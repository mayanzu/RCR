"""Regression tests for failure mechanisms, not copies of implementation."""
import unittest
from dataclasses import replace
from rosr import model
from rosr.core import (Route, enumerate_routes, solve_set_packing,
                       _violated_intervals, _selected_plan, build_plan)
from rosr.policy import acceptance_reason

def task(j, deadline=10, value=10, service=1, work=1):
    return model.Task(j, 0, 0, 0, deadline, value, 0, service, 0.2, work)

def state(tasks, status=None, ready=None, edge=None):
    tasks = {t.task_id:t for t in tasks}
    return model.Snapshot(0, tasks, status or {j:'pending' for j in tasks},
                          [model.legacy.Vehicle(0,0,0,0,2)], ready or {}, edge, 1)

class AlgorithmTests(unittest.TestCase):
    def test_tight_legal_high_value_path_survives_risk(self):
        s = state([task(0, deadline=3.5, service=1, work=2.25),
                   task(1, deadline=50, value=1)], {0:'pending',1:'ready'}, {1:0})
        paths = enumerate_routes(s,0,beta=100)
        self.assertIn((0,), [w.tasks for w in paths])
        _,_,meta = solve_set_packing(s,[paths],model.report_releases(s,{}), beta=100)
        self.assertIn(0,meta['plan'].edge_order)

    def test_preemptive_feasible_nonpreemptive_impossible_adds_report_cut(self):
        s = state([task(0,deadline=4,work=3,value=10),
                   task(1,deadline=2,work=1,value=9)],
                  {0:'ready',1:'uploading'},{0:0,1:1})
        releases = model.report_releases(s,{})
        # Mandatory overlap detects the nonpreemptive conflict directly.
        self.assertTrue(_violated_intervals(s,[],{0,1},releases,0))
        self.assertEqual((None,True), _selected_plan(s,[],{0,1}))
        _,_,meta = solve_set_packing(s, [enumerate_routes(s,0)], releases)
        self.assertGreaterEqual(meta['horn_cuts']+meta['no_good_cuts'],1)
        self.assertEqual(meta['plan'].value,10)
        self.assertEqual(meta['plan'].edge_order,(0,))

    def test_decode_failure_does_not_drop_acquisition(self):
        s = state([task(0,deadline=4,service=0.8,work=3),
                   task(1,deadline=2,work=1)],{0:'pending',1:'uploading'},{1:1})
        # Route report release is 1: both jobs cannot be served.
        w = next(w for w in enumerate_routes(s,0) if w.tasks==(0,))
        self.assertIsNone(build_plan(s,[w],{0,1}))

    def test_raw_fractional_gain_cannot_be_traded_for_short_trip(self):
        a=task(0,deadline=5,value=1,service=1,work=2)
        b=replace(task(1,deadline=5,value=1.00001,service=1,work=2),x=2)
        s=state([a,b])
        _,_,meta=solve_set_packing(s,[enumerate_routes(s,0)],{})
        self.assertEqual(meta['plan'].value,1.00001)

    def test_locked_edge_load_is_enforced(self):
        s=state([task(0,deadline=4,work=2)],{0:'ready'},{0:0},(99,0,3))
        _,_,meta=solve_set_packing(s,[enumerate_routes(s,0)],{0:0})
        self.assertEqual(meta['plan'].value,0)

    def test_anchor_survives_depth_cap(self):
        s=state([task(j,deadline=50) for j in range(4)])
        paths=enumerate_routes(s,0,max_depth=1,max_routes=2,anchor=(0,1,2,3))
        self.assertIn((0,1,2,3),[w.tasks for w in paths])
        self.assertTrue(all((j,) in [w.tasks for w in paths] for j in range(4)))

    def test_equal_value_later_plan_is_rejected(self):
        s=state([task(0), replace(task(1),x=2)])
        a=model.evaluate(s,((0,),))
        b=model.evaluate(s,((1,),))
        self.assertEqual(acceptance_reason(s,b,a),'keep_density')
        self.assertEqual(acceptance_reason(s,a,b),'equal_value_resource_dominance')

    def test_local_busy_window_proves_large_infeasible_set(self):
        # 16 selected reports: preemptively servable, nonpreemptively impossible
        # because job 14 occupies [0,9] and job 15 must finish by 2.
        jobs=[task(j,deadline=100,value=1,service=1,work=.1) for j in range(14)]
        jobs+=[task(14,deadline=10,value=10,service=1,work=9),
               task(15,deadline=2,value=10,service=1,work=1)]
        s=state(jobs,{j:'ready' for j in range(16)},
                {**{j:0 for j in range(14)},14:0,15:1})
        releases=model.report_releases(s,{})
        self.assertEqual(_selected_plan(s,[],set(range(16))),(None,True))
        from rosr.edge import EdgeOracle
        conflict=EdgeOracle(s).minimal_conflict({j:releases[j] for j in range(16)})
        self.assertLessEqual(len(conflict),14)
        self.assertEqual(set(conflict),{14,15})
        _,_,meta=solve_set_packing(s,[enumerate_routes(s,0)],releases)
        self.assertGreaterEqual(meta['horn_cuts']+meta['no_good_cuts'],1)
        # Fourteen one-point reports plus the better of the two conflicting jobs.
        self.assertEqual(meta['plan'].value,24)

    def test_feasible_local_window_leaves_large_decode_unknown(self):
        # Sixteen one-minute reports inside one 14-minute window cannot all be
        # served, but every local window subset can: no proof, therefore no cut.
        s=state([task(j,deadline=14,work=1) for j in range(16)],
                {j:'ready' for j in range(16)},{j:0 for j in range(16)})
        releases=model.report_releases(s,{})
        self.assertEqual(_selected_plan(s,[],set(range(16))),(None,False))
        from rosr.edge import EdgeOracle
        oracle=EdgeOracle(s)
        self.assertEqual(oracle.fixed({j:releases[j] for j in range(16)}),(None,False))
        with self.assertRaises(ValueError):
            oracle.minimal_conflict({j:releases[j] for j in range(16)})

    def test_solver_exhaustion_uses_density(self):
        from unittest.mock import patch
        from rosr.policy import RosrPolicy
        s=state([task(0,deadline=4,work=2),task(1,deadline=4,work=2)])
        with patch('rosr.policy.release_value_bound', return_value=20.), patch('rosr.core._Builder.solve') as solve:
            solve.return_value=type('Result',(),dict(status=1,x=None))()
            p=RosrPolicy()(s)
            self.assertTrue(solve.called)
        self.assertEqual(p.value,model.joint_resource_density(s).value)

    def test_witness_or_exact_oracle_matches_exhaustive_orders(self):
        import itertools, random
        from rosr.edge import EdgeOracle
        rng=random.Random(71)
        for _ in range(25):
            jobs=[task(j,deadline=rng.randint(3,12),work=rng.choice((1,2,3))) for j in range(5)]
            s=state(jobs,{j:'uploading' for j in range(5)},{j:rng.randint(0,4) for j in range(5)})
            releases=model.report_releases(s,{})
            def feasible(order):
                cursor=0
                for j in order:
                    cursor=max(cursor,releases[j])+s.tasks[j].work
                    if cursor>s.tasks[j].deadline+model.EPS:
                        return False
                return True
            exists=any(feasible(order) for order in itertools.permutations(range(5)))
            order,proof=EdgeOracle(s).fixed(releases)
            self.assertEqual(order is not None,exists)
            self.assertEqual(proof,not exists)
            if order is not None:
                self.assertTrue(feasible(order))

    def test_release_floor_cut_excludes_later_routes_but_not_earlier_route(self):
        from rosr.core import _Builder,_release_conflict_cut
        from rosr.edge import EdgeOracle
        s=state([task(0,deadline=6.5,work=3),task(1,deadline=4.5,work=1),task(2,deadline=30)],
                {0:'pending',1:'uploading',2:'ready'},{1:3.5,2:0})
        oracle=EdgeOracle(s)
        conflict=oracle.minimal_conflict({0:2.,1:3.5,2:0.})
        self.assertEqual(set(conflict),{0,1})
        routes=[Route(i,(0,),{0:r-.2},{0:r},0,r,r) for i,r in enumerate((2.,2.5,.5))]
        builder=_Builder()
        y={w:builder.var(('y',w.vehicle)) for w in routes}
        z={j:builder.var(('z',j)) for j in (0,1,2)}
        _release_conflict_cut(s,builder,y,z,[[w] for w in routes],{1:3.5,2:0.},conflict)
        terms=dict(zip(builder.cols,builder.data))
        self.assertEqual(terms[y[routes[0]]],1)
        self.assertEqual(terms[y[routes[1]]],1)
        self.assertNotIn(y[routes[2]],terms)
        self.assertEqual(terms[z[1]],1)
        self.assertNotIn(z[2],terms)
        self.assertEqual(builder.hi[-1],1)

    def test_cached_density_matches_frozen_decisions_on_released_states(self):
        from rosr.evaluation import density_initialize
        profile=next(p for p in model.PROFILES if p.name=='tighter_deadlines')
        tasks,vehicles=model.generate(20356004,profile,horizon=12)
        env=model.Env(tasks,vehicles,profile.edge_multiplier)
        while not env.done:
            s=env.snapshot()
            a,b=model.joint_resource_density(s),density_initialize(s)
            self.assertEqual((a.key,a.score),(b.key,b.score))
            env.apply(a)
            env.advance()

    def test_cached_route_rejects_duplicate_acquisition(self):
        from rosr.evaluation import EvaluationContext
        s=state([task(0)])
        self.assertIsNone(EvaluationContext(s).predictions(((0,0),)))

    def test_analytic_upper_skips_solver_when_density_serves_all_possible_value(self):
        from unittest.mock import patch
        from rosr.policy import RosrPolicy
        s=state([task(0)])
        with patch('rosr.core._Builder.solve',side_effect=AssertionError('solver should be skipped')):
            policy=RosrPolicy(compact_ties=False)
            p=policy(s)
        self.assertEqual(p.value,10)
        self.assertEqual(policy.decisions[-1]['meta']['stop_reason'],'independent_value_bound')

    def test_aggregate_tie_saving_cannot_newly_lock_another_idle_vehicle(self):
        s=state([task(0,service=4),task(1,service=1)])
        s.vehicles.append(model.legacy.Vehicle(1,0,0,0,2))
        a=model.evaluate(s,((0,),()))
        b=model.evaluate(s,((),(1,)))
        self.assertLess(b.wait,a.wait)
        self.assertEqual(acceptance_reason(s,b,a),'keep_density')

    def test_public_decoder_handles_fourteen_jobs_without_legacy_ten_job_cap(self):
        s=state([task(j,deadline=30) for j in range(14)],
                {j:'ready' for j in range(14)},{j:0 for j in range(14)})
        plan=build_plan(s,[],set(range(14)))
        self.assertEqual(len(plan.edge_order),14)

    def test_lazy_drop_stream_advances_past_an_infeasible_cheaper_removal(self):
        from rosr.evaluation import density_initialize
        s=state([task(0,deadline=10,value=1,work=.25),
                 task(1,deadline=4,value=2,work=2),task(2,deadline=4,value=5,work=3)],
                {0:'ready',1:'ready',2:'ready'},{0:0,1:0,2:0})
        a=density_initialize(s,selected_reports={0,1},ranked=False)
        b=density_initialize(s,selected_reports={0,1})
        self.assertEqual((a.key,a.score),(b.key,b.score))
        self.assertEqual(b.value,6)
        self.assertEqual(set(b.edge_order),{0,2})

    def test_capacity_pruning_preserves_exhaustive_density_on_loaded_states(self):
        import random
        from rosr.evaluation import density_initialize
        rng=random.Random(934)
        for _ in range(20):
            jobs=[task(j,deadline=rng.randint(4,18),value=rng.randint(1,10),
                       service=rng.choice((1,2)),work=rng.choice((.5,1,2))) for j in range(10)]
            s=state(jobs,{j:'ready' if j<6 else 'pending' for j in range(10)},
                    {j:0 for j in range(6)},(99,0,1.5))
            a=density_initialize(s,ranked=False);b=density_initialize(s)
            self.assertEqual((a.key,a.score),(b.key,b.score))

if __name__=='__main__':
    unittest.main(verbosity=2)

