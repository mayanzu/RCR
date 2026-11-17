"""Proof and boundary regressions for the current simplified policy."""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'code'))
import bootstrap as _bootstrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rosr import model
from rosr.core import _Builder, enumerate_routes, solve_set_packing
from rosr.edge import EdgeOracle
from rosr.policy import RosrPolicy, acceptance_reason, SLACK_GATE, deadline_value_bound


def task(j, x=0, deadline=105, value=10, service=1, work=1):
    return model.Task(j,0,x,0,deadline,value,0,service,.2,work)


def state(tasks,status=None,ready=None,vehicles=1):
    tasks={t.task_id:t for t in tasks}
    return model.Snapshot(0,tasks,status or {j:'pending' for j in tasks},
        [model.legacy.Vehicle(i,0,0,0,2) for i in range(vehicles)],ready or {},None,1,0)


def plan(routes,order,times,captures,value,travel=0.):
    return model.Plan(routes,order,times,captures,value,len(order),0.,travel,'test')


class RetentionEncodingTests(unittest.TestCase):
    def contested(self,fractional=False):
        # A fixed weight 100 is insufficient: retaining report 0 plus acquiring
        # task 1 costs >120 resource minutes more than taking task 2 alone.
        # Both alternatives have value 10, but tasks 0 and 2 cannot coexist.
        old=1.1 if fractional else 1
        s=state([task(0,value=old,work=5),task(1,x=80,value=10-old,work=5),
                 task(2,value=10,work=100)],
                {0:'uploading',1:'pending',2:'pending'},{0:100})
        _,_,meta=solve_set_packing(s,[enumerate_routes(s,0)],model.report_releases(s,{}),seconds=1.)
        return meta

    def test_derived_retention_bound_outweighs_global_resource_range(self):
        m=self.contested()
        self.assertIn(0,m['plan'].edge_order)
        self.assertEqual(m['plan'].value,10)
        self.assertGreater(m['resource_bound'],100)
        self.assertEqual(m['secondary_encoding'],'bounded_integer_retention')
        self.assertNotIn('retain',m['stage_solves'])
        self.assertIn('compact',m['stage_solves'])

    def test_fractional_retention_uses_exact_separate_stage(self):
        m=self.contested(True)
        self.assertIn(0,m['plan'].edge_order)
        self.assertEqual(m['secondary_encoding'],'exact_fractional_fallback')
        self.assertIn('retain',m['stage_solves'])

    def test_solver_explicitly_forwards_one_highs_thread(self):
        builder=_Builder();builder.var(('x',0))
        with patch('rosr.core.milp',return_value=SimpleNamespace()) as call:
            builder.solve(.05)
        self.assertEqual(call.call_args.kwargs['options']['threads'],1)


class GateTests(unittest.TestCase):
    def pair(self,slack=.5):
        s=state([task(0,deadline=2+slack),task(1,deadline=2+slack)])
        near=plan(((0,),),(0,),{0:(1.,2.)},{0:1.},10.,0.)
        far=plan(((1,),),(1,),{1:(1.2,2.2)},{1:1.2},10.,1.)
        return s,near,far

    def test_urgent_resource_tie_is_blocked(self):
        s,a,b=self.pair()
        self.assertEqual(acceptance_reason(s,a,b),'keep_density')

    def test_nonurgent_resource_tie_is_accepted(self):
        s,a,b=self.pair(SLACK_GATE)
        self.assertEqual(acceptance_reason(s,a,b),'equal_value_resource_dominance')

    def test_strict_value_gain_has_no_displacement_ratio_threshold(self):
        s=state([task(0,value=10),task(1,value=11)],{0:'ready',1:'pending'},{0:0})
        base=plan(((),),(0,),{0:(0,1)},{},10)
        new=plan(((1,),),(1,),{1:(1.2,2.2)},{1:1},11)
        self.assertEqual(acceptance_reason(s,new,base),'strict_value_gain')

    def test_retention_gain_does_not_bypass_resource_guard(self):
        s=state([task(0),task(1)],{0:'ready',1:'pending'},{0:0})
        base=plan(((1,),),(1,),{1:(1,2)},{1:1},10,0)
        new=plan(((),),(0,),{0:(99,100)},{},10,0)
        self.assertEqual(acceptance_reason(s,new,base),'keep_density')

    def test_retention_loss_blocks_equal_value_shortcut(self):
        s=state([task(0),task(1)],{0:'ready',1:'pending'},{0:0})
        base=plan(((),),(0,),{0:(9,10)},{},10,1)
        new=plan(((1,),),(1,),{1:(1,2)},{1:1},10,0)
        self.assertEqual(acceptance_reason(s,new,base),'keep_density')


class BoundAndWitnessTests(unittest.TestCase):
    def test_deadline_capacity_bound_is_tighter_than_individual_feasibility(self):
        s=state([task(j,deadline=10,value=10,work=5) for j in range(3)],
                {j:'ready' for j in range(3)},{j:0 for j in range(3)})
        self.assertEqual(deadline_value_bound(s,set(s.tasks)),20.)
        p=RosrPolicy()
        with patch('rosr.policy.enumerate_routes',side_effect=AssertionError('certified value is already attained')):
            chosen=p(s)
        self.assertEqual(chosen.value,20.)

    def test_deadline_bound_dominates_exact_optional_report_scheduling(self):
        import random
        rng=random.Random(819)
        for _ in range(60):
            tasks=[task(j,deadline=rng.choice([3,5,8,12]),value=rng.randint(1,10),
                        work=rng.choice([.5,1.,2.,3.])) for j in range(7)]
            s=state(tasks,{j:'ready' for j in range(7)},{j:rng.uniform(0,2) for j in range(7)})
            releases={j:s.ready_times[j] for j in s.tasks}
            exact=model.exact_edge(s,releases,required=set(),max_jobs=12)
            value=0. if exact is None else sum(s.tasks[j].value for j in exact[0])
            self.assertGreaterEqual(deadline_value_bound(s,set(s.tasks))+1e-8,value)

    def test_constant_time_uplink_matches_execution_and_independent_inverse(self):
        import random
        from rosr.transmission import Uplink
        from rosr.network_audit import finish_by_capacity
        rng=random.Random(1703)
        for n in (1,8,16,32):
            for rate in (.5,1.):
                u=Uplink(n,rate=rate)
                trials=[(rng.randrange(n),rng.uniform(0,120),rng.uniform(.001,2)) for _ in range(100)]
                trials += [(i,k*u.slot,.2) for i in range(n) for k in range(8)]
                for i,a,work in trials:
                    fast=u.finish_time(i,a,work)
                    self.assertAlmostEqual(fast,u.transmit(i,a,work)[0],places=7)
                    self.assertAlmostEqual(fast,finish_by_capacity(u,i,a,work),places=7)

    def test_visible_value_certificate_skips_pool_for_all_fleet_sizes(self):
        for n in (1,16,32):
            s=state([task(0,deadline=10)],vehicles=n)
            p=RosrPolicy()
            with patch('rosr.policy.enumerate_routes',side_effect=AssertionError('pool must be skipped')):
                chosen=p(s)
            self.assertEqual(chosen.value,10)
            self.assertEqual(p.decisions[-1]['meta']['stop_reason'],'independent_value_bound')

    def test_best_witness_reduces_completion_sum(self):
        s=state([task(0,work=10),task(1,work=1)],{0:'ready',1:'ready'},{0:0,1:0})
        order,proof=EdgeOracle(s).fixed({0:0,1:0})
        self.assertFalse(proof)
        self.assertEqual(order,(1,0))

    def test_pool_retains_density_and_slack_alternatives(self):
        s=state([task(0,value=10,deadline=50),task(1,value=8,deadline=50),
                 task(2,x=8,value=3,deadline=50)])
        full=enumerate_routes(s,0,max_routes=10000)
        multi=[w for w in full if len(w.tasks)>1]
        value=lambda w:sum(s.tasks[j].value for j in w.tasks)
        quality=lambda w:(-value(w)/max(w.occupancy,1e-9),-value(w),w.finish,w.tasks)
        slack=lambda w:min(s.tasks[j].deadline-w.reports[j]-s.tasks[j].work for j in w.tasks)
        desired={min(multi,key=quality).tasks,max(multi,key=lambda w:(slack(w),-value(w),w.tasks)).tasks}
        self.assertTrue(desired.issubset({w.tasks for w in enumerate_routes(s,0,max_routes=6)}))


if __name__=='__main__':unittest.main()
