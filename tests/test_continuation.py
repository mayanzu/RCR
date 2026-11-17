"""Execution-level tests of retained guarantees and immediate-action guards."""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'code'))
import bootstrap as _bootstrap
import unittest
from dataclasses import replace
from unittest.mock import patch
from rosr.boot import model, from_order, validate_plan, audit
from rosr.continuation import Continuation, ActionArchive
from rosr.criteria import acceptance_reason, exposure_cost, release_profile
from rosr.policy import RosrPolicy
from rosr.evaluation import density_initialize
from rosr.network_model import NetworkEnv
from rosr.network_audit import check_episode
from test_latest_algorithm import task, state


class ContinuationTests(unittest.TestCase):
    def test_same_action_keeps_a_better_executable_tail(self):
        s = state([task(0, deadline=20), task(1, deadline=20)])
        base = from_order(s, ((0,),), (0,), 'base')
        better = from_order(s, ((0,1),), (0,1), 'tail')
        self.assertEqual(acceptance_reason(s, better, base, guarded=True), 'same_action_tail_gain')
        archive = ActionArchive(s, lambda p:(-p.value,p.travel))
        archive.add(base); archive.add(better)
        self.assertEqual(len(archive.plans), 1)
        self.assertEqual(archive.duplicates, 1)
        self.assertIs(archive.best(), better)

    def test_locked_edge_is_counted_once(self):
        s = state([task(0),task(1)], {0:'active_edge',1:'ready'}, {1:0}, (0,0,1))
        p = from_order(s, ((),), (1,), 'tail')
        self.assertEqual(Continuation.value(s,p),20)
        after = replace(s, now=1, status={0:'completed',1:'ready'}, edge_active=None)
        q = from_order(after, ((),), (1,), 'tail')
        self.assertEqual(Continuation.value(after,q),20)

    def test_expired_promise_is_not_silently_dropped(self):
        s = state([task(0)])
        witness = Continuation()
        witness.remember(s, from_order(s, ((0,),), (0,), 'base'))
        self.assertEqual(witness.advance(replace(s,status={0:'expired'})),
                         (None,'promised_task_expired'))

    def test_contract_and_task_changes_invalidate_witness(self):
        s = state([task(0)])
        witness = Continuation()
        witness.remember(s, from_order(s, ((0,),), (0,), 'base'))
        self.assertEqual(witness.advance(replace(s,multiplier=2))[1],'contract_changed')
        changed = replace(s,tasks={0:replace(s.tasks[0],deadline=100)})
        self.assertEqual(witness.advance(changed)[1],'task_changed')

    def test_sunk_acquisition_is_not_an_absolute_veto_on_strict_gain(self):
        s = state([task(0,value=10),task(1,value=11)],{0:'ready',1:'pending'},{0:0})
        base = from_order(s,((),),(0,),'base')
        candidate = from_order(s,((1,),),(1,),'candidate')
        self.assertEqual(acceptance_reason(s,candidate,base,guarded=True),'strict_value_gain')
        # The declared comparison control retains its original rule.
        self.assertEqual(acceptance_reason(s,candidate,base),'strict_value_gain')

    def test_edge_lock_cost_rejects_one_point_for_two_more_minutes(self):
        s = state([task(0,deadline=12,value=5,work=4),task(1,deadline=12,value=6,work=6)],
                  {0:'ready',1:'ready'},{0:0,1:0})
        base = from_order(s,((),),(0,),'base')
        candidate = from_order(s,((),),(1,),'candidate')
        self.assertAlmostEqual(exposure_cost(s,candidate,base),2.5)
        self.assertEqual(acceptance_reason(s,candidate,base,guarded=True),'continuation_guard')

    def test_empty_baseline_does_not_forbid_tight_first_acquisition(self):
        s = state([task(0,deadline=2.2,service=.9)])
        base = from_order(s,((),),(),'empty')
        candidate = from_order(s,((0,),),(0,),'candidate')
        self.assertEqual(acceptance_reason(s,candidate,base,guarded=True),'strict_value_gain')

    def test_behavioral_rejection_is_not_a_persistent_infeasibility_cut(self):
        from rosr.core import solve_set_packing, enumerate_routes
        from rosr.edge import ConflictStore
        s = state([task(0,deadline=4,value=5,work=4),task(1,deadline=6,value=6,work=6)],
                  {0:'ready',1:'ready'},{0:0,1:0})
        base = from_order(s,((),),(0,),'base')
        store = ConflictStore()
        _,_,meta = solve_set_packing(s,[enumerate_routes(s,0)],s.ready_times,
                                     baseline=base,conflict_store=store,seconds=.5)
        self.assertGreater(meta['action_rejections'],0)
        self.assertEqual(meta['plan'].value,5)
        self.assertIsNone(store.match(s,{1:0.}))

    def test_value_ceiling_does_not_unconditionally_skip_other_actions(self):
        s = state([task(0,deadline=20,work=1),task(1,deadline=20,work=2)],
                  {0:'ready',1:'ready'},{0:0,1:0})
        base = from_order(s,((),),(0,1),'base')
        other = from_order(s,((),),(1,0),'other')
        policy = RosrPolicy(seconds=.1)
        policy.continuation.remember(s,base)
        with patch('rosr.policy.density_initialize',return_value=other), \
             patch('rosr.policy.solve_set_packing',return_value=([],set(),{'plan':base})) as solve:
            policy(s)
        self.assertTrue(solve.called)
        self.assertEqual(policy.stats.value_bound_tie_searches,1)

    def test_frozen_tail_executes_across_all_physical_events(self):
        for network in (False,True):
            tasks = [task(j,deadline=30,service=.2,work=.3) for j in range(5)]
            vehicles = [model.legacy.Vehicle(0,0,0,0,2)]
            env = (NetworkEnv if network else model.Env)(tasks,vehicles,1.)
            witness = Continuation()
            p = from_order(env.snapshot(),(tuple(range(5)),),tuple(range(5)),'initial')
            initial = p.value
            while not env.done:
                s = env.snapshot()
                if witness.plan is not None:
                    p, reason = witness.advance(s)
                    self.assertEqual(reason,'valid')
                self.assertAlmostEqual(witness.value(s,p),initial)
                witness.remember(s,p)
                env.apply(p); env.advance()
            summary, rows, logs = env.results()
            self.assertEqual(sum(t.value for t in tasks if env.status[t.task_id]=='completed'),initial)
            if network:
                check_episode(tasks,vehicles,1.,env.uplink,summary,rows,logs)
            else:
                audit.check_episode(summary,rows,logs)
                audit.audit_action_logs(tasks,vehicles,1.,rows,logs)

    def test_online_arrivals_preserve_guarantee_and_physical_audit(self):
        for network in (False,True):
            for name in ('coupled_primary','tighter_deadlines'):
                profile = next(p for p in model.PROFILES if p.name==name)
                tasks, vehicles = model.generate(20412000,profile,horizon=12)
                env = (NetworkEnv if network else model.Env)(tasks,vehicles,profile.edge_multiplier)
                policy = RosrPolicy(seconds=.005)
                previous = 0.
                while not env.done:
                    p = policy(env.snapshot())
                    self.assertTrue(validate_plan(env.snapshot(),p))
                    guarantee = policy.decisions[-1]['guaranteed_value']
                    self.assertGreaterEqual(guarantee+model.EPS,previous)
                    previous = guarantee
                    env.apply(p); env.advance()
                self.assertEqual(policy.stats.continuation_invalidations,0)
                self.assertGreaterEqual(sum(t.value for t in tasks if env.status[t.task_id]=='completed')+model.EPS,previous)
                summary,rows,logs = env.results()
                if network:
                    check_episode(tasks,vehicles,profile.edge_multiplier,env.uplink,summary,rows,logs)
                else:
                    audit.check_episode(summary,rows,logs)
                    audit.audit_action_logs(tasks,vehicles,profile.edge_multiplier,rows,logs)

    def test_policy_reset_clears_previous_episode_witness(self):
        p = RosrPolicy(seconds=.001)
        s = state([task(0)])
        p(replace(s,event_index=5))
        p(s)
        self.assertEqual(p.decisions[-1]['continuation_status'],'empty')

    def test_worse_new_density_cannot_replace_retained_plan(self):
        s = state([task(0),task(1)])
        p = RosrPolicy(seconds=.001)
        old = from_order(s,((0,1),),(0,1),'saved')
        p.continuation.remember(s,old)
        empty = from_order(s,((),),(),'empty')
        with patch('rosr.policy.density_initialize',return_value=empty):
            chosen = p(s)
        self.assertEqual(chosen.value,old.value)
        self.assertEqual(p.decisions[-1]['guaranteed_value'],old.value)

    def test_release_profile_includes_fifo_tail_and_destination(self):
        tasks = [task(0,deadline=50)]
        env = NetworkEnv(tasks,[model.legacy.Vehicle(0,0,0,0,2)],1.)
        s = env.snapshot()
        s.link_tails = (5.,)
        p = from_order(s,((0,),),(0,),'fifo')
        profile = release_profile(s,p)
        self.assertGreater(profile['upload'][0],5.)
        self.assertEqual(profile['locations'],((0,0),))


if __name__ == '__main__':
    unittest.main()
