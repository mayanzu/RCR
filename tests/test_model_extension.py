
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'code'))
import bootstrap as _bootstrap
import unittest,random
from collections import Counter
from dataclasses import replace
from rosr import model
from rosr.transmission import Uplink,route_releases
from rosr.network_model import NetworkEnv,NetworkSnapshot,SCENARIOS,generate
from rosr.network_audit import check_episode,finish_by_capacity
from rosr.evaluation import density_initialize,EvaluationContext


class ExtensionTests(unittest.TestCase):
    def test_inverse_capacity_matches_iterated_calendar(self):
        rng=random.Random(81)
        for _ in range(100):
            n=rng.choice((1,8,16,32));u=Uplink(n);v=rng.randrange(n)
            start=rng.random()*30;work=rng.random()*.8
            self.assertAlmostEqual(u.transmit(v,start,work)[0],finish_by_capacity(u,v,start,work),places=8)

    def test_global_slots_do_not_overlap(self):
        u=Uplink(8)
        segments=sorted(seg for i in range(8) for seg in u.transmit(i,0,.2)[1])
        self.assertTrue(all(a[1]<=b[0]+1e-8 for a,b in zip(segments,segments[1:])))

    def test_single_vehicle_calendar_degenerates_to_continuous_link(self):
        u=Uplink(1)
        for start in (0,.025,.031,1.5):
            self.assertAlmostEqual(u.transmit(0,start,.2)[0],start+.2,places=8)

    def test_one_vehicle_packets_still_queue_and_independent_audit_accepts_partial_slot(self):
        tasks=[model.Task(j,0,0,0,5,1,0,.02) for j in range(2)]
        vehicles=[model.legacy.Vehicle(0,0,0,0,2)]
        env=NetworkEnv(tasks,vehicles,1.)
        reports=route_releases(env.snapshot(),0,{0:.02,1:.04})
        self.assertAlmostEqual(reports[0],.22)
        self.assertAlmostEqual(reports[1],.42)
        while not env.done:
            p=density_initialize(env.snapshot());env.apply(p);env.advance()
        summary,rows,logs=env.results()
        check_episode(tasks,vehicles,1.,env.uplink,summary,rows,logs)

    def test_independent_auditor_rejects_invented_airtime(self):
        import copy
        tasks=[model.Task(0,0,0,0,5,1,0,.02)]
        vehicles=[model.legacy.Vehicle(0,0,0,0,2)]
        env=NetworkEnv(tasks,vehicles,1.)
        while not env.done:
            p=density_initialize(env.snapshot());env.apply(p);env.advance()
        summary,rows,logs=env.results();logs=copy.deepcopy(logs)
        event=next(e for e in logs if e['kind']=='upload_commit')
        event['segments']=((0.,1.),)
        with self.assertRaises(AssertionError):
            check_episode(tasks,vehicles,1.,env.uplink,summary,rows,logs)

    def test_factor_transform_preserves_task_marginals_and_slack(self):
        original,vehicles=model.generate(20360000,model.Profile('coupled_primary'))
        for sc in SCENARIOS:
            tasks,_=generate(20360000,sc)
            self.assertEqual(Counter(t.value for t in tasks),Counter(t.value for t in original))
            for a,b in zip(tasks,original):
                self.assertAlmostEqual(a.deadline-a.release,b.deadline-b.release)
                self.assertEqual((a.x,a.y,a.service,a.work,a.sensor),(b.x,b.y,b.service,b.work,b.sensor))
            if sc.batch:self.assertTrue(all(t.release%5==0 for t in tasks))
            if sc.correlated:
                ordered=sorted(tasks,key=lambda t:(t.service+1.5*t.work,t.task_id))
                self.assertEqual([t.value for t in ordered],sorted(t.value for t in tasks))

    def test_ranked_initializer_matches_exhaustive_on_both_models(self):
        for network in (False,True):
            sc=next(s for s in SCENARIOS if s.batch and s.correlated and s.network==network)
            tasks,vehicles=generate(20360001,sc)
            tasks=[t for t in tasks if t.release<10]
            env=NetworkEnv(tasks,vehicles,1.5) if network else model.Env(tasks,vehicles,1.5)
            while not env.done:
                s=env.snapshot();a=density_initialize(s,ranked=False);b=density_initialize(s)
                self.assertEqual((a.key,a.score),(b.key,b.score))
                env.apply(b);env.advance()
            if network:
                summary,rows,logs=env.results()
                check_episode(tasks,vehicles,1.5,env.uplink,summary,rows,logs)

    def test_fractional_priorities_fall_back_without_losing_gain(self):
        tasks=[model.Task(0,0,0,0,10,1.0000001,0,1),model.Task(1,0,0,0,10,1.,0,1)]
        env=model.Env(tasks,[model.legacy.Vehicle(0,0,0,0,2)],1.5)
        a=density_initialize(env.snapshot(),ranked=False);b=density_initialize(env.snapshot())
        self.assertEqual((a.key,a.score),(b.key,b.score))

    def test_path_upload_queue_accounts_for_locked_capture_and_backlog(self):
        tasks={j:model.Task(j,0,0,0,30,1,0,1) for j in range(3)}
        vehicle=model.legacy.Vehicle(0,0,0,0,2)
        vehicle.active=model.legacy.ActiveJob(0,0,1,0)
        s=NetworkSnapshot(0,tasks,{0:'active_capture',1:'pending',2:'pending'},[vehicle],{},None,1.5,0,Uplink(1),(2.,))
        reports=route_releases(s,0,{1:2.,2:3.})
        self.assertAlmostEqual(reports[1],2.4)
        self.assertAlmostEqual(reports[2],3.2)

    def test_expired_upload_drains_physical_transmission(self):
        tasks=[model.Task(0,0,0,0,3.,1,0,.1,upload=.2,work=.1)]
        vehicles=[model.legacy.Vehicle(i,0,0,0,2) for i in range(8)]
        env=NetworkEnv(tasks,vehicles,1.)
        # A feasible start is reserved, then a deadline shock is injected only
        # into this mechanism test to verify a physically locked upload drain.
        plan=density_initialize(env.snapshot());env.apply(plan);env.advance()
        self.assertTrue(env.uploads)
        env.known[0]=replace(env.known[0],deadline=env.now+.01)
        while not env.done:
            p=density_initialize(env.snapshot());env.apply(p);env.advance()
        self.assertEqual(env.status[0],'expired')
        self.assertGreater(env.now,env.known[0].deadline)
        self.assertFalse(env.uploads)


if __name__=='__main__':unittest.main(verbosity=2)
