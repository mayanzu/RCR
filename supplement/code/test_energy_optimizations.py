"""Independent schedule, bound and cache checks for certified optimizations."""
import itertools
import random
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from rosr import model
from rosr.boot import from_order, validate_plan
from rosr.capacity import (mandatory_overlap, energy_violations, deadline_value_bound,
                           release_lower_bounds, release_value_bound)
from rosr.core import enumerate_routes, solve_set_packing
from rosr.edge import ConflictStore, EdgeOracle
from rosr.evaluation import EvaluationContext, density_initialize
from rosr.scheduling import RequiredScheduler
from test_review_optimizations import task, state


class EnergyTests(unittest.TestCase):
    def test_overlap_matches_independent_start_enumeration(self):
        # Integer endpoints guarantee an integer minimizer of overlap.
        for r, p, slack, a, b in itertools.product(range(3), range(1, 4), range(4), range(6), range(1, 7)):
            if b <= a:
                continue
            d = r+p+slack
            expected = min(max(0., min(start+p, b)-max(start, a)) for start in range(r, d-p+1))
            self.assertAlmostEqual(mandatory_overlap(r, d, p, a, b), expected, places=7)

    def test_partial_overlap_detects_preemptively_feasible_conflict(self):
        self.assertTrue(energy_violations([(0., 4., 2.), (1., 3., 2.)], 0., full=True))

    def test_generated_feasible_orders_never_trigger_energy(self):
        rng = random.Random(81934)
        for n in (1, 5, 16, 40):
            for _ in range(12):
                cursor, jobs = 2., []
                for _ in range(n):
                    p = rng.uniform(.01, 3.)
                    start = cursor+rng.uniform(0., 2.)
                    jobs.append((rng.uniform(2., start), start+p+rng.uniform(0., 4.), p))
                    cursor = start+p
                self.assertEqual(energy_violations(jobs, 2., full=True), [])

    def test_optional_infeasible_report_does_not_prune_required_subset(self):
        s = state([task(0, deadline=4, work=2), task(1, deadline=3, work=2)],
                  {0:'ready', 1:'uploading'}, {0:0., 1:1.})
        ctx = EvaluationContext(s)
        p = ctx.evaluate(((),), required={0})
        self.assertIsNotNone(p)
        self.assertIn(0, p.edge_order)


class BoundTests(unittest.TestCase):
    def test_release_lp_bounds_exhaustive_optional_scheduling(self):
        rng = random.Random(8835)
        for _ in range(70):
            tasks = [task(j, deadline=rng.randint(3, 15), work=rng.choice((.5, 1., 2., 3.)),
                          value=rng.randint(1, 10)) for j in range(6)]
            s = state(tasks, {j:'uploading' for j in range(6)}, {j:rng.uniform(0., 5.) for j in range(6)})
            lower = release_lower_bounds(s, EvaluationContext(s))
            cheap = deadline_value_bound(s, lower)
            upper = release_value_bound(s, lower, cheap, seconds=.05)
            exact = model.exact_edge(s, s.ready_times, required=set(), max_jobs=12)
            opt = sum(s.tasks[j].value for j in exact[0])
            self.assertGreaterEqual(upper+1e-7, opt)
            self.assertLessEqual(upper, cheap)

    def test_release_bound_is_strictly_tighter_for_late_report_queue(self):
        s = state([task(j, deadline=10, work=2) for j in range(3)],
                  {j:'uploading' for j in range(3)}, {j:8. for j in range(3)})
        lower = release_lower_bounds(s, EvaluationContext(s))
        self.assertEqual(deadline_value_bound(s, lower), 30.)
        self.assertEqual(release_value_bound(s, lower, 30., seconds=.1), 10.)

    def test_unfinished_primal_is_never_used_as_an_upper_bound(self):
        s = state([task(j, deadline=10, work=2) for j in range(2)],
                  {j:'ready' for j in range(2)}, {j:0. for j in range(2)})
        fake = SimpleNamespace(fun=0., ineqlin=SimpleNamespace(marginals=None))
        with patch('rosr.capacity.linprog', return_value=fake):
            self.assertEqual(release_value_bound(s, s.ready_times, 20.), 20.)
        fake.ineqlin.marginals = [0.]
        with patch('rosr.capacity.linprog', return_value=fake):
            self.assertEqual(release_value_bound(s, s.ready_times, 20.), 20.)

    def test_initializer_upper_stop_preserves_complete_key_and_score(self):
        for name in ('coupled_primary', 'tighter_deadlines'):
            profile = next(p for p in model.PROFILES if p.name == name)
            tasks, vehicles = model.generate(20410000, profile, horizon=12)
            env = model.Env(tasks, vehicles, profile.edge_multiplier)
            while not env.done:
                s = env.snapshot()
                a = density_initialize(s)
                ctx = EvaluationContext(s)
                upper = deadline_value_bound(s, release_lower_bounds(s, ctx))
                b = density_initialize(s, ctx, value_upper=upper)
                self.assertEqual((a.key, a.score), (b.key, b.score))
                env.apply(a)
                env.advance()


class CertificateTests(unittest.TestCase):
    def pair(self):
        s = state([task(0, deadline=4, work=3), task(1, deadline=2, work=1)],
                  {0:'ready', 1:'uploading'}, {0:0., 1:1.})
        return s, {0:0., 1:1.}

    def test_cache_reuses_only_later_release_orthant(self):
        s, floors = self.pair()
        store = ConflictStore()
        store.add(s, floors)
        self.assertIsNotNone(store.match(s, {0:.2, 1:1.2}))
        self.assertIsNone(store.match(s, {0:0., 1:0.}))
        self.assertIsNone(store.match(s, {0:0.}))

    def test_cache_invalidates_changed_deadline_work_and_proof_base(self):
        s, floors = self.pair()
        store = ConflictStore()
        store.add(s, floors)
        changed = replace(s, tasks={**s.tasks, 0:replace(s.tasks[0], deadline=10.)})
        self.assertIsNone(store.match(changed, floors))
        self.assertIsNone(store.match(replace(s, multiplier=.5), floors))
        later = replace(s, now=1.)
        store2 = ConflictStore()
        store2.add(later, {0:1., 1:1.})
        self.assertIsNone(store2.match(s, floors))

    def test_componentwise_stronger_certificate_replaces_weaker(self):
        s, floors = self.pair()
        store = ConflictStore(capacity=2)
        store.add(s, {0:.5, 1:1.})
        store.add(s, floors)
        self.assertEqual(len(store.entries), 1)
        self.assertEqual(store.entries[0][0], floors)

    def test_strengthening_preserves_exact_infeasibility(self):
        s, floors = self.pair()
        oracle = EdgeOracle(s)
        strengthened = oracle.strengthen(floors)
        self.assertTrue(all(strengthened[j] <= floors[j] for j in floors))
        self.assertIsNone(oracle._exact_all(strengthened))

    def test_stopped_deletion_retains_a_valid_nonminimal_proof(self):
        s, floors = self.pair()
        self.assertEqual(EdgeOracle(s).minimal_conflict(floors, stop=lambda:True), floors)

    def test_tie_margin_conflicts_do_not_leak_to_original_deadlines(self):
        s = state([task(j, deadline=4, work=2) for j in range(2)],
                  {0:'ready', 1:'ready'}, {0:0., 1:0.})
        base = from_order(s, ((),), (0, 1), 'baseline')
        store = ConflictStore()
        _, _, meta = solve_set_packing(s, [enumerate_routes(s, 0)], s.ready_times,
                                      baseline=base, conflict_store=store, seconds=.5)
        self.assertEqual(meta['tie_acceptance_rows'], 4)
        self.assertEqual(store.entries, [])
        self.assertTrue(validate_plan(s, meta['plan']))

    def test_tie_constraints_never_restrict_strict_value_gain(self):
        s = state([task(0, deadline=2.2, work=1, service=.9)])
        base = from_order(s, ((),), (), 'baseline')
        _, _, meta = solve_set_packing(s, [enumerate_routes(s, 0)], {}, baseline=base, seconds=.5)
        self.assertEqual(meta['plan'].value, 10.)
        self.assertEqual(meta['tie_acceptance_rows'], 0)

    def test_stopped_pool_keeps_singletons_and_full_anchor(self):
        s = state([task(j, deadline=50) for j in range(5)])
        routes = enumerate_routes(s, 0, anchor=(0,1,2,3,4), focus={2}, stop=lambda:True)
        keys = {w.tasks for w in routes}
        self.assertIn((0,1,2,3,4), keys)
        self.assertTrue(all((j,) in keys for j in range(5)))


class WitnessTests(unittest.TestCase):
    def test_heap_witness_matches_all_three_original_rules(self):
        rng = random.Random(17018)
        for _ in range(250):
            n = rng.randint(0, 32)
            s = state([task(j, deadline=rng.uniform(2, 100), work=rng.uniform(.01, 4),
                            value=rng.randint(1, 10)) for j in range(n)],
                      {j:'uploading' for j in range(n)},
                      {j:rng.choice((0., rng.uniform(0., 20.))) for j in range(n)})
            scheduler = RequiredScheduler(s)
            for rule in ('edf', 'reserved_edf', 'density'):
                self.assertEqual(scheduler.schedule(s.ready_times, rule),
                                 model.edge_schedule(s, s.ready_times, rule, set(s.tasks)))

    def test_near_boundary_readiness_and_locked_processor(self):
        s = state([task(j, deadline=20) for j in range(3)],
                  {j:'uploading' for j in range(3)}, {0:2., 1:2.+model.EPS/2, 2:3.})
        s.edge_active = (99, 0., 2.)
        for rule in ('edf', 'reserved_edf', 'density'):
            self.assertEqual(RequiredScheduler(s).schedule(s.ready_times, rule),
                             model.edge_schedule(s, s.ready_times, rule, set(s.tasks)))


if __name__ == '__main__':
    unittest.main()
