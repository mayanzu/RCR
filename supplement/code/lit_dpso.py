"""Literature baseline: discrete particle swarm task offloading.


Implements the decision machinery of Qiu, Mao and Liu, "Joint Optimization of
Energy and Delay in Task Offloading Process of Electric Connected Vehicles",
IEEE ICC 2023, pp. 979-984 (CCF-C), doi:10.1109/ICC45041.2023.10279101,
adapted to the shared snapshot model: a particle encodes per-task vehicle
assignment plus a sequencing key, routes and processing are decoded with the
same feasibility oracle as every other method, and the swarm runs inside the
common wall budget. Fitness is completed value under physical feasibility.
"""
from __future__ import annotations
import random
import time

from rosr.boot import EPS, model, from_order
from rosr.evaluation import EvaluationContext
from rosr.edge import EdgeOracle


def _fallback_plan(s):
    """Fallback decode: reference contract keeps the frozen rule; reserved-slot
    snapshots decode through the shared network-aware density (decision rule
    unchanged, only the release predictor differs)."""
    if getattr(s, 'uplink', None) is not None:
        from rosr.evaluation import EvaluationContext, density_initialize
        plan = density_initialize(s, EvaluationContext(s))
        if plan is not None:
            return plan
    return model.make_plan(s, 'e2e_resource_density')



class DpsoPolicy:
    """Binary/discrete PSO over (assignment, order-key) with oracle decoding."""

    def __init__(self, seconds=0.1, swarm=12, iters=15, max_evaluations=512, seed=0, **kwargs):
        self.seconds, self.swarm, self.iters = seconds, swarm, iters
        self.max_evaluations, self.seed = max_evaluations, seed
        self.decisions, self.stats = [], {}

    def _decode(self, s, ctx, genes):
        """genes: {j: (vid_or_None, key)}. Build routes by key order per vehicle."""
        routes = [list() for _ in s.vehicles]
        for j, (vid, key) in genes.items():
            if vid is not None:
                routes[vid].append((key, j))
        tuples = []
        for vid, items in enumerate(routes):
            items.sort()
            path = []
            for _, j in items:
                trial = path + [j]
                if ctx.route(vid, trial) is None:
                    continue
                path = trial
            tuples.append(tuple(path))
        tuples = tuple(tuples)
        captured = {j for r in tuples for j in r}
        return ctx.evaluate(tuples, required=captured)

    def __call__(self, s):
        started = time.perf_counter()
        ctx = EvaluationContext(s)
        deadline = started + self.seconds
        rng = random.Random((self.seed * 131 + s.event_index) & 0x7fffffff)
        pending = sorted((j for j, st in s.status.items() if st == 'pending'),
                         key=lambda j: (s.tasks[j].deadline, j))
        if not pending:
            plan = self._decode(s, ctx, {})
            self.decisions.append(dict(event=s.event_index, mode='lit_dpso',
                                      chosen_value=0. if plan is None else plan.value,
                                      planning_ms=1000*(time.perf_counter()-started)))
            return plan if plan is not None else _fallback_plan(s)
        # Dimensions: vehicle index (including skip) and a sequencing key.
        compat = {j: [i for i, v in enumerate(s.vehicles)
                      if v.sensor == s.tasks[j].sensor] for j in pending}
        n = len(pending)

        def random_gene():
            j = pending[rng.randrange(n)]
            opts = compat[j] + [None]
            return {k: (rng.choice(compat[k] or [None]), rng.random())
                    if k == j else (rng.choice(compat[k] or [None]), rng.random())
                    for k in pending}

        swarm = [random_gene() for _ in range(self.swarm)]
        pbest = [dict(g) for g in swarm]
        pfit = [-1.0] * self.swarm
        gbest, gfit = None, -1.0
        vel = [{j: (0.0, 0.0) for j in pending} for _ in range(self.swarm)]
        evaluations = 0

        def exhausted():
            return (evaluations >= self.max_evaluations
                    or time.perf_counter() >= deadline)

        for i, gene in enumerate(swarm):
            if exhausted():
                break
            plan = self._decode(s, ctx, gene)
            evaluations += 1
            fit = -1.0 if plan is None else plan.value
            pfit[i] = fit
            if fit > gfit:
                gbest, gfit = dict(gene), fit
        w, c1, c2 = 0.6, 1.6, 1.6
        it = 0
        while it < self.iters and not exhausted():
            it += 1
            for i, gene in enumerate(swarm):
                if exhausted():
                    break
                for j in pending:
                    r1, r2 = rng.random(), rng.random()
                    a_key, a_vid = vel[i][j]
                    pg = pbest[i][j][1] - gene[j][1]
                    gg = (gbest[j][1] - gene[j][1]) if gbest is not None else 0.0
                    a_key = w*a_key + c1*r1*pg + c2*r2*gg
                    opts = compat[j] or [None]
                    a_vid = max(-1., min(1., w*a_vid + c1*r1*(0. if pbest[i][j][0] == gene[j][0] else 1.)
                                         + c2*r2*(0. if gbest is None or gbest[j][0] == gene[j][0] else 1.)))
                    vel[i][j] = (a_key, a_vid)
                    new_key = min(1., max(0., gene[j][1] + a_key))
                    if rng.random() < abs(a_vid):
                        new_vid = rng.choice(opts)
                    else:
                        new_vid = gene[j][0]
                    gene[j] = (new_vid, new_key)
                plan = self._decode(s, ctx, gene)
                evaluations += 1
                fit = -1.0 if plan is None else plan.value
                if fit > pfit[i]:
                    pfit[i], pbest[i] = fit, dict(gene)
                if fit > gfit:
                    gbest, gfit = dict(gene), fit
        plan = self._decode(s, ctx, gbest) if gbest is not None else None
        if plan is None:
            plan = _fallback_plan(s)
        self.decisions.append(dict(event=s.event_index, mode='lit_dpso',
                                  chosen_value=plan.value, evaluations=evaluations,
                                  iterations=it, planning_ms=1000*(time.perf_counter()-started)))
        return plan
