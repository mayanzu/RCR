"""Literature baseline: Lyapunov drift-plus-penalty online offloading.


Implements the drift-plus-penalty decision rule of Zhao, Li and Yuan, "An
online energy-saving offloading algorithm in mobile edge computing with
Lyapunov optimization", Ad Hoc Networks 163:103580, 2024 (framework after
Neely, Stochastic Network Optimization, 2010), adapted to the shared
snapshot model: virtual backlog queues price capture and processing
occupancy, each task keeps a positive drift-plus-penalty index
V*value - Q*resource, and the positive-index set is scheduled with the same
feasibility oracle as every other method.
"""
from __future__ import annotations
import time

from rosr.boot import EPS, model
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



class LyapunovPolicy:
    """Online greedy by drift-plus-penalty index with backlog-priced resources."""

    def __init__(self, seconds=0.1, V=1.0, seed=0, **kwargs):
        self.seconds, self.V, self.seed = seconds, V, seed
        self.decisions, self.stats = [], {}

    def __call__(self, s):
        started = time.perf_counter()
        ctx = EvaluationContext(s)
        # Drift-plus-penalty index: V*value - Q_edge*work - Q_veh*(service+work).
        # Backlogs are observable counts normalized by the design parameter V
        # (Neely framework; Zhao et al. 2024 use the queue-priority form). No
        # arrival distribution is required. Under light load the index reduces
        # to value-first greedy; under backlog the resource price rises.
        n_pending = sum(1 for st in s.status.values() if st == 'pending')
        n_ready = sum(1 for st in s.status.values() if st in ('ready', 'uploading'))
        known = list(s.tasks.values())
        mean_value = max(EPS, sum(t.value for t in known) / max(1, len(known)))
        mean_res = max(EPS, sum(t.service + t.work * s.multiplier for t in known) / max(1, len(known)))
        price = mean_value / mean_res
        # Saturating backlog pressure in [0,1]; the 1/16 slope is the V knob
        # trading reward against stability, set so moderate load is visible but
        # does not veto high-value work.
        q_proc = min(1.0, n_ready / 16.0)
        q_cap = min(1.0, n_pending / 16.0)
        index = {}
        for j, st in s.status.items():
            t = s.tasks[j]
            if st == 'ready':
                index[j] = t.value - q_proc * t.work * s.multiplier * price
            elif st == 'pending':
                index[j] = t.value - q_cap * (t.service + t.work * s.multiplier) * price
        # Positive-index acquisitions are greedily inserted best-index-first;
        # a task whose index cannot be realized feasibly is skipped (the
        # drift-plus-penalty bound is per-slot, not a global commitment).
        chosen = []
        routes = [list() for _ in s.vehicles]
        for j in sorted((j for j in index if index[j] > 0 and s.status[j] == 'pending'),
                        key=lambda j: (-index[j], s.tasks[j].deadline, j)):
            t = s.tasks[j]
            placed = False
            for vid, v in enumerate(s.vehicles):
                if v.sensor != t.sensor:
                    continue
                for pos in range(len(routes[vid]) + 1):
                    trial = routes[vid][:pos] + [j] + routes[vid][pos:]
                    if ctx.route(vid, trial) is not None:
                        routes[vid] = trial
                        chosen.append(j)
                        placed = True
                        break
                if placed:
                    break
        captures = {}
        for vid, path in enumerate(routes):
            if path:
                captures.update(ctx.route(vid, path)[0])
        # Processing: every capture must report; optional ready reports with a
        # positive index are offered to the shared oracle.
        releases = dict(ctx.base_reports)
        for vid, path in enumerate(routes):
            if path:
                releases.update(ctx.route_reports[vid, tuple(path)])
        optional = {j for j in index if j in ctx.base_reports and index[j] > 0}
        offer = {j: r for j, r in releases.items() if j in set(captures) | optional}
        order = EdgeOracle(s).select(offer, set(captures))
        if order is None:
            plan = _fallback_plan(s)
            mode = 'lit_lyapunov_fallback'
        else:
            from rosr.boot import from_order
            plan = from_order(s, tuple(tuple(r) for r in routes), order, 'lit_lyapunov')
            mode = 'lit_lyapunov'
        self.decisions.append(dict(event=s.event_index, mode=mode,
                                  chosen_value=plan.value,
                                  q_proc=q_proc, q_cap=q_cap,
                                  planning_ms=1000 * (time.perf_counter() - started)))
        return plan
