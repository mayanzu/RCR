"""Common-budget deterministic joint destroy/repair control, not full ALNS."""
from __future__ import annotations
import itertools
import time
from .boot import EPS, from_order, model, validate_plan, report_releases
from .evaluation import EvaluationContext, density_initialize
from .edge import EdgeOracle
from .criteria import ACCEPTED
from .policy import acceptance_reason, independent_value_bound, plan_resources


class DensityPolicy:
    def __init__(self, **kwargs):
        self.decisions = []
        self.stats = {}

    def __call__(self, s):
        start = time.perf_counter()
        ctx = EvaluationContext(s)
        plan = density_initialize(s, ctx)
        self.decisions.append(dict(event=s.event_index, planning_ms=1000*(time.perf_counter()-start),
                                   mode='density', chosen_value=plan.value))
        return plan


def plan_key(s, p):
    return (-p.value, *plan_resources(s,p), p.key)


class JointLnsPolicy:
    """Relocate, swap, remove one/two tasks, and greedy joint reinsert.

    Both edge selection and ordering can change, using the same public model,
    exact feasibility cap and anchor as the proposed policy. Search has the
    same additional wall budget and a deterministic proposal cap.
    """
    def __init__(self, seconds=0.1, compact_ties=False, max_evaluations=1024, **kwargs):
        self.seconds, self.compact_ties = seconds, compact_ties
        self.max_evaluations = max_evaluations
        self.decisions, self.stats = [], {}

    def __call__(self, s):
        started = time.perf_counter()
        ctx = EvaluationContext(s)
        base = density_initialize(s, ctx)
        initialized = time.perf_counter()
        oracle = EdgeOracle(s)
        best, evaluations = base, 0
        deadline = initialized + self.seconds
        def stop():
            return evaluations >= self.max_evaluations or time.perf_counter() >= deadline
        def evaluate(routes):
            nonlocal evaluations
            if stop():
                return None
            evaluations += 1
            pred = ctx.predictions(routes)
            if pred is None:
                return None
            captures, _ = pred
            releases = report_releases(s,captures,routes)
            order = oracle.select(releases, set(captures))
            return None if order is None else from_order(s,routes,order,'joint_destroy_repair')
        def consider(plan):
            nonlocal best
            if plan is not None and plan_key(s,plan) < plan_key(s,best):
                best = plan
        def repair(routes, strategy):
            plan = evaluate(routes)
            if plan is None:
                return
            consider(plan)
            while not stop():
                remaining = [j for j,st in s.status.items()
                             if st=='pending' and j not in plan.capture_times]
                options = []
                for j in sorted(remaining, key=lambda j:(-s.tasks[j].value,s.tasks[j].deadline,j)):
                    t = s.tasks[j]
                    for vid,v in enumerate(s.vehicles):
                        if v.sensor!=t.sensor:
                            continue
                        for pos in range(len(plan.routes[vid])+1):
                            routes = [list(r) for r in plan.routes]
                            routes[vid].insert(pos,j)
                            candidate = evaluate(routes)
                            if candidate is not None and candidate.value > plan.value+EPS:
                                gain = candidate.value-plan.value
                                denominator = max(EPS,max(0,candidate.travel-plan.travel)+t.service+t.work*s.multiplier)
                                score = gain/denominator if strategy=='density' else gain
                                options.append((-score,-gain,plan_key(s,candidate),candidate))
                                consider(candidate)
                            if stop():
                                break
                        if stop():
                            break
                    if stop():
                        break
                if not options:
                    break
                plan = min(options,key=lambda x:x[:-1])[-1]
                consider(plan)
        bound = independent_value_bound(s,ctx)
        can_change = any(v.active is None and any(st=='pending' and s.tasks[j].sensor==v.sensor
                          for j,st in s.status.items()) for v in s.vehicles) or (
                          s.edge_active is None and 'ready' in s.status.values())
        if can_change and (self.compact_ties or bound>base.value+EPS):
            # Exact fixed-route report selection is an admissible starting move.
            consider(evaluate(base.routes))
            jobs = sorted(base.capture_times,key=lambda j:(s.tasks[j].value,s.tasks[j].deadline,j))
            destroys = [()] + [(j,) for j in jobs] + list(itertools.combinations(jobs,2))
            for n,removed in enumerate(destroys):
                routes = tuple(tuple(j for j in r if j not in removed) for r in best.routes)
                repair(routes,'density' if n%2==0 else 'gain')
                if stop():
                    break
            # Relocation and pair exchange are evaluated as full joint plans.
            current = best
            positions = [(i,k,j) for i,r in enumerate(current.routes) for k,j in enumerate(r)]
            for vid,pos,j in positions:
                for other,v in enumerate(s.vehicles):
                    if v.sensor!=s.tasks[j].sensor:
                        continue
                    routes = [list(r) for r in current.routes]
                    routes[vid].pop(pos)
                    for k in range(len(routes[other])+1):
                        trial = [r[:] for r in routes]
                        trial[other].insert(k,j)
                        consider(evaluate(trial))
                        if stop():
                            break
                    if stop():
                        break
                if stop():
                    break
            if not stop():
                for (i,k,j),(v,l,h) in itertools.combinations(positions,2):
                    if s.tasks[j].sensor!=s.tasks[h].sensor:
                        continue
                    trial=[list(r) for r in current.routes]
                    trial[i][k],trial[v][l]=trial[v][l],trial[i][k]
                    consider(evaluate(trial))
                    if stop():
                        break
        reason = acceptance_reason(s,best,base,compact_ties=self.compact_ties)
        # Fairness (N1): a same-immediate-action plan with strictly higher
        # executable value is a real improvement for this control too. The
        # unguarded acceptance contract stays; only the discarded tail-gain
        # class is restored so the control matches RCR's ACCEPTED set.
        if reason == 'same_immediate_action' and best.value > base.value+EPS:
            reason = 'same_action_tail_gain'
        chosen = best if reason in ACCEPTED else base
        validate_plan(s,chosen)
        self.decisions.append(dict(event=s.event_index,time=s.now,mode=reason,
            baseline_value=base.value,chosen_value=chosen.value,
            planning_ms=1000*(time.perf_counter()-started),
            initialization_ms=1000*(initialized-started), evaluations=evaluations,
            stopped_by='budget' if time.perf_counter()>=deadline else 'work_limit' if evaluations>=self.max_evaluations else 'neighborhood_exhausted',
            edge_oracle=dict(oracle.stats),changed=chosen is not base))
        return chosen
