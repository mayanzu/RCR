"""Snapshot-local caches and an equivalent, gain-pruned density initializer.

The frozen executor is unchanged. Caches contain only data from one public
snapshot and are never reused after physical time or resource state changes.
"""
from __future__ import annotations
import heapq,math

from .boot import EPS, model, report_releases, from_order
from .transmission import route_releases
from .capacity import energy_violations
from .scheduling import RequiredScheduler


class EvaluationContext:
    def __init__(self, s, conflicts=None, energy=True):
        self.s = s
        self.base_reports = report_releases(s, {})
        self.route_cache = {}
        self.route_legs = {}
        self.route_reports = {}
        self.plan_cache = {}
        self.calls = self.hits = 0
        self.conflicts, self.energy = conflicts, energy
        self.energy_rejects = self.certificate_rejects = self.bound_stops = 0
        self.scheduler = RequiredScheduler(s)
        self.bases = [model.route_base(s, vid) for vid in range(len(s.vehicles))]
        self.compatible = {}
        for vid, vehicle in enumerate(s.vehicles):
            self.compatible.setdefault(vehicle.sensor, []).append(vid)
        self.reach_cache = {}

    def reach(self, j, vid):
        key = j, vid
        if key not in self.reach_cache:
            _, loc = self.bases[vid]
            task = self.s.tasks[j]
            self.reach_cache[key] = model.legacy.travel_time(*loc, task.x, task.y,
                                                            self.s.vehicles[vid].speed)
        return self.reach_cache[key]

    def route(self, vid, route):
        key = vid, tuple(route)
        if key in self.route_cache:
            return self.route_cache[key]
        s = self.s
        cursor, loc = self.bases[vid]
        captures, travel, legs = {}, 0.0, []
        vehicle = s.vehicles[vid]
        for j in route:
            task = s.tasks[j]
            if j in captures or s.status.get(j) != 'pending' or task.sensor != vehicle.sensor:
                self.route_cache[key] = None
                return None
            leg = model.legacy.travel_time(*loc, task.x, task.y, vehicle.speed)
            cursor += leg + task.service
            if cursor + task.upload + task.work * s.multiplier > task.deadline + EPS:
                self.route_cache[key] = None
                return None
            captures[j] = cursor
            travel += leg
            legs.append(leg)
            loc = task.x, task.y
        reports=route_releases(s,vid,captures)
        if any(reports[j]+s.tasks[j].work*s.multiplier>s.tasks[j].deadline+EPS for j in captures):
            self.route_cache[key]=None
            return None
        self.route_reports[key]=reports
        result = captures, travel
        self.route_cache[key] = result
        self.route_legs[key] = legs
        return result

    def predictions(self, routes):
        if len(routes) != len(self.s.vehicles):
            return None
        captures, travel = {}, 0.0
        for vid, route in enumerate(routes):
            result = self.route(vid, route)
            if result is None or captures.keys() & result[0].keys():
                return None
            captures.update(result[0])
            for leg in self.route_legs[vid, tuple(route)]:
                travel += leg
        return captures, travel

    def evaluate(self, routes, required=None, eligible_reports=None,
                 rules=('edf', 'reserved_edf', 'density')):
        self.calls += 1
        routes = tuple(tuple(r) for r in routes)
        required_key = None if required is None else frozenset(required)
        eligible_key = None if eligible_reports is None else frozenset(eligible_reports)
        key = routes, required_key, eligible_key, rules
        if key in self.plan_cache:
            self.hits += 1
            return self.plan_cache[key]
        pred = self.predictions(routes)
        if pred is None:
            self.plan_cache[key] = None
            return None
        s = self.s
        captures, travel = pred
        releases = dict(self.base_reports)
        for vid,route in enumerate(routes):
            releases.update(self.route_reports[vid,route])
        required = set(captures) if required is None else set(required)
        if eligible_reports is not None:
            eligible = set(eligible_reports) | set(captures)
            releases = {j: r for j, r in releases.items() if j in eligible}
        forced = {j: max(model.edge_cursor(s), releases[j]) for j in required if j in releases}
        if self.conflicts is not None and self.conflicts.match(s, forced) is not None:
            self.certificate_rejects += 1
            self.plan_cache[key] = None
            return None
        tight = self.energy and len(forced) >= 2 and any(
            2*self.scheduler.work[j] > self.scheduler.deadline[j]-r for j, r in forced.items())
        # Wide-window reports rarely violate energy. Only invoke the test when
        # a job occupies a substantial fraction of its remaining window.
        if tight and energy_violations([(r, self.scheduler.deadline[j], self.scheduler.work[j])
                                       for j, r in forced.items()], self.scheduler.base, limit=1):
            self.energy_rejects += 1
            self.plan_cache[key] = None
            return None
        plans = []
        seen_orders = set()
        all_required = required == releases.keys()
        for rule in rules:
            result = (self.scheduler.schedule(releases, rule) if all_required and rule in ('edf', 'reserved_edf', 'density')
                      else model.edge_schedule(s, releases, rule, required))
            if result is None:
                continue
            order, times = result
            if order in seen_orders:
                continue
            seen_orders.add(order)
            plans.append(model.Plan(routes, order, times, captures,
                sum(s.tasks[j].value for j in order), len(order),
                sum(end - s.tasks[j].release for j, (_, end) in times.items()),
                travel, rule))
        result = min(plans, key=lambda p: tuple(-x for x in p.score) + (p.key,)) if plans else None
        self.plan_cache[key] = result
        return result


def _ranked_choice(s,ctx,routes,selected_reports,remaining,existing,base,stop,
                   urgency_alpha=0., urgency_window=12.):
    """Lazily merge exact ordered report-drop streams for every insertion.

    Resource cost is independent of the dropped report. Drops sorted by value
    and id therefore generate nondecreasing full priority keys per insertion.
    The heap exposes the same global key order as materializing every trial.
    urgency_alpha > 0 multiplies only the density key of reports whose remaining
    processor window falls inside urgency_window, so near-deadline work ranks
    earlier without changing any value accounting or feasibility test.
    """
    route_jobs={j for route in routes for j in route}
    base_jobs=selected_reports|route_jobs
    ordered=sorted(base_jobs|remaining|existing,key=lambda j:(s.tasks[j].deadline,j))
    processing={j:s.tasks[j].work*s.multiplier for j in ordered}
    edge_cursor=model.edge_cursor(s)
    norm=max([1.,abs(edge_cursor),sum(abs(q) for q in processing.values())]
             +[abs(s.tasks[j].deadline) for j in ordered])
    tolerance=EPS*(len(ordered)+1)+8*len(ordered)*math.ulp(norm)
    capacity_cache={}
    # Prefix sums over the deadline-ordered list turn each capacity test into
    # arithmetic on base checkpoints instead of rescanning `ordered`.
    # required = (base_jobs | {j}) - {drop}; j is outside base_jobs and drop is
    # either None or inside base_jobs (drops come from selected_reports).
    # Ignore report releases: failure is a route-independent proof, while
    # passing is not a nonpreemptive feasibility certificate. The current plan
    # is valid, so base_jobs alone always satisfies every checkpoint.
    job_index={k:i for i,k in enumerate(ordered)}
    cap_minus_prefix=[]
    running=0.
    for i,k in enumerate(ordered):
        if k in base_jobs:
            running+=processing[k]
        cap_minus_prefix.append(max(0.,s.tasks[k].deadline-edge_cursor)+tolerance-running)
    ckpt=[i for i,k in enumerate(ordered) if k in base_jobs]
    def capacity_ok(j,drop):
        key=j,drop
        hit=capacity_cache.get(key)
        if hit is not None:return hit
        pj=processing[j];posj=job_index[j]
        pd=job_index[drop] if drop is not None else -1
        add_drop=processing[drop] if drop is not None else 0.
        ok=cap_minus_prefix[posj]+(add_drop if 0<=pd<=posj else 0.)>=pj
        if ok:
            for i in ckpt:
                if i==pd:continue
                adj=add_drop if 0<=pd<=i else 0.
                if cap_minus_prefix[i]+adj<(0. if i<posj else pj):
                    ok=False;break
        capacity_cache[key]=ok;return ok
    drops=[None]+sorted(selected_reports,key=lambda j:(s.tasks[j].value,j))
    streams=[];queue=[]
    # The current plan is valid. A trial changes exactly one route and inserts
    # a previously unassigned job, so no full capture-map merge is needed just
    # to rank it. Prefixes are accumulated in the original leg order; suffixes
    # remain individual legs, preserving the exact floating-point sum.
    legs=[ctx.route_legs[vid,tuple(route)] for vid,route in enumerate(routes)]
    lengths=[sum(route_legs) for route_legs in legs]
    prefixes=[];cursor=0.
    for route_legs in legs:
        prefixes.append(cursor)
        for leg in route_legs:cursor+=leg
    suffixes=[tuple(leg for later in legs[vid+1:] for leg in later) for vid in range(len(routes))]
    def admissible(j,k):
        while k<len(drops):
            drop=drops[k]
            gain=s.tasks[j].value-(s.tasks[drop].value if drop is not None else 0.)
            if gain<=EPS:return None
            if capacity_ok(j,drop):return k
            k+=1
        return None
    def travel_bound(j,vid):
        """Admissible lower bound of this vehicle's added travel for task j.

        The new route still starts at the vehicle base and must reach x_j, so
        its total travel time is at least the direct base leg and at least the
        current route length. Ignoring the rest only delays, never removes, the
        vehicle's insertion candidates.
        """
        return max(0.,ctx.reach(j,vid)-lengths[vid])
    def push(index,k):
        j,vid,pos,changed,resource=streams[index]
        k=admissible(j,k)
        if k is None:return
        drop=drops[k];gain=s.tasks[j].value-(s.tasks[drop].value if drop is not None else 0.)
        urg=1.+urgency_alpha*max(0.,urgency_window-(s.tasks[j].deadline-edge_cursor))/urgency_window
        key=(-gain*urg/resource,-gain,s.tasks[j].deadline,j,-1 if drop is None else drop,vid,pos)
        heapq.heappush(queue,(key,1,index,k))
    def stream(j,vid,pos,changed,resource):
        streams.append((j,vid,pos,changed,resource));push(len(streams)-1,0)
    for j in sorted(existing-selected_reports):
        stream(j,-1,-1,routes,max(EPS,s.tasks[j].work*s.multiplier))
    for j in sorted(remaining):
        task=s.tasks[j]
        first=admissible(j,0)
        if first is None:continue
        drop=drops[first];gain=task.value-(s.tasks[drop].value if drop is not None else 0.)
        # Ignoring the unmodelled part of marginal travel is an admissible
        # density bound. Expand one vehicle's route alternatives only when its
        # optimistic key reaches the heap head: far vehicles are deferred by a
        # larger bound and stay exactly enumerated, never pruned away.
        for vid in ctx.compatible.get(task.sensor, ()):
            urg=1.+urgency_alpha*max(0.,urgency_window-(task.deadline-edge_cursor))/urgency_window
            optimistic=(-gain*urg/max(EPS,travel_bound(j,vid)+task.service+task.work*s.multiplier),
                        -gain,task.deadline,j,-1 if drop is None else drop,-1,-1)
            heapq.heappush(queue,(optimistic,2,(j,vid),0))
    while queue:
        if stop is not None and stop():return None
        key,kind,index,k=heapq.heappop(queue)
        if kind in (0,2):
            j,vid=index if kind==2 else (index,-1)
            task=s.tasks[j]
            vehicles=enumerate(s.vehicles) if vid<0 else ((vid,s.vehicles[vid]),)
            for vid,vehicle in vehicles:
                if vehicle.sensor!=task.sensor:continue
                for pos in range(len(routes[vid])+1):
                    if stop is not None and stop():return None
                    changed=[r[:] for r in routes];changed[vid].insert(pos,j)
                    if ctx.route(vid,changed[vid]) is None:continue
                    travel=prefixes[vid]
                    for leg in ctx.route_legs[vid,tuple(changed[vid])]:travel+=leg
                    for leg in suffixes[vid]:travel+=leg
                    resource=max(EPS,max(0.,travel-base.travel)+task.service+task.work*s.multiplier)
                    stream(j,vid,pos,changed,resource)
            continue
        j,vid,pos,changed,resource=streams[index];drop=drops[k]
        retained=selected_reports-({drop} if drop is not None else set())
        required=retained|route_jobs|{j}
        trial=ctx.evaluate(changed,required=required,eligible_reports=required)
        if trial is not None:
            assert abs(trial.value-base.value+key[1])<=EPS
            return (*key,trial,retained|{j} if vid<0 else retained)
        push(index,k+1)
    return None


def local_equal_value_probe(s, ctx, baseline, trials=12):
    """Bounded equal-value neighborhood scan around a validated incumbent.

    Used at the certified value ceiling where a full density rebuild cannot
    raise value: the only open question is whether another immediate action
    (or a more compact equal-value plan) exists nearby. Cheap evidence for a
    resource-only search, never a complete enumeration of equal-value plans.
    Returns the best equal-value plan found, or None.
    """
    if baseline is None:
        return None
    best = None

    def consider(plan):
        nonlocal best
        if plan is None or plan.value < baseline.value - EPS:
            return
        if best is None or (-plan.value, plan.travel, plan.wait) < (-best.value, best.travel, best.wait):
            best = plan

    # Move each of the first few non-head reports to the front of the edge
    # order: a different first ready job is a different irreversible action.
    order = list(baseline.edge_order)
    for j in order[1:5]:
        trial_order = (j, *[k for k in order if k != j])
        try:
            consider(from_order(s, baseline.routes, trial_order, 'tie_probe_edge'))
        except (AssertionError, KeyError):
            pass
        if trials <= 0:
            return best
        trials -= 1
    # Idle vehicles can change their first acquisition; that also changes the
    # action key. Pending alternatives are probed in value order.
    pending = sorted((j for j, st in s.status.items() if st == 'pending'),
                     key=lambda j: (-s.tasks[j].value, s.tasks[j].deadline, j))
    for vid, vehicle in enumerate(s.vehicles):
        if vehicle.active is not None:
            continue
        route = baseline.routes[vid]
        for j in pending:
            if j in route or s.tasks[j].sensor != vehicle.sensor:
                continue
            # Prepending or replacing the head changes the irreversible start;
            # inserting behind the head does not and is left to the search.
            for changed in ((j, *route), (j, *route[1:])):
                if trials <= 0:
                    return best
                trials -= 1
                routes = [list(r) for r in baseline.routes]
                routes[vid] = list(changed)
                consider(ctx.evaluate(routes, required={k for r in routes for k in r}))
            break
    return best


def density_initialize(s, context=None, routes=None, selected_reports=None, stop=None,
                       ranked=True, value_upper=None, urgency_alpha=0., urgency_window=12.):
    """Frozen density decision rule with equivalent pruning and local caches.

    Optional partial routes support repair and diagnostic callers. The default
    empty start reproduces the frozen initializer's complete choice order.
    urgency_alpha > 0 only reorders positive-gain insertions inside the ranked
    stream; value accounting and feasibility are unchanged (and the default
    keeps the frozen decision rule exactly).
    """
    ctx = context or EvaluationContext(s)
    routes = [list(r) for r in (model.empty_routes(s) if routes is None else routes)]
    selected_reports = set() if selected_reports is None else set(selected_reports)
    route_jobs = {j for r in routes for j in r}
    base = ctx.evaluate(routes, required=selected_reports | route_jobs,
                        eligible_reports=selected_reports | route_jobs)
    if base is None:
        return None
    remaining = {j for j, st in s.status.items() if st == 'pending'} - route_jobs
    existing = set(ctx.base_reports)
    # Integer service priorities give exact gains independently of report order.
    # Arbitrary real-valued inputs retain exhaustive evaluation, avoiding changes
    # in floating-point ties caused by a different sum order.
    ranked = (ranked and all(t.value>0 and float(t.value).is_integer() for t in s.tasks.values())
              and sum(abs(t.value) for t in s.tasks.values()) <= 2**52)
    while True:
        # Every accepted insertion has positive raw-value gain. A certified
        # upper bound therefore proves the rest of this greedy scan is empty.
        if value_upper is not None and base.value >= value_upper:
            ctx.bound_stops += 1
            return base
        if ranked:
            chosen=_ranked_choice(s,ctx,routes,selected_reports,remaining,existing,base,stop,
                                 urgency_alpha=urgency_alpha,urgency_window=urgency_window)
            if chosen is None:return base
            j,drop,vid,pos,base,retained=chosen[3:]
            if vid>=0:
                routes[vid].insert(pos,j);remaining.remove(j)
            selected_reports=retained
            continue
        choices = []
        route_jobs = {j for route in routes for j in route}
        drops = [None, *sorted(selected_reports)]
        for drop in drops:
            retained = selected_reports - ({drop} if drop is not None else set())
            for j in sorted(existing - selected_reports):
                if stop is not None and stop():
                    return base
                gain = s.tasks[j].value - (s.tasks[drop].value if drop is not None else 0)
                if gain <= EPS:
                    continue
                req = retained | route_jobs | {j}
                trial = ctx.evaluate(routes, required=req, eligible_reports=req)
                if trial is not None:
                    gain = trial.value - base.value
                    if gain <= EPS:
                        continue
                    resource = max(EPS, s.tasks[j].work * s.multiplier)
                    choices.append((-gain/resource, -gain, s.tasks[j].deadline,
                                    j, -1 if drop is None else drop, -1, -1,
                                    trial, retained | {j}))
        for j in sorted(remaining):
            task = s.tasks[j]
            for vid, vehicle in enumerate(s.vehicles):
                if vehicle.sensor != task.sensor:
                    continue
                for pos in range(len(routes[vid]) + 1):
                    changed = [r[:] for r in routes]
                    changed[vid].insert(pos, j)
                    if ctx.route(vid, changed[vid]) is None:
                        continue
                    for drop in drops:
                        if stop is not None and stop():
                            return base
                        gain = task.value - (s.tasks[drop].value if drop is not None else 0)
                        if gain <= EPS:
                            continue
                        retained = selected_reports - ({drop} if drop is not None else set())
                        req = retained | route_jobs | {j}
                        trial = ctx.evaluate(changed, required=req, eligible_reports=req)
                        if trial is None:
                            continue
                        gain = trial.value - base.value
                        if gain <= EPS:
                            continue
                        resource = max(EPS, max(0.0, trial.travel-base.travel)
                                       + task.service + task.work*s.multiplier)
                        choices.append((-gain/resource, -gain, task.deadline,
                                        j, -1 if drop is None else drop, vid, pos,
                                        trial, retained))
        if not choices:
            return base
        chosen = min(choices, key=lambda x: x[:-2])
        j, drop, vid, pos, base, retained = chosen[3:]
        if vid >= 0:
            routes[vid].insert(pos, j)
            remaining.remove(j)
        selected_reports = retained
