"""Short-route packing with lexicographic value protection and exact decoding."""
from __future__ import annotations

import time
import math
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .boot import EPS, from_order, model, validate_plan, report_releases
from .transmission import route_releases
from .edge import EdgeOracle, EXACT_MAX
from .capacity import mandatory_overlap, energy_violations, energy_tolerance
from .criteria import acceptance_reason, plan_resources, sunk_value, slack_floor, ACCEPTED
from .continuation import ActionArchive

# Plan Step 3: a vehicle in a tight slack window chains 1--3 tasks.
MAX_DEPTH = 3
MAX_ROUTES_PER_VEHICLE = 64
# Generate a wider pool than we keep, then curate it (see enumerate_routes).
ROUTE_POOL_FACTOR = 4
# Dual-criterion curation of the multi-stop pool. The first share keeps value
# per occupied vehicle minute; the remainder keeps paths whose tightest report
# still leaves processor slack, so early-release alternatives survive for the
# release-floor cuts instead of being crowded out by value density alone.
POOL_DENSITY_SLOTS = 40
POOL_SLACK_SLOTS = 24
# Single-machine ordering is solved exactly when the job set is small enough.
EXACT_EDGE_MAX_JOBS = EXACT_MAX
# Lazy Horn separation: cut-and-solve rounds before accepting the incumbent.
HORN_ROUNDS = 6


@dataclass(frozen=True, eq=False)
class Route:
    """One feasible short path with all timings precomputed as constants."""
    vehicle: int
    tasks: tuple
    captures: dict          # j -> absolute capture finish
    reports: dict           # j -> absolute edge-ready (report release)
    travel: float
    occupancy: float        # sum(leg + service) consumed on the vehicle
    finish: float           # absolute time the vehicle frees up

    def __hash__(self):
        # Value identity is the path itself; timings are derived from it.
        return hash((self.vehicle, self.tasks))

    def __eq__(self, other):
        return (isinstance(other, Route) and self.vehicle == other.vehicle
                and self.tasks == other.tasks)

    @property
    def head(self):
        return self.tasks[0] if self.tasks else None

    @property
    def task_set(self):
        return frozenset(self.tasks)


def enumerate_routes(s, vid, max_depth=MAX_DEPTH, max_routes=MAX_ROUTES_PER_VEHICLE,
                     beta=0.0, anchor=(), focus=(), stop=None, pool_factor=ROUTE_POOL_FACTOR):
    """Enumerate physically feasible paths; risk never removes a path.

    All feasible singletons plus the density anchor survive curation. The cap
    applies to the remaining multi-stop pool, not the required anchor paths.
    beta is accepted for compatibility and is used only by the packer.
    pool_factor scales the generation budget wider than the kept cap before
    curation (default matches the frozen rule).
    """
    v = s.vehicles[vid]
    base_time, base_loc = model.route_base(s, vid)
    options = []
    for j, st in s.status.items():
        if st != 'pending':
            continue
        t = s.tasks[j]
        if t.sensor != v.sensor:
            continue
        leg = model.legacy.travel_time(*base_loc, t.x, t.y, v.speed)
        # A task cannot be captured before it is released. Vacuous for real
        # snapshots (pending implies released); needed for synthetic futures.
        if base_time + leg < t.release - EPS:
            continue
        finish = base_time + leg + t.service
        if finish + t.upload + t.work * s.multiplier > t.deadline + EPS:
            continue
        options.append((j, leg, finish))
    # Promising tasks first so the route cap keeps good paths.
    focus = set(focus)
    options.sort(key=lambda o: (o[0] not in focus, s.tasks[o[0]].deadline, -s.tasks[o[0]].value))

    # The contention load is a property of the snapshot, not of the route.
    # Risk is a secondary objective only; it never filters routes.

    # Generate wider than the cap and curate below: with a LIFO stack the
    # expansion order decides which paths survive truncation, so the stack is
    # fed in reverse and popped best-head-first.
    budget = max_routes * pool_factor
    routes = [Route(vid, (), {}, {}, 0.0, 0.0, base_time)]
    stack = [((), base_time, base_loc, 0.0, {}, {}, 0.0)]
    while stack and len(routes) < budget:
        prefix, cursor, loc, travel, captures, reports, occupancy = stack.pop()
        if prefix and stop is not None and stop():
            break
        if len(prefix) >= max_depth:
            continue
        for j, _, _ in reversed(options):
            if prefix and stop is not None and stop():
                break
            if j in captures:
                continue
            t = s.tasks[j]
            nxt_loc = (t.x, t.y)
            # Leg must be measured from the current position, not the base.
            leg = model.legacy.travel_time(*loc, t.x, t.y, v.speed)
            if cursor + leg < t.release - EPS:
                continue
            finish = cursor + leg + t.service
            if finish + t.upload + t.work * s.multiplier > t.deadline + EPS:
                continue
            cap = dict(captures); cap[j] = finish
            rep = route_releases(s,vid,cap)
            if any(rep[k]+s.tasks[k].work*s.multiplier>s.tasks[k].deadline+EPS for k in cap):
                continue
            occ = occupancy + leg + t.service
            path = prefix + (j,)
            routes.append(Route(vid, path, cap, rep, travel + leg, occ, finish))
            stack.append((path, finish, nxt_loc, travel + leg, cap, rep, occ))
    def quality(w):
        value = sum(s.tasks[k].value for k in w.tasks)
        return (-value / max(w.occupancy, EPS), -value, w.finish, w.tasks)

    def edge_slack(w):
        """Tightest report margin this path leaves on the shared processor."""
        return min(s.tasks[k].deadline - w.reports[k] - s.tasks[k].work * s.multiplier
                   for k in w.tasks)
    # Every physically feasible singleton stays available. Baseline routes are
    # injected even when longer than the search depth, protecting the anchor.
    singles = [w for w in routes if len(w.tasks) == 1]
    rest = [w for w in routes if len(w.tasks) > 1]
    kept = [routes[0]] + singles
    multi_budget = max(0, max_routes - len(kept))
    share = POOL_DENSITY_SLOTS / (POOL_DENSITY_SLOTS + POOL_SLACK_SLOTS)
    density_slots = max(1, int(round(multi_budget * share))) if multi_budget else 0
    kept += sorted(rest, key=quality)[:density_slots]
    seen = {w.tasks for w in kept}
    # The slack pool keeps early-release paths available. Under tight deadlines
    # these are the alternatives a release-floor cut needs in order to move a
    # report's release earlier rather than dropping the acquisition outright.
    by_slack = sorted((w for w in rest if w.tasks not in seen),
                      key=lambda w: (edge_slack(w),
                                     -sum(s.tasks[k].value for k in w.tasks),
                                     w.tasks), reverse=True)
    kept += by_slack[:multi_budget - density_slots]
    if focus:
        # Conflict-directed alternatives supplement, rather than evict, the
        # density/slack pool. Earlier releases can escape a stored floor cut.
        seen = {w.tasks for w in kept}
        focused = [w for w in rest if w.tasks not in seen and focus & w.task_set]
        focused.sort(key=lambda w: (sum(w.reports[j] for j in focus & w.task_set)
                                    / len(focus & w.task_set), quality(w)))
        kept += focused[:8]
    anchor = tuple(anchor)
    # A long density anchor must also have local alternatives. Otherwise the
    # pool can retain a five-stop anchor but only three-stop replacements.
    protected = [anchor]
    if len(anchor) > max_depth:
        protected += [anchor[:k]+anchor[k+1:] for k in range(len(anchor))]
        protected += [anchor[:k]+(anchor[k+1],anchor[k])+anchor[k+2:]
                      for k in range(len(anchor)-1)]
    for path in protected:
        if not path or any(w.tasks == path for w in kept):
            continue
        raw = [() for _ in s.vehicles]
        raw[vid] = path
        pred = model.route_predictions(s, raw)
        if pred is not None:
            captures, travel = pred
            reports=route_releases(s,vid,captures)
            if any(reports[j]+s.tasks[j].work*s.multiplier>s.tasks[j].deadline+EPS for j in captures):
                continue
            finish = captures[path[-1]]
            kept.append(Route(vid, path, captures,
                              reports,
                              travel, finish - base_time, finish))
    return kept


def anchor_routes(s, vid, anchor):
    """Route pool of a vehicle outside the active subset.

    Such a vehicle keeps exactly its initialized path (plus the empty route).
    Its acquisitions and report releases stay inside the master, so edge
    reordering and release-floor conflicts still see them; only its unexplored
    route alternatives are removed from the search pool.
    """
    base_time, base_loc = model.route_base(s, vid)
    kept = [Route(vid, (), {}, {}, 0.0, 0.0, base_time)]
    path = tuple(anchor)
    if not path:
        return kept
    raw = [() for _ in s.vehicles]
    raw[vid] = path
    pred = model.route_predictions(s, raw)
    if pred is None:
        return kept
    captures, travel = pred
    reports = route_releases(s, vid, captures)
    if any(reports[j] + s.tasks[j].work * s.multiplier > s.tasks[j].deadline + EPS
           for j in captures):
        return kept
    finish = captures[path[-1]]
    kept.append(Route(vid, path, captures, reports, travel, finish - base_time, finish))
    return kept


class _Builder:
    """Minimal sparse MILP builder; no time variables, no Big-M rows."""

    def __init__(self):
        self.keys, self.index, self.ub, self.integer, self.objective = [], {}, [], [], []
        self.rows, self.cols, self.data, self.lo, self.hi = [], [], [], [], []

    def var(self, key, ub=1.0, integer=True, objective=0.0):
        i = len(self.keys)
        self.keys.append(key); self.index[key] = i
        self.ub.append(ub); self.integer.append(int(integer)); self.objective.append(objective)
        return i

    def row(self, items, lo=-np.inf, hi=np.inf):
        rid = len(self.lo)
        for k, v in items.items():
            if v:
                self.rows.append(rid); self.cols.append(k); self.data.append(float(v))
        self.lo.append(lo); self.hi.append(hi)

    def solve(self, seconds):
        a = coo_array((np.array(self.data, dtype=float),
                       (np.array(self.rows, dtype=np.int32), np.array(self.cols, dtype=np.int32))),
                      shape=(len(self.lo), len(self.keys))).tocsc()
        # SciPy forwards this HiGHS option verbatim; suppress only the known
        # pass-through notice, not solver errors or other warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r"Unrecognized options detected: \{'threads'\}.*", category=RuntimeWarning)
            return milp(np.asarray(self.objective, dtype=float),
                    integrality=np.asarray(self.integer, dtype=np.int32),
                    bounds=Bounds(np.zeros(len(self.keys)), np.asarray(self.ub, dtype=float)),
                    constraints=LinearConstraint(a, np.asarray(self.lo, dtype=float),
                                                 np.asarray(self.hi, dtype=float)),
                    options={'presolve': True, 'mip_rel_gap': 0.0, 'time_limit': seconds,
                             'threads': 1})


def _horn_terms(s, y, z, routes_by_vehicle, report_release, a, bnd, margin=0.):
    """Nonpreemptive mandatory-overlap coefficients for a valid interval cut.

    Exact path-dependent releases are clamped to the end of locked edge work.
    Satisfying these rows does not certify a nonpreemptive ordering.
    """
    proc = {j: s.tasks[j].work * s.multiplier for j in s.status}
    dead = {j: s.tasks[j].deadline - s.now - margin for j in s.status}
    terms, total = {}, 0.0
    for w_list in routes_by_vehicle:
        for w in w_list:
            load = sum(mandatory_overlap(max(model.edge_cursor(s), w.reports[j])-s.now,
                                        dead[j], proc[j], a, bnd) for j in w.tasks)
            if not load:
                continue
            terms[y[w]] = terms.get(y[w], 0.0) + load
            total += load
    for j, r in report_release.items():
        load = mandatory_overlap(max(model.edge_cursor(s), r)-s.now, dead[j], proc[j], a, bnd)
        if load:
            terms[z[j]] = terms.get(z[j], 0.0) + load
            total += load
    return terms, total


def _violated_intervals(s, chosen_routes, selected_reports, report_release, edge_base,
                        limit=6, margin=0.):
    """Capacity intervals violated by one integer solution (lazy separation).

    Generating every interval up front costs O(points^2 * routes) and blows up
    on long queues; only the intervals the current solution actually violates
    can ever be needed, and those are found on the *selected* jobs alone.
    """
    proc = {j: s.tasks[j].work * s.multiplier for j in s.status}
    dead = {j: s.tasks[j].deadline - s.now - margin for j in s.status}
    jobs = []
    for w in chosen_routes:
        for j in w.tasks:
            jobs.append((max(model.edge_cursor(s), w.reports[j]) - s.now, dead[j], proc[j]))
    for j in selected_reports:
        if j in report_release:
            jobs.append((max(model.edge_cursor(s), report_release[j])-s.now, dead[j], proc[j]))
    if not jobs:
        return []
    return energy_violations(jobs, edge_base, limit, full=True)


def contention_rate(s):
    if not s.tasks:
        return 0.0
    span = max(s.now, max(t.release for t in s.tasks.values()), 1.0)
    return sum(t.work * s.multiplier for t in s.tasks.values()) / span


def route_risk(s, route, beta):
    """Bounded soft score, activated only by an actual waiting report."""
    if not any(st == 'ready' for st in s.status.values()):
        return 0.0
    rate = contention_rate(s)
    risk = 0.0
    for j in route.tasks:
        window = s.tasks[j].deadline - route.reports[j]
        slack = window - s.tasks[j].work * s.multiplier
        if slack + EPS < beta * rate * window:
            risk += 0.05 * s.tasks[j].value
    return risk


def _selected_plan(s, routes, selected, oracle=None):
    tuples = [() for _ in s.vehicles]
    for w in routes:
        tuples[w.vehicle] = w.tasks
    tuples = tuple(tuples)
    pred = model.route_predictions(s, tuples)
    if pred is None:
        raise AssertionError('packer returned invalid vehicle routes')
    releases = report_releases(s, pred[0],tuples)
    selected = set(selected)
    if not set(pred[0]).issubset(selected):
        raise AssertionError('acquisition missing from processing selection')
    oracle = oracle or EdgeOracle(s)
    order, proof = oracle.fixed({j: releases[j] for j in selected})
    return (None if order is None else from_order(s, tuples, order, 'route_checked')), proof


def _release_conflict_cut(s, builder, y, z, routes_by_vehicle, report_release, floors):
    # Each job indicator counts a route only if it releases that job no earlier
    # than the certified conflict floor. Every term is 0/1 under task uniqueness.
    terms = {}
    for routes in routes_by_vehicle:
        for w in routes:
            coefficient = sum(j in floors and max(model.edge_cursor(s), r) >= floors[j]
                              for j, r in w.reports.items())
            if coefficient:
                terms[y[w]] = coefficient
    for j, r in report_release.items():
        if j in floors and max(model.edge_cursor(s), r) >= floors[j]:
            terms[z[j]] = 1
    builder.row(terms, hi=len(floors) - 1)



def solve_set_packing(s, routes_by_vehicle, report_release, beta=0.25, seconds=0.1,
                      baseline=None, strict_value_only=False, conflict_cuts=True,
                      conflict_store=None, align_ties=True):
    """Raw-value first, compactness second; lazy nonpreemptive feasibility cuts.

    The caller passes the remaining additional-search budget, including route
    generation. At most eight certified release-floor conflicts are added.
    Conflicts cover both new acquisitions and existing report selections.
    An unknown large-instance decode failure falls back without adding a cut.
    """
    started = time.perf_counter()
    now, edge_base = s.now, model.edge_cursor(s) - s.now
    b, y = _Builder(), {}
    secondary = {}
    for vid, routes in enumerate(routes_by_vehicle):
        for w in routes:
            y[w] = b.var(('y', vid, w.tasks))
            secondary[y[w]] = (w.travel + w.occupancy
                              + sum(r - now for r in w.reports.values())
                              + route_risk(s, w, beta))
        b.row({y[w]: 1 for w in routes}, hi=1)
    captured = {}
    for ws in routes_by_vehicle:
        for w in ws:
            for j in w.tasks:
                captured.setdefault(j, []).append(w)
    jobs = sorted(set(captured) | set(report_release))
    z = {j: b.var(('z', j), objective=-s.tasks[j].value) for j in jobs}
    # An acquired report already consumed vehicle travel and collection time.
    # Retention of those reports is optimized before the resource objective, so
    # equal-value plans cannot discard a sunk acquisition for a shorter trip.
    existing_reports = set(report_release)
    retention = {z[j]: s.tasks[j].value for j in jobs if j in existing_reports}
    for j, ws in captured.items():
        b.row({z[j]: 1, **{y[w]: -1 for w in ws}}, lo=0, hi=0)
    for j in jobs:
        if s.tasks[j].deadline - now - s.tasks[j].work * s.multiplier < edge_base - EPS:
            b.row({z[j]: 1}, hi=0)
    # Clamp releases to the end of locked edge work for all capacity tests.
    effective_reports = {j: max(r, model.edge_cursor(s)) for j, r in report_release.items()}
    # A source endpoint at edge_base includes all jobs; other endpoints use
    # max(edge_base, release) in both separation and row generation.
    oracle = EdgeOracle(s, store=conflict_store if conflict_cuts else None)
    reused_cuts = 0
    if conflict_cuts and conflict_store is not None:
        for floors in conflict_store.active(s, jobs):
            _release_conflict_cut(s, b, y, z, routes_by_vehicle, effective_reports, floors)
            reused_cuts += 1
    if baseline is not None:
        b.row({z[j]: s.tasks[j].value for j in jobs}, lo=baseline.value - EPS)
    cuts = nogoods = solves = 0
    stage_solves = Counter()
    # For integer report values the retention lattice has a certified positive
    # gap. A bounded resource term (< 1/2) cannot trade away one lattice unit.
    ret_values = list(retention.values())
    integer_retention = all(float(v).is_integer() for v in ret_values) and sum(abs(v) for v in ret_values) <= 2**52
    retention_gap = math.gcd(*(int(v) for v in ret_values)) if ret_values and integer_retention else 1
    resource_bound = sum(max((secondary[y[w]] for w in routes), default=0.)
                         for routes in routes_by_vehicle)
    encoding = 'bounded_integer_retention' if integer_retention else 'exact_fractional_fallback'
    conflict_sizes = []
    plan, chosen, selected = None, [], set()
    best = baseline

    def compact(p):
        # DP minimizes makespan, not total completion. Compare actual plans.
        return p.travel + sum(p.capture_times[r[-1]] - model.route_base(s, i)[0]
                              for i, r in enumerate(p.routes) if r) + p.wait

    def rank(p):
        # Value first, then retained sunk reports, then resource compactness.
        sunk = sum(s.tasks[j].value for j in p.edge_order if j in existing_reports)
        return (-p.value, -sunk, compact(p))
    archive = ActionArchive(s, rank)
    if baseline is not None:
        archive.add(baseline)
    action_rejections = 0
    margin = slack_floor(s, baseline) if baseline is not None else 0.
    stage = 'value'
    status = None
    primary_bound = None
    primary_optimal = False
    tie_mode = False
    tie_rows = 0
    reason = 'budget_exhausted'
    separated_intervals = set()
    def stop():
        return time.perf_counter()-started >= seconds

    def align_equal_value():
        """Only necessary acceptance rows; scheduling still validates the sum."""
        old = plan_resources(s, baseline)
        b.row({y[w]: w.travel for ws in routes_by_vehicle for w in ws}, hi=old[0]+EPS)
        b.row({y[w]: w.occupancy for ws in routes_by_vehicle for w in ws}, hi=old[1]+EPS)
        existing = {j for j, st in s.status.items() if st in ('ready', 'uploading')}
        b.row({z[j]: s.tasks[j].value for j in jobs if j in existing},
              lo=sunk_value(s, baseline, existing)-EPS)
        lower_sum = {}
        for ws in routes_by_vehicle:
            for w in ws:
                lower_sum[y[w]] = sum(max(model.edge_cursor(s), w.reports[j])+s.tasks[j].work*s.multiplier-now
                                      for j in w.tasks)
                vid = w.vehicle
                old_route = baseline.routes[vid]
                old_head = baseline.capture_times[old_route[0]]-now if old_route else 0.
                if (s.vehicles[vid].active is None and w.tasks
                        and w.captures[w.tasks[0]]-now > old_head+EPS):
                    b.ub[y[w]] = 0.
                if any(max(model.edge_cursor(s), w.reports[j])+s.tasks[j].work*s.multiplier
                       > s.tasks[j].deadline-margin+EPS for j in w.tasks):
                    b.ub[y[w]] = 0.
        for j, r in effective_reports.items():
            lower_sum[z[j]] = r+s.tasks[j].work*s.multiplier-now
            if r+s.tasks[j].work*s.multiplier > s.tasks[j].deadline-margin+EPS:
                b.ub[z[j]] = 0.
        b.row(lower_sum, hi=old[2]+EPS)
        return 4
    while solves < 2 * HORN_ROUNDS + 4:
        remaining = seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        res = b.solve(max(0.001, remaining))
        solves += 1
        stage_solves[stage] += 1
        status = int(res.status)
        if status in (2, 3, 4) or res.x is None:
            reason = 'no_solver_incumbent'
            break
        chosen = [w for w, idx in y.items() if res.x[idx] > 0.5]
        selected = {j for j, idx in z.items() if res.x[idx] > 0.5}
        if stage == 'value':
            bound = getattr(res, 'mip_dual_bound', None)
            primary_bound = None if bound is None else float(-bound)
        if (strict_value_only and baseline is not None and stage == 'value'
                and res.status == 0 and -res.fun <= baseline.value + EPS):
            primary_optimal = True
            reason = 'pool_value_bound'
            break
        broken = _violated_intervals(s, chosen, selected, effective_reports, edge_base,
                                     margin=margin if tie_mode else 0.)
        if broken:
            if conflict_cuts and conflict_store is not None and not tie_mode:
                a, end = broken[0]
                releases = dict(effective_reports)
                for w in chosen:
                    releases.update({j: max(model.edge_cursor(s), r) for j, r in w.reports.items()})
                # Positive contributors to an overloaded energy interval form
                # an independent infeasibility certificate of any cardinality.
                floors = {j: releases[j] for j in selected
                          if mandatory_overlap(releases[j]-now, s.tasks[j].deadline-now,
                                               s.tasks[j].work*s.multiplier, a, end) > 0.}
                conflict_store.add(s, floors)
            added = False
            for a, end in broken:
                key = (tie_mode, round(a, 7), round(end, 7))
                if key in separated_intervals:
                    continue
                terms, _ = _horn_terms(s, y, z, routes_by_vehicle, effective_reports, a, end,
                                       margin=margin if tie_mode else 0.)
                b.row(terms, hi=end-a+EPS*(len(jobs)+2))
                separated_intervals.add(key)
                cuts += 1
                added = True
            if added:
                continue
            # A solver-tolerance repeat is sent to the checked oracle instead
            # of spending all remaining rounds on the same nearly tight row.
        plan, infeasible = _selected_plan(s, chosen, selected, oracle)
        if plan is None:
            if not infeasible:
                reason = 'decode_unknown'
                break
            if nogoods >= 8:
                reason = 'no_good_limit'
                break
            if conflict_cuts:
                releases = dict(effective_reports)
                for w in chosen:
                    releases.update({j: max(model.edge_cursor(s), r) for j, r in w.reports.items()})
                floors = oracle.minimal_conflict({j: releases[j] for j in selected}, stop=stop)
                if len(floors) <= 8 and not stop():
                    floors = oracle.strengthen(floors, stop=stop)
                _release_conflict_cut(s, b, y, z, routes_by_vehicle, effective_reports, floors)
                if conflict_store is not None and not tie_mode:
                    conflict_store.add(s, floors)
                conflict_sizes.append(len(floors))
            else:
                literals = [y[w] for w in chosen if w.tasks]
                literals += [z[j] for j in selected if j in report_release]
                b.row({idx: 1 for idx in literals}, hi=len(literals)-1)
            nogoods += 1
            continue
        validate_plan(s, plan)
        acceptable = (baseline is None or acceptance_reason(s, plan, baseline,
                      guarded=True, compact_ties=not strict_value_only) in ACCEPTED)
        if acceptable:
            archive.add(plan)
            best = archive.best()
        elif baseline is not None and plan.value > baseline.value+EPS:
            # Behavioral exclusion, not an infeasibility certificate. Exclude
            # this exact route/report assignment only; never persist it.
            ones = [i for i, x in enumerate(res.x) if x > .5]
            zeros = [i for i, x in enumerate(res.x) if x <= .5]
            b.row({**{i:1 for i in ones}, **{i:-1 for i in zeros}}, hi=len(ones)-1)
            action_rejections += 1
            if action_rejections >= 4:
                reason = 'action_guard_limit'
                break
            continue
        if stage == 'compact':
            reason = 'compact_solution'
            break
        if status != 0:
            reason = 'time_limited_value_solution'
            break
        if stage == 'value':
            primary_optimal = True
            # True lexicographic optimization: secondary terms cannot trade away
            # even a small fractional increment in raw value.
            b.row({z[j]: s.tasks[j].value for j in jobs},
                  lo=plan.value - EPS, hi=plan.value + EPS)
            if align_ties and baseline is not None and abs(plan.value-baseline.value) <= EPS:
                tie_rows = align_equal_value()
                tie_mode = True
                # Margin-derived conflicts are local to this branch and must
                # never enter the cross-event, original-deadline store.
                oracle = EdgeOracle(s, margin=margin)
            if integer_retention:
                # At most one route per vehicle, hence 0 <= C <= resource_bound.
                # This is a derived bound, not a hand-chosen big weight.
                b.objective = [-retention.get(i, 0.0)/retention_gap
                               + secondary.get(i, 0.0)/(2*resource_bound+1)
                               for i in range(len(b.keys))]
                stage = 'compact'
            else:
                # Arbitrary fractional retention values have no assumed unit
                # gap; preserve exact separate objectives rather than guessing.
                b.objective = [-retention.get(i, 0.0) for i in range(len(b.keys))]
                stage = 'retain'
            continue
        kept = sum(s.tasks[j].value for j in selected if j in existing_reports)
        b.row({z[j]: s.tasks[j].value for j in existing_reports}, lo=kept - EPS)
        scale = max(1.0, sum(secondary.values()))
        b.objective = [secondary.get(i, 0.0) / scale for i in range(len(b.keys))]
        stage = 'compact'
    meta = dict(mode='route_set_packing', status=status, stop_reason=reason,
                certification=('restricted_master_optimal' if action_rejections else 'route_pool_value_optimal')
                    if primary_optimal else 'feasible_or_fallback',
                primary_dual_bound=primary_bound,
                variables=len(b.keys), constraints=len(b.lo), horn_cuts=cuts,
                no_good_cuts=nogoods, solves=solves, plan=best,
                routes_enumerated=sum(len(r) for r in routes_by_vehicle),
                stage_solves=dict(stage_solves), secondary_encoding=encoding,
                resource_bound=resource_bound, solver_threads=1,
                elapsed_seconds=time.perf_counter() - started,
                solver_budget_seconds=seconds, edge_oracle=dict(oracle.stats),
                conflict_sizes=conflict_sizes, selected_jobs=len(selected),
                reused_conflict_cuts=reused_cuts, tie_acceptance_rows=tie_rows,
                action_candidates=len(archive.plans), action_duplicates=archive.duplicates,
                action_rejections=action_rejections,
                pool_sizes=[len(r) for r in routes_by_vehicle])
    return chosen, selected, meta


def plan_from_tuples(s, routes, selected=None):
    pred = model.route_predictions(s, routes)
    if pred is None:
        return None
    releases = report_releases(s, pred[0],routes)
    required = set(pred[0]) | (set(selected) if selected is not None else set())
    order = EdgeOracle(s).select(releases, required)
    return None if order is None else from_order(s, routes, order, 'route_checked')


def build_plan(s, chosen_routes, selected=None):
    routes = [() for _ in s.vehicles]
    for w in chosen_routes:
        routes[w.vehicle] = w.tasks
    return plan_from_tuples(s, tuple(routes), selected)
