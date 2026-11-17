"""The acceptance contract shared by search and physical dispatch."""
from .boot import EPS, action_key, model

SLACK_GATE = 1.0
ACCEPTED = ('strict_value_gain', 'equal_value_resource_dominance', 'same_action_tail_gain')


def plan_resources(s, plan):
    occupancy = sum(plan.capture_times[r[-1]]-model.route_base(s, i)[0]
                    for i, r in enumerate(plan.routes) if r)
    return (plan.travel, occupancy, sum(end-s.now for _, end in plan.edge_times.values()))


def plan_min_slack(s, plan):
    return min((s.tasks[j].deadline-end for j, (_, end) in plan.edge_times.items()), default=float('inf'))


def sunk_value(s, plan, existing):
    return sum(s.tasks[j].value for j in plan.edge_order if j in existing)


def slack_floor(s, baseline):
    return max(0., min(SLACK_GATE, plan_min_slack(s, baseline))) if baseline.edge_order else 0.


def release_profile(s, plan):
    """Immediate commitments only, including FIFO upload tail and location."""
    from .transmission import route_releases
    locks, tails, locations, heads = [], [], [], []
    for vid, (v, route) in enumerate(zip(s.vehicles, plan.routes)):
        base, location = model.route_base(s, vid)
        head = route[0] if v.active is None and route else None
        captures = {} if head is None else {head:plan.capture_times[head]}
        reports = route_releases(s, vid, captures)
        tail = max(s.now, getattr(s, 'link_tails', (s.now,)*len(s.vehicles))[vid])
        if v.active is not None and s.status[v.active.task_id] != 'expired':
            j = v.active.task_id
            uplink = getattr(s, 'uplink', None)
            tail = (uplink.finish_time(vid, max(tail, v.active.finish), s.tasks[j].upload)
                    if uplink is not None else v.active.finish+s.tasks[j].upload)
        if head is not None:
            base = plan.capture_times[head]
            location = s.tasks[head].x, s.tasks[head].y
            tail = max(tail, reports[head])
        locks.append(max(0., base-s.now))
        tails.append(max(0., tail-s.now))
        locations.append(location)
        heads.append(head)
    edge_head = (plan.edge_order[0] if s.edge_active is None and plan.edge_order
                 and s.status[plan.edge_order[0]] == 'ready' else None)
    edge_lock = (model.edge_cursor(s)-s.now if edge_head is None
                 else s.tasks[edge_head].work*s.multiplier)
    return dict(vehicle=tuple(locks), upload=tuple(tails), locations=tuple(locations),
                heads=tuple(heads), edge=max(0., edge_lock), edge_head=edge_head)


def exposure_cost(s, candidate, baseline, old=None, new=None, slack_price=0.):
    """Visible reference-density proxy, NOT a stochastic opportunity-cost bound.

    Price an extra immediate lock by the baseline head's value per lock time.
    Take the maximum stage charge to avoid adding three charges for one report.
    Idle baseline resources have zero price; position changes remain unpriced.
    Optional precomputed profiles avoid rebuilding release_profile when the
    caller already needs them (decision logging).

    slack_price >= 0 adds a processor-contention premium: surviving reports
    whose projected finish is delayed pay value * delay / remaining-slack, and
    reports dropped from the plan while their capture already happened pay
    slack_price times their value. This prices visible downstream crowding
    only; it is not a stochastic opportunity-cost bound.
    """
    if old is None:
        old = release_profile(s, baseline)
    if new is None:
        new = release_profile(s, candidate)
    charges = []
    for stage in ('vehicle', 'upload'):
        charges.append(sum(s.tasks[j].value/old[stage][i] * max(0., new[stage][i]-old[stage][i])
                           for i, j in enumerate(old['heads'])
                           if j is not None and old[stage][i] > EPS))
    j = old['edge_head']
    charges.append(0. if j is None or old['edge'] <= EPS else
                   s.tasks[j].value/old['edge'] * max(0., new['edge']-old['edge']))
    if slack_price > 0.:
        for k, (_, end_b) in baseline.edge_times.items():
            ends = candidate.edge_times.get(k)
            if ends is not None:
                delay = ends[1]-end_b
                if delay > EPS:
                    slack_b = max(1., s.tasks[k].deadline-end_b)
                    charges.append(s.tasks[k].value*delay/slack_b)
            elif k in s.status and s.status[k] in ('ready', 'uploading'):
                charges.append(slack_price*s.tasks[k].value)
    return max(charges)


def acceptance_reason(s, candidate, baseline, tolerance=1e-4, compact_ties=True,
                      guarded=False, slack_price=0.):
    # guarded=False preserves the declared comparison control's contract.
    if candidate is None:
        return 'no_feasible_candidate'
    same_action = action_key(s, candidate) == action_key(s, baseline)
    if same_action and not (guarded and candidate.value > baseline.value+EPS):
        return 'same_immediate_action'
    if guarded:
        if (candidate.value < baseline.value-EPS
                or candidate.value-baseline.value+EPS < exposure_cost(s, candidate, baseline,
                                                                     slack_price=slack_price)):
            return 'continuation_guard'
    if candidate.value > baseline.value+EPS:
        return 'same_action_tail_gain' if same_action else 'strict_value_gain'
    if compact_ties and abs(candidate.value-baseline.value) <= EPS:
        existing = {j for j, st in s.status.items() if st in ('ready', 'uploading')}
        old, new = plan_resources(s, baseline), plan_resources(s, candidate)
        def locks(plan):
            return [plan.capture_times[r[0]]-s.now if r else 0.
                    for v, r in zip(s.vehicles, plan.routes) if v.active is None]
        if (all(a <= b+EPS for a, b in zip(locks(candidate), locks(baseline)))
                and sunk_value(s, candidate, existing) >= sunk_value(s, baseline, existing)-EPS
                and all(a <= b+EPS for a, b in zip(new, old))
                and any(a < b-tolerance for a, b in zip(new, old))
                and plan_min_slack(s, candidate) >= (slack_floor(s, baseline) if guarded else SLACK_GATE)):
            return 'equal_value_resource_dominance'
    return 'keep_density'
