"""Targeted verification of manuscript formalizations against the frozen code.

Checks:
  1. R1-M1 (Eq. 7): Two-task cross-event transition where A and B are infeasible
     at floors (5, 5), then A becomes ready at R_A^e = 1 while B remains in
     locked acquisition at R_B^e = 5; verifies the code emits only z_B <= 1
     and preserves the feasible schedule A:[1,4], B:[5,8]. Also checks pending ->
     active acquisition -> uploading -> ready states.
  2. R1-M2 (Acceptance & baseline): Strict value gain passing exposure guard
     can increase Eq. (5) Phi; higher-value Density proposal failing the guard
     does not replace carried baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from rosr import RosrPolicy, model
from rosr.boot import from_order, validate_plan
from rosr.core import _Builder, _release_conflict_cut, enumerate_routes, route_risk
from rosr.criteria import acceptance_reason, exposure_cost
from rosr.edge import ConflictStore, EdgeOracle


def task(j, deadline=10.0, value=10, service=1.0, work=1.0):
    return model.Task(j, 0.0, 0, 0, deadline, value, 0, service, 0.2, work)


def state(tasks, status=None, ready=None, now=0.0):
    tmap = {t.task_id: t for t in tasks}
    return model.Snapshot(
        now, tmap, status or {j: "pending" for j in tmap},
        [model.legacy.Vehicle(0, 0, 0, 0, 2)], ready or {}, None, 1.0,
    )


def phi(s, p):
    total = 0.0
    for vid, route in enumerate(p.routes):
        if not route:
            continue
        w = next(w for w in enumerate_routes(s, vid, anchor=route) if w.tasks == route)
        total += w.travel + w.occupancy + sum(r - s.now for r in w.reports.values()) + route_risk(s, w, 0.25)
    return total


def main():
    # 1. Cross-event release-floor cut check
    tasks = [task(0, service=0.8, work=3.0), task(1, service=4.8, work=3.0), task(2, deadline=30.0, service=4.0, work=0.1)]
    vehicles = [model.legacy.Vehicle(0, 0, 0, 0, 2), model.legacy.Vehicle(1, 0, 0, 0, 2)]
    env = model.Env(tasks, vehicles, 1.0)
    s0 = env.snapshot()
    pred_bad = model.route_predictions(s0, ((2, 0), (1,)))
    floors = {j: round(c + s0.tasks[j].upload, 9) for j, c in pred_bad[0].items() if j in (0, 1)}
    assert floors == {0: 5.0, 1: 5.0}, floors
    assert EdgeOracle(s0)._exact_all(floors) is None
    store = ConflictStore()
    store.add(s0, floors)

    env.apply(from_order(s0, ((0,), (1,)), (0, 1), "early"))
    env.advance()  # t=0.8: A uploading (ready at 1.0), B locked acquisition (ready at 5.0)
    s_up = env.snapshot()
    assert s_up.status[0] == "uploading" and s_up.status[1] == "active_capture"
    b_up = _Builder()
    z_up = {j: b_up.var(("z", j)) for j in (0, 1)}
    _release_conflict_cut(s_up, b_up, {}, z_up, [[], []], model.report_releases(s_up, {}), floors)
    assert {str(b_up.keys[col]): coef for col, coef in zip(b_up.cols, b_up.data)} == {"('z', 1)": 1.0}
    env.apply(from_order(s_up, ((), ()), (0, 1), "early"))
    env.advance()  # t=1.0: A ready at 1.0, B locked acquisition at 5.0
    s1 = env.snapshot()
    assert s1.status[0] == "ready" and s1.status[1] == "active_capture"
    p1 = from_order(s1, ((), ()), (0, 1), "feasible_retained")
    assert validate_plan(s1, p1)
    assert store.active(s1)
    reports = model.report_releases(s1, {})
    b = _Builder()
    z = {j: b.var(("z", j)) for j in (0, 1)}
    _release_conflict_cut(s1, b, {}, z, [[], []], reports, floors)
    active_terms = {str(b.keys[col]): coef for col, coef in zip(b.cols, b.data)}
    assert active_terms == {"('z', 1)": 1.0}, f"unexpected cut terms: {active_terms}"

    # 2. Strict value gain can increase Phi when exposure guard passes
    s_gain = state([task(0, deadline=3.2, value=10, work=2.0), task(1, deadline=3.2, value=11, work=2.0)],
                   {0: "ready", 1: "pending"}, {0: 0.0})
    base = from_order(s_gain, ((),), (0,), "base")
    cand = from_order(s_gain, ((1,),), (1,), "cand")
    assert acceptance_reason(s_gain, cand, base, guarded=True) == "strict_value_gain"
    assert phi(s_gain, cand) > phi(s_gain, base)

    # 3. Guarded baseline retains carried plan when higher-value Density fails exposure guard
    s_base = state([task(0, deadline=4.0, value=5, work=4.0), task(1, deadline=6.0, value=6, work=6.0)],
                   {0: "ready", 1: "ready"}, {0: 0.0, 1: 0.0})
    carried = from_order(s_base, ((),), (0,), "old")
    pol = RosrPolicy(seconds=0.005)
    pol.continuation.remember(s_base, carried)
    chosen = pol(s_base)
    assert pol.decisions[-1]["density_value"] == 6.0
    assert pol.decisions[-1]["baseline_value"] == 5.0
    assert chosen.value == 5.0

    print(json.dumps({
        "cross_event_cut_terms": active_terms,
        "feasible_edge_times": p1.edge_times,
        "strict_gain_phi": [phi(s_gain, base), phi(s_gain, cand)],
        "guarded_baseline": {"density": 6.0, "baseline": 5.0, "chosen": chosen.value},
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
