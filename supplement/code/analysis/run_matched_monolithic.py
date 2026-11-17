"""Matched-snapshot, single-thread comparison of RCR LBBD vs. monolithic MILP variants.

On every search snapshot visited by RCR over the 30 paired reference streams
(20423000-20423029), evaluate three formulations on the exact same snapshot,
candidate route pool (including conflict focus), baseline value cutoff, and
0.1 s per-event remaining search budget under strictly serial execution:
  1. RCR release-floor LBBD (`solve_set_packing`)
  2. Monolithic disjunctive Big-M MILP with deadline-prefix rows only
  3. Monolithic disjunctive Big-M MILP strengthened with Eq. (6) critical-interval rows

Run from project root:
    python -B code/analysis/run_matched_monolithic.py
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from rosr import RosrPolicy, model
from rosr.boot import EPS, audit
from rosr.capacity import critical_intervals, mandatory_overlap
from rosr.core import route_risk
import rosr.policy as policy_module


def run_monolithic_variant(s, routes_by_vehicle, report_release, beta=0.25,
                           seconds=0.1, baseline=None, use_critical_intervals=False):
    started = time.perf_counter()
    now = float(s.now)
    base = float(model.edge_cursor(s))
    edge_base = base - now
    effective_reports = {
        j: max(base, float(r))
        for j, r in report_release.items()
        if max(base, float(r)) + s.tasks[j].work * s.multiplier <= s.tasks[j].deadline + EPS
    }
    pending_jobs = sorted({j for pool in routes_by_vehicle for w in pool for j in w.tasks})
    jobs = sorted(set(pending_jobs) | set(effective_reports))
    if not jobs:
        return dict(status=0, timed_out=0, build_ms=0.0, solve_ms=0.0, total_ms=0.0,
                    n_vars=0, n_rows=0, dual_bound=None, primal_val=None)

    phi_w = {}
    for vid, pool in enumerate(routes_by_vehicle):
        b_time, _ = model.route_base(s, vid)
        for w in pool:
            if not w.tasks:
                continue
            occ = max(0.0, float(w.captures[w.tasks[-1]]) - float(b_time))
            rel_sum = sum(max(0.0, float(w.reports[j]) - now) for j in w.tasks)
            phi_w[w] = float(w.travel) + occ + rel_sum + float(route_risk(s, w, beta))

    max_phi = sum(max((phi_w[w] for w in pool if w.tasks), default=0.0) for pool in routes_by_vehicle)
    sec_scale = 1.0 / (4.0 * max(1.0, max_phi + sum(s.tasks[j].value for j in effective_reports) + 100.0))
    ret_scale = sec_scale * 2.0 * (max_phi + 1.0) / max(1.0, sum(s.tasks[j].value for j in effective_reports))

    y_map, z_map, s_map, u_map = {}, {}, {}, {}
    var_lb, var_ub, var_int, var_obj = [], [], [], []

    for pool in routes_by_vehicle:
        for w in pool:
            if not w.tasks:
                continue
            y_map[w] = len(var_lb)
            var_lb.append(0.0); var_ub.append(1.0); var_int.append(1)
            var_obj.append(sec_scale * phi_w[w])

    for j in jobs:
        z_map[j] = len(var_lb)
        var_lb.append(0.0); var_ub.append(1.0); var_int.append(1)
        bonus = ret_scale * float(s.tasks[j].value) if j in effective_reports else 0.0
        var_obj.append(-float(s.tasks[j].value) - bonus)

    for j in jobs:
        dj = float(s.tasks[j].deadline)
        qj = float(s.tasks[j].work * s.multiplier)
        s_map[j] = len(var_lb)
        var_lb.append(base); var_ub.append(max(base, dj - qj + EPS)); var_int.append(0)
        var_obj.append(1e-8)

    for idx_a in range(len(jobs)):
        for idx_b in range(idx_a + 1, len(jobs)):
            u_map[(jobs[idx_a], jobs[idx_b])] = len(var_lb)
            var_lb.append(0.0); var_ub.append(1.0); var_int.append(1); var_obj.append(0.0)

    rows_i, rows_j, rows_v, row_lo, row_hi = [], [], [], [], []

    def add_row(terms, lo=-np.inf, hi=np.inf):
        r = len(row_lo)
        for col, val in terms.items():
            if val != 0.0:
                rows_i.append(r); rows_j.append(col); rows_v.append(float(val))
        row_lo.append(float(lo)); row_hi.append(float(hi))

    for pool in routes_by_vehicle:
        cols = {y_map[w]: 1.0 for w in pool if w.tasks}
        if cols:
            add_row(cols, hi=1.0)

    for j in pending_jobs:
        terms = {z_map[j]: -1.0}
        for pool in routes_by_vehicle:
            for w in pool:
                if w.tasks and j in w.tasks:
                    terms[y_map[w]] = 1.0
        add_row(terms, lo=0.0, hi=0.0)

    if baseline is not None:
        add_row({z_map[j]: float(s.tasks[j].value) for j in jobs}, lo=float(baseline.value) - EPS)

    for d in sorted({float(s.tasks[j].deadline) for j in jobs}):
        terms = {
            z_map[j]: float(s.tasks[j].work * s.multiplier)
            for j in jobs if float(s.tasks[j].deadline) <= d + EPS
        }
        add_row(terms, hi=max(0.0, d - base) + EPS * (len(jobs) + 2))

    if use_critical_intervals:
        job_bounds = []
        for j in jobs:
            cands = []
            if j in effective_reports:
                cands.append(effective_reports[j] - now)
            for pool in routes_by_vehicle:
                for w in pool:
                    if w.tasks and j in w.tasks:
                        cands.append(max(base, float(w.reports[j])) - now)
            if cands:
                job_bounds.append((min(cands), float(s.tasks[j].deadline) - now, float(s.tasks[j].work * s.multiplier)))
        for a, bnd in critical_intervals(job_bounds, edge_base, full=False):
            terms = {}
            for pool in routes_by_vehicle:
                for w in pool:
                    if not w.tasks:
                        continue
                    load = sum(
                        mandatory_overlap(max(base, float(w.reports[j])) - now,
                                          float(s.tasks[j].deadline) - now,
                                          float(s.tasks[j].work * s.multiplier), a, bnd)
                        for j in w.tasks
                    )
                    if load > 0.0:
                        terms[y_map[w]] = terms.get(y_map[w], 0.0) + load
            for j, r in effective_reports.items():
                load = mandatory_overlap(r - now, float(s.tasks[j].deadline) - now,
                                         float(s.tasks[j].work * s.multiplier), a, bnd)
                if load > 0.0:
                    terms[z_map[j]] = terms.get(z_map[j], 0.0) + load
            if sum(terms.values()) > (bnd - a) + EPS:
                add_row(terms, hi=(bnd - a) + EPS * (len(jobs) + 2))

    for j in pending_jobs:
        terms = {s_map[j]: 1.0}
        for pool in routes_by_vehicle:
            for w in pool:
                if w.tasks and j in w.tasks:
                    r_jw = max(base, float(w.reports[j]))
                    if r_jw > base:
                        terms[y_map[w]] = -(r_jw - base)
        add_row(terms, lo=base)

    for j, r_j in effective_reports.items():
        if r_j > base:
            add_row({s_map[j]: 1.0, z_map[j]: -(r_j - base)}, lo=base)

    for (ja, jb), u_col in u_map.items():
        qa = float(s.tasks[ja].work * s.multiplier)
        qb = float(s.tasks[jb].work * s.multiplier)
        da = float(s.tasks[ja].deadline)
        db = float(s.tasks[jb].deadline)
        m_ab = max(1.0, da - base + EPS)
        m_ba = max(1.0, db - base + EPS)
        add_row({s_map[jb]: 1.0, s_map[ja]: -1.0, u_col: -m_ab, z_map[ja]: -m_ab}, lo=qa - 2.0 * m_ab)
        add_row({s_map[ja]: 1.0, s_map[jb]: -1.0, u_col: m_ba, z_map[jb]: -m_ba}, lo=qb - m_ba)

    mat = csc_matrix((rows_v, (rows_i, rows_j)), shape=(len(row_lo), len(var_lb)))
    built = time.perf_counter()
    build_ms = 1000.0 * (built - started)
    rem = max(0.001, seconds - (built - started))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Unrecognized options detected: \{'threads'\}.*", category=RuntimeWarning)
        res = milp(
            c=np.asarray(var_obj, dtype=float),
            integrality=np.asarray(var_int, dtype=int),
            bounds=Bounds(np.asarray(var_lb, dtype=float), np.asarray(var_ub, dtype=float)),
            constraints=LinearConstraint(mat, np.asarray(row_lo, dtype=float), np.asarray(row_hi, dtype=float)),
            options={"time_limit": rem, "threads": 1},
        )
    ended = time.perf_counter()
    solve_ms = 1000.0 * (ended - built)
    total_ms = 1000.0 * (ended - started)
    dual = getattr(res, "mip_dual_bound", None)
    primal = None if res.x is None else float(sum(s.tasks[j].value for j in jobs if res.x[z_map[j]] > 0.5))
    return dict(
        status=int(res.status),
        timed_out=int(res.status != 0),
        build_ms=build_ms,
        solve_ms=solve_ms,
        total_ms=total_ms,
        n_vars=len(var_lb),
        n_rows=len(row_lo),
        dual_bound=None if dual is None else float(-dual),
        primal_val=primal,
        mip_nodes=getattr(res, "mip_node_count", None),
    )


def main():
    seeds = list(range(20423000, 20423030))
    workloads = [
        ("coupled_primary", 1.5, (16.0, 24.0)),
        ("edge_dominant", 2.5, (16.0, 24.0)),
        ("tighter_deadlines", 1.5, (8.0, 12.0)),
    ]
    orig_ssp = policy_module.solve_set_packing
    summary = {}
    for pname, mult, slack in workloads:
        prof = model.Profile(pname, arrival_rate=1.0, edge_multiplier=mult, slack=slack)
        snaps = []
        ep_records = []
        for seed in seeds:
            tasks, vehicles = model.generate(seed, prof)
            pol = RosrPolicy(seed=seed, seconds=0.1)

            def hooked_ssp(s, pool, base_reports, beta=0.25, seconds=0.1, baseline=None, **kw):
                t0 = time.perf_counter()
                chosen, sel, meta = orig_ssp(s, pool, base_reports, beta=beta, seconds=seconds, baseline=baseline, **kw)
                lbbd_ms = 1000.0 * (time.perf_counter() - t0)
                raw = run_monolithic_variant(s, pool, base_reports, beta=beta, seconds=seconds, baseline=baseline, use_critical_intervals=False)
                eq6 = run_monolithic_variant(s, pool, base_reports, beta=beta, seconds=seconds, baseline=baseline, use_critical_intervals=True)
                lbbd_to = int(meta.get("stop_reason") in ("budget_exhausted", "time_limited_value_solution"))
                snaps.append(dict(
                    seed=seed, event=s.event_index,
                    lbbd_ms=lbbd_ms, lbbd_solves=meta.get("solves", 0), lbbd_cuts=meta.get("horn_cuts", 0),
                    lbbd_nogoods=meta.get("no_good_cuts", 0), lbbd_to=lbbd_to,
                    lbbd_vars=meta.get("variables", 0), lbbd_rows=meta.get("constraints", 0),
                    raw=raw, eq6=eq6,
                ))
                return chosen, sel, meta

            policy_module.solve_set_packing = hooked_ssp
            try:
                ep_sum, rows, logs = model.run_episode(seed, prof, "rcr", policy=pol)
            finally:
                policy_module.solve_set_packing = orig_ssp
            audit.check_episode(ep_sum, rows, logs)
            audit.audit_action_logs(tasks, vehicles, prof.edge_multiplier, rows, logs)
            ep_records.append(dict(seed=seed, value_rate=100.0 * float(ep_sum["value_rate"]), planning_ms=float(ep_sum["planning_ms"])))

        summary[pname] = dict(
            streams=len(seeds),
            snapshots=len(snaps),
            rcr_mean_value_pct=float(np.mean([r["value_rate"] for r in ep_records])),
            rcr_mean_plan_s=float(np.mean([r["planning_ms"] for r in ep_records])) / 1000.0,
            lbbd_mean_ms=float(np.mean([x["lbbd_ms"] for x in snaps])),
            lbbd_timeout_rate=float(np.mean([x["lbbd_to"] for x in snaps])),
            lbbd_mean_vars=float(np.mean([x["lbbd_vars"] for x in snaps])),
            lbbd_mean_rows=float(np.mean([x["lbbd_rows"] for x in snaps])),
            mono_prefix_only=dict(
                timeout_rate=float(np.mean([x["raw"]["timed_out"] for x in snaps])),
                mean_total_ms=float(np.mean([x["raw"]["total_ms"] for x in snaps])),
                mean_build_ms=float(np.mean([x["raw"]["build_ms"] for x in snaps])),
                mean_solve_ms=float(np.mean([x["raw"]["solve_ms"] for x in snaps])),
                mean_vars=float(np.mean([x["raw"]["n_vars"] for x in snaps])),
                mean_rows=float(np.mean([x["raw"]["n_rows"] for x in snaps])),
            ),
            mono_eq6=dict(
                timeout_rate=float(np.mean([x["eq6"]["timed_out"] for x in snaps])),
                mean_total_ms=float(np.mean([x["eq6"]["total_ms"] for x in snaps])),
                mean_build_ms=float(np.mean([x["eq6"]["build_ms"] for x in snaps])),
                mean_solve_ms=float(np.mean([x["eq6"]["solve_ms"] for x in snaps])),
                mean_vars=float(np.mean([x["eq6"]["n_vars"] for x in snaps])),
                mean_rows=float(np.mean([x["eq6"]["n_rows"] for x in snaps])),
            ),
        )
        print(pname, json.dumps(summary[pname]), flush=True)

    (OUT / "matched_monolithic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
