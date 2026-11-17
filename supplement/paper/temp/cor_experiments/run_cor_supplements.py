"""Full multi-process runner for COR supplementary experiments:
1. Monolithic Disjunctive MILP (0.1 s) vs. RCR (LBBD, 0.1 s) on 30 paired reference streams
   (seeds 20423000-20423029) across coupled_primary, edge_dominant, tighter_deadlines.
2. Edge-bottleneck transition m in {1.0, 1.5, 2.0, 2.5, 3.0} on the 30 paired reference streams
   comparing RCR, Density, and Joint DR.
Every episode is verified by runtime.audit.check_episode and runtime.audit.audit_action_logs.
"""
from __future__ import annotations
import os
import sys
import time
import json
import math
import hashlib
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

CODE_ROOT = Path(__file__).resolve().parents[3] / 'code'
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
from scipy.stats import t as student_t, wilcoxon

from rosr import RosrPolicy, model
from rosr.comparison import DensityPolicy, JointLnsPolicy
from rosr.boot import EPS, audit, validate_plan, action_key
from rosr.core import enumerate_routes, route_risk
from rosr.evaluation import EvaluationContext, density_initialize, local_equal_value_probe
from rosr.criteria import acceptance_reason, ACCEPTED
from rosr.continuation import Continuation
from rosr.capacity import deadline_value_bound, release_lower_bounds, release_value_bound


def solve_monolithic_milp(s, routes_by_vehicle, report_release, beta=0.25, seconds=0.1, baseline=None):
    """Monolithic disjunctive MILP over the exact same path pool and snapshot as RCR."""
    started = time.perf_counter()
    now = float(s.now)
    base = float(model.edge_cursor(s))
    effective_reports = {j: max(base, float(r)) for j, r in report_release.items()
                         if max(base, float(r)) + s.tasks[j].work * s.multiplier <= s.tasks[j].deadline + EPS}
    pending_jobs = sorted({j for pool in routes_by_vehicle for w in pool for j in w.tasks})
    jobs = sorted(set(pending_jobs) | set(effective_reports))
    if not jobs:
        return baseline, dict(solves=0, status='empty', found=False)

    phi_w = {}
    for vid, pool in enumerate(routes_by_vehicle):
        b_time, _ = model.route_base(s, vid)
        for w in pool:
            if not w.tasks:
                continue
            occ = max(0.0, float(w.captures[w.tasks[-1]]) - float(b_time))
            rel_sum = sum(max(0.0, float(w.reports[j]) - now) for j in w.tasks)
            risk = route_risk(s, w, beta)
            phi_w[w] = float(w.travel) + occ + rel_sum + float(risk)

    max_phi = sum(max((phi_w[w] for w in pool if w.tasks), default=0.0) for pool in routes_by_vehicle)
    sec_scale = 1.0 / (4.0 * max(1.0, max_phi + sum(s.tasks[j].value for j in effective_reports) + 100.0))
    ret_scale = sec_scale * 2.0 * (max_phi + 1.0) / max(1.0, sum(s.tasks[j].value for j in effective_reports))

    y_map, z_map, s_map, u_map = {}, {}, {}, {}
    var_lb, var_ub, var_int, var_obj = [], [], [], []

    for vid, pool in enumerate(routes_by_vehicle):
        for w in pool:
            if not w.tasks:
                continue
            y_map[w] = len(var_lb)
            var_lb.append(0.0)
            var_ub.append(1.0)
            var_int.append(1)
            var_obj.append(sec_scale * phi_w[w])

    for j in jobs:
        z_map[j] = len(var_lb)
        var_lb.append(0.0)
        var_ub.append(1.0)
        var_int.append(1)
        bonus = ret_scale * float(s.tasks[j].value) if j in effective_reports else 0.0
        var_obj.append(-float(s.tasks[j].value) - bonus)

    for j in jobs:
        dj = float(s.tasks[j].deadline)
        qj = float(s.tasks[j].work * s.multiplier)
        s_map[j] = len(var_lb)
        var_lb.append(base)
        var_ub.append(max(base, dj - qj + EPS))
        var_int.append(0)
        var_obj.append(1e-8)

    for idx_a in range(len(jobs)):
        for idx_b in range(idx_a + 1, len(jobs)):
            ja, jb = jobs[idx_a], jobs[idx_b]
            u_map[(ja, jb)] = len(var_lb)
            var_lb.append(0.0)
            var_ub.append(1.0)
            var_int.append(1)
            var_obj.append(0.0)

    rows_i, rows_j, rows_v = [], [], []
    row_lo, row_hi = [], []

    def add_row(terms: dict[int, float], lo=-np.inf, hi=np.inf):
        r = len(row_lo)
        for col, val in terms.items():
            if val != 0.0:
                rows_i.append(r)
                rows_j.append(col)
                rows_v.append(float(val))
        row_lo.append(float(lo))
        row_hi.append(float(hi))

    for vid, pool in enumerate(routes_by_vehicle):
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
        terms = {z_map[j]: float(s.tasks[j].work * s.multiplier)
                 for j in jobs if float(s.tasks[j].deadline) <= d + EPS}
        add_row(terms, hi=max(0.0, d - base) + EPS * (len(jobs) + 2))

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
        add_row({s_map[jb]: 1.0, s_map[ja]: -1.0, u_col: -m_ab, z_map[ja]: -m_ab},
                lo=qa - 2.0 * m_ab)
        add_row({s_map[ja]: 1.0, s_map[jb]: -1.0, u_col: m_ba, z_map[jb]: -m_ba},
                lo=qb - m_ba)

    rem = max(0.001, seconds - (time.perf_counter() - started))
    mat = csc_matrix((rows_v, (rows_i, rows_j)), shape=(len(row_lo), len(var_lb)))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
        res = milp(
            c=np.asarray(var_obj, dtype=float),
            integrality=np.asarray(var_int, dtype=int),
            bounds=Bounds(np.asarray(var_lb, dtype=float), np.asarray(var_ub, dtype=float)),
            constraints=LinearConstraint(mat, np.asarray(row_lo, dtype=float), np.asarray(row_hi, dtype=float)),
            options={'time_limit': rem, 'threads': 1}
        )
    if res.x is None:
        return baseline, dict(solves=1, status=int(res.status), found=False, n_vars=len(var_lb), n_rows=len(row_lo))

    chosen_routes = [() for _ in range(len(s.vehicles))]
    for w, col in y_map.items():
        if res.x[col] > 0.5:
            chosen_routes[w.vehicle] = w.tasks
    routes_tuple = tuple(chosen_routes)
    pred = model.route_predictions(s, routes_tuple)
    if pred is None:
        return baseline, dict(solves=1, status=int(res.status), found=False, n_vars=len(var_lb), n_rows=len(row_lo))
    captures, travel = pred
    releases = model.report_releases(s, captures)
    selected_jobs = [j for j in jobs if res.x[z_map[j]] > 0.5 and j in releases]
    ordered_jobs = sorted(selected_jobs, key=lambda j: (res.x[s_map[j]], s.tasks[j].deadline, j))
    cursor = model.edge_cursor(s)
    edge_times, edge_order = {}, []
    for j in ordered_jobs:
        start = max(cursor, releases[j])
        end = start + s.tasks[j].work * s.multiplier
        if end > s.tasks[j].deadline + EPS:
            return baseline, dict(solves=1, status=int(res.status), found=False, n_vars=len(var_lb), n_rows=len(row_lo))
        edge_times[j] = (start, end)
        edge_order.append(j)
        cursor = end
    if not set(captures).issubset(edge_order):
        return baseline, dict(solves=1, status=int(res.status), found=False, n_vars=len(var_lb), n_rows=len(row_lo))
    val = sum(s.tasks[j].value for j in edge_order)
    wait = sum(edge_times[j][0] - releases[j] for j in edge_order)
    plan = model.Plan(
        routes=routes_tuple,
        edge_order=tuple(edge_order),
        edge_times=edge_times,
        capture_times=captures,
        value=val,
        count=len(edge_order),
        wait=wait,
        travel=travel,
        rule='monolithic_milp'
    )
    validate_plan(s, plan)
    return plan, dict(solves=1, status=int(res.status), found=True, n_vars=len(var_lb), n_rows=len(row_lo))


class MonolithicMilpPolicy:
    def __init__(self, seconds=0.1, beta=0.25, max_depth=3, pool_routes=64, pool_factor=4, compact_ties=True):
        self.seconds = seconds
        self.beta = beta
        self.max_depth = max_depth
        self.pool_routes = pool_routes
        self.pool_factor = pool_factor
        self.compact_ties = compact_ties
        self.continuation = Continuation()
        self.last_now = -float('inf')
        self.last_event = -1
        self.stats = dict(events=0, opportunities=0, planning_seconds=0.0, optimal_solves=0, timeouts=0,
                          analytic_skips=0)

    def __call__(self, s):
        started = time.perf_counter()
        if s.now < self.last_now or s.event_index < self.last_event:
            self.continuation = Continuation()
        self.last_now = s.now
        self.last_event = s.event_index
        ctx = EvaluationContext(s, conflicts=None)
        carried, _ = self.continuation.advance(s)
        bound_started = time.perf_counter()
        releases = release_lower_bounds(s, ctx)
        bound = deadline_value_bound(s, releases)
        bound_seconds = time.perf_counter() - bound_started

        capture_possible = any(v.active is None and any(
            status == 'pending' and s.tasks[j].sensor == v.sensor
            for j, status in s.status.items()) for v in s.vehicles)
        edge_possible = s.edge_active is None and 'ready' in s.status.values()
        ceiling = carried is not None and bound <= carried.value + EPS
        if ceiling and (not self.compact_ties or not (capture_possible or edge_possible)):
            density = carried
        elif ceiling:
            density = local_equal_value_probe(s, ctx, carried)
            density = carried if density is None else density
        else:
            density = density_initialize(s, ctx, value_upper=bound)

        baseline = carried if carried is not None else density
        if carried is not None and density is not carried and acceptance_reason(
                s, density, carried, compact_ties=self.compact_ties, guarded=True) in ACCEPTED:
            baseline = density

        initialized = time.perf_counter()
        deadline = initialized + self.seconds - bound_seconds
        self.stats['events'] += 1

        if (capture_possible or edge_possible) and bound > baseline.value + EPS:
            bound = release_value_bound(s, releases, bound,
                                        seconds=min(0.005, self.seconds / 10, max(0.0, deadline - time.perf_counter())))

        tie_search = (self.compact_ties and bound <= baseline.value + EPS
                      and density.value >= baseline.value - EPS
                      and action_key(s, density) != action_key(s, baseline))

        proposed = None
        if (capture_possible or edge_possible) and (bound > baseline.value + EPS or tie_search):
            self.stats['opportunities'] += 1
            pool = [enumerate_routes(s, vid, max_depth=self.max_depth, anchor=baseline.routes[vid],
                                     focus=set(), stop=lambda: time.perf_counter() >= deadline,
                                     max_routes=self.pool_routes, pool_factor=self.pool_factor)
                    for vid in range(len(s.vehicles))]
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                proposed, meta = solve_monolithic_milp(
                    s, pool, ctx.base_reports, beta=self.beta, seconds=remaining, baseline=baseline)
                if meta.get('status') == 0:
                    self.stats['optimal_solves'] += 1
                else:
                    self.stats['timeouts'] += 1
        else:
            self.stats['analytic_skips'] += 1

        reason = acceptance_reason(s, proposed, baseline, compact_ties=self.compact_ties, guarded=True)
        chosen = proposed if reason in ACCEPTED else baseline
        validate_plan(s, chosen)
        self.continuation.remember(s, chosen)
        self.stats['planning_seconds'] += time.perf_counter() - started
        return chosen


def run_job(job):
    kind, seed, profile_name, mult, slack, method = job
    prof = model.Profile(profile_name, arrival_rate=1.0, edge_multiplier=mult, slack=slack)
    tasks, vehicles = model.generate(seed, prof)
    if method == 'rcr':
        pol = RosrPolicy(seed=seed, seconds=0.1)
    elif method == 'density':
        pol = DensityPolicy()
    elif method == 'joint_dr':
        pol = JointLnsPolicy(seconds=0.1, compact_ties=False)
    elif method == 'monolithic_01':
        pol = MonolithicMilpPolicy(seconds=0.1)
    else:
        raise ValueError(method)
    summary, rows, logs = model.run_episode(seed, prof, method, policy=pol)
    audit.check_episode(summary, rows, logs)
    audit.audit_action_logs(tasks, vehicles, prof.edge_multiplier, rows, logs)
    stats = {}
    if hasattr(pol, 'stats'):
        stats = dict(pol.stats) if isinstance(pol.stats, dict) else dict(vars(pol.stats))
    return dict(
        kind=kind,
        seed=seed,
        profile=profile_name,
        multiplier=mult,
        method=method,
        value_rate=float(summary['value_rate']) * 100.0,
        count_rate=float(summary['count_rate']) * 100.0,
        planning_s=float(summary['planning_ms']) / 1000.0,
        events=int(summary['events']),
        stats=stats
    )


def paired_stats(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    rad = float(student_t.ppf(0.975, n - 1) * se) if se > 0 else 0.0
    wins = int(np.sum(d > 1e-9))
    ties = int(np.sum(np.abs(d) <= 1e-9))
    losses = int(np.sum(d < -1e-9))
    nz = d[np.abs(d) > 1e-9]
    p_val = float(wilcoxon(nz, zero_method='wilcox', alternative='two-sided', mode='exact').pvalue) if len(nz) > 0 else 1.0
    return dict(mean=mean, ci_lo=mean - rad, ci_hi=mean + rad, wtl=f"{wins}/{ties}/{losses}", p_wilcoxon=p_val)


def main():
    outdir = Path(__file__).resolve().parent / 'results'
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(20423000, 20423030))
    jobs = []

    # 1. Edge multiplier transition m in {1.0, 1.5, 2.0, 2.5, 3.0}
    for m in (1.0, 1.5, 2.0, 2.5, 3.0):
        pname = f"m_{m:.1f}"
        for seed in seeds:
            for method in ('rcr', 'density', 'joint_dr'):
                jobs.append(('m_sweep', seed, pname, m, (16.0, 24.0), method))

    # 2. Monolithic MILP (0.1s) on the 3 core workloads (m=1.5 primary, m=2.5 edge_limited, tight deadlines)
    for pname, m, slack in [
        ('coupled_primary', 1.5, (16.0, 24.0)),
        ('edge_dominant', 2.5, (16.0, 24.0)),
        ('tighter_deadlines', 1.5, (8.0, 12.0)),
    ]:
        for seed in seeds:
            jobs.append(('mono', seed, pname, m, slack, 'monolithic_01'))
            if pname == 'tighter_deadlines':
                jobs.append(('mono', seed, pname, m, slack, 'rcr'))

    print(f"Running {len(jobs)} audited episodes across 8 workers...")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        done_cnt = 0
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done_cnt += 1
            if done_cnt % 50 == 0 or done_cnt == len(jobs):
                print(f"  [{done_cnt}/{len(jobs)}] elapsed={time.perf_counter()-t0:.1f}s")

    (outdir / 'raw_episodes.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    # Aggregate 1: Edge multiplier sweep
    by_key = {(r['profile'], r['method'], r['seed']): r for r in results}
    # Link coupled_primary <-> m_1.5 and edge_dominant <-> m_2.5 for monolithic comparison
    for seed in seeds:
        by_key[('coupled_primary', 'rcr', seed)] = by_key[('m_1.5', 'rcr', seed)]
        by_key[('edge_dominant', 'rcr', seed)] = by_key[('m_2.5', 'rcr', seed)]

    m_summary = []
    print("\n=== EDGE MULTIPLIER SWEEP (30 paired streams per m) ===")
    for m in (1.0, 1.5, 2.0, 2.5, 3.0):
        pname = f"m_{m:.1f}"
        rcr_v = [by_key[(pname, 'rcr', s)]['value_rate'] for s in seeds]
        den_v = [by_key[(pname, 'density', s)]['value_rate'] for s in seeds]
        jdr_v = [by_key[(pname, 'joint_dr', s)]['value_rate'] for s in seeds]
        rcr_t = [by_key[(pname, 'rcr', s)]['planning_s'] for s in seeds]
        jdr_t = [by_key[(pname, 'joint_dr', s)]['planning_s'] for s in seeds]
        cuts = sum(by_key[(pname, 'rcr', s)]['stats'].get('horn_cuts', 0) for s in seeds)
        fallback_cuts = sum(by_key[(pname, 'rcr', s)]['stats'].get('no_good_cuts', 0) for s in seeds)
        reused = sum(by_key[(pname, 'rcr', s)]['stats'].get('reused_conflict_cuts', 0) for s in seeds)
        ev_tot = sum(by_key[(pname, 'rcr', s)]['stats'].get('events', 0) for s in seeds)
        skips = sum(by_key[(pname, 'rcr', s)]['stats'].get('analytic_skips', 0) for s in seeds)
        vs_den = paired_stats(rcr_v, den_v)
        vs_jdr = paired_stats(rcr_v, jdr_v)
        row = dict(
            m=m,
            rcr_mean=float(np.mean(rcr_v)),
            density_mean=float(np.mean(den_v)),
            joint_dr_mean=float(np.mean(jdr_v)),
            rcr_minus_density=vs_den,
            rcr_minus_joint_dr=vs_jdr,
            rcr_plan_s=float(np.mean(rcr_t)),
            joint_dr_plan_s=float(np.mean(jdr_t)),
            cuts=cuts,
            reused=reused,
            skip_pct=100.0 * skips / max(1, ev_tot)
        )
        m_summary.append(row)
        print(f"m={m:.1f}: RCR={row['rcr_mean']:.2f}% Den={row['density_mean']:.2f}% JDR={row['joint_dr_mean']:.2f}% | "
              f"RCR-Den={vs_den['mean']:+.2f} [{vs_den['ci_lo']:+.2f},{vs_den['ci_hi']:+.2f}] p={vs_den['p_wilcoxon']:.4g} ({vs_den['wtl']}) | "
              f"RCR-JDR={vs_jdr['mean']:+.2f} [{vs_jdr['ci_lo']:+.2f},{vs_jdr['ci_hi']:+.2f}] p={vs_jdr['p_wilcoxon']:.4g} | "
              f"cuts={cuts}({reused}) skip={row['skip_pct']:.1f}% time={row['rcr_plan_s']:.2f}s")

    # Aggregate 2: Monolithic MILP vs. RCR
    mono_summary = []
    print("\n=== MONOLITHIC MILP (0.1s) vs. RCR LBBD (0.1s) (30 paired streams) ===")
    for pname in ('coupled_primary', 'edge_dominant', 'tighter_deadlines'):
        rcr_v = [by_key[(pname, 'rcr', s)]['value_rate'] for s in seeds]
        mon_v = [by_key[(pname, 'monolithic_01', s)]['value_rate'] for s in seeds]
        rcr_t = [by_key[(pname, 'rcr', s)]['planning_s'] for s in seeds]
        mon_t = [by_key[(pname, 'monolithic_01', s)]['planning_s'] for s in seeds]
        opp = sum(by_key[(pname, 'monolithic_01', s)]['stats'].get('opportunities', 0) for s in seeds)
        timeouts = sum(by_key[(pname, 'monolithic_01', s)]['stats'].get('timeouts', 0) for s in seeds)
        vs_mon = paired_stats(rcr_v, mon_v)
        t_diff = paired_stats(mon_t, rcr_t)
        row = dict(
            workload=pname,
            rcr_mean=float(np.mean(rcr_v)),
            mono_mean=float(np.mean(mon_v)),
            rcr_minus_mono=vs_mon,
            rcr_plan_s=float(np.mean(rcr_t)),
            mono_plan_s=float(np.mean(mon_t)),
            mono_minus_rcr_time=t_diff,
            speedup_ratio=float(np.mean(mon_t) / max(1e-9, np.mean(rcr_t))),
            mono_opportunities=opp,
            mono_timeouts=timeouts,
            mono_timeout_pct=100.0 * timeouts / max(1, opp)
        )
        mono_summary.append(row)
        print(f"{pname}: RCR={row['rcr_mean']:.2f}% Mono={row['mono_mean']:.2f}% | "
              f"RCR-Mono={vs_mon['mean']:+.2f} [{vs_mon['ci_lo']:+.2f},{vs_mon['ci_hi']:+.2f}] p={vs_mon['p_wilcoxon']:.4g} ({vs_mon['wtl']}) | "
              f"Time: RCR={row['rcr_plan_s']:.2f}s vs Mono={row['mono_plan_s']:.2f}s ({row['speedup_ratio']:.2f}x) | "
              f"Mono timeouts={timeouts}/{opp} ({row['mono_timeout_pct']:.1f}%)")

    (outdir / 'summary.json').write_text(
        json.dumps(dict(m_sweep=m_summary, monolithic_vs_rcr=mono_summary), indent=2),
        encoding='utf-8'
    )


if __name__ == '__main__':
    main()
