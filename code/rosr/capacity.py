"""Conservative nonpreemptive energy tests and dual-certified value bounds."""
from __future__ import annotations

import math
import warnings
import numpy as np
from scipy.optimize import linprog
from .boot import EPS, model


def mandatory_overlap(release, deadline, work, a, b):
    """Minimum overlap of one uninterrupted job with [a,b].

    The deadline is relaxed by EPS to include every numerically valid witness.
    Individual infeasibility is handled separately by callers.
    """
    return max(0., min(work, b-a, release+work-a, b-(deadline+EPS)+work))


def energy_tolerance(jobs, a, b):
    scale = max(1., abs(a), abs(b), sum(abs(p) for _, _, p in jobs))
    return EPS*(len(jobs)+1) + 16*(len(jobs)+1)*math.ulp(scale)


def critical_intervals(jobs, base, full=False):
    """A bounded subset is sufficient for sound cuts, not for completeness."""
    windows = {(max(base, r), d+EPS) for r, d, p in jobs if d+EPS > max(base, r)}
    windows |= {(max(base, d+EPS-p), r+p) for r, d, p in jobs
                if r+p > max(base, d+EPS-p)}
    if full:
        starts = sorted({base} | {r for r, _, _ in jobs} | {d+EPS-p for _, d, p in jobs})
        ends = sorted({d+EPS for _, d, _ in jobs} | {r+p for r, _, p in jobs})
        # Avoid cubic work on a long queue. Omitted intervals only weaken tests.
        step = max(1, math.ceil(len(starts)/32))
        starts = starts[::step]
        step = max(1, math.ceil(len(ends)/32))
        ends = ends[::step]
        windows |= {(a, b) for a in starts for b in ends if b > a and a >= base}
    # Endpoints separated only by numerical deadline padding otherwise produce
    # several almost identical cuts. Keep one actual (unrounded) interval.
    unique = {}
    for a, b in sorted(windows):
        unique.setdefault((round(a, 7), round(b, 7)), (a, b))
    return sorted(unique.values(), key=lambda ab: (ab[1]-ab[0], ab))


def energy_violations(jobs, base, limit=6, full=False):
    """Return only rigorously overloaded intervals; passing is not a proof."""
    if not jobs:
        return []
    intervals = critical_intervals(jobs, base, full)
    if not intervals:
        return []
    # Vectorization keeps a complete small-set separation cheaper than Python
    # interval-by-job loops, with bounded O(n * 32**2) auxiliary storage.
    r, d, p = np.asarray(jobs, dtype=float).T
    ab = np.asarray(intervals, dtype=float)
    a, b = ab[:, 0, None], ab[:, 1, None]
    loads = np.maximum(0., np.minimum(np.minimum(p, b-a),
                       np.minimum(r+p-a, b-(d+EPS)+p))).sum(axis=1)
    scales = np.maximum(max(1., float(np.sum(np.abs(p)))), np.maximum(np.abs(ab[:, 0]), np.abs(ab[:, 1])))
    tolerances = EPS*(len(jobs)+1) + 16*(len(jobs)+1)*np.spacing(scales)
    capacities = ab[:, 1]-ab[:, 0]
    broken = np.flatnonzero(loads > capacities+tolerances)
    violations = [(float(loads[k]-capacities[k]), *intervals[k]) for k in broken]
    violations.sort(reverse=True)
    return [(a, b) for _, a, b in violations[:limit]]


def deadline_value_bound(s, possible):
    """Fractional single-prefix upper bound; preserved as the cheap first tier."""
    ids = sorted(possible, key=lambda j: (s.tasks[j].deadline, j))
    total = math.fsum(s.tasks[j].value for j in ids)
    if not ids:
        return total
    padding = max(EPS*(len(ids)+1), 8*(len(ids)+1)*math.ulp(max(1., total)))
    work = {j: s.tasks[j].work*s.multiplier for j in ids}
    ranked = sorted(ids, key=lambda j: (-s.tasks[j].value/work[j], j))
    upper = total
    prefix = set()
    prefix_value = prefix_work = 0.
    cursor = model.edge_cursor(s)
    for j in ids:
        prefix.add(j)
        prefix_value += s.tasks[j].value
        prefix_work += work[j]
        capacity = max(0., s.tasks[j].deadline-cursor)+padding
        if prefix_work <= capacity:
            continue
        retained = 0.
        remaining = capacity
        for k in ranked:
            if k not in prefix:
                continue
            amount = min(remaining, work[k])
            retained += s.tasks[k].value*(amount/work[k])
            remaining -= amount
            if remaining <= 0:
                break
        upper = min(upper, total-prefix_value+retained+padding)
    return round_upper(s, ids, upper, padding)


def round_upper(s, ids, upper, padding=0.):
    if all(float(s.tasks[j].value).is_integer() for j in ids) and math.fsum(abs(s.tasks[j].value) for j in ids) <= 2**52:
        return float(math.floor(upper+padding))
    return upper+padding


def release_lower_bounds(s, ctx):
    """Direct singleton arrival is a lower bound under metric travel and FIFO."""
    base = model.edge_cursor(s)
    releases = {}
    for j, r in ctx.base_reports.items():
        if max(base, r)+s.tasks[j].work*s.multiplier <= s.tasks[j].deadline+EPS:
            releases[j] = max(base, r)
    for j, status in s.status.items():
        if status != 'pending':
            continue
        best = float('inf')
        for vid in ctx.compatible.get(s.tasks[j].sensor, ()):
            if ctx.route(vid, (j,)) is not None:
                best = min(best, max(base, ctx.route_reports[vid, (j,)][j]))
        if best+s.tasks[j].work*s.multiplier <= s.tasks[j].deadline+EPS:
            releases[j] = best
    return releases


def release_value_bound(s, releases, cheap_upper, seconds=.005):
    """Small LP relaxation, certified from reconstructed feasible dual weights.

    For ANY lambda >= 0, b*lambda + sum(max(0,v-A.T*lambda)) is an
    upper bound with 0 <= x <= 1. Never use an unfinished primal objective.
    """
    ids = sorted(releases)
    if not ids or len(ids) > 48 or seconds <= 0:
        return cheap_upper
    base = model.edge_cursor(s)
    jobs = [(releases[j], s.tasks[j].deadline, s.tasks[j].work*s.multiplier) for j in ids]
    rows, capacities = [], []
    for d in sorted({d for _, d, _ in jobs}):
        rows.append([p if dj <= d else 0. for _, dj, p in jobs])
        capacities.append(max(0., d-base)+energy_tolerance(jobs, base, d))
    for a, b in critical_intervals(jobs, base):
        coeff = [mandatory_overlap(r, d, p, a, b) for r, d, p in jobs]
        if sum(coeff) > b-a+energy_tolerance(jobs, a, b):
            rows.append(coeff)
            capacities.append(b-a+energy_tolerance(jobs, a, b))
    values = np.asarray([s.tasks[j].value for j in ids], dtype=float)
    matrix = np.asarray(rows, dtype=float)
    rhs = np.asarray(capacities, dtype=float)
    if np.all(matrix.sum(axis=1) <= rhs):
        # x=1 attains the obvious total-value upper bound; no LP is needed.
        return cheap_upper
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
        result = linprog(-values, A_ub=matrix, b_ub=rhs, bounds=(0., 1.),
                         method='highs', options={'time_limit': seconds, 'threads': 1})
    marginals = getattr(getattr(result, 'ineqlin', None), 'marginals', None)
    if marginals is None or not np.all(np.isfinite(marginals)):
        return cheap_upper
    lam = np.maximum(0., -np.asarray(marginals))
    products = matrix.T @ lam
    dual = math.fsum(rhs*lam) + math.fsum(np.maximum(0., values-products))
    # Bound dot-product cancellation conservatively as well as solver tolerance.
    scale = max(1., abs(dual), float(np.sum(np.abs(matrix)*lam[:, None])), float(np.sum(np.abs(values))))
    padding = EPS*(len(ids)+len(rows)+1) + 32*(len(ids)+len(rows)+1)*math.ulp(scale)
    return min(cheap_upper, round_upper(s, ids, dual, padding))
