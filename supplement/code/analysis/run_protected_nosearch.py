"""Paired Protected No-Search ablation isolating LBBD search from continuation/guards.

Runs DensityPolicy (fresh constructor, no continuation/guards) and
ProtectedNoSearchPolicy (identical to RosrPolicy continuation, bounds,
tie probe, guarded baseline selection, and guarded acceptance, with only
route-pool packing / LBBD search skipped) across:
  - 30 paired grid reference streams (20423000-20423029) on m in {1.0, 1.5, 2.0, 2.5, 3.0}
    and tighter_deadlines (m=1.5, slack (8, 12))
  - 30 paired Hefei road-network streams (20480000-20480029) on primary, edge_limited,
    and tighter_deadlines (comparing against Density and fresh/retained RCR)

Run from project root:
    python -B code/analysis/run_protected_nosearch.py
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import csv
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from reanalyse_paired_results import signed_rank_exact

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
OUT = Path(__file__).resolve().parent
OSM_DIR = CODE / "osm_hefei"
if not OSM_DIR.is_dir():
    OSM_DIR = CODE / "history" / "osm_hefei_original"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(OSM_DIR))

from rosr import RosrPolicy, model
from rosr.boot import EPS, audit, validate_plan
from rosr.capacity import deadline_value_bound, release_lower_bounds
from rosr.comparison import DensityPolicy
from rosr.continuation import Continuation
from rosr.criteria import ACCEPTED, acceptance_reason
from rosr.evaluation import EvaluationContext, density_initialize, local_equal_value_probe


class ProtectedNoSearchPolicy(RosrPolicy):
    """Identical to RosrPolicy except route-pool enumeration and LBBD packing are skipped."""

    def __call__(self, s):
        started = time.perf_counter()
        if s.now < self.last_now or s.event_index < self.last_event:
            self.conflicts = self.conflicts.__class__()
            self.continuation = Continuation()
        self.last_now = s.now
        self.last_event = s.event_index
        ctx = EvaluationContext(s, conflicts=None)
        carried, _ = self.continuation.advance(s)
        previous_guaranteed = self.continuation.guaranteed
        releases = release_lower_bounds(s, ctx)
        bound = deadline_value_bound(s, releases)
        capture_possible = any(
            v.active is None and any(
                status == "pending" and s.tasks[j].sensor == v.sensor
                for j, status in s.status.items()
            )
            for v in s.vehicles
        )
        edge_possible = s.edge_active is None and "ready" in s.status.values()
        ceiling = carried is not None and bound <= carried.value + EPS
        if ceiling and (not self.compact_ties or not (capture_possible or edge_possible)):
            density = carried
        elif ceiling:
            density = local_equal_value_probe(s, ctx, carried)
            density = carried if density is None else density
        else:
            density = density_initialize(s, ctx, value_upper=bound, urgency_alpha=self.urgency_alpha)
        baseline = carried if carried is not None else density
        if carried is not None and density is not carried and acceptance_reason(
            s, density, carried, compact_ties=self.compact_ties, guarded=True, slack_price=self.slack_price
        ) in ACCEPTED:
            baseline = density
        chosen = baseline
        validate_plan(s, chosen)
        guaranteed = Continuation.value(s, chosen)
        if carried is not None and previous_guaranteed is not None:
            assert guaranteed >= previous_guaranteed - EPS
        self.continuation.remember(s, chosen)
        self.stats.events += 1
        self.stats.planning_seconds += time.perf_counter() - started
        return chosen


def run_grid_job(args):
    seed, pname, mult, slack, method = args
    prof = model.Profile(pname, arrival_rate=1.0, edge_multiplier=mult, slack=slack)
    tasks, vehicles = model.generate(seed, prof)
    pol = DensityPolicy() if method == "density" else ProtectedNoSearchPolicy(seed=seed)
    summary, rows, logs = model.run_episode(seed, prof, method, policy=pol)
    audit.check_episode(summary, rows, logs)
    audit.audit_action_logs(tasks, vehicles, prof.edge_multiplier, rows, logs)
    return (pname, method, seed, float(summary["value_rate"]) * 100.0, float(summary["planning_ms"]) / 1000.0)


def run_hefei_job(args):
    import audit_osm
    import case_contract
    seed, pname, method = args
    prof = next(p for p in model.PROFILES if p.name == pname)
    case_contract.install(model)
    tasks, vehicles = model.generate(seed, prof)
    pol = DensityPolicy() if method == "density" else ProtectedNoSearchPolicy(seed=seed)
    summary, rows, logs = model.run_episode(seed, prof, method, policy=pol)
    audit_osm.check_episode(summary, rows, logs)
    audit_osm.audit_action_logs(tasks, vehicles, prof.edge_multiplier, rows, logs)
    return (f"hefei_{pname}", method, seed, float(summary["value_rate"]) * 100.0, float(summary["planning_ms"]) / 1000.0)


def paired_stats(a, b, percent_decimals=4):
    """Means/CIs use retained values; ranks/WTL use a common observation precision.

    Grid RCR archives store percent values to four decimals. Quantize both
    arms BEFORE differencing so rounding residues cannot turn ties into wins.
    Full-precision Hefei observations use ten decimal percent precision.
    The exact conditional distribution also handles equal absolute ranks.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    if n < 2 or a.shape != b.shape or not np.all(np.isfinite(d)):
        raise ValueError("At least two finite paired observations are required")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    rad = float(student_t.ppf(0.975, n - 1) * se) if se > 0 else 0.0
    scale = 10 ** percent_decimals
    ranked = np.rint(a * scale).astype(np.int64) - np.rint(b * scale).astype(np.int64)
    statistic, p_val, n_nonzero = signed_rank_exact(ranked.tolist())
    wins, ties, losses = (int(np.sum(ranked > 0)), int(np.sum(ranked == 0)), int(np.sum(ranked < 0)))
    return dict(mean=mean, ci_lo=mean - rad, ci_hi=mean + rad,
                wtl=f"{wins}/{ties}/{losses}", p_wilcoxon=p_val,
                signed_rank_w=statistic, n_nonzero=n_nonzero,
                rank_percent_decimals=percent_decimals)


def main():
    grid_seeds = list(range(20423000, 20423030))
    hefei_seeds = list(range(20480000, 20480030))
    jobs_grid = [
        (s, pname, mult, slack, m)
        for pname, mult, slack in [
            ("m_1.0", 1.0, (16.0, 24.0)),
            ("coupled_primary", 1.5, (16.0, 24.0)),
            ("m_2.0", 2.0, (16.0, 24.0)),
            ("edge_dominant", 2.5, (16.0, 24.0)),
            ("m_3.0", 3.0, (16.0, 24.0)),
            ("tighter_deadlines", 1.5, (8.0, 12.0)),
        ]
        for s in grid_seeds
        for m in ("density", "protected_ns")
    ]
    jobs_hefei = [
        (s, pname, m)
        for pname in ("coupled_primary", "edge_dominant", "tighter_deadlines")
        for s in hefei_seeds
        for m in ("density", "protected_ns")
    ]
    res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_grid_job, j) for j in jobs_grid] + [ex.submit(run_hefei_job, j) for j in jobs_hefei]
        for fut in as_completed(futs):
            k1, k2, seed, val, sec = fut.result()
            res[(k1, k2, seed)] = (val, sec)

    hist_grid = {}
    with (ROOT / "code/history/step16_code_only_backup/results/reference_30streams.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            hist_grid[(row["condition"], row["method"], int(row["seed"]))] = float(row["value_rate"])

    hist_hefei = {}
    with (ROOT / "opt/osm_hefei/results/confirmation/episodes.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            hist_hefei[(row["profile"], row["method"], int(row["seed"]))] = 100.0 * float(row["value_rate"])

    out = {"grid": {}, "hefei": {}}
    for pname in ("m_1.0", "coupled_primary", "m_2.0", "edge_dominant", "m_3.0", "tighter_deadlines"):
        den = [res[(pname, "density", s)][0] for s in grid_seeds]
        pns = [res[(pname, "protected_ns", s)][0] for s in grid_seeds]
        entry = {
            "density_mean": float(np.mean(den)),
            "protected_ns_mean": float(np.mean(pns)),
            "protected_ns_minus_density": paired_stats(pns, den),
            "protected_ns_plan_s": float(np.mean([res[(pname, "protected_ns", s)][1] for s in grid_seeds])),
        }
        if pname in ("coupled_primary", "edge_dominant", "tighter_deadlines"):
            rcr = [hist_grid[(pname, "rosr", s)] for s in grid_seeds]
            entry["rcr_table2_mean"] = float(np.mean(rcr))
            entry["rcr_minus_protected_ns"] = paired_stats(rcr, pns)
        out["grid"][pname] = entry

    for pname in ("coupled_primary", "edge_dominant", "tighter_deadlines"):
        kp = f"hefei_{pname}"
        den = [res[(kp, "density", s)][0] for s in hefei_seeds]
        pns = [res[(kp, "protected_ns", s)][0] for s in hefei_seeds]
        rcr = [hist_hefei[(pname, "rosr", s)] for s in hefei_seeds]
        out["hefei"][pname] = {
            "density_mean": float(np.mean(den)),
            "protected_ns_mean": float(np.mean(pns)),
            "rcr_table11_mean": float(np.mean(rcr)),
            "protected_ns_minus_density": paired_stats(pns, den, percent_decimals=10),
            "rcr_minus_protected_ns": paired_stats(rcr, pns, percent_decimals=10),
            "protected_ns_plan_s": float(np.mean([res[(kp, "protected_ns", s)][1] for s in hefei_seeds])),
        }

    observations = [dict(condition=c, method=m, seed=seed, value_rate_percent=v,
                         planning_seconds=seconds, provenance="new audited execution")
                    for (c, m, seed), (v, seconds) in sorted(res.items())]
    for (condition, method, seed), value in sorted(hist_grid.items()):
        if condition in ("coupled_primary", "edge_dominant", "tighter_deadlines") and method == "rosr":
            observations.append(dict(condition=condition, method="rosr", seed=seed,
                value_rate_percent=value, planning_seconds="", provenance="retained grid CSV, four decimal percent"))
    for (condition, method, seed), value in sorted(hist_hefei.items()):
        if method == "rosr":
            observations.append(dict(condition="hefei_" + condition, method="rosr", seed=seed,
                value_rate_percent=value, planning_seconds="", provenance="retained fingerprinted Hefei episodes CSV"))
    with (OUT / "protected_nosearch_observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    sources = [Path(__file__), OUT / "reanalyse_paired_results.py",
               ROOT / "code/history/step16_code_only_backup/results/reference_30streams.csv",
               ROOT / "opt/osm_hefei/results/confirmation/episodes.csv"]
    sources += sorted((CODE / "rosr").glob("*.py")) + sorted((CODE / "runtime").glob("*.py"))
    metadata = dict(grid_rank_percent_decimals=4, hefei_rank_percent_decimals=10,
        inference="raw two-sided conditional exact signed-rank, zeros omitted, absolute-rank ties handled",
        mean_ci="unadjusted paired Student interval on retained observation values",
        workers=8, observations=len(observations),
        source_sha256={f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    (OUT / "protected_nosearch_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUT / "protected_nosearch_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
