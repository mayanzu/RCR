"""Standalone deterministic work-budget repeatability verification for RCR.

Wraps the frozen RosrPolicy with `WorkBudgetPolicy` from `code/opt/work_budget.py`
(256 cumulative MILP nodes, 16 MIP calls, 10,000 LP iterations per event) and
verifies 100% bit-identical value, execution traces, action traces, and solver
work traces across repeated runs on the same machine/software environment.

Note: This diagnostic replaces the 0.1 s wall-clock stopping rule with solver
work ceilings and therefore represents a separate repeatability check rather than
the original wall-clock policy.

Example:
    python -B code/analysis/run_work_budget_repro.py --seeds 20460400-20460402 --profiles coupled_primary
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "opt"))

from rosr import RosrPolicy, model
from rosr.boot import audit
from work_budget import WorkBudgetPolicy, WorkLimits


def parse_seeds(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def traces_hash(logs, decisions):
    exec_h, act_h, work_h = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for ev in logs:
        clean = {k: v for k, v in ev.items() if k != "method"}
        enc = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode() + b"\n"
        exec_h.update(enc)
        if ev.get("kind") in ("capture_start", "edge_start"):
            act_h.update(enc)
    for d in decisions:
        w = {
            "event": d.get("event"),
            "work_used": d.get("work_used"),
            "stop_reason": d.get("meta", {}).get("stop_reason"),
            "mode": d.get("mode"),
        }
        work_h.update(json.dumps(w, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return exec_h.hexdigest(), act_h.hexdigest(), work_h.hexdigest()


def run_once(seed, profile_name, limits):
    profile = next(p for p in model.PROFILES if p.name == profile_name)
    tasks, vehicles = model.generate(seed, profile)
    pol = WorkBudgetPolicy(RosrPolicy(seed=seed), limits=limits, mode="work")
    summary, rows, logs = model.run_episode(seed, profile, "rcr_work", policy=pol)
    audit.check_episode(summary, rows, logs)
    audit.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, logs)
    ex, ac, wk = traces_hash(logs, pol.decisions)
    return dict(
        seed=seed,
        profile=profile_name,
        completed_value=float(summary["completed_value"]),
        value_rate_pct=100.0 * float(summary["value_rate"]),
        execution_sha256=ex,
        action_sha256=ac,
        work_sha256=wk,
        wall_guard_hits=int(pol.optimization_stats.get("wall_guard_hits", 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20460400-20460401")
    ap.add_argument("--profiles", default="coupled_primary,edge_dominant,tighter_deadlines")
    ap.add_argument("--nodes", type=int, default=256)
    ap.add_argument("--calls", type=int, default=16)
    ap.add_argument("--lp-iters", type=int, default=10000)
    args = ap.parse_args()

    limits = WorkLimits(mip_nodes=args.nodes, mip_calls=args.calls, lp_iterations=args.lp_iters)
    seeds = parse_seeds(args.seeds)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    pairs, mismatches = 0, 0
    for prof in profiles:
        for seed in seeds:
            r1 = run_once(seed, prof, limits)
            r2 = run_once(seed, prof, limits)
            pairs += 1
            same = (
                r1["completed_value"] == r2["completed_value"]
                and r1["execution_sha256"] == r2["execution_sha256"]
                and r1["action_sha256"] == r2["action_sha256"]
                and r1["work_sha256"] == r2["work_sha256"]
                and r1["wall_guard_hits"] == 0
                and r2["wall_guard_hits"] == 0
            )
            if not same:
                mismatches += 1
    print(json.dumps({"pairs": pairs, "matching_pairs": pairs - mismatches, "mismatches": mismatches}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
