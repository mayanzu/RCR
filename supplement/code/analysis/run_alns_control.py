"""Standalone audited runners for the standard ALNS and adaptive ALNSv2 controls.

Reconstructs the exact ALNSPolicy (random 1--2 task destroy + SA cooling) and
ALNSv2 (adaptive roulette over random/worst/segment removal + SA cooling)
controls from the historical step16 experiment scripts on top of the current
frozen `code/rosr` simulator and audit suite.

Example (single paired stream smoke check):
    python -B code/analysis/run_alns_control.py --variant alnsv2 --seconds 0.1 --seeds 20423000 --profiles coupled_primary
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from rosr import model
from rosr.boot import audit
from rosr.evaluation import EvaluationContext, density_initialize


class ALNSPolicy:
    """Single-operator destroy/repair ALNS with exponential SA cooling."""

    def __init__(self, seed=0, seconds=0.1, destroy_k=2):
        self.seed, self.seconds, self.k = seed, seconds, destroy_k
        self.decisions = []
        self.iters = 0

    def __call__(self, s):
        started = time.perf_counter()
        deadline = started + self.seconds
        ctx = EvaluationContext(s)
        cur = density_initialize(s, ctx)
        if cur is None:
            return None
        best = cur
        rng = random.Random((self.seed * 131) ^ int(s.now * 1000) ^ s.event_index)
        temp0, temp1 = 1.5, 0.02
        iters = 0
        while time.perf_counter() < deadline:
            iters += 1
            frac = (time.perf_counter() - started) / max(self.seconds, 1e-9)
            temp = temp0 * (temp1 / temp0) ** min(1.0, frac)
            chosen = [j for r in cur.routes for j in r]
            if not chosen:
                break
            drop = rng.sample(chosen, min(self.k, len(chosen)))
            routes = [[j for j in r if j not in drop] for r in cur.routes]
            trial = density_initialize(s, ctx, routes=routes, stop=lambda: time.perf_counter() >= deadline)
            if trial is None or trial.value <= cur.value + 1e-12:
                if trial is not None and rng.random() < math.exp((trial.value - cur.value) / max(temp, 1e-9)):
                    cur = trial
            else:
                cur = trial
                if trial.value > best.value + 1e-12:
                    best = trial
        self.iters = iters
        self.decisions.append(dict(event=s.event_index, chosen_value=best.value, iters=iters))
        return best


class ALNSv2:
    """Adaptive multi-operator ALNS: random/worst/segment destroy, roulette weights, SA cooling."""

    def __init__(self, seed=0, seconds=0.1):
        self.seed, self.seconds = seed, seconds
        self.decisions = []
        self.iters = 0

    def __call__(self, s):
        started = time.perf_counter()
        deadline = started + self.seconds
        ctx = EvaluationContext(s)
        cur = density_initialize(s, ctx)
        if cur is None:
            return None
        best = cur
        rng = random.Random((self.seed * 2654435761) ^ int(s.now * 1000) ^ s.event_index)
        weights = {"random": 1.0, "worst": 1.0, "segment": 1.0}
        reward = {k: 0.0 for k in weights}
        uses = {k: 0 for k in weights}
        temp0, temp1 = 1.2, 0.02
        it = 0
        while time.perf_counter() < deadline:
            it += 1
            frac = (time.perf_counter() - started) / max(self.seconds, 1e-9)
            temp = temp0 * (temp1 / temp0) ** min(1.0, frac)
            op = rng.choices(list(weights), weights=[weights[k] for k in weights])[0]
            chosen = [j for r in cur.routes for j in r]
            if not chosen:
                break
            k = rng.randint(1, min(5, len(chosen)))
            if op == "random":
                drop = rng.sample(chosen, k)
            elif op == "worst":
                sc = {
                    j: s.tasks[j].value / max(1e-9, s.tasks[j].work * s.multiplier + s.tasks[j].service + s.tasks[j].upload)
                    for j in chosen
                }
                drop = sorted(sc, key=sc.get)[:k]
            else:
                vid = rng.randrange(len(cur.routes))
                seg = list(cur.routes[vid])
                if not seg:
                    drop = rng.sample(chosen, min(k, len(chosen)))
                else:
                    i0 = rng.randrange(len(seg))
                    drop = seg[i0:i0 + k]
            routes = [[j for j in r if j not in drop] for r in cur.routes]
            trial = density_initialize(s, ctx, routes=routes, stop=lambda: time.perf_counter() >= deadline)
            uses[op] += 1
            if trial is None:
                continue
            delta = trial.value - cur.value
            if delta > 1e-12:
                reward[op] += 1.0
                cur = trial
                if trial.value > best.value + 1e-12:
                    best = trial
            elif delta < -1e-9 and rng.random() < math.exp(delta / max(temp, 1e-9)):
                reward[op] += 0.25
                cur = trial
            if it % 25 == 0:
                for op2 in weights:
                    weights[op2] = 0.5 * weights[op2] + 0.5 * (reward[op2] / max(1, uses[op2]))
        self.iters = it
        self.decisions.append(dict(event=s.event_index, chosen_value=best.value, iters=it))
        return best


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


def run_one(args):
    seed, profile_name, variant, seconds = args
    profile = next(p for p in model.PROFILES if p.name == profile_name)
    tasks, vehicles = model.generate(seed, profile)
    pol = ALNSPolicy(seed=seed, seconds=seconds) if variant == "alns" else ALNSv2(seed=seed, seconds=seconds)
    summary, rows, logs = model.run_episode(seed, profile, variant, policy=pol)
    audit.check_episode(summary, rows, logs)
    audit.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, logs)
    return dict(
        seed=seed, profile=profile_name, variant=variant, seconds=seconds,
        value_rate_pct=100.0 * float(summary["value_rate"]),
        planning_s=float(summary["planning_ms"]) / 1000.0,
        stream_hash=summary["stream_hash"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("alns", "alnsv2"), default="alnsv2")
    ap.add_argument("--seconds", type=float, default=0.1)
    ap.add_argument("--seeds", default="20423000")
    ap.add_argument("--profiles", default="coupled_primary")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    jobs = [(s, p.strip(), args.variant, args.seconds)
            for p in args.profiles.split(",") for s in parse_seeds(args.seeds)]
    if args.workers <= 1:
        results = [run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = [f.result() for f in as_completed([ex.submit(run_one, j) for j in jobs])]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
