"""Run audited paired OSM-network episodes with frozen policies.

Usage: python -B opt/osm_hefei/run_case.py --stage pilot --workers 3
       python -B opt/osm_hefei/run_case.py --stage confirmation --workers 3

Existing episodes are reused only if every evidence fingerprint matches.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import csv
from dataclasses import asdict, is_dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import scipy
from scipy.stats import t

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

from rosr import RosrPolicy, model, verify_frozen
from rosr.comparison import DensityPolicy, JointLnsPolicy
from value_coalition import ValueCoalitionPolicy
from lit_dpso import DpsoPolicy
import audit_osm
import case_contract

PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
METHODS = tuple(PROTOCOL["methods"])
PROFILES = tuple(PROTOCOL["profiles"])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_hash() -> str:
    h = hashlib.sha256()
    for path in sorted((CODE / "rosr").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def to_json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value)}")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=to_json) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_gzip(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(obj, f, ensure_ascii=False, default=to_json, separators=(",", ":"))
    tmp.replace(path)


def configuration(stage):
    runtime_hashes = verify_frozen()
    frozen = json.loads((CODE / "frozen_latest_policy.json").read_text(encoding="utf-8"))
    if policy_hash() != frozen["source_sha256"]:
        raise RuntimeError("Frozen policy hash mismatch")
    for name in ("value_coalition.py", "lit_dpso.py"):
        if sha(CODE / name) != frozen["entrypoint_sha256"][name]:
            raise RuntimeError(f"Frozen comparison control mismatch: {name}")
    source_files = sorted(CODE.rglob("*.py")) + [CODE / "frozen_latest_policy.json", CODE / "latest_source_manifest.json"]
    sources = {p.relative_to(CODE).as_posix(): sha(p) for p in source_files}
    case_files = ("protocol.json", "case_contract.py", "audit_osm.py", "run_case.py")
    config = {
        "stage": stage, "protocol": PROTOCOL["protocol"],
        "case_sha256": {name: sha(HERE / name) for name in case_files},
        "source_sha256": policy_hash(), "runtime_sha256": runtime_hashes,
        "code_files_sha256": sources,
        "raw_osm_sha256": case_contract.META["raw_osm_sha256"],
        "network_sha256": case_contract.META["graphml_sha256"],
        "distance_sha256": case_contract.META["matrix_sha256"],
        "independent_distance_sha256": case_contract.META["audit_matrix_sha256"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "processor": platform.processor(),
                        "threads": {k: os.environ[k] for k in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "method_settings": {"RCR": "frozen defaults, additional search 0.1 s",
                            "Density": "frozen defaults", "Joint DR": "compact_ties=False, 0.1 s",
                            "Coalition-V": "frozen defaults, 0.1 s", "DPSO": "frozen defaults, 0.1 s"},
    }
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return config, hashlib.sha256(serialized).hexdigest()


def snapshot_code(out, config):
    target = out / "frozen_code_snapshot.zip"
    if target.exists():
        with zipfile.ZipFile(target) as z:
            for name, expected in config["code_files_sha256"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != expected:
                    raise RuntimeError(f"Cached snapshot differs from current source: {name}")
        return
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in config["code_files_sha256"]:
            z.write(CODE / name, arcname=name)


def make_policy(method, seed):
    if method == "rosr":
        return RosrPolicy(seed=seed)
    if method == "e2e_resource_density":
        return DensityPolicy()
    if method == "joint_lns":
        return JointLnsPolicy(compact_ties=False)
    if method == "coalition_value":
        return ValueCoalitionPolicy()
    if method == "lit_dpso":
        return DpsoPolicy(seed=seed)
    raise ValueError(method)


def run_one(seed, profile_name, method, outdir, fingerprint):
    ep = Path(outdir) / "episodes" / profile_name / str(seed) / method
    summary_path = ep / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("evidence_fingerprint") != fingerprint:
            raise RuntimeError(f"Cached episode has a different evidence fingerprint: {ep}")
        for required in ("tasks.json.gz", "events.json.gz", "decisions.json.gz"):
            if not (ep / required).exists():
                raise RuntimeError(f"Cached episode is missing required audit trace: {ep / required}")
        return summary, False
    profile = next(p for p in model.PROFILES if p.name == profile_name)
    case_contract.install(model)
    tasks, vehicles = model.generate(seed, profile)
    policy = make_policy(method, seed)
    summary, rows, events = model.run_episode(seed, profile, method, policy=policy)
    audit_osm.check_episode(summary, rows, events)
    audit_osm.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, events)
    if summary["stream_hash"] != model.signature(tasks, vehicles):
        raise RuntimeError("Simulator and audit received different task streams")
    stats = asdict(policy.stats) if is_dataclass(policy.stats) else dict(policy.stats)
    latencies = [float(d["planning_ms"]) for d in policy.decisions if "planning_ms" in d]
    plan_count = sum(e["kind"] == "plan" for e in events)
    if len(latencies) != plan_count:
        raise RuntimeError(f"Decision count does not match recorded plans for {ep}")
    summary.update({"policy_stats": stats, "event_latencies_ms": latencies,
                    "decision_events": plan_count,
                    "audit": "PASS: physical resources and independent road-distance actions",
                    "evidence_fingerprint": fingerprint,
                    "distance_sha256": case_contract.META["matrix_sha256"]})
    save_gzip(ep / "tasks.json.gz", rows)
    save_gzip(ep / "events.json.gz", events)
    save_gzip(ep / "decisions.json.gz", policy.decisions)
    # Write summary last: it is the completion marker for resume.
    save_json(summary_path, summary)
    return summary, True


def worker(job):
    seed, profile_name, outdir, fingerprint = job
    shift = (seed + PROFILES.index(profile_name)) % len(METHODS)
    order = METHODS[shift:] + METHODS[:shift]
    rows = []
    for method in order:
        begin = time.perf_counter()
        summary, ran = run_one(seed, profile_name, method, outdir, fingerprint)
        print(f'{profile_name} seed={seed} method={method} value={100*summary["value_rate"]:.2f}% '
              f'plan={summary["planning_ms"]/1000:.2f}s {"ran" if ran else "cached"} '
              f'wall={time.perf_counter()-begin:.1f}s', flush=True)
        rows.append((seed, profile_name, method, summary))
    return rows


def interval(xs):
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        raise ValueError("At least two paired streams are required")
    mean = float(np.mean(a))
    half = float(t.ppf(.975, len(a) - 1) * np.std(a, ddof=1) / math.sqrt(len(a)))
    return {"n": len(a), "mean": mean, "low95": mean - half, "high95": mean + half}


def aggregate(out, results, seeds):
    episodes = []
    metrics = ("value_rate", "count_rate", "downstream_lost_value_rate",
               "uncaptured_lost_value_rate", "wasted_capture_time", "planning_ms")
    for (seed, profile, method), summary in sorted(results.items()):
        episodes.append({"seed": seed, "profile": profile, "method": method,
                         **{key: summary[key] for key in metrics},
                         "completed_value": summary["completed_value"],
                         "total_value": summary["total_value"], "stream_hash": summary["stream_hash"]})
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    cells, comparisons = [], []
    for profile in PROFILES:
        for seed in seeds:
            hashes = {results[(seed, profile, method)]["stream_hash"] for method in METHODS}
            if len(hashes) != 1:
                raise RuntimeError(f"Unpaired task stream for {profile}/{seed}")
        for method in METHODS:
            values = [results[(seed, profile, method)] for seed in seeds]
            event_latencies = [lat for s in values for lat in s["event_latencies_ms"]]
            cells.append({"profile": profile, "method": method, "n": len(values),
                          "mean_value_pct": float(np.mean([100 * x["value_rate"] for x in values])),
                          "mean_count_pct": float(np.mean([100 * x["count_rate"] for x in values])),
                          "mean_downstream_pct": float(np.mean([100 * x["downstream_lost_value_rate"] for x in values])),
                          "mean_uncaptured_pct": float(np.mean([100 * x["uncaptured_lost_value_rate"] for x in values])),
                          "mean_wasted_minutes": float(np.mean([x["wasted_capture_time"] for x in values])),
                          "mean_planning_seconds": float(np.mean([x["planning_ms"] / 1000 for x in values])),
                          "event_planning_ms_p99": float(np.percentile(event_latencies, 99)),
                          "event_count": len(event_latencies)})
        for control in METHODS[1:]:
            rcr = [results[(seed, profile, "rosr")] for seed in seeds]
            other = [results[(seed, profile, control)] for seed in seeds]
            diffs = [100 * (a["value_rate"] - b["value_rate"]) for a, b in zip(rcr, other)]
            exact = [a["completed_value"] - b["completed_value"] for a, b in zip(rcr, other)]
            comparisons.append({"profile": profile, "control": control,
                                "value_difference_pp": interval(diffs),
                                "wins": sum(x > 0 for x in exact),
                                "ties": sum(x == 0 for x in exact),
                                "losses": sum(x < 0 for x in exact)})
    save_json(out / "analysis.json", {"cells": cells, "comparisons": comparisons,
                                      "interval": "unadjusted paired 95% Student t",
                                      "pairing": "same seed and task stream within profile"})
    for item in comparisons:
        v = item["value_difference_pp"]
        print(f'{item["profile"]} RCR - {item["control"]}: {v["mean"]:+.3f} pp '
              f'[{v["low95"]:+.3f},{v["high95"]:+.3f}], '
              f'W/T/L {item["wins"]}/{item["ties"]}/{item["losses"]}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "confirmation"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    seeds = (PROTOCOL["pilot_seeds"] if args.stage == "pilot" else
             list(range(PROTOCOL["confirmation_seed_start"], PROTOCOL["confirmation_seed_end"] + 1)))
    out = HERE / "results" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    config, fingerprint = configuration(args.stage)
    config.update({"fingerprint": fingerprint, "workers": args.workers, "seeds": seeds,
                   "profiles": PROFILES, "methods": METHODS})
    settings = out / "settings.json"
    if settings.exists():
        existing = json.loads(settings.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(config)):
            raise RuntimeError("Run settings changed; choose a new experiment rather than mixing data")
    else:
        save_json(settings, config)
    snapshot_code(out, config)
    jobs = [(seed, profile, str(out), fingerprint) for profile in PROFILES for seed in seeds]
    all_results = {}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            for seed, profile, method, summary in future.result():
                all_results[(seed, profile, method)] = summary
    if len(all_results) != len(seeds) * len(PROFILES) * len(METHODS):
        raise RuntimeError("Incomplete paired experiment")
    aggregate(out, all_results, seeds)
    print(f'Completed {len(all_results)} audited episodes in {time.perf_counter()-start:.1f}s', flush=True)


if __name__ == "__main__":
    main()
