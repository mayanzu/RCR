"""Re-audit saved OSM episodes and independently check paired summaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.stats import t

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
from rosr import model, verify_frozen
import audit_osm
import case_contract
from run_case import configuration


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def assert_close(a, b, tolerance=1e-8):
    if abs(a - b) > tolerance:
        raise AssertionError(f"{a} != {b} (tolerance {tolerance})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="confirmation")
    args = parser.parse_args()
    out = HERE / "results" / args.stage
    settings = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    verify_frozen()
    _, current_fingerprint = configuration(args.stage)
    if current_fingerprint != settings["fingerprint"]:
        raise AssertionError("Current source, data or environment differs from the experiment fingerprint")
    case_contract.install(model)
    data = {}
    plan_checks = 0
    continuation_checks = 0
    for profile_name in settings["profiles"]:
        profile = next(p for p in model.PROFILES if p.name == profile_name)
        for seed in settings["seeds"]:
            tasks, vehicles = model.generate(seed, profile)
            stream_hash = model.signature(tasks, vehicles)
            for method in settings["methods"]:
                ep = out / "episodes" / profile_name / str(seed) / method
                summary = json.loads((ep / "summary.json").read_text(encoding="utf-8"))
                rows, events, decisions = [read_gzip(ep / name) for name in
                    ("tasks.json.gz", "events.json.gz", "decisions.json.gz")]
                if summary["evidence_fingerprint"] != settings["fingerprint"]:
                    raise AssertionError(f"Different fingerprint in {ep}")
                if summary["stream_hash"] != stream_hash:
                    raise AssertionError(f"Different task stream in {ep}")
                audit_osm.check_episode(summary, rows, events)
                audit_osm.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, events)
                total = sum(r["value"] for r in rows)
                complete = sum(r["value"] for r in rows if r["status"] == "completed")
                assert_close(summary["total_value"], total)
                assert_close(summary["completed_value"], complete)
                assert_close(summary["value_rate"], complete / total if total else 0.0)
                count = sum(e["kind"] == "plan" for e in events)
                if count != len(decisions):
                    raise AssertionError(f"Plan/decision mismatch in {ep}")
                plan_checks += count
                if method == "rosr":
                    for d in decisions:
                        previous = d.get("previous_guaranteed_value")
                        if d.get("continuation_status") == "valid" and previous is not None:
                            if d["guaranteed_value"] + 1e-8 < previous:
                                raise AssertionError(f"Continuation guarantee decreased in {ep}")
                            continuation_checks += 1
                data[profile_name, seed, method] = summary
    for cell in analysis["cells"]:
        values = [data[cell["profile"], seed, cell["method"]] for seed in settings["seeds"]]
        fields = (("mean_value_pct", "value_rate", 100),
                  ("mean_count_pct", "count_rate", 100),
                  ("mean_downstream_pct", "downstream_lost_value_rate", 100),
                  ("mean_uncaptured_pct", "uncaptured_lost_value_rate", 100),
                  ("mean_wasted_minutes", "wasted_capture_time", 1),
                  ("mean_planning_seconds", "planning_ms", 0.001))
        for printed, raw, scale in fields:
            assert_close(cell[printed], statistics.mean(v[raw] * scale for v in values))
    for record in analysis["comparisons"]:
        profile, control = record["profile"], record["control"]
        diffs, exact = [], []
        for seed in settings["seeds"]:
            a, b = data[profile, seed, "rosr"], data[profile, seed, control]
            diffs.append(100 * (a["value_rate"] - b["value_rate"]))
            exact.append(a["completed_value"] - b["completed_value"])
        mu = statistics.mean(diffs)
        half = float(t.ppf(.975, len(diffs)-1)) * statistics.stdev(diffs) / math.sqrt(len(diffs))
        for key, value in (("mean", mu), ("low95", mu-half), ("high95", mu+half)):
            assert_close(record["value_difference_pp"][key], value)
        if (record["wins"], record["ties"], record["losses"]) != (
                sum(x > 0 for x in exact), sum(x == 0 for x in exact), sum(x < 0 for x in exact)):
            raise AssertionError(f"Incorrect W/T/L: {profile}/{control}")
    report = {"stage": args.stage, "episodes": len(data), "plan_checks": plan_checks,
              "valid_continuation_checks": continuation_checks,
              "mean_cells": len(analysis["cells"]), "paired_comparisons": len(analysis["comparisons"]),
              "evidence_fingerprint": settings["fingerprint"],
              "analysis_sha256": hashlib.sha256((out / "analysis.json").read_bytes()).hexdigest(),
              "result": "PASS"}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
