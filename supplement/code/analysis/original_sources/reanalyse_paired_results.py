"""Reanalyse the retained 30-stream grids without rerunning simulations.

The archived value rates are percentages rounded to four decimal places.
Signed ranks use exact integer differences at that precision, omit zero
differences, and use the exact conditional sign distribution (including
ties in absolute differences). The analysis is post hoc, not preregistered.

Run from the project root: python -B paper/analysis/reanalyse_paired_results.py
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "opt/runs/step16_code_only_backup/results"
OUT = Path(__file__).resolve().parent
CONDITIONS = ["coupled_primary", "edge_dominant", "tighter_deadlines", "ample_edge", "low_load"]
METHODS = ["rosr", "e2e_resource_density", "joint_lns", "coalition_value", "lit_dpso",
           "lit_lyapunov", "edge_adapter", "e2e_repack", "vehicle_repack_edf"]
FACTORIAL = [f"b{b}_c{c}_u{u}" for b in range(2) for c in range(2) for u in range(2)]


def read(path: Path) -> dict:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["condition"], row["method"], int(row["seed"]))
            if key in result:
                raise ValueError(f"duplicate observation: {key}")
            result[key] = row
    return result


def paired(data: dict, condition: str, method: str, seeds: list[int]) -> list[int]:
    # Integers prevent spurious tie distinctions from binary floating point.
    def scaled(key: tuple) -> int:
        value = Decimal(data[key]["value_rate"]) * 10000
        if value != value.to_integral_value():
            raise ValueError(f"unexpected precision in {key}")
        return int(value)
    return [scaled((condition, "rosr", seed)) - scaled((condition, method, seed))
            for seed in seeds]


def signed_rank_exact(differences: list[int]) -> tuple[float, float, int]:
    nonzero = [d for d in differences if d]
    if not nonzero:
        return 0.0, 1.0, 0
    rank2 = [int(round(2 * r)) for r in stats.rankdata([abs(d) for d in nonzero])]
    positive2 = sum(r for r, d in zip(rank2, nonzero) if d > 0)
    observed2 = min(positive2, sum(rank2) - positive2)
    distribution = Counter({0: 1})
    for rank in rank2:
        updated = distribution.copy()
        for total, count in distribution.items():
            updated[total + rank] += count
        distribution = updated
    p = min(1.0, 2 * sum(n for total, n in distribution.items() if total <= observed2)
            / 2 ** len(nonzero))
    return observed2 / 2, p, len(nonzero)


def holm(rows: list[dict]) -> None:
    largest = 0.0
    for rank, row in enumerate(sorted(rows, key=lambda item: item["p_unadjusted"])):
        largest = max(largest, min(1.0, (len(rows) - rank) * row["p_unadjusted"]))
        row["p_holm"] = largest


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(data: dict, conditions: list[str], methods: list[str], seed0: int) -> list[dict]:
    rows = []
    seeds = list(range(seed0, seed0 + 30))
    for condition in conditions:
        for method in methods:
            diffs = paired(data, condition, method, seeds)
            statistic, p, n_nonzero = signed_rank_exact(diffs)
            rows.append(dict(condition=condition, control=method, n_paired=len(diffs),
                             n_nonzero=n_nonzero, wins=sum(x > 0 for x in diffs),
                             ties=sum(x == 0 for x in diffs), losses=sum(x < 0 for x in diffs),
                             mean_difference_pp=float(np.mean(diffs) / 10000),
                             signed_rank_w=statistic, p_unadjusted=p, p_holm=0.0))
    holm(rows)
    return rows


def check_exact_distribution() -> None:
    # Compare the tie-aware integer DP to exhaustive signs and SciPy's exact
    # no-ties calculation; both checks also exercise zero removal.
    toy = [1, -2, 2, 0]
    rank2 = [2, 5, 5]
    observed2 = 5
    counts = [sum(r for r, sign in zip(rank2, signs) if sign)
              for signs in itertools.product([0, 1], repeat=3)]
    expected = min(1.0, 2 * sum(x <= observed2 for x in counts) / len(counts))
    assert signed_rank_exact(toy)[1] == expected
    no_ties = [1, -2, 3, 4, -5, 6]
    assert abs(signed_rank_exact(no_ties)[1] - stats.wilcoxon(no_ties, method="exact").pvalue) < 1e-12


def main() -> None:
    check_exact_distribution()
    paths = [SOURCE / "reference_30streams.csv", SOURCE / "factorial_30streams.csv",
             SOURCE / "cutfamily_alnsv2_30streams.csv"]
    reference, factorial, cut_data = map(read, paths)
    expected_reference = {(condition, method, seed) for condition in CONDITIONS for method in METHODS
                          for seed in range(20423000, 20423030)}
    if set(reference) != expected_reference:
        raise ValueError("reference matrix is incomplete or contains unexpected records")

    reference_tests = compare(reference, CONDITIONS, METHODS[1:], 20423000)
    factorial_tests = compare(factorial, FACTORIAL, METHODS[1:3], 20424000)
    write_csv("reference_signed_rank.csv", reference_tests)
    write_csv("factorial_signed_rank.csv", factorial_tests)

    omnibus = []
    for condition in CONDITIONS:
        arrays = [[float(reference[(condition, method, seed)]["value_rate"])
                   for seed in range(20423000, 20423030)] for method in METHODS]
        result = stats.friedmanchisquare(*arrays)
        omnibus.append(dict(condition=condition, n_blocks=30, n_methods=len(METHODS),
                            chi2=float(result.statistic), df=len(METHODS) - 1,
                            p_unadjusted=float(result.pvalue), p_holm=0.0))
    holm(omnibus)
    write_csv("reference_friedman.csv", omnibus)

    # Reference-stream cut decomposition with paired t intervals, as used in
    # the manuscript. These are descriptive and are not multiplicity adjusted.
    cut_rows = []
    for condition in CONDITIONS[:3]:
        seeds = list(range(20423000, 20423030))
        baseline = [float(cut_data[(condition, "cuts_off", seed)]["value_rate"]) for seed in seeds]
        for method in ["cuts_off", "conflict_plain", "rosr"]:
            values = [float(cut_data[(condition, method, seed)]["value_rate"]) for seed in seeds]
            times = [float(cut_data[(condition, method, seed)]["planning_ms"]) / 1000 for seed in seeds]
            differences = np.array(values) - np.array(baseline)
            mean = float(np.mean(differences))
            halfwidth = float(stats.t.ppf(.975, 29) * stats.sem(differences))
            cut_rows.append(dict(condition=condition, variant=method,
                                 mean_value_percent=float(np.mean(values)),
                                 mean_difference_pp=mean, ci_low_pp=mean-halfwidth,
                                 ci_high_pp=mean+halfwidth, mean_time_s=float(np.mean(times))))
    write_csv("cut_decomposition.csv", cut_rows)
    metadata = dict(analysis="post hoc; signed ranks conditional on nonzero rounded differences",
                    reference_signed_rank_family=40, factorial_signed_rank_family=16,
                    friedman_family=5, multiplicity="Holm step-down within each stated family",
                    scipy=scipy.__version__, numpy=np.__version__,
                    input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    original_event_logs_or_source_hashes_recovered=False)
    (OUT / "reanalysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in reference_tests[:8]:
        print("primary", row["control"], "W=", row["signed_rank_w"],
              "p=", f'{row["p_unadjusted"]:.6g}', "Holm=", f'{row["p_holm"]:.6g}')
    for row in omnibus:
        print("Friedman", row["condition"], f'chi2={row["chi2"]:.4f}', f'Holm={row["p_holm"]:.6g}')
    for row in factorial_tests:
        print("factorial", row["condition"], row["control"], f'Holm={row["p_holm"]:.6g}')
    for row in cut_rows:
        print("cut", row["condition"], row["variant"],
              f'{row["mean_value_percent"]:.2f}', f'{row["mean_difference_pp"]:+.2f}',
              f'[{row["ci_low_pp"]:+.2f},{row["ci_high_pp"]:+.2f}]', f'{row["mean_time_s"]:.2f}')


if __name__ == "__main__":
    main()
