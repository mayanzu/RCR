# VCS-E2E: Minimal Algorithm Reproduction Package

This package provides the RCR scheduling algorithm, its simulator, comparison
policies, experiment entry points, and mechanism tests. RCR is implemented in
`code/rosr/`; the historical internal name `rosr` remains in filenames and
command-line method names.

This package runs and verifies the algorithm standalone. The core algorithm, model, and comparison policies retain their original sources; scripts are adapted to this package layout.

## Scope

The simulator generates synthetic task streams from the supplied seeds. A
single run compares RCR with the density baseline on the same stream and
checks both trajectories against the physical and action audits.

This integrated package combines the minimal algorithm implementation and
tests with the manuscript's supplementary experiments and Hefei road-network
evidence. It contains five historical per-stream result summary tables, the
recorded Hefei network and traces, and additional experiment results.

The retained historical summaries do **not** contain the original per-event
logs or the original source fingerprints for those grid experiments. They can
be reanalysed but cannot independently establish an exact replay of every
historical run. The Hefei evidence is separately fingerprinted and can be
replayed. A new wall-clock-limited run may differ from the archived results.

This integrated package contains the algorithm, supplementary experiments, five historical per-stream summary tables, and the Hefei road-network traces. The per-event raw logs of the historical grid experiments are not recovered; the summary tables and unit tests must not be taken as per-event replays of every historical run.

## Environment

- 64-bit CPython 3.12 and Windows 11 were used for the checks described below.
- Install the pinned NumPy and SciPy versions from `requirements.txt`. SciPy
  supplies the HiGHS optimization backend; no separate solver is required for
  the included default algorithm and tests.
- A 0.1-second additional-search budget is used by default. Wall-clock limits,
  hardware, solver versions, and parallel execution can change search
  trajectories, completion values, and timings. A seed identifies the input
  stream; it does not guarantee identical measured runtimes or decisions.

## Quick start

Run these commands from this directory in PowerShell. The virtual environment
is created beside the package so that generated files stay outside it.

```powershell
python --version
python -m venv ..\VCS-E2E_venv
..\VCS-E2E_venv\Scripts\python.exe -m pip install -r requirements.txt
..\VCS-E2E_venv\Scripts\python.exe -B scripts\verify_package.py
..\VCS-E2E_venv\Scripts\python.exe -B scripts\run_rosr.py --seed 20460400 --profile tighter_deadlines
```

The verification command checks the retained Python source hashes, the frozen
model and policy identity, and runs 79 mechanism tests. The example episode
prints JSON; a successful comparison reports `stream_hashes_equal: true` and
`audited: true`. Values and timing need not match a previous run exactly.

On success, all 79 tests pass; in the single-episode output the stream hashes of both methods must agree and the audit result must be true.

To save the RCR task, event, decision, and summary files, choose a directory
outside the package, for example:

```powershell
..\VCS-E2E_venv\Scripts\python.exe -B scripts\run_rosr.py --seed 20460400 --profile tighter_deadlines --output ..\VCS-E2E_runs\smoke_20460400
```

A small paired batch can be run with:

```powershell
..\VCS-E2E_venv\Scripts\python.exe -B scripts\run_rosr_bench.py --seeds 20460400 --profiles tighter_deadlines --methods e2e_resource_density,rosr --workers 1 --label quick_check
```

Batch outputs default to a sibling `VCS-E2E_runs/` directory. Cached episodes
are checked against the relevant algorithm, control, and policy fingerprints;
use a new output directory or label when changing run settings to avoid mixing
results.

## Supplementary data and verification

All paths below are relative to the integrated package root. Install the same
`requirements.txt` as in the quick start.

```powershell
python -B scripts\verify_package.py
python -B supplement\code\verify_package.py
python -B supplement\code\analysis\verify_hefei_snapshot.py
python -B supplement\code\analysis\reanalyse_paired_results.py
python -B supplement\code\analysis\verify_review_fixes.py
python -B supplement\code\analysis\test_protected_nosearch_stats.py
python -B supplement\code\analysis\verify_protected_nosearch.py
python -B supplement\code\analysis\run_work_budget_repro.py --seeds 20460400 --profiles coupled_primary
python -B supplement\code\analysis\run_alns_control.py --variant alnsv2 --seconds 0.1 --seeds 20423000 --profiles coupled_primary
python -B supplement\code\analysis\run_protected_nosearch.py
python -B supplement\code\analysis\run_matched_monolithic.py
```

The first command checks 79 tests in the minimal layout. The second checks 77
legacy-layout tests in the retained supplementary source; the suites overlap.
The Hefei command checks the retained source/data fingerprints and 450 recorded
episodes, not the original elapsed wall times. The reanalysis reads historical
CSV summaries rather than rerunning the experiments. Both the latter commands
write files under `supplement/code/analysis/`; run them on a copy if you need
to preserve the original package hashes. The remaining commands verify the
cross-event release-floor cut indicator and guarded acceptance contract,
deterministic solver work-budget repeatability, standalone ALNS/ALNSv2 controls,
the 30-stream Protected-NS ablation, and the strictly serial matched-snapshot
monolithic MILP comparison.

- `supplement/opt/osm_hefei/`: OSM road-network snapshot and case traces.
- `supplement/paper/temp/cor_experiments/`: supplemental runner and 570
  recorded episode summaries.
- `supplement/code/history/step16_code_only_backup/results/`: five historical
  per-stream CSV summaries used by the statistical analyses.
- `supplement/PROVENANCE.md`: source, checksum and reproducibility limits.
- `supplement/requirements-network-optional.txt`: optional mapping and plotting
  dependencies for rebuilding the road graph or its figure. The retained graph
  and trace replay do not require rebuilding it or querying a live map service.

The complete manuscript and its editable source files are submitted separately.
This archive can be supplied as one supplementary item; a data repository link
or an explanation of unavailable data may still be required at submission.

## Contents

- `code/rosr/`: RCR policy, scheduling, optimization, and checks.
- `code/runtime/`: simulator, routing, and physical/action auditing.
- `code/baselines/`: retained comparison policies.
- `code/bootstrap.py`, `code/frozen_metadata.py`: path setup and source identity.
- `scripts/`: verification, single-run, batch, model, and scale entry points.
- `tests/`: mechanism and package-boundary tests.
- `requirements.txt`: pinned runtime dependencies.
- `supplement/`: supplementary experiments, road data and historical summaries.

For options, run the desired entry point with `--help`. The source hashes are
integrity checks, not a signature or independent proof of experimental
provenance.

## License

The authors' source code, simulator scripts, and synthetic experiment summaries are released under the MIT License (see `LICENSE`). Retained OpenStreetMap Hefei road-network data under `supplement/opt/osm_hefei/data/` remain subject to the Open Database License (ODbL-1.0).

## Corrected Protected-NS statistics (v1.0.0)

Grid signed ranks and win/tie/loss counts quantize both arms to four decimal
places in percentage units, matching the retained historical RCR CSV precision.
Hefei ranking uses ten decimal percentage places. The conditional exact test
handles zeros and equal absolute ranks. Means and marginal paired Student
intervals retain the observation values. These ablations report raw p-values.

`supplement/code/analysis/protected_nosearch_observations.csv` contains 540 new
audited Density/Protected-NS episodes and 180 retained RCR comparison records.
The adjacent metadata JSON records source/input fingerprints and the precision
protocol. `verify_protected_nosearch.py` verifies these records and recomputes
all nine-condition comparisons without rerunning simulations. Four dedicated
statistical regression checks run separately from the overlapping 79/77 core
mechanism suites. Rerunning the full control experiment writes a new summary,
observations CSV and metadata JSON; use a copy to preserve released checksums.
