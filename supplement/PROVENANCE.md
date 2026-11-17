# Data provenance and limits

The `supplement/` subtree is derived from the previously prepared `RCR_Supplementary_Material.zip`, SHA-256 `29a88f192396fdbd3e15a343f0c156e596c5ca91c941b90de329fcc3284848b8`. Its frozen Hefei source snapshot, network data and confirmation episodes are retained. The five historical CSV summaries were added from `code/history/step16_code_only_backup/results/` after verifying their hashes against that archive's manifest. The 77-test supplementary core and the 79-test minimal layout share identical core algorithm/model/baseline file hashes; their test suites overlap.

The historical CSVs contain per-stream summaries, not the original per-event traces or recovered original experiment-source fingerprints. Reanalysis of these summaries and a new execution from fixed seeds are distinct checks. Wall-clock limits, hardware and solver behavior can change new decisions and timings. The 450 Hefei traces can be independently replay-audited with the included fingerprinted source snapshot; this does not recreate original wall timings. The 570 supplemental episode records are retained under `paper/temp/cor_experiments/results/`.

Map data © OpenStreetMap contributors, Open Database License (ODbL): https://www.openstreetmap.org/copyright. The road network is real; task arrivals, service characteristics and deadlines are synthetic. Authors' code and synthetic summaries are released under the MIT License (see root `LICENSE`).

## Post-review matched controls and benchmark runners (2026-11-16)

- `supplement/code/analysis/verify_review_fixes.py`: verifies the cross-event release-floor cut indicator and guarded acceptance/baseline behavior on targeted edge cases.
- `supplement/code/analysis/run_protected_nosearch.py` and `protected_nosearch_summary.json`: paired 30-stream Protected-NS control isolating LBBD packing from continuation/guards on grid and Hefei.
- `supplement/code/analysis/run_matched_monolithic.py` and `matched_monolithic_summary.json`: strictly serial, matched-snapshot 30-stream comparison of LBBD, prefix-only monolithic MILP, and Eq. (6)-strengthened monolithic MILP.
- `supplement/code/analysis/run_alns_control.py`: standalone audited runners for `ALNSPolicy` and adaptive `ALNSv2`.
- `supplement/code/analysis/run_work_budget_repro.py` and `supplement/code/opt/work_budget.py`: deterministic solver work-budget repeatability verifier (256 MIP nodes, 16 MIP calls, 10,000 LP iterations).

## Statistical correction (2026-11-17, v1.0.0)

Protected-NS grid ranking now uses common four-decimal percent precision for
both archived RCR and newly executed controls, removing false nonzero pairs
caused by mixed precision. The exact conditional signed-rank implementation
handles equal absolute ranks. All 540 Density/Protected-NS episodes were
rerun and physically/action audited; 180 retained RCR observations are explicitly
labelled by their original grid/Hefei provenance. The observations CSV and
metadata hashes support summary reanalysis. This correction does not recover
missing historical grid event logs, change the core policy, or reproduce
historical wall-clock timings.
