# Supplementary analysis code (legacy layout)

This directory preserves the 77-test analysis layout from the previous supplementary archive. Install the pinned dependencies in the package-root `requirements.txt` and consult the package-root `README.md`. From the integrated package root, run `python -B supplement/code/verify_package.py` for the 77-test legacy check and `python -B supplement/code/analysis/verify_hefei_snapshot.py` for the retained Hefei trace replay.

The historic per-stream CSV summaries required by `analysis/reanalyse_paired_results.py` are included at `history/step16_code_only_backup/results/`. The reanalysis and Hefei replay write results in `analysis/`; run them on a copy when preserving the package's checksums. They do not establish recovery of the original historical per-event logs or bit-identical wall-clock behavior.
