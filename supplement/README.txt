Supplementary experiment and data subtree
=========================================

This subtree is part of the integrated RCR reproducibility package. See the package-root README.md for installation, scope, license information, and commands. It contains the retained supplementary analysis code, five historical per-stream summary CSVs, the static OpenStreetMap Hefei network and 450 recorded case episodes, and the 570 additional experiment episode summaries.

From the integrated package root (use a copy if preserving checksums):
  python -B supplement/code/verify_package.py
  python -B supplement/code/analysis/verify_hefei_snapshot.py
  python -B supplement/code/analysis/reanalyse_paired_results.py

Do not interpret the historical CSV summaries or the unit tests as the original per-event logs or as proof of identical wall-clock runs.
