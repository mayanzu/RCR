"""Road-time adapter for a frozen VCS-E2E policy and its independent audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
META = json.loads((DATA / "network_summary.json").read_text(encoding="utf-8"))


def load_matrix(name: str, expected: str) -> np.ndarray:
    path = DATA / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Road-distance matrix hash mismatch: {name}")
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (100, 100) or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError(f"Invalid road-distance matrix: {name}")
    return matrix


# The first matrix is from SciPy csgraph; the second was independently built
# with NetworkX Dijkstra and is used only by the action-log auditor.
PLANNER = load_matrix("distance_m.npy", META["matrix_sha256"])
AUDITOR = load_matrix("audit_distance_m.npy", META["audit_matrix_sha256"])
UNIT_M = float(META["distance_unit_m"])


def index(x: int, y: int) -> int:
    ix, iy = int(x), int(y)
    if ix != x or iy != y or not (0 <= ix < 10 and 0 <= iy < 10):
        raise ValueError(f"Location outside the fixed 10-by-10 anchor set: {(x, y)}")
    return ix * 10 + iy


def planner_travel_time(x1, y1, x2, y2, speed):
    if speed <= 0:
        raise ValueError("Vehicle speed must be positive")
    return float(PLANNER[index(x1, y1), index(x2, y2)]) / (UNIT_M * speed)


def audit_travel_time(x1, y1, x2, y2, speed):
    if speed <= 0:
        raise ValueError("Vehicle speed must be positive")
    return float(AUDITOR[index(x1, y1), index(x2, y2)]) / (UNIT_M * speed)


def install(model):
    """Apply only in this experiment's worker; the frozen source is untouched."""
    model.legacy.travel_time = planner_travel_time
