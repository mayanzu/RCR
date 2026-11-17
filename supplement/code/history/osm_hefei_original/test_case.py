"""Check OSM adapter invariants and sensitivity of the independent audit."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
sys.path.insert(0, str(HERE))
from rosr import model
import audit_osm
import case_contract as case


class NetworkContractTest(unittest.TestCase):
    def test_independent_all_pairs_and_directed_triangle(self):
        np.testing.assert_allclose(case.PLANNER, case.AUDITOR, atol=1e-5, rtol=0)
        self.assertGreater(float(np.max(np.abs(case.PLANNER - case.PLANNER.T))), 1.0)
        for k in range(100):
            self.assertLessEqual(float(np.max(case.PLANNER -
                                  (case.PLANNER[:, k, None] + case.PLANNER[k, None, :]))), 1e-5)

    def test_units_indices_and_input_bounds(self):
        self.assertEqual(case.index(9, 9), 99)
        self.assertEqual(case.planner_travel_time(3, 4, 3, 4, 2.0), 0.0)
        expected = case.PLANNER[case.index(0, 0), case.index(9, 9)] / (300.0 * 2.0)
        self.assertAlmostEqual(case.planner_travel_time(0, 0, 9, 9, 2.0), expected)
        with self.assertRaises(ValueError):
            case.planner_travel_time(10, 0, 1, 1, 2.0)
        with self.assertRaises(ValueError):
            case.planner_travel_time(0, 0, 1, 1, 0.0)

    def test_audit_detects_tampered_road_travel(self):
        ep = HERE / "results" / "pilot" / "episodes" / "coupled_primary" / "20480900" / "rosr"
        if not (ep / "summary.json").exists():
            self.skipTest("Pilot episode has not yet completed")
        with gzip.open(ep / "tasks.json.gz", "rt", encoding="utf-8") as f:
            rows = json.load(f)
        with gzip.open(ep / "events.json.gz", "rt", encoding="utf-8") as f:
            events = json.load(f)
        profile = next(p for p in model.PROFILES if p.name == "coupled_primary")
        tasks, vehicles = model.generate(20480900, profile)
        case.install(model)
        audit_osm.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, events)
        tampered = copy.deepcopy(events)
        start = next(e for e in tampered if e["kind"] == "capture_start")
        start["travel"] += 0.25
        with self.assertRaisesRegex(AssertionError, "actual travel time"):
            audit_osm.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, tampered)


if __name__ == "__main__":
    unittest.main()
