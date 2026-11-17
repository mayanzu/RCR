"""Import boundaries and immutable-package output policy."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from bootstrap import PACKAGE_ROOT,RUNS_ROOT,output_directory
from baselines.lit_dpso import DpsoPolicy
from baselines.lit_lyapunov import LyapunovPolicy
from baselines.value_coalition import ValueCoalitionPolicy


class ArchitectureTests(unittest.TestCase):
    def test_run_outputs_stay_outside_code_package(self):
        self.assertFalse(output_directory(label='smoke').is_relative_to(PACKAGE_ROOT))
        self.assertEqual(output_directory(label='smoke'),(RUNS_ROOT/'smoke').resolve())
        for target in [PACKAGE_ROOT,PACKAGE_ROOT/'code/generated',PACKAGE_ROOT/'scripts/../tests/output']:
            with self.assertRaises(ValueError):output_directory(target)

    def test_external_controls_have_one_shared_namespace(self):
        self.assertEqual(DpsoPolicy.__module__,'baselines.lit_dpso')
        self.assertEqual(LyapunovPolicy.__module__,'baselines.lit_lyapunov')
        self.assertEqual(ValueCoalitionPolicy.__module__,'baselines.value_coalition')
