"""Regression checks for mixed-precision archived versus new paired results."""
import unittest
from run_protected_nosearch import paired_stats

class ProtectedStatisticsTests(unittest.TestCase):
    def test_archive_rounding_residue_is_a_tie(self):
        result = paired_stats([98.7342, 50., 60.], [100 * 234 / 237, 49., 58.])
        self.assertEqual(result['wtl'], '2/1/0')
        self.assertEqual(result['n_nonzero'], 2)
        self.assertEqual(result['p_wilcoxon'], .5)

    def test_absolute_rank_ties_use_conditional_distribution(self):
        # Differences 1,1,1,-3,4 give doubled ranks 4,4,4,8,10.
        # Exactly 8 of 32 signs have rank sum <= observed 4; two-sided p=.5.
        result = paired_stats([11.,11.,11.,7.,14.], [10.]*5)
        self.assertEqual(result['signed_rank_w'], 4.)
        self.assertEqual(result['p_wilcoxon'], .5)

    def test_all_ties(self):
        result = paired_stats([50.,60.,70.], [50.,60.,70.])
        self.assertEqual(result['wtl'], '0/3/0')
        self.assertEqual(result['p_wilcoxon'], 1.)

    def test_full_precision_road_effect_is_retained(self):
        result = paired_stats([50.00001,60.00001], [50.,60.], percent_decimals=10)
        self.assertEqual(result['wtl'], '2/0/0')
        self.assertEqual(result['n_nonzero'], 2)

if __name__ == '__main__':
    unittest.main()
