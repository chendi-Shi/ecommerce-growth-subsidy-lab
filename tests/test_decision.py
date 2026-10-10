import unittest
import numpy as np
from scipy import stats
from campaign.decision import planning_n, hurdle_status, content_audience_components


class DecisionTests(unittest.TestCase):
    def test_audience_and_content_are_paired_and_reconcile(self):
        parts = content_audience_components([4, 2, -1], [3, 5, 0], [1, 1, 1])
        np.testing.assert_array_equal(parts['audience_selection'], [2, 4, -1])
        np.testing.assert_array_equal(parts['content_selection_given_audience'], [1, -3, -1])
        np.testing.assert_array_equal(parts['combined_targeting'], [3, 1, -2])

    def test_planning_uses_distance_from_business_hurdle(self):
        expected = int(np.ceil((stats.norm.ppf(.975)+stats.norm.ppf(.8))**2 * 4 / .1**2))
        self.assertEqual(planning_n(4, .15, .05), expected)
        self.assertAlmostEqual(planning_n(4, .1)/planning_n(4, .2), 4, delta=.01)
        with self.assertRaises(ValueError):
            planning_n(4, .05, .05)

    def test_nonsignificance_is_not_no_business_advantage(self):
        self.assertEqual(hurdle_status(-.279, .267, .05), 'unresolved_at_this_hurdle')
        self.assertEqual(hurdle_status(-.279, .267, .30), 'upper_bound_below_hurdle')
        self.assertEqual(hurdle_status(.08, .12, .05), 'lower_bound_above_hurdle')


if __name__ == '__main__':
    unittest.main()
