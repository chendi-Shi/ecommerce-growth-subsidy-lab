import unittest

import numpy as np
import pandas as pd

from retail.customer_drilldown import activity_groups, concentration_bridge, multiplicative_shapley, paired_window_bridge


class CustomerDrilldownTests(unittest.TestCase):
    def test_four_factor_bridge_hand_calculation_and_reversal(self):
        # Two factors move, two are fixed. 2*3*2*1 -> 4*5*2*1:
        # buyers receives (4-2)*(3+5)/2*2=16; frequency receives 12.
        a = dict(buyers=2, frequency=3, units=2, value=1)
        b = dict(buyers=4, frequency=5, units=2, value=1)
        got = multiplicative_shapley(a, b)
        self.assertEqual(got['components'], dict(buyers=16, frequency=12, units=0, value=0))
        self.assertEqual(got['revenue_delta'], 28)
        self.assertEqual(got['reconciliation_error'], 0)
        self.assertEqual(multiplicative_shapley(b, a)['components'], {k: -v for k, v in got['components'].items()})

    def test_fixed_cohort_includes_neither_period_and_activity_is_window_specific(self):
        got = activity_groups(pd.Series([2, 1, 0, 0]), pd.Series([1, 0, 3, 0]))
        self.assertEqual(got.tolist(), ['both_novembers', 'base_november_only', 'current_november_only', 'neither_november'])
        with self.assertRaises(ValueError):
            activity_groups(pd.Series([-1]), pd.Series([0]))

    def test_concentration_preserves_gain_offset_and_does_not_sum_nested_groups(self):
        # Gross losses 100, gains 60, net decline 40. Largest loss is 150%
        # of net decline but only 60% of gross losses; both are correct.
        got = concentration_bridge(np.array([-60, -30, -10, 60, 0]))
        self.assertEqual((got['gross_loss'], got['gross_gain'], got['net_decline']), (100, 60, 40))
        one, three, five = got['cumulative_largest_losses']
        self.assertEqual((one['share_of_all_gross_losses'], one['share_of_net_decline']), (.6, 1.5))
        self.assertEqual((three['gross_loss'], five['gross_loss'], five['included_customers']), (100, 100, 3))
        self.assertEqual(got['reconciliation_error'], 0)
        rising = concentration_bridge(np.array([-10, 20]))
        self.assertIsNone(rising['cumulative_largest_losses'][0]['share_of_net_decline'])

    def test_same_customer_arithmetic_coverage_does_not_attribute_other_customer_gains(self):
        # A loses 100 in Nov, earns 30 more earlier; B loses 20 but earns 50
        # more earlier; C's large earlier gain cannot cover A's shortfall.
        got = paired_window_bridge(np.array([-100, -20, 10]), np.array([30, 50, 200]))
        self.assertEqual(got['same_customer_positive_sep_oct_increase_capped_at_november_loss'], 50)
        self.assertEqual(got['november_decliners_still_down_in_sep_nov'], 1)
        self.assertEqual(got['november_decliners_still_down_sep_nov_gross_loss'], 70)
        self.assertEqual(got['all_november_decliners_sep_nov_net_delta'], -40)
        self.assertEqual(got['other_customers_sep_nov_net_delta'], 210)
        self.assertEqual(got['all_customers_sep_nov_net_delta'], 170)
        self.assertEqual(sum(r['customers'] for r in got['groups']), 3)


if __name__ == '__main__':
    unittest.main()
