import unittest

import pandas as pd

from retail.geography_drilldown import classify_customer_country, country_basket_bridge


class GeographyDrilldownTests(unittest.TestCase):
    def test_order_weighting_identifies_composition_when_country_baskets_unchanged(self):
        # The same two country baskets, but orders move to the smaller basket.
        # Buyers must not substitute for orders in the weights.
        frame = pd.DataFrame({"country": ["A", "B"], "orders_0": [1, 9], "orders_1": [9, 1],
                              "units_0": [10, 900], "units_1": [90, 100],
                              "buyers_0": [1, 1], "buyers_1": [1, 1]})
        result, _ = country_basket_bridge(frame)
        self.assertAlmostEqual(result["delta"], -72)
        self.assertAlmostEqual(result["components"]["within_common_country_basket"], 0)
        self.assertAlmostEqual(result["components"]["common_country_order_mix"], -72)
        self.assertAlmostEqual(result["reconciliation_error"], 0)

    def test_entry_exit_preserved_and_no_common_country_is_valid(self):
        frame = pd.DataFrame({"country": ["A", "B"], "orders_0": [2, 0], "orders_1": [0, 4],
                              "units_0": [100, 0], "units_1": [0, 80]})
        result, _ = country_basket_bridge(frame)
        self.assertEqual(result["common_countries"], 0)
        self.assertEqual(result["entering_countries"], 1)
        self.assertEqual(result["exiting_countries"], 1)
        self.assertAlmostEqual(result["components"]["entering_country_units_per_total_order"], 20)
        self.assertAlmostEqual(result["components"]["exiting_country_units_per_total_order"], -50)
        self.assertAlmostEqual(result["reconciliation_error"], 0)

    def test_inconsistent_country_denominator_rejected(self):
        with self.assertRaises(ValueError):
            country_basket_bridge(pd.DataFrame({"country": ["A"], "orders_0": [0], "orders_1": [1],
                                                "units_0": [20], "units_1": [20]}))

    def test_current_country_observation_does_not_imply_acquisition(self):
        frame = pd.DataFrame({"orders_0": [0, 0, 1, 1], "orders_1": [1, 1, 1, 0],
            "units_0": [0, 0, 10, 10], "units_1": [10, 10, 10, 0], "revenue_0": [0, 0, 20, 20],
            "revenue_1": [20, 20, 20, 0], "first_day": ["2010-06-01", "2011-11-03", "2010-06-01", "2010-06-01"]})
        actual = classify_customer_country(frame)
        self.assertEqual(actual.customer_group.tolist(), ["current_only_previously_observed",
            "current_only_first_observed_in_november", "common_country_november_buyers", "base_only_country_november_buyers"])
        self.assertEqual(actual.revenue_delta.sum(), 20)


if __name__ == "__main__":
    unittest.main()
