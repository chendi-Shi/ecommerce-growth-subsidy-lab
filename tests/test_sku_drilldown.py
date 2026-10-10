import unittest

import pandas as pd

from retail.sku_drilldown import (
    customer_membership_bridge, reconcile, summarize_price_bands, summarize_skus,
)


def purchases(rows):
    return pd.DataFrame(rows, columns=["stock_code", "observed_month", "customer_id", "invoice",
                                       "quantity", "unit_price", "line_revenue"])


class SkuDrilldownTests(unittest.TestCase):
    def test_concentration_uses_customer_invoice_not_line_or_invoice_string(self):
        data = purchases([
            ("x", "2010-11", "a", "same", 2, 3, 6),
            ("x", "2010-11", "a", "same", 3, 2, 6),
            ("x", "2010-11", "b", "same", 1, 3, 3),
            ("x", "2010-11", "a", "other", 4, 3, 12),
        ])
        row = summarize_skus(data).iloc[0]
        self.assertEqual(row.invoices, 3)
        self.assertEqual(row.buyers, 2)
        self.assertAlmostEqual(row.paid_value_per_unit, 2.7)
        self.assertAlmostEqual(row.maximum_invoice_unit_share, .5)
        self.assertAlmostEqual(row.maximum_buyer_unit_share, .9)

    def test_price_band_counts_can_overlap_but_money_and_units_reconcile(self):
        data = purchases([
            ("x", "2010-11", "a", "one", 2, 1, 2),
            ("x", "2010-11", "a", "one", 1, 2, 2),
            ("x", "2010-11", "b", "two", 3, 1, 3),
        ])
        bands = summarize_price_bands(data).set_index("unit_price")
        self.assertEqual(bands.buyers.sum(), 3)
        self.assertEqual(summarize_skus(data).iloc[0].buyers, 2)
        self.assertEqual(bands.invoices.sum(), 3)
        self.assertAlmostEqual(bands.loc[1, "unit_share"], 5/6)
        self.assertAlmostEqual(bands.loc[1, "revenue_share"], 5/7)
        self.assertAlmostEqual(bands.buyer_reach.sum(), 1.5)

    def test_sku_buyer_membership_preserves_all_three_sets_and_exact_bridge(self):
        data = purchases([
            ("x", "2010-11", "a", "a0", 2, 5, 10),
            ("x", "2010-11", "b", "b0", 4, 5, 20),
            ("x", "2011-11", "a", "a1", 3, 5, 15),
            ("x", "2011-11", "c", "c1", 6, 5, 30),
        ])
        bridge = customer_membership_bridge(data)
        by_group = bridge.set_index("membership")
        self.assertEqual(by_group.loc["common_sku_buyers", "revenue_delta"], 5)
        self.assertEqual(by_group.loc["base_only_sku_buyers", "revenue_delta"], -20)
        self.assertEqual(by_group.loc["current_only_sku_buyers", "revenue_delta"], 30)
        self.assertEqual(bridge.revenue_delta.sum(), 15)
        checks = reconcile(summarize_skus(data), summarize_price_bands(data), bridge)
        self.assertTrue(checks["passed"])


if __name__ == "__main__":
    unittest.main()
