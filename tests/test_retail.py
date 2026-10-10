"""Business-grain and censoring regression tests, not implementation mirror tests."""
import unittest
import tempfile
from datetime import datetime
from pathlib import Path

from retail.pipeline import RAW_SCHEMA, audit, build_marts, canonical_row, connect, load_rows, verify


class RetailTests(unittest.TestCase):
    def setUp(self):
        self.db = connect()
        self.db.execute(f"CREATE TABLE raw_lines ({RAW_SCHEMA})")
        records = [
            ("100001", "12345", "item", 2, "2009-12-01 10:00:00", 5, "A", "UK", "sheet1", 2),
            # The exact same business row appears in another sheet.
            ("100001", "12345", "item", 2, "2009-12-01 10:00:00", 5, "A", "UK", "sheet2", 2),
            # Anonymous sales remain in the money ledger, not buyer metrics.
            ("100002", "12345", "item", 3, "2009-12-01 11:00:00", 5, None, "UK", "sheet1", 3),
            ("C100003", "12345", "item", -1, "2009-12-02 10:00:00", 5, "A", "UK", "sheet1", 4),
            # Non-product charge code does not enter product revenue or orders.
            ("100001", "POST", "Postage", 1, "2009-12-01 10:00:00", 4, "A", "UK", "sheet1", 5),
            ("100004", "12345", "item", 1, "2009-12-03 10:00:00", 0, "A", "UK", "sheet1", 6),
            ("100005", "12345", "item", 0, "2009-12-03 10:00:00", 5, "A", "UK", "sheet1", 7),
            ("200001", "12345", "item", 1, "2010-03-01 10:00:00", 10, "B", "UK", "sheet1", 8),
            ("200002", "12345", "item", 1, "2010-03-01 12:00:00", 10, "B", "UK", "sheet1", 9),
            ("200003", "12345", "item", 1, "2010-03-31 10:00:00", 10, "B", "UK", "sheet1", 10),
            ("200004", "12345", "item", 1, "2010-04-01 10:00:00", 10, "B", "UK", "sheet1", 11),
            ("200005", "12345", "item", 1, "2010-06-01 10:00:00", 10, "B", "UK", "sheet1", 12),
            ("300001", "12345", "item", 2, "2011-08-30 10:00:00", 10, "E", "UK", "sheet1", 13),
            ("300002", "12345", "item", 10, "2011-09-05 10:00:00", 10, "E", "UK", "sheet1", 14),
            ("400001", "12345", "item", 1, "2011-11-25 10:00:00", 5, "C", "UK", "sheet1", 15),
            ("400002", "12345", "item", 1, "2011-12-09 10:00:00", 5, "C", "UK", "sheet2", 16),
            # Source cancellation with positive qty is signed conservatively as a return.
            ("C400003", "12345", "item", 1, "2011-12-09 11:00:00", 5, "C", "UK", "sheet2", 17),
        ]
        self.db.executemany("INSERT INTO raw_lines VALUES (?,?,?,?,?,?,?,?,?,?)", records)
        build_marts(self.db)

    def tearDown(self):
        self.db.close()

    def test_dedup_anonymous_returns_and_charge_reconcile(self):
        row = self.db.execute("""SELECT active_buyers, identified_purchase_revenue,
           anonymous_purchase_revenue, gross_product_sales, product_return_value,
           net_product_revenue FROM monthly_growth WHERE month='2009-12'""").fetchone()
        self.assertEqual(row, (1, 10.0, 15.0, 25.0, 5.0, 20.0))
        result = audit(self.db)
        self.assertEqual(result["suspected_duplicate_rows"], 1)
        self.assertEqual(result["source_missing_customer_rows"], 1)
        self.assertEqual(result["zero_price_rows"], 1)
        self.assertEqual(result["zero_quantity_rows"], 1)
        self.assertEqual(result["cancellation_positive_qty_raw_signed_amount_gbp"], 5.0)
        self.assertEqual(result["cancellation_positive_qty_sign_rule_delta_gbp"], -10.0)
        self.assertEqual(result["cancellation_positive_qty_product_ledger_delta_gbp"], -10.0)
        verify(self.db)

    def test_buyer_state_uses_prior_month_and_not_prior_order(self):
        rows = self.db.execute("""SELECT strftime(month,'%Y-%m'), buyer_state, orders
          FROM customer_month WHERE customer_id='B' ORDER BY month""").fetchall()
        self.assertEqual(rows, [("2010-03", "first_observed", 3),
                                ("2010-04", "retained", 1),
                                ("2010-06", "reactivated", 1)])
        no_returns = self.db.execute("SELECT product_return_value, anonymous_purchase_revenue "
                                    "FROM monthly_growth WHERE month='2010-03'").fetchone()
        self.assertEqual(no_returns, (0.0, 0.0))

    def test_same_day_split_order_excluded_and_horizon_boundary_included(self):
        row = self.db.execute("""SELECT days_to_repurchase, mature, repurchased, after_warmup
          FROM customer_repurchase WHERE customer_id='B' AND horizon_days=30""").fetchone()
        self.assertEqual(row, (30, True, True, True))
        # The same-day split still counts as a purchase invoice in frequency.
        self.assertEqual(self.db.execute("SELECT observed_orders FROM customer_history "
                                        "WHERE customer_id='B'").fetchone()[0], 5)

    def test_right_censoring_and_initial_cohort_warmup(self):
        row = self.db.execute("""SELECT mature, repurchased FROM customer_repurchase
          WHERE customer_id='C' AND horizon_days=30""").fetchone()
        self.assertEqual(row, (False, True))
        self.assertEqual(self.db.execute("SELECT count(*) FROM repurchase_by_cohort "
                                        "WHERE cohort_month='2011-11'").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT count(*) FROM monthly_cohort_retention "
                                        "WHERE cohort_month='2009-12'").fetchone()[0], 0)
        self.assertFalse(self.db.execute("SELECT complete_month FROM monthly_growth "
                                         "WHERE month='2011-12'").fetchone()[0])

    def test_retention_includes_zero_activity_and_reactivation(self):
        rows = self.db.execute("""SELECT month_index, retention_rate FROM monthly_cohort_retention
          WHERE cohort_month='2010-03' AND month_index IN (0,1,2,3) ORDER BY month_index""").fetchall()
        self.assertEqual(rows, [(0, 1.0), (1, 1.0), (2, 0.0), (3, 1.0)])
        common = self.db.execute("SELECT eligible_customers FROM repurchase_common_followup").fetchall()
        self.assertEqual(len(set(common)), 1)

    def test_rfm_no_future_revenue_in_segment_features(self):
        row = self.db.execute("SELECT monetary, frequency, future_revenue, future_orders "
                              "FROM rfm_holdout WHERE customer_id='E'").fetchone()
        self.assertEqual(row, (20.0, 1, 100.0, 1))

    def test_identifier_parser_does_not_create_float_customer_ids(self):
        record = canonical_row((123456.0, 12345.0, "item", 1,
                                datetime(2011, 1, 1), 5, 12345.0, "UK"), "source", 2)
        self.assertEqual(record[0], "123456")
        self.assertEqual(record[1], "12345")
        self.assertEqual(record[6], "12345")

    def test_csv_transport_preserves_inches_commas_newlines_and_missing_ids(self):
        description = 'RECORD FRAME 7" SINGLE, SIZE\nMINT'
        row = ("500001", "12345", description, 2, "2011-01-01 10:00:00",
               5, None, "UK", "sheet", 123)
        with tempfile.TemporaryDirectory() as temporary:
            count = load_rows(self.db, iter([row]), Path(temporary) / "batch.csv", batch_size=1)
        result = self.db.execute("SELECT description, customer_id FROM raw_lines "
                                 "WHERE invoice='500001'").fetchone()
        self.assertEqual(count, 1)
        self.assertEqual(result, (description, None))


if __name__ == "__main__":
    unittest.main()
