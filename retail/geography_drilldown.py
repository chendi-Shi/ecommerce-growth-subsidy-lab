"""Country, customer and basket accounting bridges for the overseas residual.

The population is identified positive-product orders outside the UK, excluding
a fixed January--October 2010 top-1% customer list. No causal effect is claimed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from retail.diagnose import _clean, price_mix_bridge, product_shapley


def country_basket_bridge(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Exact change in pooled units/order, weighted by orders, not buyers.

    Common-country conditional weights and their coverage are separated so that
    entering/exiting countries remain in the accounting identity.
    """
    required = ("country", "orders_0", "orders_1", "units_0", "units_1")
    if not set(required).issubset(frame.columns):
        raise ValueError("Country bridge needs country, order counts and unit totals")
    d = frame[list(required)].copy()
    if (d[["orders_0", "orders_1", "units_0", "units_1"]].fillna(0) < 0).any().any():
        raise ValueError("Orders and purchased units must be nonnegative")
    for t in (0, 1):
        if ((d[f"orders_{t}"].fillna(0) == 0) & (d[f"units_{t}"].fillna(0) != 0)).any():
            raise ValueError("Purchased units require a positive order count")
    mapped = d.rename(columns={"country": "key", "orders_0": "units_0", "orders_1": "units_1",
                               "units_0": "revenue_0", "units_1": "revenue_1"})
    result, common = price_mix_bridge(mapped)
    if result["status"] != "ok":
        return {"status": "undefined_no_orders_in_one_period"}, pd.DataFrame()
    names = {"within_common_key_paid_unit_value": "within_common_country_basket",
             "common_key_quantity_mix": "common_country_order_mix",
             "common_set_quantity_share": "common_country_order_coverage",
             "entering_key_revenue_per_total_unit": "entering_country_units_per_total_order",
             "exiting_key_revenue_per_total_unit": "exiting_country_units_per_total_order"}
    output = {"status": "ok", "unit": "purchased units per positive-product order",
              "units_per_order_0": result["paid_unit_value_0"], "units_per_order_1": result["paid_unit_value_1"],
              "delta": result["delta"], "orders_0": result["units_0"], "orders_1": result["units_1"],
              "common_countries": result["common_keys"], "entering_countries": result["entering_keys"],
              "exiting_countries": result["exiting_keys"],
              "common_country_order_share_0": result["common_quantity_share_0"],
              "common_country_order_share_1": result["common_quantity_share_1"],
              "common_basket_reference_units_per_order": result["common_basket_reference_paid_unit_value"],
              "components": {names[k]: v for k, v in result["components"].items()},
              "reconciliation_error": result["reconciliation_error"],
              "centered_mix_reconciliation_error": result["centered_mix_reconciliation_error"]}
    output["grouped_components"] = {"within_common_country_basket": output["components"]["within_common_country_basket"],
        "common_country_order_mix": output["components"]["common_country_order_mix"],
        "entry_exit_and_coverage_joint": sum(v for k, v in output["components"].items()
                                              if k not in ("within_common_country_basket", "common_country_order_mix"))}
    common = common.rename(columns={"key": "country", "units_0": "orders_0", "units_1": "orders_1",
        "revenue_0": "units_0", "revenue_1": "units_1", "paid_unit_value_0": "units_per_order_0",
        "paid_unit_value_1": "units_per_order_1", "common_unit_weight_0": "common_country_order_weight_0",
        "common_unit_weight_1": "common_country_order_weight_1",
        "within_key_price_contribution": "within_country_basket_contribution_units_per_order",
        "common_key_mix_contribution": "country_order_mix_contribution_units_per_order",
        "common_key_mix_contribution_centered": "centered_country_order_mix_contribution_units_per_order"})
    return output, common


def classify_customer_country(d: pd.DataFrame) -> pd.DataFrame:
    """A customer present in a country in both focal months is 'common' there."""
    f = d.copy()
    for c in ("orders_0", "orders_1", "units_0", "units_1", "revenue_0", "revenue_1"):
        f[c] = f[c].fillna(0)
    f["customer_group"] = np.select([
        (f.orders_0 > 0) & (f.orders_1 > 0),
        (f.orders_0 > 0) & (f.orders_1 == 0),
        (f.orders_0 == 0) & (f.orders_1 > 0) & (pd.to_datetime(f.first_day) >= pd.Timestamp("2011-11-01")),
    ], ["common_country_november_buyers", "base_only_country_november_buyers",
        "current_only_first_observed_in_november"], default="current_only_previously_observed")
    for c in ("units", "orders", "revenue"):
        f[f"{c}_delta"] = f[f"{c}_1"] - f[f"{c}_0"]
    return f


def run(db: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db), read_only=True, config={"memory_limit": "256MB", "threads": "2"})
    prior = con.execute("""SELECT customer_id,sum(order_revenue) prior_revenue FROM retail_orders
        WHERE customer_id IS NOT NULL AND order_date >= DATE '2010-01-01' AND order_date < DATE '2010-11-01'
        GROUP BY customer_id ORDER BY prior_revenue DESC,customer_id""").fetchdf()
    top = prior.iloc[:math.ceil(len(prior) * .01)]
    con.register("geo_excluded_customers", top[["customer_id"]])
    # Temporary views on a read-only connection do not alter the stored marts.
    con.execute("""CREATE TEMP VIEW geo_orders AS SELECT * REPLACE(coalesce(country,'[unknown]') AS country)
        FROM retail_orders WHERE customer_id IS NOT NULL AND country IS DISTINCT FROM 'United Kingdom'
        AND observed_month IN ('2010-11','2011-11') AND customer_id NOT IN (SELECT customer_id FROM geo_excluded_customers)""")
    country = con.execute("""SELECT country,
        count(*) FILTER (WHERE observed_month='2010-11') orders_0,
        count(*) FILTER (WHERE observed_month='2011-11') orders_1,
        count(DISTINCT customer_id) FILTER (WHERE observed_month='2010-11') buyers_0,
        count(DISTINCT customer_id) FILTER (WHERE observed_month='2011-11') buyers_1,
        coalesce(sum(units) FILTER (WHERE observed_month='2010-11'),0) units_0,
        coalesce(sum(units) FILTER (WHERE observed_month='2011-11'),0) units_1,
        coalesce(sum(order_revenue) FILTER (WHERE observed_month='2010-11'),0) revenue_0,
        coalesce(sum(order_revenue) FILTER (WHERE observed_month='2011-11'),0) revenue_1
        FROM geo_orders GROUP BY country""").fetchdf()
    for t in (0, 1):
        for col, num, den in (("aov", "revenue", "orders"), ("units_per_order", "units", "orders"),
                              ("orders_per_buyer", "orders", "buyers"), ("paid_unit_value", "revenue", "units")):
            country[f"{col}_{t}"] = country[f"{num}_{t}"] / country[f"{den}_{t}"].replace(0, np.nan)
        country[f"order_share_{t}"] = country[f"orders_{t}"] / country[f"orders_{t}"].sum()
    for m in ("revenue", "units", "orders", "buyers", "aov", "units_per_order", "paid_unit_value"):
        country[m+"_delta"] = country[m+"_1"] - country[m+"_0"]
        country[m+"_yoy"] = country[m+"_1"] / country[m+"_0"].replace(0, np.nan) - 1
    country["presence"] = np.select([country.orders_0.eq(0), country.orders_1.eq(0)],
                                    ["current_month_only", "base_month_only"], default="common")
    country.sort_values("revenue_delta").to_csv(output/"country_bridge.csv", index=False)
    basket, contributions = country_basket_bridge(country)
    contributions.sort_values("within_country_basket_contribution_units_per_order").to_csv(output/"country_basket_components.csv", index=False)
    totals = con.execute("""SELECT observed_month,count(*) orders,count(DISTINCT customer_id) buyers,
        sum(units) units,sum(order_revenue) revenue FROM geo_orders GROUP BY observed_month ORDER BY observed_month""").fetchdf()
    for col, num, den in (("aov", "revenue", "orders"), ("units_per_order", "units", "orders"),
                         ("orders_per_buyer", "orders", "buyers"), ("paid_unit_value", "revenue", "units")):
        totals[col] = totals[num] / totals[den]
    totals.to_csv(output/"population_metrics.csv", index=False)
    leave_country_out = country[["country"]].copy()
    for t in (0, 1):
        leave_country_out[f"revenue_{t}"] = totals.iloc[t].revenue - country[f"revenue_{t}"]
        leave_country_out[f"units_{t}"] = totals.iloc[t].units - country[f"units_{t}"]
        leave_country_out[f"orders_{t}"] = totals.iloc[t].orders - country[f"orders_{t}"]
    leave_country_out["revenue_delta"] = leave_country_out.revenue_1-leave_country_out.revenue_0
    leave_country_out["revenue_yoy"] = leave_country_out.revenue_1/leave_country_out.revenue_0-1
    leave_country_out.to_csv(output/"leave_each_country_out.csv", index=False)
    customer = con.execute("""SELECT country,customer_id,h.first_day,
        count(*) FILTER(WHERE observed_month='2010-11') orders_0,
        count(*) FILTER(WHERE observed_month='2011-11') orders_1,
        coalesce(sum(units) FILTER(WHERE observed_month='2010-11'),0) units_0,
        coalesce(sum(units) FILTER(WHERE observed_month='2011-11'),0) units_1,
        coalesce(sum(order_revenue) FILTER(WHERE observed_month='2010-11'),0) revenue_0,
        coalesce(sum(order_revenue) FILTER(WHERE observed_month='2011-11'),0) revenue_1
        FROM geo_orders JOIN customer_history h USING(customer_id) GROUP BY country,customer_id,h.first_day""").fetchdf()
    customers = classify_customer_country(customer)
    aggregates = {m: (m, "sum") for m in ("orders_0", "orders_1", "units_0", "units_1", "revenue_0", "revenue_1",
                                           "orders_delta", "units_delta", "revenue_delta")}
    customer_groups = customers.groupby(["country", "customer_group"], as_index=False).agg(customers=("customer_id", "size"), **aggregates)
    for t in (0, 1):
        for col, numerator, denominator in (("units_per_order", "units", "orders"), ("aov", "revenue", "orders"),
                                            ("paid_unit_value", "revenue", "units")):
            customer_groups[f"{col}_{t}"] = customer_groups[f"{numerator}_{t}"]/customer_groups[f"{denominator}_{t}"].replace(0, np.nan)
    customer_groups.to_csv(output/"country_customer_bridge.csv", index=False)
    # Never export customer identifiers. Common-customer groups support a follow-up
    # worklist inside a controlled environment; observed changes are not recoverable ROI.
    common = customers.query("customer_group == 'common_country_november_buyers'").copy()
    common["spend_direction"] = np.select([common.revenue_delta.lt(0), common.revenue_delta.gt(0)], ["decreased", "increased"], default="unchanged")
    common.groupby(["country", "spend_direction"], as_index=False).agg(customers=("customer_id", "size"), **aggregates).to_csv(output/"common_customer_spend_directions.csv", index=False)
    con.register("geo_common_customer_countries", common[["country", "customer_id"]])
    bands = con.execute("""WITH labelled AS (SELECT *,CASE WHEN units<100 THEN '1_<100_units'
        WHEN units<500 THEN '2_100_499_units' WHEN units<1000 THEN '3_500_999_units' ELSE '4_1000plus_units' END unit_band
        FROM geo_orders) SELECT country,observed_month,unit_band,count(*) orders,sum(units) units,sum(order_revenue) revenue
        FROM labelled GROUP BY ALL ORDER BY country,observed_month,unit_band""").fetchdf()
    bands.to_csv(output/"country_order_size_bands.csv", index=False)
    quantiles = con.execute("""SELECT country,observed_month,count(*) orders,quantile_cont(units,.5) units_p50,
        quantile_cont(units,.9) units_p90,quantile_cont(units,.95) units_p95,quantile_cont(units,.99) units_p99,
        max(units) max_order_units FROM geo_orders GROUP BY ALL ORDER BY country,observed_month""").fetchdf()
    quantiles.to_csv(output/"country_order_size_quantiles.csv", index=False)
    sku = con.execute("""WITH lines AS (SELECT o.country,o.observed_month,l.stock_code,l.description,l.quantity,l.line_revenue,
        c.customer_id IS NOT NULL is_common_country_customer FROM geo_orders o
        JOIN retail_lines l ON o.customer_id=l.customer_id AND o.invoice=l.invoice
        LEFT JOIN geo_common_customer_countries c ON o.customer_id=c.customer_id AND o.country=c.country
        WHERE l.is_product AND NOT l.is_return), scopes AS (
        SELECT *, 'all_scope_buyers' buyer_scope FROM lines UNION ALL
        SELECT *, 'common_country_november_buyers' buyer_scope FROM lines WHERE is_common_country_customer)
        SELECT country,buyer_scope,stock_code,
        arg_max(description,struct_pack(period:=observed_month,label:=description)) description,
        coalesce(sum(quantity) FILTER(WHERE observed_month='2010-11'),0) units_0,
        coalesce(sum(quantity) FILTER(WHERE observed_month='2011-11'),0) units_1,
        coalesce(sum(line_revenue) FILTER(WHERE observed_month='2010-11'),0) revenue_0,
        coalesce(sum(line_revenue) FILTER(WHERE observed_month='2011-11'),0) revenue_1
        FROM scopes GROUP BY country,buyer_scope,stock_code""").fetchdf()
    for m in ("units", "revenue"):
        sku[m+"_delta"] = sku[m+"_1"]-sku[m+"_0"]
    sku.sort_values(["country", "buyer_scope", "revenue_delta"]).to_csv(output/"country_sku_bridge.csv", index=False)
    country_assignment_audit = con.execute("""SELECT count(*) orders,
        count(*) FILTER(WHERE source_countries > 1) orders_with_multiple_source_countries,
        sum(lines_not_matching_order_country) lines_not_matching_order_country
        FROM (SELECT o.customer_id,o.invoice,count(DISTINCT coalesce(l.country,'[unknown]')) source_countries,
        count(*) FILTER(WHERE coalesce(l.country,'[unknown]') IS DISTINCT FROM o.country) lines_not_matching_order_country
        FROM geo_orders o JOIN retail_lines l ON o.customer_id=l.customer_id AND o.invoice=l.invoice
        WHERE l.is_product AND NOT l.is_return GROUP BY o.customer_id,o.invoice)""").fetchdf().iloc[0].to_dict()
    cross_country_buyers = con.execute("""SELECT observed_month,count(*) buyers_seen_in_multiple_countries
        FROM (SELECT observed_month,customer_id FROM geo_orders GROUP BY observed_month,customer_id
        HAVING count(DISTINCT country)>1) GROUP BY observed_month""").fetchdf().to_dict("records")
    def country_error(frame, groupby, expected, cols):
        observed = frame.groupby(groupby)[cols].sum()
        return max(float((observed[c]-expected.set_index(groupby)[c]).abs().max()) for c in cols)
    cols = [f"{m}_{t}" for m in ("orders", "units", "revenue") for t in (0, 1)]
    checks = {"country_basket_error_units_per_order": basket["reconciliation_error"],
        "centered_mix_error_units_per_order": basket["centered_mix_reconciliation_error"],
        "country_customer_bridge_max_error": country_error(customer_groups, "country", country, cols),
        "country_sku_bridge_max_error": country_error(sku.query("buyer_scope=='all_scope_buyers'"), "country", country,
                                                       [f"{m}_{t}" for m in ("units", "revenue") for t in (0, 1)]),
        "country_vs_population_max_error": max(abs(float(country[f"{m}_{t}"].sum())-float(totals.iloc[t][m]))
                                                 for m in ("orders", "units", "revenue") for t in (0, 1)),
        "band_vs_population_max_error": max(abs(float(bands.query("observed_month == @period")[m].sum())-float(totals.iloc[t][m]))
            for t, period in enumerate(("2010-11", "2011-11")) for m in ("orders", "units", "revenue")),
        "fixed_top_customer_count": len(top)}
    checks["country_buyer_reconstruction_max_error"] = max(abs(int(customers.loc[(customers.country == row.country) &
        (customers[f"orders_{t}"] > 0)].shape[0]) - int(getattr(row, f"buyers_{t}")))
        for row in country.itertuples() for t in (0, 1))
    aov = product_shapley(totals.iloc[0].units_per_order, totals.iloc[0].paid_unit_value,
                          totals.iloc[1].units_per_order, totals.iloc[1].paid_unit_value)
    checks["population_aov_bridge_error_gbp"] = aov["reconciliation_error"]
    checks["passed"] = all(abs(v) < 1e-6 for k, v in checks.items() if k.endswith("error") or "error_" in k)
    if not checks["passed"]:
        raise AssertionError(checks)
    summary = {"population": "Identified non-UK positive-product orders, November 2010 versus November 2011, excluding the fixed January--October 2010 top ceil(1%) customers",
        "currency": "GBP", "prior_eligible_customers": len(prior), "fixed_excluded_customers": len(top),
        "excluded_minimum_prior_revenue_gbp": float(top.prior_revenue.min()),
        "country_assignment_audit": country_assignment_audit,
        "multi_country_buyer_counts_by_month": cross_country_buyers,
        "metrics": totals.to_dict("records"), "basket_bridge": basket, "aov_bridge": aov,
        "leave_australia_out": leave_country_out.loc[leave_country_out.country.eq("Australia")].to_dict("records"),
        "largest_country_revenue_declines": country.sort_values("revenue_delta").head(5).to_dict("records"),
        "largest_country_unit_declines": country.sort_values("units_delta").head(5).to_dict("records"),
        "customer_country_warning": "Country-level distinct customer counts are not additive across countries. Country-month absence is not churn; first observed is not acquisition.",
        "limitations": ["Descriptive accounting, not an identified effect of freight, promotions, prices or inventory.",
            "Positive purchases exclude credit notes; source has no assignment, margin, shipping charge, inventory or exposure log.",
            "Country and customer groups are exploratory after the original November anomaly; validate hypotheses on new data.",
            "Order-size bands are fixed descriptive thresholds, not measured wholesale status or optimized intervention cutoffs."]}
    for name, value in (("summary.json", summary), ("checks.json", checks)):
        (output/name).write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    con.close()
    print(json.dumps(_clean({"metrics": summary["metrics"], "basket": basket, "country_declines": country[["country", "revenue_delta", "units_delta", "orders_0", "orders_1"]].sort_values("revenue_delta").head(5).to_dict("records"), "checks": checks}), ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/retail.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("reports/geography_drilldown"))
    args = parser.parse_args()
    run(args.db, args.output)
