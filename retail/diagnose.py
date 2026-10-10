"""Auditable descriptive diagnosis of the November year-over-year signal.

All money is source GBP. These are accounting decompositions, not effects of
pricing, acquisition, subsidies, or retention interventions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


FOCAL = ("2010-11", "2011-11")


def product_shapley(x0: float, p0: float, x1: float, p1: float) -> dict:
    """Exact symmetric decomposition of x*p, including the interaction equally."""
    x_effect = (x1 - x0) * (p0 + p1) / 2
    p_effect = (p1 - p0) * (x0 + x1) / 2
    delta = x1 * p1 - x0 * p0
    return {"delta": delta, "quantity_component": x_effect,
            "value_component": p_effect, "reconciliation_error": delta-x_effect-p_effect}


def price_mix_bridge(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Decompose pooled paid unit value, preserving common/entering/exiting keys.

    Input columns: key, units_0, revenue_0, units_1, revenue_1. Missing or zero
    units represent absence. A key may be a SKU or a customer-SKU pair.
    """
    d = frame.copy()
    for col in ("units_0", "revenue_0", "units_1", "revenue_1"):
        d[col] = d[col].fillna(0.0).astype(float)
    if (d[["units_0", "units_1"]] < 0).any().any():
        raise ValueError("Only positive purchase units belong in this bridge")
    if d["key"].duplicated().any():
        raise ValueError("Aggregate to one row per key before decomposition")
    totals = [float(d[f"units_{t}"].sum()) for t in (0, 1)]
    if not all(q > 0 for q in totals):
        return {"status": "undefined_no_positive_units_in_one_period"}, d
    common = (d.units_0 > 0) & (d.units_1 > 0)
    c = d.loc[common].copy()
    cq = [float(c[f"units_{t}"].sum()) for t in (0, 1)]
    shares = [cq[t]/totals[t] for t in (0, 1)]
    paid = [float(d[f"revenue_{t}"].sum())/totals[t] for t in (0, 1)]
    cp = [float(c[f"revenue_{t}"].sum())/cq[t] if cq[t] else 0 for t in (0, 1)]
    for t in (0, 1):
        c[f"paid_unit_value_{t}"] = c[f"revenue_{t}"]/c[f"units_{t}"]
        c[f"common_unit_weight_{t}"] = c[f"units_{t}"]/cq[t] if cq[t] else 0
    average_share = sum(shares)/2
    c["within_key_price_contribution"] = average_share * (
        c.common_unit_weight_0+c.common_unit_weight_1)/2 * (
        c.paid_unit_value_1-c.paid_unit_value_0)
    c["common_key_mix_contribution"] = average_share * (
        c.paid_unit_value_0+c.paid_unit_value_1)/2 * (
        c.common_unit_weight_1-c.common_unit_weight_0)
    # Subtracting a constant does not change the sum because weights each sum
    # to one. The centred allocation makes SKU drill-downs interpretable as
    # shifts above/below the common-basket reference value.
    c["common_key_mix_contribution_centered"] = average_share * (
        (c.paid_unit_value_0+c.paid_unit_value_1)/2-sum(cp)/2) * (
        c.common_unit_weight_1-c.common_unit_weight_0)
    components = {
        "within_common_key_paid_unit_value": float(c.within_key_price_contribution.sum()),
        "common_key_quantity_mix": float(c.common_key_mix_contribution.sum()),
        "common_set_quantity_share": sum(cp)/2*(shares[1]-shares[0]),
        "entering_key_revenue_per_total_unit": float(d.loc[(d.units_0 == 0), "revenue_1"].sum())/totals[1],
        "exiting_key_revenue_per_total_unit": -float(d.loc[(d.units_1 == 0), "revenue_0"].sum())/totals[0],
    }
    delta = paid[1]-paid[0]
    result = {"status": "ok", "paid_unit_value_0": paid[0], "paid_unit_value_1": paid[1],
              "delta": delta, "units_0": totals[0], "units_1": totals[1],
              "common_keys": int(common.sum()),
              "entering_keys": int(((d.units_0 == 0) & (d.units_1 > 0)).sum()),
              "exiting_keys": int(((d.units_1 == 0) & (d.units_0 > 0)).sum()),
              "common_quantity_share_0": shares[0], "common_quantity_share_1": shares[1],
              "common_basket_reference_paid_unit_value": sum(cp)/2,
              "components": components, "reconciliation_error": delta-sum(components.values())}
    result["grouped_components"] = {
        "within_common_key_paid_unit_value": components["within_common_key_paid_unit_value"],
        "common_key_quantity_mix": components["common_key_quantity_mix"],
        "entry_exit_and_common_set_share_joint": sum(components[k] for k in (
            "common_set_quantity_share", "entering_key_revenue_per_total_unit", "exiting_key_revenue_per_total_unit")),
    }
    result["centered_mix_reconciliation_error"] = float(c.common_key_mix_contribution_centered.sum()-components["common_key_quantity_mix"])
    return result, c


def customer_bridge(d: pd.DataFrame) -> pd.DataFrame:
    """Exact revenue bridge; 'current-only' is not automatically acquisition."""
    f = d.copy()
    for col in ("orders_0", "orders_1", "revenue_0", "revenue_1"):
        f[col] = f[col].fillna(0)
    f["customer_group"] = np.select([
        (f.orders_0 > 0) & (f.orders_1 > 0),
        (f.orders_0 > 0) & (f.orders_1 == 0),
        (f.orders_0 == 0) & (f.orders_1 > 0) & (pd.to_datetime(f.first_day) >= pd.Timestamp("2011-11-01")),
    ], ["common_november_buyers", "base_only_november_buyers", "current_only_first_observed_in_november"],
        default="current_only_previously_observed")
    f["revenue_delta"] = f.revenue_1-f.revenue_0
    return f


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "identity_scope" in d:
        # An anonymous invoice does not establish a distinct human buyer.
        d.loc[d.identity_scope == "anonymous", "buyers"] = np.nan
    for col, numerator, denominator in (
        ("aov", "revenue", "orders"), ("units_per_order", "units", "orders"),
        ("paid_unit_value", "revenue", "units"), ("orders_per_identified_buyer", "orders", "buyers"),
    ):
        d[col] = d[numerator]/d[denominator].replace(0, np.nan)
    if "identity_scope" in d:
        d.loc[d.identity_scope != "identified", "orders_per_identified_buyer"] = np.nan
    return d


def _comparisons(metrics: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    base = metrics.loc[metrics.year == 2010].drop(columns="year")
    current = metrics.loc[metrics.year == 2011].drop(columns="year")
    merged = base.merge(current, on=index, suffixes=("_0", "_1"), validate="one_to_one")
    for col in ("buyers", "orders", "revenue", "units", "aov", "units_per_order", "paid_unit_value"):
        merged[col+"_delta"] = merged[col+"_1"]-merged[col+"_0"]
        merged[col+"_yoy"] = merged[col+"_1"]/merged[col+"_0"].replace(0, np.nan)-1
    return merged


def run(db: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db), read_only=True, config={"memory_limit": "256MB", "threads": "2"})
    # Aggregation stays in DuckDB; no full raw ledger is copied to Python.
    print("Aggregating calendar and identity scopes", flush=True)
    monthly = con.execute("""
        WITH labelled AS (
          SELECT *, year(order_date) AS "year", month(order_date) AS "month",
            CASE WHEN customer_id IS NULL THEN 'anonymous' ELSE 'identified' END identity_scope
          FROM retail_orders WHERE order_date >= DATE '2010-01-01' AND order_date < DATE '2011-12-01'
            AND month(order_date) <= 11
        ), expanded AS (
          SELECT * EXCLUDE(identity_scope), identity_scope FROM labelled
          UNION ALL SELECT * EXCLUDE(identity_scope), 'all' AS identity_scope FROM labelled
        ) SELECT year, month, identity_scope, count(DISTINCT customer_id) buyers,
          count(*) orders, sum(order_revenue) revenue, sum(units) units
        FROM expanded GROUP BY ALL ORDER BY year,month,identity_scope
    """).fetchdf()
    monthly = _metrics(monthly)
    monthly.to_csv(output/"monthly_metrics.csv", index=False)
    monthly_yoy = _comparisons(monthly, ["month", "identity_scope"])
    monthly_yoy.to_csv(output/"monthly_yoy.csv", index=False)
    # Buyers in pooled windows are DISTINCT over the window, never sums of months.
    pooled = con.execute("""
        WITH windows(window_name,start_month,end_month) AS (VALUES
            ('jan_nov',1,11),('q1',1,3),('q2',4,6),('q3',7,9),
            ('sep_nov',9,11),('oct_nov',10,11)), labelled AS (
          SELECT *, year(order_date) AS "year", CASE WHEN customer_id IS NULL
            THEN 'anonymous' ELSE 'identified' END identity_scope FROM retail_orders
          WHERE order_date >= DATE '2010-01-01' AND order_date < DATE '2011-12-01'
        ), expanded AS (
          SELECT * EXCLUDE(identity_scope), identity_scope FROM labelled
          UNION ALL SELECT * EXCLUDE(identity_scope), 'all' AS identity_scope FROM labelled
        ) SELECT w.window_name,e.year,e.identity_scope,count(DISTINCT customer_id) buyers,
          count(*) orders,sum(order_revenue) revenue,sum(units) units
        FROM expanded e JOIN windows w ON month(e.order_date) BETWEEN start_month AND end_month
        GROUP BY ALL ORDER BY window_name,year,identity_scope
    """).fetchdf()
    pooled = _metrics(pooled)
    pooled.to_csv(output/"pooled_metrics.csv", index=False)
    pooled_yoy = _comparisons(pooled, ["window_name", "identity_scope"])
    pooled_yoy.to_csv(output/"pooled_yoy.csv", index=False)
    focal = monthly_yoy.query("month == 11 and identity_scope == 'identified'").iloc[0]
    aov = product_shapley(focal.units_per_order_0, focal.paid_unit_value_0,
                          focal.units_per_order_1, focal.paid_unit_value_1)

    print("Decomposing customer and SKU composition", flush=True)
    customer = con.execute("""
        SELECT r.customer_id,h.first_day,
          count(*) FILTER(WHERE observed_month='2010-11') orders_0,
          count(*) FILTER(WHERE observed_month='2011-11') orders_1,
          coalesce(sum(order_revenue) FILTER(WHERE observed_month='2010-11'),0) revenue_0,
          coalesce(sum(order_revenue) FILTER(WHERE observed_month='2011-11'),0) revenue_1,
          coalesce(sum(units) FILTER(WHERE observed_month='2010-11'),0) units_0,
          coalesce(sum(units) FILTER(WHERE observed_month='2011-11'),0) units_1
        FROM retail_orders r JOIN customer_history h USING(customer_id)
        WHERE r.observed_month IN ('2010-11','2011-11') GROUP BY r.customer_id,h.first_day
    """).fetchdf()
    classified = customer_bridge(customer)
    bridge = classified.groupby("customer_group", as_index=False).agg(
        customers=("customer_id","size"),orders_0=("orders_0","sum"),orders_1=("orders_1","sum"),
        revenue_0=("revenue_0","sum"),revenue_1=("revenue_1","sum"),
        units_0=("units_0","sum"),units_1=("units_1","sum"),revenue_delta=("revenue_delta","sum"))
    bridge.to_csv(output/"customer_bridge.csv", index=False)
    common_customers = [str(x) for x in classified.loc[classified.customer_group == "common_november_buyers", "customer_id"]]
    # Register an in-memory relation rather than interpolating customer identifiers.
    con.register("common_customer_ids", pd.DataFrame({"customer_id": common_customers}))

    def sku_frame(extra: str = "", pair: bool = False):
        key = "customer_id || ':' || stock_code" if pair else "stock_code"
        return con.execute(f"""
            SELECT {key} AS key,
              coalesce(sum(quantity) FILTER(WHERE observed_month='2010-11'),0) units_0,
              coalesce(sum(line_revenue) FILTER(WHERE observed_month='2010-11'),0) revenue_0,
              coalesce(sum(quantity) FILTER(WHERE observed_month='2011-11'),0) units_1,
              coalesce(sum(line_revenue) FILTER(WHERE observed_month='2011-11'),0) revenue_1
            FROM retail_lines WHERE is_product AND NOT is_return AND customer_id IS NOT NULL
              AND observed_month IN ('2010-11','2011-11') {extra} GROUP BY 1
        """).fetchdf()

    price_results = {}
    sku_descriptions = con.execute("""
        WITH counts AS (
          SELECT stock_code AS key,description,count(*) n FROM retail_lines
          WHERE is_product AND NOT is_return AND observed_month IN ('2010-11','2011-11')
            AND description IS NOT NULL GROUP BY stock_code,description
        ) SELECT key,description FROM counts
        QUALIFY row_number() OVER(PARTITION BY key ORDER BY n DESC,description)=1
    """).fetchdf()
    for label, condition, pair in (
        ("identified_sku", "", False),
        ("uk_sku", "AND country='United Kingdom'", False),
        ("other_country_sku", "AND country<>'United Kingdom'", False),
        ("common_customers_sku", "AND customer_id IN (SELECT customer_id FROM common_customer_ids)", False),
        ("customer_sku_pair", "", True),
    ):
        raw_sku = sku_frame(condition, pair)
        result, contributions = price_mix_bridge(raw_sku)
        price_results[label] = result
        # Only aggregate SKU detail is exported; no row-level customer IDs.
        if not pair:
            raw_sku.merge(sku_descriptions,on="key",how="left",validate="one_to_one").to_csv(output/f"{label}_totals.csv", index=False)
            contributions.merge(sku_descriptions,on="key",how="left",validate="one_to_one").to_csv(output/f"{label}_common_contributions.csv", index=False)
    _json(output/"paid_unit_value_decomposition.json", price_results)

    print("Testing fixed pre-period large-buyer and geography sensitivity", flush=True)
    # Definition frozen before both comparison periods, not selected on decline.
    prior = con.execute("""
        SELECT customer_id,sum(order_revenue) prior_revenue FROM retail_orders
        WHERE customer_id IS NOT NULL AND order_date >= DATE '2010-01-01'
          AND order_date < DATE '2010-11-01' GROUP BY customer_id
        ORDER BY prior_revenue DESC,customer_id
    """).fetchdf()
    n_top = math.ceil(len(prior)*.01)
    excluded = prior.iloc[:n_top]
    con.register("preperiod_top_buyers", excluded[["customer_id"]])
    concentration = {"definition": "Top ceil(1% * N) identified buyers by positive product spend during 2010-01-01..2010-10-31; deterministic customer_id tie break; fixed before November 2010",
        "eligible_prior_buyers": len(prior), "fixed_top_buyers": n_top,
        "minimum_selected_prior_revenue": float(excluded.prior_revenue.min()),
        "maximum_unselected_prior_revenue": float(prior.iloc[n_top:].prior_revenue.max()),
        "selected_prior_revenue": float(excluded.prior_revenue.sum()),
        "total_prior_revenue": float(prior.prior_revenue.sum()),
        "selected_prior_revenue_share": float(excluded.prior_revenue.sum()/prior.prior_revenue.sum())}
    sensitivity = con.execute("""
        WITH focal AS (
          SELECT *,CASE WHEN country='United Kingdom' THEN 'uk' ELSE 'other_country' END geography,
            CASE WHEN customer_id IN (SELECT customer_id FROM preperiod_top_buyers)
              THEN 'fixed_top_1pct' ELSE 'excluding_fixed_top_1pct' END buyer_scope
          FROM retail_orders WHERE customer_id IS NOT NULL AND observed_month IN ('2010-11','2011-11')
        ), geo AS (
          SELECT * FROM focal UNION ALL SELECT * REPLACE('all' AS geography) FROM focal
        ), scopes AS (
          SELECT * FROM geo UNION ALL SELECT * REPLACE('all_identified' AS buyer_scope) FROM geo
        ) SELECT observed_month,geography,buyer_scope,count(DISTINCT customer_id) buyers,
            count(*) orders,sum(order_revenue) revenue,sum(units) units
          FROM scopes GROUP BY ALL ORDER BY geography,buyer_scope,observed_month
    """).fetchdf()
    sensitivity = _metrics(sensitivity)
    sensitivity["year"] = sensitivity.observed_month.str[:4].astype(int)
    sensitivity.to_csv(output/"large_buyer_sensitivity.csv", index=False)
    sensitivity_yoy = _comparisons(sensitivity.drop(columns="observed_month"), ["geography", "buyer_scope"])
    sensitivity_yoy.to_csv(output/"large_buyer_sensitivity_yoy.csv", index=False)
    for row in sensitivity.itertuples():
        if row.geography == "all" and row.buyer_scope == "fixed_top_1pct":
            total = sensitivity.query("geography == 'all' and buyer_scope == 'all_identified' and observed_month == @row.observed_month").iloc[0]
            concentration[row.observed_month] = {"active_selected_buyers": int(row.buyers),
                "revenue": row.revenue, "identified_revenue_share": row.revenue/total.revenue,
                "identified_buyer_share": row.buyers/total.buyers}

    print("Reconciling gross/net, anonymous and duplicate-record alternatives", flush=True)
    ledger = con.execute("""
      WITH variants AS (
        SELECT 'canonical' record_rule,invoice,stock_code,quantity,invoice_date,unit_price,customer_id,country FROM canonical_lines
        UNION ALL SELECT 'raw_keep_exact_duplicates',invoice,stock_code,quantity,invoice_date,unit_price,customer_id,country FROM raw_lines
      ), valid AS (
        SELECT *,strftime(invoice_date,'%Y-%m') observed_month,
          CASE WHEN starts_with(invoice,'C') AND quantity>0 THEN -quantity*unit_price ELSE quantity*unit_price END amount,
          quantity<0 OR starts_with(invoice,'C') AS is_return,
          CASE WHEN customer_id IS NULL THEN 'anonymous' ELSE 'identified' END identity_scope
        FROM variants WHERE invoice IS NOT NULL AND stock_code IS NOT NULL AND invoice_date IS NOT NULL
          AND quantity IS NOT NULL AND unit_price IS NOT NULL AND quantity<>0 AND unit_price>0
          AND regexp_full_match(stock_code,'[0-9]{5}[A-Z]*')
          AND strftime(invoice_date,'%Y-%m') IN ('2010-11','2011-11')
      ), expanded AS (
        SELECT * FROM valid UNION ALL SELECT * REPLACE('all' AS identity_scope) FROM valid
      ) SELECT record_rule,observed_month,identity_scope,
        sum(CASE WHEN NOT is_return THEN amount ELSE 0 END) gross_purchase_revenue,
        -sum(CASE WHEN is_return THEN amount ELSE 0 END) return_value,
        sum(amount) net_revenue, count(*) lines
        FROM expanded GROUP BY ALL ORDER BY record_rule,identity_scope,observed_month
    """).fetchdf()
    ledger.to_csv(output/"ledger_sensitivity.csv", index=False)
    ledger0 = ledger.query("observed_month == '2010-11'").drop(columns="observed_month")
    ledger1 = ledger.query("observed_month == '2011-11'").drop(columns="observed_month")
    ledger_yoy = ledger0.merge(ledger1,on=["record_rule","identity_scope"],suffixes=("_0","_1"),validate="one_to_one")
    for value in ("gross_purchase_revenue", "return_value", "net_revenue"):
        ledger_yoy[value+"_delta"] = ledger_yoy[value+"_1"]-ledger_yoy[value+"_0"]
        ledger_yoy[value+"_yoy"] = ledger_yoy[value+"_1"]/ledger_yoy[value+"_0"]-1
    ledger_yoy.to_csv(output/"ledger_sensitivity_yoy.csv", index=False)

    checks = {"tolerance_gbp": 1e-6, "aov_shapley_error": aov["reconciliation_error"],
        "aov_definition_error": float(aov["delta"]-focal.aov_delta),
        "customer_revenue_bridge_error": float(bridge.revenue_delta.sum()-focal.revenue_delta),
        "customer_base_buyers_error": int(classified.orders_0.gt(0).sum()-focal.buyers_0),
        "customer_current_buyers_error": int(classified.orders_1.gt(0).sum()-focal.buyers_1),
        "customer_earliest_first_observed_after_current_month": int((pd.to_datetime(classified.first_day)>=pd.Timestamp('2011-12-01')).sum()),
        "paid_unit_value_errors": {k:v.get("reconciliation_error") for k,v in price_results.items()},
        "centered_sku_mix_max_error": max(abs(v["centered_mix_reconciliation_error"]) for v in price_results.values()),
        "gross_minus_returns_net_max_error": float(abs(ledger.gross_purchase_revenue-ledger.return_value-ledger.net_revenue).max()),
    }
    checks["sku_population_revenue_max_error"] = max(abs(
        price_results["identified_sku"][f"paid_unit_value_{t}"] *
        price_results["identified_sku"][f"units_{t}"] - focal[f"revenue_{t}"]
    ) for t in (0,1))
    checks["sku_population_units_max_error"] = max(abs(
        price_results["identified_sku"][f"units_{t}"]-focal[f"units_{t}"]
    ) for t in (0,1))
    checks["canonical_ledger_vs_order_revenue_max_error"] = max(abs(
        ledger.query("record_rule == 'canonical' and identity_scope == 'identified' and observed_month == @period").iloc[0].gross_purchase_revenue-focal[f"revenue_{t}"]
    ) for t,period in enumerate(FOCAL))
    scope_errors = []
    for _, rows in ledger.groupby(["record_rule", "observed_month"]):
        by = rows.set_index("identity_scope")
        for metric in ("gross_purchase_revenue", "return_value", "net_revenue"):
            scope_errors.append(float(by.loc["all",metric]-by.loc["identified",metric]-by.loc["anonymous",metric]))
    checks["identity_scope_money_max_error"] = max(abs(x) for x in scope_errors)
    sensitivity_errors = []
    for _, rows in sensitivity.groupby(["observed_month","geography"]):
        by = rows.set_index("buyer_scope")
        for metric in ("buyers","orders","revenue","units"):
            sensitivity_errors.append(float(by.loc["all_identified",metric]-by.loc["fixed_top_1pct",metric]-by.loc["excluding_fixed_top_1pct",metric]))
    checks["large_buyer_partition_max_error"] = max(abs(x) for x in sensitivity_errors)
    geographic_errors = []
    for _, rows in sensitivity.groupby(["observed_month", "buyer_scope"]):
        by = rows.set_index("geography")
        for metric in ("orders", "revenue", "units"):
            geographic_errors.append(float(by.loc["all",metric]-by.loc["uk",metric]-by.loc["other_country",metric]))
    checks["geographic_money_order_units_max_error"] = max(abs(x) for x in geographic_errors)
    common_pair = price_results["customer_sku_pair"]
    overall = ledger_yoy.query("record_rule == 'canonical' and identity_scope == 'all'").iloc[0]
    anon = ledger_yoy.query("record_rule == 'canonical' and identity_scope == 'anonymous'").iloc[0]
    ex_top = sensitivity_yoy.query("geography == 'all' and buyer_scope == 'excluding_fixed_top_1pct'").iloc[0]
    top = sensitivity_yoy.query("geography == 'all' and buyer_scope == 'fixed_top_1pct'").iloc[0]
    raw_identified = ledger_yoy.query("record_rule == 'raw_keep_exact_duplicates' and identity_scope == 'identified'").iloc[0]
    canonical_identified = ledger_yoy.query("record_rule == 'canonical' and identity_scope == 'identified'").iloc[0]
    full_identified = pooled_yoy.query("window_name == 'jan_nov' and identity_scope == 'identified'").iloc[0]
    hypotheses = [
      {"id":"H0", "hypothesis":"The whole business lost revenue in November", "finding":"Contradicted by the product ledger scope",
       "evidence":{"all_gross_yoy":overall.gross_purchase_revenue_yoy,"all_net_yoy":overall.net_revenue_yoy,"identified_gross_yoy":focal.revenue_yoy,"anonymous_gross_yoy":anon.gross_purchase_revenue_yoy},
       "limit":"Anonymous invoices cannot be assigned to people; identity migration is not established."},
      {"id":"H1", "hypothesis":"Lower items per order accounts for the identified AOV drop", "finding":"Part of the exact accounting bridge",
       "evidence":{"aov_delta_gbp":aov['delta'],"units_per_order_component_gbp":aov['quantity_component'],"paid_unit_value_component_gbp":aov['value_component']},
       "limit":"Product mix, customer mix and quantity discounts can jointly change transaction averages."},
      {"id":"H2", "hypothesis":"Same-SKU paid unit value explains the whole lower pooled unit value", "finding":"Separate within-SKU and mix contributions; do not label all movement a price cut",
       "evidence":price_results['identified_sku'],"limit":"Paid unit averages are not list prices and have no promotion/treatment assignment."},
      {"id":"H3", "hypothesis":"A few large buyers generate the identified revenue decline", "finding":"Fixed pre-period concentration sensitivity",
       "evidence":{"excluded_top_revenue_delta":top.revenue_delta,"excluded_top_revenue_yoy":top.revenue_yoy,"remaining_revenue_delta":ex_top.revenue_delta,"remaining_revenue_yoy":ex_top.revenue_yoy},
       "limit":"Excluding these buyers changes the estimand; no claim that they are invalid or causally responsible."},
      {"id":"H4", "hypothesis":"Buyer entry offsets missing November buyers while common buyers also change spend", "finding":"Exact four-group revenue bridge",
       "evidence":bridge.to_dict(orient='records'),"limit":"First observed is not newly acquired; November absence is not permanent churn."},
      {"id":"H5", "hypothesis":"Exact duplicates or return posting reverse the identified sign", "finding":"Alternative record and financial scopes are measured explicitly",
       "evidence":{"canonical_identified_gross_yoy":focal.revenue_yoy,"raw_identified_gross_yoy":raw_identified.gross_purchase_revenue_yoy,"canonical_identified_net_yoy":canonical_identified.net_revenue_yoy,"raw_identified_net_yoy":raw_identified.net_revenue_yoy},
       "limit":"Returns are posting-date credit notes and cannot be linked reliably to original purchases; exact duplicates may be legitimate identical lines."},
      {"id":"H6", "hypothesis":"The November pattern represents the full available calendar comparison", "finding":"Compare all 11 matched months and pooled windows",
       "evidence":{"jan_nov_identified_revenue_yoy":full_identified.revenue_yoy,"jan_nov_identified_buyers_yoy":full_identified.buyers_yoy,"nov_identified_revenue_yoy":focal.revenue_yoy},
       "limit":"Only two historical years and one retailer; December 2011 is incomplete."},
    ]
    all_errors = [v for k,v in checks.items() if k not in ("tolerance_gbp", "paid_unit_value_errors")]
    all_errors.extend(checks["paid_unit_value_errors"].values())
    checks["passed"] = all(value is not None and abs(value)<1e-6 for value in all_errors)
    _json(output/"checks.json", checks)
    summary = {"case":"November 2011 versus November 2010: opposite signs across identity and financial scopes",
        "data":"UCI Online Retail II, canonical product purchase invoices, source GBP",
        "focal_identified":focal.to_dict(),"all_scope_ledger":ledger_yoy.to_dict(orient="records"),
        "aov_shapley":aov,"paid_unit_value_bridges":price_results,
        "customer_bridge":bridge.to_dict(orient="records"),"fixed_prior_top_buyers":concentration,
        "large_buyer_sensitivity":sensitivity_yoy.to_dict(orient="records"),
        "pooled_yoy":pooled_yoy.to_dict(orient="records"),"hypotheses":hypotheses,
        "checks_passed":checks["passed"]}
    _json(output/"summary.json", summary)
    _json(output/"hypothesis_evidence.json", hypotheses)
    _plot(output, monthly_yoy, aov, price_results['identified_sku'], bridge, sensitivity_yoy)
    _technical(output, summary)
    con.close()
    if not checks["passed"]:
        raise AssertionError(f"Reconciliation failed: {checks}")
    print(json.dumps({"output":str(output),"checks_passed":True,"aov_shapley":aov,
        "sku_components":price_results['identified_sku']['components'],
        "large_buyer_sensitivity":hypotheses[3]['evidence']},indent=2),flush=True)
    return summary


def _plot(output, monthly, aov, sku, customer, sensitivity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(15,10),gridspec_kw={"hspace":.42,"wspace":.36})
    colors={"all":"#168078","identified":"#174a7e","anonymous":"#cc7a29"}
    for scope in colors:
        data=monthly.query("identity_scope == @scope").sort_values("month")
        axes[0,0].plot(data.month,data.revenue_yoy,marker='o',label=scope.capitalize(),color=colors[scope])
    axes[0,0].axhline(0,color="#aaa",lw=.8)
    axes[0,0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].set(xticks=range(1,12),xlabel="Matched calendar month (2011 vs 2010)",ylabel="Positive product purchase revenue YoY",title="A. November is a scope-dependent signal")
    axes[0,0].legend(frameon=False,ncol=3,fontsize=9)
    vals=[aov['quantity_component'],aov['value_component']]
    axes[0,1].barh(["Fewer units per invoice","Lower paid value per unit"],vals,color=["#558bb3","#ba663f"])
    for i,v in enumerate(vals):axes[0,1].text(v-.4,i,f"{v:+.2f}",ha='right',va='center')
    axes[0,1].set(xlim=(min(vals)*1.5,1),xlabel="Contribution to AOV change (GBP / invoice)",title=f"B. Identified AOV fell GBP {abs(aov['delta']):.2f}")
    labels={"common_november_buyers":"Common November buyers","base_only_november_buyers":"Base-only November buyers","current_only_first_observed_in_november":"Current-only: first observed","current_only_previously_observed":"Current-only: previously observed"}
    names=[labels[x] for x in customer.customer_group]
    vals=customer.revenue_delta/1000
    axes[1,0].barh(names,vals,color=["#ba663f" if v<0 else "#168078" for v in vals])
    for i,v in enumerate(vals):axes[1,0].text(v+(7 if v>=0 else -7),i,f"{v:+.1f}",ha='left' if v>=0 else 'right',va='center')
    axes[1,0].set(xlim=(min(vals)*1.28,max(vals)*1.40),xlabel="Revenue change (GBP thousands)",title="C. Buyer-group accounting bridge")
    axes[1,0].tick_params(axis='y',labelsize=9)
    sub=sensitivity.query("geography == 'all'").set_index('buyer_scope')
    order=['all_identified','fixed_top_1pct','excluding_fixed_top_1pct']
    values=[sub.loc[x,'revenue_yoy'] for x in order]
    axes[1,1].bar(["All identified","Fixed prior\ntop 1%","Other identified"],values,color=["#174a7e","#ba663f","#168078"])
    axes[1,1].axhline(0,color="#aaa",lw=.8)
    axes[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    for i,v in enumerate(values):axes[1,1].text(i,v+(.01 if v>=0 else -.012),f"{v:+.1%}",ha='center',va='bottom' if v>=0 else 'top')
    axes[1,1].set(ylim=(min(values)-.08,max(values)+.08),ylabel="November revenue YoY",title="D. Large-buyer definition fixed in Jan-Oct 2010")
    fig.suptitle("Retail growth diagnosis: accounting evidence before intervention",fontsize=16,y=.98)
    fig.text(.03,.025,"Canonical purchase invoices; source GBP. Paid unit value is a transaction average. Descriptive bridges do not identify causal effects.",fontsize=9,color="#555")
    fig.subplots_adjust(left=.19,right=.97,bottom=.10,top=.92)
    fig.savefig(output/"diagnosis.png",dpi=160,bbox_inches='tight')
    plt.close(fig)


def _technical(output: Path, summary: dict):
    text = """# Retail diagnosis: technical contract

Run from the repository root with `python -m retail.diagnose`. The default database is `data/processed/retail.duckdb`, opened read-only with a 256 MB memory limit and two threads. No customer identifiers or customer-level rows are exported. Every figure and JSON result is reproduced from the existing full database.

## Scope and periods

The base and current focal periods are complete November 2010 and November 2011. Revenue means positive **product** invoice value in source GBP; recognised non-product charges are outside this measure. Returns are reported separately using the existing sign rule. Identified means the source customer ID is populated; anonymous rows remain in total sales but have no countable distinct buyers. `buyers` for `all` therefore means identified unique buyers within that broader revenue scope; `buyers` for anonymous is missing, not zero actual people. Orders are invoices, not necessarily shopping sessions or business contracts. The person-level frequency `orders_per_identified_buyer` is only populated for the identified scope; all-scope and anonymous frequency are missing.

`monthly_yoy.csv` contains each January-November matched comparison. `pooled_yoy.csv` contains January-November, Q1, Q2, Q3, September-November and October-November. Pooled buyers are distinct over the whole window. September-November is an autumn window, not Q3; October-November is not a complete Q4. December 2011 is incomplete and is excluded. A single selected month cannot establish the long-term trend. The focal month was selected after examining data; all comparisons are retrospective descriptive analysis.

## Exact AOV bridge

For identified purchases, let U=units/orders and P=revenue/units, so AOV=U*P. The symmetric two-factor Shapley bridge is:

* Units-per-order contribution: `(U1-U0)*(P0+P1)/2`.
* Paid-unit-value contribution: `(P1-P0)*(U0+U1)/2`.

These sum exactly to AOV1-AOV0. The interaction is split equally. This is an accounting allocation, not causal attribution.

## Exact paid-unit-value bridge

For the union of product codes in both months, C is the intersection, s_t is C's fraction of total purchased units, and P_C,t is revenue/units within C. Within C, `w_i,t=q_i,t/sum_C(q_t)` and `p_i,t=r_i,t/q_i,t`.

`Delta P_C = sum_i [(w_i,0+w_i,1)/2 * (p_i,1-p_i,0)] + sum_i [(p_i,0+p_i,1)/2 * (w_i,1-w_i,0)]`.

The overall change is the sum of five exact components:

1. `mean(s) * within-common-SKU paid-unit-value change`.
2. `mean(s) * within-common-SKU quantity-weight mix change`.
3. `mean(P_C) * (s1-s0)`, the common-set quantity share contribution.
4. Revenue from current-only SKUs divided by **all** current-period units.
5. Negative revenue from base-only SKUs divided by **all** base-period units.

Components 3-5 jointly describe entry/exit and its displacement of the common set. Components 4 and 5 must not individually be read as their complete net effect. The same exact bridge is supplied for UK, other countries, buyers present in both Novembers, and customer-SKU pairs. The pair bridge reduces between-customer composition within a key but cannot hold order size, discount, variant, timing or contract terms fixed. In the actual data, only 4,723 customer-SKU pairs occur in both Novembers, covering 16.012% of base-period units and 14.134% of current-period units. Their positive within-pair component cannot establish rising paid prices for the overall customer base. Paid-unit values are transaction-weighted averages, **not** posted or list prices, and SKU codes need not guarantee unchanged product quality. No 'price cut caused decline' conclusion is identified.

The SKU contribution tables include both the direct mix allocation and a centred allocation subtracting `mean(P_C)` from each SKU's mean paid unit value before multiplying by its weight change. The total is unchanged because weight changes sum to zero. Use the centred field for SKU drill-downs: loss of weight for an above-reference SKU or gain of weight for a below-reference SKU contributes negatively. SKU-level attribution depends on this declared reference; it is not a unique causal effect. Product descriptions use the most frequent description across the two focal periods, with lexical tie breaking. Descriptions assist review but do not establish a consistent product taxonomy or unchanged product quality.

If the common set is empty its weighted components are zero and entry/exit supplies the full change. If either period has no positive units, a paid-unit average is undefined and no decomposition is asserted.

## Customer and large-buyer comparisons

`customer_bridge.csv` partitions the union of identified November buyers into: present in both months, base-only, current-only first observed during November 2011, and current-only previously observed. Group revenue changes sum exactly to the total identified change; current-only and base-only are period membership, not causal acquisition or permanent churn. First-observed dates are left-censored by the dataset starting in December 2009. Previously observed includes customers first seen after the base period and customers last seen earlier; it does not prove a win-back campaign.

The fixed large-buyer set is selected on January-October 2010 positive product purchase spend, before **both** focal Novembers: `ceil(N*1%)` buyers ordered by spend descending and customer ID for tie breaking. It is not defined using outcome-period spend. This set is held fixed for both years and reported with counts, revenue and shares. Sensitivities show UK, other countries and their pooled totals, including the retained large buyers and the complementary population. This tests concentration dependence; removing high-value customers changes the estimand and does not imply those observations should be deleted.

## Record and financial reconciliation

`ledger_sensitivity_yoy.csv` compares the canonical exact-row deduplication rule with all source rows retained, each for identified, anonymous and total product ledgers. The source has no stable line ID; identical legitimate lines cannot be distinguished from duplicates, so both assumptions are shown. Gross-minus-posted-credit equals net; identified plus anonymous equals total. A return's invoice date is its posting date. Without a reliable original-order link the return fraction is a posting-period accounting ratio, not a purchase cohort's refund probability. No date-shifting of returns is assumed.

`checks.json` verifies monetary partitions and every exact bridge with an absolute tolerance of GBP 0.000001 (or the corresponding per-unit scale). The dataset's monetary fields originate as floating-point values. Small summation errors below this tolerance are numerical, not hidden residual categories. No residual is silently discarded. Meaningful tests use hand-computable examples, disjoint SKU sets, undefined empty periods and buyer-group reconciliation.

## Decision boundary

These results narrow competing descriptive explanations and identify missing evidence. The customer-group, large-buyer and basket/SKU bridges are overlapping views of the same transactions: their components cannot be added across bridges as if they were independent causes. They cannot establish marketing ROI, coupon uplift, buyer identity migration, demand elasticities, lost acquisition, reasons for customer absence, or which segment should receive a subsidy. Those require exposure, assignment, identity quality, marketing cost, margin, inventory, discount and customer feedback data plus a separately designed intervention evaluation. This historical UK retailer cannot establish Douyin's growth performance.
"""
    (output/"TECHNICAL.md").write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,default=Path("data/processed/retail.duckdb"))
    parser.add_argument("--output",type=Path,default=Path("reports/retail_diagnosis"))
    args=parser.parse_args()
    run(args.db,args.output)


if __name__ == "__main__":
    main()
