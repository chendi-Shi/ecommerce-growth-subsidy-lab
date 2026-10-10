"""Aggregate follow-up of the fixed pre-November high-spend customer cohort.

Historical purchase records support descriptive bridges, not retention, stockout,
discount, or subsidy effects. Customer identifiers stay in process and are never
written to the report directory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


FACTORS = ("active_buyers", "orders_per_buyer", "units_per_order", "paid_unit_value")
WINDOWS = {"november": (11, 11), "oct_nov": (10, 11), "sep_nov": (9, 11), "jan_nov": (1, 11)}
GROUPS = ("both_novembers", "base_november_only", "current_november_only", "neither_november")


def multiplicative_shapley(base: dict, current: dict) -> dict:
    """Average all factor orderings; interaction terms are allocated symmetrically."""
    names = tuple(base)
    if set(names) != set(current) or not names:
        raise ValueError("Both periods must have the same nonempty factor set")
    if not all(math.isfinite(float(v)) and v >= 0 for v in [*base.values(), *current.values()]):
        raise ValueError("Factors must be finite and nonnegative")
    components = dict.fromkeys(names, 0.0)
    for ordering in itertools.permutations(names):
        state = dict(base)
        previous = math.prod(state.values())
        for name in ordering:
            state[name] = current[name]
            updated = math.prod(state.values())
            components[name] += updated - previous
            previous = updated
    components = {k: v / math.factorial(len(names)) for k, v in components.items()}
    delta = math.prod(current.values()) - math.prod(base.values())
    return {"revenue_delta": delta, "components": components,
            "reconciliation_error": delta - sum(components.values())}


def activity_groups(orders_0: pd.Series, orders_1: pd.Series) -> np.ndarray:
    """Absence is window-specific; it does not imply permanent customer loss."""
    if len(orders_0) != len(orders_1) or (orders_0 < 0).any() or (orders_1 < 0).any():
        raise ValueError("Order counts must have equal length and be nonnegative")
    return np.select([
        (orders_0 > 0) & (orders_1 > 0),
        (orders_0 > 0) & (orders_1 == 0),
        (orders_0 == 0) & (orders_1 > 0),
    ], GROUPS[:3], default=GROUPS[3])


def concentration_bridge(deltas: np.ndarray) -> dict:
    """Cumulative anonymous loss concentration keeps gain offsets visible.

    Loss magnitudes are positive. Net decline may be negative if gains dominate;
    top-loss shares of net decline are then undefined, not silently clipped.
    """
    d = np.asarray(deltas, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("Deltas must be finite")
    losses = np.sort(-d[d < 0])[::-1]
    gross_loss, gross_gain = float(losses.sum()), float(d[d > 0].sum())
    net_decline = gross_loss - gross_gain
    return {"declining_customers": int((d < 0).sum()), "increasing_customers": int((d > 0).sum()),
            "unchanged_customers": int((d == 0).sum()), "gross_loss": gross_loss,
            "gross_gain": gross_gain, "net_decline": net_decline, "revenue_delta": float(d.sum()),
            "reconciliation_error": float(d.sum()) - (gross_gain - gross_loss),
            "cumulative_largest_losses": [{"requested_top_k": k, "included_customers": min(k, len(losses)),
                "gross_loss": float(losses[:k].sum()),
                "share_of_all_gross_losses": float(losses[:k].sum()) / gross_loss if gross_loss else None,
                "share_of_net_decline": float(losses[:k].sum()) / net_decline if net_decline > 0 else None}
                for k in (1, 3, 5)]}


def paired_window_bridge(november_delta: np.ndarray, sep_oct_delta: np.ndarray) -> dict:
    """Check same-customer arithmetic coverage before interpreting pooled timing.

    min(positive Sep-Oct change, November loss) is compatible amount coverage,
    never evidence that orders actually moved between months.
    """
    nov, prior = np.asarray(november_delta, dtype=float), np.asarray(sep_oct_delta, dtype=float)
    if nov.shape != prior.shape or nov.ndim != 1 or not (np.isfinite(nov).all() and np.isfinite(prior).all()):
        raise ValueError("Matched customer deltas must be finite one-dimensional arrays of equal length")
    wider = nov + prior
    declining = nov < 0
    matched = np.minimum(np.maximum(prior, 0), np.maximum(-nov, 0))
    losses = float(-nov[declining].sum())
    groups = []
    for nov_decline in (True, False):
        for sepnov_decline in (True, False):
            mask = (declining == nov_decline) & ((wider < 0) == sepnov_decline)
            groups.append({"november_decline": nov_decline, "sep_nov_decline": sepnov_decline,
                "customers": int(mask.sum()), "november_delta": float(nov[mask].sum()),
                "sep_oct_delta": float(prior[mask].sum()), "sep_nov_delta": float(wider[mask].sum()),
                "arithmetic_amount_covered_by_same_customer_sep_oct_increase": float(matched[mask].sum())})
    return {"november_declining_customers": int(declining.sum()),
        "november_gross_loss_in_declining_customers": losses,
        "same_customer_positive_sep_oct_increase_capped_at_november_loss": float(matched.sum()),
        "arithmetic_coverage_of_november_gross_losses": float(matched.sum()) / losses if losses else None,
        "november_decliners_still_down_in_sep_nov": int((declining & (wider < 0)).sum()),
        "november_decliners_still_down_sep_nov_gross_loss": float(-wider[declining & (wider < 0)].sum()),
        "all_november_decliners_sep_nov_net_delta": float(wider[declining].sum()),
        "other_customers_sep_nov_net_delta": float(wider[~declining].sum()),
        "all_customers_sep_nov_net_delta": float(wider.sum()),
        "groups": groups,
        "interpretation": "Same-customer arithmetic coverage does not match invoices or establish purchasing brought forward; cohort selection overlaps the base Sep-Oct window."}


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _write_json(path: Path, value):
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metrics(row: dict) -> dict:
    r = dict(row)
    for key, numerator, denominator in (("orders_per_buyer", "orders", "active_buyers"),
            ("units_per_order", "units", "orders"), ("paid_unit_value", "revenue", "units"),
            ("aov", "revenue", "orders")):
        r[key] = r[numerator] / r[denominator] if r[denominator] else None
    return r


def _factors(row):
    return {k: float(row[k]) for k in FACTORS}


def run(db: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db), read_only=True, config={"memory_limit": "256MB", "threads": "2"})
    prior = con.execute("""SELECT customer_id, sum(order_revenue) prior_revenue
        FROM retail_orders WHERE customer_id IS NOT NULL
        AND order_date >= DATE '2010-01-01' AND order_date < DATE '2010-11-01'
        GROUP BY customer_id ORDER BY prior_revenue DESC, customer_id""").fetchdf()
    n_top = math.ceil(len(prior) * .01)
    cohort = prior.head(n_top).copy()
    con.register("fixed_top_buyers", cohort[["customer_id"]])
    selected = con.execute("""SELECT r.customer_id, r.order_date, r.order_revenue, r.units
        FROM retail_orders r JOIN fixed_top_buyers t USING(customer_id)
        WHERE order_date >= DATE '2010-01-01' AND order_date < DATE '2011-12-01'""").fetchdf()
    selected["year"] = selected.order_date.dt.year
    selected["month"] = selected.order_date.dt.month

    window_rows = []
    for name, (start, end) in WINDOWS.items():
        frame = con.execute("""WITH labelled AS (
            SELECT *, CASE WHEN customer_id IN (SELECT customer_id FROM fixed_top_buyers)
                THEN 'fixed_top_1pct' ELSE 'other_identified' END customer_scope
            FROM retail_orders WHERE customer_id IS NOT NULL
                AND order_date >= DATE '2010-01-01' AND order_date < DATE '2011-12-01'
                AND month(order_date) BETWEEN ? AND ?), expanded AS (
            SELECT * FROM labelled UNION ALL
            SELECT * REPLACE ('all_identified' AS customer_scope) FROM labelled)
            SELECT customer_scope, year(order_date) AS year, count(DISTINCT customer_id) active_buyers,
                count(*) AS orders, sum(units) AS units, sum(order_revenue) AS revenue
            FROM expanded GROUP BY ALL ORDER BY customer_scope, year""", [start, end]).fetchdf()
        for row in frame.to_dict("records"):
            window_rows.append(_metrics({"window": name,
                "base_window_overlaps_cohort_selection": name != "november", **row}))
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "window_metrics.csv", index=False)
    yoy = windows[windows.year == 2010].drop(columns="year").merge(
        windows[windows.year == 2011].drop(columns="year"),
        on=["window", "customer_scope", "base_window_overlaps_cohort_selection"],
        suffixes=("_0", "_1"), validate="one_to_one")
    for metric in ("revenue", "orders", "units", "active_buyers", "aov"):
        yoy[metric + "_delta"] = yoy[metric + "_1"] - yoy[metric + "_0"]
        yoy[metric + "_yoy"] = yoy[metric + "_1"] / yoy[metric + "_0"] - 1
    yoy.to_csv(output / "window_yoy.csv", index=False)

    nov = selected[selected.month == 11]
    customers = cohort[["customer_id"]].copy()
    for i, year in enumerate((2010, 2011)):
        totals = nov[nov.year == year].groupby("customer_id").agg(
            orders=("order_revenue", "size"), revenue=("order_revenue", "sum"), units=("units", "sum"))
        customers = customers.join(totals.add_suffix(f"_{i}"), on="customer_id").fillna(0)
    customers["activity_group"] = activity_groups(customers.orders_0, customers.orders_1)
    customers["revenue_delta"] = customers.revenue_1 - customers.revenue_0
    bridge = customers.groupby("activity_group").agg(
        customers=("customer_id", "size"), orders_0=("orders_0", "sum"), orders_1=("orders_1", "sum"),
        units_0=("units_0", "sum"), units_1=("units_1", "sum"),
        revenue_0=("revenue_0", "sum"), revenue_1=("revenue_1", "sum"),
        revenue_delta=("revenue_delta", "sum")).reindex(GROUPS, fill_value=0).reset_index()
    bridge.to_csv(output / "november_activity_bridge.csv", index=False)
    concentration = concentration_bridge(customers.revenue_delta.to_numpy())
    pd.DataFrame(concentration["cumulative_largest_losses"]).to_csv(output / "loss_concentration.csv", index=False)
    for i, year in enumerate((2010, 2011)):
        earlier = selected[(selected.year == year) & selected.month.between(9, 10)]
        total = earlier.groupby("customer_id").order_revenue.sum().rename(f"sep_oct_revenue_{i}")
        customers = customers.join(total, on="customer_id").fillna(0)
    timing = paired_window_bridge(customers.revenue_delta.to_numpy(),
        (customers.sep_oct_revenue_1 - customers.sep_oct_revenue_0).to_numpy())
    pd.DataFrame(timing["groups"]).to_csv(output / "same_customer_window_bridge.csv", index=False)

    # A frozen invoice-size cutoff allows a like-for-like class comparison.
    # It is an investigation threshold, not a wholesale/customer-type label.
    historical_orders = selected[(selected.year == 2010) & (selected.month < 11)]
    threshold = float(historical_orders.order_revenue.quantile(.9))
    dist, bins = [], []
    for year in (2010, 2011):
        d = nov[nov.year == year]
        index = 0 if year == 2010 else 1
        spend = customers[f"revenue_{index}"]
        dist.append({"year": year, "cohort_customers": n_top,
            "active_buyers": int((spend > 0).sum()), "orders": len(d),
            "order_revenue_median": float(d.order_revenue.median()),
            "order_revenue_p90": float(d.order_revenue.quantile(.9)),
            "order_units_median": float(d.units.median()), "order_units_p90": float(d.units.quantile(.9)),
            "customer_spend_median_all_fixed_customers": float(spend.median()),
            "customer_spend_p90_all_fixed_customers": float(spend.quantile(.9)),
            "customer_spend_median_active_only": float(spend[spend > 0].median())})
        for label, mask in (("below_prior_order_p90", d.order_revenue < threshold),
                            ("at_or_above_prior_order_p90", d.order_revenue >= threshold)):
            b = d[mask]
            bins.append(_metrics({"year": year, "size_bin": label,
                "threshold_gbp": threshold, "active_buyers": b.customer_id.nunique(),
                "orders": len(b), "units": float(b.units.sum()), "revenue": float(b.order_revenue.sum())}))
    distribution = pd.DataFrame(dist)
    distribution.to_csv(output / "november_distributions.csv", index=False)
    size_bins = pd.DataFrame(bins)
    size_bins.to_csv(output / "november_order_size_bins.csv", index=False)
    size_yoy = size_bins[size_bins.year == 2010].drop(columns="year").merge(
        size_bins[size_bins.year == 2011].drop(columns="year"), on=["size_bin", "threshold_gbp"],
        suffixes=("_0", "_1"), validate="one_to_one")
    for metric in ("revenue", "orders", "units", "aov"):
        size_yoy[metric + "_delta"] = size_yoy[metric + "_1"] - size_yoy[metric + "_0"]
    size_yoy.to_csv(output / "november_order_size_yoy.csv", index=False)

    decomposition = {}
    fixed = windows[(windows.window == "november") & (windows.customer_scope == "fixed_top_1pct")].set_index("year")
    decomposition["all_fixed_customers"] = multiplicative_shapley(_factors(fixed.loc[2010]), _factors(fixed.loc[2011]))
    common = bridge[bridge.activity_group == "both_novembers"].iloc[0]
    common_metrics = [_metrics({"active_buyers": int(common.customers), "orders": common[f"orders_{i}"],
        "units": common[f"units_{i}"], "revenue": common[f"revenue_{i}"]}) for i in (0, 1)]
    decomposition["both_novembers_only"] = multiplicative_shapley(_factors(common_metrics[0]), _factors(common_metrics[1]))
    decomp_rows = [{"population": group, "factor": factor, "contribution_gbp": amount}
        for group, result in decomposition.items() for factor, amount in result["components"].items()]
    pd.DataFrame(decomp_rows).to_csv(output / "november_four_factor_bridge.csv", index=False)

    checks = {"fixed_cohort_vs_activity_groups_count_error": int(bridge.customers.sum()) - n_top,
        "activity_revenue_delta_error": float(bridge.revenue_delta.sum()) - float(fixed.loc[2011, "revenue"] - fixed.loc[2010, "revenue"]),
        "loss_gain_reconciliation_error": concentration["reconciliation_error"],
        "all_four_factor_error": decomposition["all_fixed_customers"]["reconciliation_error"],
        "common_four_factor_error": decomposition["both_novembers_only"]["reconciliation_error"],
        "invoice_bins_revenue_delta_error": float(size_yoy.revenue_delta.sum()) - float(bridge.revenue_delta.sum()),
        "same_customer_window_partition_count_error": sum(r["customers"] for r in timing["groups"]) - n_top,
        "same_customer_november_delta_error": sum(r["november_delta"] for r in timing["groups"]) - float(bridge.revenue_delta.sum()),
        "same_customer_sep_nov_delta_error": timing["all_customers_sep_nov_net_delta"] - float(yoy[(yoy.window == "sep_nov") & (yoy.customer_scope == "fixed_top_1pct")].iloc[0].revenue_delta)}
    for year in (2010, 2011):
        for metric in ("revenue", "units", "orders"):
            checks[f"invoice_bins_{year}_{metric}_error"] = float(size_bins[size_bins.year == year][metric].sum() - fixed.loc[year, metric])
    partition_errors = []
    for _, group in windows.groupby(["window", "year"]):
        g = group.set_index("customer_scope")
        for metric in ("revenue", "units", "orders", "active_buyers"):
            partition_errors.append(float(g.loc["all_identified", metric] - g.loc["fixed_top_1pct", metric] - g.loc["other_identified", metric]))
    checks["all_window_cohort_partition_max_error"] = max(abs(x) for x in partition_errors)
    check_result = {"tolerance": 1e-6, "checks": checks, "passed": all(abs(x) < 1e-6 for x in checks.values())}
    if not check_result["passed"]:
        raise AssertionError(f"Aggregate reconciliation failed: {check_result}")
    summary = {"data": "UCI Online Retail II; canonical positive product invoices with identified customers",
        "currency": "GBP", "cohort": {"eligible_prior_customers": len(prior), "fixed_customers": n_top,
            "selection_start": "2010-01-01", "selection_end_exclusive": "2010-11-01",
            "definition": "ceil(1% * eligible customers), prior purchase spend descending; customer_id tie break",
            "minimum_selected_prior_spend": float(cohort.prior_revenue.min()),
            "fully_predetermined_comparison": "November 2010 versus November 2011",
            "overlap_warning": "Other 2010 comparison windows overlap the selection period; selection and regression-to-mean can affect those descriptive comparisons."},
        "window_comparisons": yoy.to_dict("records"), "november_activity_bridge": bridge.to_dict("records"),
        "same_customer_window_bridge": timing,
        "november_loss_concentration": concentration, "november_factor_decomposition": decomposition,
        "november_distributions": distribution.to_dict("records"),
        "order_size_threshold": {"prior_period_invoice_p90_gbp": threshold,
            "prior_period_orders": len(historical_orders), "quantile_method": "linear; numpy/pandas default",
            "scope": "Invoices from the same fixed cohort, January-October 2010; no November outcome used"},
        "november_order_size_comparisons": size_yoy.to_dict("records"),
        "interpretation": ["Paid unit value is transaction-weighted revenue per unit, not a like-for-like price index.",
            "The activity and four-factor bridges are alternative views; do not add across decompositions.",
            "Gross losses exclude increases; net decline includes gain offsets. Loss concentration can exceed 100% of net decline.",
            "No December 2011 recovery claim: the extract ends during December.",
            "Large invoice counts do not establish why orders shrank or whether orders moved to another month.",
            "No customer identifiers, invoice identifiers, or customer-level rankings are exported."]}
    _write_json(output / "checks.json", check_result)
    _write_json(output / "summary.json", summary)
    _write_findings(output, summary)
    con.close()
    print(json.dumps({"fixed_customers": n_top, "checks_passed": check_result["passed"],
        "november_net_decline_gbp": concentration["net_decline"],
        "fixed_cohort_windows": [{"window": r["window"], "revenue_yoy": r["revenue_yoy"]}
            for r in summary["window_comparisons"] if r["customer_scope"] == "fixed_top_1pct"]}, ensure_ascii=False, indent=2))
    return summary


def _write_findings(output: Path, s: dict):
    cohort = s["cohort"]
    comparisons = {r["window"]: r for r in s["window_comparisons"] if r["customer_scope"] == "fixed_top_1pct"}
    bridge = {r["activity_group"]: r for r in s["november_activity_bridge"]}
    losses = s["november_loss_concentration"]
    factors = s["november_factor_decomposition"]
    timing = s["same_customer_window_bridge"]
    size = {r["size_bin"]: r for r in s["november_order_size_comparisons"]}
    d0, d1 = s["november_distributions"]
    fmt = lambda v: f"£{v:,.2f}" if v >= 0 else f"−£{abs(v):,.2f}"
    window_table = "\n".join(f"| {name} | {fmt(r['revenue_0'])} | {fmt(r['revenue_1'])} | {r['revenue_yoy']:+.2%} | {'否' if name == 'november' else '是'} |" for name, r in comparisons.items())
    group_names = {"both_novembers": "两年11月均购买", "base_november_only": "仅2010年11月购买",
        "current_november_only": "仅2011年11月购买", "neither_november": "两年11月均未购买"}
    group_table = "\n".join(f"| {group_names[k]} | {r['customers']} | {fmt(r['revenue_0'])} | {fmt(r['revenue_1'])} | {fmt(r['revenue_delta'])} |" for k, r in bridge.items())
    factor_names = {"active_buyers": "活跃人数", "orders_per_buyer": "人均订单数", "units_per_order": "每单件数", "paid_unit_value": "每件成交金额"}
    factor_table = "\n".join(f"| {factor_names[k]} | {fmt(v)} | {fmt(factors['both_novembers_only']['components'][k])} |" for k, v in factors['all_fixed_customers']['components'].items())
    concentration_table = "\n".join(f"| 前{r['requested_top_k']}名减额客户合计 | {fmt(r['gross_loss'])} | {r['share_of_all_gross_losses']:.2%} | {r['share_of_net_decline']:.2%} |" for r in losses['cumulative_largest_losses'])
    bins_table = "\n".join(f"| {'低于阈值' if k.startswith('below') else '达到或超过阈值'} | {r['orders_0']} → {r['orders_1']} | {fmt(r['revenue_delta'])} | {fmt(r['aov_0'])} → {fmt(r['aov_1'])} |" for k, r in size.items())
    paired_table = "\n".join(f"| {'11月减额' if r['november_decline'] else '11月未减额'} / {'9—11月减额' if r['sep_nov_decline'] else '9—11月未减额'} | {r['customers']} | {fmt(r['november_delta'])} | {fmt(r['sep_oct_delta'])} | {fmt(r['sep_nov_delta'])} |" for r in timing['groups'])
    content = f"""# 高历史购买额客户复核：11月下降是否意味着持续流失？

**结论先收窄：这份名单的11月下降成立，但不能把组总额的宽窗增长解释为客户已经恢复或提前采购。** 固定{cohort['fixed_customers']}人的11月购买金额同比{comparisons['november']['revenue_yoy']:+.2%}；扩大到9—11月后为{comparisons['sep_nov']['revenue_yoy']:+.2%}。配对到同一客户后，11月减额者中仍有{timing['november_decliners_still_down_in_sep_nov']}人在9—11月同比减少；这与组总额增长可以同时成立。

数据口径为 UCI Online Retail II 去重后、正向商品购买发票，单位英镑。按2010年1—10月购买额从{cohort['eligible_prior_customers']}名可识别客户中固定前1%（向上取整38人）；名单不使用两年11月结果。它表示历史购买金额较高，不能直接等同于批发商、利润高、可触达或应获补贴。

## 1. 时间窗口先改变问题的范围

| 比较窗口 | 2010年购买金额 | 2011年购买金额 | 同比 | 基期窗口与名单选择期重叠 |
| --- | ---: | ---: | ---: | :---: |
{window_table}

**只有11月对11月的比较，名单才完全先于两侧结果。** 1—11月、9—11月和10—11月的2010年基期均包含名单选择期内交易；因名单按高消费选取，存在选择偏差和均值回归风险。这些窗口是交易账簿上的描述性稳健性检查，不能作为该名单具有长期因果效应的检验。宽窗增长也不能证明客户没有流失或未来会恢复。未使用不完整的2011年12月做恢复判断。

### 宽窗增长来自同一客户的前月增额，还是其他客户抵消？

11月金额下降的{timing['november_declining_customers']}人合计减额{fmt(timing['november_gross_loss_in_declining_customers'])}。逐人计算 `min(max(9—10月同比增额, 0), max(11月同比减额, 0))`，可被同一客户前两个月增额覆盖的金额只有 **{fmt(timing['same_customer_positive_sep_oct_increase_capped_at_november_loss'])}（占11月全部减额{timing['arithmetic_coverage_of_november_gross_losses']:.2%}）**。

其中{timing['november_decliners_still_down_in_sep_nov']}人的9—11月仍减少，减额合计{fmt(timing['november_decliners_still_down_sep_nov_gross_loss'])}。全部11月减额客户合起来，9—11月净变化为{fmt(timing['all_november_decliners_sep_nov_net_delta'])}；其余客户为{fmt(timing['other_customers_sep_nov_net_delta'])}，合并后才得到整组的{fmt(timing['all_customers_sep_nov_net_delta'])}。

| 同一客户的双窗口状态 | 人数 | 11月变化 | 9—10月变化 | 9—11月变化 |
| --- | ---: | ---: | ---: | ---: |
{paired_table}

由此，组总额的正增长掩盖了客户之间的大额抵消，不能据此关闭对18名跨窗口减额者的调查。这里“跨窗口减额”仍是账簿描述，不等于需求永久流失。

**该覆盖量只是同客户的金额匹配上限，不是“提前采购金额”。** 没有同一订单的计划日期和实际日期，也没有反事实正常采购额。名单选择又与2010年9—10月交易重叠；同客配对可以排除“纯粹来自其他客户增加”的部分算术解释，不能识别采购时点移动。完整的四类聚合见 [same_customer_window_bridge.csv](same_customer_window_bridge.csv)。

业务问题应拆成“哪些同客减额跨窗口仍存在”与“哪些变化可能与订单发生时间或组间抵消有关”，分别核对可比客户的预定交货、下单频率、跨月订单与商品可售情况。

## 2. 把客户缺席和持续购买者减额拆开

| 固定名单内的11月活动状态 | 人数 | 2010年金额 | 2011年金额 | 金额变化 |
| --- | ---: | ---: | ---: | ---: |
{group_table}

“仅基期11月购买”表示该月没有正向发票，不能称永久流失；“仅当前11月购买”也不是新获客，因为所有38人在名单选择期已有购买。两年11月均未购买的人仍留在38人名单内，避免通过事后活跃状态缩小样本。各组金额变化可相加得到整组净变化。

## 3. 下降在哪个交易环节？

按 **金额 = 活跃人数 × 人均订单数 × 每单件数 × 每件成交金额**，对4个因子的24种替换顺序取均值，精确分摊交互项。

| 因子 | 固定38人金额变化分量 | 两年11月均购买者金额变化分量 |
| --- | ---: | ---: |
{factor_table}

左列总和为{fmt(factors['all_fixed_customers']['revenue_delta'])}；右列总和为{fmt(factors['both_novembers_only']['revenue_delta'])}。右列人数固定，更接近持续购买者自身交易结构的描述。两个总体不能相加；本节也不能与上一节活动分组分量相加。每件成交金额同时包含商品和客户组合变化，不是可比SKU价格指数，也不支持推算促销弹性。

## 4. 先看增减抵消，再讨论少数客户集中度

38人中，{losses['declining_customers']}人金额减少、{losses['increasing_customers']}人增加、{losses['unchanged_customers']}人持平（含两期均无购买）。减额绝对值合计 **{fmt(losses['gross_loss'])}**，被增额 **{fmt(losses['gross_gain'])}** 抵消，最后净下降 **{fmt(losses['net_decline'])}**。

| 累计去标识减额汇总 | 减额绝对值 | 占全部减额 | 占净下降 |
| --- | ---: | ---: | ---: |
{concentration_table}

这三个集合彼此嵌套，不能相加；占净下降超过100%是增额抵消造成的，并不表示统计错误。此表仅定位核查优先级，不证明少数客户造成可干预的业务损失，也不等于可用优惠券追回的收入。没有导出客户标识、客户级数据或排行榜。

## 5. 大订单是否减少？使用比较期前固定的门槛

用同一名单2010年1—10月订单金额的90分位数 **{fmt(s['order_size_threshold']['prior_period_invoice_p90_gbp'])}** 固定“大订单”核查门槛；两年11月使用同一门槛。它是相对这个历史高购买额群体的订单规模标签，不是业务预设的批发订单定义。

| 订单组 | 订单数（2010→2011） | 金额变化 | 单均金额（2010→2011） |
| --- | ---: | ---: | ---: |
{bins_table}

整体订单金额中位数 {fmt(d0['order_revenue_median'])} → {fmt(d1['order_revenue_median'])}；90分位数 {fmt(d0['order_revenue_p90'])} → {fmt(d1['order_revenue_p90'])}。固定38人中、包含零购买者的客户金额中位数 {fmt(d0['customer_spend_median_all_fixed_customers'])} → {fmt(d1['customer_spend_median_all_fixed_customers'])}。订单分位数来自当期所有发票，客户分位数来自完整固定名单，二者分母不同。

达到门槛的订单变化与较小订单变化必须同时阅读。单张大订单可受备货周期、合同、商品可售、交期或记账日期影响；当前公开账簿不能区分这些竞争解释，也不能判断一张订单拆成多张发票的情况。

这里也纠正一个容易过早下的结论：**固定门槛以上订单数没有减少（20→21），减少的是该组金额和单均金额；整体订单中位数和90分位数反而上升。** 因而不能概括为所有订单变小，也不能只凭总额说大订单“数量流失”。更具体的核查对象是上端订单规模、合同与商品组合，同时保留小订单数量减少这一条线索。

## 下一步业务动作和停止规则

1. **先核订单日期及履约链路。** 将11月减少与9—10月增加并列，向业务核对订单创建、承诺发货、实际发货、取消和跨月确认字段；只有这些记录能区分需求变化与时间移动。
2. **聚焦大额订单的可售与交付约束。** 对固定门槛上下两组核对商品缺货、最小起订量、运费门槛、合同价格和客服反馈。若主要是缺货/交付问题，应先修供给或履约，补贴不是这份账簿支持的第一动作。
3. **设置进入干预评估的证据门槛。** 只有确认交易日期可比、身份稳定且存在可干预阻力后，才针对明确可触达群体设计随机实验；预先约定平台或商家贡献口径，核对券成本是否已冲减收入，按增量净贡献决策，避免重复扣除成本。口径约定见 [实验设计](../../docs/EXPERIMENT_DESIGN.md)。当前数据没有这些字段，无法给出补贴预算或因果ROI。

本页不会把宽窗增长称为“恢复”，也不会把11月缺席称为“流失”。所有归因是会计分解；相同交易的不同切片不构成独立因果证据。

复现：`.venv312/Scripts/python.exe -m retail.customer_drilldown --db data/processed/retail.duckdb`。机器可读结果见 [summary.json](summary.json)、[checks.json](checks.json)；代码见 [customer_drilldown.py](../../retail/customer_drilldown.py)。
"""
    (output / "FINDINGS.md").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/retail.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("reports/customer_drilldown"))
    args = parser.parse_args()
    run(args.db, args.output)


if __name__ == "__main__":
    main()
