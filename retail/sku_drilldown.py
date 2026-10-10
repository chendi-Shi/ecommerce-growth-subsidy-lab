"""Falsifiable follow-up of three SKUs already selected in the earlier memo.

Read the canonical transaction database without modifying it. Only aggregate
tables leave this module: neither invoice IDs nor customer IDs are exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SKUS = ("22423", "22578", "16014")
PERIODS = ("2010-11", "2011-11")
GROUP_KEYS = ["stock_code", "observed_month"]
ORDER_KEYS = GROUP_KEYS + ["customer_id", "invoice"]


def aggregate_orders(lines: pd.DataFrame) -> pd.DataFrame:
    """The existing retail_orders grain is (customer_id, invoice)."""
    return lines.groupby(ORDER_KEYS, as_index=False).agg(
        units=("quantity", "sum"), revenue=("line_revenue", "sum"))


def summarize_skus(lines: pd.DataFrame) -> pd.DataFrame:
    orders = aggregate_orders(lines)
    buyers = orders.groupby(GROUP_KEYS + ["customer_id"], as_index=False).agg(
        units=("units", "sum"), revenue=("revenue", "sum"))
    metrics = orders.groupby(GROUP_KEYS, as_index=False).agg(
        invoices=("invoice", "size"), buyers=("customer_id", "nunique"),
        units=("units", "sum"), revenue=("revenue", "sum"),
        maximum_invoice_units=("units", "max"))
    maxima = buyers.groupby(GROUP_KEYS, as_index=False).agg(
        maximum_buyer_units=("units", "max"))
    metrics = metrics.merge(maxima, on=GROUP_KEYS, validate="one_to_one")
    metrics["paid_value_per_unit"] = metrics.revenue / metrics.units
    metrics["maximum_invoice_unit_share"] = metrics.maximum_invoice_units / metrics.units
    metrics["maximum_buyer_unit_share"] = metrics.maximum_buyer_units / metrics.units
    return metrics.sort_values(GROUP_KEYS).reset_index(drop=True)


def summarize_price_bands(lines: pd.DataFrame) -> pd.DataFrame:
    """Use source unit_price values, not bins of the weighted SKU average.

    A buyer or an invoice may occur in several price bands. Their counts are
    deliberately not treated as additive across bands.
    """
    keys = GROUP_KEYS + ["unit_price"]
    bands = lines.groupby(keys, as_index=False).agg(
        source_lines=("quantity", "size"), buyers=("customer_id", "nunique"),
        units=("quantity", "sum"), revenue=("line_revenue", "sum"))
    invoice_counts = lines.drop_duplicates(keys + ["customer_id", "invoice"]).groupby(
        keys, as_index=False).size().rename(columns={"size": "invoices"})
    bands = bands.merge(invoice_counts, on=keys, validate="one_to_one")
    totals = summarize_skus(lines)[GROUP_KEYS + ["units", "revenue", "buyers"]].rename(
        columns={"units": "sku_units", "revenue": "sku_revenue", "buyers": "sku_buyers"})
    bands = bands.merge(totals, on=GROUP_KEYS, validate="many_to_one")
    bands["unit_share"] = bands.units / bands.sku_units
    bands["revenue_share"] = bands.revenue / bands.sku_revenue
    bands["buyer_reach"] = bands.buyers / bands.sku_buyers
    return bands.sort_values(keys).reset_index(drop=True)


def customer_membership_bridge(lines: pd.DataFrame) -> pd.DataFrame:
    """Partition SKU buyers by membership in the two fixed November windows."""
    orders = aggregate_orders(lines)
    customer_period = orders.groupby(GROUP_KEYS + ["customer_id"], as_index=False).agg(
        invoices=("invoice", "size"), units=("units", "sum"), revenue=("revenue", "sum"))
    wide = []
    for idx, period in enumerate(PERIODS):
        one = customer_period.loc[customer_period.observed_month == period].drop(columns="observed_month")
        wide.append(one.rename(columns={col: f"{col}_{idx}" for col in ("invoices", "units", "revenue")}))
    paired = wide[0].merge(wide[1], on=["stock_code", "customer_id"], how="outer", validate="one_to_one")
    measures = [f"{col}_{t}" for col in ("invoices", "units", "revenue") for t in (0, 1)]
    paired[measures] = paired[measures].fillna(0)
    paired["membership"] = np.select(
        [(paired.invoices_0 > 0) & (paired.invoices_1 > 0), paired.invoices_0 > 0],
        ["common_sku_buyers", "base_only_sku_buyers"], default="current_only_sku_buyers")
    result = paired.groupby(["stock_code", "membership"], as_index=False).agg(
        buyers=("customer_id", "size"), **{col: (col, "sum") for col in measures})
    for col in ("invoices", "units", "revenue"):
        result[f"{col}_delta"] = result[f"{col}_1"] - result[f"{col}_0"]
    return result.sort_values(["stock_code", "membership"]).reset_index(drop=True)


def reconcile(metrics: pd.DataFrame, bands: pd.DataFrame, bridge: pd.DataFrame) -> dict:
    checks: dict[str, float] = {}
    for row in metrics.itertuples():
        selected_bands = bands.loc[(bands.stock_code == row.stock_code) & (bands.observed_month == row.observed_month)]
        selected_bridge = bridge.loc[bridge.stock_code == row.stock_code]
        t = PERIODS.index(row.observed_month)
        checks[f"{row.stock_code}/{row.observed_month}/price_units"] = float(selected_bands.units.sum() - row.units)
        checks[f"{row.stock_code}/{row.observed_month}/price_revenue"] = float(selected_bands.revenue.sum() - row.revenue)
        for col in ("units", "revenue", "invoices"):
            checks[f"{row.stock_code}/{row.observed_month}/bridge_{col}"] = float(selected_bridge[f"{col}_{t}"].sum() - getattr(row, col))
        active = selected_bridge.loc[selected_bridge[f"invoices_{t}"] > 0, "buyers"].sum()
        checks[f"{row.stock_code}/{row.observed_month}/bridge_buyers"] = float(active - row.buyers)
        checks[f"{row.stock_code}/{row.observed_month}/paid_value"] = float(row.paid_value_per_unit * row.units - row.revenue)
    return {"tolerance": 1e-6, "errors": checks,
            "maximum_absolute_error": max(map(abs, checks.values()), default=0.0),
            "passed": all(abs(value) <= 1e-6 for value in checks.values())}


def write_findings(output: Path, metrics: pd.DataFrame, bands: pd.DataFrame, bridge: pd.DataFrame):
    def metric(sku, period):
        return metrics.loc[(metrics.stock_code == sku) & (metrics.observed_month == period)].iloc[0]

    stars = metric("22578", "2011-11")
    stars0 = metric("22578", "2010-11")
    scissors = metric("16014", "2011-11")
    cake0, cake1 = metric("22423", "2010-11"), metric("22423", "2011-11")
    low = bands.loc[(bands.stock_code == "22578") & (bands.observed_month == "2011-11") & np.isclose(bands.unit_price, .29)].iloc[0]
    scissor_low = bands.loc[(bands.stock_code == "16014") & (bands.observed_month == "2011-11") & np.isclose(bands.unit_price, .32)].iloc[0]
    scissor_other = bands.loc[(bands.stock_code == "16014") & (bands.observed_month == "2011-11") & np.isclose(bands.unit_price, .42)].iloc[0]
    cake_bridge = bridge.loc[bridge.stock_code == "22423"].set_index("membership")
    table = "\n".join(
        f"|{r.stock_code}|{r.observed_month}|£{r.revenue:,.2f}|{r.units:,.0f}|{r.invoices}|{r.buyers}|{r.paid_value_per_unit:.5f}|{r.maximum_invoice_unit_share:.2%}|{r.maximum_buyer_unit_share:.2%}|"
        for r in metrics.itertuples())
    text = f"""# SKU下钻：大单集中、成交价格档位与购买客户减少

**三个商品对应三条不同调查路径。** 小剪刀的件数增长高度集中在一张发票；圣诞装饰的低成交价格档位广泛覆盖客户；蛋糕架的销量下降伴随购买客户数和发票数下降。这进一步反对把所有商品归为统一的“用户价值下降”，但没有识别任何促销或价格的因果效果。

固定上一版[经营诊断备忘录](../retail_diagnosis/DECISION_MEMO.md)已展示的三个SKU：22423、22578、16014，没有在这一步重新筛选商品。两期均为完整11月（2010、2011）；仅保留规范去重账簿中有客户ID、正数量、正单价、非退货的商品行。金额为源GBP。

|SKU|月份|商品金额|件数|发票数|购买客户数|加权成交金额/件|最大单张发票件数占比|最大客户件数占比|
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table}

## 1. 16014小剪刀：先查集中大单，不能当作广泛偏好变化

2011年11月共{scissors.units:,.0f}件，其中最大单张发票{scissors.maximum_invoice_units:,.0f}件，占{scissors.maximum_invoice_unit_share:.2%}；最大客户亦占{scissors.maximum_buyer_unit_share:.2%}。源`unit_price=£0.32`的{int(scissor_low.source_lines)}条商品行共{scissor_low.units:,.0f}件，另{scissor_other.units:,.0f}件仍以£0.42成交，见[原始成交单价档位](price_bands.csv)。

因此应优先核查批量采购、特殊订单条款及订单履约，而不是把该SKU件数增长解释成大量普通用户转向低价商品。最大订单份额是本月该SKU全部件数的分母；不能外推到全店增长，也不表示该订单一定异常或应被删除。

## 2. 22578星形圣诞装饰：低成交价格跨客户出现，不能只归于一张大单

基期{stars0.units:,.0f}件全部记录为`unit_price=£0.85`。当前期{low.units:,.0f}/{stars.units:,.0f}件（{low.unit_share:.2%}）记录为£0.29，涉及{int(low.buyers)}名客户，占该SKU当期{int(stars.buyers)}名买家的{low.buyer_reach:.2%}。最大单张发票仅占{stars.maximum_invoice_unit_share:.2%}，最大客户占{stars.maximum_buyer_unit_share:.2%}。

“一个客户或一张大单拉低均价”不足以解释该SKU现象。下一步应查活动价生效范围、库存清理、客户条款或SKU定义变化；先核对毛利和商品属性，再讨论是否复制促销。£0.29是交易记录字段，不是已验证的挂牌价；买家和件数上升不证明降价引发需求增长，更不能由两期记录估计价格弹性。

## 3. 22423三层蛋糕架：购买覆盖收缩，进一步拆客户集合

购买客户从{int(cake0.buyers)}减至{int(cake1.buyers)}，发票从{int(cake0.invoices)}减至{int(cake1.invoices)}，件数从{cake0.units:,.0f}减至{cake1.units:,.0f}，加权成交金额/件由£{cake0.paid_value_per_unit:.2f}升至£{cake1.paid_value_per_unit:.2f}。最大单张发票占比由{cake0.maximum_invoice_unit_share:.2%}降至{cake1.maximum_invoice_unit_share:.2%}；这组指标并不呈现由当前某一张大单支配的结构。

|该SKU购买客户集合|客户数|基期金额|当期金额|金额变化|
|---|---:|---:|---:|---:|
"""
    labels = {"common_sku_buyers": "两期均购买该SKU", "base_only_sku_buyers": "仅基期购买该SKU", "current_only_sku_buyers": "仅当期购买该SKU"}
    for membership, row in cake_bridge.iterrows():
        text += f"|{labels[membership]}|{int(row.buyers)}|£{row.revenue_0:,.2f}|£{row.revenue_1:,.2f}|£{row.revenue_delta:+,.2f}|\n"
    text += """
客户集合桥接说明销量缺口来自哪些购买集合，供商品运营进一步核查替代品、商品可得性和采购周期。仅基期购买该SKU不表示该客户离开了店铺；当前期才买该SKU也不表示平台新客。没有曝光、库存、活动和反馈记录，不能区分需求变化与供给约束。

## 口径与行动边界

* 发票口径为`(customer_id, invoice)`，与仓库`retail_orders`一致；同一发票中多条相同SKU商品行先聚合。发票不是一次浏览、一个购物会话或一个采购合同。
* “加权成交金额/件”是`sum(line_revenue)/sum(quantity)`；价格档位直接按源`unit_price`分组，两者不是同一个指标。档位没有按平均价格二次分箱。
* 每个价格档位的买家数是该档位去重客户数。同一个客户、发票可能出现在多个档位，**买家数与发票数不能跨价格档位直接相加**；件数与金额可以按SKU×月份对账。`buyer_reach`可跨档位合计超过100%。
* SKU客户桥接在两个月购买该SKU的客户并集上划分三个互斥集合。对每个SKU，金额、件数及发票分录精确相加；不同SKU的客户与发票可能重叠，不能跨SKU相加为全店人数或订单数。
* 三个SKU是上一轮分析挑出的回顾性例子，不是预注册总体样本；不得把本结果推广到所有商品，也不报告选择后显著性。

所有导出均为聚合结果，不含客户ID或发票ID。[SKU指标](sku_metrics.csv)、[价格档位](price_bands.csv)、[客户集合桥接](customer_membership_bridge.csv)和[守恒校验](checks.json)均由同一只读模块生成。

复现：`python -m retail.sku_drilldown`。数据源、清洗及许可沿用[数据说明](../../docs/DATA_PROVENANCE.md)；该报告不含利润、优惠券成本、挂牌价、库存或曝光信息。
"""
    (output / "FINDINGS.md").write_text(text, encoding="utf-8")


def run(db: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db), read_only=True, config={"memory_limit": "256MB", "threads": "2"}) as con:
        lines = con.execute("""
            SELECT stock_code, observed_month, customer_id, invoice, quantity,
                   unit_price, line_revenue
            FROM retail_lines WHERE is_product AND NOT is_return
              AND customer_id IS NOT NULL
              AND observed_month IN (?, ?)
              AND stock_code IN (?, ?, ?)
        """, [*PERIODS, *SKUS]).fetchdf()
    if lines.empty or (lines.quantity <= 0).any() or (lines.unit_price <= 0).any():
        raise ValueError("Expected positive identified product purchase lines")
    if lines["invoice"].isna().any() or lines["customer_id"].isna().any():
        raise ValueError("Expected complete invoice/customer keys")
    if not np.isfinite(lines[["quantity", "unit_price", "line_revenue"]].to_numpy()).all():
        raise ValueError("Non-finite source transaction values")
    if set(map(tuple, lines[GROUP_KEYS].drop_duplicates().to_numpy())) != {(s, p) for s in SKUS for p in PERIODS}:
        raise ValueError("All three fixed SKUs must be observed in both fixed periods")
    metrics = summarize_skus(lines)
    bands = summarize_price_bands(lines)
    bridge = customer_membership_bridge(lines)
    checks = reconcile(metrics, bands, bridge)
    if not checks["passed"]:
        raise AssertionError(f"SKU aggregation failed reconciliation: {checks}")
    metrics.to_csv(output / "sku_metrics.csv", index=False)
    bands.to_csv(output / "price_bands.csv", index=False)
    bridge.to_csv(output / "customer_membership_bridge.csv", index=False)
    summary = {"fixed_skus": list(SKUS), "periods": list(PERIODS),
               "selection": "The three SKUs already named in the previous retail diagnosis memo; retrospective selection",
               "currency": "GBP", "source_lines": len(lines),
               "source_scope": "canonical positive identified product lines, excluding returns",
               "order_grain": ["customer_id", "invoice"],
               "price_band_buyer_and_invoice_counts_are_not_additive": True,
               "metrics": metrics.to_dict(orient="records"), "checks_passed": checks["passed"]}
    for name, payload in (("checks.json", checks), ("summary.json", summary)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_findings(output, metrics, bands, bridge)
    print(json.dumps({"output": str(output), "source_lines": len(lines),
                      "checks_passed": checks["passed"], "maximum_absolute_error": checks["maximum_absolute_error"]}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/processed/retail.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("reports/sku_drilldown"))
    args = parser.parse_args()
    run(args.db, args.output)


if __name__ == "__main__":
    main()
