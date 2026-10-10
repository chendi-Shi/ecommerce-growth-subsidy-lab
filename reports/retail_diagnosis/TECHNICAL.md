# Retail diagnosis: technical contract

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
