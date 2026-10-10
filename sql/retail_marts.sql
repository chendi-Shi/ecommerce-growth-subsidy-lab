-- Executed by DuckDB 1.3.2. Money is source GBP, not a platform GMV/profit proxy.
-- A row is an invoice line, not a customer/order. Source has no unique line ID.
CREATE OR REPLACE TABLE canonical_lines AS
SELECT invoice, stock_code, description, quantity, invoice_date, unit_price,
       customer_id, country, min(source_sheet) AS source_sheet,
       count(*) AS source_occurrences
FROM raw_lines
GROUP BY invoice, stock_code, description, quantity, invoice_date, unit_price,
         customer_id, country;

CREATE OR REPLACE TABLE retail_lines AS
SELECT invoice, stock_code, description, quantity, invoice_date, unit_price,
       customer_id, country, source_sheet,
       CASE WHEN starts_with(invoice, 'C') AND quantity > 0
            THEN -quantity * unit_price ELSE quantity * unit_price END AS line_revenue,
       regexp_full_match(stock_code, '[0-9]{5}[A-Z]*') AS is_product,
       quantity < 0 OR starts_with(invoice, 'C') AS is_return,
       strftime(invoice_date, '%Y-%m') AS observed_month
FROM canonical_lines
WHERE invoice IS NOT NULL AND stock_code IS NOT NULL AND invoice_date IS NOT NULL
  AND quantity IS NOT NULL AND unit_price IS NOT NULL
  AND quantity <> 0 AND unit_price > 0;

CREATE OR REPLACE TABLE observation AS
SELECT min(CAST(invoice_date AS DATE)) AS starts_on,
       max(CAST(invoice_date AS DATE)) AS ends_on,
       CAST(date_trunc('month', max(invoice_date)) AS DATE) AS last_month,
       CAST(date_trunc('month', min(invoice_date)) + INTERVAL '3 months' AS DATE)
           AS cohort_warmup_end
FROM raw_lines;

-- Positive product invoices only; credit notes remain in retail_lines financial mart.
CREATE OR REPLACE TABLE retail_orders AS
SELECT customer_id, invoice, min(invoice_date) AS order_date,
       sum(line_revenue) AS order_revenue, sum(quantity) AS units,
       count(DISTINCT stock_code) AS distinct_products, min(country) AS country,
       strftime(min(invoice_date), '%Y-%m') AS observed_month
FROM retail_lines
WHERE is_product AND NOT is_return
GROUP BY customer_id, invoice;

CREATE OR REPLACE TABLE customer_history AS
SELECT customer_id, min(CAST(order_date AS DATE)) AS first_day,
       min(date_trunc('month', order_date)) AS first_month,
       max(CAST(order_date AS DATE)) AS last_day,
       count(*) AS observed_orders, sum(order_revenue) AS observed_purchase_revenue
FROM retail_orders WHERE customer_id IS NOT NULL GROUP BY customer_id;

CREATE OR REPLACE TABLE customer_month AS
WITH activity AS (
    SELECT customer_id, date_trunc('month', order_date) AS month,
           count(*) AS orders, sum(order_revenue) AS purchase_revenue
    FROM retail_orders WHERE customer_id IS NOT NULL GROUP BY customer_id, month
), lagged AS (
    SELECT *, lag(month) OVER (PARTITION BY customer_id ORDER BY month) AS previous_month
    FROM activity
)
SELECT a.*, c.first_month,
       CASE WHEN month = first_month THEN 'first_observed'
            WHEN date_diff('month', previous_month, month) = 1 THEN 'retained'
            ELSE 'reactivated' END AS buyer_state
FROM lagged a JOIN customer_history c USING (customer_id);

CREATE OR REPLACE TABLE monthly_growth AS
WITH known AS (
    SELECT month, count(*) AS active_buyers,
        count(*) FILTER (WHERE buyer_state = 'first_observed') AS first_observed_buyers,
        count(*) FILTER (WHERE buyer_state = 'retained') AS retained_buyers,
        count(*) FILTER (WHERE buyer_state = 'reactivated') AS reactivated_buyers,
        sum(orders) AS identified_orders, sum(purchase_revenue) AS identified_purchase_revenue
    FROM customer_month GROUP BY month
), money AS (
    SELECT date_trunc('month', invoice_date) AS month,
        coalesce(sum(line_revenue) FILTER (WHERE is_product AND NOT is_return),0) AS gross_product_sales,
        -coalesce(sum(line_revenue) FILTER (WHERE is_product AND is_return),0) AS product_return_value,
        coalesce(sum(line_revenue) FILTER (WHERE is_product),0) AS net_product_revenue,
        coalesce(sum(line_revenue) FILTER (WHERE is_product AND NOT is_return AND customer_id IS NULL),0)
            AS anonymous_purchase_revenue
    FROM retail_lines GROUP BY month
)
SELECT strftime(k.month, '%Y-%m') AS month, k.* EXCLUDE(month),
       retained_buyers + reactivated_buyers AS repeat_buyers,
       identified_orders * 1.0 / active_buyers AS orders_per_buyer,
       identified_purchase_revenue / identified_orders AS revenue_per_order,
       identified_purchase_revenue / active_buyers AS revenue_per_buyer,
       m.* EXCLUDE(month),
       k.month < o.last_month AS complete_month
FROM known k JOIN money m USING (month) CROSS JOIN observation o ORDER BY k.month;

-- Exact multiplicative decomposition: log(sales_t/sales_t-1) = log(buyers ratio)
-- + log(order frequency ratio) + log(order value ratio), same identified population.
CREATE OR REPLACE TABLE monthly_decomposition AS
WITH ratios AS (
  SELECT month, identified_purchase_revenue,
    ln(identified_purchase_revenue /
       lag(identified_purchase_revenue) OVER (ORDER BY month)) AS log_sales_change,
    ln(active_buyers * 1.0 / lag(active_buyers) OVER (ORDER BY month)) AS log_buyer_change,
    ln(orders_per_buyer / lag(orders_per_buyer) OVER (ORDER BY month)) AS log_frequency_change,
    ln(revenue_per_order / lag(revenue_per_order) OVER (ORDER BY month)) AS log_order_value_change
  FROM monthly_growth WHERE complete_month
)
SELECT *, log_sales_change - log_buyer_change - log_frequency_change - log_order_value_change
    AS reconciliation_error FROM ratios ORDER BY month;

CREATE OR REPLACE TABLE monthly_cohort_retention AS
WITH cohort_sizes AS (
    SELECT first_month, count(*) AS cohort_customers FROM customer_history
    CROSS JOIN observation WHERE first_day >= cohort_warmup_end GROUP BY first_month
), offsets AS (SELECT unnest(range(0, 22)) AS month_index)
SELECT strftime(s.first_month, '%Y-%m') AS cohort_month, offsets.month_index,
       s.cohort_customers, count(c.customer_id) AS active_customers,
       count(c.customer_id) * 1.0 / s.cohort_customers AS retention_rate
FROM cohort_sizes s CROSS JOIN offsets CROSS JOIN observation o
LEFT JOIN customer_month c ON c.first_month = s.first_month
  AND date_diff('month', c.first_month, c.month) = offsets.month_index
WHERE s.first_month + offsets.month_index * INTERVAL '1 month' < o.last_month
GROUP BY s.first_month, offsets.month_index, s.cohort_customers ORDER BY 1, 2;

-- Repurchase means another purchase on a strictly later calendar day: same-day
-- split invoices are not counted. Recent users without full follow-up are excluded.
CREATE OR REPLACE TABLE customer_repurchase AS
WITH next_dates AS (
  SELECT h.customer_id, h.first_day, min(CAST(r.order_date AS DATE)) AS next_day
  FROM customer_history h LEFT JOIN retail_orders r ON h.customer_id = r.customer_id
    AND CAST(r.order_date AS DATE) > h.first_day
  GROUP BY h.customer_id, h.first_day
), horizons AS (SELECT unnest([30, 60, 90]) AS horizon_days)
SELECT n.*, h.horizon_days, date_diff('day', first_day, next_day) AS days_to_repurchase,
       first_day + horizon_days * INTERVAL '1 day' <= ends_on AS mature,
       coalesce(next_day <= first_day + horizon_days * INTERVAL '1 day', false) AS repurchased,
       first_day >= cohort_warmup_end AS after_warmup,
       last_day(first_day) + horizon_days * INTERVAL '1 day' <= ends_on AS full_cohort_mature
FROM next_dates n CROSS JOIN horizons h CROSS JOIN observation;

CREATE OR REPLACE TABLE repurchase_by_cohort AS
SELECT strftime(first_day, '%Y-%m') AS cohort_month, horizon_days,
       count(*) AS eligible_customers,
       count(*) FILTER (WHERE repurchased) AS repurchased_customers,
       count(*) FILTER (WHERE repurchased) * 1.0 / count(*) AS repurchase_rate,
       bool_and(full_cohort_mature) AS full_cohort_mature
FROM customer_repurchase WHERE mature AND after_warmup
GROUP BY 1, 2 ORDER BY 1, 2;

-- Same 90-day-complete cohorts across windows permit valid cumulative comparison.
CREATE OR REPLACE TABLE repurchase_common_followup AS
SELECT horizon_days, count(*) AS eligible_customers,
       count(*) FILTER (WHERE repurchased) AS repurchased_customers,
       count(*) FILTER (WHERE repurchased) * 1.0 / count(*) AS repurchase_rate
FROM customer_repurchase CROSS JOIN observation
WHERE after_warmup AND last_day(first_day) + INTERVAL '90 days' <= ends_on
GROUP BY horizon_days ORDER BY horizon_days;

CREATE OR REPLACE TABLE country_metrics AS
SELECT country, count(*) AS orders, count(DISTINCT customer_id) AS identified_buyers,
       sum(order_revenue) AS purchase_revenue, avg(order_revenue) AS revenue_per_order,
       sum(order_revenue) FILTER (WHERE customer_id IS NULL) AS anonymous_purchase_revenue
FROM retail_orders GROUP BY country ORDER BY purchase_revenue DESC;

CREATE OR REPLACE TABLE price_band_metrics AS
SELECT CASE WHEN unit_price < 2 THEN 'low: < GBP2'
            WHEN unit_price < 5 THEN 'mid: GBP2-<5' ELSE 'high: >= GBP5' END AS price_band,
       count(*) AS purchase_lines, count(DISTINCT stock_code) AS products,
       count(DISTINCT customer_id) AS identified_buyers,
       sum(quantity) AS units, sum(line_revenue) AS purchase_revenue
FROM retail_lines WHERE is_product AND NOT is_return GROUP BY 1 ORDER BY 1;

-- Pre-period R/F rules; monetary profiles value but is not part of the label.
-- The segment outcomes are evaluated on a disjoint future
-- period (2011-09-01..2011-11-30); no holdout spend enters segment assignment.
CREATE OR REPLACE TABLE rfm_holdout AS
WITH train AS (
  SELECT customer_id,
    date_diff('day', max(CAST(order_date AS DATE)), DATE '2011-08-31') AS recency_days,
    count(*) AS frequency, sum(order_revenue) AS monetary,
    min(CAST(order_date AS DATE)) AS first_day
  FROM retail_orders WHERE customer_id IS NOT NULL AND order_date < TIMESTAMP '2011-09-01'
  GROUP BY customer_id
), scores AS (
  SELECT *, ntile(5) OVER (ORDER BY recency_days DESC, customer_id) AS r_score,
       ntile(5) OVER (ORDER BY frequency, customer_id) AS f_score
  FROM train
), segment AS (
 SELECT *, CASE WHEN r_score <= 2 THEN 'dormant'
       WHEN r_score >= 4 AND f_score >= 4 THEN 'loyal'
       WHEN r_score >= 4 AND f_score <= 2 THEN 'recent_low_frequency'
       ELSE 'regular' END AS segment
 FROM scores
), test AS (
 SELECT customer_id, count(*) AS future_orders, sum(order_revenue) AS future_revenue
 FROM retail_orders WHERE customer_id IS NOT NULL
   AND order_date >= TIMESTAMP '2011-09-01' AND order_date < TIMESTAMP '2011-12-01'
 GROUP BY customer_id
)
SELECT s.*, coalesce(t.future_orders, 0) AS future_orders,
       coalesce(t.future_revenue, 0) AS future_revenue
FROM segment s LEFT JOIN test t USING (customer_id);

CREATE OR REPLACE TABLE rfm_segment_validation AS
SELECT segment, count(*) AS customers,
    count(*) FILTER (WHERE future_orders > 0) AS future_buyers,
    count(*) FILTER (WHERE future_orders > 0) * 1.0 / count(*) AS future_purchase_rate,
    avg(future_revenue) AS future_revenue_per_customer,
    sum(future_revenue) AS future_purchase_revenue,
    avg(recency_days) AS prior_recency_days, avg(frequency) AS prior_frequency,
    avg(monetary) AS prior_revenue_per_customer
FROM rfm_holdout GROUP BY segment ORDER BY future_revenue_per_customer DESC;
