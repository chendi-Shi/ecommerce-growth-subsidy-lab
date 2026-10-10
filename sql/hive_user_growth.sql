-- Hive 3.x template for genuine headerless Tianchi UserBehavior CSV.
-- Not executed by the Python demo. Requires a Hive/HDFS environment.
-- Five original fields only; no fabricated scene/cost/revenue columns.
SET hive.local.time.zone=Asia/Shanghai;
CREATE DATABASE IF NOT EXISTS ecommerce_growth;
USE ecommerce_growth;

CREATE EXTERNAL TABLE IF NOT EXISTS ods_behavior (
  user_id BIGINT, item_id BIGINT, category_id BIGINT,
  behavior STRING, event_ts BIGINT
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE LOCATION '/data/ecommerce/userbehavior';

CREATE TABLE IF NOT EXISTS dwd_behavior STORED AS ORC AS
SELECT DISTINCT user_id, item_id, category_id, behavior, event_ts,
  TO_DATE(FROM_UNIXTIME(event_ts)) AS dt
FROM ods_behavior
WHERE user_id IS NOT NULL AND item_id IS NOT NULL
  AND event_ts IS NOT NULL AND behavior IN ('pv','cart','fav','buy');
-- DISTINCT treats identical records as duplicates; document this assumption.

CREATE TABLE IF NOT EXISTS ads_daily_growth STORED AS ORC AS
SELECT dt, COUNT(DISTINCT user_id) AS dau,
  COUNT(DISTINCT CASE WHEN behavior='pv' THEN user_id END) AS browsing_users,
  COUNT(DISTINCT CASE WHEN behavior='cart' THEN user_id END) AS cart_users,
  COUNT(DISTINCT CASE WHEN behavior='buy' THEN user_id END) AS buyers,
  COUNT(DISTINCT CASE WHEN behavior='buy' THEN user_id END)*1.0 /
    COUNT(DISTINCT user_id) AS active_to_purchase_rate
FROM dwd_behavior GROUP BY dt;

-- Strict ordered funnel within the same user/item/day.
-- Take the first pv, first cart AFTER pv, then purchase AFTER cart.
CREATE TABLE IF NOT EXISTS ads_ordered_funnel STORED AS ORC AS
WITH views AS (
  SELECT user_id,item_id,dt,MIN(event_ts) AS view_ts
  FROM dwd_behavior WHERE behavior='pv' GROUP BY user_id,item_id,dt
), carts AS (
  SELECT v.user_id,v.item_id,v.dt,v.view_ts,MIN(c.event_ts) AS cart_ts
  FROM views v LEFT JOIN dwd_behavior c
    ON v.user_id=c.user_id AND v.item_id=c.item_id AND v.dt=c.dt
    AND c.behavior='cart' AND c.event_ts>v.view_ts
  GROUP BY v.user_id,v.item_id,v.dt,v.view_ts
), buys AS (
  SELECT c.user_id,c.item_id,c.dt,c.cart_ts,MIN(b.event_ts) AS buy_ts
  FROM carts c LEFT JOIN dwd_behavior b
    ON c.user_id=b.user_id AND c.item_id=b.item_id AND c.dt=b.dt
    AND b.behavior='buy' AND b.event_ts>c.cart_ts
  GROUP BY c.user_id,c.item_id,c.dt,c.cart_ts
)
SELECT dt,COUNT(*) AS viewed_user_item_pairs,
  SUM(CASE WHEN cart_ts IS NOT NULL THEN 1 ELSE 0 END) AS ordered_cart_pairs,
  SUM(CASE WHEN buy_ts IS NOT NULL THEN 1 ELSE 0 END) AS ordered_purchase_pairs
FROM buys GROUP BY dt;

-- First OBSERVED activity cohort, not platform registration/new user.
-- 7-day retention only for cohorts with a complete seven-day follow-up.
CREATE TABLE IF NOT EXISTS ads_cohort_retention STORED AS ORC AS
WITH active AS (SELECT DISTINCT user_id,dt FROM dwd_behavior),
cohorts AS (SELECT user_id,MIN(dt) AS cohort_dt FROM active GROUP BY user_id),
bounds AS (SELECT MAX(dt) AS last_dt FROM active),
eligible AS (
  SELECT c.* FROM cohorts c CROSS JOIN bounds b
  WHERE DATE_ADD(c.cohort_dt,7)<=b.last_dt
)
SELECT e.cohort_dt,COUNT(DISTINCT e.user_id) AS cohort_users,
  COUNT(DISTINCT CASE WHEN DATEDIFF(a.dt,e.cohort_dt)=1 THEN e.user_id END)*1.0/
    COUNT(DISTINCT e.user_id) AS d1_retention,
  COUNT(DISTINCT CASE WHEN DATEDIFF(a.dt,e.cohort_dt)=7 THEN e.user_id END)*1.0/
    COUNT(DISTINCT e.user_id) AS d7_retention
FROM eligible e LEFT JOIN active a ON e.user_id=a.user_id
GROUP BY e.cohort_dt;
