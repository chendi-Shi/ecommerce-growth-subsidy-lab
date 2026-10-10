-- Executed by infra/run_hive.py against Apache Hive 4.2.1 HiveServer2.
-- Source: real Hillstrom 64,000-customer, three-arm randomized email trial.
-- customer_id is the stable source-row identifier, not an original customer ID.
-- The outcome window is 14 days; spend is observed USD revenue, not profit.
SET hive.execution.engine=tez;
SET tez.local.mode=true;
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.max.dynamic.partitions=100;
SET hive.exec.max.dynamic.partitions.pernode=100;
SET hive.vectorized.execution.enabled=true;
SET hive.cbo.enable=true;

CREATE DATABASE IF NOT EXISTS growth_lab;
USE growth_lab;

CREATE EXTERNAL TABLE IF NOT EXISTS ods_campaign (
  customer_id STRING, recency STRING, history STRING,
  mens STRING, womens STRING, newbie STRING,
  zip_code STRING, channel STRING, treatment STRING,
  visit STRING, conversion STRING, spend STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"', 'escapeChar'='\\')
STORED AS TEXTFILE
LOCATION '${hivevar:input_dir}'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE TABLE IF NOT EXISTS dwd_campaign (
  customer_id STRING, recency INT, history DOUBLE,
  mens INT, womens INT, newbie INT,
  zip_code STRING, channel STRING,
  visit INT, conversion INT, spend DOUBLE
)
PARTITIONED BY (treatment INT)
STORED AS ORC
TBLPROPERTIES ('orc.compress'='SNAPPY');

INSERT OVERWRITE TABLE dwd_campaign PARTITION (treatment)
SELECT customer_id, CAST(recency AS INT), CAST(history AS DOUBLE),
       CAST(mens AS INT), CAST(womens AS INT), CAST(newbie AS INT),
       zip_code, channel, CAST(visit AS INT), CAST(conversion AS INT),
       CAST(spend AS DOUBLE), CAST(treatment AS INT)
FROM ods_campaign;

-- Raw-field validation occurs before load in the runner. Do not silently drop
-- invalid outcomes: arm counts and totals must reconcile to the original CSV.
CREATE TABLE IF NOT EXISTS ads_campaign_arm (
  treatment INT, users BIGINT, visitors BIGINT, buyers BIGINT,
  total_spend DOUBLE, visit_rate DOUBLE, conversion_rate DOUBLE,
  spend_per_user DOUBLE, spend_variance DOUBLE
)
STORED AS ORC;
INSERT OVERWRITE TABLE ads_campaign_arm
SELECT treatment, COUNT(*) AS users,
       SUM(visit) AS visitors, SUM(conversion) AS buyers,
       SUM(spend) AS total_spend, AVG(visit) AS visit_rate,
       AVG(conversion) AS conversion_rate, AVG(spend) AS spend_per_user,
       VAR_SAMP(spend) AS spend_variance
FROM dwd_campaign
GROUP BY treatment;

-- All segment dimensions below precede randomization. Merchandise flags are
-- historical product preferences; channel is historical Phone/Web/Multichannel.
CREATE TABLE IF NOT EXISTS ads_campaign_segment (
  treatment INT, channel STRING, merchandise_preference STRING,
  recency_band STRING, users BIGINT, buyers BIGINT,
  spend_per_user DOUBLE, conversion_rate DOUBLE
)
STORED AS ORC;
INSERT OVERWRITE TABLE ads_campaign_segment
SELECT treatment, channel,
       CASE WHEN mens=1 AND womens=1 THEN 'both'
            WHEN mens=1 THEN 'mens_only'
            WHEN womens=1 THEN 'womens_only' ELSE 'neither' END,
       CASE WHEN recency<=3 THEN '01-03_months'
            WHEN recency<=6 THEN '04-06_months' ELSE '07-12_months' END,
       COUNT(*), SUM(conversion), AVG(spend), AVG(conversion)
FROM dwd_campaign
GROUP BY treatment, channel,
       CASE WHEN mens=1 AND womens=1 THEN 'both'
            WHEN mens=1 THEN 'mens_only'
            WHEN womens=1 THEN 'womens_only' ELSE 'neither' END,
       CASE WHEN recency<=3 THEN '01-03_months'
            WHEN recency<=6 THEN '04-06_months' ELSE '07-12_months' END;

CREATE TABLE IF NOT EXISTS ads_campaign_itt (
  treatment INT, treatment_users BIGINT, control_users BIGINT,
  visit_delta DOUBLE, conversion_delta DOUBLE, spend_delta DOUBLE,
  spend_standard_error DOUBLE
)
STORED AS ORC;
INSERT OVERWRITE TABLE ads_campaign_itt
SELECT a.treatment, a.users, c.users,
       a.visit_rate-c.visit_rate, a.conversion_rate-c.conversion_rate,
       a.spend_per_user-c.spend_per_user,
       SQRT(a.spend_variance/a.users+c.spend_variance/c.users)
FROM ads_campaign_arm a
CROSS JOIN ads_campaign_arm c
WHERE a.treatment IN (1,2) AND c.treatment=0;

ANALYZE TABLE dwd_campaign PARTITION (treatment) COMPUTE STATISTICS;
-- Captured in reports/hive/explain.txt; treatment predicate should prune ORC
-- partitions. This is a real engine plan, not a translated DuckDB query.
EXPLAIN FORMATTED
SELECT zip_code, COUNT(*) AS users, AVG(spend) AS spend_per_user
FROM dwd_campaign
WHERE treatment=1 AND channel='Web'
GROUP BY zip_code;
