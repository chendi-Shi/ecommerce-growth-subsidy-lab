-- Optional genuine retail CSV reconciliation in the SAME Apache Hive runtime.
-- Sample is a sequential slice for engine verification, not a population sample.
-- CSV exporter must normalize CR/LF in descriptions: OpenCSVSerde cannot read
-- quoted multiline records. Raw Excel and parquet retain original descriptions.
SET hive.execution.engine=tez;
SET tez.local.mode=true;
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
USE growth_lab;
CREATE EXTERNAL TABLE IF NOT EXISTS ods_retail_sample (
  invoice STRING, stock_code STRING, description STRING,
  quantity STRING, invoice_date STRING, unit_price STRING,
  customer_id STRING, country STRING, source_sheet STRING,
  line_revenue STRING, is_product STRING, is_return STRING,
  observed_month STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"', 'escapeChar'='\\')
STORED AS TEXTFILE
LOCATION '${hivevar:retail_dir}'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE TABLE IF NOT EXISTS dwd_retail_sample (
  invoice STRING, stock_code STRING, description STRING,
  quantity DOUBLE, invoice_date TIMESTAMP, unit_price DOUBLE,
  customer_id STRING, country STRING, source_sheet STRING,
  line_revenue DOUBLE, is_product BOOLEAN, is_return BOOLEAN
)
PARTITIONED BY (observed_month STRING)
STORED AS ORC;

INSERT OVERWRITE TABLE dwd_retail_sample PARTITION (observed_month)
SELECT invoice, stock_code, description, CAST(quantity AS DOUBLE),
       CAST(invoice_date AS TIMESTAMP), CAST(unit_price AS DOUBLE),
       CASE WHEN customer_id='' THEN NULL ELSE customer_id END,
       country, source_sheet, CAST(line_revenue AS DOUBLE),
       CASE WHEN LOWER(is_product) IN ('true','1') THEN TRUE ELSE FALSE END,
       CASE WHEN LOWER(is_return) IN ('true','1') THEN TRUE ELSE FALSE END,
       observed_month
FROM ods_retail_sample;

CREATE TABLE IF NOT EXISTS ads_retail_sample_month (
  observed_month STRING, line_rows BIGINT, identified_buyers BIGINT,
  net_product_revenue DOUBLE, positive_product_revenue DOUBLE,
  return_product_revenue DOUBLE
)
STORED AS ORC;
INSERT OVERWRITE TABLE ads_retail_sample_month
SELECT observed_month, COUNT(*),
       COUNT(DISTINCT CASE WHEN is_product AND quantity>0 AND unit_price>0
                           THEN customer_id END),
       SUM(CASE WHEN is_product THEN line_revenue ELSE 0 END),
       SUM(CASE WHEN is_product AND quantity>0 AND unit_price>0
                THEN line_revenue ELSE 0 END),
       SUM(CASE WHEN is_product AND is_return THEN line_revenue ELSE 0 END)
FROM dwd_retail_sample
GROUP BY observed_month;

EXPLAIN FORMATTED
SELECT country, SUM(line_revenue)
FROM dwd_retail_sample
WHERE observed_month=(SELECT MIN(observed_month) FROM ads_retail_sample_month)
  AND is_product
GROUP BY country;
