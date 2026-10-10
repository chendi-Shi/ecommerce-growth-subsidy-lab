# Actual Apache Hive verification

`sql/campaign_hive.sql` executes in the official `apache/hive:4.2.1` image.
HiveServer2 uses its embedded Derby metastore and local Tez execution. It builds
a CSV external ODS table, typed ORC DWD table partitioned by experimental arm,
and ORC ADS tables for arm statistics, pre-treatment segments and ITT differences.

With an already running Docker daemon:

```bash
python -m campaign.download
python infra/run_hive.py --input data/processed/hillstrom.csv --output reports/hive
```

The runner does not launch Docker Desktop or expose any host network ports.
Its Beeline process gets a container pseudo-terminal and Java 21 preview support:
the bundled JLine FFM provider in this Hive release otherwise fails before JDBC
connects. The server and metastore use the official image's startup defaults.
It validates the canonical source, waits for HiveServer2, executes through
Beeline/JDBC, checks ORC storage and the three partitions, and reconciles all
64,000 rows' arm counts, visitor/buyer counts, total spend and per-user means
against an independent Python CSV calculation. Monetary comparisons allow
`1e-6` absolute / `1e-10` relative floating-point tolerance; counts must match
exactly. Original Hillstrom customer IDs were not published, so `customer_id`
is a documented stable source-row identifier.

The `Real data and Apache Hive verification` GitHub Actions workflow performs
the same commands on an Ubuntu runner. It uploads `verification.json`, metric
CSVs, the engine's `EXPLAIN FORMATTED` output, storage/partition evidence and
server/Beeline logs, including on failure. A checked-in workflow alone does not
prove a successful run: use its completed run and generated artifact as evidence.

Optional retail engine check after the retail pipeline exports its genuine
50,000-row sequential CSV slice:

```bash
python infra/run_hive.py --retail-input data/processed/retail_hive_sample.csv
```

The retail exporter replaces CR/LF inside CSV descriptions with spaces because
Hive's OpenCSVSerde cannot parse multiline records. Original Excel/parquet
descriptions remain intact. The verifier builds a month-partitioned ORC DWD
table and checks per-month row counts, identifiable purchasing customers, net
product revenue, positive product revenue and returns against the input CSV.
Null customer IDs stay in monetary totals and are excluded from distinct-buyer
counts. This is sample-level engine validation, not full-dataset Hive execution.

This setup establishes reproducible Hive SQL execution and query plans. It does
not establish production distributed-cluster deployment or operating experience.

Sources: [Apache's Docker quickstart](https://hive.apache.org/development/quickstart/)
and the [original randomized experiment description](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).
