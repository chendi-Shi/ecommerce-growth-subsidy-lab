"""Execute and reconcile REAL Apache Hive, never a substitute SQL engine.

python infra/run_hive.py --input data/processed/hillstrom.csv
An already-running Docker daemon is required. The runner does not start desktop
applications. The Apache image uses an embedded Derby metastore and local Tez;
this demonstrates Hive execution, not a production distributed Hadoop cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "apache/hive:4.2.1"
CAMPAIGN_COLUMNS = [
    "customer_id", "recency", "history", "mens", "womens", "newbie",
    "zip_code", "channel", "treatment", "visit", "conversion", "spend",
]
CAMPAIGN_METRICS = [
    "treatment", "users", "visitors", "buyers", "total_spend", "visit_rate",
    "conversion_rate", "spend_per_user",
]
RETAIL_COLUMNS = [
    "invoice", "stock_code", "description", "quantity", "invoice_date",
    "unit_price", "customer_id", "country", "source_sheet", "line_revenue",
    "is_product", "is_return", "observed_month",
]
RETAIL_METRICS = [
    "observed_month", "line_rows", "identified_buyers", "net_product_revenue",
    "positive_product_revenue", "return_product_revenue",
]


def read_rows(path: Path, columns: list[str]):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"{path}: expected exact ordered header {columns}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV record at line {line}")
            if any("\n" in value or "\r" in value for value in row.values()):
                raise ValueError(f"{path}: Hive OpenCSVSerde cannot read multiline records")
            yield row


def finite(value: str, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}")
    return result


def campaign_expected(path: Path) -> dict[str, dict]:
    arms = {str(a): {"users": 0, "visitors": 0, "buyers": 0, "spends": []}
            for a in (0, 1, 2)}
    identifiers = set()
    for row in read_rows(path, CAMPAIGN_COLUMNS):
        customer = row["customer_id"]
        if not customer or customer in identifiers:
            raise ValueError("Campaign source-row identifiers must be unique and nonempty")
        identifiers.add(customer)
        arm = str(int(row["treatment"]))
        if arm not in arms:
            raise ValueError("Campaign treatment must be 0, 1 or 2")
        visit, conversion = int(row["visit"]), int(row["conversion"])
        if visit not in (0, 1) or conversion not in (0, 1):
            raise ValueError("Campaign visit and conversion must be binary")
        spend = finite(row["spend"], "campaign spend")
        if spend < 0:
            raise ValueError("Campaign spend must be nonnegative")
        finite(row["history"], "campaign history")
        int(row["recency"])
        for name in ("mens", "womens", "newbie"):
            if int(row[name]) not in (0, 1):
                raise ValueError(f"Campaign {name} must be binary")
        current = arms[arm]
        current["users"] += 1
        current["visitors"] += visit
        current["buyers"] += conversion
        current["spends"].append(spend)
    for arm, stats in arms.items():
        n = stats["users"]
        if not n:
            raise ValueError(f"Campaign arm {arm} is empty")
        stats["total_spend"] = math.fsum(stats.pop("spends"))
        stats["visit_rate"] = stats["visitors"] / n
        stats["conversion_rate"] = stats["buyers"] / n
        stats["spend_per_user"] = stats["total_spend"] / n
    return arms


def as_bool(value: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid retail boolean {value!r}")


def retail_expected(path: Path) -> dict[str, dict]:
    months = {}
    for row in read_rows(path, RETAIL_COLUMNS):
        month = row["observed_month"]
        stats = months.setdefault(month, {
            "line_rows": 0, "buyers": set(), "net": [], "positive": [], "returns": [],
        })
        stats["line_rows"] += 1
        quantity = finite(row["quantity"], "retail quantity")
        price = finite(row["unit_price"], "retail unit_price")
        revenue = finite(row["line_revenue"], "retail line_revenue")
        product, returned = as_bool(row["is_product"]), as_bool(row["is_return"])
        if product:
            stats["net"].append(revenue)
            if quantity > 0 and price > 0 and not returned:
                stats["positive"].append(revenue)
                if row["customer_id"]:
                    stats["buyers"].add(row["customer_id"])
            if returned:
                stats["returns"].append(revenue)
    if not months:
        raise ValueError("Retail sample is empty")
    return {month: {
        "line_rows": stats["line_rows"],
        "identified_buyers": len(stats["buyers"]),
        "net_product_revenue": math.fsum(stats["net"]),
        "positive_product_revenue": math.fsum(stats["positive"]),
        "return_product_revenue": math.fsum(stats["returns"]),
    } for month, stats in months.items()}


def result_rows(output: str, columns: list[str]) -> list[dict[str, str]]:
    # Beeline's csv2 output may include informational lines before the header.
    parsed = list(csv.reader(io.StringIO(output)))
    for index, row in enumerate(parsed):
        # Hive can qualify JDBC column labels (table.column) by default.
        labels = [label.rsplit(".", 1)[-1] for label in row]
        if labels == columns:
            result = [dict(zip(columns, values)) for values in parsed[index + 1:]
                      if len(values) == len(columns)]
            if not result:
                raise ValueError("Hive result header found, but no metric rows")
            return result
    raise ValueError(f"Expected Hive result header missing: {columns}")


def reconcile(rows: list[dict], expected: dict, key: str,
              integer_metrics: set[str]) -> list[dict]:
    observed = {str(row[key]): row for row in rows}
    if len(observed) != len(rows) or set(observed) != set(expected):
        raise AssertionError(f"Hive grouping keys differ: {set(observed)} != {set(expected)}")
    checks = []
    for value, reference in expected.items():
        for metric, target in reference.items():
            raw = observed[value][metric]
            actual = int(raw) if metric in integer_metrics else float(raw)
            matched = actual == target if metric in integer_metrics else math.isclose(
                actual, target, rel_tol=1e-10, abs_tol=1e-6)
            checks.append(dict(group=value, metric=metric, python=target, hive=actual,
                               absolute_difference=abs(actual - target), passed=matched))
            if not matched:
                raise AssertionError(f"Hive mismatch {value}/{metric}: {actual} != {target}")
    return checks


class HiveRunner:
    def __init__(self, container: str, output: Path):
        self.container, self.output = container, output
        self.started = False
        self.stage: Path | None = None

    def command(self, args: list[str], timeout: int = 600):
        return subprocess.run(args, text=True, capture_output=True, timeout=timeout,
                              encoding="utf-8", errors="replace")

    def beeline(self, sql: str | None = None, file: str | None = None,
                timeout: int = 600):
        # Hive 4.2.1's bundled JLine FFM provider needs Java 21 preview support.
        # BeeLine explicitly requires a system terminal even for -e/-f, so give
        # the client a container PTY. This does not expose a desktop window.
        args = ["docker", "exec", "-t",
                "--env", "JAVA_TOOL_OPTIONS=--enable-preview --enable-native-access=ALL-UNNAMED",
                "--env", "TERM=dumb", self.container, "/opt/hive/bin/beeline",
                "-u", "jdbc:hive2://localhost:10000/default", "-n", "hive",
                "--silent=true", "--showHeader=true", "--outputformat=csv2",
                "--color=false", "--force=false"]
        if file:
            # This release's JLine -f path can echo/truncate the script and
            # return zero without sending statements to HiveServer2. Read the
            # exact staged SQL ourselves and use the working -e batch path.
            if self.stage is None:
                raise RuntimeError("Hive SQL staging directory is unavailable")
            content = (self.stage / Path(file).name).read_text(encoding="utf-8")
            statements = " ".join(line.strip() for line in content.splitlines()
                                  if line.strip() and not line.lstrip().startswith("--"))
            args += ["--hivevar", "input_dir=file:///project/input/ods",
                     "--hivevar", "retail_dir=file:///project/input/retail", "-e", statements]
        else:
            args += ["-e", sql or "SELECT 1"]
        return self.command(args, timeout=timeout)

    def save_query(self, name: str, sql: str | None = None,
                   file: str | None = None, timeout: int = 600) -> str:
        result = self.beeline(sql, file, timeout)
        (self.output / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (self.output / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Hive {name} failed ({result.returncode}); inspect saved stderr")
        combined = result.stdout + result.stderr
        if "FAILED:" in combined or "Error:" in combined:
            raise RuntimeError(f"Hive {name} reported an error; inspect saved logs")
        return result.stdout

    def start(self, stage: Path):
        self.stage = stage
        result = self.command([
            "docker", "run", "-d", "--name", self.container, "--memory=5g", "--cpus=2",
            "--env", "SERVICE_NAME=hiveserver2",
            "--mount", f"type=bind,source={stage},target=/project/input,readonly",
            IMAGE,
        ], timeout=300)
        (self.output / "docker_start.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Cannot start Apache Hive container; inspect docker_start.txt")
        self.started = True
        deadline = time.monotonic() + 420
        attempt = 0
        with (self.output / "readiness.log").open("w", encoding="utf-8") as log:
            while time.monotonic() < deadline:
                attempt += 1
                try:
                    ready = self.beeline("SELECT 1 AS hive_ready", timeout=40)
                    log.write(f"Attempt {attempt}\n{ready.stdout}\n{ready.stderr}\n")
                    log.flush()
                    if ready.returncode == 0 and "hive_ready" in ready.stdout:
                        print(f"Apache HiveServer2 ready after {attempt} probes", flush=True)
                        return
                    if any(message in ready.stdout + ready.stderr for message in (
                            "Unable to create a terminal", "UnsupportedClassVersionError")):
                        raise RuntimeError("Beeline client JVM/terminal initialization failed; inspect readiness.log")
                except subprocess.TimeoutExpired:
                    log.write(f"Attempt {attempt}: probe timed out\n")
                    log.flush()
                time.sleep(5)
        raise TimeoutError("HiveServer2 did not become ready within 420 seconds")

    def finish(self, keep: bool):
        if self.started:
            logs = self.command(["docker", "logs", self.container], timeout=30)
            (self.output / "hiveserver2.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
            metadata = self.command(["docker", "inspect", self.container], timeout=30)
            (self.output / "container_inspect.json").write_text(metadata.stdout, encoding="utf-8")
            if not keep:
                self.command(["docker", "rm", "-f", self.container], timeout=30)


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/hillstrom.csv")
    parser.add_argument("--retail-input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hive")
    parser.add_argument("--keep-container", action="store_true", help="Keep container for debugging")
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    expected = campaign_expected(args.input)
    retail_reference = retail_expected(args.retail_input.resolve()) if args.retail_input else None
    args.output.mkdir(parents=True, exist_ok=True)
    source_rows = sum(arm["users"] for arm in expected.values())
    if source_rows != 64_000:
        raise ValueError(f"Expected the full 64,000-row Hillstrom release, got {source_rows}")
    stage = args.output / f"staging-{uuid.uuid4().hex[:10]}"
    (stage / "ods").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, stage / "ods/hillstrom.csv")
    shutil.copyfile(ROOT / "sql/campaign_hive.sql", stage / "campaign_hive.sql")
    if args.retail_input:
        (stage / "retail").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.retail_input, stage / "retail/sample.csv")
        shutil.copyfile(ROOT / "infra/retail_hive.sql", stage / "retail_hive.sql")
    # Do not expose JDBC ports: all clients use docker exec inside this container.
    runner = HiveRunner(f"growth-hive-{uuid.uuid4().hex[:10]}", args.output)
    report = dict(engine="Apache Hive", image=IMAGE, status="running",
                  execution="HiveServer2 / embedded Derby / local Tez / ORC",
                  source_sha256=sha256(args.input), source_rows=sum(a["users"] for a in expected.values()),
                  distributed_cluster=False, checks=[])
    try:
        runner.start(stage)
        runner.save_query("campaign_init", file="/project/input/campaign_hive.sql")
        output = runner.save_query("campaign_metrics", sql=(
            "SELECT treatment,users,visitors,buyers,total_spend,visit_rate,"
            "conversion_rate,spend_per_user FROM growth_lab.ads_campaign_arm ORDER BY treatment"))
        rows = result_rows(output, CAMPAIGN_METRICS)
        report["checks"] += reconcile(rows, expected, "treatment", {"users", "visitors", "buyers"})
        write_csv(args.output / "hive_metrics.csv", rows, CAMPAIGN_METRICS)
        plan = runner.save_query("explain", sql=(
            "EXPLAIN FORMATTED SELECT zip_code,COUNT(*) AS users,AVG(spend) "
            "FROM growth_lab.dwd_campaign WHERE treatment=1 AND channel='Web' GROUP BY zip_code"))
        (args.output / "explain.txt").write_text(plan, encoding="utf-8")
        schema = runner.save_query("schema", sql="DESCRIBE FORMATTED growth_lab.dwd_campaign")
        if "OrcInputFormat" not in schema:
            raise AssertionError("Hive DWD storage format is not ORC")
        partitions = runner.save_query("partitions", sql="SHOW PARTITIONS growth_lab.dwd_campaign")
        if not all(f"treatment={a}" in partitions for a in (0, 1, 2)):
            raise AssertionError("Expected three treatment partitions absent")
        if args.retail_input:
            runner.save_query("retail_init", file="/project/input/retail_hive.sql")
            retail_output = runner.save_query("retail_metrics", sql=(
                "SELECT observed_month,line_rows,identified_buyers,net_product_revenue,"
                "positive_product_revenue,return_product_revenue "
                "FROM growth_lab.ads_retail_sample_month ORDER BY observed_month"))
            retail_rows = result_rows(retail_output, RETAIL_METRICS)
            report["retail_checks"] = reconcile(retail_rows, retail_reference, "observed_month",
                                                 {"line_rows", "identified_buyers"})
            report["retail_sample_rows"] = sum(v["line_rows"] for v in retail_reference.values())
            report["retail_source_sha256"] = sha256(args.retail_input)
            write_csv(args.output / "retail_hive_metrics.csv", retail_rows, RETAIL_METRICS)
        report["status"] = "passed"
        print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "retail_checks")},
                         ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        runner.finish(args.keep_container)
        # Keep source data out of uploaded engine evidence; it remains in data/.
        if not args.keep_container:
            if stage.resolve().parent != args.output or stage.is_symlink():
                raise RuntimeError("Refusing cleanup outside the explicitly named output directory")
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
