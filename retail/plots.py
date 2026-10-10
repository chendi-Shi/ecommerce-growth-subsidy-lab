"""Small, reviewable charts from committed aggregate tables, never raw PII rows."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def render(report_dir: Path) -> Path:
    growth = [row for row in read(report_dir / "monthly_growth.csv") if row["complete_month"] == "true"]
    cohorts = read(report_dir / "monthly_cohort_retention.csv")
    rfm = read(report_dir / "rfm_segment_validation.csv")
    blue, orange, teal = "#2457a7", "#e18734", "#167d8d"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#344256"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.3), gridspec_kw={"hspace": .48, "wspace": .40})
    fig.patch.set_facecolor("#f7f9fc")
    x = np.arange(len(growth))
    ticks = list(range(0, len(growth), 4))
    first = np.array([float(row["first_observed_buyers"]) for row in growth])
    repeat = np.array([float(row["repeat_buyers"]) for row in growth])
    ax = axes[0, 0]
    ax.bar(x, first, color=orange, label="First observed buyers", width=.78)
    ax.bar(x, repeat, bottom=first, color=blue, label="Repeat buyers", width=.78)
    ax.set_xticks(ticks, [growth[i]["month"] for i in ticks], rotation=20)
    ax.set_title("Buyer growth: acquisition observation vs repeat", loc="left", pad=12)
    ax.set_ylabel("Identified purchasing customers")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=.12)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    ax.plot(x, [float(row["gross_product_sales"])/1e6 for row in growth], color=teal,
            marker="o", ms=3, label="Gross product sales")
    ax.plot(x, [float(row["net_product_revenue"])/1e6 for row in growth], color=blue,
            marker="o", ms=3, label="Net product revenue")
    ax.set_xticks(ticks, [growth[i]["month"] for i in ticks], rotation=20)
    ax.set_title("Sales ledger includes anonymous transactions", loc="left", pad=12)
    ax.set_ylabel("GBP millions; sales less product returns")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=.12)

    ax = axes[1, 0]
    months = sorted(set(row["cohort_month"] for row in cohorts))
    matrix = np.full((len(months), 13), np.nan)
    for row in cohorts:
        index = int(row["month_index"])
        if index <= 12:
            matrix[months.index(row["cohort_month"]), index] = float(row["retention_rate"])
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=.6)
    ax.set_xticks(range(13), [str(i) for i in range(13)])
    ax.set_yticks(range(len(months)), months, fontsize=7.8)
    ax.set_title("Cohort calendar-month purchase retention", loc="left", pad=12)
    ax.set_xlabel("Month after first observed purchase; blank = immature")
    fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label="Share purchasing", extend="max")

    ax = axes[1, 1]
    labels = [row["segment"].replace("recent_low_frequency", "Recent / low\nfrequency")
              .replace("dormant", "Dormant").replace("loyal", "Loyal").replace("regular", "Regular")
              for row in rfm]
    rates = [float(row["future_purchase_rate"])*100 for row in rfm]
    bars = ax.barh(labels, rates, color=[blue, teal, orange, "#8da7cc"][:len(rfm)])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    for bar, row, rate in zip(bars, rfm, rates):
        ax.text(rate + 2, bar.get_y() + bar.get_height()/2,
                f"{rate:.1f}%  (n={int(row['customers']):,})", va="center", fontsize=9)
    ax.set_title("RF rules checked on unseen future transactions", loc="left", pad=12)
    ax.set_xlabel("Sep-Nov 2011 purchase rate (%); segments frozen Aug 31")
    ax.grid(axis="x", alpha=.12)
    ax.set_axisbelow(True)
    fig.suptitle("Online Retail II | observed growth and repurchase opportunities",
                 x=.07, y=.975, ha="left", fontsize=18, weight="bold", color="#21344f")
    fig.text(.07, .936, "1,067,371 source invoice lines | Dec 2009-Nov 2011 complete months | descriptive analysis",
             fontsize=10.5, color="#5d6e85")
    fig.text(.07, .015, "Source: UCI Online Retail II, CC BY 4.0. No treatment, subsidy, traffic logs or margin data.\n"
             "First observed is not first-ever purchase. Initial 3 months excluded from cohorts; final partial month excluded.",
             fontsize=8, color="#65738a")
    fig.subplots_adjust(top=.88, bottom=.11, left=.07, right=.95)
    target = report_dir / "retail_overview.png"
    fig.savefig(target, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target
