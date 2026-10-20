"""Exploratory Data Analysis (EDA) and Visualization Module for Retail Forecasting.

Generates presentation-ready figures saved directly to `reports/figures/`:
1. `01_missing_data_profile.png` - Missing data breakdown across raw features.
2. `02_daily_net_sales_trend.png` - Daily sales time series with 7-day and 30-day moving averages.
3. `03_day_of_week_seasonality.png` - Day-of-week sales volume & Saturday closure insight.
4. `04_monthly_sales_trend.png` - Monthly revenue (£) and unit volume trends.
5. `05_top_10_products_by_revenue.png` - Top 10 catalog products by gross revenue.
6. `06_sales_vs_cancellations.png` - Gross sales vs return rates across time.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Ensure project root is on sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure headless matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import RAW_DATA_DIR, PROCESSED_DATA_DIR, load_raw_data, DEFAULT_EXCEL_FILENAME
from src.cleaner import clean_transactions, aggregate_daily_sales

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Apply consistent aesthetic style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_missing_data(raw_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate bar chart of missing value percentages across raw columns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "01_missing_data_profile.png"

    missing_counts = raw_df.isnull().sum()
    missing_pct = (missing_counts / len(raw_df)) * 100
    df_missing = pd.DataFrame({"MissingCount": missing_counts, "Percentage": missing_pct})
    df_missing = df_missing.sort_values(by="Percentage", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(
        x=df_missing.index,
        y="Percentage",
        data=df_missing,
        palette="Blues_r",
        hue=df_missing.index,
        legend=False,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_title("Missing Value Distribution in Raw Dataset (541,909 Records)", weight="bold", pad=12)
    ax.set_ylabel("Missing Percentage (%)")
    ax.set_xlabel("Feature")
    ax.set_ylim(0, max(df_missing["Percentage"].max() * 1.2, 5))
    plt.xticks(rotation=30, ha="right")

    for p in bars.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
                weight="semibold",
            )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)
    return fig_path


def plot_daily_sales_trend(daily_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate daily net sales line plot with rolling trends and holiday annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "02_daily_net_sales_trend.png"

    df = daily_df.copy()
    df["sales_7d_ma"] = df["net_sales"].rolling(window=7, min_periods=1).mean()
    df["sales_30d_ma"] = df["net_sales"].rolling(window=30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df.index, df["net_sales"], label="Daily Net Sales (£)", color="#90CAF9", alpha=0.6, linewidth=1.2)
    ax.plot(df.index, df["sales_7d_ma"], label="7-Day Moving Average", color="#1976D2", linewidth=2.0)
    ax.plot(df.index, df["sales_30d_ma"], label="30-Day Moving Average", color="#D32F2F", linewidth=2.2, linestyle="--")

    # Annotate Q4 Peak
    q4_peak_date = df["net_sales"].idxmax()
    q4_peak_val = df["net_sales"].max()
    ax.annotate(
        f"Peak Q4 Surge: £{q4_peak_val:,.0f}\n({q4_peak_date.strftime('%b %d, %Y')})",
        xy=(q4_peak_date, q4_peak_val),
        xytext=(q4_peak_date - pd.Timedelta(days=75), q4_peak_val * 0.90),
        arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        weight="semibold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D32F2F", lw=1.2),
    )

    ax.set_title("Daily Net Sales Trend (£) with 7-Day and 30-Day Moving Averages", weight="bold", pad=12)
    ax.set_ylabel("Net Sales (£)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter("£{x:,.0f}")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)
    return fig_path


def plot_day_of_week_seasonality(daily_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate day-of-week sales revenue and transaction count distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "03_day_of_week_seasonality.png"

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = daily_df.groupby("day_name").agg(
        mean_sales=("net_sales", "mean"),
        total_sales=("net_sales", "sum"),
        mean_orders=("total_orders", "mean"),
    ).reindex(day_order).fillna(0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    colors = ["#1976D2", "#1976D2", "#1976D2", "#0D47A1", "#1976D2", "#E0E0E0", "#42A5F5"]
    bars = ax1.bar(dow_stats.index, dow_stats["mean_sales"], color=colors, edgecolor="black", linewidth=0.7)

    ax1.set_title("Weekly Seasonality: Average Daily Revenue by Day of Week", weight="bold", pad=12)
    ax1.set_ylabel("Average Daily Net Sales (£)", color="#0D47A1", weight="bold")
    ax1.yaxis.set_major_formatter("£{x:,.0f}")
    ax1.set_xlabel("Day of Week")

    # Add text label for Saturday closure
    sat_idx = day_order.index("Saturday")
    ax1.annotate(
        "Closed on Saturdays\n(0 Sales)",
        xy=(sat_idx, 0),
        xytext=(sat_idx, dow_stats["mean_sales"].max() * 0.25),
        ha="center",
        arrowprops=dict(facecolor="#757575", shrink=0.08, width=1.2, headwidth=6),
        fontsize=9,
        weight="semibold",
        color="#616161",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FAFAFA", ec="#BDBDBD"),
    )

    for p in bars:
        h = p.get_height()
        if h > 0:
            ax1.annotate(
                f"£{h:,.0f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
            )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)
    return fig_path


def plot_monthly_sales_and_volume(daily_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate monthly sales revenue and product unit volume trend."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "04_monthly_sales_trend.png"

    df = daily_df.copy()
    df["YearMonth"] = df.index.to_period("M").astype(str)

    monthly = df.groupby("YearMonth").agg(
        net_sales=("net_sales", "sum"),
        net_quantity=("net_quantity", "sum"),
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(monthly))
    width = 0.45

    bars = ax1.bar(x, monthly["net_sales"], width=width, label="Net Revenue (£)", color="#1976D2", alpha=0.85, edgecolor="black")
    lines = ax2.plot(x, monthly["net_quantity"], label="Net Units Sold", color="#FF9800", marker="o", linewidth=2.5, markersize=6)

    ax1.set_title("Monthly Sales Dynamics: Revenue (£) vs. Unit Volume Sold", weight="bold", pad=12)
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("Net Sales Revenue (£)", color="#1976D2", weight="bold")
    ax2.set_ylabel("Net Units Sold", color="#E65100", weight="bold")
    ax1.yaxis.set_major_formatter("£{x:,.0f}")
    ax2.yaxis.set_major_formatter("{x:,.0f}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly.index, rotation=35, ha="right")
    ax2.grid(False)

    # Combined legend
    plots = [bars, lines[0]]
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)
    return fig_path


def plot_top_products_by_revenue(cleaned_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate horizontal bar chart of top 10 products by gross sales revenue."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "05_top_10_products_by_revenue.png"

    # Only include positive sales
    sales_df = cleaned_df[cleaned_df["Quantity"] > 0]
    top_products = (
        sales_df.groupby("Description")["LineTotal"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = sns.barplot(
        x=top_products.values,
        y=top_products.index,
        palette="viridis",
        hue=top_products.index,
        legend=False,
        edgecolor="black",
        linewidth=0.7,
        ax=ax,
    )

    ax.set_title("Top 10 Products by Gross Sales Revenue (£)", weight="bold", pad=12)
    ax.set_xlabel("Total Sales Revenue (£)")
    ax.set_ylabel("Product Description")
    ax.xaxis.set_major_formatter("£{x:,.0f}")

    for p in bars.patches:
        w = p.get_width()
        ax.annotate(
            f"£{w:,.0f}",
            (w, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            weight="semibold",
        )

    plt.xlim(0, top_products.max() * 1.15)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)
    return fig_path


def plot_sales_vs_cancellations(daily_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate comparison of monthly gross sales vs return values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "06_sales_vs_cancellations.png"

    df = daily_df.copy()
    df["YearMonth"] = df.index.to_period("M").astype(str)

    monthly = df.groupby("YearMonth").agg(
        gross_sales=("gross_sales", "sum"),
        returns_amount=("returns_amount", "sum"),
    )
    monthly["return_rate_pct"] = (monthly["returns_amount"] / monthly["gross_sales"]) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    x = np.arange(len(monthly))
    width = 0.38

    ax1.bar(x - width/2, monthly["gross_sales"], width=width, label="Gross Sales (£)", color="#2E7D32", alpha=0.85, edgecolor="black")
    ax1.bar(x + width/2, monthly["returns_amount"], width=width, label="Returns Value (£)", color="#C62828", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Monetary Value (£)", weight="bold")
    ax1.yaxis.set_major_formatter("£{x:,.0f}")
    ax1.set_title("Monthly Breakdown: Gross Sales vs. Customer Returns / Cancellations", weight="bold", pad=10)
    ax1.legend(loc="upper left")

    ax2.plot(x, monthly["return_rate_pct"], color="#C62828", marker="s", linewidth=1.8, markersize=5)
    ax2.set_ylabel("Return Rate (%)", color="#C62828", weight="bold")
    ax2.set_xlabel("Year-Month")
    ax2.set_xticks(x)
    ax2.set_xticklabels(monthly.index, rotation=35, ha="right")
    ax2.set_ylim(0, max(monthly["return_rate_pct"].max() * 1.3, 5))

    for idx, rate in enumerate(monthly["return_rate_pct"]):
        ax2.annotate(
            f"{rate:.1f}%",
            (idx, rate),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
            weight="semibold",
        )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)
    return fig_path


def generate_all_eda_figures() -> list[Path]:
    """Execute complete figure generation pipeline from raw and processed data."""
    raw_excel_path = RAW_DATA_DIR / DEFAULT_EXCEL_FILENAME
    processed_daily_path = PROCESSED_DATA_DIR / "daily_sales.csv"

    if not raw_excel_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file '{raw_excel_path}' not found. Run 'python src/data_loader.py --download' first."
        )

    logger.info("Loading raw dataset for EDA profiling...")
    raw_df = load_raw_data(raw_excel_path)

    logger.info("Cleaning transactions...")
    cleaned_df = clean_transactions(raw_df)

    if processed_daily_path.is_file():
        logger.info("Loading existing processed daily sales from '%s'...", processed_daily_path)
        daily_df = pd.read_csv(processed_daily_path, index_col=0, parse_dates=True)
    else:
        logger.info("Aggregating daily sales...")
        daily_df = aggregate_daily_sales(cleaned_df)
        daily_df.to_csv(processed_daily_path, index=True)

    generated_paths: list[Path] = []
    generated_paths.append(plot_missing_data(raw_df))
    generated_paths.append(plot_daily_sales_trend(daily_df))
    generated_paths.append(plot_day_of_week_seasonality(daily_df))
    generated_paths.append(plot_monthly_sales_and_volume(daily_df))
    generated_paths.append(plot_top_products_by_revenue(cleaned_df))
    generated_paths.append(plot_sales_vs_cancellations(daily_df))

    logger.info("Successfully generated %d figures in '%s'.", len(generated_paths), FIGURES_DIR)
    return generated_paths


if __name__ == "__main__":
    generate_all_eda_figures()
