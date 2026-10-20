"""Data Cleaning and Daily Sales Aggregation Pipeline for Retail Forecasting.

This module provides reproducible, fully documented functions to:
- Clean raw transactional retail logs with transparent business rules.
- Distinguish genuine sales transactions from cancellations, returns, and warehouse write-offs.
- Aggregate transactions into daily sales time series with explicit metrics
  (Gross Sales, Returns, Net Sales, Gross Quantity, Net Quantity).
- Export processed datasets for exploratory analysis and downstream modeling.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

# Ensure project root is on sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import (
    DEFAULT_EXCEL_FILENAME,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    load_raw_data,
)

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

# Administrative / non-merchandise stock codes
NON_PRODUCT_CODES: set[str] = {
    "POST",         # Postage
    "D",            # Discount
    "M",            # Manual entry
    "BANK CHARGES", # Bank fee
    "CRST",         # Manual credit/adjustment
    "AMAZONFEE",    # Amazon marketplace fee
    "DOT",          # Dotcom postage
    "S",            # Samples
}


# ------------------------------------------------------------------------------
# Cleaning Pipeline
# ------------------------------------------------------------------------------

def clean_transactions(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    remove_non_products: bool = False,
) -> pd.DataFrame:
    """Clean raw retail transactions with explicit business assumptions.

    Business Rules & Transformations Applied:
    -----------------------------------------
    1. Deduplication: Removes exact duplicate rows (typically caused by client-side
       re-clicks or server logging retries).
    2. String Trimming: Strips leading/trailing whitespace from InvoiceNo, StockCode,
       Description, and Country.
    3. Missing Description & Invalid Prices:
       - Records with UnitPrice <= 0 are warehouse write-offs (damaged, lost, audit checks)
         or accounting entries ('Adjust bad debt'). They are excluded because they do not
         represent customer purchase demand.
    4. Missing CustomerID:
       - Kept in the dataset. 24.9% of transactions represent guest checkouts.
         Discarding them would underestimate true product sales volume and store demand.
         Missing CustomerID values are imputed with -1 to preserve integer typing.
    5. Cancellations & Returns:
       - Identified by InvoiceNo starting with 'C' (or negative quantities).
       - Kept and flagged with `IsCancellation` and `IsReturn` columns so returns can be
         analyzed and subtracted from gross sales to yield accurate net daily demand.
    6. Date Handling:
       - Standardizes InvoiceDate into pandas datetime and extracts a normalized Date column.
    7. Monetary Calculation:
       - Computes `LineTotal = Quantity * UnitPrice`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions DataFrame.
    drop_duplicates : bool, default True
        Whether to drop exact duplicate rows.
    remove_non_products : bool, default False
        Whether to filter out postage, discounts, and fee codes (e.g. POST, D, M).

    Returns
    -------
    pd.DataFrame
        Cleaned transactional DataFrame with audit columns.
    """
    logger.info("Starting transaction cleaning on %d initial records...", len(df))
    cleaned = df.copy()

    initial_count = len(cleaned)

    # 1. Deduplication
    if drop_duplicates:
        dups_count = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates()
        logger.info("Removed %d duplicate records (%.2f%%).", dups_count, (dups_count / initial_count) * 100)

    # 2. String trimming
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # 3. Ensure Datetime
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["Date"] = cleaned["InvoiceDate"].dt.floor("D")

    # 4. Handle invalid prices (warehouse inventory adjustments & bad debt entries)
    # UnitPrice <= 0 indicates damaged stock, sample tests, or administrative adjustments
    invalid_price_mask = cleaned["UnitPrice"] <= 0
    invalid_price_count = int(invalid_price_mask.sum())
    cleaned = cleaned[~invalid_price_mask].copy()
    logger.info(
        "Removed %d records with UnitPrice <= 0 (administrative/warehouse adjustments).",
        invalid_price_count,
    )

    # 5. Flag cancellations & returns
    cleaned["IsCancellation"] = cleaned["InvoiceNo"].str.startswith(("C", "c"))
    cleaned["IsReturn"] = cleaned["Quantity"] < 0

    # 6. Filter non-product service codes if requested
    if remove_non_products and "StockCode" in cleaned.columns:
        stock_upper = cleaned["StockCode"].str.upper()
        non_product_mask = stock_upper.isin(NON_PRODUCT_CODES)
        non_product_count = int(non_product_mask.sum())
        cleaned = cleaned[~non_product_mask].copy()
        logger.info("Filtered %d non-merchandise service records (POST, D, M, etc.).", non_product_count)

    # 7. Customer ID standardization (preserve guest checkouts as -1)
    if "CustomerID" in cleaned.columns:
        cleaned["CustomerID"] = (
            pd.to_numeric(cleaned["CustomerID"], errors="coerce")
            .fillna(-1)
            .astype(np.int64)
        )

    # 8. Compute LineTotal (£)
    cleaned["LineTotal"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    logger.info(
        "Cleaning complete: Retained %d records (%.2f%% of initial raw records).",
        len(cleaned),
        (len(cleaned) / initial_count) * 100,
    )

    return cleaned


# ------------------------------------------------------------------------------
# Daily Aggregation Pipeline
# ------------------------------------------------------------------------------

def aggregate_daily_sales(
    cleaned_df: pd.DataFrame,
    fill_missing_dates: bool = True,
) -> pd.DataFrame:
    """Aggregate cleaned transactions into a daily sales time series.

    Sales & Demand Metrics Defined:
    -------------------------------
    - `gross_sales`: Sum of LineTotal for positive purchases (£).
    - `returns_amount`: Absolute sum of LineTotal for returned/cancelled orders (£).
    - `net_sales`: `gross_sales - returns_amount` (True net daily revenue).
    - `gross_quantity`: Total product units purchased.
    - `returned_quantity`: Total product units returned/cancelled.
    - `net_quantity`: `gross_quantity - returned_quantity` (True net daily unit demand).
    - `total_orders`: Number of unique completed purchase invoices.
    - `total_cancellations`: Number of unique cancellation invoices.
    - `total_transactions`: Total unique invoices (`total_orders + total_cancellations`).
    - `unique_customers`: Number of distinct identified customers (excluding guest checkouts).
    - `unique_products`: Number of distinct StockCodes sold.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned transactions DataFrame from clean_transactions().
    fill_missing_dates : bool, default True
        Whether to reindex the time series to include missing calendar dates (e.g. Saturdays
        when the business is closed) with zero sales.

    Returns
    -------
    pd.DataFrame
        Daily aggregated time series with calendar features.
    """
    logger.info("Aggregating %d transactions into daily sales metrics...", len(cleaned_df))

    # Split into positive sales vs returns
    sales_mask = cleaned_df["Quantity"] > 0
    returns_mask = cleaned_df["Quantity"] < 0

    sales_df = cleaned_df[sales_mask]
    returns_df = cleaned_df[returns_mask]

    # Daily gross sales and volume
    daily_sales = (
        sales_df.groupby("Date")
        .agg(
            gross_sales=("LineTotal", "sum"),
            gross_quantity=("Quantity", "sum"),
            total_orders=("InvoiceNo", "nunique"),
            unique_products=("StockCode", "nunique"),
        )
    )

    # Unique identified customers on positive orders
    identified_cust = sales_df[sales_df["CustomerID"] != -1]
    daily_customers = (
        identified_cust.groupby("Date")["CustomerID"]
        .nunique()
        .rename("unique_customers")
    )
    daily_sales = daily_sales.join(daily_customers, how="left")

    # Daily returns
    daily_returns = (
        returns_df.groupby("Date")
        .agg(
            returns_amount=("LineTotal", lambda x: abs(float(x.sum()))),
            returned_quantity=("Quantity", lambda x: abs(int(x.sum()))),
            total_cancellations=("InvoiceNo", "nunique"),
        )
    )

    # Combine gross and returns
    daily = daily_sales.join(daily_returns, how="outer").fillna(0)

    # Compute Net figures
    daily["net_sales"] = daily["gross_sales"] - daily["returns_amount"]
    daily["net_quantity"] = daily["gross_quantity"] - daily["returned_quantity"]
    daily["total_transactions"] = daily["total_orders"] + daily["total_cancellations"]

    # Reindex to full continuous daily calendar if requested
    if fill_missing_dates and not daily.empty:
        min_date = daily.index.min()
        max_date = daily.index.max()
        full_index = pd.date_range(start=min_date, end=max_date, freq="D", name="Date")
        daily = daily.reindex(full_index).fillna(0)

    # Extract useful calendar features for exploratory analysis & forecasting
    daily["day_of_week"] = daily.index.dayofweek
    daily["day_name"] = daily.index.day_name()
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["month"] = daily.index.month
    daily["year"] = daily.index.year

    # Round currency fields to 2 decimal places
    currency_cols = ["gross_sales", "returns_amount", "net_sales"]
    daily[currency_cols] = daily[currency_cols].round(2)

    logger.info("Daily aggregation complete: %d total days (%s to %s).", len(daily), daily.index.min().date(), daily.index.max().date())
    return daily


def export_processed_data(
    daily_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """Save the processed daily sales DataFrame to disk (CSV format).

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily sales time series.
    output_path : Path or str, optional
        Target file path. Defaults to data/processed/daily_sales.csv.

    Returns
    -------
    Path
        Destination path of the saved file.
    """
    if output_path is None:
        target_path = PROCESSED_DATA_DIR / "daily_sales.csv"
    else:
        target_path = Path(output_path).resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(target_path, index=True)
    logger.info("Exported processed daily sales to: %s (%d rows)", target_path, len(daily_df))
    return target_path


# ------------------------------------------------------------------------------
# CLI Pipeline Runner
# ------------------------------------------------------------------------------

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end data loading, cleaning, and aggregation pipeline.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Cleaned transactions DataFrame and daily sales DataFrame.
    """
    raw_excel_path = RAW_DATA_DIR / DEFAULT_EXCEL_FILENAME
    if not raw_excel_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file not found at '{raw_excel_path}'. "
            f"Please run 'python src/data_loader.py --download' first."
        )

    logger.info("Loading raw dataset from '%s'...", raw_excel_path)
    raw_df = load_raw_data(raw_excel_path)

    logger.info("Executing transaction cleaning pipeline...")
    cleaned_df = clean_transactions(raw_df)

    logger.info("Aggregating transactions into daily sales...")
    daily_df = aggregate_daily_sales(cleaned_df)

    export_path = export_processed_data(daily_df)
    logger.info("Pipeline executed successfully! Output saved to %s", export_path)

    return cleaned_df, daily_df


def main() -> None:
    """CLI entry point for src.cleaner module."""
    parser = argparse.ArgumentParser(
        description="Retail Sales & Demand Forecasting: Transaction Cleaning & Daily Aggregation"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the complete cleaning and daily aggregation pipeline.",
    )
    args = parser.parse_args()

    if args.run or len(sys.argv) == 1:
        run_pipeline()


if __name__ == "__main__":
    main()
