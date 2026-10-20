"""Unit tests for src.cleaner module (Data Cleaning and Daily Sales Aggregation).

Tests use isolated synthetic fixtures. No network requests or external dataset
files are required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.cleaner import clean_transactions, aggregate_daily_sales


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def synthetic_raw_transactions() -> pd.DataFrame:
    """Provide synthetic transactional data covering normal sales, cancellations,
    warehouse adjustments, duplicates, and missing customer IDs.
    """
    return pd.DataFrame({
        "InvoiceNo": [
            "5001",       # Valid purchase Day 1
            "5001",       # Exact duplicate row Day 1
            "5002",       # Valid purchase Day 1 (guest checkout)
            "C5003",      # Cancellation Day 1
            "5004",       # Zero price warehouse write-off Day 1
            "5005",       # Valid purchase Day 3 (leaving Day 2 empty)
        ],
        "StockCode": [
            "21001",
            "21001",
            "21002",
            "21001",
            "21003",
            "21004",
        ],
        "Description": [
            " Item A ",
            " Item A ",
            " Item B ",
            " Item A ",
            " damaged ",
            " Item C ",
        ],
        "Quantity": [
            10,
            10,
            5,
            -2,
            -5,
            20,
        ],
        "InvoiceDate": [
            "2021-06-01 09:00:00",
            "2021-06-01 09:00:00",
            "2021-06-01 11:30:00",
            "2021-06-01 14:00:00",
            "2021-06-01 16:00:00",
            "2021-06-03 10:00:00",
        ],
        "UnitPrice": [
            2.50,
            2.50,
            4.00,
            2.50,
            0.00,   # UnitPrice <= 0 invalid adjustment
            3.00,
        ],
        "CustomerID": [
            12345.0,
            12345.0,
            None,   # Guest checkout
            12345.0,
            None,
            67890.0,
        ],
        "Country": [
            " United Kingdom ",
            " United Kingdom ",
            " United Kingdom ",
            " United Kingdom ",
            " United Kingdom ",
            " France ",
        ],
    })


# ------------------------------------------------------------------------------
# Cleaning Tests
# ------------------------------------------------------------------------------

def test_clean_transactions_deduplication(synthetic_raw_transactions: pd.DataFrame) -> None:
    """Test that exact duplicate records are removed."""
    cleaned = clean_transactions(synthetic_raw_transactions, drop_duplicates=True)
    # 6 rows initial - 1 duplicate - 1 zero-price = 4 rows retained
    assert len(cleaned) == 4
    assert cleaned["InvoiceNo"].tolist().count("5001") == 1


def test_clean_transactions_removes_zero_and_negative_prices(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """Test that records with UnitPrice <= 0 are removed as warehouse adjustments."""
    cleaned = clean_transactions(synthetic_raw_transactions)
    assert not (cleaned["UnitPrice"] <= 0).any()
    assert "5004" not in cleaned["InvoiceNo"].values


def test_clean_transactions_string_trimming(synthetic_raw_transactions: pd.DataFrame) -> None:
    """Test that leading/trailing whitespaces are properly trimmed."""
    cleaned = clean_transactions(synthetic_raw_transactions)
    descriptions = cleaned["Description"].tolist()
    assert "Item A" in descriptions
    assert "Item B" in descriptions
    assert all(not d.startswith(" ") and not d.endswith(" ") for d in descriptions)


def test_clean_transactions_flags_and_linetotal(synthetic_raw_transactions: pd.DataFrame) -> None:
    """Test cancellation/return flags and LineTotal calculation."""
    cleaned = clean_transactions(synthetic_raw_transactions)

    # LineTotal = Quantity * UnitPrice
    row_c = cleaned[cleaned["InvoiceNo"] == "C5003"].iloc[0]
    assert bool(row_c["IsCancellation"]) is True
    assert bool(row_c["IsReturn"]) is True
    assert row_c["LineTotal"] == -2 * 2.50  # -5.00

    row_sale = cleaned[cleaned["InvoiceNo"] == "5001"].iloc[0]
    assert bool(row_sale["IsCancellation"]) is False
    assert bool(row_sale["IsReturn"]) is False
    assert row_sale["LineTotal"] == 10 * 2.50  # 25.00


def test_clean_transactions_imputes_guest_customer_id(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """Test that missing CustomerID is imputed with -1 instead of dropping rows."""
    cleaned = clean_transactions(synthetic_raw_transactions)
    guest_row = cleaned[cleaned["InvoiceNo"] == "5002"].iloc[0]
    assert guest_row["CustomerID"] == -1
    assert not cleaned["CustomerID"].isnull().any()


# ------------------------------------------------------------------------------
# Aggregation Tests
# ------------------------------------------------------------------------------

def test_aggregate_daily_sales_metrics(synthetic_raw_transactions: pd.DataFrame) -> None:
    """Test mathematical accuracy of daily sales aggregation."""
    cleaned = clean_transactions(synthetic_raw_transactions)
    daily = aggregate_daily_sales(cleaned, fill_missing_dates=False)

    # We expect 2 active days: 2021-06-01 and 2021-06-03
    assert len(daily) == 2

    day1 = daily.loc[pd.Timestamp("2021-06-01")]
    # Day 1:
    # 5001: 10 * 2.50 = 25.00 gross sales, 10 units
    # 5002: 5 * 4.00 = 20.00 gross sales, 5 units
    # Gross Sales = 45.00, Gross Quantity = 15
    # C5003: 2 * 2.50 = 5.00 return, 2 units
    # Net Sales = 45.00 - 5.00 = 40.00
    # Net Quantity = 15 - 2 = 13
    assert day1["gross_sales"] == 45.00
    assert day1["returns_amount"] == 5.00
    assert day1["net_sales"] == 40.00
    assert day1["gross_quantity"] == 15
    assert day1["returned_quantity"] == 2
    assert day1["net_quantity"] == 13
    assert day1["total_orders"] == 2
    assert day1["total_cancellations"] == 1
    assert day1["total_transactions"] == 3


def test_aggregate_daily_sales_continuous_calendar(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """Test that fill_missing_dates=True generates complete calendar range with 0-filled days."""
    cleaned = clean_transactions(synthetic_raw_transactions)
    daily = aggregate_daily_sales(cleaned, fill_missing_dates=True)

    # Min date: 2021-06-01, Max date: 2021-06-03 -> 3 full days (June 1, 2, 3)
    assert len(daily) == 3
    assert pd.Timestamp("2021-06-02") in daily.index

    # Check that Day 2 is zero-filled
    day2 = daily.loc[pd.Timestamp("2021-06-02")]
    assert day2["net_sales"] == 0.0
    assert day2["net_quantity"] == 0
    assert day2["total_orders"] == 0

    # Check calendar features
    assert "day_of_week" in daily.columns
    assert "day_name" in daily.columns
    assert "is_weekend" in daily.columns
    assert "month" in daily.columns
    assert "year" in daily.columns
