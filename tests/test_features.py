"""Unit tests for src.features module.

Tests cover:
- Exact time-series lag calculations.
- Strict data leakage prevention in rolling statistics (zero lookahead).
- Calendar and cyclical transformation validity.
- Holiday proximity metrics.
- Warmup missing value handling.
- Chronological train / validation / test splits (ordering & non-overlap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_feature_matrix,
    chronological_train_test_split,
    create_calendar_features,
    create_holiday_features,
    create_lag_features,
    create_rolling_features,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def synthetic_daily_sales() -> pd.DataFrame:
    """Create 60 days of synthetic continuous daily sales data."""
    dates = pd.date_range("2021-01-01", periods=60, freq="D", name="Date")
    # Linear trend + constant to test arithmetic easily
    sales_values = [float(100.0 + i * 5.0) for i in range(60)]
    return pd.DataFrame(
        {
            "net_sales": sales_values,
            "gross_sales": [s + 10.0 for s in sales_values],
            "returns_amount": [10.0] * 60,
            "net_quantity": [int(10 + i) for i in range(60)],
            "total_orders": [5] * 60,
        },
        index=dates,
    )


# ------------------------------------------------------------------------------
# Lag Feature Tests
# ------------------------------------------------------------------------------

def test_create_lag_features(synthetic_daily_sales: pd.DataFrame) -> None:
    """Test that lag features correctly shift values by exact day offsets."""
    lags = [1, 7, 14, 28]
    lags_df = create_lag_features(synthetic_daily_sales, target_col="net_sales", lags=lags)

    assert list(lags_df.columns) == [f"net_sales_lag_{k}" for k in lags]

    # At day 0 (index 0), all lags should be NaN
    assert np.isnan(lags_df["net_sales_lag_1"].iloc[0])
    assert np.isnan(lags_df["net_sales_lag_7"].iloc[6])

    # At day 1 (index 1), lag 1 should equal sales from day 0
    assert lags_df["net_sales_lag_1"].iloc[1] == synthetic_daily_sales["net_sales"].iloc[0]

    # At day 28 (index 28), all lags should match their historical targets
    for k in lags:
        expected_val = synthetic_daily_sales["net_sales"].iloc[28 - k]
        assert lags_df[f"net_sales_lag_{k}"].iloc[28] == expected_val


# ------------------------------------------------------------------------------
# Strict Data Leakage Prevention Tests
# ------------------------------------------------------------------------------

def test_create_rolling_features_leakage_prevention(synthetic_daily_sales: pd.DataFrame) -> None:
    """Strictly verify that rolling statistics at time t DO NOT include target(t)."""
    windows = [7, 14, 28]
    rolling_df = create_rolling_features(synthetic_daily_sales, target_col="net_sales", windows=windows)

    # At day 7 (index 7), rolling_mean_7 must equal average of indices 0 through 6
    expected_mean_day7 = synthetic_daily_sales["net_sales"].iloc[0:7].mean()
    actual_mean_day7 = rolling_df["net_sales_rolling_mean_7"].iloc[7]
    assert np.isclose(actual_mean_day7, expected_mean_day7)

    # DEFINITIVE LEAKAGE TEST:
    # If we perturb the target value at day t, rolling_mean_7 at day t MUST NOT CHANGE.
    df_perturbed = synthetic_daily_sales.copy()
    original_target_t = df_perturbed["net_sales"].iloc[7]
    df_perturbed.loc[df_perturbed.index[7], "net_sales"] = 999999.0  # Massive spike at t

    perturbed_rolling = create_rolling_features(df_perturbed, target_col="net_sales", windows=windows)
    perturbed_mean_day7 = perturbed_rolling["net_sales_rolling_mean_7"].iloc[7]

    # Rolling mean at day 7 should remain 100% identical because target(t) is excluded
    assert np.isclose(perturbed_mean_day7, expected_mean_day7)
    # But rolling mean at day 8 should reflect the spike from day 7
    assert perturbed_rolling["net_sales_rolling_mean_7"].iloc[8] > actual_mean_day7


# ------------------------------------------------------------------------------
# Calendar & Cyclical Feature Tests
# ------------------------------------------------------------------------------

def test_create_calendar_features(synthetic_daily_sales: pd.DataFrame) -> None:
    """Test calendar components and cyclical transformations."""
    cal_df = create_calendar_features(synthetic_daily_sales)

    assert "day_of_week" in cal_df.columns
    assert "day_of_month" in cal_df.columns
    assert "month" in cal_df.columns
    assert "quarter" in cal_df.columns
    assert "year" in cal_df.columns
    assert "is_weekend" in cal_df.columns
    assert "is_saturday" in cal_df.columns
    assert "sin_day_of_week" in cal_df.columns
    assert "cos_day_of_week" in cal_df.columns

    # Cyclical values must be bounded between -1.0 and 1.0
    assert (cal_df["sin_day_of_week"].abs() <= 1.0001).all()
    assert (cal_df["cos_day_of_week"].abs() <= 1.0001).all()
    assert (cal_df["sin_month"].abs() <= 1.0001).all()
    assert (cal_df["cos_month"].abs() <= 1.0001).all()

    # 2021-01-01 is a Friday (dayofweek = 4, is_weekend = 0)
    assert cal_df["day_of_week"].iloc[0] == 4
    assert cal_df["is_weekend"].iloc[0] == 0
    # 2021-01-02 is a Saturday (dayofweek = 5, is_weekend = 1, is_saturday = 1)
    assert cal_df["day_of_week"].iloc[1] == 5
    assert cal_df["is_weekend"].iloc[1] == 1
    assert cal_df["is_saturday"].iloc[1] == 1


# ------------------------------------------------------------------------------
# Holiday Proximity Tests
# ------------------------------------------------------------------------------

def test_create_holiday_features() -> None:
    """Test days until, days since, and is_holiday indicator."""
    dates = pd.date_range("2021-12-23", "2021-12-27", freq="D", name="Date")
    dummy_df = pd.DataFrame({"dummy": range(5)}, index=dates)

    holidays = ["2021-12-25", "2021-12-26"]
    hol_df = create_holiday_features(dummy_df, holidays=holidays)

    # 2021-12-24: 1 day until Christmas, not a holiday
    assert hol_df.loc["2021-12-24", "days_until_holiday"] == 1
    assert hol_df.loc["2021-12-24", "is_holiday"] == 0

    # 2021-12-25: Christmas Day
    assert hol_df.loc["2021-12-25", "is_holiday"] == 1
    assert hol_df.loc["2021-12-25", "days_until_holiday"] == 0
    assert hol_df.loc["2021-12-25", "days_since_holiday"] == 0

    # 2021-12-27: 1 day since Boxing Day (2021-12-26)
    assert hol_df.loc["2021-12-27", "days_since_holiday"] == 1


# ------------------------------------------------------------------------------
# Feature Matrix Assembler & Missing Values Tests
# ------------------------------------------------------------------------------

def test_build_feature_matrix_drop_warmup(synthetic_daily_sales: pd.DataFrame) -> None:
    """Test feature matrix assembly with warmup row trimming."""
    # With 60 initial days, max lag/window=28, dropping warmup leaves 60 - 28 = 32 rows
    feature_matrix = build_feature_matrix(
        synthetic_daily_sales,
        target_col="net_sales",
        lags=[1, 7, 14, 28],
        windows=[7, 14, 28],
        drop_warmup=True,
    )

    assert len(feature_matrix) == 32
    assert feature_matrix.isnull().sum().sum() == 0  # Zero NaNs
    assert "net_sales_lag_28" in feature_matrix.columns
    assert "net_sales_rolling_mean_28" in feature_matrix.columns


def test_build_feature_matrix_retain_warmup(synthetic_daily_sales: pd.DataFrame) -> None:
    """Test feature matrix assembly retaining warmup rows."""
    feature_matrix = build_feature_matrix(
        synthetic_daily_sales,
        target_col="net_sales",
        lags=[1, 7, 14, 28],
        windows=[7, 14, 28],
        drop_warmup=False,
    )

    assert len(feature_matrix) == 60
    assert feature_matrix.isnull().sum().sum() > 0  # Warmup NaNs present


# ------------------------------------------------------------------------------
# Chronological Split Tests
# ------------------------------------------------------------------------------

def test_chronological_train_test_split_temporal_order(synthetic_daily_sales: pd.DataFrame) -> None:
    """Test chronological partition sizes, ordering, and complete non-overlap."""
    train_df, val_df, test_df = chronological_train_test_split(
        synthetic_daily_sales,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # Check total lengths
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_daily_sales)
    assert len(train_df) == 42   # 60 * 0.70
    assert len(val_df) == 9      # 60 * 0.15
    assert len(test_df) == 9     # 60 - 42 - 9

    # Strict temporal ordering: max(train) < min(val) and max(val) < min(test)
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

    # Zero index intersection
    assert len(set(train_df.index).intersection(set(val_df.index))) == 0
    assert len(set(val_df.index).intersection(set(test_df.index))) == 0


def test_chronological_train_test_split_invalid_ratios(synthetic_daily_sales: pd.DataFrame) -> None:
    """Test that invalid split ratios raise ValueError."""
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        chronological_train_test_split(synthetic_daily_sales, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
