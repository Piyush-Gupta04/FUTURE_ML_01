"""Feature Engineering Pipeline for Retail Sales & Demand Forecasting.

This module creates rich, leak-free time-series features for daily sales forecasting:
1. Target Lags: Historical demand at t-1, t-7, t-14, t-28.
2. Rolling Statistics: 7-day, 14-day, and 28-day rolling mean & standard deviation.
   CRITICAL: All rolling calculations are strictly shifted by 1 day (shift(1)) so
   the current day's target is never used in feature computation (zero lookahead leakage).
3. Calendar Features: Day of week, day of month, month, quarter, year, weekend indicator,
   and Saturday closure flag, plus cyclical sine/cosine encodings.
4. Holiday Proximity: Days until next UK retail holiday, days since last holiday,
   and binary holiday indicator (covering UK bank holidays, Christmas, and Black Friday).
5. Chronological Splitting: Strictly ordered Train / Validation / Test partitions to
   guarantee no temporal data leakage.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys

# Ensure project root is on sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import PROCESSED_DATA_DIR
from src.cleaner import run_pipeline

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

# Default time-series feature parameters
DEFAULT_LAGS: list[int] = [1, 7, 14, 28]
DEFAULT_WINDOWS: list[int] = [7, 14, 28]

# Relevant UK bank holidays and retail demand drivers (2010-2011)
UK_RETAIL_HOLIDAYS: list[str] = [
    "2010-08-30",  # Summer Bank Holiday 2010 (historical anchor)
    "2010-12-25",  # Christmas Day 2010
    "2010-12-26",  # Boxing Day 2010
    "2010-12-27",  # Christmas Substitute Bank Holiday 2010
    "2010-12-28",  # Boxing Day Substitute Bank Holiday 2010
    "2011-01-01",  # New Year's Day 2011
    "2011-01-03",  # New Year Substitute Bank Holiday 2011
    "2011-04-22",  # Good Friday 2011
    "2011-04-25",  # Easter Monday 2011
    "2011-04-29",  # Royal Wedding Public Holiday 2011
    "2011-05-02",  # Early May Bank Holiday 2011
    "2011-05-30",  # Spring Bank Holiday 2011
    "2011-08-29",  # Summer Bank Holiday 2011
    "2011-10-31",  # Halloween (Retail driver) 2011
    "2011-11-25",  # Black Friday (Major retail driver) 2011
    "2011-12-25",  # Christmas Day 2011
    "2011-12-26",  # Boxing Day 2011
    "2012-01-01",  # New Year's Day 2012 (forward anchor)
]


# ------------------------------------------------------------------------------
# Time-Series Feature Functions
# ------------------------------------------------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "net_sales",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create lagged historical observations of the target variable.

    Lag t-k represents the observation value k days prior to the forecast date.

    Parameters
    ----------
    df : pd.DataFrame
        Daily sales DataFrame with DatetimeIndex.
    target_col : str, default 'net_sales'
        Column name to create lags for.
    lags : list[int], optional
        List of day offsets to lag. Defaults to [1, 7, 14, 28].

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the generated lag columns.
    """
    if lags is None:
        lags = DEFAULT_LAGS

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    lag_data: dict[str, pd.Series] = {}
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        lag_data[col_name] = df[target_col].shift(lag)

    return pd.DataFrame(lag_data, index=df.index)


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "net_sales",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create rolling statistics of the target variable with strictly enforced lookahead prevention.

    LEAKAGE PREVENTION:
    -------------------
    The target series is explicitly shifted by 1 day (shift(1)) PRIOR to computing
    rolling aggregates. Therefore, at any forecast date t, the rolling window encompasses
    observations from [t-window, ..., t-1]. The observation at date t is never used.

    Parameters
    ----------
    df : pd.DataFrame
        Daily sales DataFrame with DatetimeIndex.
    target_col : str, default 'net_sales'
        Column name to compute rolling statistics on.
    windows : list[int], optional
        Window lengths in days. Defaults to [7, 14, 28].

    Returns
    -------
    pd.DataFrame
        DataFrame containing rolling mean and standard deviation columns.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    # Strict lookahead prevention: shift(1) ensures observation at t is never in the window
    shifted_series = df[target_col].shift(1)

    rolling_data: dict[str, pd.Series] = {}
    for w in windows:
        mean_col = f"{target_col}_rolling_mean_{w}"
        std_col = f"{target_col}_rolling_std_{w}"

        rolling_data[mean_col] = shifted_series.rolling(window=w, min_periods=w).mean()
        rolling_data[std_col] = shifted_series.rolling(window=w, min_periods=w).std()

    return pd.DataFrame(rolling_data, index=df.index)


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal calendar features and cyclical encodings from the DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Calendar features DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

    cal = pd.DataFrame(index=df.index)

    # Core calendar parts
    cal["day_of_week"] = df.index.dayofweek          # 0=Monday ... 6=Sunday
    cal["day_of_month"] = df.index.day
    cal["month"] = df.index.month
    cal["quarter"] = df.index.quarter
    cal["year"] = df.index.year

    # Binary flags
    cal["is_weekend"] = cal["day_of_week"].isin([5, 6]).astype(int)
    cal["is_saturday"] = (cal["day_of_week"] == 5).astype(int)  # 0 sales operational closure
    cal["is_month_start"] = df.index.is_month_start.astype(int)
    cal["is_month_end"] = df.index.is_month_end.astype(int)

    # Cyclical sine/cosine transformations (preserves continuous periodic boundaries)
    # Day of week (period = 7)
    cal["sin_day_of_week"] = np.sin(2 * np.pi * cal["day_of_week"] / 7.0)
    cal["cos_day_of_week"] = np.cos(2 * np.pi * cal["day_of_week"] / 7.0)

    # Month of year (period = 12)
    cal["sin_month"] = np.sin(2 * np.pi * (cal["month"] - 1) / 12.0)
    cal["cos_month"] = np.cos(2 * np.pi * (cal["month"] - 1) / 12.0)

    return cal


def create_holiday_features(
    df: pd.DataFrame,
    holidays: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate holiday proximity metrics: days until next holiday, days since last, and holiday flag.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    holidays : list[str], optional
        List of holiday date strings. Defaults to UK_RETAIL_HOLIDAYS.

    Returns
    -------
    pd.DataFrame
        Holiday features DataFrame containing:
        - `is_holiday`: Binary flag (1 if date is a designated holiday, 0 otherwise).
        - `days_until_holiday`: Count of days until the upcoming holiday.
        - `days_since_holiday`: Count of days since the most recent past holiday.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

    if holidays is None:
        holidays = UK_RETAIL_HOLIDAYS

    hol_dates = pd.to_datetime(holidays).sort_values()
    df_dates = df.index.normalize()

    days_until: list[int] = []
    days_since: list[int] = []
    is_holiday: list[int] = []

    for current_date in df_dates:
        # Future holidays (h >= current_date)
        future_hols = hol_dates[hol_dates >= current_date]
        if not future_hols.empty:
            until_val = (future_hols.min() - current_date).days
        else:
            until_val = 999  # Large boundary constant if none remaining

        # Past holidays (h <= current_date)
        past_hols = hol_dates[hol_dates <= current_date]
        if not past_hols.empty:
            since_val = (current_date - past_hols.max()).days
        else:
            since_val = 999  # Large boundary constant if none prior

        is_hol_val = 1 if until_val == 0 else 0

        days_until.append(until_val)
        days_since.append(since_val)
        is_holiday.append(is_hol_val)

    return pd.DataFrame(
        {
            "is_holiday": is_holiday,
            "days_until_holiday": days_until,
            "days_since_holiday": days_since,
        },
        index=df.index,
    )


# ------------------------------------------------------------------------------
# Feature Matrix Assembler
# ------------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "net_sales",
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Assemble all engineered time-series, calendar, and holiday features into a unified matrix.

    Missing Values / Warmup Period:
    -------------------------------
    Because the maximum lag and rolling window is 28 days, the initial 28 rows
    of the series (the warmup period) will have NaN values for 28-day lags/windows.
    If `drop_warmup=True` (recommended), these initial 28 warmup rows are dropped,
    yielding a 100% complete feature matrix without any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Daily sales DataFrame with DatetimeIndex.
    target_col : str, default 'net_sales'
        Primary target variable for forecasting.
    lags : list[int], optional
        Offsets for lag features.
    windows : list[int], optional
        Window lengths for rolling statistics.
    drop_warmup : bool, default True
        Whether to drop the initial warmup rows containing NaNs.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for machine learning model training.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")

    logger.info("Building feature matrix for target '%s' (%d initial days)...", target_col, len(df))

    lags_df = create_lag_features(df, target_col=target_col, lags=lags)
    rolling_df = create_rolling_features(df, target_col=target_col, windows=windows)
    calendar_df = create_calendar_features(df)
    holiday_df = create_holiday_features(df)

    # Retain the target and key daily totals
    base_cols = [c for c in [target_col, "gross_sales", "returns_amount", "net_quantity", "total_orders"] if c in df.columns]
    feature_matrix = pd.concat([df[base_cols], lags_df, rolling_df, calendar_df, holiday_df], axis=1)

    initial_len = len(feature_matrix)
    if drop_warmup:
        feature_matrix = feature_matrix.dropna()
        dropped_count = initial_len - len(feature_matrix)
        logger.info(
            "Dropped %d warmup rows (from %s to %s). Retained %d clean rows with 0 NaNs.",
            dropped_count,
            df.index.min().date(),
            feature_matrix.index.min().date(),
            len(feature_matrix),
        )

    return feature_matrix


# ------------------------------------------------------------------------------
# Chronological Splitter (Zero Data Leakage)
# ------------------------------------------------------------------------------

def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically into Train, Validation, and Test partitions.

    LEAKAGE PREVENTION:
    -------------------
    Random splits shuffle future dates into the past, causing severe lookahead bias.
    This function strictly partitions data by chronological order:
    - Train: [0 : n_train]
    - Validation: [n_train : n_train + n_val]
    - Test: [n_train + n_val : end]

    Parameters
    ----------
    df : pd.DataFrame
        Ordered time-series DataFrame with DatetimeIndex.
    train_ratio : float, default 0.70
        Fraction of data for training.
    val_ratio : float, default 0.15
        Fraction of data for validation / hyperparameter tuning.
    test_ratio : float, default 0.15
        Fraction of data for final out-of-sample evaluation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        If the ratios do not sum to 1.0 or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(f"Ratios must sum to 1.0; got {total_ratio:.4f}")

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    logger.info(
        "Chronological Split: Train=%d days (%s to %s) | Val=%d days (%s to %s) | Test=%d days (%s to %s)",
        len(train_df), train_df.index.min().date(), train_df.index.max().date(),
        len(val_df), val_df.index.min().date(), val_df.index.max().date(),
        len(test_df), test_df.index.min().date(), test_df.index.max().date(),
    )

    return train_df, val_df, test_df


# ------------------------------------------------------------------------------
# Export & Pipeline Runner
# ------------------------------------------------------------------------------

def export_featured_data(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """Save the engineered feature matrix to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix.
    output_path : Path or str, optional
        Destination path. Defaults to data/processed/featured_daily_sales.csv.

    Returns
    -------
    Path
        Path of the saved file.
    """
    if output_path is None:
        target_path = PROCESSED_DATA_DIR / "featured_daily_sales.csv"
    else:
        target_path = Path(output_path).resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=True)
    logger.info("Exported feature matrix to: %s (%d rows, %d columns)", target_path, len(df), len(df.columns))
    return target_path


def load_daily_sales(daily_path: Path | str | None = None) -> pd.DataFrame:
    """Load daily sales dataset from disk, executing the cleaner pipeline if missing."""
    if daily_path is None:
        path = PROCESSED_DATA_DIR / "daily_sales.csv"
    else:
        path = Path(daily_path).resolve()

    if not path.is_file():
        logger.info("Daily sales file '%s' not found. Executing Phase 2 pipeline...", path)
        _, daily_df = run_pipeline()
        return daily_df

    logger.info("Loading daily sales from '%s'...", path)
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def run_feature_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end feature engineering and chronological train/val/test split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (feature_matrix, train_df, val_df, test_df)
    """
    daily_df = load_daily_sales()
    feature_matrix = build_feature_matrix(daily_df, target_col="net_sales", drop_warmup=True)
    export_featured_data(feature_matrix)

    train_df, val_df, test_df = chronological_train_test_split(feature_matrix)
    return feature_matrix, train_df, val_df, test_df


def main() -> None:
    """CLI entry point for src.features module."""
    parser = argparse.ArgumentParser(
        description="Retail Sales & Demand Forecasting: Feature Engineering Pipeline"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Build feature matrix and perform chronological train/val/test split.",
    )
    args = parser.parse_args()

    if args.run or len(sys.argv) == 1:
        run_feature_pipeline()


if __name__ == "__main__":
    main()
