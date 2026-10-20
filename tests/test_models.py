"""Unit tests for src.models and src.evaluate modules.

Tests cover:
- Reproducibility of model training with fixed random seeds.
- Estimator fit and predict shapes for all baselines and ML models.
- Mathematical accuracy of evaluation metrics (MAE, RMSE, WAPE, zero-masked MAPE, sMAPE).
- Safe handling of zero actual values in MAPE without division-by-zero errors.
- End-to-end model comparison integrity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calculate_mae,
    calculate_rmse,
    calculate_smape,
    calculate_wape,
    calculate_zero_masked_mape,
    compare_models,
    evaluate_predictions,
)
from src.models import (
    MovingAverageRegressor,
    PersistenceNaiveRegressor,
    SeasonalNaiveRegressor,
    generate_predictions,
    get_default_models,
    get_feature_columns,
    train_models,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def synthetic_feature_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic training features and target series."""
    np.random.seed(42)
    n_samples = 50
    dates = pd.date_range("2021-01-01", periods=n_samples, freq="D", name="Date")

    data = {
        "net_sales_lag_1": np.random.uniform(100, 500, n_samples),
        "net_sales_lag_7": np.random.uniform(100, 500, n_samples),
        "net_sales_lag_14": np.random.uniform(100, 500, n_samples),
        "net_sales_lag_28": np.random.uniform(100, 500, n_samples),
        "net_sales_rolling_mean_7": np.random.uniform(100, 500, n_samples),
        "net_sales_rolling_std_7": np.random.uniform(10, 50, n_samples),
        "day_of_week": [d.dayofweek for d in dates],
        "is_weekend": [int(d.dayofweek in [5, 6]) for d in dates],
        "is_holiday": [0] * n_samples,
        "days_until_holiday": [10] * n_samples,
        "days_since_holiday": [5] * n_samples,
    }
    X = pd.DataFrame(data, index=dates)
    # Target is combination of lag_1 + noise
    y = X["net_sales_lag_1"] * 0.8 + X["net_sales_lag_7"] * 0.2 + np.random.normal(0, 10, n_samples)
    y.name = "net_sales"
    return X, y


# ------------------------------------------------------------------------------
# Baseline Model Tests
# ------------------------------------------------------------------------------

def test_persistence_naive_prediction(synthetic_feature_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test Persistence Naive outputs exact net_sales_lag_1 values."""
    X, _ = synthetic_feature_data
    model = PersistenceNaiveRegressor()
    model.fit(X)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    np.testing.assert_array_equal(preds, X["net_sales_lag_1"].values)


def test_seasonal_naive_prediction(synthetic_feature_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test Seasonal Naive outputs exact net_sales_lag_7 values."""
    X, _ = synthetic_feature_data
    model = SeasonalNaiveRegressor()
    model.fit(X)
    preds = model.predict(X)

    assert len(preds) == len(X)
    np.testing.assert_array_equal(preds, X["net_sales_lag_7"].values)


def test_moving_average_prediction(synthetic_feature_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test Moving Average outputs exact net_sales_rolling_mean_7 values."""
    X, _ = synthetic_feature_data
    model = MovingAverageRegressor()
    model.fit(X)
    preds = model.predict(X)

    assert len(preds) == len(X)
    np.testing.assert_array_equal(preds, X["net_sales_rolling_mean_7"].values)


# ------------------------------------------------------------------------------
# Machine Learning Training & Reproducibility Tests
# ------------------------------------------------------------------------------

def test_train_models_reproducibility(synthetic_feature_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Verify that training ML models with random_state=42 yields 100% identical predictions."""
    X, y = synthetic_feature_data

    models1 = train_models(X, y, random_state=42)
    models2 = train_models(X, y, random_state=42)

    rf1 = models1["Random Forest"].predict(X)
    rf2 = models2["Random Forest"].predict(X)
    np.testing.assert_allclose(rf1, rf2)

    gb1 = models1["Gradient Boosting"].predict(X)
    gb2 = models2["Gradient Boosting"].predict(X)
    np.testing.assert_allclose(gb1, gb2)


def test_get_feature_columns() -> None:
    """Verify that contemporaneous targets/totals are excluded from X feature list."""
    df = pd.DataFrame({
        "net_sales": [100.0],
        "gross_sales": [120.0],
        "returns_amount": [20.0],
        "net_quantity": [50],
        "total_orders": [5],
        "net_sales_lag_1": [90.0],
        "day_of_week": [1],
    })
    feature_cols = get_feature_columns(df)
    assert "net_sales_lag_1" in feature_cols
    assert "day_of_week" in feature_cols
    assert "net_sales" not in feature_cols
    assert "gross_sales" not in feature_cols
    assert "returns_amount" not in feature_cols
    assert "net_quantity" not in feature_cols
    assert "total_orders" not in feature_cols


# ------------------------------------------------------------------------------
# Metric Calculation Tests
# ------------------------------------------------------------------------------

def test_metric_calculations_exact() -> None:
    """Test MAE, RMSE, and WAPE on known synthetic vectors."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 330.0, 380.0])
    # Absolute errors: 10, 10, 30, 20 -> sum = 70
    # Squared errors: 100, 100, 900, 400 -> sum = 1500, mean = 375

    expected_mae = 70.0 / 4.0   # 17.5
    expected_rmse = np.sqrt(375.0)  # ~19.3649
    expected_wape = (70.0 / 1000.0) * 100.0  # 7.0%

    assert np.isclose(calculate_mae(y_true, y_pred), expected_mae)
    assert np.isclose(calculate_rmse(y_true, y_pred), expected_rmse)
    assert np.isclose(calculate_wape(y_true, y_pred), expected_wape)


def test_zero_masked_mape_handling() -> None:
    """Verify that zero-sales days do not cause division by zero or infinite errors."""
    y_true = np.array([0.0, 100.0, 200.0, 0.0])  # Contains 2 zero-sales days (e.g. Saturdays)
    y_pred = np.array([50.0, 110.0, 180.0, 10.0])

    mape, zero_days = calculate_zero_masked_mape(y_true, y_pred)

    assert zero_days == 2
    # Active days: index 1 (10/100 = 10%) and index 2 (20/200 = 10%) -> mean = 10%
    assert np.isclose(mape, 10.0)
    assert not np.isnan(mape)
    assert not np.isinf(mape)


def test_smape_bounded() -> None:
    """Verify that sMAPE is non-negative and bounded <= 200%."""
    y_true = np.array([0.0, 50.0, 100.0])
    y_pred = np.array([10.0, 0.0, 200.0])

    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 200.0


def test_compare_models_structure() -> None:
    """Test compare_models produces properly formatted and sorted DataFrame."""
    y_true = np.array([100.0, 200.0, 300.0])
    preds = {
        "Model A": np.array([100.0, 200.0, 300.0]),  # Perfect prediction (WAPE = 0%)
        "Model B": np.array([150.0, 250.0, 350.0]),  # Error
    }

    df_comp = compare_models(y_true, preds)
    assert len(df_comp) == 2
    assert "Model" in df_comp.columns
    assert "MAE (GBP)" in df_comp.columns
    assert "WAPE (%)" in df_comp.columns
    assert "Zero-Days Excluded" in df_comp.columns

    # Model A should rank #1 because it has lower WAPE
    assert df_comp.iloc[0]["Model"] == "Model A"
    assert df_comp.iloc[0]["WAPE (%)"] == 0.0


def test_champion_retraining_workflow(synthetic_feature_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Verify that retraining the validation champion on Train+Val generates valid test predictions."""
    X, y = synthetic_feature_data
    n_train = 30
    n_val = 10
    n_test = 10

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    # 1. Fit models on train partition only
    fitted_models = train_models(X_train, y_train, random_state=42)

    # 2. Model selection on validation partition
    val_preds = generate_predictions(fitted_models, X_val)
    val_comp = compare_models(y_val, val_preds)
    champion_name = val_comp.iloc[0]["Model"]

    # 3. Retrain champion on Train + Val
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    models_dict = get_default_models(random_state=42)
    champion_estimator = models_dict[champion_name]
    champion_estimator.fit(X_train_val, y_train_val)

    test_preds = champion_estimator.predict(X_test)
    assert len(test_preds) == len(X_test)
    assert not np.isnan(test_preds).any()

    # Evaluation on test partition
    test_metrics = evaluate_predictions(y_test, test_preds, model_name=f"{champion_name} (Updated)")
    assert test_metrics["Model"] == f"{champion_name} (Updated)"
    assert test_metrics["WAPE (%)"] >= 0.0

