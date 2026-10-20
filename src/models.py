"""Forecasting Models and Training Utilities for Retail Demand Forecasting.

This module implements:
1. Baseline Heuristics:
   - Persistence Naive (predicts previous day t-1).
   - Seasonal Naive (predicts same day last week t-7).
   - 7-Day Moving Average (predicts mean of preceding 7 days).
2. Machine Learning Regressors:
   - Ridge Regression (L2 regularized linear model with StandardScaler).
   - Random Forest Regressor (non-linear bagged tree ensemble).
   - Gradient Boosting Regressor (sequential boosted trees).
3. Zero-Leakage Training Workflows:
   - Preprocessing is fitted strictly on training data.
   - Fixed random seeds (random_state=42) for 100% reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Protocol

# Ensure project root is on sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Feature column exclusion list (contemporaneous target/totals that must never be in X)
EXCLUDED_FROM_X: list[str] = [
    "net_sales",
    "gross_sales",
    "returns_amount",
    "net_quantity",
    "total_orders",
]


# ------------------------------------------------------------------------------
# Baseline Estimators (scikit-learn compatible interface)
# ------------------------------------------------------------------------------

class PersistenceNaiveRegressor(BaseEstimator, RegressorMixin):
    """Persistence Naive Baseline: Predicts previous day's sales (t-1).

    Uses `net_sales_lag_1` feature, which is strictly known at the forecast origin.
    """

    def __init__(self, lag_col: str = "net_sales_lag_1") -> None:
        self.lag_col = lag_col
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> PersistenceNaiveRegressor:
        """Fit estimator (validates column presence)."""
        if self.lag_col not in X.columns:
            raise KeyError(f"Required lag column '{self.lag_col}' not found in X.")
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate t-1 predictions."""
        if not self.is_fitted_:
            self.fit(X)
        return np.asarray(X[self.lag_col].values, dtype=float)


class SeasonalNaiveRegressor(BaseEstimator, RegressorMixin):
    """Seasonal Naive Baseline: Predicts sales from 7 days earlier (t-7).

    Captures weekly operational cycle (e.g., Saturday store closures, weekday rhythms).
    """

    def __init__(self, seasonal_col: str = "net_sales_lag_7") -> None:
        self.seasonal_col = seasonal_col
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> SeasonalNaiveRegressor:
        """Fit estimator (validates column presence)."""
        if self.seasonal_col not in X.columns:
            raise KeyError(f"Required seasonal lag column '{self.seasonal_col}' not found in X.")
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate t-7 predictions."""
        if not self.is_fitted_:
            self.fit(X)
        return np.asarray(X[self.seasonal_col].values, dtype=float)


class MovingAverageRegressor(BaseEstimator, RegressorMixin):
    """Moving Average Baseline: Predicts average of preceding 7 days.

    Uses `net_sales_rolling_mean_7` feature (which is strictly shifted by 1 day
    to prevent lookahead leakage).
    """

    def __init__(self, rolling_col: str = "net_sales_rolling_mean_7") -> None:
        self.rolling_col = rolling_col
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> MovingAverageRegressor:
        """Fit estimator (validates column presence)."""
        if self.rolling_col not in X.columns:
            raise KeyError(f"Required rolling column '{self.rolling_col}' not found in X.")
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate rolling average predictions."""
        if not self.is_fitted_:
            self.fit(X)
        return np.asarray(X[self.rolling_col].values, dtype=float)


# ------------------------------------------------------------------------------
# Model Factory & Configurations
# ------------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all valid feature columns, excluding contemporaneous targets/totals."""
    return [c for c in df.columns if c not in EXCLUDED_FROM_X]


def get_default_models(random_state: int = 42) -> dict[str, Any]:
    """Instantiate standard baseline heuristics and ML regressors.

    Parameters
    ----------
    random_state : int, default 42
        Seed for reproducible initialization.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model names to estimator instances.
    """
    return {
        "Persistence Naive (t-1)": PersistenceNaiveRegressor(),
        "Seasonal Naive (t-7)": SeasonalNaiveRegressor(),
        "Moving Average (7d)": MovingAverageRegressor(),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=random_state)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=random_state,
        ),
    }


# ------------------------------------------------------------------------------
# Training & Prediction Runners
# ------------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    models: dict[str, Any] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train all specified forecasting models on training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features matrix (strictly pre-cutoff).
    y_train : pd.Series or np.ndarray
        Training target vector.
    models : dict[str, Any], optional
        Custom model dictionary. Defaults to get_default_models().
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary of fitted estimators.
    """
    if models is None:
        models = get_default_models(random_state=random_state)

    fitted_models: dict[str, Any] = {}
    logger.info("Training %d forecasting models on %d observations...", len(models), len(X_train))

    for name, model in models.items():
        logger.info("Fitting model: %s...", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model

    logger.info("All models fitted successfully.")
    return fitted_models


def generate_predictions(
    fitted_models: dict[str, Any],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Generate predictions across all fitted models for a feature matrix X.

    Parameters
    ----------
    fitted_models : dict[str, Any]
        Dictionary of fitted estimators.
    X : pd.DataFrame
        Input feature matrix.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping model names to predicted 1-D numpy arrays.
    """
    preds: dict[str, np.ndarray] = {}
    for name, model in fitted_models.items():
        preds[name] = model.predict(X)
    return preds
