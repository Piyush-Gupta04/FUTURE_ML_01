"""Evaluation Metrics and Model Comparison Module for Retail Demand Forecasting.

This module provides standard time-series evaluation metrics and comparison reporting:
1. MAE (Mean Absolute Error): Average absolute magnitude of errors in £.
2. RMSE (Root Mean Squared Error): Penalizes large unexpected forecast spikes in £.
3. WAPE (Weighted Absolute Percentage Error):
   WAPE = (sum |y - y_hat| / sum y) * 100
   Standard retail demand metric that naturally and stably handles zero-sales days
   (such as Saturday store closures) without division by zero.
4. Zero-Masked MAPE: Standard percentage error computed exclusively over non-zero days,
   with the count of excluded zero-sales days explicitly reported.
5. sMAPE (Symmetric MAPE): Percentage error bounded between 0% and 200%.
6. R² (Coefficient of Determination): Proportion of variance explained.
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

# Configure headless matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import PROJECT_ROOT
from src.features import (
    build_feature_matrix,
    chronological_train_test_split,
    load_daily_sales,
)
from src.models import (
    generate_predictions,
    get_default_models,
    get_feature_columns,
    train_models,
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

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ------------------------------------------------------------------------------
# Core Metric Functions
# ------------------------------------------------------------------------------

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (£)."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (£)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE %).

    Formula: (sum |y - y_hat| / sum y) * 100

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        WAPE percentage (e.g. 25.4 for 25.4%). Returns 0.0 if total actual sales is 0.
    """
    total_actual = float(np.sum(y_true))
    if total_actual <= 0:
        return 0.0
    total_abs_error = float(np.sum(np.abs(y_true - y_pred)))
    return (total_abs_error / total_actual) * 100.0


def calculate_zero_masked_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 1e-5,
) -> tuple[float, int]:
    """Calculate MAPE exclusively over non-zero sales days with explicit policy documentation.

    Policy:
    In retail, store closures and holidays produce true zero-sales days (y=0). Traditional
    MAPE formula |y - y_hat| / y results in division by zero. This metric filters out days
    where |y| <= zero_threshold, computes standard MAPE on the remaining active operating days,
    and returns both the percentage error and the count of zero-sales days excluded.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    zero_threshold : float, default 1e-5
        Threshold below which values are treated as zero.

    Returns
    -------
    tuple[float, int]
        (mape_percentage, count_zero_days_excluded)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    non_zero_mask = np.abs(y_true_arr) > zero_threshold
    zero_count = int(np.sum(~non_zero_mask))

    if not np.any(non_zero_mask):
        return 0.0, zero_count

    active_true = y_true_arr[non_zero_mask]
    active_pred = y_pred_arr[non_zero_mask]
    mape_val = float(np.mean(np.abs((active_true - active_pred) / active_true)) * 100.0)
    return mape_val, zero_count


def calculate_smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-5,
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE %).

    Bounded between 0% and 200%.
    Formula: (1/n) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) ) * 100
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr) + epsilon
    return float(np.mean(2.0 * np.abs(y_true_arr - y_pred_arr) / denominator) * 100.0)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Coefficient of Determination (R²)."""
    return float(r2_score(y_true, y_pred))


# ------------------------------------------------------------------------------
# Model Evaluation Summary & Reporting
# ------------------------------------------------------------------------------

def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Compute full suite of evaluation metrics for a single model prediction.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True target values.
    y_pred : pd.Series or np.ndarray
        Predicted values.
    model_name : str, default 'Model'
        Display name of the model.

    Returns
    -------
    dict[str, Any]
        Dictionary with model evaluation results.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mae = calculate_mae(yt, yp)
    rmse = calculate_rmse(yt, yp)
    wape = calculate_wape(yt, yp)
    mape, zero_days = calculate_zero_masked_mape(yt, yp)
    smape = calculate_smape(yt, yp)
    r2 = calculate_r2(yt, yp)

    return {
        "Model": model_name,
        "MAE (GBP)": round(mae, 2),
        "RMSE (GBP)": round(rmse, 2),
        "WAPE (%)": round(wape, 2),
        "MAPE (%) [Zero-Masked]": round(mape, 2),
        "sMAPE (%)": round(smape, 2),
        "R2 Score": round(r2, 4),
        "Zero-Days Excluded": zero_days,
    }


def compare_models(
    y_true: pd.Series | np.ndarray,
    predictions_dict: dict[str, np.ndarray | pd.Series],
) -> pd.DataFrame:
    """Generate a clean comparative evaluation table across multiple models.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth values.
    predictions_dict : dict[str, np.ndarray | pd.Series]
        Mapping from model name to prediction array.

    Returns
    -------
    pd.DataFrame
        Structured comparison DataFrame sorted by WAPE (ascending).
    """
    rows: list[dict[str, Any]] = []
    for name, y_pred in predictions_dict.items():
        row = evaluate_predictions(y_true, y_pred, model_name=name)
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    return df_comp.sort_values(by="WAPE (%)", ascending=True).reset_index(drop=True)


# ------------------------------------------------------------------------------
# Forecast Visualization
# ------------------------------------------------------------------------------

def plot_forecast_vs_actual(
    y_true: pd.Series,
    predictions_dict: dict[str, np.ndarray | pd.Series],
    title: str = "Forecast vs. Actual Net Daily Sales (£)",
    highlight_models: list[str] | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Generate publication-ready forecast vs. actual visualization.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth series with DatetimeIndex.
    predictions_dict : dict[str, np.ndarray | pd.Series]
        Dictionary of predictions.
    title : str
        Figure title.
    highlight_models : list[str], optional
        Models to include in plot. If None, plots top 2 models + Seasonal Naive.
    output_path : Path or str, optional
        Target save path. Defaults to reports/figures/07_forecast_vs_actual.png.

    Returns
    -------
    Path
        Saved figure path.
    """
    if output_path is None:
        target_path = FIGURES_DIR / "07_forecast_vs_actual.png"
    else:
        target_path = Path(output_path).resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Plot ground truth
    ax.plot(
        y_true.index,
        y_true.values,
        label="Actual Net Sales (£)",
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        alpha=0.9,
    )

    palette = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00"]
    model_items = list(predictions_dict.items())
    if highlight_models is not None:
        model_items = [(k, v) for k, v in model_items if k in highlight_models]

    for idx, (name, preds) in enumerate(model_items[:4]):
        color = palette[idx % len(palette)]
        ax.plot(
            y_true.index,
            preds,
            label=name,
            color=color,
            linewidth=1.8,
            linestyle="--" if "Naive" in name or "Average" in name else "-",
            alpha=0.85,
        )

    ax.set_title(title, weight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Sales (£)")
    ax.yaxis.set_major_formatter("£{x:,.0f}")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    fig.savefig(target_path, dpi=300)
    plt.close(fig)
    logger.info("Saved forecast comparison plot: %s", target_path)
    return target_path


def export_evaluation_report(
    val_results: pd.DataFrame,
    test_results: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """Export formatted evaluation markdown report and CSV summary to disk."""
    if output_path is None:
        target_path = REPORTS_DIR / "model_comparison.csv"
    else:
        target_path = Path(output_path).resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Save validation CSV
    val_results["Partition"] = "Validation"
    test_results["Partition"] = "Test"
    combined = pd.concat([val_results, test_results], ignore_index=True)
    combined.to_csv(target_path, index=False)
    logger.info("Exported model comparison metrics to: %s", target_path)
    return target_path


# ------------------------------------------------------------------------------
# End-to-End Evaluation Pipeline
# ------------------------------------------------------------------------------

def run_evaluation_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Execute model training, validation selection, and test evaluation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, Path]
        (val_comparison_df, test_comparison_df, plot_path)
    """
    daily = load_daily_sales()
    feature_matrix = build_feature_matrix(daily, drop_warmup=True)
    train_df, val_df, test_df = chronological_train_test_split(feature_matrix)

    target_col = "net_sales"
    feature_cols = get_feature_columns(feature_matrix)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # 1. Fit models on training partition only
    fitted_models = train_models(X_train, y_train)

    # 2. Evaluate on Validation set (Model Selection strictly on validation data)
    val_preds = generate_predictions(fitted_models, X_val)
    val_comparison = compare_models(y_val, val_preds)

    best_val_model_name = val_comparison.iloc[0]["Model"]
    logger.info("Best model on validation set: %s (WAPE: %.2f%%)", best_val_model_name, val_comparison.iloc[0]["WAPE (%)"])

    # 3. Evaluate on Out-of-Sample Test set
    test_preds = generate_predictions(fitted_models, X_test)

    # 4. Standard Operational Practice: Retrain the selected champion architecture
    # on Train + Validation before out-of-sample deployment
    train_val_df = pd.concat([train_df, val_df])
    X_train_val = train_val_df[feature_cols]
    y_train_val = train_val_df[target_col]

    all_fresh_models = get_default_models(random_state=42)
    if best_val_model_name in all_fresh_models:
        champion_estimator = all_fresh_models[best_val_model_name]
        logger.info(
            "Retraining champion model '%s' on combined Train+Validation partition (%d observations)...",
            best_val_model_name,
            len(train_val_df),
        )
        champion_estimator.fit(X_train_val, y_train_val)
        updated_model_name = f"{best_val_model_name} (Updated)"
        test_preds[updated_model_name] = champion_estimator.predict(X_test)
        logger.info("Added '%s' to out-of-sample test evaluation.", updated_model_name)

    test_comparison = compare_models(y_test, test_preds)

    # 5. Generate forecast plot on test partition
    updated_name = f"{best_val_model_name} (Updated)"
    plot_models = [
        updated_name if updated_name in test_preds else best_val_model_name,
        best_val_model_name,
        "Seasonal Naive (t-7)",
    ]
    plot_path = plot_forecast_vs_actual(
        y_test,
        test_preds,
        title=f"Test Set Forecast vs. Actual Sales ({test_df.index.min().date()} to {test_df.index.max().date()})",
        highlight_models=plot_models,
    )

    # 6. Export summary reports
    export_evaluation_report(val_comparison, test_comparison)

    return val_comparison, test_comparison, plot_path


def main() -> None:
    """CLI entry point for src.evaluate module."""
    parser = argparse.ArgumentParser(
        description="Retail Sales & Demand Forecasting: Model Evaluation & Comparison"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Train models and evaluate across validation and test partitions.",
    )
    args = parser.parse_args()

    if args.run or len(sys.argv) == 1:
        val_comp, test_comp, plot_p = run_evaluation_pipeline()
        print("\n" + "=" * 80)
        print("VALIDATION SET MODEL COMPARISON (Model Selection Partition)")
        print("=" * 80)
        print(val_comp.to_string(index=False))
        print("\n" + "=" * 80)
        print("OUT-OF-SAMPLE TEST SET PERFORMANCE (Final Evaluation Partition)")
        print("=" * 80)
        print(test_comp.to_string(index=False))
        print("\nForecast visualization saved to:", plot_p)


if __name__ == "__main__":
    main()
