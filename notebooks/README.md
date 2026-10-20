# Notebooks Directory

This folder contains Jupyter Notebooks used for interactive exploration, data visualization, feature engineering, and model experimentation.

## Naming Convention

Notebooks should follow a sequential numbering prefix to reflect the project workflow:

- `01_exploratory_data_analysis.ipynb` — Data profiling, business cleaning rules, cancellation auditing, and 6 publication-ready visual insights (Phase 2).
- `02_feature_engineering.ipynb` — 31 leak-free time-series features (lags, shifted rolling stats, calendar encodings, holiday distance) and chronological splits (Phase 3).
- `03_model_training.ipynb` — Baseline models (Persistence, Seasonal Naive, Moving Average), ML regressors (Ridge, Random Forest, Gradient Boosting), validation selection, and out-of-sample test evaluation (Phase 4).

## Best Practices

1. **Keep Notebooks Reproducible**: Run all cells from top to bottom before sharing or committing.
2. **Clear Outputs Before Committing**: Clear large outputs and intermediate plots or use `.gitignore` to prevent repository bloat.
3. **Refactor Reusable Code**: Once a helper function or data processing routine is validated, move it into `src/` modules rather than copy-pasting across notebooks.
