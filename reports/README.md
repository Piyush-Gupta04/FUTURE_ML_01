# Reports & Deliverables Directory

This directory stores evaluation deliverables, business reports, performance benchmark tables, and publication-ready visualizations.

## Directory Structure

```text
reports/
├── figures/                       <- Publication-ready visualizations (PNG, 300 DPI)
│   ├── 01_missing_data_profile.png
│   ├── 02_daily_net_sales_trend.png
│   ├── 03_day_of_week_seasonality.png
│   ├── 04_monthly_sales_trend.png
│   ├── 05_top_10_products_by_revenue.png
│   ├── 06_sales_vs_cancellations.png
│   └── 07_forecast_vs_actual.png
├── final_project_report.md        <- Comprehensive final executive summary & strategic findings
├── submission_checklist.md        <- Internship submission checklist & LinkedIn post draft
├── model_comparison.csv           <- Exported benchmark metrics across Validation and Test sets
└── README.md                      <- Directory index and overview (this file)
```

## Key Deliverables Summary

- **`final_project_report.md`**: Complete analytical report detailing dataset provenance, transparent cleaning logic, feature engineering, baseline comparisons, empirical metrics, business strategy recommendations, and operational limitations.
- **`submission_checklist.md`**: Verification checklist covering project execution, repository cleanliness, and a drafted LinkedIn update ready for post-internship publishing.
- **`model_comparison.csv`**: Machine-readable benchmarks (MAE, RMSE, WAPE, zero-masked MAPE, sMAPE, R²) comparing 6 forecasting models across chronological Validation and Test partitions.
- **`figures/`**: 7 high-resolution figures illustrating missing data rates, daily trends, weekly seasonality (Saturday closures), monthly dynamics, catalog Pareto distributions, cancellation rates, and out-of-sample forecast vs. actual trajectories.
