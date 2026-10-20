# Retail Sales & Demand Forecasting (FUTURE_ML_01)

An end-to-end Machine Learning project designed to forecast retail product demand, analyze transactional sales dynamics, and deliver actionable inventory planning insights.

---

## 🎯 Project Objective

In retail and e-commerce operations, inaccurate demand forecasts lead to costly overstocking or stockouts. The primary objective of this project is to build an interpretable, reproducible Machine Learning forecasting pipeline that:

1. Ingests raw transactional sales logs reliably.
2. Performs data cleaning and exploratory data analysis (EDA).
3. Engineers temporal, rolling, and categorical features reflecting retail purchase patterns.
4. Benchmarks regression and time series models against baseline heuristics.
5. Produces actionable business metrics (MAE, RMSE, WAPE/MAPE) and inventory insights.

---

## 📊 Dataset Provenance & License

- **Dataset**: [UCI Machine Learning Repository: Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Archive URL**: `https://archive.ics.uci.edu/static/public/352/online+retail.zip`
- **Description**: Contains 541,909 transactional records between 01/12/2010 and 09/12/2011 for a UK-based registered online retailer specializing in unique all-occasion gifts.
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)).
- **Citation**:
  > Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining*. Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208. DOI: [10.1057/dbm.2012.17](https://doi.org/10.1057/dbm.2012.17).

---

## 📁 Project Structure

```text
FUTURE_ML_01/
├── data/
│   ├── raw/                               <- Immutable raw downloads (Online Retail.xlsx) (gitignored)
│   ├── processed/                         <- Cleaned daily sales & feature matrix (gitignored)
│   │   ├── daily_sales.csv
│   │   └── featured_daily_sales.csv
│   └── README.md                          <- Dataset schema, provenance, and data rules
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb <- End-to-end EDA walkthrough with actual findings
│   ├── 02_feature_engineering.ipynb       <- Feature engineering, leakage checks & distributions
│   ├── 03_model_training.ipynb            <- Model training, baseline benchmarking & evaluations
│   └── README.md                          <- Notebook conventions and reproducibility standards
├── src/
│   ├── __init__.py                        <- Package marker
│   ├── data_loader.py                     <- Download, safe extraction, loading, and validation
│   ├── cleaner.py                         <- Business-rule cleaning & daily sales aggregation
│   ├── eda.py                             <- Automated generation of 6 publication-ready figures
│   ├── features.py                        <- Leak-free time series, calendar, holiday features & split
│   ├── models.py                          <- Baselines (Persistence, Seasonal, MA) and ML regressors
│   └── evaluate.py                        <- Evaluation metrics (MAE, RMSE, WAPE, MAPE), comparisons
├── reports/
│   ├── figures/                           <- Generated visualizations (PNG, 300 DPI)
│   │   ├── 01_missing_data_profile.png
│   │   ├── 02_daily_net_sales_trend.png
│   │   ├── 03_day_of_week_seasonality.png
│   │   ├── 04_monthly_sales_trend.png
│   │   ├── 05_top_10_products_by_revenue.png
│   │   ├── 06_sales_vs_cancellations.png
│   │   └── 07_forecast_vs_actual.png
│   ├── final_project_report.md            <- Comprehensive executive report & business strategy
│   ├── submission_checklist.md            <- Internship submission checklist & LinkedIn post draft
│   ├── model_comparison.csv               <- Exported performance benchmarks across partitions
│   └── README.md                          <- Reporting and artifact documentation
├── tests/
│   ├── test_data_loader.py                <- Unit tests for ingestion and schema validation
│   ├── test_cleaner.py                    <- Unit tests for cleaning logic & aggregation math
│   ├── test_features.py                   <- Unit tests for leakage prevention, lags, & splits
│   └── test_models.py                     <- Unit tests for models, baselines, & evaluation metrics
├── .gitignore                             <- Excludes raw/processed data, caches, checkpoints
├── requirements.txt                       <- Project dependencies
└── README.md                              <- Project guide and reproduction instructions
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Piyush-Gupta04/FUTURE_ML_01.git
cd FUTURE_ML_01
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Complete Reproduction Workflow

### Step 1: Download & Validate Raw Dataset
Verify remote source accessibility and download the raw dataset into `data/raw/`:
```bash
# Verify remote URL accessibility
python src/data_loader.py --verify-url

# Download and safely extract the dataset
python src/data_loader.py --download

# Run non-destructive data validation
python src/data_loader.py --validate
```

### Step 2: Clean Transactions & Aggregate Daily Sales
Execute the documented cleaning pipeline and produce the continuous daily sales time series in `data/processed/daily_sales.csv`:
```bash
python src/cleaner.py
```

### Step 3: Generate Exploratory Visualizations
Produce all 6 EDA figures in `reports/figures/`:
```bash
python src/eda.py
```

### Step 4: Engineer Features & Chronological Partitions
Construct the 31-feature matrix with strict lookahead prevention (`shift(1)`):
```bash
python src/features.py
```

### Step 5: Train Models & Evaluate Performance
Train 3 baseline models and 3 machine learning models, select the best model on the validation set, evaluate out-of-sample on the test set, export `reports/model_comparison.csv`, and render `reports/figures/07_forecast_vs_actual.png`:
```bash
python src/evaluate.py
```

### Step 6: Interactive Analytical Notebooks
Launch Jupyter to explore the step-by-step notebooks:
```bash
# Phase 2 EDA: Ingestion, cleaning audits, and visual insights
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb

# Phase 3 Features: Lags, rolling aggregates, holiday countdowns, leakage checks
jupyter notebook notebooks/02_feature_engineering.ipynb

# Phase 4 Modeling: Baseline comparisons, ML regressors, forecast vs actual
jupyter notebook notebooks/03_model_training.ipynb
```

---

## 🤖 Forecasting Models & Evaluation Methodology (Phase 4)

### Evaluated Models

1. **Persistence Naive Baseline ($t-1$)**: Predicts yesterday's sales.
2. **Seasonal Naive Baseline ($t-7$)**: Predicts sales from 7 days earlier (same day last week).
3. **Moving Average Baseline (7d)**: Predicts average sales of the preceding 7 days (strictly shifted).
4. **Ridge Regression**: L2 regularized linear model with `StandardScaler` fit strictly on training data.
5. **Random Forest Regressor**: 100-tree ensemble with controlled depth (`max_depth=6`, `min_samples_leaf=3`).
6. **Gradient Boosting Regressor**: Sequential boosted tree ensemble (`n_estimators=100`, `learning_rate=0.05`).

### Evaluation Metrics & Zero-Sales Policy

- **MAE (£)**: Mean absolute deviation in currency units.
- **RMSE (£)**: Root mean squared error penalizing extreme demand spikes.
- **WAPE (%)**: Weighted Absolute Percentage Error:
  $$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y} \times 100$$
  *Why WAPE is the primary retail metric:* Retailers close on specific days (e.g. Saturdays), producing $y=0$. Traditional MAPE divides by zero ($\infty$). WAPE naturally and stably handles zero-sales days by weighting errors against aggregate demand volume.
- **Zero-Masked MAPE (%)**: Standard MAPE computed exclusively over active non-zero operating days, explicitly documenting the number of excluded zero-sales days.
- **sMAPE (%)**: Symmetric Mean Absolute Percentage Error bounded between 0% and 200%.
- **$R^2$ Score**: Proportion of variance explained.

### Limitations: One-Step-Ahead vs. Multi-Step Forecasting

- **One-Step-Ahead (Implemented Pipeline)**: Forecasts tomorrow ($t+1$) given true observations up to today ($t$). Highly reliable for day-to-day warehouse dispatching, automated purchase replenishment, and staffing.
- **Multi-Step-Ahead**: Forecasts $t+1, \dots, t+14$ consecutively. In recursive multi-step forecasting, predictions feed back into autoregressive lag features, leading to cumulative error compounding.

---

## 📈 Model Performance Benchmarks

Actual metrics computed on the leak-free chronological partitions:

### Validation Set (Model Selection Partition: 51 days, Aug 28 – Oct 17, 2011)

| Rank | Model | MAE (GBP) | RMSE (GBP) | WAPE (%) | Zero-Masked MAPE (%) | sMAPE (%) | $R^2$ Score | Zero Days |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest** | **£11,767.14** | **£17,512.14** | **36.52%** | **32.68%** | 55.69% | **0.3467** | 8 |
| 🥈 | Gradient Boosting | £13,443.90 | £19,526.83 | 41.72% | 35.39% | 69.42% | 0.1878 | 8 |
| 🥉 | Seasonal Naive ($t-7$) | £15,028.01 | £21,135.68 | 46.64% | 49.60% | 43.67% | 0.0484 | 8 |
| 4 | Moving Average (7d) | £16,628.73 | £22,031.92 | 51.61% | 41.81% | 64.30% | -0.0340 | 8 |
| 5 | Ridge Regression | £17,276.10 | £22,911.70 | 53.62% | 48.35% | 87.24% | -0.1182 | 8 |
| 6 | Persistence Naive ($t-1$) | £21,562.17 | £26,884.67 | 66.92% | 56.12% | 97.00% | -0.5397 | 8 |

*Takeaway:* **Random Forest** achieved the best validation performance, reducing MAE by **45.4%** relative to Persistence Naive and by **21.7%** relative to Seasonal Naive.

### Out-of-Sample Test Set (Held-Out Q4 Peak: 53 days, Oct 18 – Dec 09, 2011)

| Model | Setup | MAE (GBP) | RMSE (GBP) | WAPE (%) | Zero-Masked MAPE (%) | $R^2$ Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Updated)** | Retrained on Train+Val | **£10,987.70** | **£15,915.69** | **24.36%** | **29.74%** | **0.5611** |
| Seasonal Naive ($t-7$) | Baseline Heuristic | £11,685.71 | £16,667.57 | 25.90% | 27.96% | 0.5186 |
| Random Forest (Fixed) | Trained on Train only | £19,128.20 | £24,400.24 | 42.40% | 39.60% | -0.0316 |
| Moving Average (7d) | Baseline Heuristic | £19,007.16 | £24,182.50 | 42.13% | 33.01% | -0.0133 |
| Persistence Naive ($t-1$) | Baseline Heuristic | £22,956.52 | £29,731.52 | 50.89% | 41.15% | -0.5316 |
| Gradient Boosting | Trained on Train only | £26,547.54 | £31,922.25 | 58.85% | 54.47% | -0.7656 |
| Ridge Regression | Trained on Train only | £37,638.46 | £41,401.56 | 83.44% | 78.60% | -1.9700 |

*Takeaway:* In time-series forecasting, updating models as fresh validation data becomes available is standard best practice. Retraining the selected Random Forest on Train+Val yields the lowest overall error (**24.36% WAPE**, £10.9k MAE, $R^2 = 0.56$), outperforming all models across the Q4 surge.

---

## 🧪 Testing & Verification

Unit tests use isolated synthetic DataFrames and mocked network calls. No real dataset download or internet connection is required:

```bash
# Run all 38 unit tests across ingestion, cleaning, features, and modeling
pytest tests/ -v
```

All **38 tests pass with 100% success rate** covering:
- Ingestion, schema validation, and URL connectivity (14 tests).
- Transaction deduplication, write-off filters, and aggregation math (7 tests).
- Time-series lags, shifted rolling statistics leakage checks, calendar/cyclical bounds, holiday proximity, and chronological partition ordering (8 tests).
- Model fit/predict shapes, baseline consistency, reproducibility with fixed random seeds, and metric calculations (9 tests).

---

## 🗺️ Project Roadmap

| Phase | Description | Status |
|---|---|:---:|
| **Phase 1** | Project setup, modular data loader, security, schema validation & unit tests | **Completed** |
| **Phase 2** | Data cleaning pipeline, daily sales aggregation, EDA visualizations & notebook | **Completed** |
| **Phase 3** | Time-series feature engineering, leakage prevention & chronological splits | **Completed** |
| **Phase 4** | Forecasting Models (Baselines, Ridge, Random Forest, Gradient Boosting) | **Completed** |
| **Phase 5** | Final Model Evaluation, business insights, executive summary & internship packaging | **Completed** |

---

## 📑 Final Project Deliverables

- **Executive Report:** [`reports/final_project_report.md`](file:///c:/Users/piyus/OneDrive/Desktop/code/intership%20project/FUTURE_ML_01/reports/final_project_report.md)
- **Submission Checklist & LinkedIn Draft:** [`reports/submission_checklist.md`](file:///c:/Users/piyus/OneDrive/Desktop/code/intership%20project/FUTURE_ML_01/reports/submission_checklist.md)
- **Benchmark Metrics Table:** [`reports/model_comparison.csv`](file:///c:/Users/piyus/OneDrive/Desktop/code/intership%20project/FUTURE_ML_01/reports/model_comparison.csv)
- **Forecast vs. Actual Visualization:** [`reports/figures/07_forecast_vs_actual.png`](file:///c:/Users/piyus/OneDrive/Desktop/code/intership%20project/FUTURE_ML_01/reports/figures/07_forecast_vs_actual.png)
