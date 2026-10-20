# Final Project Report: Retail Sales & Demand Forecasting

**Project Identifier:** FUTURE_ML_01
**Project Title:** End-to-End Retail Sales & Demand Forecasting Pipeline
**Domain:** E-Commerce, Retail Analytics & Supply Chain Management
**Stack:** Python 3.14, pandas, NumPy, scikit-learn, matplotlib, seaborn, pytest
**Author:** Machine Learning Engineering Intern

---

## 1. Executive Summary

This project delivers an end-to-end, reproducible Machine Learning pipeline designed to forecast daily sales and product demand for an online retail business. Operating on 541,909 raw transactional records from the UCI Online Retail dataset (December 2010 to December 2011), the pipeline performs non-destructive schema validation, applies transparent business cleaning rules, aggregates transactions into continuous daily time series, constructs 31 leak-free time-series features, and benchmarks Machine Learning regressors against standard industry heuristics.

### Key Results Summary
- **Data Ingestion & Cleaning:** 541,909 raw records processed; 5,268 duplicates removed (0.97%) and 2,512 zero/negative price warehouse adjustments filtered; 534,129 valid transactions retained (98.56%).
- **Volume & Revenue Discovered:** The business generated **Â£9,748,131.07** in net revenue and sold **5,296,860 units** across 374 calendar days. Customer returns totaled **Â£893,979.73**, representing an **8.40%** return rate relative to gross sales.
- **Model Selection (Validation Set):** **Random Forest Regressor** demonstrated superior forecasting accuracy on the validation partition, achieving a **36.52% WAPE** and **Â£11,767.14 MAE** (a **45.4% error reduction** over Persistence Naive baseline and a **21.7% error reduction** over Seasonal Naive baseline).
- **Out-of-Sample Test Evaluation:** Evaluating across the peak holiday volume surge (October 18 to December 9, 2011), the retrained Random Forest achieved **24.36% WAPE**, **Â£10,987.70 MAE**, and an **$R^2$ score of 0.5611**, successfully capturing pre-holiday demand dynamics without data leakage.

---

## 2. Business Problem & Operational Context

In modern retail and e-commerce supply chains, stockouts lead to lost customer orders and brand attrition, while overstocking ties up working capital in perishable inventory and warehouse holding costs.

### Core Operational Questions
1. **Pacing & Replenishment:** How much net inventory will be demanded on any given day?
2. **Weekly Seasonality:** How do store closures and weekday shopping patterns affect daily staffing and fulfillment schedules?
3. **Holiday Surges:** When does the pre-Christmas wholesale ordering rush begin, when does it peak, and what safety stock is required?
4. **Returns Impact:** How significant are customer cancellations and returns, and how do they offset gross receipts?

---

## 3. Dataset Provenance & Data Cleaning Methodology

### 3.1 Provenance
- **Dataset Source:** [UCI Machine Learning Repository: Online Retail (ID: 352)](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Archive URL:** `https://archive.ics.uci.edu/static/public/352/online+retail.zip`
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)).
- **Citation:**
  > Daqing Chen, Sai Liang Sain, and Kun Guo (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining*, Journal of Database Marketing & Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208. DOI: 10.1057/dbm.2012.17.

### 3.2 Raw Schema & Observed Characteristics
The raw spreadsheet contains 8 features:
- `InvoiceNo`: 6-digit transaction identifier (starts with 'C' for cancellations).
- `StockCode`: 5-digit product item code.
- `Description`: Item description/name.
- `Quantity`: Numeric quantities per line item (negative for returns).
- `InvoiceDate`: Transaction timestamp (2010-12-01 08:26:00 to 2011-12-09 12:50:00).
- `UnitPrice`: Product price in Sterling (Â£).
- `CustomerID`: 5-digit customer identifier.
- `Country`: Customer country of residence (38 unique countries; 91.4% UK).

### 3.3 Transparent Cleaning Rules (Why Decisions Were Made)

```text
541,909 Raw Records
 â”œâ”€â”€ [Removed]  5,268 exact duplicate rows (0.97%) from client re-clicks or server logging retries.
 â”œâ”€â”€ [Removed]  2,512 records where UnitPrice <= 0:
 â”‚              - 1,454 records with missing Description and UnitPrice == 0.
 â”‚              - Internal warehouse inventory adjustments: "damages", "lost", "check", "thrown away".
 â”‚              - Exactly 2 accounting debt entries ("Adjust bad debt", -Â£11,062.06).
 â””â”€â”€ [Retained] 534,129 valid retail transactions (98.56% retained).
```

1. **Guest Checkout Preservation:** 135,080 records (24.93%) have missing `CustomerID`. Deleting these would discard **Â£1.44M** in genuine customer demand. These were retained and imputed with `-1` (Guest/Unknown Customer).
2. **Cancellation Accounting:** 10,624 records feature negative quantities. These were not silently deleted; line totals were computed as $\text{LineTotal} = \text{Quantity} \times \text{UnitPrice}$ so returns accurately counterbalance gross receipts.
3. **Daily Aggregation Metric Definitions:**
   - $\text{Gross Sales} = \sum (\text{LineTotal} \mid \text{Quantity} > 0)$
   - $\text{Returns Amount} = \sum |\text{LineTotal} \mid \text{Quantity} < 0|$
   - $\text{Net Sales} = \text{Gross Sales} - \text{Returns Amount}$
   - $\text{Net Quantity} = \text{Gross Units} - \text{Returned Units}$
   - Missing calendar days (e.g., Saturday closures) were re-indexed and explicitly filled with `0` so time-series algorithms process an unbroken daily timeline.

---

## 4. Exploratory Data Analysis & Empirical Findings

### 4.1 Missing Data Profile
`CustomerID` is the only substantial missing field (24.93%), representing guest checkouts. Missing `Description` (0.27%) overlaps 100% with `UnitPrice == 0` administrative records.

### 4.2 Time-Series Trajectory & Q4 Peak
- Mean daily net sales: **Â£26,064.52 / day** (Median: Â£22,233.15 / day).
- Baseline trading remains steady from January through August 2011 (Â£15k to Â£25k/day).
- Starting in September 2011, demand accelerates dramatically as wholesale buyers stock up for the Christmas gift season, reaching an all-time peak of **Â£111,958.62 on November 14, 2011**.

### 4.3 Day-of-Week Seasonality: The Saturday Closure Insight
- The business trades Sunday through Friday.
- **Saturdays consistently record 0 sales across all 53 weeks** in the recording period.
- **Thursday is the highest volume trading day**, averaging Â£38,500/day.

### 4.4 Catalog Concentration (Pareto Principle)
Top revenue-generating products include:
- `DOTCOM POSTAGE`
- `REGENCY CAKESTAND 3 TIER`
- `WHITE HANGING HEART T-LIGHT HOLDER`
- `PARTY BUNTING`

Demand follows a classic long-tail Pareto distribution, where the top 20% of catalog items generate over 75% of total sales volume.

---

## 5. Feature Engineering & Zero-Lookahead Architecture

To provide tabular regression models with temporal context while strictly preventing data leakage, 31 features were engineered:

```text
Feature Matrix (346 Clean Operating Days x 31 Columns)
â”œâ”€â”€ Target & Volume:   net_sales, gross_sales, returns_amount, net_quantity, total_orders
â”œâ”€â”€ Autoregressive:    net_sales_lag_1, lag_7, lag_14, lag_28
â”œâ”€â”€ Shifted Rolling:   rolling_mean_7, rolling_mean_14, rolling_mean_28
â”‚                      rolling_std_7, rolling_std_14, rolling_std_28
â”‚                      (STRICT LEAKAGE RULE: Computed on shift(1); date t never in window)
â”œâ”€â”€ Calendar Partitions: day_of_week, day_of_month, month, quarter, year, is_weekend
â”œâ”€â”€ Operational Flags:   is_saturday (closure flag), is_month_start, is_month_end
â”œâ”€â”€ Cyclical Encodings:  sin_day_of_week, cos_day_of_week, sin_month, cos_month
â””â”€â”€ Holiday Proximity:   is_holiday, days_until_holiday, days_since_holiday
```

### Warmup Period Handling
The maximum lag and rolling window spans 28 calendar days. The first 28 rows (December 1 to December 28, 2010) naturally have NaN values. `drop_warmup=True` removes these 28 warmup rows, yielding 346 clean rows with **0 missing values**.

### Strict Chronological Splitting
Random shuffling causes future data to contaminate the past. We partitioned data strictly in chronological order:

```text
[------------------ Train (70%) ------------------][--- Val (15%) ---][--- Test (15%) ---]
2010-12-29                                2011-08-27 2011-08-28   2011-10-17 2011-10-18   2011-12-09
           242 Days (Â£21.8k/day mean)                   51 Days (Â£30.7k)        53 Days (Â£36.5k)
```

$$\max(\text{Train}) < \min(\text{Val}) < \max(\text{Val}) < \min(\text{Test})$$

---

## 6. Model Benchmarking & Empirical Results

### 6.1 Evaluated Architectures
1. **Persistence Naive ($t-1$)**: Predicts yesterday's demand.
2. **Seasonal Naive ($t-7$)**: Predicts same day last week.
3. **Moving Average (7d)**: Predicts mean of preceding 7 days (shifted).
4. **Ridge Regression**: L2 regularized linear model with `StandardScaler` fit strictly on training data.
5. **Random Forest Regressor**: 100-tree ensemble (`max_depth=6`, `min_samples_leaf=3`, `random_state=42`).
6. **Gradient Boosting Regressor**: 100 sequential boosting estimators (`learning_rate=0.05`, `max_depth=3`).

### 6.2 Validation Partition Performance (Model Selection)

| Rank | Model | MAE (GBP) | RMSE (GBP) | WAPE (%) | Zero-Masked MAPE (%) | sMAPE (%) | $R^2$ Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ðŸ¥‡ | **Random Forest** | **Â£11,767.14** | **Â£17,512.14** | **36.52%** | **32.68%** | 55.69% | **0.3467** |
| ðŸ¥ˆ | Gradient Boosting | Â£13,443.90 | Â£19,526.83 | 41.72% | 35.39% | 69.42% | 0.1878 |
| ðŸ¥‰ | Seasonal Naive ($t-7$) | Â£15,028.01 | Â£21,135.68 | 46.64% | 49.60% | 43.67% | 0.0484 |
| 4 | Moving Average (7d) | Â£16,628.73 | Â£22,031.92 | 51.61% | 41.81% | 64.30% | -0.0340 |
| 5 | Ridge Regression | Â£17,276.10 | Â£22,911.70 | 53.62% | 48.35% | 87.24% | -0.1182 |
| 6 | Persistence Naive ($t-1$) | Â£21,562.17 | Â£26,884.67 | 66.92% | 56.12% | 97.00% | -0.5397 |

### 6.3 Test Partition Performance (Held-out Evaluation)

| Model | Training Configuration | MAE (GBP) | RMSE (GBP) | WAPE (%) | Zero-Masked MAPE (%) | $R^2$ Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Updated)** | Retrained on Train+Val | **Â£10,987.70** | **Â£15,915.69** | **24.36%** | **29.74%** | **0.5611** |
| Seasonal Naive ($t-7$) | Baseline Heuristic | Â£11,685.71 | Â£16,667.57 | 25.90% | 27.96% | 0.5186 |
| Random Forest (Fixed) | Trained on Train only | Â£19,128.20 | Â£24,400.24 | 42.40% | 39.60% | -0.0316 |
| Moving Average (7d) | Baseline Heuristic | Â£19,007.16 | Â£24,182.50 | 42.13% | 33.01% | -0.0133 |
| Persistence Naive ($t-1$) | Baseline Heuristic | Â£22,956.52 | Â£29,731.52 | 50.89% | 41.15% | -0.5316 |
| Gradient Boosting | Trained on Train only | Â£26,547.54 | Â£31,922.25 | 58.85% | 54.47% | -0.7656 |
| Ridge Regression | Trained on Train only | Â£37,638.46 | Â£41,401.56 | 83.44% | 78.60% | -1.9700 |

### 6.4 Key Evaluation Insights
1. **Why Random Forest Wins on Validation:** Decision trees naturally segment feature space, handling the sharp binary contrast between Saturday closures (0 sales) and high-volume Thursdays without forcing a linear slope.
2. **The Test-Set Seasonality Shift:** In the test set (late October through December), daily volume surged to Â£36.5k/day. Seasonal Naive ($t-7$) performed remarkably well because prior-week observations were already elevated. Once Random Forest was retrained on `Train + Validation` (the standard operational practice before deployment), it incorporated the early autumn ramp and achieved the best overall result (**24.36% WAPE**, **$R^2 = 0.5611$**).
3. **Metric Interpretation Policy:** WAPE is confirmed as the robust benchmark metric because zero-sales days cause traditional MAPE to produce infinite errors. In validation, 8 days were zero-sales; in test, 7 days were zero-sales.

---

## 7. Business Recommendations & Inventory Strategy

*(Recommendations are derived from observed empirical patterns and standard supply chain practices, without asserting unverified causal claims.)*

### 1. Pre-Holiday Inventory Buildup (The Mid-September Ramp)
- **Observed Pattern:** Daily wholesale net sales accelerate in late September and peak around **November 14th (Â£111,958)**, approximately 6 weeks ahead of Christmas Day.
- **Actionable Strategy:** Inbound inventory purchasing for giftware should be scheduled to arrive at fulfillment centers by **mid-September**, rather than late October, to prevent stockouts during the primary wholesale purchasing window.

### 2. Operational Fulfillment on Saturdays
- **Observed Pattern:** 0 customer sales across all 53 Saturdays in the dataset.
- **Actionable Strategy:** Packing, dispatch, and carrier pickup shifts can be eliminated on Saturdays, shifting labor hours to Sundays and Thursdays (the peak trading day, averaging Â£38.5k). Warehouse replenishment and stocktaking should be concentrated on Saturdays without disrupting live customer dispatches.

### 3. Customer Returns Financial Buffer (The 8.4% Rule)
- **Observed Pattern:** Customer returns and order cancellations accounted for **Â£893,979.73 (8.40% of gross revenue)**, remaining relatively stable between 6% and 12% monthly.
- **Actionable Strategy:** Financial planning and procurement models should maintain an **8.5% reserve buffer** against gross bookings. Reverse logistics workflows should be staffed to handle return surges immediately following holiday peak weeks.

### 4. High-Demand SKU Prioritization
- **Observed Pattern:** Top revenue items (`DOTCOM POSTAGE`, `REGENCY CAKESTAND`, `WHITE HANGING HEART`) drive disproportionate volume.
- **Actionable Strategy:** Implement dedicated safety stock buffers for the top 50 revenue SKUs, using the macro daily demand forecast as an envelope to allocate SKU-level reorder points.

---

## 8. Limitations, Operational Risks & Future Enhancements

### 8.1 Model & Data Limitations
1. **One-Step-Ahead Horizon:** The models forecast next-day sales ($t+1$) given true historical observations through day $t$. While ideal for daily fulfillment staffing and automated dispatch, multi-step forecasts ($t+1$ to $t+14$) require recursive rollouts where prediction errors compound over time.
2. **Single Annual Cycle:** The dataset spans 12.3 months. Because it contains only one complete annual cycle, models cannot statistically disentangle true calendar month effects from idiosyncratic 2011 market events without multi-year longitudinal data.
3. **Macro vs. SKU Granularity:** The pipeline forecasts aggregate daily business revenue and total units. While critical for warehouse capacity and cash flow planning, individual warehouse slotting requires SKU-level modeling.

### 8.2 Operational Risks
- **Supply Shock Vulnerability:** Statistical regression assumes historical purchasing behavior continues. Sudden macroeconomic disruptions, supplier lead-time delays, or carrier strikes will deviate from historical predictions.
- **Holiday Drift:** Black Friday and Easter dates change annually. The holiday feature generator must be updated annually with relevant calendar dates.

### 8.3 Recommended Next Steps (Phase 6 / Production Roadmap)
1. **SKU-Level Hierarchy:** Implement hierarchical reconciliation (e.g. MinT / Bottom-Up) to disaggregate macro daily forecasts into product-category and top-SKU level forecasts.
2. **Deep Learning Sequence Models:** Benchmark Temporal Fusion Transformers (TFT) or N-BEATS to capture complex multi-horizon seasonal interactions if multi-year data becomes available.
3. **Automated Weekly Retraining:** Deploy a weekly automated retraining job that appends newly finalized weekly sales to the training set, maintaining fresh rolling averages in production.

---

## 9. Conclusion

The `FUTURE_ML_01` Retail Sales & Demand Forecasting project successfully delivers a robust, transparent, and tested machine learning workflow. By coupling strict zero-lookahead feature engineering with baseline benchmarking, the pipeline provides clear operational guidance for inventory procurement, holiday safety stock planning, and warehouse labor scheduling. All source code, tests, and documentation are structured for turnkey reproducibility.
