# Data Directory

This directory stores datasets used in the **Sales & Demand Forecasting** machine learning project.

## Directory Structure

```text
data/
├── raw/         <- Original, immutable raw data downloads (never modified or committed)
├── processed/   <- Cleaned, transformed, and feature-engineered datasets ready for modeling
└── README.md    <- Documentation and dataset provenance (this file)
```

## Dataset Candidate: UCI Online Retail Dataset

- **Dataset Name**: Online Retail
- **UCI Repository ID**: 352
- **Source URL**: [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Direct Download Archive**: `https://archive.ics.uci.edu/static/public/352/online+retail.zip`
- **Citation**:
  > Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining*. Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208. DOI: [10.1057/dbm.2012.17](https://doi.org/10.1057/dbm.2012.17).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset Schema

| Variable | Type | Description |
|---|---|---|
| `InvoiceNo` | Nominal / String | 6-digit number uniquely assigned to each transaction. Starts with 'C' for cancellations. |
| `StockCode` | Nominal / String | 5-digit integral product/item code uniquely assigned to each distinct product. |
| `Description` | String | Product name/description. |
| `Quantity` | Numeric / Integer | Quantities of each product per transaction. Negative values represent returns/cancellations. |
| `InvoiceDate` | Datetime | Date and time when the transaction was generated (2010-12-01 to 2011-12-09). |
| `UnitPrice` | Numeric / Float | Product price per unit in Sterling (£). |
| `CustomerID` | Nominal / Integer | 5-digit number uniquely assigned to each customer. Missing for guest checkouts. |
| `Country` | Nominal / String | Name of the country where the customer resides. |

## Important Data Handling Rules

1. **Raw data is immutable**: Never edit, overwrite, or delete files directly inside `data/raw/`. Any preprocessing steps must produce new files in `data/processed/`.
2. **Git tracking**: Neither `data/raw/` nor `data/processed/` dataset files are committed to Git. The `.gitignore` file enforces this rule to prevent bloating the repository with large binary/tabular assets.
