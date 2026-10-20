"""Data Loading and Raw Validation Module for Retail Sales & Demand Forecasting.

This module provides reliable utilities to:
- Verify remote dataset accessibility.
- Download and safely extract raw datasets to the local filesystem.
- Load spreadsheet data (.xlsx, .xls, .csv) into pandas DataFrames.
- Perform non-destructive data validation (schema checks, missing values, duplicates,
  date parsing, anomalies such as negative quantities/prices) without dropping records.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import zipfile
from typing import Any

import pandas as pd
import requests

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Candidate dataset: UCI Online Retail Dataset (ID 352)
UCI_ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DEFAULT_EXCEL_FILENAME = "Online Retail.xlsx"

DEFAULT_REQUIRED_COLUMNS: list[str] = [
    "InvoiceNo",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
]

EXPECTED_COLUMNS: list[str] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Set up module logger
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


# ------------------------------------------------------------------------------
# URL Accessibility & Download Functions
# ------------------------------------------------------------------------------

def verify_source_accessible(url: str = UCI_ONLINE_RETAIL_URL, timeout: int = 10) -> bool:
    """Verify that the remote dataset URL is reachable.

    Performs a lightweight HTTP HEAD or streaming GET request without downloading
    the full body.

    Parameters
    ----------
    url : str
        URL of the dataset to verify.
    timeout : int, default 10
        Connection and read timeout in seconds.

    Returns
    -------
    bool
        True if the resource responded with an HTTP 200 OK status code.

    Raises
    ------
    requests.exceptions.RequestException
        If the server is unreachable or returns a non-200 HTTP status.
    """
    logger.info("Verifying accessibility of remote source: %s", url)
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        # Some servers disallow HEAD or return 405 Method Not Allowed; fallback to stream GET
        if response.status_code in (403, 405):
            response = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        logger.info("Remote source is accessible (HTTP %s).", response.status_code)
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to access remote dataset URL '%s': %s", url, exc)
        raise


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> list[Path]:
    """Safely extract all members of a zip file, preventing Zip Slip path traversal.

    Parameters
    ----------
    zip_path : Path
        Path to the zip archive.
    target_dir : Path
        Destination directory where files should be extracted.

    Returns
    -------
    list[Path]
        List of extracted file paths.

    Raises
    ------
    ValueError
        If an archive member attempts directory traversal outside target_dir.
    zipfile.BadZipFile
        If the file is not a valid zip archive.
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            # Resolve the intended absolute destination
            member_path = (target_dir / member.filename).resolve()
            # Ensure the resolved path remains strictly within target_dir
            if not str(member_path).startswith(str(target_dir)):
                raise ValueError(
                    f"Security Error: Archive member '{member.filename}' attempts "
                    f"path traversal outside target directory '{target_dir}'."
                )

        archive.extractall(target_dir)
        for member in archive.infolist():
            extracted_paths.append((target_dir / member.filename).resolve())

    return extracted_paths


def download_and_extract_data(
    url: str = UCI_ONLINE_RETAIL_URL,
    target_dir: Path | str = RAW_DATA_DIR,
    force: bool = False,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,  # 1 MB chunks
) -> Path:
    """Download the dataset archive and safely extract it into the raw data directory.

    If the uncompressed dataset (e.g., 'Online Retail.xlsx') already exists in
    target_dir and `force=False`, download is skipped.

    Parameters
    ----------
    url : str
        Direct download URL for the zip archive.
    target_dir : Path or str, default RAW_DATA_DIR
        Directory where raw data should be stored.
    force : bool, default False
        If True, re-downloads even if the dataset already exists.
    timeout : int, default 60
        Network timeout in seconds.
    chunk_size : int, default 1MB
        Download chunk size in bytes.

    Returns
    -------
    Path
        Path to the primary extracted spreadsheet file (.xlsx).

    Raises
    ------
    requests.exceptions.RequestException
        If the download fails due to network or server issues.
    zipfile.BadZipFile
        If the downloaded archive is corrupted or not a valid zip.
    FileNotFoundError
        If no spreadsheet file is found after extraction.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if extracted file already exists
    potential_file = target_path / DEFAULT_EXCEL_FILENAME
    if not force and potential_file.is_file():
        logger.info(
            "Raw dataset already exists at '%s'. Skipping download. (Use force=True to re-download).",
            potential_file,
        )
        return potential_file

    logger.info("Initiating download from: %s", url)
    verify_source_accessible(url, timeout=min(timeout, 15))

    zip_dest = target_path / "online_retail_raw.zip"
    temp_zip_dest = target_path / "online_retail_raw.zip.tmp"

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total_bytes = 0
            with open(temp_zip_dest, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f_out.write(chunk)
                        total_bytes += len(chunk)

        # Atomic rename after full download completes
        if temp_zip_dest.exists():
            temp_zip_dest.replace(zip_dest)

        logger.info(
            "Download completed successfully (%0.2f MB). Extracting...",
            total_bytes / (1024 * 1024),
        )

        extracted_files = _safe_extract_zip(zip_dest, target_path)
        logger.info("Extracted %d file(s) to '%s'.", len(extracted_files), target_path)

        # Find the primary Excel file
        excel_candidates = [
            f for f in extracted_files if f.suffix.lower() in (".xlsx", ".xls")
        ]
        if not excel_candidates:
            # Fallback search in target_path
            excel_candidates = list(target_path.glob("*.xlsx")) + list(target_path.glob("*.xls"))

        if not excel_candidates:
            raise FileNotFoundError(
                f"No Excel file found in '{target_path}' after extracting '{zip_dest}'."
            )

        primary_file = excel_candidates[0]
        logger.info("Primary raw dataset ready at: %s", primary_file)
        return primary_file

    except Exception as exc:
        # Clean up partial temporary file if an error occurs
        if temp_zip_dest.exists():
            try:
                temp_zip_dest.unlink()
            except OSError:
                pass
        logger.error("Download or extraction failed: %s", exc)
        raise


# ------------------------------------------------------------------------------
# Data Loading Functions
# ------------------------------------------------------------------------------

def load_raw_data(
    file_path: Path | str,
    sheet_name: str | int = 0,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load the raw dataset spreadsheet into a pandas DataFrame.

    Supports Excel (.xlsx, .xls) and CSV (.csv) formats.

    Parameters
    ----------
    file_path : Path or str
        Path to the dataset file.
    sheet_name : str or int, default 0
        Worksheet name or index (for Excel files).
    nrows : int, optional
        Number of rows to read (useful for quick previews or testing).

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame without any records removed.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist on disk.
    ValueError
        If file extension is unsupported or file cannot be parsed.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. "
            f"Please verify the path or run download_and_extract_data()."
        )

    ext = path.suffix.lower()
    logger.info("Loading raw data from '%s' (extension: %s)...", path, ext)

    if ext in (".xlsx", ".xls"):
        try:
            # openpyxl engine is standard for .xlsx
            engine = "openpyxl" if ext == ".xlsx" else None
            df = pd.read_excel(path, sheet_name=sheet_name, nrows=nrows, engine=engine)
        except ImportError as exc:
            raise ImportError(
                "Missing Excel engine 'openpyxl'. Install it via: pip install openpyxl"
            ) from exc
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file at '{path}': {exc}") from exc
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file at '{path}': {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Expected .xlsx, .xls, or .csv."
        )

    logger.info(
        "Successfully loaded %d rows and %d columns from '%s'.",
        len(df),
        len(df.columns),
        path.name,
    )
    return df


# ------------------------------------------------------------------------------
# Data Validation Functions (Non-destructive)
# ------------------------------------------------------------------------------

def validate_raw_data(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Validate raw DataFrame schema, missing values, duplicates, and anomalies.

    This function does NOT modify or drop any records. It returns a comprehensive
    dictionary detailing the health and characteristics of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to validate.
    required_columns : list[str], optional
        Columns that must be present in the DataFrame.
        Defaults to DEFAULT_REQUIRED_COLUMNS.

    Returns
    -------
    dict[str, Any]
        Dictionary with validation results:
        - 'is_valid': bool (True if all required columns present)
        - 'total_rows': int
        - 'total_columns': int
        - 'columns': list of column names
        - 'missing_required_columns': list of missing required column names
        - 'unexpected_columns': list of columns not in EXPECTED_COLUMNS
        - 'missing_values': dict mapping column name -> {count, percentage}
        - 'duplicate_records_count': int
        - 'duplicate_records_percentage': float
        - 'negative_quantities_count': int (indicates cancellations/returns)
        - 'zero_or_negative_unit_prices_count': int (adjustments/free items)
        - 'date_parsing': dict with parse status, unparseable count, min/max dates
        - 'memory_usage_mb': float

    Raises
    ------
    ValueError
        If df is empty or not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty (0 rows).")

    if required_columns is None:
        required_columns = DEFAULT_REQUIRED_COLUMNS

    actual_columns = list(df.columns)
    missing_required = [col for col in required_columns if col not in actual_columns]
    unexpected = [col for col in actual_columns if col not in EXPECTED_COLUMNS]

    # Missing values assessment
    missing_counts = df.isnull().sum().to_dict()
    missing_summary = {
        col: {
            "missing_count": int(count),
            "missing_percentage": round(float(count / len(df)) * 100, 2),
        }
        for col, count in missing_counts.items()
    }

    # Duplicate records
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round(float(duplicate_count / len(df)) * 100, 2)

    # Date parsing check
    date_summary: dict[str, Any] = {
        "column_found": "InvoiceDate" in df.columns,
        "is_parsed_as_datetime": False,
        "unparseable_count": 0,
        "min_date": None,
        "max_date": None,
    }

    if "InvoiceDate" in df.columns:
        date_series = df["InvoiceDate"]
        if pd.api.types.is_datetime64_any_dtype(date_series):
            date_summary["is_parsed_as_datetime"] = True
            valid_dates = date_series.dropna()
            if not valid_dates.empty:
                date_summary["min_date"] = str(valid_dates.min())
                date_summary["max_date"] = str(valid_dates.max())
        else:
            # Try parsing non-destructively
            parsed = pd.to_datetime(date_series, errors="coerce")
            unparseable = int(parsed.isna().sum() - date_series.isna().sum())
            date_summary["unparseable_count"] = max(0, unparseable)
            valid_dates = parsed.dropna()
            if not valid_dates.empty:
                date_summary["min_date"] = str(valid_dates.min())
                date_summary["max_date"] = str(valid_dates.max())

    # Quantity check (negative quantities identify returns/cancellations)
    negative_qty_count = 0
    if "Quantity" in df.columns:
        numeric_qty = pd.to_numeric(df["Quantity"], errors="coerce")
        negative_qty_count = int((numeric_qty < 0).sum())

    # Unit price check (negative or zero unit prices)
    negative_or_zero_price_count = 0
    if "UnitPrice" in df.columns:
        numeric_price = pd.to_numeric(df["UnitPrice"], errors="coerce")
        negative_or_zero_price_count = int((numeric_price <= 0).sum())

    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)

    is_valid = len(missing_required) == 0

    return {
        "is_valid": is_valid,
        "total_rows": len(df),
        "total_columns": len(actual_columns),
        "columns": actual_columns,
        "missing_required_columns": missing_required,
        "unexpected_columns": unexpected,
        "missing_values": missing_summary,
        "duplicate_records_count": duplicate_count,
        "duplicate_records_percentage": duplicate_percentage,
        "negative_quantities_count": negative_qty_count,
        "zero_or_negative_unit_prices_count": negative_or_zero_price_count,
        "date_parsing": date_summary,
        "memory_usage_mb": memory_mb,
    }


def format_validation_summary(report: dict[str, Any]) -> str:
    """Format the validation results dictionary into a readable markdown/text report.

    Parameters
    ----------
    report : dict[str, Any]
        Output dictionary from validate_raw_data().

    Returns
    -------
    str
        Human-readable summary report.
    """
    lines: list[str] = [
        "=" * 70,
        "DATASET VALIDATION SUMMARY REPORT",
        "=" * 70,
        f"Validation Status: {'PASSED (Schema valid)' if report['is_valid'] else 'FAILED (Missing required columns)'}",
        f"Total Records:      {report['total_rows']:,}",
        f"Total Columns:      {report['total_columns']}",
        f"Memory Usage:       {report['memory_usage_mb']:.2f} MB",
        "",
        "COLUMNS PRESENT:",
        f"  {', '.join(report['columns'])}",
    ]

    if report["missing_required_columns"]:
        lines.append(f"  WARNING - Missing required: {report['missing_required_columns']}")

    lines.extend([
        "",
        "MISSING VALUES BREAKDOWN:",
    ])
    for col, stats in report["missing_values"].items():
        count = stats["missing_count"]
        pct = stats["missing_percentage"]
        lines.append(f"  - {col:<15}: {count:>8,} missing ({pct:>5.2f}%)")

    lines.extend([
        "",
        "DUPLICATE RECORDS:",
        f"  Count:      {report['duplicate_records_count']:,} rows ({report['duplicate_records_percentage']:.2f}%)",
        "",
        "BUSINESS LOGIC ANOMALIES (Recorded, not deleted):",
        f"  - Negative Quantities (Returns/Cancellations): {report['negative_quantities_count']:,}",
        f"  - Zero or Negative Unit Prices (Adjustments):   {report['zero_or_negative_unit_prices_count']:,}",
        "",
        "DATE PARSING:",
        f"  - InvoiceDate Found:     {report['date_parsing']['column_found']}",
        f"  - Already Datetime:      {report['date_parsing']['is_parsed_as_datetime']}",
        f"  - Unparseable Dates:     {report['date_parsing']['unparseable_count']}",
        f"  - Earliest Transaction:  {report['date_parsing']['min_date']}",
        f"  - Latest Transaction:    {report['date_parsing']['max_date']}",
        "=" * 70,
    ])

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Command Line Interface (CLI)
# ------------------------------------------------------------------------------

def main() -> None:
    """Entry point for executing the data loader module directly."""
    parser = argparse.ArgumentParser(
        description="Retail Sales & Demand Forecasting: Data Loader and Validator"
    )
    parser.add_argument(
        "--verify-url",
        action="store_true",
        help="Verify accessibility of the remote UCI dataset URL without downloading.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the UCI Online Retail dataset into data/raw/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the raw dataset file already exists.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the raw dataset and display a comprehensive summary report.",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Custom path to raw spreadsheet file for loading and validation.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Number of rows to preview (default loads all rows).",
    )

    args = parser.parse_args()

    # If no flags provided, display help
    if not (args.verify_url or args.download or args.validate):
        parser.print_help()
        print("\n[Tip] Example quick start:")
        print("  python src/data_loader.py --verify-url")
        print("  python src/data_loader.py --download")
        print("  python src/data_loader.py --validate")
        return

    if args.verify_url:
        try:
            accessible = verify_source_accessible()
            print(f"Source URL is accessible: {accessible}")
        except Exception as exc:
            print(f"Source accessibility check failed: {exc}", file=sys.stderr)
            sys.exit(1)

    dataset_path: Path | None = None
    if args.file_path:
        dataset_path = Path(args.file_path).resolve()
    else:
        dataset_path = RAW_DATA_DIR / DEFAULT_EXCEL_FILENAME

    if args.download:
        try:
            dataset_path = download_and_extract_data(force=args.force)
            print(f"Dataset successfully downloaded and available at: {dataset_path}")
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.validate:
        if dataset_path is None or not dataset_path.is_file():
            print(
                f"Cannot validate: Dataset file '{dataset_path}' does not exist. "
                f"Run with --download first or specify --file-path.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            print(f"Loading '{dataset_path}' for validation...")
            df = load_raw_data(dataset_path, nrows=args.sample_rows)
            report = validate_raw_data(df)
            print(format_validation_summary(report))
        except Exception as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
