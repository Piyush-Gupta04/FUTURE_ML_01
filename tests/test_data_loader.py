"""Unit tests for src.data_loader module.

All tests utilize synthetic in-memory DataFrames or temporary files.
No external network requests or real dataset downloads are required.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_loader import (
    DEFAULT_EXCEL_FILENAME,
    _safe_extract_zip,
    download_and_extract_data,
    format_validation_summary,
    load_raw_data,
    validate_raw_data,
    verify_source_accessible,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Provide a minimal valid synthetic retail DataFrame."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536366", "C536367", "536368"],
        "StockCode": ["85123A", "71053", "84406B", "84029G"],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "CREAM CUPID HEARTS COAT HANGER",
            "KNITTED UNION FLAG HOT WATER BOTTLE",
        ],
        "Quantity": [6, 6, -1, 4],  # Contains 1 cancellation/negative quantity
        "InvoiceDate": [
            "2010-12-01 08:26:00",
            "2010-12-01 08:28:00",
            "2010-12-01 08:34:00",
            "2010-12-01 09:00:00",
        ],
        "UnitPrice": [2.55, 3.39, 2.75, 0.00],  # Contains 1 zero unit price
        "CustomerID": [17850.0, 17850.0, 13047.0, None],  # Contains 1 missing CustomerID
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France"],
    })


@pytest.fixture
def sample_df_with_duplicates(sample_valid_df: pd.DataFrame) -> pd.DataFrame:
    """Provide a synthetic DataFrame with duplicate rows."""
    duplicate_row = sample_valid_df.iloc[[0]]
    return pd.concat([sample_valid_df, duplicate_row], ignore_index=True)


# ------------------------------------------------------------------------------
# Tests for Data Validation
# ------------------------------------------------------------------------------

def test_validate_raw_data_valid(sample_valid_df: pd.DataFrame) -> None:
    """Test validation on a well-formed DataFrame."""
    report = validate_raw_data(sample_valid_df)

    assert report["is_valid"] is True
    assert report["total_rows"] == 4
    assert report["total_columns"] == 8
    assert report["missing_required_columns"] == []
    assert report["negative_quantities_count"] == 1
    assert report["zero_or_negative_unit_prices_count"] == 1
    assert report["duplicate_records_count"] == 0
    assert report["missing_values"]["CustomerID"]["missing_count"] == 1
    assert report["date_parsing"]["column_found"] is True
    assert report["date_parsing"]["min_date"] == "2010-12-01 08:26:00"
    assert report["date_parsing"]["max_date"] == "2010-12-01 09:00:00"


def test_validate_raw_data_missing_required_columns() -> None:
    """Test validation detects missing required columns."""
    df = pd.DataFrame({
        "StockCode": ["85123A"],
        "Quantity": [5],
    })
    report = validate_raw_data(df)

    assert report["is_valid"] is False
    assert "InvoiceNo" in report["missing_required_columns"]
    assert "InvoiceDate" in report["missing_required_columns"]
    assert "UnitPrice" in report["missing_required_columns"]


def test_validate_raw_data_does_not_mutate_or_drop_records(
    sample_df_with_duplicates: pd.DataFrame,
) -> None:
    """Test that validation is non-destructive and preserves all rows."""
    original_len = len(sample_df_with_duplicates)
    original_data = sample_df_with_duplicates.copy(deep=True)

    report = validate_raw_data(sample_df_with_duplicates)

    # DataFrame should remain completely untouched
    assert len(sample_df_with_duplicates) == original_len
    pd.testing.assert_frame_equal(sample_df_with_duplicates, original_data)
    assert report["duplicate_records_count"] == 1
    assert report["duplicate_records_percentage"] == round(1 / original_len * 100, 2)


def test_validate_raw_data_date_parsing_with_unparseable() -> None:
    """Test validation flags unparseable date values without crashing."""
    df = pd.DataFrame({
        "InvoiceNo": ["1001", "1002"],
        "Quantity": [1, 2],
        "UnitPrice": [10.0, 15.0],
        "InvoiceDate": ["2021-01-01 10:00:00", "not-a-valid-date"],
    })
    report = validate_raw_data(df)

    assert report["date_parsing"]["unparseable_count"] == 1
    assert report["date_parsing"]["min_date"] == "2021-01-01 10:00:00"


def test_validate_raw_data_empty_or_invalid_input() -> None:
    """Test that passing an empty DataFrame or non-DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_raw_data(pd.DataFrame())

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        validate_raw_data("not a dataframe")  # type: ignore[arg-type]


def test_format_validation_summary(sample_valid_df: pd.DataFrame) -> None:
    """Test formatted report generation."""
    report = validate_raw_data(sample_valid_df)
    summary_text = format_validation_summary(report)

    assert "DATASET VALIDATION SUMMARY REPORT" in summary_text
    assert "PASSED (Schema valid)" in summary_text
    assert "InvoiceNo" in summary_text
    assert "Negative Quantities" in summary_text


# ------------------------------------------------------------------------------
# Tests for Data Loading
# ------------------------------------------------------------------------------

def test_load_raw_data_file_not_found(tmp_path: Path) -> None:
    """Test load_raw_data raises FileNotFoundError for missing path."""
    non_existent = tmp_path / "does_not_exist.xlsx"
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_raw_data(non_existent)


def test_load_raw_data_csv(tmp_path: Path, sample_valid_df: pd.DataFrame) -> None:
    """Test loading a synthetic dataset from CSV."""
    csv_file = tmp_path / "synthetic_retail.csv"
    sample_valid_df.to_csv(csv_file, index=False)

    loaded = load_raw_data(csv_file)
    assert len(loaded) == 4
    assert list(loaded.columns) == list(sample_valid_df.columns)


def test_load_raw_data_excel(tmp_path: Path, sample_valid_df: pd.DataFrame) -> None:
    """Test loading a synthetic dataset from Excel (.xlsx)."""
    xlsx_file = tmp_path / "synthetic_retail.xlsx"
    sample_valid_df.to_excel(xlsx_file, index=False, engine="openpyxl")

    loaded = load_raw_data(xlsx_file)
    assert len(loaded) == 4
    assert "InvoiceNo" in loaded.columns


def test_load_raw_data_unsupported_format(tmp_path: Path) -> None:
    """Test that unsupported file extensions raise ValueError."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("dummy content")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_raw_data(txt_file)


# ------------------------------------------------------------------------------
# Tests for Safe Extraction & Download Logic
# ------------------------------------------------------------------------------

def test_safe_extract_zip_prevents_slip(tmp_path: Path) -> None:
    """Test that _safe_extract_zip refuses archives attempting directory traversal."""
    bad_zip_path = tmp_path / "malicious.zip"
    extract_target = tmp_path / "extracted"
    extract_target.mkdir()

    # Create a zip containing a member attempting path traversal
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("../evil.txt", "malicious payload")

    bad_zip_path.write_bytes(zip_buffer.getvalue())

    with pytest.raises(ValueError, match="Security Error: Archive member"):
        _safe_extract_zip(bad_zip_path, extract_target)


def test_download_skips_when_file_exists(tmp_path: Path) -> None:
    """Test download_and_extract_data skips download when raw file already exists."""
    existing_file = tmp_path / DEFAULT_EXCEL_FILENAME
    existing_file.write_text("dummy excel content")

    with patch("requests.get") as mock_get:
        result_path = download_and_extract_data(target_dir=tmp_path, force=False)
        assert result_path == existing_file
        mock_get.assert_not_called()


def test_download_handles_network_error(tmp_path: Path) -> None:
    """Test that network errors during download are raised cleanly."""
    with patch(
        "src.data_loader.verify_source_accessible",
        side_effect=requests.exceptions.ConnectionError("Connection failed"),
    ):
        with pytest.raises(requests.exceptions.RequestException):
            download_and_extract_data(target_dir=tmp_path, force=True)


def test_verify_source_accessible_success() -> None:
    """Test verify_source_accessible with mocked successful HTTP response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.head", return_value=mock_resp):
        assert verify_source_accessible("https://example.com/data.zip") is True
