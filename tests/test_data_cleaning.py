"""Unit tests for RetailIQ data validation, mapping, and cleaning pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.data_cleaning import clean_retail_data
from src.data_validation import detect_outliers_iqr, map_columns, validate_dataset


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
  """Creates a synthetic raw retail DataFrame with unstandardized column names,

  duplicates, missing values, and outliers.
  """
  return pd.DataFrame({
      "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-3", "ORD-4", "ORD-5"],
      "order_date": [
          "2017-01-05",
          "2017-01-06",
          "2017-01-07",
          "2017-01-07",
          "2017-01-08",
          "2017-01-09",
      ],
      "ship_date": [
          "2017-01-08",
          "2017-01-10",
          "2017-01-11",
          "2017-01-11",
          None,
          "2017-01-14",
      ],
      "customer_segment": [
          "Consumer",
          "Corporate",
          "Home Office",
          "Home Office",
          "Consumer",
          "Consumer",
      ],
      "region": ["West", "East", "Central", "Central", "South", "West"],
      "category": [
          "Technology",
          "Furniture",
          "Office Supplies",
          "Office Supplies",
          "Furniture",
          "Technology",
      ],
      "sub_category": [
          "Phones",
          "Chairs",
          "Paper",
          "Paper",
          "Tables",
          "Accessories",
      ],
      "product_name": [
          "Smartphone Alpha",
          "Ergo Chair",
          "A4 Paper",
          "A4 Paper",
          "Conference Desk",
          "Mouse Pad",
      ],
      "sales": [500.0, 250.0, 50.0, 50.0, 1200.0, 25.0],
      "profit": [100.0, -30.0, 12.0, 12.0, -450.0, 8.0],
      "quantity": [2, 1, 5, 5, 4, 1],
      "discount": [0.10, 0.40, 0.0, 0.0, 0.60, 0.0],
  })


def test_map_columns(sample_raw_data):
  """Tests that alias mapping correctly identifies canonical columns."""
  matched, unmatched = map_columns(sample_raw_data)
  assert "Order Date" in matched
  assert "Sales" in matched
  assert "Profit" in matched
  assert "Region" in matched
  assert matched["Order Date"] == "order_date"
  assert matched["Sales"] == "sales"


def test_validate_dataset(sample_raw_data):
  """Tests dataset health and conformance report generation."""
  report = validate_dataset(sample_raw_data)
  assert report.is_valid is True
  assert report.total_rows == 6
  assert report.duplicate_rows == 1  # ORD-3 is duplicated
  assert len(report.missing_required_columns) == 0


def test_clean_retail_data(sample_raw_data):
  """Tests that cleaning deduplicates, standardizes schema, and engineers features."""
  cleaned_df, audit = clean_retail_data(
      sample_raw_data, handle_outliers=False, save_processed=False
  )

  # Check deduplication (6 initial -> 5 final)
  assert len(cleaned_df) == 5
  assert audit["duplicates_removed"] == 1

  # Check standardized columns
  assert "Order Date" in cleaned_df.columns
  assert "Sales" in cleaned_df.columns
  assert "Profit" in cleaned_df.columns

  # Check feature engineering
  assert "Order_Year" in cleaned_df.columns
  assert "Order_Month" in cleaned_df.columns
  assert "Profit_Margin_Pct" in cleaned_df.columns
  assert "Unit_Price" in cleaned_df.columns
  assert "Shipping_Duration_Days" in cleaned_df.columns

  # Verify values
  row_first = cleaned_df[cleaned_df["Order ID"] == "ORD-1"].iloc[0]
  assert row_first["Order_Year"] == 2017
  assert row_first["Profit_Margin_Pct"] == 20.0
  assert row_first["Unit_Price"] == 250.0


def test_outlier_detection():
  """Tests IQR outlier calculation on skewed numeric series."""
  df = pd.DataFrame(
      {"Sales": [10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 12.0, 11.0, 2500.0]}
  )
  outlier_res = detect_outliers_iqr(df, columns=["Sales"])
  assert "Sales" in outlier_res
  assert outlier_res["Sales"]["count"] == 1
  assert outlier_res["Sales"]["max"] == 2500.0
