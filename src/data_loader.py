"""RetailIQ Data Loader Module.

Handles robust dataset loading from local storage, Streamlit file uploads,
and multi-encoding fallback detection with metadata introspection.
"""

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from config.settings import (
    DEFAULT_PROCESSED_DATASET,
    DEFAULT_RAW_DATASET,
)
from src.data_cleaning import clean_retail_data


def read_csv_with_fallback_encoding(
    source: Union[str, Path, BytesIO],
    **kwargs: Any,
) -> pd.DataFrame:
  """Reads a CSV with automatic encoding fallback (utf-8, latin1, cp1252, iso-8859-1)."""
  encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
  last_err = None

  for enc in encodings:
    try:
      if isinstance(source, BytesIO):
        source.seek(0)
      return pd.read_csv(source, encoding=enc, **kwargs)
    except UnicodeDecodeError as e:
      last_err = e
      continue
    except Exception as e:
      raise e

  raise ValueError(
      f"Failed to read CSV with supported encodings {encodings}. Error: {last_err}"
  )


def load_raw_data(
    source: Optional[Union[str, Path, BytesIO]] = None,
) -> pd.DataFrame:
  """Loads raw retail data from specified file path, file object, or default raw storage."""
  target = source if source is not None else DEFAULT_RAW_DATASET

  if isinstance(target, (str, Path)):
    path = Path(target)
    if not path.exists():
      raise FileNotFoundError(
          f"Raw dataset file not found at: {path}. "
          "Please upload a valid CSV or ensure the sample dataset is placed in data/raw/."
      )

  df = read_csv_with_fallback_encoding(target)
  return df


def load_clean_data(
    source: Optional[Union[str, Path, BytesIO]] = None,
    force_reclean: bool = False,
) -> pd.DataFrame:
  """Loads cleaned retail data.

  If clean dataset does not exist or force_reclean is True, cleans raw data and
  persists it.
  """
  if not force_reclean and source is None and DEFAULT_PROCESSED_DATASET.exists():
    df = read_csv_with_fallback_encoding(DEFAULT_PROCESSED_DATASET)
    # Ensure date parsing
    if "Order Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Order Date"]):
      df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if "Ship Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Ship Date"]):
      df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    return df

  raw_df = load_raw_data(source)
  clean_df, _ = clean_retail_data(raw_df, save_processed=True)
  return clean_df


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
  """Computes high-level metadata and executive summary metrics for a DataFrame."""
  if df is None or df.empty:
    return {
        "rows": 0,
        "columns": 0,
        "date_range": "N/A",
        "total_revenue": 0.0,
        "total_profit": 0.0,
        "profit_margin": 0.0,
        "total_orders": 0,
        "total_customers": 0,
        "total_products": 0,
    }

  date_col = "Order Date" if "Order Date" in df.columns else None
  date_range = "N/A"
  if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    if pd.notna(min_date) and pd.notna(max_date):
      date_range = f"{min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}"

  sales = float(df["Sales"].sum()) if "Sales" in df.columns else 0.0
  profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0.0
  margin = round((profit / sales) * 100, 2) if sales > 0 else 0.0
  orders = int(df["Order ID"].nunique()) if "Order ID" in df.columns else len(df)
  customers = int(df["Customer ID"].nunique()) if "Customer ID" in df.columns else 0
  products = int(df["Product ID"].nunique()) if "Product ID" in df.columns else 0

  return {
      "rows": len(df),
      "columns": len(df.columns),
      "date_range": date_range,
      "total_revenue": round(sales, 2),
      "total_profit": round(profit, 2),
      "profit_margin": margin,
      "total_orders": orders,
      "total_customers": customers,
      "total_products": products,
  }
