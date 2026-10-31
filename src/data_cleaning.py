"""RetailIQ Data Cleaning and Preprocessing Pipeline.

Performs schema standardization, missing-value imputation, duplicate removal,
date parsing, outlier handling, and business metric feature engineering.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import DEFAULT_PROCESSED_DATASET, REQUIRED_COLUMNS
from src.data_validation import map_columns


def parse_dates_robust(series: pd.Series) -> pd.Series:
  """Robustly parses date series supporting multiple US and ISO date conventions."""
  return pd.to_datetime(series, errors="coerce", format="mixed")


def clean_retail_data(
    df: pd.DataFrame,
    handle_outliers: bool = True,
    outlier_method: str = "clip",
    iqr_multiplier: float = 2.5,
    save_processed: bool = True,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
  """Full end-to-end data cleaning and transformation pipeline.

  Args:
      df: Raw input DataFrame
      handle_outliers: Whether to apply outlier treatment
      outlier_method: 'clip' to cap at IQR bounds, 'flag' to add boolean column,
        or 'none'
      iqr_multiplier: IQR threshold multiplier (default 2.5 for retail data)
      save_processed: Whether to persist the cleaned dataset to disk
      output_path: Destination path if saving

  Returns:
      cleaned_df: Cleaned and feature-enriched DataFrame
      audit: Dictionary summarizing preprocessing actions and metrics
  """
  if df is None or df.empty:
    raise ValueError("Input DataFrame is empty or None.")

  raw_shape = df.shape
  audit: Dict[str, Any] = {
      "initial_rows": raw_shape[0],
      "initial_cols": raw_shape[1],
      "duplicates_removed": 0,
      "missing_imputed": {},
      "outliers_treated": {},
      "actions": [],
  }

  df_clean = df.copy()

  # Step 1: Column Mapping to Canonical Schema
  matched, _ = map_columns(df_clean)
  inverse_map = {orig_col: canon for canon, orig_col in matched.items()}
  df_clean.rename(columns=inverse_map, inplace=True)
  audit["actions"].append(f"Mapped {len(matched)} columns to canonical schema.")

  # Verify minimum essential columns are present
  missing_critical = [c for c in ["Order Date", "Sales"] if c not in df_clean.columns]
  if missing_critical:
    raise ValueError(
        f"Critical columns missing after mapping: {missing_critical}"
    )

  # Step 2: Remove Duplicates
  initial_count = len(df_clean)
  df_clean.drop_duplicates(inplace=True)
  duplicates_dropped = initial_count - len(df_clean)
  audit["duplicates_removed"] = duplicates_dropped
  if duplicates_dropped > 0:
    audit["actions"].append(f"Removed {duplicates_dropped:,} duplicate records.")

  # Step 3: Date Parsing and Chronological Ordering
  if "Order Date" in df_clean.columns:
    df_clean["Order Date"] = parse_dates_robust(df_clean["Order Date"])
    # Drop rows where Order Date is completely invalid
    invalid_dates = df_clean["Order Date"].isna().sum()
    if invalid_dates > 0:
      df_clean = df_clean.dropna(subset=["Order Date"]).copy()
      audit["actions"].append(f"Dropped {invalid_dates} rows with unparseable Order Date.")

  if "Ship Date" in df_clean.columns:
    df_clean["Ship Date"] = parse_dates_robust(df_clean["Ship Date"])
    # If ship date missing, fallback to order date + 3 days median shipping
    ship_na = df_clean["Ship Date"].isna().sum()
    if ship_na > 0:
      df_clean["Ship Date"] = df_clean["Ship Date"].fillna(
          df_clean["Order Date"] + pd.Timedelta(days=3)
      )
      audit["missing_imputed"]["Ship Date"] = int(ship_na)

  # Step 4: Numerical Conversions and Missing Value Handling
  numeric_fields = ["Sales", "Profit", "Quantity", "Discount"]
  for col in numeric_fields:
    if col in df_clean.columns:
      df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
      missing_num = int(df_clean[col].isna().sum())
      if missing_num > 0:
        median_val = df_clean[col].median()
        if pd.isna(median_val):
          median_val = 0.0
        df_clean[col] = df_clean[col].fillna(median_val)
        audit["missing_imputed"][col] = missing_num
        audit["actions"].append(f"Imputed {missing_num} missing values in '{col}' with median ({median_val:.2f}).")

  # Fill missing categoricals
  cat_fields = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode", "State", "City"]
  for col in cat_fields:
    if col in df_clean.columns:
      missing_cat = int(df_clean[col].isna().sum())
      if missing_cat > 0:
        df_clean[col] = df_clean[col].fillna("Unknown").astype(str).str.strip()
        audit["missing_imputed"][col] = missing_cat
      else:
        df_clean[col] = df_clean[col].astype(str).str.strip()

  # Ensure Quantity is positive integer
  if "Quantity" in df_clean.columns:
    df_clean["Quantity"] = df_clean["Quantity"].clip(lower=1).round().astype(int)

  # Ensure Discount is bounded between 0 and 1
  if "Discount" in df_clean.columns:
    df_clean["Discount"] = df_clean["Discount"].clip(lower=0.0, upper=1.0)

  # Ensure Sales is non-negative
  if "Sales" in df_clean.columns:
    df_clean["Sales"] = df_clean["Sales"].clip(lower=0.0)

  # Step 5: Feature Engineering & Business Metrics
  if "Order Date" in df_clean.columns:
    df_clean["Order_Year"] = df_clean["Order Date"].dt.year
    df_clean["Order_Month"] = df_clean["Order Date"].dt.month
    df_clean["Order_Month_Name"] = df_clean["Order Date"].dt.strftime("%b")
    df_clean["Order_Year_Month"] = df_clean["Order Date"].dt.to_period("M").astype(str)
    df_clean["Order_Day"] = df_clean["Order Date"].dt.day
    df_clean["Order_Day_Of_Week"] = df_clean["Order Date"].dt.day_name()
    df_clean["Order_Quarter"] = "Q" + df_clean["Order Date"].dt.quarter.astype(str)
    df_clean["Is_Weekend"] = df_clean["Order Date"].dt.dayofweek.isin([5, 6]).astype(int)

  if "Ship Date" in df_clean.columns and "Order Date" in df_clean.columns:
    ship_days = (df_clean["Ship Date"] - df_clean["Order Date"]).dt.days
    df_clean["Shipping_Duration_Days"] = ship_days.clip(lower=0)

  # Derived Financial Metrics
  if "Sales" in df_clean.columns and "Profit" in df_clean.columns:
    # Profit Margin % = (Profit / Sales) * 100
    df_clean["Profit_Margin_Pct"] = np.where(
        df_clean["Sales"] > 0,
        (df_clean["Profit"] / df_clean["Sales"]) * 100.0,
        0.0
    ).round(2)
    df_clean["Is_Profitable"] = (df_clean["Profit"] > 0).astype(int)

  if "Sales" in df_clean.columns and "Quantity" in df_clean.columns:
    df_clean["Unit_Price"] = np.where(
        df_clean["Quantity"] > 0,
        df_clean["Sales"] / df_clean["Quantity"],
        df_clean["Sales"]
    ).round(2)

  # Step 6: Outlier Handling
  if handle_outliers and outlier_method != "none":
    for col in ["Sales", "Profit"]:
      if col not in df_clean.columns:
        continue
      s = df_clean[col]
      q1 = s.quantile(0.25)
      q3 = s.quantile(0.75)
      iqr = q3 - q1
      low_b = q1 - iqr_multiplier * iqr
      high_b = q3 + iqr_multiplier * iqr

      outlier_mask = (s < low_b) | (s > high_b)
      outlier_count = int(outlier_mask.sum())

      if outlier_count > 0:
        audit["outliers_treated"][col] = {
            "count": outlier_count,
            "lower_bound": round(float(low_b), 2),
            "upper_bound": round(float(high_b), 2),
            "method": outlier_method,
        }

        if outlier_method == "clip":
          # Keep unclipped original in col_Raw, then clip main column
          df_clean[f"{col}_Unclipped"] = df_clean[col]
          df_clean[col] = df_clean[col].clip(lower=low_b, upper=high_b)
          audit["actions"].append(
              f"Clipped {outlier_count:,} outliers in '{col}' to range [{low_b:.2f}, {high_b:.2f}]."
          )
        elif outlier_method == "flag":
          df_clean[f"Is_{col}_Outlier"] = outlier_mask.astype(int)
          audit["actions"].append(f"Flagged {outlier_count:,} outliers in '{col}'.")

  # Sort chronologically by Order Date
  if "Order Date" in df_clean.columns:
    df_clean.sort_values(by="Order Date", ascending=True, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

  audit["final_rows"] = len(df_clean)
  audit["final_cols"] = len(df_clean.columns)

  # Step 7: Persist Processed Data
  if save_processed:
    dest = output_path or DEFAULT_PROCESSED_DATASET
    dest.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(dest, index=False)
    audit["saved_path"] = str(dest)
    audit["actions"].append(f"Saved cleaned dataset to {dest}.")

  return df_clean, audit
