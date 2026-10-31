"""RetailIQ Feature Engineering Module.

Generates temporal, cyclical, rolling statistical, and lag features for
chronological time-series sales forecasting and retail analytics without data
leakage.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def aggregate_sales_by_frequency(
    df: pd.DataFrame,
    date_col: str = "Order Date",
    target_col: str = "Sales",
    freq: str = "D",
) -> pd.DataFrame:
  """Aggregates transactional retail data into regular time intervals.

  Fills missing dates in the range with zero sales to ensure a continuous time
  series.

  Args:
      df: Cleaned retail transactions DataFrame
      date_col: Name of datetime column
      target_col: Revenue target column
      freq: Pandas frequency string ('D' for daily, 'W-MON' for weekly, 'MS' for
        monthly)

  Returns:
      Time-series DataFrame indexed by date with aggregated metrics.
  """
  data = df.copy()
  if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
  data = data.dropna(subset=[date_col])

  data.set_index(date_col, inplace=True)
  data.sort_index(inplace=True)

  # Aggregations
  agg_dict = {
      target_col: "sum",
      "Profit": "sum",
      "Quantity": "sum",
      "Discount": "mean",
      "Order ID": "nunique",
  }
  # Filter only existing columns
  actual_aggs = {k: v for k, v in agg_dict.items() if k in data.columns}

  ts = data.resample(freq).agg(actual_aggs)
  ts.rename(
      columns={
          target_col: "Sales",
          "Order ID": "Order_Count",
          "Quantity": "Total_Units",
          "Profit": "Total_Profit",
          "Discount": "Avg_Discount",
      },
      inplace=True,
  )

  # Fill missing calendar intervals with 0 sales and units
  ts["Sales"] = ts["Sales"].fillna(0.0)
  if "Total_Units" in ts.columns:
    ts["Total_Units"] = ts["Total_Units"].fillna(0)
  if "Order_Count" in ts.columns:
    ts["Order_Count"] = ts["Order_Count"].fillna(0)
  if "Total_Profit" in ts.columns:
    ts["Total_Profit"] = ts["Total_Profit"].fillna(0.0)
  if "Avg_Discount" in ts.columns:
    ts["Avg_Discount"] = ts["Avg_Discount"].fillna(0.0)

  ts.reset_index(inplace=True)
  ts.rename(columns={date_col: "Date"}, inplace=True)
  return ts


def create_forecasting_features(
    ts_df: pd.DataFrame,
    target_col: str = "Sales",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
  """Builds rich time-series features: calendar, cyclical, lag, and rolling statistics.

  Ensures all rolling features use shifted values so the current target is never
  leaked into predictors.
  """
  df_feat = ts_df.copy()
  if "Date" not in df_feat.columns:
    raise ValueError("Input DataFrame must have a 'Date' column.")

  if not pd.api.types.is_datetime64_any_dtype(df_feat["Date"]):
    df_feat["Date"] = pd.to_datetime(df_feat["Date"])

  df_feat.sort_values(by="Date", ascending=True, inplace=True)
  df_feat.reset_index(drop=True, inplace=True)

  # 1. Calendar Features
  df_feat["day_of_week"] = df_feat["Date"].dt.dayofweek
  df_feat["day_of_month"] = df_feat["Date"].dt.day
  df_feat["day_of_year"] = df_feat["Date"].dt.dayofyear
  df_feat["week_of_year"] = df_feat["Date"].dt.isocalendar().week.astype(int)
  df_feat["month"] = df_feat["Date"].dt.month
  df_feat["quarter"] = df_feat["Date"].dt.quarter
  df_feat["year"] = df_feat["Date"].dt.year
  df_feat["is_weekend"] = df_feat["day_of_week"].isin([5, 6]).astype(int)
  df_feat["is_month_start"] = df_feat["Date"].dt.is_month_start.astype(int)
  df_feat["is_month_end"] = df_feat["Date"].dt.is_month_end.astype(int)

  # 2. Cyclical Features (Sine / Cosine encoding)
  df_feat["sin_dow"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
  df_feat["cos_dow"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)
  df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12)
  df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12)
  df_feat["sin_doy"] = np.sin(2 * np.pi * df_feat["day_of_year"] / 365.25)
  df_feat["cos_doy"] = np.cos(2 * np.pi * df_feat["day_of_year"] / 365.25)

  # 3. Deterministic Trend Step
  df_feat["trend_step"] = np.arange(len(df_feat))

  # 4. Lag Features (Shifted by 1 or more periods to prevent leakage)
  if lags is None:
    lags = [1, 2, 7, 14, 21, 28, 30]

  for lag in lags:
    if len(df_feat) > lag:
      df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

  # 5. Rolling Statistical Features (Calculated strictly on lag_1 to avoid leakage)
  if windows is None:
    windows = [7, 14, 30]

  # Shift target by 1 first so rolling window only sees past observations
  shifted_target = df_feat[target_col].shift(1)
  for w in windows:
    if len(df_feat) > w:
      df_feat[f"rolling_mean_{w}"] = (
          shifted_target.rolling(window=w, min_periods=1).mean()
      )
      df_feat[f"rolling_std_{w}"] = (
          shifted_target.rolling(window=w, min_periods=1).std().fillna(0.0)
      )
      df_feat[f"rolling_min_{w}"] = (
          shifted_target.rolling(window=w, min_periods=1).min()
      )
      df_feat[f"rolling_max_{w}"] = (
          shifted_target.rolling(window=w, min_periods=1).max()
      )

  # Drop initial rows where maximum lag is NaN
  max_lag = max(lags) if lags else 1
  df_valid = df_feat.iloc[max_lag:].copy().reset_index(drop=True)

  return df_valid


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    target_col: str = "Sales",
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
  """Performs a strict chronological train-test split for time-series data.

  The training set contains all historical data up to split point T, and the
  test set contains subsequent future data from T to end.

  Returns:
      X_train, X_test, y_train, y_test, df_train, df_test
  """
  data = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)
  n = len(data)
  split_idx = int(n * (1.0 - test_size))

  df_train = data.iloc[:split_idx].copy()
  df_test = data.iloc[split_idx:].copy()

  exclude_cols = [date_col, target_col, "Total_Profit", "Profit"]
  feature_cols = [
      c for c in data.columns
      if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])
  ]

  X_train = df_train[feature_cols].copy()
  y_train = df_train[target_col].copy()
  X_test = df_test[feature_cols].copy()
  y_test = df_test[target_col].copy()

  return X_train, X_test, y_train, y_test, df_train, df_test
