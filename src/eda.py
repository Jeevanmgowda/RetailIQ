"""RetailIQ Exploratory Data Analysis (EDA) Module.

Computes comprehensive statistical summaries, distribution metrics, correlation
matrices,
category/regional/segment breakdowns, and automated analytical observations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
  """Computes structural metadata and column breakdown."""
  mem_bytes = df.memory_usage(deep=True).sum()
  mem_mb = round(mem_bytes / (1024 * 1024), 2)

  col_info = []
  for col in df.columns:
    col_info.append({
        "Column": col,
        "Type": str(df[col].dtype),
        "Non-Null Count": int(df[col].notnull().sum()),
        "Null Count": int(df[col].isnull().sum()),
        "Null %": round((df[col].isnull().sum() / len(df)) * 100, 2),
        "Unique Values": int(df[col].nunique()),
    })

  return {
      "total_rows": len(df),
      "total_columns": len(df.columns),
      "memory_mb": mem_mb,
      "columns_table": pd.DataFrame(col_info),
      "duplicate_count": int(df.duplicated().sum()),
  }


def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
  """Computes advanced descriptive statistics for numerical columns."""
  num_cols = [
      c for c in ["Sales", "Profit", "Quantity", "Discount", "Profit_Margin_Pct", "Unit_Price", "Shipping_Duration_Days"]
      if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
  ]
  if not num_cols:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

  summary_rows = []
  for col in num_cols:
    s = df[col].dropna()
    if len(s) == 0:
      continue
    skew_val = float(stats.skew(s)) if len(s) > 2 else 0.0
    kurt_val = float(stats.kurtosis(s)) if len(s) > 3 else 0.0

    summary_rows.append({
        "Metric": col,
        "Count": int(len(s)),
        "Mean": round(float(s.mean()), 2),
        "Std": round(float(s.std()), 2),
        "Min": round(float(s.min()), 2),
        "25% (Q1)": round(float(s.quantile(0.25)), 2),
        "Median (Q2)": round(float(s.median()), 2),
        "75% (Q3)": round(float(s.quantile(0.75)), 2),
        "Max": round(float(s.max()), 2),
        "IQR": round(float(s.quantile(0.75) - s.quantile(0.25)), 2),
        "Skewness": round(skew_val, 2),
        "Kurtosis": round(kurt_val, 2),
    })

  return pd.DataFrame(summary_rows)


def get_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
  """Computes correlation matrix for numerical business columns."""
  candidate_cols = [
      "Sales", "Profit", "Quantity", "Discount", "Profit_Margin_Pct",
      "Unit_Price", "Shipping_Duration_Days"
  ]
  cols = [c for c in candidate_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
  if len(cols) < 2:
    return pd.DataFrame()
  return df[cols].corr(method=method).round(3)


def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
  """Summarizes performance metrics by Category and Sub-Category."""
  if "Category" not in df.columns:
    return pd.DataFrame()

  group_cols = ["Category"]
  if "Sub-Category" in df.columns:
    group_cols.append("Sub-Category")

  summary = df.groupby(group_cols).agg(
      Total_Sales=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Total_Quantity=("Quantity", "sum"),
      Avg_Discount=("Discount", "mean"),
      Order_Count=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
  ).reset_index()

  summary["Profit_Margin_Pct"] = np.where(
      summary["Total_Sales"] > 0,
      (summary["Total_Profit"] / summary["Total_Sales"]) * 100.0,
      0.0
  ).round(2)

  summary["Total_Sales"] = summary["Total_Sales"].round(2)
  summary["Total_Profit"] = summary["Total_Profit"].round(2)
  summary["Avg_Discount"] = (summary["Avg_Discount"] * 100).round(1)

  return summary.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)


def get_regional_breakdown(df: pd.DataFrame) -> pd.DataFrame:
  """Summarizes performance metrics by Region and State."""
  if "Region" not in df.columns:
    return pd.DataFrame()

  reg_summary = df.groupby("Region").agg(
      Total_Sales=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Total_Quantity=("Quantity", "sum"),
      Avg_Discount=("Discount", "mean"),
      Orders=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
      Customers=("Customer ID", "nunique") if "Customer ID" in df.columns else ("Sales", "count"),
  ).reset_index()

  reg_summary["Profit_Margin_Pct"] = np.where(
      reg_summary["Total_Sales"] > 0,
      (reg_summary["Total_Profit"] / reg_summary["Total_Sales"]) * 100.0,
      0.0
  ).round(2)

  total_sales_all = reg_summary["Total_Sales"].sum()
  reg_summary["Sales_Contribution_Pct"] = np.where(
      total_sales_all > 0,
      (reg_summary["Total_Sales"] / total_sales_all) * 100.0,
      0.0
  ).round(2)

  reg_summary["Total_Sales"] = reg_summary["Total_Sales"].round(2)
  reg_summary["Total_Profit"] = reg_summary["Total_Profit"].round(2)
  reg_summary["Avg_Discount"] = (reg_summary["Avg_Discount"] * 100).round(1)

  return reg_summary.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)


def get_segment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
  """Summarizes performance metrics across customer market segments."""
  if "Segment" not in df.columns:
    return pd.DataFrame()

  seg_summary = df.groupby("Segment").agg(
      Total_Sales=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Total_Quantity=("Quantity", "sum"),
      Orders=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
      Avg_Discount=("Discount", "mean"),
  ).reset_index()

  seg_summary["Profit_Margin_Pct"] = np.where(
      seg_summary["Total_Sales"] > 0,
      (seg_summary["Total_Profit"] / seg_summary["Total_Sales"]) * 100.0,
      0.0
  ).round(2)

  seg_summary["Average_Order_Value"] = np.where(
      seg_summary["Orders"] > 0,
      seg_summary["Total_Sales"] / seg_summary["Orders"],
      0.0
  ).round(2)

  seg_summary["Total_Sales"] = seg_summary["Total_Sales"].round(2)
  seg_summary["Total_Profit"] = seg_summary["Total_Profit"].round(2)
  seg_summary["Avg_Discount"] = (seg_summary["Avg_Discount"] * 100).round(1)

  return seg_summary.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)


def generate_eda_observations(df: pd.DataFrame) -> List[str]:
  """Generates dynamic, data-driven analytical observations based on calculated dataset metrics."""
  observations = []
  if df is None or df.empty:
    return ["Insufficient data to generate observations."]

  # Total volume observations
  tot_sales = df["Sales"].sum() if "Sales" in df.columns else 0
  tot_profit = df["Profit"].sum() if "Profit" in df.columns else 0
  margin = (tot_profit / tot_sales * 100) if tot_sales > 0 else 0

  observations.append(
      f"Overall enterprise sales reached **${tot_sales:,.2f}** generating **${tot_profit:,.2f}** "
      f"in net profit, reflecting an aggregate profit margin of **{margin:.1f}%**."
  )

  # Category observations
  if "Category" in df.columns and "Sales" in df.columns and "Profit" in df.columns:
    cat_grp = df.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    top_cat_rev = cat_grp.sort_values(by="Sales", ascending=False).iloc[0]
    top_cat_prof = cat_grp.sort_values(by="Profit", ascending=False).iloc[0]
    lowest_cat_margin = cat_grp.assign(Margin=cat_grp["Profit"]/cat_grp["Sales"]*100).sort_values(by="Margin").iloc[0]

    observations.append(
        f"**{top_cat_rev['Category']}** generates the largest revenue share (${top_cat_rev['Sales']:,.2f}), "
        f"while **{top_cat_prof['Category']}** delivers the highest cumulative profit (${top_cat_prof['Profit']:,.2f})."
    )
    if lowest_cat_margin["Margin"] < 10:
      observations.append(
          f"**{lowest_cat_margin['Category']}** has a compressed profit margin of **{lowest_cat_margin['Margin']:.1f}%**, "
          "indicating heavy discounting or high fulfillment costs."
      )

  # Regional observations
  if "Region" in df.columns and "Sales" in df.columns:
    reg_grp = df.groupby("Region").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    top_reg = reg_grp.sort_values(by="Sales", ascending=False).iloc[0]
    bot_reg = reg_grp.sort_values(by="Profit", ascending=True).iloc[0]
    observations.append(
        f"The **{top_reg['Region']}** region leads commercial volume with ${top_reg['Sales']:,.2f} in sales, "
        f"whereas the **{bot_reg['Region']}** region generates lowest operating profit (${bot_reg['Profit']:,.2f})."
    )

  # Discount vs Profit impact
  if "Discount" in df.columns and "Profit" in df.columns:
    high_disc = df[df["Discount"] >= 0.3]
    low_disc = df[df["Discount"] < 0.3]
    high_disc_loss_rate = (high_disc["Profit"] < 0).mean() * 100 if len(high_disc) > 0 else 0
    low_disc_loss_rate = (low_disc["Profit"] < 0).mean() * 100 if len(low_disc) > 0 else 0

    observations.append(
        f"Orders with deep discounts (>=30%) suffer an unprofitable transaction rate of **{high_disc_loss_rate:.1f}%**, "
        f"compared to only **{low_disc_loss_rate:.1f}%** for standard discounts."
    )

  # Outlier skewness
  if "Sales" in df.columns:
    skew = stats.skew(df["Sales"].dropna())
    if skew > 2.0:
      observations.append(
          f"Sales distribution exhibits high positive skewness (**{skew:.2f}**), "
          "indicating that a relatively small percentage of high-ticket transactions accounts for a major portion of turnover."
      )

  return observations
