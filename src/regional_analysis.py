"""RetailIQ Regional and Customer Analysis Module.

Analyzes geographic sales distributions, regional profit contributions, state
rankings, customer segment behaviors, and purchasing patterns.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def get_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates aggregated performance metrics by Region."""
  if "Region" not in df.columns:
    return pd.DataFrame()

  reg = df.groupby("Region").agg(
      Total_Revenue=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Total_Quantity=("Quantity", "sum"),
      Order_Count=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
      Customer_Count=("Customer ID", "nunique") if "Customer ID" in df.columns else ("Sales", "count"),
      Avg_Discount=("Discount", "mean"),
  ).reset_index()

  total_rev = reg["Total_Revenue"].sum()
  reg["Revenue_Share_Pct"] = np.where(
      total_rev > 0,
      (reg["Total_Revenue"] / total_rev) * 100.0,
      0.0
  ).round(2)

  reg["Profit_Margin_Pct"] = np.where(
      reg["Total_Revenue"] > 0,
      (reg["Total_Profit"] / reg["Total_Revenue"]) * 100.0,
      0.0
  ).round(2)

  reg["Average_Order_Value"] = np.where(
      reg["Order_Count"] > 0,
      reg["Total_Revenue"] / reg["Order_Count"],
      0.0
  ).round(2)

  reg["Total_Revenue"] = reg["Total_Revenue"].round(2)
  reg["Total_Profit"] = reg["Total_Profit"].round(2)
  reg["Avg_Discount"] = (reg["Avg_Discount"] * 100).round(1)

  return reg.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)


def get_state_performance(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
  """Calculates state-level performance ranking."""
  if "State" not in df.columns:
    return pd.DataFrame()

  state_df = df.groupby(["State", "Region"] if "Region" in df.columns else "State").agg(
      Total_Revenue=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Order_Count=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
      Avg_Discount=("Discount", "mean"),
  ).reset_index()

  state_df["Profit_Margin_Pct"] = np.where(
      state_df["Total_Revenue"] > 0,
      (state_df["Total_Profit"] / state_df["Total_Revenue"]) * 100.0,
      0.0
  ).round(2)

  state_df["Total_Revenue"] = state_df["Total_Revenue"].round(2)
  state_df["Total_Profit"] = state_df["Total_Profit"].round(2)
  state_df["Avg_Discount"] = (state_df["Avg_Discount"] * 100).round(1)

  return state_df.sort_values(by="Total_Revenue", ascending=False).head(top_n).reset_index(drop=True)


def get_customer_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
  """Analyzes performance metrics, profitability, and purchasing patterns across customer segments."""
  if "Segment" not in df.columns:
    return pd.DataFrame()

  seg = df.groupby("Segment").agg(
      Total_Revenue=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Total_Quantity=("Quantity", "sum"),
      Order_Count=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
      Unique_Customers=("Customer ID", "nunique") if "Customer ID" in df.columns else ("Sales", "count"),
      Avg_Discount=("Discount", "mean"),
  ).reset_index()

  total_sales = seg["Total_Revenue"].sum()
  seg["Revenue_Share_Pct"] = np.where(
      total_sales > 0,
      (seg["Total_Revenue"] / total_sales) * 100.0,
      0.0
  ).round(2)

  seg["Profit_Margin_Pct"] = np.where(
      seg["Total_Revenue"] > 0,
      (seg["Total_Profit"] / seg["Total_Revenue"]) * 100.0,
      0.0
  ).round(2)

  seg["AOV"] = np.where(
      seg["Order_Count"] > 0,
      seg["Total_Revenue"] / seg["Order_Count"],
      0.0
  ).round(2)

  seg["Orders_Per_Customer"] = np.where(
      seg["Unique_Customers"] > 0,
      seg["Order_Count"] / seg["Unique_Customers"],
      0.0
  ).round(2)

  seg["Revenue_Per_Customer"] = np.where(
      seg["Unique_Customers"] > 0,
      seg["Total_Revenue"] / seg["Unique_Customers"],
      0.0
  ).round(2)

  seg["Total_Revenue"] = seg["Total_Revenue"].round(2)
  seg["Total_Profit"] = seg["Total_Profit"].round(2)
  seg["Avg_Discount"] = (seg["Avg_Discount"] * 100).round(1)

  return seg.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)


def generate_regional_insights(df: pd.DataFrame) -> List[str]:
  """Generates concise, data-driven analytical observations for regional & customer dynamics."""
  insights = []
  if df is None or df.empty:
    return ["Insufficient data for regional analytics."]

  reg_df = get_regional_performance(df)
  if not reg_df.empty:
    top_reg = reg_df.iloc[0]
    insights.append(
        f"**{top_reg['Region']}** is the strongest commercial territory, contributing "
        f"**${top_reg['Total_Revenue']:,.2f}** ({top_reg['Revenue_Share_Pct']}% of total revenue) "
        f"with an operating profit margin of **{top_reg['Profit_Margin_Pct']}%**."
    )

    lowest_margin_reg = reg_df.sort_values(by="Profit_Margin_Pct").iloc[0]
    if lowest_margin_reg["Profit_Margin_Pct"] < top_reg["Profit_Margin_Pct"]:
      insights.append(
          f"**{lowest_margin_reg['Region']}** generates the lowest operating margin at **{lowest_margin_reg['Profit_Margin_Pct']}%**, "
          f"driven by higher promotional discounts averaging **{lowest_margin_reg['Avg_Discount']}%**."
      )

  seg_df = get_customer_segment_analysis(df)
  if not seg_df.empty:
    top_seg = seg_df.iloc[0]
    highest_aov_seg = seg_df.sort_values(by="AOV", ascending=False).iloc[0]
    insights.append(
        f"The **{top_seg['Segment']}** segment generates highest gross volume (**${top_seg['Total_Revenue']:,.2f}**), "
        f"while **{highest_aov_seg['Segment']}** holds the highest Average Order Value at **${highest_aov_seg['AOV']:,.2f}** per transaction."
    )

  return insights
