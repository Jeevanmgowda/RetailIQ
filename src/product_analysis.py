"""RetailIQ Product Performance Analysis Module.

Analyzes product portfolio profitability, revenue drivers, loss-making SKUs,
discount sensitivity, and high-revenue low-margin product traps.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def get_top_products_by_revenue(
    df: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
  """Returns top N products ranked by total sales revenue."""
  if "Product Name" not in df.columns or "Sales" not in df.columns:
    return pd.DataFrame()

  group_cols = ["Product Name"]
  if "Category" in df.columns:
    group_cols.append("Category")
  if "Sub-Category" in df.columns:
    group_cols.append("Sub-Category")

  prod_sales = df.groupby(group_cols).agg(
      Total_Sales=("Sales", "sum"),
      Total_Profit=("Profit", "sum") if "Profit" in df.columns else ("Sales", "count"),
      Quantity_Sold=("Quantity", "sum") if "Quantity" in df.columns else ("Sales", "count"),
      Order_Count=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
  ).reset_index()

  if "Profit" in df.columns:
    prod_sales["Profit_Margin_Pct"] = np.where(
        prod_sales["Total_Sales"] > 0,
        (prod_sales["Total_Profit"] / prod_sales["Total_Sales"]) * 100.0,
        0.0
    ).round(2)
    prod_sales["Total_Profit"] = prod_sales["Total_Profit"].round(2)

  prod_sales["Total_Sales"] = prod_sales["Total_Sales"].round(2)
  return prod_sales.sort_values(by="Total_Sales", ascending=False).head(n).reset_index(drop=True)


def get_top_products_by_profit(
    df: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
  """Returns top N products ranked by cumulative operating profit."""
  if "Product Name" not in df.columns or "Profit" not in df.columns:
    return pd.DataFrame()

  group_cols = ["Product Name"]
  if "Category" in df.columns:
    group_cols.append("Category")
  if "Sub-Category" in df.columns:
    group_cols.append("Sub-Category")

  prod_profit = df.groupby(group_cols).agg(
      Total_Profit=("Profit", "sum"),
      Total_Sales=("Sales", "sum") if "Sales" in df.columns else ("Profit", "count"),
      Quantity_Sold=("Quantity", "sum") if "Quantity" in df.columns else ("Profit", "count"),
  ).reset_index()

  if "Sales" in df.columns:
    prod_profit["Profit_Margin_Pct"] = np.where(
        prod_profit["Total_Sales"] > 0,
        (prod_profit["Total_Profit"] / prod_profit["Total_Sales"]) * 100.0,
        0.0
    ).round(2)
    prod_profit["Total_Sales"] = prod_profit["Total_Sales"].round(2)

  prod_profit["Total_Profit"] = prod_profit["Total_Profit"].round(2)
  return prod_profit.sort_values(by="Total_Profit", ascending=False).head(n).reset_index(drop=True)


def get_lowest_performing_products(
    df: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
  """Returns lowest N performing products (greatest cumulative loss)."""
  if "Product Name" not in df.columns or "Profit" not in df.columns:
    return pd.DataFrame()

  group_cols = ["Product Name"]
  if "Category" in df.columns:
    group_cols.append("Category")
  if "Sub-Category" in df.columns:
    group_cols.append("Sub-Category")

  prod_loss = df.groupby(group_cols).agg(
      Total_Profit=("Profit", "sum"),
      Total_Sales=("Sales", "sum") if "Sales" in df.columns else ("Profit", "count"),
      Quantity_Sold=("Quantity", "sum") if "Quantity" in df.columns else ("Profit", "count"),
      Avg_Discount=("Discount", "mean") if "Discount" in df.columns else ("Profit", "count"),
  ).reset_index()

  if "Sales" in df.columns:
    prod_loss["Profit_Margin_Pct"] = np.where(
        prod_loss["Total_Sales"] > 0,
        (prod_loss["Total_Profit"] / prod_loss["Total_Sales"]) * 100.0,
        0.0
    ).round(2)
    prod_loss["Total_Sales"] = prod_loss["Total_Sales"].round(2)

  prod_loss["Total_Profit"] = prod_loss["Total_Profit"].round(2)
  if "Discount" in df.columns:
    prod_loss["Avg_Discount"] = (prod_loss["Avg_Discount"] * 100).round(1)

  return prod_loss.sort_values(by="Total_Profit", ascending=True).head(n).reset_index(drop=True)


def identify_high_revenue_low_profit_traps(
    df: pd.DataFrame,
    sales_percentile: float = 0.70,
    max_margin_pct: float = 5.0,
) -> pd.DataFrame:
  """Identifies 'Revenue Trap' products: high sales volume but negligible or negative profit margins.

  These products inflate top-line numbers while eroding net margins.
  """
  if "Product Name" not in df.columns or "Sales" not in df.columns or "Profit" not in df.columns:
    return pd.DataFrame()

  group_cols = ["Product Name"]
  if "Category" in df.columns:
    group_cols.append("Category")
  if "Sub-Category" in df.columns:
    group_cols.append("Sub-Category")

  prod_agg = df.groupby(group_cols).agg(
      Total_Sales=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Quantity_Sold=("Quantity", "sum"),
      Avg_Discount=("Discount", "mean") if "Discount" in df.columns else ("Sales", "count"),
  ).reset_index()

  prod_agg["Profit_Margin_Pct"] = np.where(
      prod_agg["Total_Sales"] > 0,
      (prod_agg["Total_Profit"] / prod_agg["Total_Sales"]) * 100.0,
      0.0
  ).round(2)

  threshold_sales = prod_agg["Total_Sales"].quantile(sales_percentile)

  traps = prod_agg[
      (prod_agg["Total_Sales"] >= threshold_sales) &
      (prod_agg["Profit_Margin_Pct"] <= max_margin_pct)
  ].copy()

  traps["Total_Sales"] = traps["Total_Sales"].round(2)
  traps["Total_Profit"] = traps["Total_Profit"].round(2)
  if "Discount" in df.columns:
    traps["Avg_Discount"] = (traps["Avg_Discount"] * 100).round(1)

  return traps.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)


def get_discount_vs_profit_binned(
    df: pd.DataFrame,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
  """Bins discounts to analyze how varying discount tiers affect sales and profit margins."""
  if "Discount" not in df.columns or "Profit" not in df.columns or "Sales" not in df.columns:
    return pd.DataFrame()

  if bins is None:
    bins = [-0.01, 0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
  if labels is None:
    labels = ["0% (No Discount)", "1-10%", "11-20%", "21-30%", "31-50%", "50%+"]

  df_binned = df.copy()
  df_binned["Discount_Tier"] = pd.cut(df_binned["Discount"], bins=bins, labels=labels)

  tier_summary = df_binned.groupby("Discount_Tier", observed=False).agg(
      Transaction_Count=("Sales", "count"),
      Total_Sales=("Sales", "sum"),
      Total_Profit=("Profit", "sum"),
      Avg_Profit_Per_Order=("Profit", "mean"),
  ).reset_index()

  tier_summary["Profit_Margin_Pct"] = np.where(
      tier_summary["Total_Sales"] > 0,
      (tier_summary["Total_Profit"] / tier_summary["Total_Sales"]) * 100.0,
      0.0
  ).round(2)

  tier_summary["Total_Sales"] = tier_summary["Total_Sales"].round(2)
  tier_summary["Total_Profit"] = tier_summary["Total_Profit"].round(2)
  tier_summary["Avg_Profit_Per_Order"] = tier_summary["Avg_Profit_Per_Order"].round(2)

  return tier_summary


def generate_product_recommendations(df: pd.DataFrame) -> List[Dict[str, str]]:
  """Generates targeted business recommendations based on product performance metrics."""
  recs = []

  # 1. Top loss makers
  lowest = get_lowest_performing_products(df, n=5)
  if not lowest.empty and lowest.iloc[0]["Total_Profit"] < 0:
    worst_prod = lowest.iloc[0]
    total_worst_loss = abs(float(lowest["Total_Profit"].sum()))
    recs.append({
        "type": "Cost Reduction / Rationalization",
        "title": f"Review or Phase Out Top Loss-Maker: {worst_prod['Product Name'][:45]}",
        "description": (
            f"The top 5 loss-making products collectively burned **${total_worst_loss:,.2f}** in capital. "
            f"**{worst_prod['Product Name'][:45]}** single-handedly caused a loss of **${abs(worst_prod['Total_Profit']):,.2f}** "
            f"with an average discount of **{worst_prod.get('Avg_Discount', 0)}%**. "
            "Recommendation: Renegotiate supplier unit costs, enforce a discount ceiling of 15%, or de-list the SKU."
        ),
        "impact": "High Profit Recovery",
        "status": "Critical",
    })

  # 2. Revenue traps
  traps = identify_high_revenue_low_profit_traps(df)
  if not traps.empty:
    trap_count = len(traps)
    trap_sales = float(traps["Total_Sales"].sum())
    recs.append({
        "type": "Pricing Optimization",
        "title": f"Restructure Pricing on {trap_count} High-Volume Margin Trap SKUs",
        "description": (
            f"Identified {trap_count} products delivering high turnover (**${trap_sales:,.2f}** total revenue) "
            "but generating margins below 5%. These items act as unmonetized volume drivers. "
            "Recommendation: Introduce bundle deals with high-margin accessories and trim discretionary discounts by 3-5%."
        ),
        "impact": "Margin Expansion",
        "status": "Actionable",
    })

  # 3. Top profit champion
  top_prof = get_top_products_by_profit(df, n=3)
  if not top_prof.empty:
    best_p = top_prof.iloc[0]
    recs.append({
        "type": "Growth / Inventory Priority",
        "title": f"Protect Inventory & Accelerate Marketing for {best_p['Product Name'][:45]}",
        "description": (
            f"**{best_p['Product Name'][:45]}** is your primary profit anchor, generating **${best_p['Total_Profit']:,.2f}** "
            f"in net profit ({best_p.get('Profit_Margin_Pct', 0)}% margin). "
            "Recommendation: Secure safety stock agreements with distributors to avoid stockouts during demand spikes."
        ),
        "impact": "Revenue & Profit Continuity",
        "status": "Strategic",
    })

  return recs
