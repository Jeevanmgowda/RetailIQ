"""RetailIQ Business Intelligence & Recommendation Engine.

Generates targeted, data-backed operational and strategic recommendations
derived directly from empirical transaction metrics, pricing elasticity,
and category/regional margins.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class RecommendationItem:
  """Structured data-backed recommendation."""

  domain: str           # Pricing, Inventory, Product Mix, Regional, Seasonality
  title: str
  priority: str         # Critical, High, Medium, Opportunity
  metric_evidence: str
  prescriptive_action: str
  expected_impact: str

  def to_dict(self) -> Dict[str, str]:
    return asdict(self)


def generate_business_recommendations(df: pd.DataFrame) -> List[RecommendationItem]:
  """Computes quantitative metrics across categories, regions, products, and temporal patterns

  to formulate precise, high-ROI business recommendations.
  """
  recommendations: List[RecommendationItem] = []
  if df is None or df.empty:
    return recommendations

  # -------------------------------------------------------------
  # 1. PRICING & EXCESSIVE DISCOUNTING AUDIT
  # -------------------------------------------------------------
  if "Discount" in df.columns and "Profit" in df.columns and "Sales" in df.columns:
    deep_disc_mask = df["Discount"] >= 0.30
    deep_disc_df = df[deep_disc_mask]
    deep_disc_loss = deep_disc_df[deep_disc_df["Profit"] < 0]["Profit"].sum()
    deep_disc_orders = len(deep_disc_df)
    total_profit = df["Profit"].sum()

    if deep_disc_orders > 0 and deep_disc_loss < -100:
      loss_val = abs(float(deep_disc_loss))
      recommendations.append(
          RecommendationItem(
              domain="Pricing & Discounts",
              title="Establish Strict Margin-Protective Discount Floors",
              priority="Critical",
              metric_evidence=(
                  f"Transactions with >=30% discounts accounted for {deep_disc_orders:,} orders "
                  f"and produced an aggregate net loss of **${loss_val:,.2f}**."
              ),
              prescriptive_action=(
                  "Implement an automated POS approval threshold requiring regional manager sign-off "
                  "for any discretionary customer discount exceeding 20% on low-margin categories."
              ),
              expected_impact=(
                  f"Potential annual recovery of up to **${loss_val * 0.75:,.2f}** in previously eroded margin."
              ),
          )
      )

  # -------------------------------------------------------------
  # 2. LOW-PROFIT / LOSS-MAKING PRODUCTS
  # -------------------------------------------------------------
  if "Product Name" in df.columns and "Profit" in df.columns:
    prod_profit = df.groupby("Product Name").agg(
        Total_Profit=("Profit", "sum"),
        Total_Sales=("Sales", "sum") if "Sales" in df.columns else ("Profit", "count"),
        Avg_Discount=("Discount", "mean") if "Discount" in df.columns else ("Profit", "count"),
    ).reset_index()

    loss_makers = prod_profit[prod_profit["Total_Profit"] < -500].sort_values(by="Total_Profit", ascending=True)
    if not loss_makers.empty:
      top_loss_sku = loss_makers.iloc[0]
      cumulative_loss = abs(float(loss_makers["Total_Profit"].sum()))
      count_skus = len(loss_makers)

      recommendations.append(
          RecommendationItem(
              domain="Product Portfolio",
              title=f"Rationalize {count_skus} Chronic Capital-Draining SKUs",
              priority="Critical",
              metric_evidence=(
                  f"{count_skus} products generated chronic deficits totaling **${cumulative_loss:,.2f}** in losses. "
                  f"The single largest deficit was '{top_loss_sku['Product Name'][:40]}' (-${abs(top_loss_sku['Total_Profit']):,.2f})."
              ),
              prescriptive_action=(
                  "Review supplier cost contracts for these specific SKUs. If supplier unit costs cannot be reduced "
                  "by at least 18%, discontinue SKU stocking and reallocate working capital to positive-margin alternatives."
              ),
              expected_impact=(
                  f"Immediate preservation of **${cumulative_loss:,.2f}** in operating cash flow."
              ),
          )
      )

  # -------------------------------------------------------------
  # 3. HIGH-MARGIN CATEGORIES & EXPANSION
  # -------------------------------------------------------------
  if "Category" in df.columns and "Profit" in df.columns and "Sales" in df.columns:
    cat_summary = df.groupby("Category").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
    ).reset_index()
    cat_summary["Margin_Pct"] = np.where(
        cat_summary["Total_Sales"] > 0,
        (cat_summary["Total_Profit"] / cat_summary["Total_Sales"]) * 100,
        0.0
    )

    best_cat = cat_summary.sort_values(by="Margin_Pct", ascending=False).iloc[0]
    if best_cat["Margin_Pct"] > 12.0:
      recommendations.append(
          RecommendationItem(
              domain="Category Expansion",
              title=f"Scale Marketing & Prime Shelf Placement for {best_cat['Category']}",
              priority="High",
              metric_evidence=(
                  f"**{best_cat['Category']}** is your highest-margin category with a profit margin of "
                  f"**{best_cat['Margin_Pct']:.1f}%**, generating **${best_cat['Total_Profit']:,.2f}** profit on "
                  f"**${best_cat['Total_Sales']:,.2f}** revenue."
              ),
              prescriptive_action=(
                  f"Increase marketing spend on {best_cat['Category']} campaigns by 15-20% and negotiate co-op "
                  "promotional placement with top vendors in this department."
              ),
              expected_impact=(
                  "Accelerates high-margin mix shift, expanding enterprise operating margins by 120-180 bps."
              ),
          )
      )

  # -------------------------------------------------------------
  # 4. REGIONAL PERFORMANCE / DECLINING REVENUE & MARGINS
  # -------------------------------------------------------------
  if "Region" in df.columns and "Profit" in df.columns and "Sales" in df.columns:
    reg_summary = df.groupby("Region").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Avg_Discount=("Discount", "mean") if "Discount" in df.columns else ("Sales", "count"),
    ).reset_index()
    reg_summary["Margin_Pct"] = (reg_summary["Total_Profit"] / reg_summary["Total_Sales"]) * 100

    lowest_reg = reg_summary.sort_values(by="Margin_Pct").iloc[0]
    highest_reg = reg_summary.sort_values(by="Margin_Pct", ascending=False).iloc[0]

    if lowest_reg["Margin_Pct"] < highest_reg["Margin_Pct"] - 4.0:
      recommendations.append(
          RecommendationItem(
              domain="Regional Strategy",
              title=f"Optimize Operational Efficiency in {lowest_reg['Region']} Region",
              priority="High",
              metric_evidence=(
                  f"The **{lowest_reg['Region']}** region trails with an operating margin of only "
                  f"**{lowest_reg['Margin_Pct']:.1f}%** (vs **{highest_reg['Margin_Pct']:.1f}%** in {highest_reg['Region']}), "
                  f"while discounting averages **{lowest_reg['Avg_Discount']*100:.1f}%**."
              ),
              prescriptive_action=(
                  f"Conduct a freight and logistics audit for {lowest_reg['Region']} distribution routes, "
                  "and align regional commission incentives to gross profit rather than gross revenue."
              ),
              expected_impact=(
                  f"Lifting {lowest_reg['Region']} margin to company benchmark would unlock "
                  f"**${(lowest_reg['Total_Sales'] * ((highest_reg['Margin_Pct'] - lowest_reg['Margin_Pct'])/100)):,.2f}** in incremental profit."
              ),
          )
      )

  # -------------------------------------------------------------
  # 5. SEASONAL SALES & INVENTORY PLANNING
  # -------------------------------------------------------------
  if "Order Date" in df.columns and "Sales" in df.columns:
    date_series = pd.to_datetime(df["Order Date"], errors="coerce")
    valid_dates = df.dropna(subset=["Order Date"]).copy()
    valid_dates["Month"] = date_series.dt.month
    valid_dates["Quarter"] = date_series.dt.quarter

    monthly_sales = valid_dates.groupby("Month")["Sales"].sum()
    if len(monthly_sales) >= 12:
      peak_month_idx = int(monthly_sales.idxmax())
      trough_month_idx = int(monthly_sales.idxmin())
      month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
      peak_month = month_names[peak_month_idx - 1]
      peak_vol = float(monthly_sales[peak_month_idx])
      avg_vol = float(monthly_sales.mean())
      surge_pct = round(((peak_vol - avg_vol) / avg_vol) * 100, 1)

      recommendations.append(
          RecommendationItem(
              domain="Inventory & Seasonality",
              title=f"Pre-Build Safety Stocks 45 Days Ahead of {peak_month} Peak Demand",
              priority="Medium",
              metric_evidence=(
                  f"Historical purchasing exhibits high seasonality: **{peak_month}** surges **+{surge_pct}%** "
                  f"above monthly average with **${peak_vol:,.2f}** in aggregate turnover."
              ),
              prescriptive_action=(
                  f"Initiate purchase orders for top velocity SKUs 6-8 weeks prior to {peak_month}. "
                  "Secure secondary 3PL logistics carriers early to safeguard on-time delivery rates."
              ),
              expected_impact=(
                  "Mitigates stockouts during peak shopping periods, capturing an estimated 8-12% in otherwise lost revenue."
              ),
          )
      )

  return recommendations
