"""RetailIQ Module A: Executive Dashboard.

Delivers an interactive C-suite business overview featuring core financial KPIs,
temporal trends, category and regional splits, and top-performing products.
"""

from components.charts import (
    plot_monthly_comparison,
    plot_profit_over_time,
    plot_revenue_over_time,
    plot_sales_by_category,
    plot_sales_by_region,
    plot_top_products_bar,
)
from components.metrics import render_executive_kpis
import numpy as np
import pandas as pd
from src.product_analysis import get_top_products_by_revenue
import streamlit as st


def render_overview_page(df: pd.DataFrame):
  """Renders the complete Executive Overview dashboard."""
  if df is None or df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your filter criteria in the sidebar.")
    return

  # 1. Compute Executive KPIs
  total_revenue = float(df["Sales"].sum()) if "Sales" in df.columns else 0.0
  total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0.0
  total_orders = int(df["Order ID"].nunique()) if "Order ID" in df.columns else len(df)
  total_products_sold = int(df["Quantity"].sum()) if "Quantity" in df.columns else 0
  avg_order_value = (total_revenue / total_orders) if total_orders > 0 else 0.0
  profit_margin_pct = ((total_profit / total_revenue) * 100.0) if total_revenue > 0 else 0.0

  # Render Executive KPI Ribbon
  render_executive_kpis(
      total_revenue=total_revenue,
      total_profit=total_profit,
      total_orders=total_orders,
      total_products_sold=total_products_sold,
      avg_order_value=avg_order_value,
      profit_margin_pct=profit_margin_pct,
  )

  st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

  # 2. Main Trend Charts (Two-Column Layout)
  col_trend_left, col_trend_right = st.columns([1, 1])

  with col_trend_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fig_rev = plot_revenue_over_time(df)
    st.plotly_chart(fig_rev, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  with col_trend_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fig_prof = plot_profit_over_time(df)
    st.plotly_chart(fig_prof, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  # 3. Categorical & Regional Performance
  col_cat, col_reg = st.columns([1, 1])

  with col_cat:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if "Category" in df.columns:
      fig_cat = plot_sales_by_category(df)
      st.plotly_chart(fig_cat, use_container_width=True)
    else:
      st.info("Category column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

  with col_reg:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if "Region" in df.columns:
      fig_reg = plot_sales_by_region(df)
      st.plotly_chart(fig_reg, use_container_width=True)
    else:
      st.info("Region column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

  # 4. Monthly YoY Comparison & Top Products
  col_yoy, col_top = st.columns([1, 1])

  with col_yoy:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fig_yoy = plot_monthly_comparison(df)
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  with col_top:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    top_prods = get_top_products_by_revenue(df, n=8)
    if not top_prods.empty:
      fig_top = plot_top_products_bar(top_prods, metric="Total_Sales", n=8)
      st.plotly_chart(fig_top, use_container_width=True)
    else:
      st.info("Product details not available.")
    st.markdown("</div>", unsafe_allow_html=True)
