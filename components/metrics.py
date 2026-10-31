"""RetailIQ KPI Metrics Component.

Renders modern, styled KPI summary cards matching SaaS and Power BI enterprise
standards.
"""

from typing import Optional
import streamlit as st


def format_currency(val: float) -> str:
  """Formats large numeric values into human-readable currency ($1.59M, $45.2K, $850.00)."""
  abs_val = abs(val)
  sign = "-" if val < 0 else ""
  if abs_val >= 1_000_000:
    return f"{sign}${abs_val / 1_000_000:.2f}M"
  elif abs_val >= 1_000:
    return f"{sign}${abs_val / 1_000:.1f}K"
  else:
    return f"{sign}${abs_val:.2f}"


def format_number(val: float) -> str:
  """Formats large integers into human-readable format (12.4K, 5,009)."""
  if abs(val) >= 1_000_000:
    return f"{val / 1_000_000:.2f}M"
  elif abs(val) >= 10_000:
    return f"{val / 1_000:.1f}K"
  else:
    return f"{int(val):,}"


def render_kpi_card(
    title: str,
    value: str,
    context: str = "",
    status: str = "neutral",  # positive, negative, neutral
):
  """Renders a single styled KPI card."""
  context_class = f"context-{status}"
  html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-context {context_class}">{context}</div>
    </div>
    """
  st.markdown(html, unsafe_allow_html=True)


def render_executive_kpis(
    total_revenue: float,
    total_profit: float,
    total_orders: int,
    total_products_sold: int,
    avg_order_value: float,
    profit_margin_pct: float,
):
  """Renders a 6-column executive KPI card row."""
  col1, col2, col3, col4, col5, col6 = st.columns(6)

  with col1:
    render_kpi_card(
        title="Total Revenue",
        value=format_currency(total_revenue),
        context="Gross commercial turnover",
        status="positive" if total_revenue > 0 else "neutral",
    )

  with col2:
    margin_status = "positive" if total_profit > 0 else "negative"
    render_kpi_card(
        title="Total Profit",
        value=format_currency(total_profit),
        context=f"Net earnings",
        status=margin_status,
    )

  with col3:
    margin_status = "positive" if profit_margin_pct >= 10 else ("negative" if profit_margin_pct < 0 else "neutral")
    render_kpi_card(
        title="Profit Margin",
        value=f"{profit_margin_pct:.1f}%",
        context="Net margin on revenue",
        status=margin_status,
    )

  with col4:
    render_kpi_card(
        title="Total Orders",
        value=format_number(total_orders),
        context="Unique sales transactions",
        status="neutral",
    )

  with col5:
    render_kpi_card(
        title="Units Sold",
        value=format_number(total_products_sold),
        context="Total volume dispatched",
        status="neutral",
    )

  with col6:
    render_kpi_card(
        title="Avg Order Value",
        value=format_currency(avg_order_value),
        context="Revenue per order transaction",
        status="neutral",
    )
