"""RetailIQ Module D: Regional and Customer Analysis Page.

Analyzes geographic market dynamics, state profitability, customer segment
economics,
purchasing patterns, and territory expansion opportunities.
"""

from components.charts import plot_sales_by_region
import pandas as pd
import plotly.express as px
from src.regional_analysis import (
    generate_regional_insights,
    get_customer_segment_analysis,
    get_regional_performance,
    get_state_performance,
)
import streamlit as st


def render_regional_page(df: pd.DataFrame):
  """Renders the comprehensive Regional & Customer Analysis module."""
  if df is None or df.empty:
    st.warning("⚠️ No data available for Regional & Customer Analysis. Please adjust your filter criteria in the sidebar.")
    return

  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">🗺️ Regional & Customer Market Dynamics</div>
        <div class="section-subtitle">
            Evaluate geographic revenue concentrations, state operating margins, customer segment value, and purchasing behaviors.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Automated Regional Insights
  insights = generate_regional_insights(df)
  with st.expander("💡 Regional & Customer Insights", expanded=True):
    for ins in insights:
      st.markdown(f"- {ins}")

  tab_reg, tab_state, tab_cust = st.tabs(
      [
          "🌍 Regional Performance",
          "📍 State-Level Rankings",
          "👥 Customer Segment Economics",
      ]
  )

  with tab_reg:
    reg_df = get_regional_performance(df)

    col_donut, col_bar = st.columns([1, 1.2])
    with col_donut:
      fig_donut = plot_sales_by_region(df)
      st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
      fig_bar = px.bar(
          reg_df,
          x="Region",
          y=["Total_Revenue", "Total_Profit"],
          barmode="group",
          title="<b>Regional Revenue vs Net Profit</b>",
          color_discrete_sequence=["#1E3A8A", "#0D9488"],
      )
      fig_bar.update_layout(template="plotly_white")
      st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("##### Detailed Regional Performance Matrix")
    st.dataframe(reg_df, use_container_width=True)

  with tab_state:
    st.markdown("##### Top State Performers by Commercial Revenue")
    top_states = get_state_performance(df, top_n=15)
    if not top_states.empty:
      fig_state = px.bar(
          top_states,
          x="Total_Revenue",
          y="State",
          orientation="h",
          color="Profit_Margin_Pct",
          color_continuous_scale="Tealgrn",
          title="<b>Top 15 States by Revenue & Profit Margin</b>",
      )
      fig_state.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
      st.plotly_chart(fig_state, use_container_width=True)

      st.dataframe(top_states, use_container_width=True)
    else:
      st.info("State data not available.")

  with tab_cust:
    st.markdown("##### Customer Segment Purchasing Behavior")
    cust_df = get_customer_segment_analysis(df)
    if not cust_df.empty:
      col_aov, col_ord = st.columns([1, 1])

      with col_aov:
        fig_aov = px.bar(
            cust_df,
            x="Segment",
            y="AOV",
            color="Segment",
            title="<b>Average Order Value (AOV) by Segment</b>",
            color_discrete_sequence=["#1E3A8A", "#3B82F6", "#0D9488"],
        )
        fig_aov.update_layout(template="plotly_white")
        st.plotly_chart(fig_aov, use_container_width=True)

      with col_ord:
        fig_rev_cust = px.bar(
            cust_df,
            x="Segment",
            y="Revenue_Per_Customer",
            color="Segment",
            title="<b>Annual Revenue per Customer by Segment</b>",
            color_discrete_sequence=["#1E3A8A", "#3B82F6", "#0D9488"],
        )
        fig_rev_cust.update_layout(template="plotly_white")
        st.plotly_chart(fig_rev_cust, use_container_width=True)

      st.markdown("##### Detailed Segment Economics Table")
      st.dataframe(cust_df, use_container_width=True)
    else:
      st.info("Customer segment data not available.")
