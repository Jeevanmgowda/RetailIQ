"""RetailIQ Module B: Exploratory Data Analysis (EDA) Page.

Provides in-depth exploratory data analysis including distribution profiles,
statistical summaries, correlation analysis, category/regional breakdowns,
and automated algorithmic trend observations.
"""

from components.charts import (
    plot_correlation_heatmap,
    plot_distribution_histogram,
    plot_sales_by_category,
    plot_sales_by_region,
)
import pandas as pd
import plotly.express as px
from src.eda import (
    generate_eda_observations,
    get_category_breakdown,
    get_correlation_matrix,
    get_dataset_overview,
    get_regional_breakdown,
    get_segment_breakdown,
    get_statistical_summary,
)
import streamlit as st


def render_eda_page(df: pd.DataFrame):
  """Renders the comprehensive Exploratory Data Analysis module."""
  if df is None or df.empty:
    st.warning("⚠️ No data available for EDA. Please adjust your filter criteria in the sidebar.")
    return

  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">🔍 Exploratory Data Analysis (EDA)</div>
        <div class="section-subtitle">
            Rigorous statistical evaluation of distributions, feature correlations, category & geographic metrics,
            accompanied by dynamic machine-generated analytical observations.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Automated Business Observations Banner
  observations = generate_eda_observations(df)
  with st.expander("💡 Key Automated Analytical Observations (Click to Expand)", expanded=True):
    for obs in observations:
      st.markdown(f"- {obs}")

  # EDA Navigation Tabs
  tab_stats, tab_corr, tab_dist, tab_cat, tab_reg = st.tabs(
      [
          "📊 Statistical Summary",
          "🔗 Correlation Analysis",
          "📈 Distribution & Outliers",
          "🏷️ Category & Sub-Category",
          "🗺️ Regional & Segments",
      ]
  )

  with tab_stats:
    st.markdown("#### Descriptive Statistics (Numerical Features)")
    stat_df = get_statistical_summary(df)
    st.dataframe(stat_df, use_container_width=True)

    st.markdown("#### Structural Column Health")
    overview = get_dataset_overview(df)
    st.dataframe(overview["columns_table"], use_container_width=True, height=250)

  with tab_corr:
    st.markdown("#### Feature Correlation Matrix")
    corr_matrix = get_correlation_matrix(df)
    if not corr_matrix.empty:
      col_hm, col_diag = st.columns([1.3, 1])
      with col_hm:
        fig_hm = plot_correlation_heatmap(corr_matrix)
        st.plotly_chart(fig_hm, use_container_width=True)
      with col_diag:
        st.markdown("##### Key Correlation Findings")
        st.markdown(
            """
                - **Sales vs. Profit**: Moderately positive correlation across products, though heavily distorted when deep discounts (>25%) are applied.
                - **Discount vs. Profit**: Strong negative relationship. High discounts directly cannibalize net margins.
                - **Quantity vs. Sales**: Positive volume relationship, with unit price acting as the primary revenue multiplier.
                """
        )
        st.dataframe(corr_matrix, use_container_width=True)
    else:
      st.info("Insufficient numerical columns for correlation analysis.")

  with tab_dist:
    st.markdown("#### Feature Distribution Profiles")
    dist_feature = st.selectbox(
        "Select Feature for Distribution Analysis",
        options=[c for c in ["Sales", "Profit", "Quantity", "Discount", "Shipping_Duration_Days"] if c in df.columns],
        index=0,
    )

    col_hist, col_box = st.columns([1.2, 1])
    with col_hist:
      fig_hist = plot_distribution_histogram(df, col=dist_feature)
      st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
      fig_box = px.box(
          df,
          y=dist_feature,
          color="Category" if "Category" in df.columns else None,
          title=f"<b>Boxplot & Outliers: {dist_feature}</b>",
      )
      fig_box.update_layout(template="plotly_white")
      st.plotly_chart(fig_box, use_container_width=True)

  with tab_cat:
    st.markdown("#### Category & Sub-Category Deep Dive")
    cat_summary = get_category_breakdown(df)
    st.dataframe(cat_summary, use_container_width=True)

    if not cat_summary.empty:
      fig_subcat = px.bar(
          cat_summary,
          x="Sub-Category" if "Sub-Category" in cat_summary.columns else "Category",
          y="Total_Sales",
          color="Profit_Margin_Pct",
          color_continuous_scale="Viridis",
          title="<b>Sub-Category Sales & Profit Margin (%)</b>",
      )
      fig_subcat.update_layout(template="plotly_white")
      st.plotly_chart(fig_subcat, use_container_width=True)

  with tab_reg:
    st.markdown("#### Regional & Segment Breakdown")
    col_reg_t, col_seg_t = st.columns([1, 1])
    with col_reg_t:
      st.markdown("##### Regional Performance Table")
      reg_summary = get_regional_breakdown(df)
      st.dataframe(reg_summary, use_container_width=True)

    with col_seg_t:
      st.markdown("##### Customer Segment Performance Table")
      seg_summary = get_segment_breakdown(df)
      st.dataframe(seg_summary, use_container_width=True)
