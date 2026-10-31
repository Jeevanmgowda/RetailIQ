"""RetailIQ Module C: Product Performance Analysis Page.

Evaluates SKU performance, profit drivers, loss-making items, discount
elasticity,
revenue traps, and generates targeted merchandising recommendations.
"""

from components.charts import (
    plot_discount_vs_profit_tiers,
    plot_quantity_vs_revenue_scatter,
    plot_top_products_bar,
)
import pandas as pd
import plotly.express as px
from src.product_analysis import (
    generate_product_recommendations,
    get_discount_vs_profit_binned,
    get_lowest_performing_products,
    get_top_products_by_profit,
    get_top_products_by_revenue,
    identify_high_revenue_low_profit_traps,
)
import streamlit as st


def render_product_page(df: pd.DataFrame):
  """Renders the comprehensive Product Portfolio & Performance module."""
  if df is None or df.empty:
    st.warning("⚠️ No data available for Product Analysis. Please adjust your filter criteria in the sidebar.")
    return

  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">📦 Product Portfolio & Profitability Analysis</div>
        <div class="section-subtitle">
            Identify top revenue drivers, profit champions, chronic loss-makers, and unmonetized volume traps.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Product Actionable Recommendations
  recs = generate_product_recommendations(df)
  if recs:
    st.markdown("#### Strategic Merchandising Recommendations")
    for r in recs:
      st.markdown(
          f"""
            <div class="rec-card {r['status'].lower()}">
                <div class="rec-header-row">
                    <span class="rec-domain-tag">{r['type']}</span>
                    <span class="rec-priority-badge badge-{r['status'].lower()}">{r['status']}</span>
                </div>
                <div class="rec-title">{r['title']}</div>
                <div class="rec-text">{r['description']}</div>
                <div class="rec-text" style="font-size: 0.8rem; color: #1E3A8A; font-weight: 600;">🎯 Expected Impact: {r['impact']}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

  # Tabs: Top Revenue, Top Profit, Loss Leaders & Traps, Discount Elasticity
  tab_top, tab_loss, tab_traps, tab_disc, tab_vol = st.tabs(
      [
          "🏆 Top Revenue & Profit",
          "⚠️ Loss-Making Products",
          "🪤 High-Revenue Margin Traps",
          "🏷️ Discount vs Profit Impact",
          "📦 Quantity vs Revenue Dynamics",
      ]
  )

  with tab_top:
    col_rev_top, col_prof_top = st.columns([1, 1])

    with col_rev_top:
      st.markdown("##### Top 10 Products by Revenue")
      top_rev = get_top_products_by_revenue(df, n=10)
      if not top_rev.empty:
        fig_r = plot_top_products_bar(top_rev, metric="Total_Sales", n=10)
        st.plotly_chart(fig_r, use_container_width=True)
        st.dataframe(top_rev[["Product Name", "Category", "Total_Sales", "Total_Profit", "Profit_Margin_Pct"]], use_container_width=True)
      else:
        st.info("No product revenue data available.")

    with col_prof_top:
      st.markdown("##### Top 10 Products by Operating Profit")
      top_prof = get_top_products_by_profit(df, n=10)
      if not top_prof.empty:
        fig_p = plot_top_products_bar(top_prof, metric="Total_Profit", n=10)
        st.plotly_chart(fig_p, use_container_width=True)
        st.dataframe(top_prof[["Product Name", "Category", "Total_Profit", "Total_Sales", "Profit_Margin_Pct"]], use_container_width=True)
      else:
        st.info("No product profit data available.")

  with tab_loss:
    st.markdown("##### Lowest-Performing SKUs (Greatest Cumulative Losses)")
    lowest_df = get_lowest_performing_products(df, n=12)
    if not lowest_df.empty:
      st.dataframe(lowest_df, use_container_width=True)
      fig_loss = px.bar(
          lowest_df,
          x="Total_Profit",
          y=lowest_df["Product Name"].str.slice(0, 40),
          orientation="h",
          color="Avg_Discount",
          color_continuous_scale="Reds",
          title="<b>Loss Magnitude & Associated Discounting</b>",
      )
      fig_loss.update_layout(template="plotly_white")
      st.plotly_chart(fig_loss, use_container_width=True)
    else:
      st.success("No loss-making products detected under active filters.")

  with tab_traps:
    st.markdown("##### High-Revenue, Low-Profit 'Margin Traps'")
    st.caption("Products generating top 30% volume but delivering profit margins under 5%.")
    traps_df = identify_high_revenue_low_profit_traps(df)
    if not traps_df.empty:
      st.dataframe(traps_df, use_container_width=True)
    else:
      st.info("No margin trap products identified with active filters.")

  with tab_disc:
    st.markdown("##### Discount Elasticity & Profit Inflection Analysis")
    binned_tier = get_discount_vs_profit_binned(df)
    if not binned_tier.empty:
      col_chart, col_tbl = st.columns([1.3, 1])
      with col_chart:
        fig_tier = plot_discount_vs_profit_tiers(binned_tier)
        st.plotly_chart(fig_tier, use_container_width=True)
      with col_tbl:
        st.dataframe(binned_tier, use_container_width=True)
        st.markdown(
            """
                **Takeaway**: Note how net margins turn negative beyond 20% discounts.
                Discounts between 0-10% preserve robust margins (>25%), whereas 30%+ discounts consistently produce severe net losses.
                """
        )

  with tab_vol:
    st.markdown("##### Transaction Volume vs Revenue Analysis")
    fig_scatter = plot_quantity_vs_revenue_scatter(df)
    st.plotly_chart(fig_scatter, use_container_width=True)
