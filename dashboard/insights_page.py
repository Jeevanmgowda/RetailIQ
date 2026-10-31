"""RetailIQ Module F: Business Intelligence & Recommendations Page.

Translates empirical retail transactions, pricing elasticity, and category margins
into prescriptive, high-ROI executive recommendations.
"""

from typing import List
import pandas as pd
from src.recommendation_engine import (
    RecommendationItem,
    generate_business_recommendations,
)
import streamlit as st


def render_insights_page(df: pd.DataFrame):
  """Renders the prescriptive Business Intelligence and Recommendation engine."""
  if df is None or df.empty:
    st.warning("⚠️ No data available for Business Intelligence. Please adjust your filter criteria in the sidebar.")
    return

  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">💡 Prescriptive Business Intelligence & Strategy Engine</div>
        <div class="section-subtitle">
            Actionable executive recommendations derived dynamically from empirical data metrics.
            Target capital re-allocation, discount ceilings, high-velocity inventory buffers, and territory expansion.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Generate data-backed recommendations
  recommendations: List[RecommendationItem] = generate_business_recommendations(df)

  if not recommendations:
    st.info("No critical business anomalies or optimization triggers detected under current filter selection.")
    return

  # Domain filter
  all_domains = sorted(list({r.domain for r in recommendations}))
  selected_domain = st.selectbox(
      "Filter Recommendations by Strategic Domain:",
      options=["All Domains"] + all_domains,
      index=0,
  )

  filtered_recs = recommendations
  if selected_domain != "All Domains":
    filtered_recs = [r for r in recommendations if r.domain == selected_domain]

  # Executive Summary Banner
  critical_count = sum(1 for r in recommendations if r.priority == "Critical")
  high_count = sum(1 for r in recommendations if r.priority == "High")
  medium_count = sum(1 for r in recommendations if r.priority == "Medium")

  col_s1, col_s2, col_s3, col_s4 = st.columns(4)
  with col_s1:
    st.metric("Total Recommendations", len(recommendations))
  with col_s2:
    st.metric("Critical Alerts", critical_count, delta="Immediate Action Needed", delta_color="inverse")
  with col_s3:
    st.metric("High Priority", high_count)
  with col_s4:
    st.metric("Medium / Strategic", medium_count)

  st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

  # Render Recommendation Cards
  for rec in filtered_recs:
    p_lower = rec.priority.lower()
    html_card = f"""
        <div class="rec-card {p_lower}">
            <div class="rec-header-row">
                <span class="rec-domain-tag">{rec.domain}</span>
                <span class="rec-priority-badge badge-{p_lower}">{rec.priority} Priority</span>
            </div>
            <div class="rec-title">{rec.title}</div>
            
            <div class="rec-section-label">📊 Quantitative Evidence:</div>
            <div class="rec-text">{rec.metric_evidence}</div>
            
            <div class="rec-section-label">🛠️ Prescriptive Action:</div>
            <div class="rec-text">{rec.prescriptive_action}</div>
            
            <div class="rec-section-label" style="color: #10B981;">🎯 Expected Business Impact:</div>
            <div class="rec-text" style="color: #065F46; font-weight: 600;">{rec.expected_impact}</div>
        </div>
        """
    st.markdown(html_card, unsafe_allow_html=True)
