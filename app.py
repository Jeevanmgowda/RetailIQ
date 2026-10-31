"""RetailIQ: Intelligent Retail Sales Analytics & Forecasting Platform.

Main Streamlit Application Entrypoint.
"""

from components.filters import render_sidebar_filters
from components.styling import apply_custom_css, render_brand_header
from dashboard.about_page import render_about_page
from dashboard.data_management import render_data_management_page
from dashboard.eda_page import render_eda_page
from dashboard.insights_page import render_insights_page
from dashboard.overview import render_overview_page
from dashboard.prediction_page import render_prediction_page
from dashboard.product_page import render_product_page
from dashboard.regional_page import render_regional_page
import pandas as pd
from src.data_loader import load_clean_data
from src.data_validation import validate_dataset
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="RetailIQ - Retail Sales Analytics & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Enterprise Design System & Styles
apply_custom_css()


def initialize_app_state():
  """Loads the default dataset into session state if not already initialized."""
  if "dataset" not in st.session_state or st.session_state["dataset"] is None:
    try:
      df = load_clean_data()
      st.session_state["dataset"] = df
      st.session_state["dataset_name"] = "Sample Superstore (Canonical)"
      st.session_state["validation_report"] = validate_dataset(df)
    except Exception as e:
      st.error(f"Failed to load default dataset: {e}")
      st.session_state["dataset"] = pd.DataFrame()
      st.session_state["dataset_name"] = "No Data"


def main():
  initialize_app_state()

  df_master = st.session_state.get("dataset")
  dataset_name = st.session_state.get("dataset_name", "Sample Superstore")

  # 3. Sidebar Navigation
  st.sidebar.markdown(
      """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.25rem;">
            <div style="background: linear-gradient(135deg, #1E3A8A, #3B82F6); color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.1rem;">IQ</div>
            <div>
                <h2 style="font-size: 1.25rem; font-weight: 800; color: #1E3A8A; margin: 0; line-height: 1;">RetailIQ</h2>
                <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">ENTERPRISE ANALYTICS</span>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

  menu_options = [
      "Dashboard Overview",
      "Data Management",
      "Exploratory Analysis",
      "Product Analysis",
      "Regional Analysis",
      "Sales Prediction",
      "Business Insights",
      "About Project",
  ]

  selected_page = st.sidebar.radio(
      "Navigation Menu",
      options=menu_options,
      index=0,
      label_visibility="collapsed",
  )

  # 4. Global Interactive Sidebar Filters (for analytical views)
  filter_pages = [
      "Dashboard Overview",
      "Exploratory Analysis",
      "Product Analysis",
      "Regional Analysis",
      "Business Insights",
  ]

  active_df = df_master
  if selected_page in filter_pages and df_master is not None and not df_master.empty:
    active_df, _ = render_sidebar_filters(df_master)

  # 5. Top Branding Header
  render_brand_header(current_page=selected_page, dataset_info=dataset_name)

  # 6. Page Routing
  if selected_page == "Dashboard Overview":
    render_overview_page(active_df)
  elif selected_page == "Data Management":
    render_data_management_page()
  elif selected_page == "Exploratory Analysis":
    render_eda_page(active_df)
  elif selected_page == "Product Analysis":
    render_product_page(active_df)
  elif selected_page == "Regional Analysis":
    render_regional_page(active_df)
  elif selected_page == "Sales Prediction":
    render_prediction_page(df_master)
  elif selected_page == "Business Insights":
    render_insights_page(active_df)
  elif selected_page == "About Project":
    render_about_page()

  # Sidebar Footer
  st.sidebar.markdown("---")
  st.sidebar.caption(
      "RetailIQ Platform v1.0.0<br>"
      "Production Data Science Architecture<br>"
      "© 2026 RetailIQ Analytics",
      unsafe_allow_html=True,
  )


if __name__ == "__main__":
  main()
