"""RetailIQ Interactive Global Filters Component.

Provides dynamic, stateful sidebar filtering across dates, geographic regions,
product categories, and customer market segments.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st


@dataclass
class FilterState:
  """Container for active user filter selections."""

  start_date: Optional[date] = None
  end_date: Optional[date] = None
  regions: List[str] = None
  categories: List[str] = None
  segments: List[str] = None


def render_sidebar_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
  """Renders interactive multi-criteria filter controls in the Streamlit sidebar

  and applies them to the dataset.
  """
  st.sidebar.markdown("---")
  st.sidebar.markdown("### Global Filters")

  if df is None or df.empty:
    return df, {}

  filtered_df = df.copy()

  # 1. Date Range Filter
  date_col = "Order Date"
  min_date = None
  max_date = None

  if date_col in filtered_df.columns:
    if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
      filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce")

    valid_dates = filtered_df[date_col].dropna()
    if not valid_dates.empty:
      min_date = valid_dates.min().date()
      max_date = valid_dates.max().date()

      date_range = st.sidebar.date_input(
          "Date Range",
          value=(min_date, max_date),
          min_value=min_date,
          max_value=max_date,
          help="Filter all metrics and charts between selected dates.",
      )

      if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt, end_dt = date_range
        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_dt)
            & (filtered_df[date_col].dt.date <= end_dt)
        ]

  # 2. Region Filter
  all_regions = (
      sorted(df["Region"].dropna().unique().tolist())
      if "Region" in df.columns
      else []
  )
  selected_regions = all_regions
  if all_regions:
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=all_regions,
        default=all_regions,
        help="Select one or more geographic sales regions.",
    )
    if selected_regions:
      filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]
    else:
      filtered_df = filtered_df.iloc[0:0]  # empty if nothing selected

  # 3. Product Category Filter
  all_categories = (
      sorted(df["Category"].dropna().unique().tolist())
      if "Category" in df.columns
      else []
  )
  selected_categories = all_categories
  if all_categories:
    selected_categories = st.sidebar.multiselect(
        "Product Category",
        options=all_categories,
        default=all_categories,
        help="Filter transactions by product line category.",
    )
    if selected_categories:
      filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    else:
      filtered_df = filtered_df.iloc[0:0]

  # 4. Customer Segment Filter
  all_segments = (
      sorted(df["Segment"].dropna().unique().tolist())
      if "Segment" in df.columns
      else []
  )
  selected_segments = all_segments
  if all_segments:
    selected_segments = st.sidebar.multiselect(
        "Customer Segment",
        options=all_segments,
        default=all_segments,
        help="Filter by purchasing customer segment.",
    )
    if selected_segments:
      filtered_df = filtered_df[filtered_df["Segment"].isin(selected_segments)]
    else:
      filtered_df = filtered_df.iloc[0:0]

  # Filter summary info in sidebar
  st.sidebar.caption(
      f"Showing **{len(filtered_df):,}** of **{len(df):,}** records "
      f"({(len(filtered_df)/len(df)*100):.1f}% active)"
  )

  filter_state = {
      "regions": selected_regions,
      "categories": selected_categories,
      "segments": selected_segments,
      "records_matched": len(filtered_df),
  }

  return filtered_df, filter_state
