"""RetailIQ Data Management & Ingestion Module.

Handles CSV file uploads, automatic schema mapping & validation, missing value &
duplicate detection, outlier remediation, and cleaned dataset export.
"""

from io import BytesIO
from config.settings import DEFAULT_PROCESSED_DATASET, DEFAULT_RAW_DATASET
import pandas as pd
from src.data_cleaning import clean_retail_data
from src.data_loader import load_clean_data, load_raw_data
from src.data_validation import validate_dataset
import streamlit as st


def render_data_management_page():
  """Renders the comprehensive Data Ingestion, Validation, and Management suite."""
  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">📁 Dataset Ingestion & Schema Conformance</div>
        <div class="section-subtitle">
            Upload custom enterprise transaction CSVs or utilize the built-in canonical Sample Superstore dataset.
            The pipeline automatically performs column mapping, date parsing, duplicate deduction, and outlier handling.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Data source selection tabs
  tab_upload, tab_active, tab_audit = st.tabs(
      ["📤 Upload Dataset", "📊 Active Schema & Health", "⚙️ Cleaning Pipeline & Export"]
  )

  with tab_upload:
    st.markdown("#### Upload New Transaction CSV")
    uploaded_file = st.file_uploader(
        "Choose a CSV retail file",
        type=["csv"],
        help="Upload a retail transaction CSV. RetailIQ will automatically detect and map standard column aliases.",
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
      if uploaded_file is not None:
        if st.button("Process & Activate Uploaded Dataset", type="primary"):
          with st.spinner("Ingesting, validating, and cleaning uploaded dataset..."):
            try:
              raw_df = pd.read_csv(uploaded_file, encoding="latin1")
              val_report = validate_dataset(raw_df)

              if not val_report.is_valid:
                st.error(
                    f"❌ Schema Validation Failed: Missing required columns: {val_report.missing_required_columns}. "
                    f"Detected columns: {list(raw_df.columns)}"
                )
              else:
                cleaned_df, audit = clean_retail_data(raw_df, save_processed=True)
                st.session_state["dataset"] = cleaned_df
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["validation_report"] = val_report
                st.session_state["cleaning_audit"] = audit
                st.success(
                    f"✅ Successfully ingested **{uploaded_file.name}**! "
                    f"Processed {len(cleaned_df):,} records across {len(cleaned_df.columns)} standardized columns."
                )
                st.rerun()
            except Exception as e:
              st.error(f"Error parsing file: {e}")

    with col_btn2:
      if st.button("Restore Default Superstore Dataset"):
        with st.spinner("Reloading canonical Sample Superstore dataset..."):
          df = load_clean_data(force_reclean=True)
          st.session_state["dataset"] = df
          st.session_state["dataset_name"] = "Sample Superstore (Default)"
          st.session_state["validation_report"] = validate_dataset(df)
          st.success("Canonical Sample Superstore dataset restored!")
          st.rerun()

    # Column Mapping Reference Table
    with st.expander("ℹ️ Supported Column Aliases & Mapping Rules"):
      st.markdown(
          """
            RetailIQ intelligently maps common transactional column variations:
            - **Order Date**: `Order Date`, `order_date`, `date`, `transaction_date`
            - **Sales**: `Sales`, `revenue`, `turnover`, `total_sales`, `amount`
            - **Profit**: `Profit`, `net_profit`, `earnings`, `margin`
            - **Quantity**: `Quantity`, `qty`, `units`, `items_sold`
            - **Category**: `Category`, `product_category`, `dept`
            - **Region**: `Region`, `zone`, `territory`
            - **Customer Segment**: `Segment`, `customer_segment`, `market_segment`
            """
      )

  # Active Dataset Status & Health
  active_df = st.session_state.get("dataset")
  if active_df is None or active_df.empty:
    st.info("Loading active dataset...")
    active_df = load_clean_data()
    st.session_state["dataset"] = active_df
    st.session_state["dataset_name"] = "Sample Superstore (Default)"
    st.session_state["validation_report"] = validate_dataset(active_df)

  report = st.session_state.get("validation_report") or validate_dataset(active_df)

  with tab_active:
    st.markdown(f"#### Active Dataset: **{st.session_state.get('dataset_name', 'Sample Superstore')}**")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
      st.metric("Total Records", f"{len(active_df):,}")
    with col_m2:
      st.metric("Total Features", f"{len(active_df.columns)}")
    with col_m3:
      st.metric("Duplicate Rows", f"{report.duplicate_rows:,}")
    with col_m4:
      st.metric("Missing Values", f"{report.total_missing_values:,}")

    # Column Mapping Table
    st.markdown("##### Schema Conformance & Column Mapping")
    map_rows = []
    for canon, actual in report.matched_columns.items():
      map_rows.append({
          "Canonical Feature": canon,
          "Source Column": actual,
          "Data Type": report.data_types.get(actual, "Unknown"),
          "Status": "✅ Mapped",
      })
    for unmapped in report.unmatched_columns:
      map_rows.append({
          "Canonical Feature": unmapped,
          "Source Column": "—",
          "Data Type": "N/A",
          "Status": "⚠️ Optional (Missing)",
      })
    st.dataframe(pd.DataFrame(map_rows), use_container_width=True, height=260)

    # Preview Dataset
    st.markdown("##### Raw Dataset Preview (First 15 Rows)")
    st.dataframe(active_df.head(15), use_container_width=True)

  with tab_audit:
    st.markdown("#### Outlier Detection & Preprocessing Parameters")
    
    col_out_left, col_out_right = st.columns([1, 1])
    with col_out_left:
      st.markdown("##### Outlier Analysis (IQR Method)")
      outlier_data = report.outlier_summary
      if outlier_data:
        out_rows = []
        for col_name, stats in outlier_data.items():
          out_rows.append({
              "Feature": col_name,
              "Outlier Count": stats["count"],
              "Outlier %": f"{stats['percentage']}%",
              "Lower Bound": f"${stats['lower_bound']:,.2f}",
              "Upper Bound": f"${stats['upper_bound']:,.2f}",
          })
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True)
      else:
        st.info("No numerical outlier summary available.")

    with col_out_right:
      st.markdown("##### Export Cleaned Dataset")
      st.write(
          "Download the fully preprocessed, cleaned, and feature-engineered dataset for external reporting or modeling."
      )
      
      csv_buffer = BytesIO()
      active_df.to_csv(csv_buffer, index=False)
      csv_bytes = csv_buffer.getvalue()

      st.download_button(
          label="📥 Download Cleaned CSV",
          data=csv_bytes,
          file_name="retailiq_cleaned_dataset.csv",
          mime="text/csv",
          type="primary",
      )

      st.markdown(
          f"""
            - **Cleaned File Size**: ~{len(csv_bytes) / 1024:.1f} KB
            - **Standard Format**: UTF-8 Encoded CSV
            - **Engineered Columns**: Temporal features, profit margins, shipping duration
            """
      )
