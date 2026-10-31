"""RetailIQ Module E: Machine Learning Sales Prediction & Forecasting Page.

Enables time-aware model training, cross-model benchmarking (Linear Regression,
Random Forest, Gradient Boosting), chronological test-set evaluation, recursive
multi-step future sales forecasting with 95% confidence intervals, and prediction
exports.
"""

from io import BytesIO
from components.charts import (
    plot_actual_vs_predicted,
    plot_future_forecast_with_intervals,
    plot_model_comparison_bar,
)
import numpy as np
import pandas as pd
from src.sales_forecasting import (
    generate_future_forecast,
    train_and_evaluate_all_models,
)
import streamlit as st


def render_prediction_page(df: pd.DataFrame):
  """Renders the end-to-end Machine Learning Forecasting Studio."""
  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">🤖 Predictive Intelligence & Sales Forecasting Studio</div>
        <div class="section-subtitle">
            Train and benchmark machine learning algorithms using strict chronological time-aware validation.
            Generate future revenue projections with empirical 95% uncertainty intervals.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Check if model bundle is cached in session state
  bundle = st.session_state.get("ml_bundle")
  comparison_df = st.session_state.get("ml_comparison")
  best_model = st.session_state.get("ml_best_model")

  # Sidebar ML Training Settings
  with st.expander("⚙️ Forecasting Model Configuration & Training Controls", expanded=(bundle is None)):
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
      time_freq = st.selectbox(
          "Aggregation Frequency",
          options=["D", "W-MON"],
          format_func=lambda x: "Daily ('D')" if x == "D" else "Weekly ('W-MON')",
          index=0,
          help="Time frequency for sequence aggregation.",
      )
    with col_c2:
      test_ratio = st.slider(
          "Validation Split Ratio (Test Size)",
          min_value=0.10,
          max_value=0.35,
          value=0.20,
          step=0.05,
          help="Chronological portion of timeline reserved for out-of-sample evaluation.",
      )
    with col_c3:
      forecast_horizon = st.slider(
          "Future Forecast Horizon (Periods)",
          min_value=7,
          max_value=90,
          value=30,
          step=7,
          help="Number of future intervals to forecast recursively.",
      )

    train_btn = st.button("🚀 Train & Benchmark Models", type="primary")

  if train_btn or bundle is None:
    with st.spinner("Engineering lag & rolling features, training algorithms chronologically..."):
      try:
        bundle, comparison_df, best_model, feat_df = train_and_evaluate_all_models(
            df=df,
            freq=time_freq,
            test_size=test_ratio,
            save_best=True,
        )
        st.session_state["ml_bundle"] = bundle
        st.session_state["ml_comparison"] = comparison_df
        st.session_state["ml_best_model"] = best_model
        st.session_state["ml_horizon"] = forecast_horizon
        st.success(f"✅ Models trained successfully! Best performing model: **{best_model}**")
      except Exception as e:
        st.error(f"Error training models: {e}")
        return

  # Model Selection & Comparison Tabs
  tab_comp, tab_eval, tab_future, tab_export = st.tabs(
      [
          "🏆 Model Comparison & Leaderboard",
          "🎯 Test Set Validation",
          "🔮 Future Sales Forecast",
          "📥 Export Predictions",
      ]
  )

  with tab_comp:
    st.markdown("#### Chronological Cross-Model Performance Leaderboard")
    st.caption("Ranked by Root Mean Squared Error (RMSE) on out-of-sample chronological test set.")

    col_tb, col_chart = st.columns([1.2, 1])
    with col_tb:
      st.dataframe(comparison_df, use_container_width=True)
      st.info(
          f"🏆 **Winner: {best_model}** exhibited optimal generalization with lowest prediction error across the evaluation window."
      )

    with col_chart:
      fig_comp = plot_model_comparison_bar(comparison_df)
      st.plotly_chart(fig_comp, use_container_width=True)

  # Active Model for Prediction Analysis
  active_model_name = st.selectbox(
      "Select Model to Inspect:",
      options=list(bundle["results"].keys()),
      index=list(bundle["results"].keys()).index(best_model),
  )

  active_model_res = bundle["results"][active_model_name]
  test_dates = bundle["df_test"]["Date"]
  y_test = bundle["y_test"]
  y_pred = active_model_res["y_pred"]
  metrics = active_model_res["metrics"]

  with tab_eval:
    st.markdown(f"#### Test Set Performance: **{active_model_name}**")

    # Evaluation KPI cards
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
      st.metric("Test RMSE", f"${metrics.rmse:,.2f}")
    with col_e2:
      st.metric("Test MAE", f"${metrics.mae:,.2f}")
    with col_e3:
      st.metric("R² Score", f"{metrics.r2:.4f}")
    with col_e4:
      st.metric("WAPE", f"{metrics.wape:.1f}%")

    fig_actual_pred = plot_actual_vs_predicted(
        dates=test_dates,
        y_true=y_test,
        y_pred=y_pred,
        model_name=active_model_name,
    )
    st.plotly_chart(fig_actual_pred, use_container_width=True)

  with tab_future:
    st.markdown(f"#### Future Sales Forecast: **{active_model_name}**")
    horizon = st.session_state.get("ml_horizon", 30)

    with st.spinner("Generating recursive multi-step future projections..."):
      forecast_df = generate_future_forecast(
          model_bundle=bundle,
          model_name=active_model_name,
          horizon=horizon,
      )

    history_full = pd.concat([bundle["df_train"], bundle["df_test"]]).sort_values(by="Date")

    fig_future = plot_future_forecast_with_intervals(
        history_df=history_full,
        forecast_df=forecast_df,
        model_name=active_model_name,
        recent_days=90,
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Future Summary KPIs
    tot_proj_sales = forecast_df["Predicted_Sales"].sum()
    avg_proj_sales = forecast_df["Predicted_Sales"].mean()
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
      st.metric(f"Projected Total Revenue ({horizon} Days)", f"${tot_proj_sales:,.2f}")
    with col_f2:
      st.metric("Projected Daily Average", f"${avg_proj_sales:,.2f}")
    with col_f3:
      st.metric("Forecast Horizon", f"{horizon} Periods Ahead")

    st.markdown("##### Detailed Forecast Schedule")
    st.dataframe(forecast_df, use_container_width=True)

  with tab_export:
    st.markdown("#### Download Prediction Results")
    st.write(
        "Export the generated future sales forecast and confidence intervals as a structured CSV for business planning."
    )

    csv_buffer = BytesIO()
    forecast_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Future Forecast CSV",
        data=csv_bytes,
        file_name=f"retailiq_{active_model_name.lower().replace(' ', '_')}_forecast.csv",
        mime="text/csv",
        type="primary",
    )
