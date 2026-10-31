"""RetailIQ About Project & Documentation Page.

Comprehensive overview of architecture, data science pipeline, machine learning
methodology, technology stack, and business objectives.
"""

import streamlit as st


def render_about_page():
  """Renders the comprehensive project overview, methodology, and tech stack details."""
  st.markdown(
      """
    <div class="section-card">
        <div class="section-header">ℹ️ About RetailIQ Platform</div>
        <div class="section-subtitle">
            An Intelligent Retail Sales Analytics & Predictive Forecasting Platform engineered for commercial enterprise intelligence.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  col1, col2 = st.columns([1.2, 1])

  with col1:
    st.markdown("### 🎯 Problem Statement & Mission")
    st.write(
        """
        Modern omnichannel retail businesses face complex operational challenges: unmonetized volume traps,
        chronic discounting margin erosion, regional supply-chain inefficiencies, and inaccurate inventory forecasting.
        
        **RetailIQ** transforms raw transaction streams into actionable, quantitative decisions through:
        1. **Executive Visibility**: Unified C-suite monitoring of revenue, gross margin, AOV, and order velocity.
        2. **Deep Exploratory Intelligence**: Statistical distributions, correlation mechanics, and geographic contribution.
        3. **Portfolio Rationalization**: Pinpointing loss-leader SKUs and excessive promotional discount cannibalization.
        4. **Time-Aware Machine Learning**: Leakage-free sales forecasting with empirical 95% uncertainty intervals.
        5. **Prescriptive Action**: Automatically generated, data-backed operational mandates.
        """
    )

    st.markdown("### 🏗️ Technical Architecture")
    st.markdown(
        """
        ```
        RetailIQ Platform
        ├── Ingestion & Validation  ──> Flexible schema mapping, multi-encoding fallback, outlier clipping
        ├── Feature Engineering    ──> Cyclical transforms, calendar lags, leakage-free rolling statistics
        ├── Machine Learning Pipeline──> Chronological time-split, Linear/RF/GBM benchmarking, recursive horizon
        ├── BI Recommendation Engine──> Data-backed pricing elasticity, discount audits, inventory planning
        └── Streamlit Dashboard UI ──> Enterprise SaaS design system, interactive Plotly charts, global filters
        ```
        """
    )

  with col2:
    st.markdown("### 💻 Technology Stack")
    st.markdown(
        """
        - **Core Language**: Python 3.11+
        - **Data Processing**: Pandas 3.x, NumPy 2.x, SciPy
        - **Visualization**: Plotly Interactive Charts, Matplotlib, Seaborn
        - **Machine Learning**: Scikit-learn (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor)
        - **Application Framework**: Streamlit with Custom CSS & Enterprise BI Design System
        - **Model Persistence**: Joblib serialization
        - **Quality Assurance**: Automated Pytest test suite
        """
    )

    st.markdown("### 🔬 Machine Learning Methodology")
    st.markdown(
        """
        - **Target Variable**: Aggregate Sales Revenue (`Sales`)
        - **Leakage Prevention**: All rolling windows (`7D`, `14D`, `30D`) are computed strictly over shifted past values (`lag_1`).
        - **Validation Strategy**: Strict chronological split (80% historical training, 20% future held-out testing).
        - **Benchmark Metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Weighted Absolute Percentage Error (WAPE), and R² Score.
        - **Uncertainty Modeling**: Dynamic prediction intervals expanding as $\\sigma \\cdot \\sqrt{1 + 0.04 \\cdot h}$ across future horizons.
        """
    )
