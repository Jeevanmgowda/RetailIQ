"""RetailIQ UI Design System & Styling.

Provides custom CSS, theme variables, brand header, and card styles for a
premium SaaS / Power BI / Stripe-style analytics experience.
"""

import streamlit as st


def apply_custom_css():
  """Injects custom enterprise CSS for a polished, light-mode modern BI aesthetic."""
  css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    /* Top Brand Header */
    .brand-header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .brand-logo-icon {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.3rem;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.25);
    }
    .brand-name {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-tagline {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        margin: 0;
    }
    .brand-badge {
        background: #EFF6FF;
        color: #1E3A8A;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid #BFDBFE;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }
    .metric-context {
        font-size: 0.78rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .context-positive {
        color: #10B981;
    }
    .context-negative {
        color: #EF4444;
    }
    .context-neutral {
        color: #64748B;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 1.25rem;
    }

    /* Recommendation Cards */
    .rec-card {
        background: #FFFFFF;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .rec-card.critical {
        border-left-color: #EF4444;
    }
    .rec-card.high {
        border-left-color: #F59E0B;
    }
    .rec-card.medium {
        border-left-color: #3B82F6;
    }
    .rec-card.opportunity {
        border-left-color: #10B981;
    }

    .rec-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .rec-domain-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .rec-priority-badge {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .badge-critical {
        background: #FEE2E2;
        color: #991B1B;
    }
    .badge-high {
        background: #FEF3C7;
        color: #92400E;
    }
    .badge-medium {
        background: #EFF6FF;
        color: #1E40AF;
    }
    .badge-opportunity {
        background: #D1FAE5;
        color: #065F46;
    }

    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }
    .rec-section-label {
        font-weight: 700;
        color: #334155;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .rec-text {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 0.4rem;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    /* Clean Buttons */
    .stButton>button {
        background-color: #1E3A8A;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        color: #FFFFFF;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #64748B;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
    }
    </style>
    """
  st.markdown(css, unsafe_allow_html=True)


def render_brand_header(current_page: str = "Overview", dataset_info: str = "Sample Superstore"):
  """Renders the top branding header banner."""
  html = f"""
    <div class="brand-header-container">
        <div class="brand-title-wrap">
            <div class="brand-logo-icon">IQ</div>
            <div>
                <h1 class="brand-name">RetailIQ</h1>
                <p class="brand-tagline">Intelligent Retail Sales Analytics & Predictive Intelligence</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="brand-badge" style="background: #F1F5F9; color: #475569; border-color: #CBD5E1;">
                Active Module: <strong>{current_page}</strong>
            </span>
            <span class="brand-badge">
                Dataset: <strong>{dataset_info}</strong>
            </span>
        </div>
    </div>
    """
  st.markdown(html, unsafe_allow_html=True)
