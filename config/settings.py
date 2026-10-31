"""RetailIQ Configuration and Settings.

Centralized configuration for paths, schema definitions, UI styling, and ML
parameters.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure essential directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    VISUALIZATIONS_DIR,
    NOTEBOOKS_DIR,
]:
  directory.mkdir(parents=True, exist_ok=True)

# Default dataset path
DEFAULT_RAW_DATASET = RAW_DATA_DIR / "Sample - Superstore.csv"
DEFAULT_PROCESSED_DATASET = PROCESSED_DATA_DIR / "cleaned_superstore.csv"

# Canonical Column Schema
CANONICAL_COLUMNS = [
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
    "Customer Name",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Product ID",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]

REQUIRED_COLUMNS = [
    "Order Date",
    "Sales",
    "Profit",
    "Quantity",
    "Category",
    "Sub-Category",
    "Region",
    "Segment",
]

# Aliases for flexible column mapping
COLUMN_ALIASES = {
    "order_id": ["order id", "orderid", "order_number", "id"],
    "order_date": ["order date", "orderdate", "date", "transaction_date"],
    "ship_date": ["ship date", "shipdate", "shipping_date"],
    "ship_mode": ["ship mode", "shipmode", "shipping_mode"],
    "customer_id": ["customer id", "customerid", "client_id"],
    "customer_name": ["customer name", "customername", "client_name"],
    "segment": ["segment", "customer_segment", "market_segment"],
    "country": ["country", "nation"],
    "city": ["city", "town"],
    "state": ["state", "province"],
    "postal_code": ["postal code", "postalcode", "zip", "zip code", "zipcode"],
    "region": ["region", "zone", "territory"],
    "product_id": ["product id", "productid", "item_id", "sku"],
    "category": ["category", "product_category", "dept"],
    "sub_category": ["sub-category", "subcategory", "sub_category", "product_type"],
    "product_name": ["product name", "productname", "item_name", "title"],
    "sales": ["sales", "revenue", "turnover", "total_sales", "amount"],
    "quantity": ["quantity", "qty", "units", "items_sold"],
    "discount": ["discount", "disc", "discount_rate"],
    "profit": ["profit", "net_profit", "earnings", "margin"],
}

# Date Format Parsing Candidates
DATE_FORMATS = [
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
]

# UI / Brand Theme Palette
THEME = {
    "primary": "#1E3A8A",      # Deep Navy Blue
    "primary_light": "#3B82F6",# Vibrant Blue
    "accent": "#0D9488",       # Teal / Emerald
    "accent_hover": "#0F766E",
    "success": "#10B981",      # Emerald Green
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Rose Red
    "background": "#F8FAFC",   # Crisp Slate Gray 50
    "surface": "#FFFFFF",      # Pure White Card
    "surface_subtle": "#F1F5F9",
    "text": "#0F172A",         # Slate 900
    "text_muted": "#64748B",   # Slate 500
    "border": "#E2E8F0",       # Slate 200
}

# Plotly Color Scales
PLOTLY_TEMPLATE = "plotly_white"
CHART_COLOR_PALETTE = [
    "#1E3A8A",  # Navy
    "#0D9488",  # Teal
    "#3B82F6",  # Blue
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#10B981",  # Green
    "#64748B",  # Slate
]

# Machine Learning Defaults
ML_TARGET_COLUMN = "Sales"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_FORECAST_HORIZON = 30  # Days
DEFAULT_TIME_FREQUENCY = "D"   # Daily aggregation
RANDOM_STATE = 42
