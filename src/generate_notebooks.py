"""Generates high-quality, executable Jupyter Notebooks for RetailIQ."""

from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook_01():
  nb = nbf.v4.new_notebook()
  cells = [
      nbf.v4.new_markdown_cell("""# RetailIQ — 01 Data Understanding & Ingestion
## Intelligent Retail Sales Analytics & Predictive Forecasting Platform

This notebook performs initial exploratory inspection and structural understanding of the **Sample Superstore** retail transaction dataset.

### Objectives:
1. Ingest raw retail transactions.
2. Inspect shape, dimensions, memory consumption, and schema types.
3. Detect missing values, duplicates, and initial anomalies.
4. Review statistical summary of core commercial fields (`Sales`, `Profit`, `Quantity`, `Discount`).
"""),
      nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Set project root
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from config.settings import DEFAULT_RAW_DATASET
from src.data_loader import load_raw_data, get_dataset_metadata
from src.data_validation import validate_dataset
"""),
      nbf.v4.new_markdown_cell("### 1. Ingest Raw Dataset"),
      nbf.v4.new_code_cell("""# Load raw dataset
raw_df = load_raw_data(DEFAULT_RAW_DATASET)
print(f"Dataset Shape: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")
raw_df.head(5)
"""),
      nbf.v4.new_markdown_cell("### 2. Schema Validation & Health Report"),
      nbf.v4.new_code_cell("""val_report = validate_dataset(raw_df)
print(f"Is Schema Conforming: {val_report.is_valid}")
print(f"Total Missing Values: {val_report.total_missing_values}")
print(f"Duplicate Rows: {val_report.duplicate_rows}")
print("Matched Canonical Columns:")
for canon, actual in val_report.matched_columns.items():
    print(f"  - {canon} <- '{actual}'")
"""),
      nbf.v4.new_markdown_cell("### 3. High-Level Executive Summary Metrics"),
      nbf.v4.new_code_cell("""meta = get_dataset_metadata(raw_df)
for k, v in meta.items():
    print(f"{k.replace('_', ' ').title()}: {v}")
"""),
      nbf.v4.new_markdown_cell("""### 4. Summary & Next Steps
- The dataset contains transactions spanning multiple operating years with orders across Furniture, Office Supplies, and Technology.
- Proceed to `02_data_cleaning.ipynb` for date parsing, outlier treatment, and feature enrichment.
"""),
  ]
  nb.cells = cells
  with open(NOTEBOOKS_DIR / "01_data_understanding.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
  print("Created 01_data_understanding.ipynb")


def create_notebook_02():
  nb = nbf.v4.new_notebook()
  cells = [
      nbf.v4.new_markdown_cell("""# RetailIQ — 02 Data Cleaning & Transformation Pipeline
## Schema Standardization, Date Parsing, Outlier Treatment & Metric Engineering

This notebook implements the complete data preprocessing and cleaning pipeline.

### Steps:
1. Column mapping and schema normalization.
2. Removal of duplicate records.
3. Multi-format date parsing (`Order Date`, `Ship Date`).
4. Financial and operational feature engineering (`Profit_Margin_Pct`, `Unit_Price`, `Shipping_Duration_Days`).
5. Outlier detection using IQR and clipping remediation.
6. Persistence to `data/processed/cleaned_superstore.csv`.
"""),
      nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path().resolve().parent
sys.path.append(str(project_root))

from src.data_loader import load_raw_data
from src.data_cleaning import clean_retail_data
"""),
      nbf.v4.new_markdown_cell("### 1. Execute Data Cleaning Pipeline"),
      nbf.v4.new_code_cell("""raw_df = load_raw_data()
clean_df, audit = clean_retail_data(
    raw_df,
    handle_outliers=True,
    outlier_method="clip",
    save_processed=True
)

print("Preprocessing Audit Report:")
print(f"  - Initial Shape: {audit['initial_rows']:,} rows, {audit['initial_cols']} cols")
print(f"  - Final Clean Shape: {audit['final_rows']:,} rows, {audit['final_cols']} cols")
print(f"  - Duplicates Dropped: {audit['duplicates_removed']}")
print(f"  - Saved Destination: {audit.get('saved_path')}")
print("\\nAudit Actions Taken:")
for act in audit['actions']:
    print(f"  * {act}")
"""),
      nbf.v4.new_markdown_cell("### 2. Inspect Engineered Temporal & Financial Columns"),
      nbf.v4.new_code_cell("""engineered_cols = [
    'Order Date', 'Ship Date', 'Shipping_Duration_Days', 
    'Order_Year', 'Order_Month', 'Order_Quarter', 
    'Sales', 'Profit', 'Profit_Margin_Pct', 'Unit_Price'
]
clean_df[engineered_cols].head(5)
"""),
  ]
  nb.cells = cells
  with open(NOTEBOOKS_DIR / "02_data_cleaning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
  print("Created 02_data_cleaning.ipynb")


def create_notebook_03():
  nb = nbf.v4.new_notebook()
  cells = [
      nbf.v4.new_markdown_cell("""# RetailIQ — 03 Exploratory Data Analysis (EDA)
## Statistical Distributions, Correlation Mechanics, and Portfolio Profitability

This notebook provides statistical evaluations, feature correlations, category/regional breakdowns, and automated business observations.
"""),
      nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path().resolve().parent
sys.path.append(str(project_root))

from src.data_loader import load_clean_data
from src.eda import (
    get_statistical_summary,
    get_correlation_matrix,
    get_category_breakdown,
    get_regional_breakdown,
    get_segment_breakdown,
    generate_eda_observations
)
"""),
      nbf.v4.new_markdown_cell("### 1. Load Processed Clean Dataset"),
      nbf.v4.new_code_cell("""df = load_clean_data()
print(f"Clean Data: {df.shape[0]:,} rows, {df.shape[1]} columns")
"""),
      nbf.v4.new_markdown_cell("### 2. Descriptive Statistics (Skewness & Kurtosis)"),
      nbf.v4.new_code_cell("""stats_df = get_statistical_summary(df)
stats_df
"""),
      nbf.v4.new_markdown_cell("### 3. Feature Correlation Matrix"),
      nbf.v4.new_code_cell("""corr = get_correlation_matrix(df)
corr
"""),
      nbf.v4.new_markdown_cell("### 4. Category & Sub-Category Performance"),
      nbf.v4.new_code_cell("""cat_df = get_category_breakdown(df)
cat_df.head(10)
"""),
      nbf.v4.new_markdown_cell("### 5. Algorithmic Trend Observations"),
      nbf.v4.new_code_cell("""obs = generate_eda_observations(df)
for o in obs:
    print(f"• {o}")
"""),
  ]
  nb.cells = cells
  with open(NOTEBOOKS_DIR / "03_exploratory_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
  print("Created 03_exploratory_analysis.ipynb")


def create_notebook_04():
  nb = nbf.v4.new_notebook()
  cells = [
      nbf.v4.new_markdown_cell("""# RetailIQ — 04 Machine Learning Sales Forecasting Pipeline
## Chronological Time-Aware Modeling, Multi-Algorithm Benchmark & Future Projections

This notebook implements the complete machine learning workflow:
1. Daily and weekly sequence aggregation.
2. Lag and rolling statistical feature engineering (avoiding data leakage).
3. Chronological train-test splitting (80% train, 20% held-out test).
4. Training and benchmarking:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. Metric evaluation: MAE, MSE, RMSE, R², WAPE.
6. Recursive multi-step future sales forecasting with 95% uncertainty intervals.
7. Model persistence.
"""),
      nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path().resolve().parent
sys.path.append(str(project_root))

from src.data_loader import load_clean_data
from src.sales_forecasting import (
    train_and_evaluate_all_models,
    generate_future_forecast,
    load_trained_model
)
"""),
      nbf.v4.new_markdown_cell("### 1. Train and Benchmark Candidate Models"),
      nbf.v4.new_code_cell("""df = load_clean_data()
bundle, comp_df, best_model, feat_df = train_and_evaluate_all_models(
    df, freq='D', test_size=0.20, save_best=True
)

print(f"Top Performing Model: {best_model}\\n")
print("Model Comparison Leaderboard:")
comp_df
"""),
      nbf.v4.new_markdown_cell("### 2. Multi-Step Recursive Future Forecasting"),
      nbf.v4.new_code_cell("""forecast_df = generate_future_forecast(
    model_bundle=bundle,
    model_name=best_model,
    horizon=30
)
print("Forecast Head (Next 30 Days):")
forecast_df.head(10)
"""),
      nbf.v4.new_markdown_cell("### 3. Verify Model Serialization & Reloading"),
      nbf.v4.new_code_cell("""saved = load_trained_model(best_model)
print(f"Loaded Serialized Model: {saved['model_name']}")
print(f"Trained Metrics: {saved['metrics']}")
print(f"Features: {len(saved['feature_names'])} total predictors")
"""),
  ]
  nb.cells = cells
  with open(NOTEBOOKS_DIR / "04_machine_learning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
  print("Created 04_machine_learning.ipynb")


if __name__ == "__main__":
  create_notebook_01()
  create_notebook_02()
  create_notebook_03()
  create_notebook_04()
  print("All 4 Jupyter Notebooks generated successfully!")
