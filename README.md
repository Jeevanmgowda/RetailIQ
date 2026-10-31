# RetailIQ: Intelligent Retail Sales Analytics & Forecasting Platform

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Charts-Plotly-239120.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

RetailIQ is an end-to-end, production-grade retail intelligence and predictive sales analytics platform designed for commercial enterprise decision-makers. It unifies transaction ingestion, exploratory data analysis, product portfolio profitability, geographic market analysis, leak-free machine learning forecasting, and data-backed business recommendations.

---

## Table of Contents
1. [Project Overview & Problem Statement](#project-overview--problem-statement)
2. [Key Objectives](#key-objectives)
3. [System Architecture](#system-architecture)
4. [Core Application Modules](#core-application-modules)
5. [Technology Stack](#technology-stack)
6. [Dataset Information & Ingestion](#dataset-information--ingestion)
7. [Machine Learning Methodology](#machine-learning-methodology)
8. [Installation & Setup](#installation--setup)
9. [Running the Application](#running-the-application)
10. [Running Automated Tests](#running-automated-tests)
11. [Project Structure](#project-structure)
12. [Project Limitations & Future Roadmap](#project-limitations--future-roadmap)
13. [License](#license)

---

## Project Overview & Problem Statement

Modern omnichannel retail businesses frequently suffer from **unmonetized volume traps**, **promotional discount margin erosion**, **regional supply-chain imbalances**, and **inaccurate demand forecasting**. Raw point-of-sale data often remains siloed in ERPs, leaving executives with backward-looking dashboards rather than forward-looking prescriptive intelligence.

**RetailIQ** solves this challenge by transforming transactional retail streams into predictive, high-ROI operational decisions. By combining statistical data profiling, multi-model regression benchmarking, recursive multi-step forecasting with 95% confidence intervals, and a dynamic recommendation engine, RetailIQ enables businesses to protect margins and scale revenue sustainably.

---

## Key Objectives

- **Executive Visibility**: Unified C-suite monitoring of revenue, gross profit, order velocity, unit volume, and profit margins.
- **Data Health & Resilient Ingestion**: Automatic column alias mapping, multi-encoding fallback, missing value imputation, duplicate deduction, and statistical outlier remediation.
- **Deep Exploratory Analysis**: Descriptive statistics (skewness, kurtosis), correlation mechanics, distribution profiles, and algorithmic narrative observations.
- **Product Portfolio Optimization**: Isolating top revenue drivers, profit anchors, chronic loss-maker SKUs, and margin trap products.
- **Leakage-Free Sales Forecasting**: Strict chronological time-aware train-test validation benchmarking Linear Regression, Random Forest, and Gradient Boosting Regressors.
- **Prescriptive Strategy**: Generating quantifiable, data-backed operational mandates spanning pricing floors, safety stock buffers, and regional freight efficiency.

---

## System Architecture

```
                                  ┌─────────────────────────────┐
                                  │   Raw Retail Transaction    │
                                  │   CSV / Superstore Dataset  │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │   Data Ingestion & Mapping  │
                                  │   (Aliases & Fallback Encod)│
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │ Data Cleaning & Validation  │
                                  │ (Deduplication, IQR Outlier)│
                                  └──────────────┬──────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   │                             │                             │
                   ▼                             ▼                             ▼
    ┌─────────────────────────────┐┌───────────────────────────┐┌─────────────────────────────┐
    │  Exploratory Data Analysis  ││ Product & Regional Analyt ││ Feature Engineering Pipeline│
    │ (Stats, Skew, Correlations) ││ (SKU Traps, Regional Split││ (Cyclical, Lags, Rollings)  │
    └──────────────┬──────────────┘└─────────────┬─────────────┘└──────────────┬──────────────┘
                   │                             │                             │
                   │                             │                             ▼
                   │                             │              ┌─────────────────────────────┐
                   │                             │              │ Chronological Split (80/20) │
                   │                             │              │ Linear / RF / Gradient Boost│
                   │                             │              └──────────────┬──────────────┘
                   │                             │                             │
                   ▼                             ▼                             ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                       Prescriptive Recommendation Engine                                │
    │              (Pricing Floors, Margin Rationalization, Seasonality Stock)                │
    └────────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    Streamlit Enterprise UI (SaaS / Power BI Aesthetic)                  │
    │           (Interactive Plotly Charts, Dynamic Filters, Forecast Interval Bands)         │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Application Modules

### 1. Executive Dashboard (Module A)
- **Top Financial KPIs**: Total Revenue, Total Operating Profit, Overall Profit Margin %, Total Unique Orders, Total Units Sold, and Average Order Value (AOV).
- **Interactive Visualizations**:
  - Revenue Over Time (Smooth Area Chart with Month-Year Aggregations).
  - Net Operating Profit Trend (Diverging Color Bars for Positive vs Negative Months).
  - Revenue & Profit by Product Category.
  - Regional Revenue Contribution (Donut Share Breakdown).
  - Year-over-Year Monthly Revenue Performance.
  - Top 8 Revenue-Generating Products.
- **Global Sidebar Filters**: Dynamic multi-criteria filtering by Date Range, Region, Product Category, and Customer Segment.

### 2. Exploratory Data Analysis (Module B)
- **Descriptive Statistics**: Metric counts, mean, standard deviation, interquartile ranges (Q1, Q2, Q3), skewness, and kurtosis.
- **Feature Correlation Matrix**: Interactive heatmap analyzing collinearity between financial and operational parameters (`Sales`, `Profit`, `Quantity`, `Discount`, `Unit Price`, `Shipping Duration`).
- **Distribution Profiles & Outlier Diagnostics**: Histograms and boxplots evaluating transactional skewness.
- **Category & Sub-Category Performance Tables**: Sub-category rankings by sales volume and margin percentage.
- **Automated Algorithmic Observations**: Data-driven analytical insights generated dynamically from calculated figures.

### 3. Product Performance Analysis (Module C)
- **Top 10 Products by Revenue & Profit**: Contrast top gross volume drivers against net margin anchors.
- **Chronic Loss-Making SKUs**: Identifies bottom-performing products causing large cumulative capital deficits.
- **High-Revenue Margin Traps**: Identifies high-volume products (>70th percentile sales) delivering sub-5% profit margins.
- **Discount Sensitivity & Elasticity**: Binned analysis demonstrating how discounts exceeding 20% trigger negative net margins.
- **Transaction Volume Dynamics**: Scatter plot evaluating quantity sold against turnover.

### 4. Regional & Customer Market Dynamics (Module D)
- **Geographic Contribution**: Regional revenue share, profit margins, and Average Order Value comparisons.
- **State-Level Rankings**: Top 15 states ranked by sales turnover and operational margin.
- **Customer Segment Economics**: Comparative purchasing patterns across Consumer, Corporate, and Home Office segments (AOV, orders per customer, revenue per customer).

### 5. Machine Learning – Sales Prediction & Forecasting (Module E)
- **Chronological Out-of-Sample Splitting**: 80% training set strictly precedes the 20% test set timeline to prevent temporal data leakage.
- **Multi-Algorithm Benchmarking**:
  1. *Linear Regression* (Standardized continuous baseline)
  2. *Random Forest Regressor* (Ensemble bagging trees with 150 estimators)
  3. *Gradient Boosting Regressor* (Boosting trees with regularized learning rate)
- **Evaluation Metrics**: MAE, MSE, RMSE, R² Score, and Weighted Absolute Percentage Error (WAPE).
- **Recursive Multi-Step Future Forecasting**: Projects sales 7 to 90 periods ahead with empirical 95% uncertainty intervals expanding as $\sigma_{\text{resid}} \cdot \sqrt{1 + 0.04 \cdot h}$.
- **Export Facility**: Download future forecast schedules and confidence intervals directly as CSV.

### 6. Business Intelligence & Strategy Engine (Module F)
- **Data-Backed Prescriptive Actions**: Formulates actionable business mandates directly supported by empirical metrics:
  - *Pricing Floors*: Halts margin erosion from discretionary discounts $\ge 30\%$.
  - *Portfolio Rationalization*: Advises vendor renegotiation or discontinuation for chronic deficit SKUs.
  - *Category Scaling*: Identifies highest-margin categories for expanded marketing allocation.
  - *Regional Efficiency*: Targets freight and logistics audits in underperforming territories.
  - *Seasonality Buffers*: Recommends inventory safety stock pre-building 45 days prior to peak shopping months.

---

## Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core runtime environment |
| **Data Processing** | Pandas 3.x, NumPy 2.x, SciPy | Vectorized manipulations, descriptive stats, correlations |
| **Interactive UI** | Streamlit | Modern reactive web application |
| **Visualizations** | Plotly Express & Graph Objects | Enterprise charts, custom tooltips, responsive grids |
| **Machine Learning** | Scikit-learn | Regressors, pipelines, scalers, cross-validation |
| **Model Persistence** | Joblib | Serialization and reuse of trained ML pipelines |
| **Testing** | Pytest | Automated unit and integration test suite |
| **Styling** | Custom CSS & Inter Typography | Light-mode enterprise SaaS / Power BI aesthetic |

---

## Dataset Information & Ingestion

RetailIQ natively integrates the canonical **Sample Superstore** transaction dataset:
- **Total Records**: 9,994 transactions.
- **Date Span**: January 2015 – December 2018.
- **Attributes**: `Order ID`, `Order Date`, `Ship Date`, `Ship Mode`, `Customer ID`, `Customer Name`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`, `Product ID`, `Category`, `Sub-Category`, `Product Name`, `Sales`, `Quantity`, `Discount`, `Profit`.

### Custom Dataset Ingestion
RetailIQ provides a flexible CSV ingestion engine:
1. Navigate to **Data Management** in the sidebar.
2. Upload any retail transactional CSV.
3. RetailIQ will match column aliases automatically (e.g., `turnover` $\rightarrow$ `Sales`, `date` $\rightarrow$ `Order Date`).
4. Review schema validation reports and download the standardized, cleaned CSV.

---

## Machine Learning Methodology

### 1. Leakage Prevention & Aggregation
To model physical sales without lookahead bias, transactions are aggregated to a continuous time-series (daily or weekly). All rolling statistical windows (7, 14, 30 periods) are calculated strictly on shifted prior observations (`lag_1`), guaranteeing zero contamination from future sales.

### 2. Time-Aware Validation
Standard random k-fold cross-validation is invalid for chronological sequences. RetailIQ utilizes a strict time-aware partition: the earliest 80% of dates constitute the training set, and the final 20% constitute the held-out evaluation window.

### 3. Algorithm Leaderboard
| Model | Optimization Objective | Key Hyperparameters |
| :--- | :--- | :--- |
| **Linear Regression** | Ordinary Least Squares | Standardized Scaler |
| **Random Forest Regressor** | Variance Reduction (Bagging) | 150 Trees, `max_depth=12`, `min_samples_split=4` |
| **Gradient Boosting Regressor** | Residual Gradient Minimization | 150 Estimators, `lr=0.05`, `max_depth=5` |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/RetailIQ.git
cd RetailIQ
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to access RetailIQ.

---

## Running Automated Tests

RetailIQ includes comprehensive automated test coverage for data validation, cleaning, feature engineering, model training, and recommendation logic:
```bash
pytest -v tests/
```

All 10 unit tests should pass with green status:
```
tests/test_data_cleaning.py::test_map_columns PASSED
tests/test_data_cleaning.py::test_validate_dataset PASSED
tests/test_data_cleaning.py::test_clean_retail_data PASSED
tests/test_data_cleaning.py::test_outlier_detection PASSED
tests/test_forecasting.py::test_aggregate_sales_by_frequency PASSED
tests/test_forecasting.py::test_create_forecasting_features PASSED
tests/test_forecasting.py::test_chronological_train_test_split PASSED
tests/test_forecasting.py::test_train_and_evaluate_all_models PASSED
tests/test_forecasting.py::test_future_forecast_generation PASSED
tests/test_recommendations.py::test_generate_business_recommendations PASSED
```

---

## Project Structure

```
RetailIQ/
│
├── app.py                      # Main Streamlit application entrypoint
├── requirements.txt            # Production dependencies
├── README.md                   # Comprehensive platform documentation
├── LICENSE                     # MIT Open-Source License
├── .gitignore                  # Git exclude specifications
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Schema rules, UI theme palette & paths
│
├── data/
│   ├── raw/                    # Canonical raw Sample Superstore CSV
│   └── processed/              # Cleaned & feature-engineered dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Multi-encoding loader & metadata extractor
│   ├── data_cleaning.py        # Schema mapping, dates, outliers, deduplication
│   ├── data_validation.py      # Schema conformance & health audit
│   ├── feature_engineering.py  # Cyclical, lag, rolling statistical features
│   ├── eda.py                  # Statistical summaries & automated observations
│   ├── product_analysis.py     # SKU rankings, margin traps, discount tiers
│   ├── regional_analysis.py    # Territory metrics, state ranking, customer AOV
│   ├── sales_forecasting.py    # Model pipelines, recursive future forecasting
│   ├── model_evaluation.py     # Regression metrics (MAE, RMSE, WAPE, R2)
│   ├── recommendation_engine.py# Data-backed prescriptive BI recommendation logic
│   └── generate_notebooks.py   # Notebook generation script
│
├── models/
│   └── saved_models/           # Serialized Joblib model pipelines
│
├── dashboard/
│   ├── __init__.py
│   ├── overview.py             # Module A: Executive C-suite dashboard
│   ├── data_management.py      # Data upload, schema audit & export
│   ├── eda_page.py             # Module B: Exploratory Data Analysis
│   ├── product_page.py         # Module C: Product Performance Analysis
│   ├── regional_page.py        # Module D: Regional & Customer Analysis
│   ├── prediction_page.py      # Module E: ML Sales Forecasting Studio
│   ├── insights_page.py        # Module F: Prescriptive Business Intelligence
│   └── about_page.py           # System documentation & tech stack
│
├── components/
│   ├── __init__.py
│   ├── styling.py              # Custom CSS design system & brand headers
│   ├── metrics.py              # KPI card renderers
│   ├── charts.py               # Enterprise Plotly charts library
│   └── filters.py              # Interactive sidebar filter component
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   └── 04_machine_learning.ipynb
│
└── tests/
    ├── __init__.py
    ├── test_data_cleaning.py
    ├── test_forecasting.py
    └── test_recommendations.py
```

---

## Project Limitations & Future Roadmap

### Current Limitations
1. **Univariate Time-Series Scope**: Future multi-step projections currently forecast aggregate commercial revenue; SKU-level multi-series forecasting can be added for fine-grained inventory replenishment.
2. **Exogenous Factors**: The model does not currently ingest macro-economic inflation rates, competitor pricing indices, or weather disruptions.

### Future Roadmap
- [ ] **Hierarchical Forecasting**: Implement reconciled bottom-up and top-down hierarchical forecasting (Category $\rightarrow$ Sub-Category $\rightarrow$ SKU).
- [ ] **Deep Learning Models**: Integrate Temporal Fusion Transformers (TFT) and N-BEATS for non-linear multi-horizon sequences.
- [ ] **Automated PDF Export**: One-click C-suite executive briefing PDF report compilation.
- [ ] **Real-time Webhook / ERP Integration**: Direct connectors for Shopify, Stripe, SAP, and Snowflake data warehouses.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
