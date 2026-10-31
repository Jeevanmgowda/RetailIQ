"""Unit tests for RetailIQ machine learning sales forecasting pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    aggregate_sales_by_frequency,
    chronological_train_test_split,
    create_forecasting_features,
)
from src.model_evaluation import calculate_metrics
from src.sales_forecasting import (
    generate_future_forecast,
    train_and_evaluate_all_models,
)


@pytest.fixture
def synthetic_time_series_data() -> pd.DataFrame:
  """Generates 120 days of synthetic transactional retail records."""
  np.random.seed(42)
  dates = pd.date_range(start="2018-01-01", periods=120, freq="D")

  rows = []
  for i, dt in enumerate(dates):
    # Repeat transactions on same date occasionally
    base_sales = 200 + 50 * np.sin(2 * np.pi * i / 7) + 2 * i
    sales = max(10.0, float(base_sales + np.random.normal(0, 15)))
    profit = float(sales * 0.15 + np.random.normal(0, 5))
    rows.append({
        "Order ID": f"ORD-{i}",
        "Order Date": dt,
        "Ship Date": dt + pd.Timedelta(days=2),
        "Sales": sales,
        "Profit": profit,
        "Quantity": int(np.random.randint(1, 8)),
        "Discount": float(np.random.choice([0.0, 0.1, 0.2])),
        "Category": np.random.choice(
            ["Technology", "Furniture", "Office Supplies"]
        ),
        "Sub-Category": "Item",
        "Region": "West",
        "Segment": "Consumer",
    })
  return pd.DataFrame(rows)


def test_aggregate_sales_by_frequency(synthetic_time_series_data):
  """Tests that daily aggregation produces contiguous time-series with zero-fill."""
  ts_df = aggregate_sales_by_frequency(synthetic_time_series_data, freq="D")
  assert "Date" in ts_df.columns
  assert "Sales" in ts_df.columns
  assert len(ts_df) == 120
  assert ts_df["Sales"].isna().sum() == 0


def test_create_forecasting_features(synthetic_time_series_data):
  """Tests feature extraction and verifies no future target leakage."""
  ts_df = aggregate_sales_by_frequency(synthetic_time_series_data, freq="D")
  feat_df = create_forecasting_features(ts_df, lags=[1, 7], windows=[7])

  assert "sin_dow" in feat_df.columns
  assert "lag_1" in feat_df.columns
  assert "lag_7" in feat_df.columns
  assert "rolling_mean_7" in feat_df.columns

  # Verify lag_1 equals previous day's sales exactly
  # (feat_df starts at max_lag=7, so index 1 in feat_df corresponds to index 8 in ts_df)
  assert feat_df.iloc[1]["lag_1"] == feat_df.iloc[0]["Sales"]


def test_chronological_train_test_split(synthetic_time_series_data):
  """Verifies strict chronological partition without data leakage."""
  ts_df = aggregate_sales_by_frequency(synthetic_time_series_data, freq="D")
  feat_df = create_forecasting_features(ts_df, lags=[1, 2], windows=[7])

  X_train, X_test, y_train, y_test, df_train, df_test = (
      chronological_train_test_split(
          feat_df, test_size=0.25, target_col="Sales", date_col="Date"
      )
  )

  # Check lengths
  assert len(X_train) + len(X_test) == len(feat_df)
  # Verify max train date is strictly strictly earlier than min test date
  assert df_train["Date"].max() < df_test["Date"].min()


def test_train_and_evaluate_all_models(synthetic_time_series_data):
  """Tests model training, evaluation metrics, and comparison table."""
  bundle, comp_df, best_model, feat_df = train_and_evaluate_all_models(
      synthetic_time_series_data,
      freq="D",
      test_size=0.20,
      save_best=False,
  )

  assert len(bundle["results"]) == 3
  assert "Linear Regression" in bundle["results"]
  assert "Random Forest Regressor" in bundle["results"]
  assert "Gradient Boosting Regressor" in bundle["results"]
  assert not comp_df.empty
  assert best_model in bundle["results"]

  # Check metrics valid
  metrics = bundle["results"][best_model]["metrics"]
  assert metrics.rmse > 0
  assert metrics.mae > 0


def test_future_forecast_generation(synthetic_time_series_data):
  """Tests recursive multi-step forecasting and confidence interval calculation."""
  bundle, _, best_model, _ = train_and_evaluate_all_models(
      synthetic_time_series_data, freq="D", test_size=0.20, save_best=False
  )

  forecast_df = generate_future_forecast(
      model_bundle=bundle,
      model_name=best_model,
      horizon=14,
  )

  assert len(forecast_df) == 14
  assert "Predicted_Sales" in forecast_df.columns
  assert "Lower_Bound_95" in forecast_df.columns
  assert "Upper_Bound_95" in forecast_df.columns
  # Ensure uncertainty bounds are coherent (lower <= prediction <= upper)
  for _, row in forecast_df.iterrows():
    assert row["Lower_Bound_95"] <= row["Predicted_Sales"]
    assert row["Predicted_Sales"] <= row["Upper_Bound_95"]
