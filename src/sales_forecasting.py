"""RetailIQ Machine Learning Sales Forecasting Pipeline.

Implements model training, chronological cross-validation, hyperparameter
configuration,
recursive multi-step future sales forecasting, uncertainty interval estimation,
and model persistence.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config.settings import RANDOM_STATE, SAVED_MODELS_DIR
from src.feature_engineering import (
    aggregate_sales_by_frequency,
    chronological_train_test_split,
    create_forecasting_features,
)
from src.model_evaluation import (
    RegressionMetrics,
    calculate_metrics,
    calculate_residuals,
    compare_models,
)


def get_model_pipeline(model_type: str, random_state: int = RANDOM_STATE):
  """Builds a scikit-learn pipeline for the specified regressor model."""
  m = model_type.lower()
  if "linear" in m:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
  elif "random forest" in m or "rf" in m:
    return Pipeline([
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    ])
  elif "gradient" in m or "gb" in m:
    return Pipeline([
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
            ),
        )
    ])
  else:
    raise ValueError(f"Unsupported model type: {model_type}")


def train_and_evaluate_all_models(
    df: pd.DataFrame,
    freq: str = "D",
    test_size: float = 0.20,
    save_best: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame, str, pd.DataFrame]:
  """Runs the entire ML pipeline:

  aggregation, feature generation, time-aware splitting, training and evaluation
  across Linear Regression, Random Forest, and Gradient Boosting.

  Returns:
      results: dict containing models, test predictions, metrics, and residuals
      comparison_df: ranked performance comparison table
      best_model_name: name of top performing model
      feat_df: engineered feature dataset
  """
  # 1. Aggregate time-series
  ts_df = aggregate_sales_by_frequency(df, freq=freq)
  if len(ts_df) < 60:
    # If daily too short or weekly requested, adjust frequency
    freq = "W-MON"
    ts_df = aggregate_sales_by_frequency(df, freq=freq)

  # 2. Engineer features
  feat_df = create_forecasting_features(ts_df)

  # 3. Chronological Train-Test Split
  X_train, X_test, y_train, y_test, df_train, df_test = (
      chronological_train_test_split(
          feat_df, test_size=test_size, target_col="Sales", date_col="Date"
      )
  )

  candidate_models = {
      "Linear Regression": get_model_pipeline("linear"),
      "Random Forest Regressor": get_model_pipeline("random forest"),
      "Gradient Boosting Regressor": get_model_pipeline("gradient boosting"),
  }

  results: Dict[str, Any] = {}

  for name, pipeline in candidate_models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    metrics = calculate_metrics(y_test, y_pred)
    res_diag = calculate_residuals(y_test, y_pred)

    results[name] = {
        "pipeline": pipeline,
        "metrics": metrics,
        "residuals": res_diag,
        "y_pred": y_pred,
        "feature_names": list(X_train.columns),
    }

  comparison_df, best_model_name = compare_models(results)

  # Save best model to disk
  if save_best:
    save_trained_model(
        pipeline=results[best_model_name]["pipeline"],
        model_name=best_model_name,
        feature_names=results[best_model_name]["feature_names"],
        metrics=results[best_model_name]["metrics"],
        freq=freq,
    )

  bundle = {
      "results": results,
      "comparison_df": comparison_df,
      "best_model_name": best_model_name,
      "X_train": X_train,
      "X_test": X_test,
      "y_train": y_train,
      "y_test": y_test,
      "df_train": df_train,
      "df_test": df_test,
      "freq": freq,
  }

  return bundle, comparison_df, best_model_name, feat_df


def save_trained_model(
    pipeline: Any,
    model_name: str,
    feature_names: List[str],
    metrics: RegressionMetrics,
    freq: str = "D",
    output_dir: Optional[Path] = None,
) -> Path:
  """Serializes trained model pipeline and metadata to disk."""
  dest_dir = output_dir or SAVED_MODELS_DIR
  dest_dir.mkdir(parents=True, exist_ok=True)

  clean_name = model_name.lower().replace(" ", "_")
  model_path = dest_dir / f"{clean_name}_pipeline.joblib"

  payload = {
      "model_name": model_name,
      "pipeline": pipeline,
      "feature_names": feature_names,
      "metrics": metrics.to_dict(),
      "freq": freq,
  }

  joblib.dump(payload, model_path)
  return model_path


def load_trained_model(model_name: str, model_dir: Optional[Path] = None) -> Dict[str, Any]:
  """Loads serialized pipeline and metadata from disk."""
  dest_dir = model_dir or SAVED_MODELS_DIR
  clean_name = model_name.lower().replace(" ", "_")
  path = dest_dir / f"{clean_name}_pipeline.joblib"
  if not path.exists():
    raise FileNotFoundError(f"Model file not found at: {path}")
  return joblib.load(path)


def generate_future_forecast(
    model_bundle: Dict[str, Any],
    model_name: str,
    horizon: int = 30,
) -> pd.DataFrame:
  """Generates iterative multi-step forward predictions beyond the historical dataset.

  Calculates dynamic prediction intervals (95% uncertainty envelope) based on
  empirical test residuals that expand gracefully over the forecast horizon.
  """
  model_info = model_bundle["results"][model_name]
  pipeline = model_info["pipeline"]
  feature_names = model_info["feature_names"]
  std_resid = model_info["residuals"]["std_residual"]
  freq = model_bundle.get("freq", "D")

  # Historical timeline
  df_full = pd.concat([model_bundle["df_train"], model_bundle["df_test"]]).sort_values(by="Date").reset_index(drop=True)
  last_date = df_full["Date"].max()

  # Create future date range
  if freq == "D":
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
  elif "W" in freq:
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
  elif "M" in freq:
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
  else:
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq=freq)

  # Iterative forecast loop
  forecast_records = []
  history_series = df_full.set_index("Date")["Sales"].copy()

  for step_idx, cur_date in enumerate(future_dates):
    # Construct dynamic features
    day_of_week = cur_date.dayofweek
    day_of_month = cur_date.day
    day_of_year = cur_date.dayofyear
    week_of_year = int(cur_date.isocalendar().week)
    month = cur_date.month
    quarter = cur_date.quarter
    year = cur_date.year

    row_feat: Dict[str, float] = {
        "day_of_week": float(day_of_week),
        "day_of_month": float(day_of_month),
        "day_of_year": float(day_of_year),
        "week_of_year": float(week_of_year),
        "month": float(month),
        "quarter": float(quarter),
        "year": float(year),
        "is_weekend": float(1 if day_of_week in [5, 6] else 0),
        "is_month_start": float(1 if cur_date.is_month_start else 0),
        "is_month_end": float(1 if cur_date.is_month_end else 0),
        "sin_dow": np.sin(2 * np.pi * day_of_week / 7),
        "cos_dow": np.cos(2 * np.pi * day_of_week / 7),
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "sin_doy": np.sin(2 * np.pi * day_of_year / 365.25),
        "cos_doy": np.cos(2 * np.pi * day_of_year / 365.25),
        "trend_step": float(len(history_series) + step_idx),
    }

    # Lags: pull from history_series
    for lag in [1, 2, 7, 14, 21, 28, 30]:
      if f"lag_{lag}" in feature_names:
        if len(history_series) >= lag:
          row_feat[f"lag_{lag}"] = float(history_series.iloc[-lag])
        else:
          row_feat[f"lag_{lag}"] = float(history_series.mean())

    # Rolling windows
    for w in [7, 14, 30]:
      if f"rolling_mean_{w}" in feature_names:
        window_slice = history_series.iloc[-w:] if len(history_series) >= w else history_series
        row_feat[f"rolling_mean_{w}"] = float(window_slice.mean())
        row_feat[f"rolling_std_{w}"] = float(window_slice.std() if len(window_slice) > 1 else 0.0)
        row_feat[f"rolling_min_{w}"] = float(window_slice.min())
        row_feat[f"rolling_max_{w}"] = float(window_slice.max())

    # Align to model features
    X_single = pd.DataFrame([{col: row_feat.get(col, 0.0) for col in feature_names}])

    # Predict
    pred_val = float(pipeline.predict(X_single)[0])
    pred_val = max(0.0, pred_val)

    # Uncertainty factor: expands with horizon sqrt(1 + 0.04 * h)
    uncertainty_mult = 1.96 * std_resid * np.sqrt(1.0 + 0.04 * (step_idx + 1))
    lower_bound = max(0.0, pred_val - uncertainty_mult)
    upper_bound = pred_val + uncertainty_mult

    forecast_records.append({
        "Date": cur_date,
        "Predicted_Sales": round(pred_val, 2),
        "Lower_Bound_95": round(lower_bound, 2),
        "Upper_Bound_95": round(upper_bound, 2),
    })

    # Append prediction to history series for subsequent recursive lags
    history_series.loc[cur_date] = pred_val

  return pd.DataFrame(forecast_records)
