"""RetailIQ Model Evaluation Module.

Computes standard regression evaluation metrics (MAE, MSE, RMSE, R2, MAPE,
WAPE),
compares multiple algorithms, and performs residual diagnostics.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
  """Structured regression performance metrics."""

  mae: float
  mse: float
  rmse: float
  r2: float
  mape: float
  wape: float

  def to_dict(self) -> Dict[str, float]:
    return {
        "MAE": round(self.mae, 2),
        "MSE": round(self.mse, 2),
        "RMSE": round(self.rmse, 2),
        "R2": round(self.r2, 4),
        "MAPE (%)": round(self.mape, 2),
        "WAPE (%)": round(self.wape, 2),
    }


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> RegressionMetrics:
  """Calculates comprehensive regression metrics.

  Includes WAPE (Weighted Absolute Percentage Error) to guard against zero
  division in MAPE.
  """
  y_true_arr = np.asarray(y_true, dtype=float)
  y_pred_arr = np.asarray(y_pred, dtype=float)

  # Non-negative clip for physical sales predictions
  y_pred_clipped = np.clip(y_pred_arr, a_min=0.0, a_max=None)

  mae = float(mean_absolute_error(y_true_arr, y_pred_clipped))
  mse = float(mean_squared_error(y_true_arr, y_pred_clipped))
  rmse = float(np.sqrt(mse))
  r2 = float(r2_score(y_true_arr, y_pred_clipped))

  # MAPE (excluding true values of 0 to avoid inf)
  nonzero_mask = y_true_arr > 0
  if nonzero_mask.sum() > 0:
    mape = float(
        np.mean(
            np.abs(
                (y_true_arr[nonzero_mask] - y_pred_clipped[nonzero_mask])
                / y_true_arr[nonzero_mask]
            )
        )
        * 100.0
    )
  else:
    mape = 0.0

  # WAPE = sum(|y - y_hat|) / sum(y) * 100
  total_true = np.sum(y_true_arr)
  wape = (
      float(np.sum(np.abs(y_true_arr - y_pred_clipped)) / total_true * 100.0)
      if total_true > 0
      else 0.0
  )

  return RegressionMetrics(
      mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape, wape=wape
  )


def compare_models(
    results: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, str]:
  """Constructs a side-by-side comparison table and identifies best model.

  Args:
      results: Mapping of model_name -> {'metrics': RegressionMetrics, ...}

  Returns:
      comparison_df: DataFrame with model metrics ranked by RMSE
      best_model_name: Name of top performing model
  """
  rows = []
  for model_name, res in results.items():
    metrics: RegressionMetrics = res["metrics"]
    row = {"Model": model_name, **metrics.to_dict()}
    rows.append(row)

  comparison_df = pd.DataFrame(rows)
  # Rank primarily by lowest RMSE, secondarily by lowest MAE
  comparison_df.sort_values(by=["RMSE", "MAE"], ascending=True, inplace=True)
  comparison_df.reset_index(drop=True, inplace=True)

  best_model_name = comparison_df.iloc[0]["Model"]
  return comparison_df, best_model_name


def calculate_residuals(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, Any]:
  """Computes error residuals and diagnostic statistics."""
  y_true_arr = np.asarray(y_true, dtype=float)
  y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)
  residuals = y_true_arr - y_pred_arr

  return {
      "residuals": residuals,
      "mean_residual": round(float(np.mean(residuals)), 2),
      "std_residual": round(float(np.std(residuals)), 2),
      "max_overprediction": round(float(np.min(residuals)), 2),
      "max_underprediction": round(float(np.max(residuals)), 2),
  }
