"""RetailIQ Data Validation Module.

Provides robust validation, schema conformance checks, missing value &
duplicate detection, and data health reporting for retail datasets.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import CANONICAL_COLUMNS, COLUMN_ALIASES, REQUIRED_COLUMNS


@dataclass
class ValidationReport:
  """Structured data validation report."""

  is_valid: bool = False
  total_rows: int = 0
  total_columns: int = 0
  matched_columns: Dict[str, str] = field(default_factory=dict)
  unmatched_columns: List[str] = field(default_factory=list)
  missing_required_columns: List[str] = field(default_factory=list)
  duplicate_rows: int = 0
  total_missing_values: int = 0
  missing_by_column: Dict[str, int] = field(default_factory=dict)
  data_types: Dict[str, str] = field(default_factory=dict)
  numeric_columns: List[str] = field(default_factory=list)
  date_columns: List[str] = field(default_factory=list)
  categorical_columns: List[str] = field(default_factory=list)
  outlier_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
  issues: List[str] = field(default_factory=list)
  warnings: List[str] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


def normalize_col_name(col: str) -> str:
  """Normalize column name by lowercasing, stripping, and replacing separators."""
  return (
      str(col)
      .strip()
      .lower()
      .replace("-", " ")
      .replace("_", " ")
      .replace(".", " ")
  )


def map_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
  """Map incoming DataFrame columns to canonical column names using aliases.

  Returns:
      matched: mapping of canonical_column -> uploaded_column_name
      unmatched: list of canonical columns that could not be matched
  """
  matched: Dict[str, str] = {}
  raw_cols = list(df.columns)
  normalized_raw = {normalize_col_name(c): c for c in raw_cols}

  # First check exact canonical names
  for canonical in CANONICAL_COLUMNS:
    norm_canonical = normalize_col_name(canonical)
    if norm_canonical in normalized_raw:
      matched[canonical] = normalized_raw[norm_canonical]

  # Check aliases for remaining unmatched canonical columns
  for canonical in CANONICAL_COLUMNS:
    if canonical in matched:
      continue
    canonical_key = canonical.lower().replace("-", "_").replace(" ", "_")
    aliases = COLUMN_ALIASES.get(canonical_key, [normalize_col_name(canonical)])

    for alias in aliases:
      norm_alias = normalize_col_name(alias)
      if (
          norm_alias in normalized_raw
          and normalized_raw[norm_alias] not in matched.values()
      ):
        matched[canonical] = normalized_raw[norm_alias]
        break

  unmatched = [col for col in CANONICAL_COLUMNS if col not in matched]
  return matched, unmatched


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 1.5,
) -> Dict[str, Dict[str, Any]]:
  """Detect outliers using the Interquartile Range (IQR) technique."""
  outlier_summary: Dict[str, Dict[str, Any]] = {}
  if columns is None:
    columns = [
        c
        for c in ["Sales", "Profit", "Quantity", "Discount"]
        if c in df.columns
    ]

  for col in columns:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
      continue
    series = df[col].dropna()
    if len(series) < 4:
      continue

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = float(q1 - threshold * iqr)
    upper_bound = float(q3 + threshold * iqr)

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    count = int(len(outliers))
    pct = round((count / len(series)) * 100, 2) if len(series) > 0 else 0.0

    outlier_summary[col] = {
        "count": count,
        "percentage": pct,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "min": (
            round(float(outliers.min()), 4) if count > 0 else round(float(series.min()), 4)
        ),
        "max": (
            round(float(outliers.max()), 4) if count > 0 else round(float(series.max()), 4)
        ),
    }

  return outlier_summary


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
  """Run full validation suite on uploaded or loaded retail DataFrame."""
  report = ValidationReport()

  if df is None or df.empty:
    report.is_valid = False
    report.issues.append("Dataset is empty or could not be loaded.")
    return report

  report.total_rows = len(df)
  report.total_columns = len(df.columns)

  # Check column mappings
  matched, unmatched = map_columns(df)
  report.matched_columns = matched
  report.unmatched_columns = unmatched

  # Check required columns
  missing_required = [
      col for col in REQUIRED_COLUMNS if col not in matched
  ]
  report.missing_required_columns = missing_required

  if missing_required:
    report.issues.append(
        f"Missing required columns: {', '.join(missing_required)}. "
        f"Found matched columns: {list(matched.keys())}."
    )
    report.is_valid = False
  else:
    report.is_valid = True

  # Check duplicates
  duplicate_count = int(df.duplicated().sum())
  report.duplicate_rows = duplicate_count
  if duplicate_count > 0:
    report.warnings.append(
        f"Detected {duplicate_count:,} duplicate rows ({round((duplicate_count / report.total_rows) * 100, 2)}% of dataset)."
    )

  # Check missing values
  missing_counts = df.isnull().sum()
  report.total_missing_values = int(missing_counts.sum())
  report.missing_by_column = {
      col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0
  }

  if report.total_missing_values > 0:
    report.warnings.append(
        f"Detected {report.total_missing_values:,} total missing values across {len(report.missing_by_column)} columns."
    )

  # Classify column types
  for col in df.columns:
    report.data_types[col] = str(df[col].dtype)
    if pd.api.types.is_numeric_dtype(df[col]):
      report.numeric_columns.append(col)
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
      report.date_columns.append(col)
    else:
      # Check if string might be a date column
      sample = df[col].dropna().head(10).astype(str)
      is_date_like = any(keyword in col.lower() for keyword in ["date", "time", "day"])
      if is_date_like:
        report.date_columns.append(col)
      else:
        report.categorical_columns.append(col)

  # Outlier detection on numerical columns
  num_cols = [c for c in ["Sales", "Profit", "Quantity", "Discount"] if c in df.columns or any(matched.get(c) == col for col in df.columns)]
  report.outlier_summary = detect_outliers_iqr(df)

  return report
