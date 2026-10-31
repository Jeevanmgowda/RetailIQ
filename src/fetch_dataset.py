"""Helper script to download and verify the canonical Sample Superstore dataset."""

import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
)
DEST_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Sample - Superstore.csv"


def download_dataset():
  DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
  if DEST_PATH.exists() and DEST_PATH.stat().st_size > 100000:
    print(f"Dataset already exists at {DEST_PATH} ({DEST_PATH.stat().st_size} bytes)")
    return True

  print(f"Downloading Sample Superstore dataset from {DATASET_URL}...")
  headers = {"User-Agent": "Mozilla/5.0"}
  req = urllib.request.Request(DATASET_URL, headers=headers)
  with urllib.request.urlopen(req) as response:
    content = response.read()

  with open(DEST_PATH, "wb") as f:
    f.write(content)

  print(f"Successfully saved {len(content)} bytes to {DEST_PATH}")

  # Validate reading
  df = pd.read_csv(DEST_PATH, encoding="latin1")
  print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
  print("Columns:", list(df.columns))
  return True


if __name__ == "__main__":
  download_dataset()
