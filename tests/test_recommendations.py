"""Unit tests for RetailIQ business intelligence recommendation engine."""

import pandas as pd
import pytest
from src.recommendation_engine import (
    RecommendationItem,
    generate_business_recommendations,
)


@pytest.fixture
def business_test_data() -> pd.DataFrame:
  """Generates sample transactional data with known anomalies:

  - Chronic loss-maker SKU
  - Unprofitable deep discounts
  - High-margin category
  - Lagging region
  """
  rows = []

  # Normal profitable transactions
  for i in range(50):
    rows.append({
        "Order ID": f"ORD-NORM-{i}",
        "Order Date": "2018-05-10",
        "Product Name": "Profitable Server",
        "Category": "Technology",
        "Region": "West",
        "Sales": 1000.0,
        "Profit": 300.0,
        "Discount": 0.05,
    })

  # Chronic loss-maker SKU with heavy discounts
  for i in range(20):
    rows.append({
        "Order ID": f"ORD-LOSS-{i}",
        "Order Date": "2018-06-15",
        "Product Name": "Toxic Heavy Desk",
        "Category": "Furniture",
        "Region": "Central",
        "Sales": 800.0,
        "Profit": -250.0,
        "Discount": 0.50,
    })

  # Lagging region
  for i in range(30):
    rows.append({
        "Order ID": f"ORD-REG-{i}",
        "Order Date": "2018-07-20",
        "Product Name": "Standard Folder",
        "Category": "Office Supplies",
        "Region": "South",
        "Sales": 200.0,
        "Profit": 4.0,  # 2% margin
        "Discount": 0.35,
    })

  return pd.DataFrame(rows)


def test_generate_business_recommendations(business_test_data):
  """Tests that recommendation engine detects pricing, portfolio, and regional anomalies."""
  recs = generate_business_recommendations(business_test_data)
  assert len(recs) >= 3

  domains = [r.domain for r in recs]
  assert "Pricing & Discounts" in domains
  assert "Product Portfolio" in domains

  # Verify deep discount recommendation flagged
  disc_rec = next(r for r in recs if r.domain == "Pricing & Discounts")
  assert disc_rec.priority == "Critical"
  assert ">=30% discounts" in disc_rec.metric_evidence

  # Verify product portfolio loss-maker flagged
  prod_rec = next(r for r in recs if r.domain == "Product Portfolio")
  assert "Toxic Heavy Desk" in prod_rec.metric_evidence
