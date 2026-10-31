"""RetailIQ Plotly Interactive Charts Component.

Provides a unified, enterprise-grade data visualization library styled to match
SaaS and Power BI BI standards.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import CHART_COLOR_PALETTE, THEME


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
  """Applies clean enterprise styling, typography, and grid formatting to Plotly figure."""
  fig.update_layout(
      template="plotly_white",
      height=height,
      margin=dict(l=20, r=20, t=40, b=20),
      font=dict(family="Inter, sans-serif", size=12, color=THEME["text"]),
      plot_bgcolor="#FFFFFF",
      paper_bgcolor="#FFFFFF",
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1,
          font=dict(size=11),
      ),
      hoverlabel=dict(
          bgcolor="#FFFFFF",
          font_size=12,
          font_family="Inter, sans-serif",
          bordercolor="#E2E8F0",
      ),
  )
  fig.update_xaxes(
      showgrid=True,
      gridwidth=1,
      gridcolor="#F1F5F9",
      linecolor="#CBD5E1",
      zeroline=False,
  )
  fig.update_yaxes(
      showgrid=True,
      gridwidth=1,
      gridcolor="#F1F5F9",
      linecolor="#CBD5E1",
      zeroline=False,
  )
  return fig


# -------------------------------------------------------------
# Module A & Overview Charts
# -------------------------------------------------------------
def plot_revenue_over_time(
    df: pd.DataFrame, freq: str = "ME", date_col: str = "Order Date"
) -> go.Figure:
  """Creates smooth line/area chart of revenue over time."""
  data = df.dropna(subset=[date_col]).copy()
  data.set_index(date_col, inplace=True)
  ts = data["Sales"].resample(freq).sum().reset_index()

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=ts[date_col],
          y=ts["Sales"],
          mode="lines+markers",
          name="Revenue",
          line=dict(color=THEME["primary_light"], width=2.5),
          marker=dict(size=5, color=THEME["primary"]),
          fill="tozeroy",
          fillcolor="rgba(59, 130, 246, 0.08)",
          hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>",
      )
  )
  fig.update_layout(
      title=dict(
          text="<b>Revenue Trend Over Time</b>", font=dict(size=14, color="#1E3A8A")
      ),
      xaxis_title="",
      yaxis_title="Revenue ($)",
  )
  return style_figure(fig)


def plot_profit_over_time(
    df: pd.DataFrame, freq: str = "ME", date_col: str = "Order Date"
) -> go.Figure:
  """Creates profit trend bar chart with positive/negative color coding."""
  data = df.dropna(subset=[date_col]).copy()
  data.set_index(date_col, inplace=True)
  ts = data["Profit"].resample(freq).sum().reset_index()

  colors = [
      THEME["success"] if p >= 0 else THEME["danger"] for p in ts["Profit"]
  ]

  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          x=ts[date_col],
          y=ts["Profit"],
          name="Net Profit",
          marker_color=colors,
          hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Profit:</b> $%{y:,.2f}<extra></extra>",
      )
  )
  fig.update_layout(
      title=dict(
          text="<b>Net Operating Profit Trend</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="",
      yaxis_title="Profit ($)",
  )
  return style_figure(fig)


def plot_sales_by_category(df: pd.DataFrame) -> go.Figure:
  """Creates horizontal bar chart comparing Sales and Profit across Categories."""
  cat_df = (
      df.groupby("Category")
      .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
      .reset_index()
  )
  cat_df.sort_values(by="Sales", ascending=True, inplace=True)

  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          y=cat_df["Category"],
          x=cat_df["Sales"],
          name="Revenue",
          orientation="h",
          marker_color=THEME["primary"],
          hovertemplate="<b>%{y}</b><br>Sales: $%{x:,.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Bar(
          y=cat_df["Category"],
          x=cat_df["Profit"],
          name="Profit",
          orientation="h",
          marker_color=THEME["accent"],
          hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.2f}<extra></extra>",
      )
  )
  fig.update_layout(
      barmode="group",
      title=dict(
          text="<b>Revenue & Profit by Category</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Amount ($)",
      yaxis_title="",
  )
  return style_figure(fig)


def plot_sales_by_region(df: pd.DataFrame) -> go.Figure:
  """Creates modern donut chart showing regional revenue contribution."""
  reg_df = df.groupby("Region")["Sales"].sum().reset_index()

  fig = go.Figure(
      data=[
          go.Pie(
              labels=reg_df["Region"],
              values=reg_df["Sales"],
              hole=0.55,
              marker=dict(colors=CHART_COLOR_PALETTE),
              textinfo="label+percent",
              hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
          )
      ]
  )
  fig.update_layout(
      title=dict(
          text="<b>Regional Revenue Contribution</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      showlegend=True,
  )
  return style_figure(fig)


def plot_monthly_comparison(df: pd.DataFrame) -> go.Figure:
  """Creates Year-over-Year monthly sales comparison bar chart."""
  data = df.copy()
  if "Order Date" in data.columns:
    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month
    data["Month_Name"] = data["Order Date"].dt.strftime("%b")

    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    pivot = (
        data.pivot_table(
            index="Month_Name",
            columns="Year",
            values="Sales",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(month_order)
        .reset_index()
    )

    fig = go.Figure()
    years = [col for col in pivot.columns if col != "Month_Name"]
    for i, yr in enumerate(years):
      color = CHART_COLOR_PALETTE[i % len(CHART_COLOR_PALETTE)]
      fig.add_trace(
          go.Bar(
              x=pivot["Month_Name"],
              y=pivot[yr],
              name=str(yr),
              marker_color=color,
              hovertemplate=f"<b>%{{x}} {yr}</b><br>Sales: $%{{y:,.2f}}<extra></extra>",
          )
      )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Monthly Revenue Comparison (YoY)</b>",
            font=dict(size=14, color="#1E3A8A"),
        ),
        xaxis_title="",
        yaxis_title="Revenue ($)",
    )
    return style_figure(fig)
  return go.Figure()


def plot_top_products_bar(
    prod_df: pd.DataFrame, metric: str = "Total_Sales", n: int = 10
) -> go.Figure:
  """Creates clean horizontal bar chart for top products."""
  data = prod_df.head(n).iloc[::-1]  # Reverse for ascending horizontal display
  val_col = metric
  color = THEME["primary_light"] if "Sales" in metric else THEME["accent"]

  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          y=data["Product Name"].str.slice(0, 38),
          x=data[val_col],
          orientation="h",
          marker_color=color,
          hovertemplate="<b>%{y}</b><br>Amount: $%{x:,.2f}<extra></extra>",
      )
  )
  fig.update_layout(
      title=dict(
          text=f"<b>Top {n} Products by {'Revenue' if 'Sales' in metric else 'Profit'}</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Amount ($)",
      yaxis_title="",
  )
  return style_figure(fig, height=400)


# -------------------------------------------------------------
# Module B & EDA Charts
# -------------------------------------------------------------
def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
  """Creates interactive correlation heatmap."""
  if corr_df.empty:
    return go.Figure()

  fig = go.Figure(
      data=go.Heatmap(
          z=corr_df.values,
          x=corr_df.columns,
          y=corr_df.index,
          colorscale="Blues",
          zmin=-1,
          zmax=1,
          text=np.round(corr_df.values, 2),
          texttemplate="%{text}",
          textfont={"size": 11},
          hoverongaps=False,
      )
  )
  fig.update_layout(
      title=dict(
          text="<b>Correlation Matrix (Numerical Features)</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis=dict(tickangle=-45),
  )
  return style_figure(fig, height=420)


def plot_distribution_histogram(
    df: pd.DataFrame, col: str = "Sales"
) -> go.Figure:
  """Creates distribution histogram with statistical summary."""
  series = df[col].dropna()
  fig = px.histogram(
      df,
      x=col,
      nbins=40,
      color_discrete_sequence=[THEME["primary_light"]],
      opacity=0.85,
      marginal="box",
  )
  fig.update_layout(
      title=dict(
          text=f"<b>Distribution of {col}</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title=col,
      yaxis_title="Count",
  )
  return style_figure(fig, height=380)


# -------------------------------------------------------------
# Module C & Product Analysis Charts
# -------------------------------------------------------------
def plot_discount_vs_profit_tiers(tier_df: pd.DataFrame) -> go.Figure:
  """Plots dual-axis chart showing total sales and profit margin across discount tiers."""
  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          x=tier_df["Discount_Tier"],
          y=tier_df["Total_Sales"],
          name="Revenue ($)",
          marker_color=THEME["primary_light"],
          yaxis="y",
          hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=tier_df["Discount_Tier"],
          y=tier_df["Profit_Margin_Pct"],
          name="Profit Margin (%)",
          mode="lines+markers",
          line=dict(color=THEME["danger"], width=3),
          marker=dict(size=8, color=THEME["danger"]),
          yaxis="y2",
          hovertemplate="<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>",
      )
  )
  fig.update_layout(
      title=dict(
          text="<b>Discount Sensitivity: Revenue vs Profit Margin</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Discount Tier",
      yaxis=dict(title="Total Sales ($)"),
      yaxis2=dict(
          title="Profit Margin (%)",
          overlaying="y",
          side="right",
          showgrid=False,
      ),
  )
  return style_figure(fig)


def plot_quantity_vs_revenue_scatter(df: pd.DataFrame) -> go.Figure:
  """Scatter plot of Quantity vs Sales colored by Category and Profit."""
  sample_df = df.sample(min(len(df), 1200), random_state=42)
  fig = px.scatter(
      sample_df,
      x="Quantity",
      y="Sales",
      color="Category",
      size="Quantity",
      hover_data=["Sub-Category", "Profit"],
      color_discrete_sequence=CHART_COLOR_PALETTE,
      opacity=0.75,
  )
  fig.update_layout(
      title=dict(
          text="<b>Transaction Volume vs Sales Revenue</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Quantity Sold",
      yaxis_title="Sales ($)",
  )
  return style_figure(fig)


# -------------------------------------------------------------
# Module E & Machine Learning Charts
# -------------------------------------------------------------
def plot_actual_vs_predicted(
    dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray, model_name: str
) -> go.Figure:
  """Plots chronological actual vs predicted sales on test set."""
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=dates,
          y=y_true,
          mode="lines",
          name="Actual Sales",
          line=dict(color="#64748B", width=1.8),
          hovertemplate="<b>Actual:</b> $%{y:,.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=dates,
          y=y_pred,
          mode="lines",
          name=f"Predicted ({model_name})",
          line=dict(color=THEME["primary_light"], width=2.2),
          hovertemplate="<b>Predicted:</b> $%{y:,.2f}<extra></extra>",
      )
  )
  fig.update_layout(
      title=dict(
          text=f"<b>Test Set Evaluation: Actual vs Predicted Sales ({model_name})</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Date",
      yaxis_title="Sales ($)",
  )
  return style_figure(fig, height=400)


def plot_future_forecast_with_intervals(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str,
    recent_days: int = 90,
) -> go.Figure:
  """Plots recent historical actuals joined with future forecast and 95% uncertainty envelope."""
  recent_hist = (
      history_df.sort_values(by="Date").tail(recent_days).reset_index(drop=True)
  )

  fig = go.Figure()

  # 1. Historical Actuals
  fig.add_trace(
      go.Scatter(
          x=recent_hist["Date"],
          y=recent_hist["Sales"],
          mode="lines",
          name="Historical Sales",
          line=dict(color="#475569", width=2),
          hovertemplate="<b>Actual Date:</b> %{x|%b %d, %Y}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>",
      )
  )

  # 2. Upper Bound Ribbon
  fig.add_trace(
      go.Scatter(
          x=forecast_df["Date"],
          y=forecast_df["Upper_Bound_95"],
          mode="lines",
          line=dict(width=0),
          showlegend=False,
          hoverinfo="skip",
      )
  )

  # 3. Lower Bound Ribbon with Fill
  fig.add_trace(
      go.Scatter(
          x=forecast_df["Date"],
          y=forecast_df["Lower_Bound_95"],
          mode="lines",
          line=dict(width=0),
          fill="tonexty",
          fillcolor="rgba(59, 130, 246, 0.15)",
          name="95% Confidence Interval",
          hovertemplate="<b>95% Range:</b> $%{y:,.2f} - $%{text:,.2f}<extra></extra>",
          text=forecast_df["Upper_Bound_95"],
      )
  )

  # 4. Forecast Line
  fig.add_trace(
      go.Scatter(
          x=forecast_df["Date"],
          y=forecast_df["Predicted_Sales"],
          mode="lines+markers",
          name=f"Forecast ({model_name})",
          line=dict(color=THEME["primary_light"], width=2.5, dash="solid"),
          marker=dict(size=4, color=THEME["primary"]),
          hovertemplate="<b>Forecast Date:</b> %{x|%b %d, %Y}<br><b>Projected Sales:</b> $%{y:,.2f}<extra></extra>",
      )
  )

  fig.update_layout(
      title=dict(
          text=f"<b>Future Sales Forecast ({len(forecast_df)} Periods Ahead) with 95% Prediction Interval</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Date",
      yaxis_title="Sales ($)",
  )
  return style_figure(fig, height=440)


def plot_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
  """Plots side-by-side metric comparison bar chart across models."""
  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          x=comparison_df["Model"],
          y=comparison_df["RMSE"],
          name="RMSE ($)",
          marker_color=THEME["primary"],
          hovertemplate="<b>%{x}</b><br>RMSE: $%{y:,.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Bar(
          x=comparison_df["Model"],
          y=comparison_df["MAE"],
          name="MAE ($)",
          marker_color=THEME["accent"],
          hovertemplate="<b>%{x}</b><br>MAE: $%{y:,.2f}<extra></extra>",
      )
  )
  fig.update_layout(
      barmode="group",
      title=dict(
          text="<b>Model Performance Comparison (Error Metrics)</b>",
          font=dict(size=14, color="#1E3A8A"),
      ),
      xaxis_title="Algorithm",
      yaxis_title="Error ($)",
  )
  return style_figure(fig, height=350)
