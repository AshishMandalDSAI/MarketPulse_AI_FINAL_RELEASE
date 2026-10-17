"""Forecast page — historical target metric + near-term forecast with prediction interval.

Dynamic column detection: uses canonical metadata to resolve date and target
column names instead of hardcoding 'Date' and 'Revenue'. Works with both
the legacy demo forecast and the generic pipeline forecast.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_marketing_data, load_forecast, load_canonical_metadata,
)
from dashboard.components import session_paths
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


@safe_page("Forecast", required_capability="forecasting")
def render():
    st.markdown('<div class="mp-header-badge">FORECAST</div>', unsafe_allow_html=True)
    st.title("Revenue Forecast")

    df = load_marketing_data()
    forecast = load_forecast()

    if forecast.empty:
        if session_paths.is_user_dataset_mode():
            st.warning("Forecast not available for this dataset. Re-run the analysis pipeline from the Upload page.")
        else:
            st.warning("Forecast not found. Run `python src/forecasting/forecast.py` first.")
        return

    # Resolve date column dynamically
    date_col = session_paths.get_date_column() or "Date"

    # Resolve target/revenue column dynamically
    meta = load_canonical_metadata()
    revenue_col = (meta.get("revenue_column") if meta else None) or "Revenue"

    # Detect forecast column names (handles both legacy and generic output)
    forecast_date_col = next((c for c in ["Date", "date"] if c in forecast.columns), None)
    forecast_val_col = next((c for c in ["Forecast_Revenue", "forecast"] if c in forecast.columns), None)
    upper_col = next((c for c in ["Upper_90", "upper_90"] if c in forecast.columns), None)
    lower_col = next((c for c in ["Lower_90", "lower_90"] if c in forecast.columns), None)

    if not forecast_val_col:
        st.warning("Forecast data format not recognized.")
        return

    st.markdown("A trend + seasonal Ridge regression forecast — chosen deliberately for "
                "reliability over complexity, with an approximate 90% prediction interval.")

    fig = go.Figure()

    # Historical data overlay (use dynamic column names)
    if date_col in df.columns and revenue_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[revenue_col],
            name=f"Historical {revenue_col}", line=dict(color=CHART_COLORWAY[0])
        ))

    # Forecast line
    x_axis = forecast[forecast_date_col] if forecast_date_col else forecast.index
    fig.add_trace(go.Scatter(
        x=x_axis, y=forecast[forecast_val_col],
        name="Forecast", line=dict(color=CHART_COLORWAY[3])
    ))

    # Prediction interval
    if upper_col and lower_col and upper_col in forecast.columns and lower_col in forecast.columns:
        fig.add_trace(go.Scatter(
            x=list(x_axis) + list(x_axis[::-1]),
            y=list(forecast[upper_col]) + list(forecast[lower_col][::-1]),
            fill="toself", fillcolor="rgba(217,119,6,0.18)", line=dict(color="rgba(0,0,0,0)"),
            name="90% Prediction Interval", showlegend=True,
        ))

    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Forecast Table</div>', unsafe_allow_html=True)
    st.dataframe(forecast, use_container_width=True, hide_index=True)
