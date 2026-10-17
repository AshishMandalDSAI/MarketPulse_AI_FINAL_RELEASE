"""Marketing Mix Model page — model performance, diagnostics, curves.

Dual rendering: handles both the legacy demo MMM metrics format (with
model_comparison, diagnostics, actual_vs_predicted) and the generic
Ridge regression format from the dataset-first pipeline.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_model_metrics, load_model_diagnostics, load_actual_vs_predicted,
)
from dashboard.components.kpi_card import render_kpi_row
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


@safe_page("Marketing Mix Model", required_capability="marketing_mix_modeling")
def render():
    st.markdown('<div class="mp-header-badge">MARKETING MIX MODEL</div>', unsafe_allow_html=True)
    st.title("Marketing Mix Model (MMM)")

    metrics = load_model_metrics()
    diagnostics = load_model_diagnostics()
    avp = load_actual_vs_predicted()

    if not metrics:
        st.warning("Model metrics not found. Run the analysis pipeline first.")
        return

    # Detect format: legacy (has model_comparison) vs generic (flat dict)
    selected = metrics.get("selected_model")
    model_comparison = metrics.get("model_comparison")

    if selected and model_comparison and selected in model_comparison:
        # Legacy format with full model comparison
        test_m = model_comparison[selected]["test"]
        model_type_label = selected
    else:
        # Generic format — build test_m from flat keys
        test_m = {
            "r2": metrics.get("r2", 0),
            "rmse": metrics.get("rmse", 0),
            "mae": metrics.get("mae", 0),
            "mape": metrics.get("mape", 0),
        }
        model_type_label = metrics.get("model_type", "Ridge Regression")

    st.markdown(f"**Model:** `{model_type_label}`")

    render_kpi_row([
        {"label": "R² (test)", "value": f"{test_m['r2']*100:.1f}%" if test_m['r2'] < 1 else f"{test_m['r2']:.1f}%"},
        {"label": "RMSE (test)", "value": f"{test_m['rmse']:,.0f}"},
        {"label": "MAE (test)", "value": f"{test_m['mae']:,.0f}"},
        {"label": "MAPE (test)", "value": f"{test_m['mape']:.1f}%"},
    ])

    # Explanation callout
    model_type = metrics.get("model_type", "")
    if model_type == "generic_ridge_linear":
        st.markdown(
            '<div class="mp-callout">'
            'This is a lightweight linear model (Ridge regression) that estimates how each '
            'marketing channel\'s spend contributes to the target metric. It does not model '
            'adstock (carryover) or saturation (diminishing returns) — those diagnostics are '
            'not available for this model type.'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="mp-callout">'
            'How this works, in plain terms: the model assumes this week\'s marketing spend keeps '
            'influencing sales for a few weeks after (adstock / carryover), and that spending more on a '
            'channel helps less and less the more you already spend on it (saturation / diminishing '
            'returns). It combines these effects with business controls (price, promotions, holidays, '
            'competitor and economic conditions) to predict weekly revenue, and is evaluated on weeks it '
            'never saw during training.</div>', unsafe_allow_html=True
        )

    # Model comparison table (only if multiple models were compared)
    if model_comparison and len(model_comparison) > 1:
        st.markdown('<div class="mp-section-title">Model Comparison</div>', unsafe_allow_html=True)
        comp_rows = []
        for name, m in model_comparison.items():
            comp_rows.append({
                "Model": name, "Test R²": round(m["test"]["r2"], 3),
                "Test RMSE": round(m["test"]["rmse"], 0), "Test MAE": round(m["test"]["mae"], 0),
                "Test MAPE %": round(m["test"]["mape"], 2), "Selected": "✅" if name == selected else "",
            })
        st.dataframe(comp_rows, use_container_width=True, hide_index=True)

    # Features list (for generic models)
    features = metrics.get("features", [])
    n_obs = metrics.get("n_observations")
    if features:
        with st.expander(f"Model features ({len(features)} feature(s)" + (f", {n_obs} observations" if n_obs else "") + ")"):
            st.write(", ".join(features))

    # Actual vs Predicted (available for legacy demo)
    if not avp.empty:
        st.markdown('<div class="mp-section-title">Actual vs Predicted Revenue (Test Period)</div>', unsafe_allow_html=True)
        # Detect date column dynamically
        date_col = "Date"
        for candidate in ["Date", "date", "Week", "Period"]:
            if candidate in avp.columns:
                date_col = candidate
                break

        actual_col = next((c for c in ["Actual_Revenue", "actual", "Actual"] if c in avp.columns), None)
        pred_col = next((c for c in ["Predicted_Revenue", "predicted", "Predicted"] if c in avp.columns), None)
        resid_col = next((c for c in ["Residual", "residual"] if c in avp.columns), None)

        if actual_col and pred_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=avp[date_col], y=avp[actual_col], name="Actual", line=dict(color=CHART_COLORWAY[0])))
            fig.add_trace(go.Scatter(x=avp[date_col], y=avp[pred_col], name="Predicted", line=dict(color=CHART_COLORWAY[3], dash="dash")))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        if resid_col:
            st.markdown('<div class="mp-section-title">Residual Analysis</div>', unsafe_allow_html=True)
            fig2 = go.Figure(go.Bar(x=avp[date_col], y=avp[resid_col], marker_color=CHART_COLORWAY[5]))
            fig2.update_layout(template=PLOTLY_TEMPLATE, height=280, title="Residuals (Actual − Predicted)", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    # Diagnostics (adstock/saturation curves — only for legacy MMM, not generic Ridge)
    if diagnostics:
        curves = diagnostics.get("curves", {})
        if curves:
            st.markdown('<div class="mp-section-title">Adstock & Saturation Curves</div>', unsafe_allow_html=True)
            channel_choice = st.selectbox("Channel", list(curves.keys()))
            c = curves[channel_choice]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Scatter(y=c["adstock_decay_curve"], mode="lines+markers", line=dict(color=CHART_COLORWAY[2])))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                   title=f"Adstock Decay Curve — decay={c['decay_rate']}",
                                   xaxis_title="Weeks after spend", yaxis_title="Carryover effect",
                                   margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(go.Scatter(x=c["spend_grid"], y=c["saturation_response_curve"],
                                            mode="lines", line=dict(color=CHART_COLORWAY[4])))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                   title=f"Saturation / Response Curve — k={c['half_saturation_k']:.0f}",
                                   xaxis_title="Weekly spend", yaxis_title="Saturated response",
                                   margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
