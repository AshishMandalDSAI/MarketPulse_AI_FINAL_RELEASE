"""Executive Overview page — dynamic, dataset-agnostic.

Phase 3: uses dynamic KPI discovery, automatic chart recommendations,
automatic insight detection, and relationship analysis from the Phase 3
analytics engines.  Falls back to legacy marketing-specific view when
the dataset has marketing capabilities AND no canonical metadata is
available (backward compatibility).

Never hardcodes Revenue, Date, Google Ads, Total_Marketing_Spend, etc.
in the dynamic path.  Every KPI has source, calculation, confidence.
Every chart has score and reason.  Every insight has evidence.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_marketing_data, load_kpis, load_channel_attribution,
    load_budget_optimization, load_budget_table,
    load_dynamic_analytics, load_capabilities, has_capability,
    load_generic_kpis,
)
from dashboard.components.kpi_card import render_kpi_row, render_kpi_card, format_inr
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


# ---------------------------------------------------------------------------
# Legacy marketing-specific rendering (preserved for marketing datasets)
# ---------------------------------------------------------------------------

def _generate_legacy_insights(kpis, attribution, budget_opt, budget_table):
    """Short, dynamically-generated bullet insights for marketing datasets."""
    insights = []
    if not attribution.empty:
        top = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).iloc[0]
        insights.append(
            f"💡 **{top['Channel']}** is the model's most efficient channel — an estimated "
            f"₹{top['Model_Estimated_Efficiency']:.2f} of incremental revenue per ₹1 spent."
        )
    if budget_opt:
        imp = budget_opt.get("expected_improvement_pct", 0)
        if imp > 0.5:
            insights.append(
                f"💡 Reallocating the current budget toward the optimizer's recommendation is projected to "
                f"lift predicted weekly revenue by **{imp:+.1f}%** — with no increase in total spend."
            )
    if not budget_table.empty:
        biggest_shift = budget_table.reindex(budget_table["Change"].abs().sort_values(ascending=False).index).iloc[0]
        direction = "increasing" if biggest_shift["Change"] > 0 else "decreasing"
        insights.append(
            f"💡 The single largest recommended move is **{direction} {biggest_shift['Channel']}** by "
            f"{format_inr(abs(biggest_shift['Change']), compact=False)} ({biggest_shift['Change_Pct']:+.1f}%)."
        )
    growth = kpis.get("revenue_growth_h1_to_h2")
    if growth is not None:
        direction = "grown" if growth >= 0 else "declined"
        insights.append(f"💡 Revenue has **{direction} {abs(growth)*100:.1f}%** comparing the first half to the "
                         f"second half of the observed period.")
    return insights


def _render_legacy_overview():
    """Full legacy marketing-specific overview (unchanged from pre-Phase 3)."""
    st.markdown('<div class="mp-header-badge">MARKETPULSE AI · EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)
    st.title("What's happening, why, and what to do next")
    st.caption("AI-Driven Marketing Mix Modeling & Budget Optimization — Turn Marketing Spend into Smarter "
               "Business Decisions.")

    df = load_marketing_data()
    kpis = load_kpis().get("kpis", {})
    attribution = load_channel_attribution()
    budget_opt = load_budget_optimization()
    budget_table = load_budget_table()

    overall_roas = kpis.get("overall_naive_roas", 0)
    revenue_growth = kpis.get("revenue_growth_h1_to_h2", 0) * 100
    predicted_rev = budget_opt.get("current_predicted_revenue", 0)
    optimized_rev = budget_opt.get("optimized_predicted_revenue", 0)
    improvement = budget_opt.get("expected_improvement_pct", 0)

    render_kpi_row([
        {"label": "Total Revenue", "value": format_inr(kpis.get("total_revenue", 0)),
         "delta": f"{revenue_growth:+.1f}% H1→H2", "positive": revenue_growth >= 0},
        {"label": "Marketing Spend", "value": format_inr(kpis.get("total_marketing_spend", 0)),
         "delta": f"{kpis.get('marketing_growth_h1_to_h2', 0)*100:+.1f}% H1→H2",
         "positive": kpis.get("marketing_growth_h1_to_h2", 0) >= 0},
        {"label": "Overall ROAS", "value": f"{overall_roas:.2f}x"},
        {"label": "Potential Improvement", "value": f"{improvement:+.1f}%",
         "delta": "via budget optimization", "positive": improvement >= 0},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    if predicted_rev and optimized_rev:
        render_kpi_row([
            {"label": "Model Predicted Revenue (current mix)", "value": format_inr(predicted_rev, compact=False)},
            {"label": "Optimized Revenue (AI recommended mix)", "value": format_inr(optimized_rev, compact=False)},
        ])

    st.markdown('<div class="mp-section-title">Revenue & Marketing Spend Trend</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Revenue"], name="Revenue",
                                  line=dict(color=CHART_COLORWAY[0], width=2)))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, l=10, r=10, b=10),
                           title="Weekly Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Total_Marketing_Spend"], name="Marketing Spend",
                                  line=dict(color=CHART_COLORWAY[1], width=2)))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, l=10, r=10, b=10),
                           title="Weekly Marketing Spend Trend")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Channel Contribution & Budget Distribution</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        if not attribution.empty:
            fig = go.Figure(go.Pie(labels=attribution["Channel"], values=attribution["Estimated_Contribution"],
                                    hole=0.45, marker=dict(colors=CHART_COLORWAY)))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10),
                               title="Model-Estimated Revenue Contribution by Channel")
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if not attribution.empty:
            spend_dist = attribution.sort_values("Spend", ascending=False)
            fig = go.Figure(go.Bar(x=spend_dist["Channel"], y=spend_dist["Spend"],
                                    marker_color=CHART_COLORWAY[2]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10),
                               title="Current Marketing Budget Distribution")
            st.plotly_chart(fig, use_container_width=True)

    if not budget_table.empty:
        st.markdown('<div class="mp-section-title">Current vs Optimized Budget</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=budget_table["Channel"], y=budget_table["Current_Budget"],
                              marker_color=CHART_COLORWAY[0]))
        fig.add_trace(go.Bar(name="AI Recommended", x=budget_table["Channel"], y=budget_table["Recommended_Budget"],
                              marker_color=CHART_COLORWAY[2]))
        fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=340, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Full channel-by-channel numbers and a live re-run of the optimizer are on the Budget "
                   "Optimizer page.")

    st.markdown('<div class="mp-section-title">Top Performing Channels</div>', unsafe_allow_html=True)
    if not attribution.empty:
        top3 = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).head(3)
        cols = st.columns(3)
        for col, (_, row) in zip(cols, top3.iterrows()):
            with col:
                st.markdown(
                    f"""<div class="mp-kpi-card">
                        <div class="mp-kpi-label">{row['Channel']}</div>
                        <div class="mp-kpi-value">₹{row['Model_Estimated_Efficiency']:.2f}</div>
                        <div class="mp-kpi-delta-pos">est. revenue per ₹1 spent</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="mp-section-title">🤖 AI-Driven Business Insights</div>', unsafe_allow_html=True)
    insights = _generate_legacy_insights(kpis, attribution, budget_opt, budget_table)
    for insight in insights:
        st.markdown(f'<div class="mp-callout" style="margin-bottom:8px;">{insight}</div>', unsafe_allow_html=True)

    with st.expander("What does this page mean, and how should management use it?"):
        st.markdown(
            "- **What's happening:** the KPI cards and trend charts above summarize actual revenue and spend "
            "over the observed period.\n"
            "- **Why it's happening:** the pie/bar charts show how the fitted Marketing Mix Model attributes "
            "revenue across channels, and how current budget is actually distributed.\n"
            "- **What to do:** the *Potential Improvement* figure and the Current-vs-Optimized chart come "
            "from a real optimization run against the fitted model — see the Budget Optimizer page to adjust "
            "assumptions and re-run it live."
        )


# ---------------------------------------------------------------------------
# Dynamic dataset-agnostic rendering (Phase 3)
# ---------------------------------------------------------------------------

def _render_dynamic_kpis(analytics: dict) -> None:
    """Render dynamically discovered KPIs as cards."""
    kpis = analytics.get("kpis", [])
    if not kpis:
        st.info("No KPIs could be discovered for this dataset.")
        return

    # Sort by priority (lower = more important)
    kpis_sorted = sorted(kpis, key=lambda k: k.get("priority", 999))

    # Show top 4 as primary KPI row
    primary = kpis_sorted[:4]
    cards = []
    for kpi in primary:
        card = {
            "label": kpi["label"],
            "value": kpi.get("formatted_value") or str(kpi.get("value", "")),
        }
        if kpi.get("delta"):
            card["delta"] = kpi["delta"]
            card["positive"] = kpi.get("delta_positive", True)
        cards.append(card)
    render_kpi_row(cards)

    # Show remaining KPIs in a second row if present
    secondary = kpis_sorted[4:8]
    if secondary:
        st.markdown("<br>", unsafe_allow_html=True)
        cards2 = []
        for kpi in secondary:
            card = {
                "label": kpi["label"],
                "value": kpi.get("formatted_value") or str(kpi.get("value", "")),
            }
            if kpi.get("delta"):
                card["delta"] = kpi["delta"]
                card["positive"] = kpi.get("delta_positive", True)
            cards2.append(card)
        render_kpi_row(cards2)

    # Expandable detail: all KPIs with explanation
    with st.expander("📊 KPI Details & Methodology"):
        for kpi in kpis_sorted:
            conf = kpi.get("confidence", 0)
            conf_label = "🟢" if conf >= 0.8 else ("🟡" if conf >= 0.5 else "🔴")
            st.markdown(
                f"**{kpi['label']}** = {kpi.get('formatted_value', kpi.get('value', ''))} "
                f"&nbsp;&nbsp;{conf_label} confidence {conf:.0%}\n\n"
                f"*Calculation:* `{kpi.get('calculation', 'N/A')}` &nbsp;|&nbsp; "
                f"*Source:* `{kpi.get('source_column', 'N/A')}` &nbsp;|&nbsp; "
                f"*Category:* {kpi.get('category', 'N/A')}\n\n"
                f"{kpi.get('explanation', '')}\n\n---"
            )


def _render_dynamic_charts(analytics: dict, df) -> None:
    """Render top recommended visualizations using Plotly."""
    charts = analytics.get("charts", [])
    if not charts:
        return

    # Take top 4 charts
    top_charts = sorted(charts, key=lambda c: -c.get("score", 0))[:4]

    for i in range(0, len(top_charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(top_charts):
                break
            chart = top_charts[idx]
            with col:
                _render_single_chart(chart, df)


def _render_single_chart(chart: dict, df) -> None:
    """Render a single chart recommendation as a Plotly figure."""
    chart_type = chart.get("chart_type", "bar")
    title = chart.get("title", "Chart")
    x_col = chart.get("x_column")
    y_col = chart.get("y_column")
    y_cols = chart.get("y_columns")
    aggregation = chart.get("aggregation", "none")
    top_n = chart.get("top_n")

    try:
        fig = go.Figure()

        if chart_type == "line" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col], y=plot_df[y_col],
                    name=y_col, mode="lines",
                    line=dict(color=CHART_COLORWAY[0], width=2),
                ))

        elif chart_type == "area" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col], y=plot_df[y_col],
                    name=y_col, fill="tozeroy", mode="lines",
                    line=dict(color=CHART_COLORWAY[1], width=2),
                ))

        elif chart_type == "bar" and x_col:
            if x_col in df.columns:
                if aggregation == "count":
                    counts = df[x_col].value_counts(dropna=True)
                    if top_n:
                        counts = counts.head(top_n)
                    fig.add_trace(go.Bar(
                        x=counts.index.astype(str), y=counts.values,
                        marker_color=CHART_COLORWAY[2],
                    ))
                elif y_col and y_col in df.columns:
                    import pandas as pd
                    if aggregation == "mean":
                        grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                    elif aggregation == "sum":
                        grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    else:
                        grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                    if top_n:
                        grouped = grouped.head(top_n)
                    fig.add_trace(go.Bar(
                        x=grouped.index.astype(str), y=grouped.values,
                        marker_color=CHART_COLORWAY[2],
                    ))

        elif chart_type == "horizontal_bar" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                import pandas as pd
                if aggregation == "sum":
                    grouped = df.groupby(y_col)[x_col].sum().sort_values(ascending=True)
                else:
                    grouped = df.groupby(y_col)[x_col].mean().sort_values(ascending=True)
                if top_n:
                    grouped = grouped.tail(top_n)
                fig.add_trace(go.Bar(
                    y=grouped.index.astype(str), x=grouped.values,
                    orientation="h", marker_color=CHART_COLORWAY[3],
                ))

        elif chart_type == "scatter" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                plot_df = df[[x_col, y_col]].dropna()
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col], y=plot_df[y_col],
                    mode="markers", name=f"{x_col} vs {y_col}",
                    marker=dict(color=CHART_COLORWAY[4], size=6, opacity=0.7),
                ))

        elif chart_type == "histogram" and x_col:
            if x_col in df.columns:
                import pandas as pd
                series = pd.to_numeric(df[x_col], errors="coerce").dropna()
                fig.add_trace(go.Histogram(
                    x=series, nbinsx=30,
                    marker_color=CHART_COLORWAY[5],
                ))

        elif chart_type == "donut" and x_col:
            if x_col in df.columns:
                counts = df[x_col].value_counts(dropna=True).head(8)
                fig = go.Figure(go.Pie(
                    labels=counts.index.astype(str), values=counts.values,
                    hole=0.45, marker=dict(colors=CHART_COLORWAY),
                ))

        elif chart_type == "box" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                fig.add_trace(go.Box(
                    x=df[x_col].astype(str), y=df[y_col],
                    marker_color=CHART_COLORWAY[6],
                ))

        elif chart_type == "heatmap" and y_cols:
            import pandas as pd
            valid_cols = [c for c in y_cols if c in df.columns]
            if len(valid_cols) >= 2:
                numeric_df = df[valid_cols].apply(pd.to_numeric, errors="coerce")
                corr = numeric_df.corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                    text=corr.values.round(2), texttemplate="%{text}",
                ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            margin=dict(t=30, l=10, r=10, b=10),
            title=dict(text=title, font=dict(size=14)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Score badge
        score = chart.get("score", 0)
        st.caption(f"Score: {score:.2f} — {chart.get('reason', '')}")

    except Exception as e:
        st.warning(f"Could not render chart '{title}': {e}")


def _render_dynamic_insights(analytics: dict) -> None:
    """Render automatically discovered insights."""
    insights = analytics.get("insights", [])
    if not insights:
        return

    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "ℹ️"}

    for insight in insights[:8]:
        icon = severity_icons.get(insight.get("severity", "info"), "ℹ️")
        conf = insight.get("confidence", 0)

        st.markdown(
            f'<div class="mp-callout" style="margin-bottom:8px;">'
            f'{icon} **{insight.get("title", "")}** (confidence: {conf:.0%})<br>'
            f'{insight.get("description", "")}<br>'
            f'<small style="color: #9AA4C2;">Evidence: {insight.get("evidence", "N/A")}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_dynamic_relationships(analytics: dict) -> None:
    """Render detected relationships."""
    relationships = analytics.get("relationships", [])
    if not relationships:
        return

    for rel in relationships[:5]:
        cols_str = " ↔ ".join(rel.get("columns", []))
        strength = rel.get("strength", 0)
        strength_bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))

        st.markdown(
            f"**{cols_str}** — {rel.get('relationship_type', 'unknown')} "
            f"({rel.get('direction', '')})\n\n"
            f"`{strength_bar}` strength: {strength:.3f}\n\n"
            f"{rel.get('description', '')}\n\n"
            f"*Evidence:* {rel.get('evidence', 'N/A')}\n\n---"
        )


def _render_data_quality_summary(analytics: dict) -> None:
    """Render a brief data quality summary from KPIs."""
    kpis = analytics.get("kpis", [])
    quality_kpi = next((k for k in kpis if k.get("category") == "quality"), None)
    if quality_kpi:
        score = quality_kpi.get("value", 100)
        color = "#16A34A" if score >= 80 else ("#D97706" if score >= 50 else "#DC2626")
        st.markdown(
            f'<div class="mp-kpi-card" style="display:inline-block; margin-right:16px;">'
            f'<div class="mp-kpi-label">Data Quality Score</div>'
            f'<div class="mp-kpi-value" style="color:{color}">{score:.1f}%</div>'
            f'<div style="color:#6B7280; font-size:12px;">{quality_kpi.get("explanation", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_capability_summary() -> None:
    """Show which analysis capabilities are available/unavailable."""
    caps = load_capabilities()
    if not caps:
        return

    available = []
    unavailable = []
    for key, entry in caps.items():
        if isinstance(entry, dict):
            name = entry.get("name", key)
            if entry.get("available", False):
                available.append(name)
            else:
                unavailable.append((name, entry.get("reason", "")))

    if available:
        st.success(f"**Available:** {', '.join(available)}")
    if unavailable:
        reasons = "; ".join(f"{n} ({r})" for n, r in unavailable[:5])
        st.info(f"**Not available for this dataset:** {reasons}")


def _render_dynamic_overview():
    """Full dynamic, dataset-agnostic executive overview."""
    analytics = load_dynamic_analytics()
    df = load_marketing_data()

    st.markdown('<div class="mp-header-badge">MARKETPULSE AI · EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)
    st.title("Executive Overview")
    st.caption("AI-driven automatic analysis — KPIs, trends, and insights discovered dynamically from your data.")

    # Section 1: Data Quality Summary
    st.markdown('<div class="mp-section-title">📋 Data Quality</div>', unsafe_allow_html=True)
    _render_data_quality_summary(analytics)

    # Section 2: Primary KPIs
    st.markdown('<div class="mp-section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    _render_dynamic_kpis(analytics)

    # Section 3: Recommended Visualizations
    st.markdown('<div class="mp-section-title">📈 Recommended Visualizations</div>', unsafe_allow_html=True)
    if df is not None and not df.empty:
        _render_dynamic_charts(analytics, df)
    else:
        st.info("No data available for visualization.")

    # Section 4: Key Insights
    st.markdown('<div class="mp-section-title">🤖 AI-Discovered Insights</div>', unsafe_allow_html=True)
    _render_dynamic_insights(analytics)

    # Section 5: Relationships
    with st.expander("🔗 Detected Relationships & Correlations"):
        _render_dynamic_relationships(analytics)

    # Section 6: Available Analysis Capabilities
    with st.expander("🧪 Analysis Capabilities"):
        _render_capability_summary()

    # Explainer
    with st.expander("What does this page show, and how was it generated?"):
        st.markdown(
            "This executive overview is **automatically generated** from your dataset. "
            "Every KPI, chart, insight, and relationship was discovered by analyzing "
            "the data's structure, types, and statistical properties — no column names "
            "were assumed or hardcoded.\n\n"
            "- **KPIs** are discovered by identifying target/measure columns, computing "
            "totals, averages, growth rates, and data quality metrics.\n"
            "- **Charts** are recommended based on column types and relationships "
            "(time + numeric → line; category + numeric → bar; etc.).\n"
            "- **Insights** detect trends, concentration, outliers, correlations, and "
            "data quality issues using statistical tests.\n"
            "- **Relationships** find Pearson correlations, group comparisons, and "
            "time-based trends.\n\n"
            "Each element includes its confidence level, evidence, and methodology."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@safe_page("Executive Overview")
def render():
    """Route to dynamic or legacy overview based on data availability."""
    # Try dynamic analytics first (Phase 3 path)
    analytics = load_dynamic_analytics()
    if analytics and analytics.get("kpis"):
        _render_dynamic_overview()
        return

    # Check if this is a marketing dataset with legacy capabilities
    if has_capability("marketing_mix_modeling") or has_capability("channel_attribution"):
        try:
            _render_legacy_overview()
            return
        except (KeyError, Exception):
            pass

    # Final fallback: dynamic overview even with limited analytics
    _render_dynamic_overview()


render()
