"""
MarketPulse AI — AI-Driven Marketing Mix Modeling & Budget Optimization Platform
====================================================================================
Main Streamlit entry point. Run with:

    streamlit run app.py

Requires the analytics/model pipeline to have been run at least once:

    python src/data/generate_dataset.py
    python src/data/data_cleaning.py
    python src/features/feature_engineering.py
    python src/analytics/marketing_analytics.py
    python src/models/marketing_mix_model.py
    python src/models/model_evaluation.py
    python src/models/channel_contribution.py
    python src/optimization/budget_optimizer.py
    python src/optimization/simulator.py
    python src/forecasting/forecast.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dashboard.styles.theme import THEME_CSS
from dashboard.components.data_loader import pipeline_ready
from dashboard.components import session_paths

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)

PAGES = {
    "00 · 📤 Upload Data": "dashboard.pages.0_upload_data",
    "01 · 📊 Executive Overview": "dashboard.pages.1_executive_overview",
    "02 · 📈 Marketing Performance": "dashboard.pages.2_marketing_performance",
    "03 · 🧠 Marketing Mix Model": "dashboard.pages.3_marketing_mix_model",
    "04 · 💰 Channel ROI": "dashboard.pages.4_channel_roi",
    "05 · 🎯 Budget Optimizer": "dashboard.pages.5_budget_optimizer",
    "06 · 🔮 What-If Simulator": "dashboard.pages.6_what_if_simulator",
    "07 · 📅 Forecast": "dashboard.pages.7_forecast",
    "08 · 🤖 AI Marketing Analyst": "dashboard.pages.8_ai_marketing_analyst",
    "09 · 💡 Recommendations": "dashboard.pages.9_recommendations",
    "10 · 📋 Reports": "dashboard.pages.10_reports",
    "11 · 🔮 Predictions": "dashboard.pages.11_predictions",
    "12 · 🔍 Anomalies & Drivers": "dashboard.pages.12_anomalies_drivers",
    "13 · 🧠 Decision Intelligence": "dashboard.pages.13_decision_intelligence",
}


def main():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 20px 0;">
                <div style="font-size: 22px; font-weight: 800; color: #E7EAF3;">📈 MarketPulse AI</div>
                <div style="font-size: 12px; color: #9AA4C2; margin-top: 4px;">
                    AI-Driven Marketing Mix Modeling &amp; Budget Optimization
                </div>
                <div style="font-size: 11px; color: #5B6CFF; margin-top: 6px; font-style: italic;">
                    "Turn Marketing Spend into Smarter Business Decisions."
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page_keys = list(PAGES.keys())
        default_page = st.session_state.pop("mp_force_page", None)
        default_index = page_keys.index(default_page) if default_page in page_keys else 0
        selection = st.radio("Navigate", page_keys, index=default_index, label_visibility="collapsed")
        st.markdown("---")
        # Phase 1: Active dataset indicator — shows clearly what dataset is active
        if session_paths.dataset_chosen():
            active = session_paths.get_active_dataset()
            if active and active["is_demo"]:
                st.caption("**Active Dataset:** NovaMart (demo)")
                st.caption("Synthetic dataset for academic demonstration")
            elif active:
                st.caption(f"**Active Dataset:** {active['filename'] or 'Uploaded'}")
                st.caption(f"ID: `{active['dataset_id'][:16]}`")
                if active.get("n_rows") and active.get("n_cols"):
                    st.caption(f"Rows: {active['n_rows']:,} | Cols: {active['n_cols']}")
                if st.button("↩️ Switch to demo dataset"):
                    session_paths.reset_to_demo_dataset()
                    st.rerun()
            # Phase 1: Reset Analysis button
            if st.button("🔄 Reset Analysis"):
                session_paths.reset_analysis()
                st.rerun()
        else:
            st.caption("No dataset active")

    # Dataset-first gate: before rendering any page except Upload Data, make
    # the person explicitly choose a dataset (uploaded, or the demo) rather
    # than silently defaulting to demo data on first load.
    if selection != "00 · 📤 Upload Data" and not session_paths.dataset_chosen():
        st.info("### No dataset is currently active. Upload a CSV/XLSX file to begin.")
        st.markdown("MarketPulse AI works with **your own data** — any CSV or Excel file. "
                    "You can also explore the platform with the built-in demo dataset first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Go to Upload Data", type="primary", use_container_width=True):
                st.session_state["mp_force_page"] = "00 · 📤 Upload Data"
                st.rerun()
        with col2:
            if st.button("▶️ Try Demo Dataset instead", use_container_width=True):
                session_paths.reset_to_demo_dataset()
                st.rerun()
        return

    if selection == "00 · 📤 Upload Data":
        import importlib
        module = importlib.import_module(PAGES[selection])
        module.render()
        return

    if not pipeline_ready():
        st.error("⚠️ Pipeline outputs not found — this dashboard needs the analytics/model pipeline to have "
                 "run at least once before it can display real numbers.")
        st.markdown(
            "**Option 1 — run it now (recommended for a live demo):**"
        )
        if st.button("▶️ Initialize Demo Data (run full pipeline)", type="primary"):
            with st.spinner("Generating dataset, training the MMM, and running optimization — this takes under a minute..."):
                import subprocess
                steps = [
                    "src/data/generate_dataset.py", "src/data/data_cleaning.py",
                    "src/features/feature_engineering.py", "src/analytics/marketing_analytics.py",
                    "src/models/marketing_mix_model.py", "src/models/model_evaluation.py",
                    "src/models/channel_contribution.py", "src/optimization/budget_optimizer.py",
                    "src/optimization/simulator.py", "src/forecasting/forecast.py",
                    "src/utils/database.py",
                ]
                failed_step = None
                for step in steps:
                    result = subprocess.run([sys.executable, step], cwd=str(ROOT), capture_output=True, text=True)
                    if result.returncode != 0:
                        failed_step = (step, result.stderr[-800:])
                        break
                if failed_step:
                    st.error(f"Pipeline step failed: `{failed_step[0]}`\n\n```\n{failed_step[1]}\n```")
                else:
                    st.cache_data.clear()
                    st.success("Demo data initialized — reloading...")
                    st.rerun()
        st.markdown("**Option 2 — run manually from a terminal:**")
        st.code(
            "python src/data/generate_dataset.py\n"
            "python src/data/data_cleaning.py\n"
            "python src/features/feature_engineering.py\n"
            "python src/analytics/marketing_analytics.py\n"
            "python src/models/marketing_mix_model.py\n"
            "python src/models/model_evaluation.py\n"
            "python src/models/channel_contribution.py\n"
            "python src/optimization/budget_optimizer.py\n"
            "python src/optimization/simulator.py\n"
            "python src/forecasting/forecast.py\n"
            "python src/utils/database.py",
            language="bash",
        )
        st.caption("See README.md → *Quick Start* for full details.")
        return

    import importlib
    module = importlib.import_module(PAGES[selection])
    module.render()


if __name__ == "__main__":
    main()
