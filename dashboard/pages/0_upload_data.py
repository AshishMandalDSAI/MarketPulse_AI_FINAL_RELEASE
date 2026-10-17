"""Upload Data page — the entry point for the dataset-first platform.

Lets the user upload any CSV/XLSX, shows profiling results and each
semantic-role guess with a confidence badge, auto-selects HIGH-confidence
roles, requires explicit confirmation for MEDIUM/LOW, and on Confirm runs
the full analysis pipeline and shows the resulting capabilities + a generic
KPI preview.
"""

import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DEMO_DATASET_ID
from dashboard.components import session_paths
from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
from src.data_engine.capability_engine import detect_capabilities
from src.data_engine.analysis_runner import run_full_analysis
from src.kpi.generic_kpis import compute_generic_kpis

BADGE = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🔴 LOW"}


def _load_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def render():
    st.title("📤 Upload Data")
    st.caption("The uploaded dataset is the single source of truth — MarketPulse AI adapts to whatever "
               "columns your file actually has.")

    if st.button("▶️ Or just try the built-in demo dataset"):
        session_paths.reset_to_demo_dataset()
        st.session_state["mp_force_page"] = "01 · 📊 Executive Overview"
        st.rerun()

    uploaded_file = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        return

    try:
        df = _load_dataframe(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return

    if df.empty:
        st.warning("This file has no rows.")
        return

    profile, mapping = profile_and_map(df)

    st.subheader("Profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{profile['n_rows']:,}")
    c2.metric("Columns", f"{profile['n_cols']:,}")
    c3.metric("Missing data (avg)", f"{profile['overall_missing_pct']:.1f}%")

    st.subheader("Column role guesses")
    st.caption("HIGH-confidence roles are auto-selected. MEDIUM/LOW roles need your confirmation below.")

    confirmed_roles = set()
    overrides = {}

    single_roles = ["date", "revenue", "sales", "cost", "impressions", "clicks",
                     "conversions", "orders", "profit", "margin"]
    all_columns = ["— none —"] + list(df.columns)

    for role in single_roles:
        guess = getattr(mapping, role)
        if not guess.is_mapped:
            continue
        level = guess.confidence_level
        label = f"**{role}** — guessed: `{guess.column}` ({BADGE.get(level, level)}, {guess.confidence:.2f})"
        if level == "HIGH":
            st.markdown(f"✅ {label}")
            confirmed_roles.add(role)
        else:
            st.markdown(label)
            default_idx = all_columns.index(guess.column) if guess.column in all_columns else 0
            chosen = st.selectbox(
                f"Confirm column for '{role}'", all_columns, index=default_idx, key=f"role_select_{role}",
            )
            if chosen != "— none —":
                confirmed_roles.add(role)
                if role == "date":
                    overrides["date_column"] = chosen
                elif role == "revenue":
                    overrides["revenue_column"] = chosen
            else:
                if role == "date":
                    overrides["date_column"] = None
                elif role == "revenue":
                    overrides["revenue_column"] = None

    st.markdown("**Marketing channel / spend columns**")
    candidate_channels = mapping.spend_like_columns
    if candidate_channels:
        selected_channels = []
        for guess in mapping.channels:
            level = guess.confidence_level
            checked = st.checkbox(
                f"{guess.column} ({BADGE.get(level, level)}, {guess.confidence:.2f})",
                value=(level == "HIGH"), key=f"channel_{guess.column}",
            )
            if checked:
                selected_channels.append(guess.column)
        overrides["channel_columns"] = selected_channels
        if selected_channels:
            confirmed_roles.add("channels")
    else:
        st.caption("No candidate spend/channel columns were detected.")

    other_cols = [c for c in df.columns if c not in (
        [overrides.get("date_column"), overrides.get("revenue_column")] + list(mapping.spend_like_columns)
    )]
    with st.expander(f"Other columns kept as generic measures/dimensions ({len(other_cols)})"):
        st.write(other_cols)

    if st.button("✅ Confirm & Analyze", type="primary"):
        # Phase 1: ALWAYS generate a fresh UUID. Never reuse mp_pending_dataset_id
        # or any previous ID. Even re-uploading the same file gets a new ID.
        dataset_id = f"upload_{uuid.uuid4().hex[:12]}"

        result = run_full_analysis(dataset_id, df=df, mapping=mapping,
                                    confirmed_roles=confirmed_roles, overrides=overrides)
        session_paths.set_active_dataset(dataset_id, filename=uploaded_file.name)

        st.success(f"Dataset analyzed as `{dataset_id}`.")
        st.subheader("Capabilities")
        for cap in result["capabilities"].capabilities:
            icon = "✅" if cap.available else "⛔"
            st.markdown(f"{icon} **{cap.name}** — {cap.reason}")

        st.subheader("Generic KPI preview")
        kpis = result["generic_kpis"]
        if kpis:
            cols = st.columns(min(4, len(kpis)))
            for i, kpi in enumerate(kpis[:4]):
                cols[i % len(cols)].metric(kpi.label, kpi.value)
            with st.expander("All generic KPIs"):
                for kpi in kpis:
                    st.write(f"**{kpi.label}**: {kpi.value} — {kpi.note}")

        with st.expander("Mapping decisions"):
            for line in result["report"]:
                st.write(f"- {line}")

        if st.button("Go to Executive Overview →"):
            st.session_state["mp_force_page"] = "01 · 📊 Executive Overview"
            st.rerun()
