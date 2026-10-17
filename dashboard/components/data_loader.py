"""Cached data-loading helpers shared across all dashboard pages.

Centralizing loads here means expensive operations (reading CSVs, unpickling
the trained model) happen once per session via st.cache_data / st.cache_resource
rather than on every page render (Phase 15 — Performance requirement).

Dataset-first update: these loaders now resolve DATA_DIR/MODELS_DIR against
the *active* dataset (built-in demo, or an uploaded dataset) via
dashboard.components.session_paths, and tolerate EmptyDataError since some
pipeline stages legitimately write empty CSVs (e.g. a dataset with 0
marketing channels writes an empty channel_roi.csv).

Phase 1: st.cache_data is cleared on every dataset switch (via
session_paths._clear_page_caches), so cached results cannot leak between
datasets. has_capability() now returns False for uploaded datasets with
no capabilities.json, preventing silent fallback to demo behavior.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components import session_paths

# Phase 1: these module-level constants are ONLY for backward compat
# in demo mode. They must NOT be used for uploaded datasets.
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def _safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    """Reads a CSV, gracefully handling a missing file or a legitimately
    empty one (pandas.errors.EmptyDataError) instead of crashing."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates) if parse_dates else pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data
def load_marketing_data() -> pd.DataFrame:
    data_dir = session_paths.get_processed_dir()
    # dataset-first: standardized_data.csv for an uploaded dataset,
    # marketing_data_clean.csv for the built-in demo dataset.
    path = data_dir / "marketing_data_clean.csv"
    if not path.exists():
        path = data_dir / "standardized_data.csv"
    df = _safe_read_csv(path)
    date_col = session_paths.get_date_column() or "Date"
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass
    return df


@st.cache_data
def load_kpis() -> dict:
    return _safe_read_json(session_paths.get_processed_dir() / "marketing_kpis.json")


@st.cache_data
def load_generic_kpis() -> list:
    return _safe_read_json(session_paths.get_processed_dir() / "generic_kpis.json") or []


@st.cache_data
def load_capabilities() -> dict:
    return _safe_read_json(session_paths.get_processed_dir() / "capabilities.json")


def has_capability(key: str) -> bool:
    """Check if the active dataset supports a given capability.

    Phase 1 fix: For uploaded (non-demo) datasets, a missing capabilities.json
    means the pipeline hasn't run, so capabilities are UNKNOWN = unavailable.
    Only the demo dataset assumes 'available' when capabilities.json is missing
    (backward compatibility with the pre-pipeline demo data).
    """
    caps = load_capabilities()
    if not caps:
        # No capabilities.json found
        if session_paths.is_user_dataset_mode():
            # Uploaded dataset: pipeline didn't produce capabilities -> unavailable
            return False
        else:
            # Demo dataset: backward compat, assume available
            return True
    entry = caps.get(key)
    if entry is None:
        if session_paths.is_user_dataset_mode():
            return False
        return True
    return bool(entry.get("available", True))


def capability_reason(key: str) -> str:
    caps = load_capabilities()
    entry = caps.get(key) if caps else None
    if entry:
        return entry.get("reason", "")
    if session_paths.is_user_dataset_mode():
        return "Capability information not available for this dataset."
    return ""


@st.cache_data
def load_channel_performance() -> pd.DataFrame:
    return _safe_read_csv(session_paths.get_processed_dir() / "channel_performance.csv")


@st.cache_data
def load_channel_attribution() -> pd.DataFrame:
    data_dir = session_paths.get_processed_dir()
    df = _safe_read_csv(data_dir / "channel_attribution.csv")
    if df.empty:
        df = _safe_read_csv(session_paths.get_models_dir() / "channel_attribution.csv")
    return df


@st.cache_data
def load_model_metrics() -> dict:
    return _safe_read_json(session_paths.get_models_dir() / "model_metrics.json")


@st.cache_data
def load_model_diagnostics() -> dict:
    return _safe_read_json(session_paths.get_models_dir() / "model_diagnostics.json")


@st.cache_data
def load_actual_vs_predicted() -> pd.DataFrame:
    # Phase 1: use dynamic date column instead of hardcoded ["Date"]
    date_col = session_paths.get_date_column()
    parse_dates = [date_col] if date_col else None
    return _safe_read_csv(session_paths.get_processed_dir() / "actual_vs_predicted.csv", parse_dates=parse_dates)


@st.cache_data
def load_budget_optimization() -> dict:
    return _safe_read_json(session_paths.get_processed_dir() / "budget_optimization_summary.json")


@st.cache_data
def load_budget_table() -> pd.DataFrame:
    return _safe_read_csv(session_paths.get_processed_dir() / "budget_optimization_result.csv")


@st.cache_data
def load_scenario_comparison() -> pd.DataFrame:
    return _safe_read_csv(session_paths.get_processed_dir() / "scenario_comparison.csv")


@st.cache_data
def load_forecast() -> pd.DataFrame:
    # Phase 1: use dynamic date column instead of hardcoded ["Date"]
    date_col = session_paths.get_date_column()
    parse_dates = [date_col] if date_col else None
    data_dir = session_paths.get_processed_dir()
    df = _safe_read_csv(data_dir / "revenue_forecast.csv", parse_dates=parse_dates)
    if df.empty:
        df = _safe_read_csv(session_paths.get_models_dir() / "revenue_forecast.csv")
    return df


@st.cache_data
def load_monthly_performance() -> pd.DataFrame:
    return _safe_read_csv(session_paths.get_processed_dir() / "monthly_performance.csv")


@st.cache_data
def load_quarterly_performance() -> pd.DataFrame:
    return _safe_read_csv(session_paths.get_processed_dir() / "quarterly_performance.csv")


@st.cache_data
def load_canonical_metadata() -> dict:
    """Load the Phase 2 canonical metadata for the active dataset."""
    return _safe_read_json(session_paths.get_processed_dir() / "canonical_metadata.json")


def load_analysis_context():
    """Reconstruct a Phase 2 AnalysisContext from persisted canonical metadata + dataframe.

    Returns None if canonical metadata is not available (e.g. pre-Phase 2 dataset).
    """
    meta = load_canonical_metadata()
    if not meta:
        return None

    try:
        from src.data_engine.canonical import CanonicalDataset, AnalysisContext

        # Load the dataframe
        df = load_marketing_data()
        if df is None or df.empty:
            return None

        canonical = CanonicalDataset.from_metadata_dict(meta, df=df)
        return AnalysisContext.from_canonical(canonical)
    except Exception:
        return None


@st.cache_data
def load_dynamic_analytics() -> dict:
    """Run Phase 3 dynamic analytics engines and return the AnalyticsResult as a dict.

    Cached per dataset (st.cache_data is cleared on dataset switch).
    Returns an empty dict if canonical metadata is unavailable.
    """
    meta = load_canonical_metadata()
    if not meta:
        return {}

    try:
        from src.data_engine.canonical import CanonicalDataset, AnalysisContext
        from src.analytics.contracts import AnalyticsResult
        from src.analytics.kpi_engine import discover_kpis
        from src.analytics.visualization_engine import recommend_charts
        from src.analytics.insight_engine import discover_insights
        from src.analytics.relationship_engine import analyze_relationships

        df = load_marketing_data()
        if df is None or df.empty:
            return {}

        canonical = CanonicalDataset.from_metadata_dict(meta, df=df)
        context = AnalysisContext.from_canonical(canonical)

        result = AnalyticsResult(
            dataset_id=canonical.dataset_id,
            kpis=discover_kpis(context),
            charts=recommend_charts(context),
            insights=discover_insights(context),
            relationships=analyze_relationships(context),
        )
        return result.to_dict()
    except Exception:
        return {}


@st.cache_data
def load_prediction_report() -> dict:
    """Load the Phase 4 prediction report for the active dataset.

    Returns an empty dict if the prediction report is unavailable.
    """
    return _safe_read_json(session_paths.get_processed_dir() / "prediction_report.json")


@st.cache_data
def load_phase5_analysis() -> dict:
    """Load the Phase 5 anomaly + driver analysis for the active dataset.

    Returns an empty dict if the Phase 5 analysis is unavailable.
    """
    return _safe_read_json(session_paths.get_processed_dir() / "phase5_analysis.json")


def pipeline_ready() -> bool:
    """Whether the active dataset has a standardized/cleaned data file -
    the only thing every page needs at minimum. Individual pages handle
    their own missing MMM-specific files (channel_attribution.csv,
    model_metrics.json) gracefully instead of gating the whole app on them."""
    data_dir = session_paths.get_processed_dir()
    candidates = [data_dir / "marketing_data_clean.csv", data_dir / "standardized_data.csv"]
    return any(p.exists() for p in candidates)
