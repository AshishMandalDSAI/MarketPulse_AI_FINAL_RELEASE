# MarketPulse AI — Architecture

## Dataset-First Layer (`src/data_engine/`)

Before the legacy pipeline described below ever runs, every dataset —
uploaded or the built-in demo — can flow through the dataset-first
("Universal Data Engine") layer, which has no dependency on any fixed
column schema:

```
   raw upload (CSV/XLSX)
          |
          v
   profiler.py            -> profile_dataset(): per-column inferred type,
                              missing %, cardinality, basic stats
          |
          v
   column_classifier.py   -> classify_columns(): alias-exact / alias-fuzzy /
                              statistical-fallback role guesses + confidence
          |
          v
   pipeline.py             -> profile_and_map() / confirm_and_standardize():
                               applies user confirmations/overrides,
                               enforces that MEDIUM/LOW guesses are never
                               auto-applied
          |
          v
   data_standardizer.py    -> standardize(): builds a StandardizedDataset
                               (date/revenue/channels + generic control /
                               dimension / identifier columns) — nothing
                               with no confirmed role is dropped
          |
          v
   capability_engine.py    -> detect_capabilities(): per-dataset, per-
                               analysis available/unavailable + reason
          |
          v
   analysis_runner.py      -> run_full_analysis(): orchestrates the above
                               for one dataset_id, persists standardized
                               data + capabilities.json + generic_kpis.json
                               + universal_kpis.json, and conditionally
                               runs a lightweight generic MMM
                               (Ridge regression), a lightweight generic
                               forecast (Ridge on a linear time index),
                               and ROI/ROAS — each stage only if its
                               capability is available, with every skip
                               recorded with an explicit reason
```

`src/kpi/generic_kpis.py` sits alongside this layer: `compute_generic_kpis()`
derives KPIs (record count, date range, per-measure averages/sums, most
common categorical values, missing-data rate) purely from whatever
columns a `StandardizedDataset` actually has — no hardcoded column names.

### How it plugs into the existing packages

The dataset-first layer does not replace `src/analytics/`, `src/models/`,
`src/forecasting/`, or `src/optimization/` — those packages remain the
full-fidelity implementation used for the built-in NovaMart demo dataset
(adstock, Hill saturation, leave-one-channel-out attribution, SLSQP
budget optimization). For an arbitrary uploaded dataset that satisfies
`StandardizedDataset.mmm_readiness()` (date + revenue + ≥2 channel
columns), `analysis_runner.py` instead runs its own **simplified**
generic MMM/forecast directly (not by calling into `src/models/` or
`src/forecasting/`, whose code assumes the legacy 8-channel schema) —
see `docs/model_documentation.md` for exactly how the two differ.
`api/main.py` is the seam between the two worlds: legacy endpoints
(`/optimize-budget`, `/simulate-budget`, `/attribution`, etc.) continue to
read the legacy pipeline's `data/processed/`/`models/` outputs, while the
new endpoints (`/upload`, `/datasets/{id}/confirm`) and the `dataset_id`
query parameter on `/kpis` are backed by the dataset-first layer.

### Dataset isolation

Every uploaded dataset gets its own directory tree, created by
`config.settings.ensure_dataset_dirs()`:

```
data/uploads/<dataset_id>/
  raw/         # the originally uploaded file, unmodified
  processed/   # standardized_data.csv, generic_kpis.json, universal_kpis.json,
               # capabilities.json, and (if applicable) revenue_forecast.csv,
               # channel_roi.csv
  models/      # (if applicable) model_metrics.json, channel_attribution.csv
```

This is structurally separate from the demo dataset's `data/processed/`
and `models/` directories (no `uploads/` in their path), and separate
from every other `dataset_id`'s directory. `dataset_id`s are generated as
random hex strings (or user-chosen upload IDs) and validated against
`^[a-zA-Z0-9_-]{1,64}$` before being used to build any filesystem path,
which rejects path-traversal attempts outright. The Streamlit session
layer (`dashboard/components/session_paths.py`) mirrors this isolation in
memory: switching the active dataset clears `st.cache_data` and all
page-level cached `session_state` keys so no stale results from one
dataset can leak into a view of another.

## System Overview

*The diagram below is the legacy pipeline that produces the built-in
NovaMart demo dataset's outputs. An uploaded dataset instead flows through
the dataset-first layer described above.*

```
                    MARKETPULSE AI
                          |
                          v
   generate_dataset.py -> RAW DATA (data/raw/)
                          |
                          v
   data_cleaning.py -> VALIDATED + CLEANED (stage 1)
                          |
                          v
   feature_engineering.py -> ANALYSIS-READY DATASET (data/processed/marketing_data_clean.csv)
                          |
             +------------+------------+
             v                         v
   marketing_analytics.py     src/models/*
   (KPIs, channel perf,       (adstock, saturation,
    monthly/quarterly,         marketing_mix_model,
    holiday/promo)             model_evaluation,
             |                 channel_contribution)
             |                         |
             +------------+------------+
                          v
              src/optimization/*
     (budget_optimizer.py — scipy SLSQP,
      simulator.py — what-if scenarios)
                          |
             +------------+------------+
             v                         v
   src/forecasting/forecast.py   src/ai/analyst.py
   (trend + seasonal Ridge)      (rule-based / optional LLM)
             |                         |
             +------------+------------+
                          v
              +-----------+-----------+
              v                       v
        app.py (Streamlit)      api/main.py (FastAPI)
        dashboard/pages/*             |
                          v           v
                 src/utils/database.py (SQLite,
                 marketpulse.db — read-model store)
```

## Layer Responsibilities

- **Data layer** (`src/data/`): synthetic dataset generation, validation, and cleaning.
- **Feature layer** (`src/features/`): time features, spend aggregates/shares, growth rates.
- **Analytics layer** (`src/analytics/`): descriptive/observational KPIs and breakdowns — no modeling.
- **Model layer** (`src/models/`): the Marketing Mix Model itself — adstock, saturation, regression,
  evaluation, and attribution. This is the only layer that produces causal-flavored (model-estimated)
  claims, and it consistently labels them as such.
- **Optimization layer** (`src/optimization/`): budget optimization (scipy.optimize) and what-if simulation,
  both built directly on top of the fitted model from the model layer.
- **Forecasting layer** (`src/forecasting/`): a separate, deliberately simple time-series forecast — not
  part of the MMM causal story, just a near-term revenue projection.
- **AI layer** (`src/ai/`): a rule-based analyst grounded in the above layers' outputs, with an optional
  LLM narrative layer that never touches the underlying numbers.
- **Presentation layer**: `app.py` + `dashboard/` (Streamlit) and `api/main.py` (FastAPI) are two
  independent consumers of the same underlying pipeline outputs and fitted model — they do not duplicate
  business logic.
- **Persistence**: `data/processed/marketpulse.db` (SQLite) is a convenience read-model built from the
  same CSV/JSON pipeline outputs, for anyone who wants to query the project's results with SQL.

## Why this layering

Each layer only depends on layers "above" it in the diagram and writes its outputs to `data/processed/` or
`models/` as CSV/JSON/pickle files. This means:

- Every layer can be re-run independently (`python src/models/marketing_mix_model.py`, etc.) without
  needing the dashboard or API running.
- The dashboard and API never compute business logic themselves — they only read pipeline outputs or call
  the same `src/` functions the pipeline scripts use, so the two presentation layers can never disagree
  with each other or with the CLI pipeline output.
