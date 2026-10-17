<div align="center">

# 📊 MarketPulse AI

### AI-Driven Marketing Mix Modeling & Budget Optimization Platform

**Turn Marketing Spend into Smarter Business Decisions.**

<br>

<a href="https://marketpulse-ai-f8wwdykbtkqk2zrfktukg.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-MarketPulse%20AI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</a>

<a href="https://github.com/AshishMandalDSAI/MarketPulse-AI">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github">
</a>

</div>

---

## 🚀 Live Demo

### Try MarketPulse AI Online

👉 **[🌐 OPEN LIVE APPLICATION](https://marketpulse-ai-f8wwdykbtkqk2zrfktukg.streamlit.app/)**

The complete Streamlit dashboard is deployed and available online.

---

## 🎯 Project Overview

**MarketPulse AI** is an end-to-end Marketing Mix Modeling (MMM) and
budget-optimization platform developed as an MBA Data Science & AI
final-year project.

The platform analyzes marketing channels, measures their contribution
to revenue, models diminishing returns, forecasts revenue, and recommends
how to allocate a fixed marketing budget to maximize predicted revenue.

> **Important:** The built-in demo dataset is entirely synthetic and
> generated for academic/project demonstration purposes. It does not
> represent a real company or actual business performance.

---

## 🗂️ Dataset-First: Bring Your Own Data

MarketPulse AI is **schema-agnostic**. There is no fixed, required set of
CSV/XLSX columns — you can upload any reasonably-structured dataset (CSV,
or Excel `.xlsx`/`.xls` via `openpyxl`) and the platform adapts to whatever
columns it actually contains. The built-in NovaMart demo dataset (see
below) is one supported path among several now, not the only one — a
dataset with **zero** marketing/revenue/date columns still works.

**How it works, end to end:**

1. **Upload** (`POST /upload` or the "Upload Data" dashboard page) — the
   file is parsed and profiled: row/column counts, per-column inferred
   type (numeric / categorical / datetime / text / identifier), missing
   data %, and cardinality. No hardcoded column names are involved.
2. **Semantic column classification** — every column is matched against
   generic alias vocabularies (e.g. "revenue", "date", "google ads spend")
   in three tiers: exact alias match, fuzzy/substring alias match, and a
   statistical/dtype fallback (e.g. "this numeric column looks like a
   currency measure," "this text column parses as a date"). Each guess
   gets a confidence score and level:
   - **HIGH** (≥ 0.80) — auto-selected, no user action needed.
   - **MEDIUM** (0.55–0.79) and **LOW** (< 0.55) — shown to the user with
     the guessed column and confidence, but **never silently applied**.
     The user must explicitly confirm or override them (or pick a
     different column, or decline the role entirely) before they're used.
3. **Confirm & standardize** (`POST /datasets/{dataset_id}/confirm`) —
   once the user confirms/overrides ambiguous fields, the dataset is
   standardized into semantic roles (date, revenue, marketing channels)
   plus generic numeric "control" measures, categorical "dimension"
   columns, and near-unique "identifier" columns for everything else.
   Nothing is dropped — a column that isn't mapped to a specific role is
   still kept and usable as a generic measure or dimension.
4. **Capability detection** — a Capability Engine inspects the
   standardized dataset and determines, per dataset, which analyses are
   actually usable, each with a plain-English reason:
   - **Always available:** data profiling, generic KPIs, AI analyst,
     business report.
   - **Conditionally available**, based on what the dataset actually
     contains: category/distribution analysis (needs ≥1 categorical
     column), trend analysis (needs a date column), forecasting (needs a
     date column + a numeric target + enough time points), Marketing Mix
     Modeling / channel attribution / model diagnostics (needs date +
     revenue + ≥2 spend/channel columns), ROI/ROAS (needs revenue + ≥1
     spend column), budget optimization and what-if simulation (both
     require a trained MMM).
5. **Dynamic KPIs, charts, and dashboard** — Generic KPIs (total records,
   date range, averages/sums of whatever numeric measures exist, most
   common categorical values, missing-data rate) are computed purely from
   the columns present — never fabricated, never hardcoded to a specific
   schema. The dashboard and `/kpis` endpoint surface marketing-specific
   KPIs only when the Capability Engine says they're available, and mark
   them "unavailable" with a reason otherwise.
6. **Complete isolation** — every uploaded dataset gets its own directory
   under `data/uploads/<dataset_id>/{raw,processed,models}/`, entirely
   separate from other uploads and from the built-in demo dataset's
   `data/processed/`/`models/`. Nothing from one dataset leaks into
   another, and the Streamlit session clears all cached page state on
   every dataset switch.

MMM, ROI, forecasting, and budget optimization are **genuinely optional**
— they simply won't appear (and will say why) for a dataset that lacks
the columns they need; profiling, generic KPIs, the AI analyst, and
report generation always work regardless.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Executive Overview | High-level marketing and revenue KPIs |
| 📈 Marketing Performance | Analyze marketing channels and trends |
| 🧠 Marketing Mix Model | Estimate channel contribution and ROI |
| 💰 Channel ROI | Compare marketing efficiency |
| 🎯 Budget Optimizer | Optimize marketing budget allocation |
| 🔮 What-If Simulator | Test different marketing scenarios |
| 📅 Revenue Forecast | Forecast future revenue |
| 🤖 AI Marketing Analyst | Generate analytical insights |
| 💡 Recommendations | Data-driven marketing recommendations |
| 📄 Reports | Generate and explore project reports |

---

## 🧠 Machine Learning & Analytics

The project includes:

- Marketing Mix Modeling (full adstock/saturation for demo; Ridge-based for uploads)
- Channel Contribution Analysis with signed coefficients
- Revenue Forecasting with frequency-aware date projection
- Budget Optimization
- Scenario Simulation
- Marketing KPI Analysis
- AI-assisted Business Insights
- Anomaly Detection & Driver Analysis
- Decision Intelligence

The **full** adstock/saturation MMM is trained for the built-in NovaMart
demo dataset via the legacy `src/models/` pipeline. An arbitrary uploaded
dataset that satisfies the Capability Engine's MMM readiness check (date +
revenue + ≥2 spend/channel columns) instead gets a **generic** MMM — a
Ridge regression with chronological 80/20 train/test split, signed
coefficients (preserving negative associations), and separate train vs.
holdout metrics. The generic forecast infers data frequency
(daily/weekly/monthly/quarterly/yearly) from the date column and uses
calendar-aware date projection. All model outputs include explicit
disclaimers that Ridge coefficients measure statistical association, not
causal marketing impact. See `docs/model_documentation.md` and
`docs/optimization_methodology.md` for details.

---

## 🛠️ Technology Stack

### Programming

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Data Science

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Dashboard & API

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

### Deployment

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

### Version Control

![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)

---

## 🏗️ System Architecture

*The diagram below shows the legacy pipeline used for the built-in demo
dataset. For an uploaded dataset, requests instead flow through the
dataset-first pipeline (upload → profile → classify → standardize →
capability detection → conditional analytics) described above and in
`docs/architecture.md`.*

```text
                    ┌─────────────────────┐
                    │   Marketing Data    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning &     │
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Marketing Mix Model │
                    │      (MMM)          │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
          Channel ROI    Revenue Forecast   Attribution
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Budget Optimization │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.11 or later
- pip (Python package manager)
- Docker & Docker Compose (optional, for containerized deployment)

### Local Setup (Windows PowerShell)

```powershell
# Clone the repository
git clone https://github.com/AshishMandalDSAI/MarketPulse-AI.git
cd MarketPulse-AI

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional — the app works without .env)
Copy-Item .env.example .env
# Edit .env to set ANTHROPIC_API_KEY for AI analyst, API_KEY for auth, etc.

# Run the initialization pipeline (generates demo data & models)
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

# Start the Streamlit dashboard
streamlit run app.py

# Start the FastAPI backend (in a separate terminal)
uvicorn api.main:app --reload
```

### Local Setup (macOS / Linux)

```bash
git clone https://github.com/AshishMandalDSAI/MarketPulse-AI.git
cd MarketPulse-AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional
streamlit run app.py
```

### Docker Deployment

```powershell
# Copy environment template (optional — Docker works without .env)
Copy-Item .env.example .env

# Validate configuration
docker compose config

# Build and start all services
docker compose up --build
```

- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

Docker handles initialization automatically via `docker-entrypoint.sh`:
- Uses `flock` for race-condition-safe initialization across containers
- The API service waits for the dashboard to be healthy before starting
- Demo data and models are generated on first run only
- Existing data in `./data` and `./models` is preserved across restarts

---

## 🔒 Environment Configuration

All environment variables are optional. The application works fully offline with sensible defaults.

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Enables LLM-powered AI Marketing Analyst | Demo AI Mode (offline) |
| `API_KEY` | Secures write/compute API endpoints with X-API-Key header | All endpoints open |
| `CORS_ORIGINS` | Comma-separated allowed origins for API CORS | Localhost dev origins |
| `DATABASE_PATH` | SQLite database path | `data/processed/marketpulse.db` |
| `API_HOST` | FastAPI server host | `127.0.0.1` |
| `API_PORT` | FastAPI server port | `8000` |
| `STREAMLIT_PORT` | Streamlit server port | `8501` |

**API Authentication:** When `API_KEY` is set, write and compute endpoints (`/upload`, `/optimize-budget`, `/simulate-budget`, `/ask`, `/datasets/*/confirm`, `/datasets/*/decisions/run`) require an `X-API-Key` header. Read-only endpoints (`/health`, `/channels`, `/attribution`, `/model-performance`, `/kpis`) remain public.

---

## 🧪 Running Tests

```bash
# Compile all Python files (syntax check)
python -m compileall .

# Run the full test suite
python -m pytest -q

# Run specific test modules
python -m pytest tests/test_docker_entrypoint.py -v
python -m pytest tests/test_api.py -v
```

---

## 📄 License

This project was developed as an MBA Data Science & AI final-year project.
