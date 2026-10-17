<div align="center">

# 📊 MarketPulse AI

### AI-Powered Marketing Mix Modeling & Budget Optimization

**Transform Marketing Data into Smarter Business Decisions.**

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Machine Learning">
</p>

<p>
  <a href="https://marketpulse-ai-f8wwdykbtkqk2zrfktukg.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Try_MarketPulse_AI-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/AshishMandalDSAI/MarketPulse_AI_FINAL_RELEASE">
    <img src="https://img.shields.io/badge/GitHub-View_Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

</div>

---

## 📌 Overview

**MarketPulse AI** is an end-to-end Marketing Mix Modeling (MMM) and
marketing budget optimization platform developed as an MBA Data Science
& AI final-year project.

It helps users analyze marketing performance, understand relationships
between marketing expenditure and revenue, forecast future performance,
and explore data-driven budget allocation strategies.

The platform combines data analytics, machine learning, forecasting,
and interactive dashboards in one application.

> **Note:** The built-in demonstration dataset is synthetic and intended
> for educational and demonstration purposes. It does not represent
> actual company performance.

---

## 🌐 Live Application

### Try MarketPulse AI Online

🚀 **[Launch MarketPulse AI](https://marketpulse-ai-f8wwdykbtkqk2zrfktukg.streamlit.app/)**

Explore the interactive dashboard and experience the application's
available analytics features.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Executive Dashboard | View key performance indicators and dataset summaries. |
| 📂 Flexible Data Upload | Upload supported CSV and Excel datasets. |
| 🔍 Data Profiling | Examine data types, missing values, and dataset structure. |
| 📈 Marketing Analytics | Explore marketing performance and available trends. |
| 🧠 Marketing Mix Modeling | Estimate statistical relationships between marketing channels and revenue. |
| 💰 ROI & ROAS Analysis | Evaluate marketing efficiency when the required data is available. |
| 🎯 Budget Optimization | Explore budget allocation recommendations when a trained model is available. |
| 🔮 Revenue Forecasting | Forecast future values when the dataset supports time-series analysis. |
| 🧪 What-If Simulation | Explore alternative marketing budget scenarios when supported. |
| 🤖 AI Marketing Analyst | Generate analytical insights using the available AI or demo mode. |
| 💡 Recommendations | Review data-driven findings and potential business actions. |
| 📄 Business Reports | Generate reports based on available analytical results. |

---

## 📂 Flexible Dataset Support

MarketPulse AI follows a dataset-first approach.

It can profile uploaded CSV and supported Excel files without requiring
every dataset to use the same column names.

### How It Works

1. **Upload Data** — Import a supported CSV or Excel file.
2. **Profile the Dataset** — Inspect rows, columns, data types, and missing values.
3. **Identify Column Roles** — Suggest possible dates, revenue measures, and marketing channels.
4. **Confirm Column Mapping** — Review and confirm ambiguous column assignments.
5. **Detect Available Capabilities** — Determine which analyses the dataset supports.
6. **Explore Insights** — Use the dashboard, KPIs, reports, and available models.

### Important

Not every dataset supports every analytical feature.

For example:

- Trend analysis requires a usable date column.
- Forecasting requires suitable date and numeric target data.
- Marketing Mix Modeling requires appropriate date, revenue, and marketing-channel data.
- Budget optimization requires a trained marketing mix model.

When required data is unavailable, the corresponding feature may be unavailable.

---

## 🧠 Machine Learning & Analytics

The project incorporates the following analytical capabilities:

- Marketing Mix Modeling
- Marketing Channel Contribution Analysis
- Revenue Forecasting
- Marketing ROI and ROAS Analysis
- Budget Optimization
- What-If Scenario Simulation
- Anomaly Detection and Driver Analysis
- Marketing KPI Analysis
- AI-Assisted Business Insights
- Decision Support and Business Reporting

### Model Interpretation

The built-in demonstration workflow uses an adstock and saturation-based
Marketing Mix Modeling pipeline.

For supported uploaded datasets, the platform can use a Ridge regression
approach with chronological training and holdout evaluation.

**Important:** Regression coefficients indicate statistical associations.
They do not, by themselves, establish that marketing expenditure caused
a change in revenue.

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Ridge Regression
- Marketing Mix Modeling

### Dashboard
- Streamlit

### API
- FastAPI
- Uvicorn

### Data Formats
- CSV
- Excel

### Deployment & Development
- Git
- GitHub
- Docker
- Docker Compose

---

## 🏗️ System Architecture

```text
              ┌────────────────────────┐
              │      Marketing Data    │
              │      CSV / Excel       │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Data Upload &       │
              │       Profiling        │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Column Classification │
              │  & Standardization      │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Capability Detection   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Analytics & ML Models  │
              │ MMM / Forecasting      │
              │ ROI / Budget Analysis  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Insights & Reports    │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Streamlit Dashboard    │
              │      FastAPI           │
              └────────────────────────┘
