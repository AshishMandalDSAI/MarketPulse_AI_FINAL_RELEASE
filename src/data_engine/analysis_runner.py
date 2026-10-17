"""Orchestrates the full dataset-first pipeline for one dataset_id:

  load raw -> profile+map -> standardize -> detect capabilities
  -> persist standardized data + capabilities.json + generic_kpis.json
     + universal_kpis.json to data/uploads/<dataset_id>/{processed,models}/
  -> conditionally run a lightweight generic MMM stage (if capability available)
  -> conditionally run a lightweight generic forecast stage (if capability available)
  -> conditionally run ROI stage (if capability available)

Every skipped stage is recorded with an explicit reason — never silently
skipped, never fabricated.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import ensure_dataset_dirs, dataset_processed_dir, dataset_models_dir, dataset_raw_path
from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
from src.data_engine.capability_engine import detect_capabilities
from src.data_engine.canonical import CanonicalDataset, AnalysisContext, UserOverrides
from src.kpi.generic_kpis import compute_generic_kpis

DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def normalize_duplicate_columns(df: pd.DataFrame) -> tuple:
    """Detect and rename duplicate column names deterministically.

    Returns (df_with_unique_columns, rename_map_or_None).
    - If no duplicates, returns (df, None).
    - If duplicates exist, renames them as ``Name``, ``Name_1``, ``Name_2``, …,
      avoiding collisions with existing column names.
    - Column order and all data values are preserved.
    """
    cols = list(df.columns)
    if len(cols) == len(set(cols)):
        return df, None

    existing_names = set(cols)
    seen: dict[str, int] = {}
    new_cols = []
    rename_map = {}

    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            candidate = f"{col}_{seen[col]}"
            # Avoid collisions with any existing or already-assigned name
            while candidate in existing_names or candidate in set(new_cols):
                seen[col] += 1
                candidate = f"{col}_{seen[col]}"
            new_cols.append(candidate)
            existing_names.add(candidate)
            rename_map[candidate] = col  # new_name -> original_name

    df = df.copy()
    df.columns = new_cols
    return df, rename_map


def validate_dataset_id(dataset_id: str) -> str:
    if not dataset_id or not DATASET_ID_PATTERN.match(dataset_id):
        raise ValueError(
            f"Invalid dataset_id {dataset_id!r} — must match {DATASET_ID_PATTERN.pattern}."
        )
    return dataset_id


def _run_generic_mmm(dataset, models_dir: Path, processed_dir: Path, report: list) -> None:
    """A lightweight, genuinely-computed (not fabricated) linear
    channel-contribution model: Ridge regression of revenue on channel
    spend + numeric controls, with proper train/test chronological split.

    **Attribution methodology**: each channel's estimated contribution is
    ``coefficient × total_channel_spend``, preserving the sign of the
    coefficient.  A negative coefficient means the model estimates a
    negative association between that channel's spend and revenue — this
    is reported honestly rather than converted to a positive contribution.

    **Important caveat written into the output**: linear regression
    coefficients measure statistical *association*, not proof of causal
    marketing impact.  Confounders, multicollinearity and reverse
    causality can distort the estimates.

    Outputs are written in the SAME schema the dashboard pages expect
    (channel_attribution.csv with Rank, Channel, Spend, Estimated_Contribution,
    Contribution_Pct, Coefficient, Coefficient_Direction, Spend_Revenue_Ratio,
    Model_Estimated_Efficiency; model_metrics.json with selected_model +
    model_comparison structure containing both train and test metrics)
    so pages 3, 4, and 9 can consume them without adapters."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

        df = dataset.df.copy()
        channel_cols = list(dataset.channels)
        control_cols = [c for c in dataset.control_cols if c in df.columns]
        X_cols = channel_cols + control_cols
        X = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = pd.to_numeric(df[dataset.revenue_col], errors="coerce").fillna(0.0)

        n = len(y)
        if n < 4:
            report.append(f"Generic MMM skipped: only {n} observation(s) — need at least 4.")
            return

        # --- Chronological train/test split ---
        # If a date column exists, sort chronologically first
        if dataset.has_date and dataset.date_col in df.columns:
            sort_idx = pd.to_datetime(df[dataset.date_col], errors="coerce").argsort()
            X = X.iloc[sort_idx].reset_index(drop=True)
            y = y.iloc[sort_idx].reset_index(drop=True)

        split_point = max(2, int(n * 0.8))  # 80/20 split, at least 2 in train
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        has_holdout = len(X_test) > 0

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Training metrics
        train_preds = model.predict(X_train)
        train_r2 = float(r2_score(y_train, train_preds)) if len(y_train) > 1 else 0.0
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds))) if len(y_train) > 1 else 0.0
        train_mae = float(mean_absolute_error(y_train, train_preds)) if len(y_train) > 1 else 0.0

        # Holdout metrics (if we have test data)
        if has_holdout:
            test_preds = model.predict(X_test)
            test_r2 = float(r2_score(y_test, test_preds)) if len(y_test) > 1 else 0.0
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds))) if len(y_test) > 1 else 0.0
            test_mae = float(mean_absolute_error(y_test, test_preds)) if len(y_test) > 1 else 0.0
            try:
                y_nonzero = y_test.replace(0, np.nan).dropna()
                test_mape = float(mean_absolute_percentage_error(
                    y_nonzero, pd.Series(test_preds, index=y_test.index).loc[y_nonzero.index]
                )) * 100
            except Exception:
                test_mape = None
        else:
            test_r2 = test_rmse = test_mae = 0.0
            test_mape = None

        # --- Build channel_attribution.csv in the dashboard-expected schema ---
        # Use full-data coefficients for attribution (fitted on train only)
        coefs = {col: float(coef) for col, coef in zip(X_cols, model.coef_)}
        total_revenue = float(y.sum()) or 1.0

        attribution_rows = []
        for ch in channel_cols:
            spend_total = float(df[ch].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum())
            coef = coefs[ch]

            # Estimated contribution: coefficient * total spend (signed — preserves direction)
            estimated_contribution = coef * spend_total

            # Direction label for dashboard consumers
            if coef > 0.001:
                direction = "positive"
            elif coef < -0.001:
                direction = "negative"
            else:
                direction = "neutral"

            # Spend-to-revenue ratio (NOT attributed ROAS — this is total revenue / channel spend)
            spend_revenue_ratio = (total_revenue / spend_total) if spend_total > 0 else 0.0

            # Model efficiency: signed coefficient (revenue-per-unit-spend association)
            efficiency = coef

            attribution_rows.append({
                "Channel": ch,
                "Spend": round(spend_total, 2),
                "Estimated_Contribution": round(estimated_contribution, 2),
                "Coefficient": round(coef, 6),
                "Coefficient_Direction": direction,
                "Spend_Revenue_Ratio": round(spend_revenue_ratio, 2),
                "Model_Estimated_Efficiency": round(efficiency, 4),
            })

        # Contribution_Pct: use absolute values for sizing but preserve sign info
        total_abs_contribution = sum(abs(r["Estimated_Contribution"]) for r in attribution_rows) or 1.0
        for row in attribution_rows:
            # Signed percentage — negative channels get negative share
            row["Contribution_Pct"] = round(
                row["Estimated_Contribution"] / total_abs_contribution * 100, 2
            )

        # Sort by efficiency descending and rank
        attribution_rows.sort(key=lambda r: r["Model_Estimated_Efficiency"], reverse=True)
        for i, row in enumerate(attribution_rows, 1):
            row["Rank"] = i

        attr_df = pd.DataFrame(attribution_rows)
        # Write to BOTH models/ and processed/ so loaders find it
        attr_df.to_csv(models_dir / "channel_attribution.csv", index=False)
        attr_df.to_csv(processed_dir / "channel_attribution.csv", index=False)

        # --- Build model_metrics.json in the dashboard-expected schema ---
        model_name = "generic_ridge_linear"
        test_metrics = {
            "r2": round(test_r2, 4),
            "rmse": round(test_rmse, 2),
            "mae": round(test_mae, 2),
            "mape": round(test_mape, 2) if test_mape is not None else 0.0,
        } if has_holdout else None

        metrics_dict = {
            "selected_model": model_name,
            "model_comparison": {
                model_name: {
                    "train": {
                        "r2": round(train_r2, 4),
                        "rmse": round(train_rmse, 2),
                        "mae": round(train_mae, 2),
                    },
                    **({"test": test_metrics} if test_metrics else {}),
                }
            },
            # Extra metadata for transparency
            "n_observations": int(n),
            "n_train": int(split_point),
            "n_test": int(n - split_point),
            "has_holdout": has_holdout,
            "features": X_cols,
            "model_type": model_name,
            "note": ("Ridge regression coefficients measure statistical association, "
                     "not proof of causal marketing impact. Confounders, "
                     "multicollinearity and reverse causality can distort estimates."),
        }
        with open(models_dir / "model_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)

        holdout_msg = f", holdout R2={test_r2:.3f}" if has_holdout else " (no holdout — insufficient data)"
        report.append(
            f"Generic MMM (Ridge regression) trained on {split_point} rows: "
            f"train R2={train_r2:.3f}{holdout_msg}, {len(X_cols)} feature(s)."
        )
    except Exception as e:
        report.append(f"Marketing mix modeling was available but failed to train: {e}")


def _infer_frequency(dates_series: pd.Series) -> tuple:
    """Infer the time-series frequency from a sorted datetime Series.

    Returns (freq_label, timedelta) where freq_label is one of
    'daily', 'weekly', 'monthly', 'quarterly', 'yearly', or 'irregular',
    and timedelta is the corresponding pd.Timedelta (or median gap for irregular).
    """
    if len(dates_series) < 2:
        return "unknown", pd.Timedelta(weeks=1)

    gaps = dates_series.diff().dropna()
    if gaps.empty:
        return "unknown", pd.Timedelta(weeks=1)

    median_days = gaps.median().days

    # Match to standard calendar periods
    if 0 <= median_days <= 1:
        return "daily", pd.Timedelta(days=1)
    elif 5 <= median_days <= 9:
        return "weekly", pd.Timedelta(weeks=1)
    elif 25 <= median_days <= 35:
        return "monthly", pd.Timedelta(days=30)
    elif 85 <= median_days <= 95:
        return "quarterly", pd.Timedelta(days=91)
    elif 350 <= median_days <= 380:
        return "yearly", pd.Timedelta(days=365)
    else:
        return "irregular", gaps.median()


def _run_generic_forecast(dataset, models_dir: Path, processed_dir: Path, report: list) -> None:
    """Lightweight linear-trend forecast with chronological train/test split.

    Validates date ordering, infers frequency, uses a holdout period for
    evaluation, and produces honestly-labeled approximate prediction intervals.

    Outputs revenue_forecast.csv with the SAME column names the dashboard
    page 7 expects: Date, Forecast_Revenue, Lower_90, Upper_90."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_absolute_error

        target_col = dataset.revenue_col or (dataset.control_cols[0] if dataset.control_cols else None)
        if not target_col:
            report.append("Forecasting skipped: no numeric target column available.")
            return

        df = dataset.df.copy()

        # --- Validate and sort dates ---
        dates_raw = pd.to_datetime(df[dataset.date_col], errors="coerce")
        valid_mask = dates_raw.notna()
        if valid_mask.sum() < 4:
            report.append(f"Forecasting skipped: only {valid_mask.sum()} valid dates — need at least 4.")
            return

        df = df[valid_mask].copy()
        df["_parsed_date"] = dates_raw[valid_mask]

        # Remove duplicate dates (keep first occurrence)
        n_dups = df["_parsed_date"].duplicated().sum()
        if n_dups > 0:
            df = df.drop_duplicates(subset=["_parsed_date"], keep="first")

        # Sort chronologically
        df = df.sort_values("_parsed_date").reset_index(drop=True)

        y = pd.to_numeric(df[target_col], errors="coerce").ffill().fillna(0.0).values
        n = len(y)

        if n < 4:
            report.append(f"Forecasting skipped: only {n} observations after cleaning — need at least 4.")
            return

        # --- Infer frequency ---
        dates_sorted = df["_parsed_date"].reset_index(drop=True)
        freq_label, freq_delta = _infer_frequency(dates_sorted)

        # --- Chronological train/test split for evaluation ---
        split_point = max(3, int(n * 0.8))
        idx_all = np.arange(n).reshape(-1, 1)
        idx_train = idx_all[:split_point]
        y_train = y[:split_point]
        has_holdout = split_point < n

        model = Ridge(alpha=1.0)
        model.fit(idx_train, y_train)

        # Evaluate on holdout if available
        holdout_msg = ""
        if has_holdout:
            idx_test = idx_all[split_point:]
            y_test = y[split_point:]
            test_preds = model.predict(idx_test)
            if len(y_test) > 1:
                holdout_r2 = float(r2_score(y_test, test_preds))
                holdout_mae = float(mean_absolute_error(y_test, test_preds))
                holdout_msg = f", holdout R2={holdout_r2:.3f}, MAE={holdout_mae:.1f}"
            else:
                holdout_msg = ", holdout=1 obs (too few for R2)"

        # Refit on all data for the final forecast
        model_full = Ridge(alpha=1.0)
        model_full.fit(idx_all, y)

        # Residual standard deviation for prediction interval
        resid = y - model_full.predict(idx_all)
        resid_std = float(np.std(resid)) if n > 1 else 0.0

        # Forecast horizon
        horizon = min(12, max(1, n // 4))
        future_idx = np.arange(n, n + horizon).reshape(-1, 1)
        forecast_vals = model_full.predict(future_idx)

        # --- Project dates forward using inferred frequency ---
        last_date = dates_sorted.iloc[-1]
        if freq_label == "monthly":
            # Use pd.DateOffset for calendar months
            future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(horizon)]
        elif freq_label == "quarterly":
            future_dates = [last_date + pd.DateOffset(months=3 * (i + 1)) for i in range(horizon)]
        elif freq_label == "yearly":
            future_dates = [last_date + pd.DateOffset(years=i + 1) for i in range(horizon)]
        else:
            # daily, weekly, irregular — use the inferred timedelta
            future_dates = [last_date + freq_delta * (i + 1) for i in range(horizon)]

        out = pd.DataFrame({
            "Date": future_dates,
            "Forecast_Revenue": forecast_vals,
            "Lower_90": forecast_vals - 1.645 * resid_std,
            "Upper_90": forecast_vals + 1.645 * resid_std,
        })
        out.to_csv(processed_dir / "revenue_forecast.csv", index=False)

        interval_note = ("Prediction intervals are approximate (±1.645 × residual std dev) "
                         "and assume residuals are roughly normally distributed.")
        dup_note = f" ({n_dups} duplicate date(s) removed)" if n_dups > 0 else ""
        report.append(
            f"Generic forecast (linear trend, {freq_label} frequency) for '{target_col}': "
            f"{horizon} step(s) ahead from {n} observation(s){dup_note}{holdout_msg}. "
            f"{interval_note}"
        )
    except Exception as e:
        report.append(f"Forecasting was available but failed to run: {e}")


def run_full_analysis(dataset_id: str, df: pd.DataFrame = None, mapping=None, confirmed_roles=None, overrides=None):
    """Runs the entire dataset-first pipeline for `dataset_id`. If `df` is
    not given, reads the previously-saved raw file for this dataset_id.
    Returns a dict with dataset, capabilities, generic_kpis, report."""
    validate_dataset_id(dataset_id)
    ensure_dataset_dirs(dataset_id)
    processed_dir = dataset_processed_dir(dataset_id)
    models_dir = dataset_models_dir(dataset_id)

    if df is None:
        raw_dir = dataset_raw_path(dataset_id, "").parent
        candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(f"No raw upload found for dataset_id={dataset_id!r}.")
        path = candidates[0]
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)

    # --- Duplicate column guard (M-1) ---
    df, dup_rename_map = normalize_duplicate_columns(df)

    if mapping is None:
        _, mapping = profile_and_map(df)

    dataset, map_report = confirm_and_standardize(df, mapping, confirmed_roles=confirmed_roles, overrides=overrides)
    capabilities = detect_capabilities(dataset)

    report = list(map_report)

    if dup_rename_map:
        report.insert(0, f"Duplicate column names detected and normalized: "
                         f"{', '.join(f'{v!r} -> {k!r}' for k, v in dup_rename_map.items())}.")

    # Persist standardized data
    dataset.df.to_csv(processed_dir / "standardized_data.csv", index=False)

    # Generic KPIs (always computed — never fabricated)
    generic_kpis = compute_generic_kpis(dataset)
    with open(processed_dir / "generic_kpis.json", "w") as f:
        json.dump([k.to_dict() for k in generic_kpis], f, indent=2)
    with open(processed_dir / "universal_kpis.json", "w") as f:
        json.dump({
            "has_date": dataset.has_date, "has_revenue": dataset.has_revenue,
            "n_channels": len(dataset.channels), "n_rows": len(dataset.df),
            "n_controls": len(dataset.control_cols), "n_dimensions": len(dataset.dimension_cols),
            "date_column": dataset.date_col, "revenue_column": dataset.revenue_col,
            "channel_columns": dataset.channels, "dimension_columns": dataset.dimension_cols,
            "control_columns": dataset.control_cols, "identifier_columns": dataset.identifier_cols,
        }, f, indent=2)

    with open(processed_dir / "capabilities.json", "w") as f:
        json.dump(capabilities.to_dict(), f, indent=2)

    # Legacy marketing_analytics stage: only meaningful when the dataset has
    # date + revenue (the minimum the legacy KPI shape needs).
    if dataset.has_date and dataset.has_revenue:
        report.append(f"Date+revenue present — generic KPI/trend analytics are computed for '{dataset.revenue_col}'.")
    else:
        report.append("Legacy-style revenue/date analytics skipped: date and/or revenue column not identified.")

    # Marketing Mix Modeling stage
    if capabilities.is_available("marketing_mix_modeling"):
        _run_generic_mmm(dataset, models_dir, processed_dir, report)
    else:
        report.append(f"Marketing mix modeling stage skipped: {capabilities.get('marketing_mix_modeling').reason}")

    # Forecast stage
    if capabilities.is_available("forecasting"):
        _run_generic_forecast(dataset, models_dir, processed_dir, report)
    else:
        report.append(f"Forecasting stage skipped: {capabilities.get('forecasting').reason}")

    # ROI stage — compute spend-to-revenue ratio (NOT attributed ROAS)
    if capabilities.is_available("roi_roas"):
        try:
            df2 = dataset.df
            revenue_total = pd.to_numeric(df2[dataset.revenue_col], errors="coerce").fillna(0.0).sum()
            rows = []
            for ch in dataset.channels:
                spend = pd.to_numeric(df2[ch], errors="coerce").fillna(0.0).sum()
                # This is total_revenue / channel_spend — a naive ratio, NOT
                # attributed ROAS.  True attributed ROAS requires a causal model
                # isolating each channel's incremental revenue contribution.
                spend_revenue_ratio = (float(revenue_total / spend)) if spend else None
                rows.append({
                    "channel": ch,
                    "total_spend": float(spend),
                    "spend_revenue_ratio": spend_revenue_ratio,
                    "note": "Naive total-revenue-to-channel-spend ratio, not attributed ROAS",
                })
            pd.DataFrame(rows).to_csv(processed_dir / "channel_roi.csv", index=False)
            report.append(f"Spend-revenue ratios computed for {len(rows)} channel(s).")
        except Exception as e:
            report.append(f"ROI/ROAS stage was available but failed: {e}")
    else:
        report.append(f"ROI/ROAS stage skipped: {capabilities.get('roi_roas').reason}")

    if not capabilities.is_available("budget_optimization"):
        report.append(f"Budget optimization skipped: {capabilities.get('budget_optimization').reason}")
    if not capabilities.is_available("what_if_simulation"):
        report.append(f"What-if simulation skipped: {capabilities.get('what_if_simulation').reason}")

    # Phase 2: Build canonical dataset + analysis context (backward-compatible addition)
    canonical = CanonicalDataset.from_standardized(
        dataset_id=dataset_id,
        std_dataset=dataset,
        filename="",
        capability_report=capabilities,
    )
    context = AnalysisContext.from_canonical(canonical)

    # Persist canonical metadata alongside existing artifacts
    with open(processed_dir / "canonical_metadata.json", "w") as f:
        json.dump(canonical.to_metadata_dict(), f, indent=2, default=str)

    # Phase 3: Run dynamic analytics engines and persist results
    analytics_result_dict = None
    try:
        from src.analytics.contracts import AnalyticsResult
        from src.analytics.kpi_engine import discover_kpis
        from src.analytics.visualization_engine import recommend_charts
        from src.analytics.insight_engine import discover_insights
        from src.analytics.relationship_engine import analyze_relationships

        analytics_result = AnalyticsResult(
            dataset_id=dataset_id,
            kpis=discover_kpis(context),
            charts=recommend_charts(context),
            insights=discover_insights(context),
            relationships=analyze_relationships(context),
        )
        analytics_result_dict = analytics_result.to_dict()
        with open(processed_dir / "dynamic_analytics.json", "w") as f:
            json.dump(analytics_result_dict, f, indent=2, default=str)
        report.append(
            f"Phase 3 dynamic analytics: {len(analytics_result.kpis)} KPI(s), "
            f"{len(analytics_result.charts)} chart(s), "
            f"{len(analytics_result.insights)} insight(s), "
            f"{len(analytics_result.relationships)} relationship(s)."
        )
    except Exception as e:
        report.append(f"Phase 3 dynamic analytics failed (non-fatal): {e}")

    # Phase 4: Run predictive analytics engines and persist results
    prediction_report_dict = None
    try:
        from src.prediction.task_detector import detect_predictive_tasks
        from src.prediction.contracts import PredictionReport
        from src.prediction.forecasting_engine import run_forecast
        from src.prediction.regression_engine import run_regression
        from src.prediction.classification_engine import run_classification
        from src.prediction.segmentation_engine import run_segmentation

        tasks = detect_predictive_tasks(context)
        pred_report = PredictionReport(dataset_id=dataset_id, tasks=tasks)

        for task in tasks:
            if not task.requirements_met:
                continue
            try:
                if task.task_type == "forecasting":
                    pred_report.forecast = run_forecast(context, target_column=task.target_column)
                elif task.task_type == "regression":
                    pred_report.regression = run_regression(context, target_column=task.target_column)
                elif task.task_type == "classification":
                    pred_report.classification = run_classification(context, target_column=task.target_column)
                elif task.task_type == "segmentation":
                    pred_report.segmentation = run_segmentation(context)
            except Exception as task_err:
                report.append(f"Phase 4 {task.task_type} failed (non-fatal): {task_err}")

        prediction_report_dict = pred_report.to_dict()
        with open(processed_dir / "prediction_report.json", "w") as f:
            json.dump(prediction_report_dict, f, indent=2, default=str)

        completed = [t for t in ["forecast", "regression", "classification", "segmentation"]
                     if getattr(pred_report, t) is not None]
        report.append(
            f"Phase 4 prediction: {len(tasks)} task(s) detected, "
            f"{len(completed)} completed ({', '.join(completed) if completed else 'none'})."
        )
    except Exception as e:
        report.append(f"Phase 4 prediction failed (non-fatal): {e}")

    # Phase 5: Run anomaly detection + driver intelligence and persist results
    phase5_result_dict = None
    try:
        from src.anomaly.detector import detect_anomalies
        from src.anomaly.driver_engine import analyze_drivers
        from src.anomaly.contracts import Phase5Result

        anomaly_report = detect_anomalies(context, dataset_id)
        driver_report = analyze_drivers(context, dataset_id)

        phase5_result = Phase5Result(
            dataset_id=dataset_id,
            anomaly_report=anomaly_report,
            driver_report=driver_report,
        )
        phase5_result_dict = phase5_result.to_dict()
        with open(processed_dir / "phase5_analysis.json", "w") as f:
            json.dump(phase5_result_dict, f, indent=2, default=str)

        n_anomalies = len(anomaly_report.anomalies) if anomaly_report else 0
        n_drivers = len(driver_report.drivers) if driver_report else 0
        target_info = f"target='{driver_report.target_column}'" if driver_report and driver_report.target_column else "no target"
        report.append(
            f"Phase 5 anomaly+driver analysis: {n_anomalies} anomaly(ies), "
            f"{n_drivers} driver(s) ({target_info})."
        )
    except Exception as e:
        report.append(f"Phase 5 anomaly+driver analysis failed (non-fatal): {e}")

    # Phase 6: Run marketing intelligence pipeline and persist results
    phase6_result_dict = None
    try:
        from src.marketing.orchestrator import run_marketing_analysis

        phase6_result = run_marketing_analysis(
            canonical=canonical,
            context=context,
            dataset_id=dataset_id,
        )
        phase6_result_dict = phase6_result.to_dict()
        with open(processed_dir / "phase6_marketing.json", "w") as f:
            json.dump(phase6_result_dict, f, indent=2, default=str)

        readiness = "NOT_READY"
        n_channels = 0
        if phase6_result.marketing_report:
            readiness = phase6_result.marketing_report.readiness.value
            n_channels = len(phase6_result.marketing_report.channel_fields)
        has_mmm = phase6_result.mmm_result is not None and phase6_result.mmm_result._model is not None
        report.append(
            f"Phase 6 marketing intelligence: readiness={readiness}, "
            f"{n_channels} channel(s), MMM={'fitted' if has_mmm else 'not fitted'}."
        )
    except Exception as e:
        report.append(f"Phase 6 marketing intelligence failed (non-fatal): {e}")

    # Phase 7: Run Decision Intelligence orchestrator and persist results
    decision_report_dict = None
    phase7_path = processed_dir / "phase7_decisions.json"
    # Stale-result protection: remove any previous report before running
    # so a failure cannot leave an old report masquerading as current.
    if phase7_path.exists():
        phase7_path.unlink()
    try:
        from src.decision.orchestrator import DecisionOrchestrator, OrchestrationRequest

        request = OrchestrationRequest(
            dataset_id=dataset_id,
            analytics_result=analytics_result_dict,
            prediction_report=prediction_report_dict,
            phase5_result=phase5_result_dict,
            marketing_result=phase6_result_dict,
        )
        orchestrator = DecisionOrchestrator()
        orch_report = orchestrator.run(request)
        decision_report_dict = orch_report.to_dict()
        with open(phase7_path, "w") as f:
            json.dump(decision_report_dict, f, indent=2, default=str)

        n_decisions = len(orch_report.decisions())
        report.append(
            f"Phase 7 decision intelligence: status={orch_report.status}, "
            f"{n_decisions} decision(s) generated."
        )
    except Exception as e:
        report.append(f"Phase 7 decision intelligence failed (non-fatal): {e}")
        # Persist a failure marker so consumers can distinguish
        # "failed" from "never run".
        try:
            with open(phase7_path, "w") as f:
                json.dump({
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "error": str(e),
                }, f, indent=2)
        except Exception:
            pass  # Best-effort — filesystem errors don't crash the pipeline

    return {
        "dataset": dataset,
        "capabilities": capabilities,
        "generic_kpis": generic_kpis,
        "report": report,
        # Phase 2 additions (backward-compatible — new keys only)
        "canonical": canonical,
        "context": context,
        # Phase 4 addition
        "prediction_report": prediction_report_dict,
        # Phase 5 addition
        "phase5_result": phase5_result_dict,
        # Phase 6 addition
        "phase6_result": phase6_result_dict,
        # Phase 7 addition
        "decision_report": decision_report_dict,
    }
