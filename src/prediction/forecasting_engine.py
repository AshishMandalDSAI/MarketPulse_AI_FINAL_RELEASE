"""Phase 4 -- General Forecasting Engine.

Consumes an AnalysisContext (from the canonical data contract) and produces a
ForecastResult.  Trains multiple simple models, selects the best via
time-aware validation, and generates point forecasts with confidence intervals.

No hardcoded column names.  Confidence derives from actual validation metrics.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.contracts import ForecastResult, ModelCandidate, FeatureImportance
from src.data_engine.canonical import AnalysisContext
from src.prediction.task_detector import MAX_ROWS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ROWS = 10
_TRAIN_FRACTION = 0.80
_CONFIDENCE_Z = 1.96  # 95 % prediction interval


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_time_attr(time_info, attr: str, default=None):
    """Read a TimeInfo attribute regardless of whether it is a dataclass or dict."""
    if time_info is None:
        return default
    if hasattr(time_info, attr):
        return getattr(time_info, attr)
    if isinstance(time_info, dict):
        return time_info.get(attr, default)
    return default


def _detect_frequency(time_info) -> str:
    """Map time_info.time_granularity to a normalised frequency label."""
    raw = _safe_time_attr(time_info, "time_granularity", "irregular")
    mapping = {
        "daily": "daily",
        "weekly": "weekly",
        "monthly": "monthly",
        "quarterly": "monthly",   # treat quarterly as monthly for offset logic
        "yearly": "yearly",
    }
    return mapping.get(raw, "irregular")


def _generate_future_dates(last_date: pd.Timestamp, horizon: int,
                           frequency: str) -> list[str]:
    """Generate *horizon* future date strings starting after *last_date*."""
    offset_map = {
        "daily": pd.DateOffset(days=1),
        "weekly": pd.DateOffset(weeks=1),
        "monthly": pd.DateOffset(months=1),
        "yearly": pd.DateOffset(years=1),
    }
    offset = offset_map.get(frequency, pd.DateOffset(days=1))
    dates: list[str] = []
    current = last_date
    for _ in range(horizon):
        current = current + offset
        dates.append(current.strftime("%Y-%m-%d"))
    return dates


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE, and R-squared on the validation set."""
    residuals = actual - predicted
    n = len(actual)
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # MAPE -- guard against zero actuals
    nonzero = actual != 0
    if nonzero.any():
        mape = float(np.mean(np.abs(residuals[nonzero] / actual[nonzero])) * 100)
    else:
        mape = float("inf")

    # R-squared
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def _train_naive(train: np.ndarray, val: np.ndarray,
                 horizon: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray]:
    """Naive forecast: repeat the last training value."""
    t0 = time.time()
    last = train[-1]
    val_pred = np.full(len(val), last)
    future_pred = np.full(horizon, last)
    metrics = _compute_metrics(val, val_pred)
    elapsed = time.time() - t0
    candidate = ModelCandidate(
        name="Naive (Last Value)",
        model_type="naive",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred, future_pred


def _train_seasonal_naive(train: np.ndarray, val: np.ndarray,
                          horizon: int,
                          season_length: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray]:
    """Seasonal naive: repeat the last seasonal cycle."""
    t0 = time.time()
    # Validation predictions
    val_pred = np.empty(len(val))
    for i in range(len(val)):
        idx = len(train) - season_length + (i % season_length)
        val_pred[i] = train[idx] if 0 <= idx < len(train) else train[-1]

    # Future predictions
    full = np.concatenate([train, val])
    future_pred = np.empty(horizon)
    for i in range(horizon):
        idx = len(full) - season_length + (i % season_length)
        future_pred[i] = full[idx] if 0 <= idx < len(full) else full[-1]

    metrics = _compute_metrics(val, val_pred)
    elapsed = time.time() - t0
    candidate = ModelCandidate(
        name="Seasonal Naive",
        model_type="seasonal_naive",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred, future_pred


def _train_moving_average(train: np.ndarray, val: np.ndarray,
                          horizon: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray]:
    """Moving-average forecast using a rolling window."""
    t0 = time.time()
    window = min(12, max(2, len(train) // 3))

    # Validation: one-step-ahead rolling mean
    series = list(train)
    val_pred = np.empty(len(val))
    for i in range(len(val)):
        val_pred[i] = np.mean(series[-window:])
        series.append(val[i])

    # Future: keep extending with predictions
    series_full = list(np.concatenate([train, val]))
    future_pred = np.empty(horizon)
    for i in range(horizon):
        pred = np.mean(series_full[-window:])
        future_pred[i] = pred
        series_full.append(pred)

    metrics = _compute_metrics(val, val_pred)
    elapsed = time.time() - t0
    candidate = ModelCandidate(
        name=f"Moving Average (window={window})",
        model_type="moving_average",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred, future_pred


def _train_linear_trend(train: np.ndarray, val: np.ndarray,
                        horizon: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray, float]:
    """Linear trend via sklearn LinearRegression on time index.

    Returns the candidate, validation predictions, future predictions,
    and the trend coefficient.
    """
    from sklearn.linear_model import LinearRegression

    t0 = time.time()
    X_train = np.arange(len(train)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, train)

    # Validation
    X_val = np.arange(len(train), len(train) + len(val)).reshape(-1, 1)
    val_pred = model.predict(X_val)

    # Future (retrained on all data)
    full = np.concatenate([train, val])
    X_full = np.arange(len(full)).reshape(-1, 1)
    model.fit(X_full, full)
    X_future = np.arange(len(full), len(full) + horizon).reshape(-1, 1)
    future_pred = model.predict(X_future)

    coef = float(model.coef_[0])

    # Metrics on validation set (from the train-only fit)
    model_train_only = LinearRegression().fit(X_train, train)
    val_pred_eval = model_train_only.predict(X_val)
    metrics = _compute_metrics(val, val_pred_eval)
    elapsed = time.time() - t0

    candidate = ModelCandidate(
        name="Linear Trend",
        model_type="linear_trend",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred_eval, future_pred, coef


def _train_exponential_smoothing(
    train: np.ndarray, val: np.ndarray, horizon: int, frequency: str,
) -> Optional[tuple[ModelCandidate, np.ndarray, np.ndarray]]:
    """Exponential smoothing via statsmodels.  Returns None if unavailable."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return None

    t0 = time.time()
    try:
        # Choose seasonal periods if enough data
        seasonal_periods_map = {
            "daily": 7,
            "weekly": 52,
            "monthly": 12,
            "yearly": 1,
        }
        sp = seasonal_periods_map.get(frequency, 1)
        use_seasonal = sp > 1 and len(train) >= 2 * sp

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=sp if use_seasonal else None,
        ).fit(optimized=True)

        val_pred = model.forecast(len(val))

        # Retrain on all data for future predictions
        full = np.concatenate([train, val])
        model_full = ExponentialSmoothing(
            full,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=sp if use_seasonal else None,
        ).fit(optimized=True)
        future_pred = model_full.forecast(horizon)

        # Replace any NaN predictions with the last known value
        fallback = full[-1]
        val_pred = np.where(np.isfinite(val_pred), val_pred, fallback)
        future_pred = np.where(np.isfinite(future_pred), future_pred, fallback)

        metrics = _compute_metrics(val, val_pred)
        elapsed = time.time() - t0

        candidate = ModelCandidate(
            name="Exponential Smoothing",
            model_type="exponential_smoothing",
            metrics=metrics,
            score=metrics["MAE"],
            training_time_seconds=elapsed,
        )
        return candidate, val_pred, future_pred

    except Exception as exc:
        logger.debug("Exponential smoothing failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Season-length heuristic
# ---------------------------------------------------------------------------

def _guess_season_length(frequency: str, n_obs: int) -> int:
    """Return a reasonable seasonal period length based on the frequency."""
    defaults = {
        "daily": 7,
        "weekly": 52,
        "monthly": 12,
        "yearly": 1,
    }
    sp = defaults.get(frequency, 1)
    # Only use seasonality if we have at least two full cycles
    if sp > 1 and n_obs < 2 * sp:
        sp = 1
    return max(sp, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_forecast(
    context: AnalysisContext,
    target_column: str = None,
    horizon: int = None,
) -> ForecastResult:
    """Run the general forecasting engine.

    Parameters
    ----------
    context : AnalysisContext
        The canonical analysis context containing the dataset, time info,
        target candidates, and data quality metadata.
    target_column : str, optional
        The column to forecast.  If not provided, the first target candidate
        from ``context.dataset.target_candidates`` is used.
    horizon : int, optional
        Number of future periods to forecast.  Defaults to
        ``min(12, n_observations // 5)``.

    Returns
    -------
    ForecastResult
        A fully populated forecast result with predictions, confidence
        intervals, model candidates, and validation metrics.
    """
    ds = context.dataset
    df = ds.dataframe

    # ------------------------------------------------------------------
    # Guard: no DataFrame
    # ------------------------------------------------------------------
    if df is None or df.empty:
        return ForecastResult(
            status="unavailable",
            explanation="No data available for forecasting.",
            limitations=["DataFrame is empty or missing."],
        )

    # ------------------------------------------------------------------
    # Time column detection
    # ------------------------------------------------------------------
    time_info = ds.time_info
    time_col = _safe_time_attr(time_info, "time_column") if hasattr(time_info, "time_column") or isinstance(time_info, dict) else None

    if not time_col or time_col not in df.columns:
        return ForecastResult(
            status="unavailable",
            explanation="No time column detected in the dataset.",
            limitations=["Forecasting requires a datetime column for temporal ordering."],
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    if target_column and target_column in df.columns:
        target_col = target_column
    elif ds.target_candidates:
        first_candidate = ds.target_candidates[0]
        target_col = first_candidate.column if hasattr(first_candidate, "column") else None
    else:
        target_col = None

    if not target_col or target_col not in df.columns:
        return ForecastResult(
            status="unavailable",
            time_column=time_col,
            explanation="No suitable numeric target column found for forecasting.",
            limitations=["No numeric column qualifies as a forecast target."],
        )

    # Verify the target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return ForecastResult(
            status="unavailable",
            target_column=target_col,
            time_column=time_col,
            explanation=f"Target column '{target_col}' is not numeric.",
            limitations=["Forecasting requires a numeric target column."],
        )

    # ------------------------------------------------------------------
    # Sample down large datasets
    # ------------------------------------------------------------------
    working_df = df[[time_col, target_col]].copy()
    if len(working_df) > MAX_ROWS:
        logger.info("Sampling dataset from %d to %d rows.", len(working_df), MAX_ROWS)
        working_df = working_df.sample(n=MAX_ROWS, random_state=42)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    # Parse time column and sort
    working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
    working_df = working_df.dropna(subset=[time_col])
    working_df = working_df.sort_values(time_col).reset_index(drop=True)

    # Drop rows where target is NaN
    working_df = working_df.dropna(subset=[target_col])

    n_obs = len(working_df)

    # ------------------------------------------------------------------
    # Edge case: too few rows
    # ------------------------------------------------------------------
    if n_obs < _MIN_ROWS:
        return ForecastResult(
            status="insufficient_data",
            target_column=target_col,
            time_column=time_col,
            explanation=(
                f"Only {n_obs} valid observations after cleaning. "
                f"Forecasting requires at least {_MIN_ROWS}."
            ),
            limitations=[
                f"Row count ({n_obs}) is below the minimum ({_MIN_ROWS}).",
                "Consider providing more data for reliable forecasting.",
            ],
        )

    # ------------------------------------------------------------------
    # Edge case: all-constant target
    # ------------------------------------------------------------------
    target_values = working_df[target_col].values.astype(float)
    if np.std(target_values) == 0:
        return ForecastResult(
            status="failed",
            target_column=target_col,
            time_column=time_col,
            explanation=(
                f"Target column '{target_col}' has zero variance "
                f"(constant value {target_values[0]:.4g}). "
                "Forecasting is not meaningful."
            ),
            limitations=["Target column is constant; no pattern to learn."],
        )

    # ------------------------------------------------------------------
    # Frequency and horizon
    # ------------------------------------------------------------------
    frequency = _detect_frequency(time_info)
    if horizon is None:
        horizon = min(12, max(1, n_obs // 5))

    # ------------------------------------------------------------------
    # Train / validation split (time-aware)
    # ------------------------------------------------------------------
    split_idx = max(2, int(n_obs * _TRAIN_FRACTION))
    train_vals = target_values[:split_idx]
    val_vals = target_values[split_idx:]

    if len(val_vals) < 1:
        # Fall back: use last observation as single-point validation
        split_idx = n_obs - 1
        train_vals = target_values[:split_idx]
        val_vals = target_values[split_idx:]

    # ------------------------------------------------------------------
    # Train model candidates
    # ------------------------------------------------------------------
    candidates: list[ModelCandidate] = []
    val_predictions: dict[str, np.ndarray] = {}
    future_predictions: dict[str, np.ndarray] = {}
    trend_coef: Optional[float] = None

    # 1. Naive
    cand, vp, fp = _train_naive(train_vals, val_vals, horizon)
    candidates.append(cand)
    val_predictions[cand.name] = vp
    future_predictions[cand.name] = fp

    # 2. Seasonal Naive
    season_len = _guess_season_length(frequency, n_obs)
    if season_len > 1:
        cand, vp, fp = _train_seasonal_naive(train_vals, val_vals, horizon, season_len)
        candidates.append(cand)
        val_predictions[cand.name] = vp
        future_predictions[cand.name] = fp

    # 3. Moving Average
    cand, vp, fp = _train_moving_average(train_vals, val_vals, horizon)
    candidates.append(cand)
    val_predictions[cand.name] = vp
    future_predictions[cand.name] = fp

    # 4. Linear Trend
    try:
        cand, vp, fp, coef = _train_linear_trend(train_vals, val_vals, horizon)
        candidates.append(cand)
        val_predictions[cand.name] = vp
        future_predictions[cand.name] = fp
        trend_coef = coef
    except Exception as exc:
        logger.debug("Linear trend training failed: %s", exc)

    # 5. Exponential Smoothing (optional dependency)
    es_result = _train_exponential_smoothing(train_vals, val_vals, horizon, frequency)
    if es_result is not None:
        cand, vp, fp = es_result
        candidates.append(cand)
        val_predictions[cand.name] = vp
        future_predictions[cand.name] = fp

    # ------------------------------------------------------------------
    # Model selection: lowest MAE on validation set
    # ------------------------------------------------------------------
    best_candidate: Optional[ModelCandidate] = None
    best_mae = float("inf")
    for cand in candidates:
        cand_mae = cand.metrics.get("MAE", float("inf"))
        if cand_mae < best_mae:
            best_mae = cand_mae
            best_candidate = cand

    if best_candidate is None:
        return ForecastResult(
            status="failed",
            target_column=target_col,
            time_column=time_col,
            horizon=horizon,
            frequency=frequency,
            explanation="All model candidates failed during training.",
            model_candidates=candidates,
            limitations=["No model produced valid predictions."],
        )

    best_candidate.selected = True
    selected_name = best_candidate.name
    preds = future_predictions[selected_name]
    val_preds = val_predictions[selected_name]

    # ------------------------------------------------------------------
    # Confidence intervals from validation residuals
    # ------------------------------------------------------------------
    residuals = val_vals - val_preds
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

    lower_bound = (preds - _CONFIDENCE_Z * residual_std).tolist()
    upper_bound = (preds + _CONFIDENCE_Z * residual_std).tolist()

    # ------------------------------------------------------------------
    # Future dates
    # ------------------------------------------------------------------
    last_date = working_df[time_col].iloc[-1]
    future_dates = _generate_future_dates(last_date, horizon, frequency)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    feature_importance: list[FeatureImportance] = []
    if best_candidate.model_type == "linear_trend" and trend_coef is not None:
        feature_importance.append(
            FeatureImportance(feature="time_index", importance=trend_coef, rank=1)
        )

    # ------------------------------------------------------------------
    # Confidence score from validation R-squared
    # ------------------------------------------------------------------
    best_r2 = best_candidate.metrics.get("R2", 0.0)
    # Clamp between 0 and 1
    confidence_score = max(0.0, min(1.0, best_r2))

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------
    limitations: list[str] = []
    if n_obs < 30:
        limitations.append(
            f"Only {n_obs} observations; forecasts may be unreliable."
        )
    if frequency == "irregular":
        limitations.append(
            "Irregular time intervals detected; forecasts assume regular spacing."
        )
    if len(val_vals) < 5:
        limitations.append(
            f"Validation set has only {len(val_vals)} point(s); "
            "error estimates may be imprecise."
        )
    if residual_std > 0 and best_mae > 0:
        mean_val = float(np.mean(np.abs(target_values)))
        if mean_val > 0 and best_mae / mean_val > 0.3:
            limitations.append(
                "Model error is high relative to the target's magnitude; "
                "treat predictions as rough estimates."
            )
    if len(working_df) < len(df):
        limitations.append(
            f"Dataset was sampled from {len(df)} to {len(working_df)} rows."
        )
    if not feature_importance:
        limitations.append(
            "Selected model does not produce feature-importance scores."
        )

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    explanation = (
        f"Trained {len(candidates)} model(s) on '{target_col}' over "
        f"{n_obs} observations ({frequency} frequency). "
        f"Best model: {selected_name} "
        f"(validation MAE={best_mae:.4f}, R2={best_r2:.4f}). "
        f"Forecasting {horizon} step(s) ahead with 95% confidence intervals."
    )

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return ForecastResult(
        status="success",
        target_column=target_col,
        time_column=time_col,
        horizon=horizon,
        frequency=frequency,
        model_name=selected_name,
        predictions=preds.tolist(),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        future_dates=future_dates,
        metrics=best_candidate.metrics,
        confidence=confidence_score,
        model_candidates=candidates,
        feature_importance=feature_importance,
        limitations=limitations,
        explanation=explanation,
    )
