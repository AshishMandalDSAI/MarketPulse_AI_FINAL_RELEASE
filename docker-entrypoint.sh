#!/usr/bin/env bash
# MarketPulse AI — Docker container entrypoint
#
# Initializes demo data/models on first run ONLY when needed.
# Idempotent: skips initialization when data already exists (user uploads,
# previous runs, or pre-populated host volumes).
#
# Race-condition safe: uses flock(1) on a shared lock file so that when
# multiple containers (dashboard + API) start against the same bind-mounted
# volumes, only one runs the pipeline — the other waits for the lock and
# then sees the sentinels already present.
#
# Sentinel files checked:
#   data/processed/marketing_data_clean.csv  — demo dataset
#   models/mmm_model.pkl                     — trained MMM
#
# If both exist, initialization is skipped entirely — existing user data
# and uploads in data/uploads/ are never touched or overwritten.

set -e

DATA_SENTINEL="data/processed/marketing_data_clean.csv"
MODEL_SENTINEL="models/mmm_model.pkl"
LOCK_FILE="data/.init.lock"

# Ensure base directories exist before attempting the lock file
mkdir -p data/raw data/processed data/uploads models

# Use flock for exclusive locking so only one container initializes.
# The lock is held only during the sentinel check + pipeline run;
# once released, the other container re-checks sentinels and skips.
(
    flock -x 200

    if [ ! -f "$DATA_SENTINEL" ] || [ ! -f "$MODEL_SENTINEL" ]; then
        echo "[MarketPulse] Demo data/models not found — running initialization pipeline..."

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
        python src/utils/database.py 2>/dev/null || true  # database.py is optional

        echo "[MarketPulse] Initialization complete."
    else
        echo "[MarketPulse] Demo data and models already present — skipping initialization."
    fi

) 200>"$LOCK_FILE"

echo "[MarketPulse] Starting service: $@"
exec "$@"
