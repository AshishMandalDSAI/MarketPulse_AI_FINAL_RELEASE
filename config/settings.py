"""Centralized, environment-aware paths & limits for MarketPulse AI.

Everything here is computed relative to the project root or read from the
environment — never a hardcoded absolute personal path.
"""

import os
from pathlib import Path

# Project root = two levels up from this file (config/settings.py -> repo root)
BASE_DIR = Path(__file__).resolve().parents[1]

# --- Built-in / legacy demo dataset directories (backward compatible) ------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# --- Multi-dataset ("dataset-first") directories ----------------------------
UPLOADS_DIR = DATA_DIR / "uploads"

DEMO_DATASET_ID = "demo"


def dataset_upload_dir(dataset_id: str) -> Path:
    """Root directory for one uploaded dataset's artifacts."""
    return UPLOADS_DIR / dataset_id


def dataset_raw_path(dataset_id: str, filename: str) -> Path:
    return dataset_upload_dir(dataset_id) / "raw" / filename


def dataset_processed_dir(dataset_id: str) -> Path:
    return dataset_upload_dir(dataset_id) / "processed"


def dataset_models_dir(dataset_id: str) -> Path:
    return dataset_upload_dir(dataset_id) / "models"


def ensure_dataset_dirs(dataset_id: str) -> None:
    for d in (
        dataset_upload_dir(dataset_id) / "raw",
        dataset_processed_dir(dataset_id),
        dataset_models_dir(dataset_id),
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- API auth (optional; no-op if unset) ------------------------------------
API_KEY = os.environ.get("MARKETPULSE_API_KEY") or os.environ.get("API_KEY") or None

# --- Performance / safety limits --------------------------------------------
MAX_ROWS = int(os.environ.get("MARKETPULSE_MAX_ROWS", "100000"))
MAX_UPLOAD_MB = int(os.environ.get("MARKETPULSE_MAX_UPLOAD_MB", "50"))
