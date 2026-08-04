"""Shared path helpers for the Camera Tutor standalone app.

All runtime data lives under one data directory:
- default: <app_root>/.camera-tutor-data/  (app_root = parent of this package)
- override: CAMERA_TUTOR_DATA_DIR environment variable
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Return the Camera Tutor data directory (created by callers as needed)."""
    return Path(os.environ.get(
        "CAMERA_TUTOR_DATA_DIR", str(APP_ROOT / ".camera-tutor-data")))
