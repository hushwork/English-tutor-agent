#!/usr/bin/env python3
"""Run the English Tutor CLI."""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from english_tutor.cli import run

if __name__ == "__main__":
    run()
