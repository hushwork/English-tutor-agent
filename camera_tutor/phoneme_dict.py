"""CMUdict phoneme dictionary — word → phoneme sequence for viseme alignment.

Downloads and caches the Carnegie Mellon Pronunciation Dictionary (CMUdict)
locally. Maps any English word to its ARPABET phoneme sequence, then to visemes.

Usage:
    from camera_tutor.phoneme_dict import word_to_visemes

    visemes = word_to_visemes("hello")
    # → [(0.0, Viseme.V12_H), (0.25, Viseme.V01_AE_AH), ...]
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx

from camera_tutor.avatar import PHONEME_TO_VISEME, Viseme

# Cache location
_CACHE_DIR = Path.home() / ".camera-tutor-data"
_CMU_PATH = _CACHE_DIR / "cmudict.dict"

# CMUdict source URL
_CMU_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"

# Global dictionary (lazy load)
_cmu: Optional[dict[str, list[str]]] = None


def _download_cmu():
    """Download CMUdict if not already cached."""
    if _CMU_PATH.exists():
        return

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try multiple sources
    urls = [
        _CMU_URL,
        "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.dict",
    ]
    for url in urls:
        try:
            resp = httpx.get(url, timeout=15)
            if resp.status_code == 200:
                _CMU_PATH.write_bytes(resp.content)
                assert _CMU_PATH.exists()
                return
        except Exception:
            continue

    raise RuntimeError(
        "Failed to download CMUdict. "
        "Download manually and place at: " + str(_CMU_PATH)
    )


def _load_cmu():
    """Parse CMUdict into a dict mapping word → phoneme list."""
    global _cmu
    if _cmu is not None:
        return

    _download_cmu()

    _cmu = {}
    with open(_CMU_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            # word is first part, strip parenthesized suffix: "a(2)" → "a"
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
            phonemes = parts[1:]
            # Keep first pronunciation only (most common)
            if word not in _cmu:
                _cmu[word] = phonemes

    assert len(_cmu) > 100000, f"CMUdict too small: {len(_cmu)} entries"


def get_phonemes(word: str) -> Optional[list[str]]:
    """Get ARPABET phoneme sequence for a word.

    Returns None if word not in dictionary.
    Phonemes include stress markers: HH1, AH0, L, OW1, etc.
    """
    _load_cmu()
    return _cmu.get(word.lower())


def phoneme_to_viseme(phoneme: str) -> Viseme:
    """Convert a single ARPABET phoneme to a Viseme.

    Strips stress markers (AH1 → AH, AH0 → AH) before lookup.
    """
    key = re.sub(r"[0-2]$", "", phoneme).lower()
    return PHONEME_TO_VISEME.get(key, Viseme.V00_SIL)


def word_to_viseme_timeline(word: str, start_s: float, end_s: float) -> list[tuple[float, Viseme]]:
    """Build a viseme timeline for one word.

    Args:
        word: The word text
        start_s: Word start time in seconds
        end_s: Word end time in seconds

    Returns:
        List of (time_s, Viseme) pairs, evenly distributed across the word duration.
        Falls back to a single word-level viseme if phonemes not available.
    """
    phonemes = get_phonemes(word)

    if not phonemes or len(phonemes) < 1:
        # Fallback to word-level single viseme
        from camera_tutor.avatar import PHONEME_TO_VISEME as PMAP
        word_lower = word.lower().strip(",.!?\"'")
        for key, v in sorted(PMAP.items(), key=lambda x: len(x[0]), reverse=True):
            if key in word_lower:
                return [(start_s, v)]
        return [(start_s, Viseme.V02_AA)]

    duration = end_s - start_s
    if duration <= 0:
        duration = 0.3  # fallback

    per_phoneme = duration / len(phonemes)
    timeline = []
    for i, p in enumerate(phonemes):
        t = start_s + i * per_phoneme
        timeline.append((t, phoneme_to_viseme(p)))

    return timeline
