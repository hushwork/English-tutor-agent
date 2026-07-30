"""Lightweight spectral viseme detection — audio waveform → mouth shape.

Uses FFT-based spectral features (centroid + spread) to classify
audio into ~10 viseme categories at ~0.1ms per 20ms window.

Much faster than LPC formant extraction (~0.1ms vs ~3ms) and
works for both vowels and consonants.

Vowels: distinguished by spectral centroid position
  - Low centroid (200-600Hz): /u/, /o/  → V07_UW_W, V08_OW
  - Mid centroid (600-1200Hz): /a/, /æ/ → V02_AA, V01_AE_AH
  - High centroid (1200-2500Hz): /i/, /ɪ/ → V06_IY_IH

Consonants: distinguished by centroid + spread
  - Very high centroid (>3000Hz): /s/, /ʃ/ → V15_S_Z, V16_SH_ZH
  - High centroid + wide spread: /f/, /θ/ → V18_F_V, V17_TH_DH
  - Burst (short, broadband): /p/, /t/, /k/ → V21_P_B_M, V19_T_D_N
"""

from __future__ import annotations

import math
import numpy as np
from camera_tutor.avatar import Viseme


def _spectral_features(signal: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Compute (centroid_Hz, spread_Hz, energy) from audio window.

    Centroid: center of mass of spectrum (brightness indicator)
    Spread: spectral bandwidth (how wide the frequency distribution is)
    Energy: RMS power (silence detection)

    All computed from a single FFT — ~0.05ms on modern CPU.
    """
    n = len(signal)
    windowed = signal * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    total = np.sum(spec)
    if total < 1e-10:
        return 0.0, 0.0, 0.0

    # Centroid
    centroid = float(np.sum(freqs * spec) / total)

    # Spread
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total))

    # Energy (RMS)
    energy = float(np.sqrt(np.mean(signal ** 2)))

    return centroid, spread, energy


def classify_viseme(signal: np.ndarray, sr: int) -> Viseme:
    """Classify a 20ms audio window into a Viseme.

    Uses spectral centroid + spread as a 2D feature space.

    Args:
        signal: float32 array, 20ms at sample rate sr, normalized to [-1, 1]
        sr: sample rate (24000)

    Returns:
        Viseme classification
    """
    centroid, spread, energy = _spectral_features(signal, sr)

    # ── Silence ──
    if energy < 0.005:
        return Viseme.V00_SIL

    # ── Consonants (high centroid, wide spread, or short/transient) ──
    if centroid > 3000:
        # Sibilants: very high centroid
        if spread > 1500:
            return Viseme.V16_SH_ZH   # /ʃ/, /ʒ/, /tʃ/, /dʒ/
        return Viseme.V15_S_Z         # /s/, /z/

    if centroid > 2000 and spread > 1200:
        # Fricatives: medium-high centroid, moderately wide spread
        if centroid > 2600:
            return Viseme.V18_F_V     # /f/, /v/ — upper teeth on lower lip ☆
        return Viseme.V17_TH_DH       # /θ/, /ð/ — tongue between teeth ☆☆☆

    if centroid > 1600 and energy < 0.02:
        # Short/burst-like: stop consonants or /h/
        if spread < 800:
            return Viseme.V12_H       # /h/ — narrow spectrum
        return Viseme.V19_T_D_N       # /t/, /d/, /n/ — alveolar

    # ── Vowels & sonorants (lower centroid, narrower spread) ──
    # Centroid correlates with vowel frontness/height
    if centroid < 400:
        return Viseme.V07_UW_W        # /u/, /w/ — very low centroid
    if centroid < 550:
        return Viseme.V08_OW          # /oʊ/ — low centroid
    if centroid < 700:
        return Viseme.V03_AO          # /ɔ/ — low-mid centroid
    if centroid < 850:
        return Viseme.V02_AA          # /ɑ/ — mid-low centroid ☆
    if centroid < 1100:
        return Viseme.V05_ER          # /ɝ/ — mid centroid, r-colored ☆
    if centroid < 1400:
        return Viseme.V01_AE_AH       # /æ/, /ʌ/ — mid-high centroid ☆
    if centroid < 1700:
        return Viseme.V04_EH_EY       # /ɛ/, /eɪ/
    return Viseme.V06_IY_IH           # /i/, /ɪ/ — highest vowel centroid


def chunk_to_visemes(pcm_chunk: bytes, sr: int) -> list[Viseme]:
    """Sliding-window analysis: multiple visemes per audio chunk.

    30ms stride → ~33fps per chunk. Each window ~0.1ms (FFT only).
    Returns deduped viseme sequence for the chunk.
    """
    signal = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0

    window_samples = int(sr * 0.020)
    hop_samples = int(sr * 0.030)

    if len(signal) < window_samples:
        return []

    visemes = []
    for start in range(0, len(signal) - window_samples + 1, hop_samples):
        window = signal[start:start + window_samples]
        visemes.append(classify_viseme(window, sr))

    if not visemes:
        return []

    # Dedup consecutive duplicates
    result = [visemes[0]]
    for v in visemes[1:]:
        if v != result[-1]:
            result.append(v)
    return result
