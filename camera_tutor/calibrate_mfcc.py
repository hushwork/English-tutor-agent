#!/usr/bin/env python3
"""MFCC calibration tool — analyzes real Emma speech to find good thresholds.

Usage:
  # Record Emma speaking and save as WAV (24kHz mono 16-bit)
  # Then run:
  python3 camera_tutor/calibrate_mfcc.py path/to/emma_speech.wav

  # Or generate a test tone:
  python3 camera_tutor/calibrate_mfcc.py --sweep

Output: prints MFCC coefficient ranges (min/max/mean/std) and
viseme distribution, so you can tune the classify_viseme thresholds.
"""

import argparse
import struct
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_tutor.spectral_viseme import _mfcc, classify_viseme
from camera_tutor.avatar import Viseme


def analyze_file(wav_path: str) -> None:
    """Analyze a WAV file frame-by-frame."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"File: {wav_path}")
    print(f"Sample rate: {sr} Hz, duration: {n_frames/sr:.1f}s, channels: {wf.getnchannels()}")
    print()

    # Process in 20ms windows with 30ms stride
    win_samples = int(sr * 0.020)
    hop_samples = int(sr * 0.030)

    mfcc_buf = []       # all MFCC vectors (silence excluded)
    viseme_counts = {}  # viseme → count
    total = 0

    for start in range(0, len(signal) - win_samples + 1, hop_samples):
        window = signal[start:start + win_samples]
        window = window * np.hanning(len(window))
        energy = float(np.sqrt(np.mean(window**2)))
        if energy < 0.005:
            continue  # skip silence

        mfcc = _mfcc(window, sr, n_mfcc=13)
        mfcc_buf.append(mfcc)
        v = classify_viseme(window, sr)
        label = v.name
        viseme_counts[label] = viseme_counts.get(label, 0) + 1
        total += 1

    mfcc_arr = np.array(mfcc_buf)  # (n_frames, 13)

    print(f"Analyzed {total} non-silent frames ({len(signal)/sr:.1f}s audio)")
    print()

    # Per-coefficient statistics
    print("=== MFCC Coefficient Ranges ===")
    print(f"{'Coef':>5} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 48)
    for i in range(13):
        col = mfcc_arr[:, i]
        print(f"C{i:<4d} {col.min():10.1f} {col.max():10.1f} {col.mean():10.1f} {col.std():10.1f}")

    # Derived features
    print()
    print("=== Derived Features ===")
    energy_arr = np.array([np.sqrt(np.mean(w**2)) for w in [
        signal[s:s+win_samples] for s in range(0, len(signal)-win_samples+1, hop_samples)
    ][:len(mfcc_arr)]])
    hf_arr = np.abs(mfcc_arr[:, 6]) + np.abs(mfcc_arr[:, 7])
    c3_arr = mfcc_arr[:, 3]
    for name, arr in [("Energy", energy_arr), ("hf_rough", hf_arr), ("|c3|", np.abs(c3_arr))]:
        print(f"  {name:10s}: min={arr.min():.4f} max={arr.max():.4f}  "
              f"p50={np.percentile(arr,50):.4f} p90={np.percentile(arr,90):.4f}")

    # Viseme distribution
    print()
    print("=== Viseme Distribution ===")
    sorted_v = sorted(viseme_counts.items(), key=lambda x: -x[1])
    for name, count in sorted_v:
        bar = "█" * int(count / max(1, total) * 40)
        print(f"  {name:14s}: {count:4d} ({count/total*100:5.1f}%) {bar}")

    # Vowel vs consonant breakdown
    vowel_labels = {"V01_AE_AH","V02_AA","V03_AO","V04_EH_EY","V05_ER",
                    "V06_IY_IH","V07_UW_W","V08_OW","V09_AW","V10_OY","V11_AY"}
    consonant_labels = {"V12_H","V13_R","V14_L","V15_S_Z","V16_SH_ZH",
                        "V17_TH_DH","V18_F_V","V19_T_D_N","V20_K_G_NG","V21_P_B_M"}
    vowel_count = sum(c for n, c in viseme_counts.items() if n in vowel_labels)
    cons_count = sum(c for n, c in viseme_counts.items() if n in consonant_labels)
    sil_count = viseme_counts.get("V00_SIL", 0)
    print(f"\n  Vowels: {vowel_count} ({vowel_count/total*100:.0f}%)  "
          f"Consonants: {cons_count} ({cons_count/total*100:.0f}%)  "
          f"Silence: {sil_count}")


def run_sweep() -> None:
    """Analyze a frequency sweep to understand classifier behavior."""
    import numpy as np
    sr = 24000
    win_samples = int(sr * 0.020)

    print("=== Frequency Sweep Test ===")
    print(f"{'Freq':>8} {'Energy':>10} {'c1':>8} {'c2':>8} {'hf':>7} {'→ viseme'}")
    print("-" * 60)

    for freq in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                 1200, 1500, 1800, 2000, 2500, 3000, 4000]:
        t = np.arange(0, win_samples / sr, 1 / sr)
        signal = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
        mfcc = _mfcc(signal * np.hanning(len(signal)), sr, n_mfcc=13)
        v = classify_viseme(signal * np.hanning(len(signal)), sr)
        hf = abs(mfcc[6]) + abs(mfcc[7])
        energy = float(np.sqrt(np.mean(signal**2)))
        print(f"  {freq:5}Hz   {energy:.6f}  {mfcc[1]:+7.1f}  {mfcc[2]:+7.1f}  "
              f"{hf:5.1f}   → {v.name}")

    # Noise test
    print()
    print("  noise    ...")
    rng = np.random.RandomState(42)
    for amp in [0.01, 0.03, 0.06, 0.1, 0.2]:
        noise = (rng.randn(win_samples) * amp).astype(np.float32)
        mfcc = _mfcc(noise * np.hanning(len(noise)), sr, n_mfcc=13)
        energy = float(np.sqrt(np.mean(noise**2)))
        v = classify_viseme(noise * np.hanning(len(noise)), sr)
        hf = abs(mfcc[6]) + abs(mfcc[7])
        print(f"  noise x{amp:.2f}  {energy:.6f}  {mfcc[1]:+7.1f}  {mfcc[2]:+7.1f}  "
              f"{hf:5.1f}   → {v.name}")


def main():
    p = argparse.ArgumentParser(description="MFCC viseme calibration tool")
    p.add_argument("wav_file", nargs="?", help="WAV file to analyze")
    p.add_argument("--sweep", action="store_true", help="Run frequency sweep test")
    args = p.parse_args()

    if args.sweep:
        run_sweep()
    elif args.wav_file:
        analyze_file(args.wav_file)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
