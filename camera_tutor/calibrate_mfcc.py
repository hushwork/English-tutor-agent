#!/usr/bin/env python3
"""MFCC calibration — analyze recorded Emma audio for threshold tuning.

Two-step workflow:

  1. Record audio:
     SAVE_CALIBRATION_AUDIO=1 python3 camera_tutor/realtime_demo.py
     # Talk to Emma for a few sentences, then Ctrl+C
     # WAV files saved to .camera-tutor-data/calibration/

  2. Analyze:
     python3 camera_tutor/calibrate_mfcc.py .camera-tutor-data/calibration/
     # Prints MFCC ranges and recommended thresholds for classify_viseme
"""

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_tutor.spectral_viseme import _mfcc


def analyze_dir(wav_dir: str) -> None:
    """Analyze all WAV files in a directory."""
    wav_files = sorted(Path(wav_dir).glob("*.wav"))
    if not wav_files:
        print(f"❌ No WAV files found in {wav_dir}")
        print("   Run: SAVE_CALIBRATION_AUDIO=1 python3 camera_tutor/realtime_demo.py")
        return

    print(f"=== Analyzing {len(wav_files)} WAV files ===")
    all_arr = []
    total_frames = 0

    for wf in wav_files:
        try:
            with wave.open(str(wf), "rb") as f:
                sr = f.getframerate()
                raw = f.readframes(f.getnframes())
        except Exception as e:
            print(f"  ⚠️  Skip {wf.name}: {e}")
            continue

        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        win_samples = int(sr * 0.020)
        hop_samples = int(sr * 0.030)

        count = 0
        for start in range(0, len(signal) - win_samples + 1, hop_samples):
            window = signal[start:start + win_samples]
            window = window * np.hanning(len(window))
            if float(np.sqrt(np.mean(window**2))) < 0.005:
                continue
            mfcc = _mfcc(window, sr, n_mfcc=13)
            all_arr.append(mfcc)
            count += 1

        dur = len(signal) / sr
        print(f"  {wf.name}: {count} speech frames, {dur:.1f}s")
        total_frames += count

    if not all_arr:
        print("❌ No speech frames found")
        return

    arr = np.array(all_arr)  # (n_frames, 13)
    hf = np.abs(arr[:, 6]) + np.abs(arr[:, 7])
    energy = np.array([float(np.sqrt(np.mean(w**2))) for w in [
        np.hanning(len(w)) * w for w in [arr[i] for i in range(min(len(arr), 100))]
    ]])  # rough proxy

    print(f"\n=== MFCC Statistics ({total_frames} frames) ===")
    print(f"{'Coeff':>6} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print("-" * 42)
    for i in range(13):
        col = arr[:, i]
        print(f"C{i:<5d} {col.min():8.1f} {col.max():8.1f} {col.mean():8.1f} {col.std():8.1f}")

    print(f"\nDerived: hf_rough [p50={np.percentile(hf,50):.1f}, p75={np.percentile(hf,75):.1f}, p90={np.percentile(hf,90):.1f}]")

    print(f"\n=== Suggested classify_viseme thresholds ===")
    c1 = arr[:, 1]
    c2 = arr[:, 2]
    c1_p = [np.percentile(c1, p) for p in [10, 25, 50, 75, 90]]
    c2_p = [np.percentile(c2, p) for p in [10, 25, 50, 75, 90]]
    hf_p = [np.percentile(hf, p) for p in [50, 75, 90, 95]]

    print(f"  # Silence: energy < 0.005")
    print(f"  # Sibilants: hf > {hf_p[3]:.0f} AND energy > 0.02")
    print(f"  # Fricatives: hf > {hf_p[2]:.0f}")
    print(f"  # c1 range: [{int(c1_p[0])}, {int(c1_p[4])}]")
    print(f"  # c2 range: [{int(c2_p[0])}, {int(c2_p[4])}]")
    print(f"  # c1 boundaries: {int(c1_p[0])}, {int(c1_p[1])}, {int(c1_p[2])}, {int(c1_p[3])}, {int(c1_p[4])}")
    print(f"  # c2 boundaries: {int(c2_p[0])}, {int(c2_p[1])}, {int(c2_p[2])}, {int(c2_p[3])}, {int(c2_p[4])}")


def main():
    if len(sys.argv) < 2:
        default = Path(__file__).resolve().parent.parent / ".camera-tutor-data" / "calibration"
        if default.is_dir():
            analyze_dir(str(default))
        else:
            print(__doc__)
        return

    path = sys.argv[1]
    if Path(path).is_dir():
        analyze_dir(path)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
