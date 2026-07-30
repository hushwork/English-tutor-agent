#!/usr/bin/env python3
"""Audio diagnostic — test mic quality, detect dropouts, find right gain.

Usage:
    source venv/bin/activate
    python3 camera_tutor/audio_diagnostic.py          # default mic
    python3 camera_tutor/audio_diagnostic.py 3        # device index 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
import numpy as np
import wave
import time
import os
from collections import deque

SAMPLE_RATE = 16000
DURATION = 8.0


def countdown(secs: int):
    for i in range(secs, 0, -1):
        print(f"\r   Starting in {i}...  ", end="", flush=True)
        time.sleep(1)
    print("\r   🔴 RECORDING NOW — speak clearly!          ")


def find_best_gain(speech_arr: np.ndarray) -> tuple[float, str]:
    peak = float(np.abs(speech_arr).max())
    if peak < 1:
        return 10.0, "silent"
    max_gain = 32767.0 / peak
    best = max(1.0, int(max_gain * 0.7))
    boosted = np.clip(speech_arr * best, -32767, 32767)
    rms = float(np.sqrt(np.mean(boosted ** 2)))
    if rms > 2000:
        return max(1.0, best * 0.5), "loud — headroom added"
    elif rms > 500:
        return best, "good"
    elif rms > 100:
        return best, "quiet but usable"
    else:
        return best, "too quiet"


# ── Device selection ──────────────────────────────────────────────

devices = sd.query_devices()
print("=" * 60)
print("  Microphone Diagnostic")
print("=" * 60)
print()

print("Available input devices:")
for i, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        marker = " ← DEFAULT" if i == sd.default.device[0] else ""
        print(f"  [{i}] {dev['name']} ({dev['default_samplerate']:.0f}Hz){marker}")

if len(sys.argv) > 1:
    try:
        default_in = int(sys.argv[1])
    except ValueError:
        default_in = sd.default.device[0]
else:
    default_in = sd.default.device[0]

print(f"\nUsing device [{default_in}]: {devices[default_in]['name']}")
print()

# ── Step 1: Noise floor (callback-based — no overflow on macOS) ──

input("Step 1/4: Stay SILENT. Press Enter to record 3s of background noise...")
print("   Recording ambient noise...")

ring: deque[bytes] = deque()
def noise_callback(indata, frames, time_info, status):
    ring.append(indata.tobytes())

mic = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="int16",
    blocksize=800, device=default_in, callback=noise_callback,
)
mic.start()
time.sleep(3.0)
mic.stop()
mic.close()

noise_raw = b"".join(ring)
ring.clear()
noise_arr = np.frombuffer(noise_raw, dtype=np.int16).astype(np.float32)
noise_rms = float(np.sqrt(np.mean(noise_arr ** 2)))
print(f"   Noise floor: RMS={noise_rms:.1f}  ({20*np.log10(max(noise_rms,1)/32767):.1f} dBFS)")
print()

# ── Step 2: Speech with callback (zero overflow) ──────────────────

print(f"Step 2/4: You will have {int(DURATION)} seconds to speak.")
print(f"   Say a full sentence: 'Hello, I am testing my microphone quality.'")
print()
countdown(3)

CALLBACK_FRAMES = 800  # 50ms @ 16kHz
speech_ring: deque[tuple[bytes, float]] = deque()  # (data, timestamp)

def speech_callback(indata, frames, time_info, status):
    speech_ring.append((indata.tobytes(), time.time()))

mic = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="int16",
    blocksize=CALLBACK_FRAMES, device=default_in, callback=speech_callback,
)
mic.start()

# Show live level meter while recording
t0 = time.time()
while time.time() - t0 < DURATION:
    time.sleep(0.1)
    # Show latest chunk level
    if speech_ring:
        latest = speech_ring[-1][0]
        arr = np.frombuffer(latest, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        bars = "▁▂▃▄▅▆▇█"
        idx = min(int(rms / 430 * 7), 7)
        bar = bars[max(idx,0)] * max(1, idx + 1)
        print(f"\r   Level: {bar}  (RMS={rms:.0f})  ", end="", flush=True)

mic.stop()
mic.close()
print()

# ── Dropout analysis ──────────────────────────────────────────────

# Check timing of callback chunks
speech_chunks_bytes = [item[0] for item in speech_ring]
speech_times = [item[1] for item in speech_ring]
gaps = []
for i in range(1, len(speech_times)):
    gaps.append(speech_times[i] - speech_times[i-1])

expected_interval = CALLBACK_FRAMES / SAMPLE_RATE  # 0.05s
gaps_over_2x = sum(1 for g in gaps if g > expected_interval * 2)
max_gap = max(gaps) if gaps else 0
expected_chunks = int(DURATION / expected_interval)
actual_chunks = len(speech_chunks_bytes)

print(f"\n   📊 Recording integrity (callback-based):")
print(f"   Chunks captured:  {actual_chunks} (expected ~{expected_chunks})")
print(f"   Max gap:          {max_gap*1000:.1f} ms  {'⚠️  GAP!' if max_gap > expected_interval*2.5 else '✅ tight'}")
print(f"   Gaps > 2x normal: {gaps_over_2x} of {len(gaps)} chunks")

if gaps_over_2x > 5:
    print(f"   ⚠️  Timing gaps detected — audio may have minor stutter")
elif gaps_over_2x > 0 and gaps_over_2x <= 3:
    print(f"   ⚠️  {gaps_over_2x} small gap(s) — likely harmless")
else:
    print(f"   ✅ No significant gaps — stream is clean")
print()

# Concatenate all callback chunks into continuous audio
speech_raw = b"".join(speech_chunks_bytes)
speech_arr = np.frombuffer(speech_raw, dtype=np.int16).astype(np.float32)

window_samples = int(0.5 * SAMPLE_RATE)
max_window_rms = 0.0
max_window_peak = 0.0
for start in range(0, len(speech_arr) - window_samples, window_samples // 2):
    win = speech_arr[start:start + window_samples]
    wrms = float(np.sqrt(np.mean(win ** 2)))
    if wrms > max_window_rms:
        max_window_rms = wrms
        max_window_peak = float(np.abs(win).max())

overall_rms = float(np.sqrt(np.mean(speech_arr ** 2)))
overall_peak = float(np.abs(speech_arr).max())
snr_db = 20 * np.log10(max(max_window_rms, 1) / max(noise_rms, 1))
best_gain, quality = find_best_gain(speech_arr)

print("=" * 60)
print("  Audio Quality Results")
print("=" * 60)
print()
print(f"  Noise floor:              RMS={noise_rms:.1f}")
print(f"  Speech (overall):         RMS={overall_rms:.0f}, Peak={overall_peak:.0f}")
print(f"  Speech (loudest 0.5s):    RMS={max_window_rms:.0f}, Peak={max_window_peak:.0f}")
print(f"  SNR:                      {snr_db:.1f} dB")
print()

print("  Gain simulation:")
print(f"  {'Gain':<8} {'RMS':<10} {'Peak':<10} {'dBFS':<8} Status")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*18}")
best_marker_placed = False
for gain in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
    boosted = np.clip(speech_arr * gain, -32767, 32767)
    rms = float(np.sqrt(np.mean(boosted ** 2)))
    peak = float(np.abs(boosted).max())
    dbfs = 20 * np.log10(max(rms, 0.01) / 32767)
    if peak >= 32767:
        status = "⚠️  CLIPPING!"
    elif rms < 30:
        status = "❌ Too quiet"
    elif rms < 100:
        status = "⚠️  Still quiet"
    elif rms < 1000:
        status = "✅ Good"
    else:
        status = "✅ Loud & clear"
    marker = ""
    if not best_marker_placed and gain >= best_gain:
        marker = "  ← RECOMMENDED"
        best_marker_placed = True
    print(f"  {gain:<8.0f}x {rms:<10.0f} {peak:<10.0f} {dbfs:<8.1f} {status}{marker}")
    if peak >= 32767:
        break

# ── Save WAVs ─────────────────────────────────────────────────────

outdir = "/tmp"
out_raw = os.path.join(outdir, "mic_diag_raw.wav")
with wave.open(out_raw, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(speech_raw)

boosted = np.clip(speech_arr * best_gain, -32767, 32767).astype(np.int16)
out_best = os.path.join(outdir, f"mic_diag_{best_gain:.0f}x.wav")
with wave.open(out_best, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(boosted.tobytes())

# ── Verdict ───────────────────────────────────────────────────────

print()

dropout_warning = ""
if gaps_over_2x > 5:
    dropout_warning = (
        "\n  ⚠️  TIMING GAPS DETECTED — this is why the LLM can't hear you!\n"
        "  Audio callbacks are not firing at regular intervals.\n"
        "  Possible causes:\n"
        "    1. USB bandwidth contention (try a different USB port)\n"
        "    2. Bluetooth interference (Jabra is wireless — try wired mic)\n"
        "    3. System CPU load (close heavy apps during recording)\n"
    )

if snr_db < 10:
    verdict = "❌ BAD — mic barely picks up your voice"
elif snr_db < 20:
    verdict = "⚠️  MARGINAL — LLM may struggle"
elif best_gain == 1.0 and max_window_rms > 500:
    verdict = "✅ EXCELLENT — audio level is perfect"
elif best_gain <= 3.0:
    verdict = f"✅ GOOD — use {best_gain:.0f}x gain"
else:
    verdict = f"⚠️  NEEDS GAIN — add MIC_GAIN={best_gain:.0f} to .env"

print(f"  VERDICT: {verdict}")
print(dropout_warning)

# ── Step 3-4: Listen ──────────────────────────────────────────────

input("Step 3/4: Press Enter to play back (check for choppiness)...")
print("   Playing raw recording at 1x...")
# Play raw first (the one that matters)
raw_int16 = speech_arr.astype(np.int16)
try:
    sd.play(raw_int16, samplerate=SAMPLE_RATE)
    sd.wait()
except Exception as e:
    print(f"   Playback error: {e}")

print()
input("Step 4/4: Press Enter to play boosted version...")
print(f"   Playing at {best_gain:.0f}x gain...")
try:
    sd.play(boosted, samplerate=SAMPLE_RATE)
    sd.wait()
except Exception as e:
    print(f"   Playback error: {e}")

print()
print(f"  💾 Raw:  {out_raw}")
print(f"  💾 {best_gain:.0f}x:   {out_best}")
print()
print("  👉 Also try opening the WAV files in QuickTime / VLC.")
print("     If they sound smooth there but choppy in this script,")
print("     the playback API is at fault (NOT the recording).")
print()

if best_gain > 1.0:
    print(f"  To apply gain in realtime_demo, add to .env:")
    print(f"      MIC_GAIN={best_gain:.0f}")
    print()

if gaps_over_2x > 5:
    print("  ⚠️  To fix stutter in realtime_demo, try:")
    print("     - Use wired headphones/mic instead of Bluetooth Jabra")
    print("     - Plug Jabra USB dongle into a different USB port")
    print("     - Close CPU-heavy apps while running the demo")
    print()
