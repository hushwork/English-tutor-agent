#!/usr/bin/env python3
"""Standalone checks for camera_tutor/rtc_device.py (no pytest needed).

Run:  python3 tests/test_rtc_device.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_tutor.rtc_device import (  # noqa: E402
    MIC_READ_BYTES, RATE_MIC, RATE_SPK, RTC_RATE, RTC_FRAME_SAMPLES,
    SPK_SAMPLES_PER_TICK,
    RTCAudioManager, RTCFrameSource,
    make_resampler, pcm16_to_frame, resample_frame, upsample_24k_to_48k,
)


def test_resample_48k_to_16k():
    """Continuous 48kHz stream → ~1/3 sample count @ 16kHz.

    AudioResampler has a one-time filter delay, so check the total
    over several frames rather than per-call alignment.
    """
    resampler = make_resampler(RATE_MIC)
    total = 0
    for _ in range(10):
        samples = (np.sin(np.arange(960) / 10.0) * 10000).astype(np.int16)
        frame = pcm16_to_frame(samples.tobytes(), RTC_RATE, 0)
        total += len(np.frombuffer(resample_frame(resampler, frame), dtype=np.int16))
    assert abs(total - 3200) <= 64, f"expected ~3200 samples over 10 frames, got {total}"
    print("✓ resample 48k→16k")


def test_upsample_24k_to_48k():
    """One 20ms 24kHz tick (480 samples) → exactly 960 samples @ 48kHz."""
    samples = (np.sin(np.arange(480) / 5.0) * 10000).astype(np.int16)
    out = upsample_24k_to_48k(samples.tobytes())
    arr = np.frombuffer(out, dtype=np.int16)
    assert len(arr) == RTC_FRAME_SAMPLES, \
        f"expected {RTC_FRAME_SAMPLES} samples, got {len(arr)}"
    print("✓ upsample 24k→48k (exact)")


def test_read_mic_alignment():
    """read_mic returns exactly 200ms chunks, or None when short."""
    mgr = RTCAudioManager()
    mgr.start()

    assert mgr.read_mic() is None                       # empty
    mgr._feed_mic(b"\x01\x00" * (MIC_READ_BYTES // 4))  # half a chunk
    assert mgr.read_mic() is None                       # still short
    mgr._feed_mic(b"\x02\x00" * (MIC_READ_BYTES // 4))  # complete it
    data = mgr.read_mic()
    assert data is not None and len(data) == MIC_READ_BYTES
    assert data[:2] == b"\x01\x00" and data[-2:] == b"\x02\x00"  # FIFO order
    assert mgr.read_mic() is None                       # drained

    # Oversized feed: returns one chunk, keeps the remainder
    mgr._feed_mic(b"\x00\x00" * MIC_READ_BYTES * 2)
    assert mgr.read_mic() is not None
    rest = mgr.read_mic()
    assert rest is not None and len(rest) == MIC_READ_BYTES
    mgr.stop()
    print("✓ read_mic alignment")


def test_mic_gain_and_level():
    mgr = RTCAudioManager(mic_gain=2.0)
    mgr.start()
    mgr._feed_mic((np.ones(MIC_READ_BYTES // 2, dtype=np.int16) * 1000).tobytes())
    data = mgr.read_mic()
    arr = np.frombuffer(data, dtype=np.int16)
    assert np.all(arr == 2000), f"gain not applied: {arr[:4]}"
    assert mgr.mic_level_rms > 0
    mgr.stop()
    print("✓ mic gain & level tracking")


def test_write_spk_no_peer_dropped():
    mgr = RTCAudioManager()
    mgr.start()
    mgr.write_spk(b"\x00" * 960, visemes=[{"mouth_open": 1.0}])  # no peer
    buf, visemes = mgr._pop_spk(960)
    assert buf == b"\x00" * 960 and visemes == []     # zero-padded, chunk dropped
    mgr.stop()
    print("✓ write_spk drops without peer")


def test_spk_pop_and_visemes():
    mgr = RTCAudioManager(viseme_lead_ms=0)
    fired: list[dict] = []
    mgr.set_viseme_handler(fired.append)
    mgr.start()
    mgr._peer_connected = True

    payload = {"viseme": "A", "mouth_open": 0.8}
    chunk = b"\x05\x00" * SPK_SAMPLES_PER_TICK        # exactly one 20ms tick
    mgr.write_spk(chunk, visemes=[payload])

    buf, visemes = mgr._pop_spk(SPK_SAMPLES_PER_TICK * 2)
    assert buf == chunk, "pop returned wrong audio"
    assert visemes == [payload], "viseme not paired with its audio"

    mgr._schedule_visemes(visemes)
    deadline = time.time() + 1.0
    while not fired and time.time() < deadline:
        time.sleep(0.02)
    assert fired == [payload], f"viseme handler not fired: {fired}"

    # Underflow: next pop is zero-padded silence
    buf, visemes = mgr._pop_spk(SPK_SAMPLES_PER_TICK * 2)
    assert buf == b"\x00" * SPK_SAMPLES_PER_TICK * 2 and visemes == []
    mgr.stop()
    print("✓ spk pop + viseme scheduling")


def test_frame_source():
    src = RTCFrameSource()
    src.start()
    ok, frame = src.read_frame()
    assert not ok and frame is None

    fake = np.zeros((720, 1280, 3), dtype=np.uint8)
    with src._lock:
        src._latest = fake
    ok, frame = src.read_frame()
    assert ok and frame.shape == (720, 1280, 3)
    frame[0, 0] = 255
    assert src._latest[0, 0, 0] == 0, "read_frame must return a copy"
    src.stop()
    ok, _ = src.read_frame()
    assert not ok
    print("✓ frame source")


if __name__ == "__main__":
    test_resample_48k_to_16k()
    test_upsample_24k_to_48k()
    test_read_mic_alignment()
    test_mic_gain_and_level()
    test_write_spk_no_peer_dropped()
    test_spk_pop_and_visemes()
    test_frame_source()
    print("\nAll rtc_device checks passed ✓")
