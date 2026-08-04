#!/usr/bin/env python3
"""Loopback integration check for WebRTC device mode.

Simulates the browser side with a second in-process RTCPeerConnection
(dummy sine mic + green-screen camera), runs the real offer/answer
against RTCDeviceManager, and verifies all four seams end to end:

  dummy mic  ──► manager.audio.read_mic()   (16kHz PCM out)
  TTS PCM    ──► browser-side audio track    (48kHz frames received)
  dummy cam  ──► manager.camera.read_frame() (BGR frames)
  visemes    ──► handler fired alongside outbound audio

Run:  python3 tests/test_rtc_loopback.py
"""

from __future__ import annotations

import asyncio
import fractions
import sys
import time
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack  # noqa: E402

from camera_tutor.rtc_device import (  # noqa: E402
    MIC_READ_BYTES, RTC_RATE, RTC_FRAME_SAMPLES,
    RTCDeviceManager, pcm16_to_frame,
)


class DummyMicTrack(AudioStreamTrack):
    """Browser-side mic: 48kHz sine, 20ms frames."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._start = None
        self._pts = 0
        self._phase = 0

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        else:
            self._pts += RTC_FRAME_SAMPLES
            wait = self._start + self._pts / RTC_RATE - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        t = (np.arange(RTC_FRAME_SAMPLES) + self._phase) * 440 / RTC_RATE
        self._phase += RTC_FRAME_SAMPLES
        samples = (np.sin(2 * np.pi * t) * 8000).astype(np.int16)
        return pcm16_to_frame(samples.tobytes(), RTC_RATE, self._pts)


class DummyCamTrack(VideoStreamTrack):
    """Browser-side camera: 30fps green frames."""

    kind = "video"

    def __init__(self):
        super().__init__()
        self._pts = 0
        self._start = None

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        else:
            self._pts += 90000 // 30  # 90kHz clock
            wait = self._start + self._pts / 90000 - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 1] = 200
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, 90000)
        return frame


async def main() -> None:
    manager = RTCDeviceManager()
    manager.audio.start()
    manager.camera.start()

    visemes_fired: list[dict] = []
    manager.audio.set_viseme_handler(visemes_fired.append)

    browser = RTCPeerConnection()
    browser.addTrack(DummyMicTrack())
    browser.addTrack(DummyCamTrack())

    audio_frames_received = 0

    @browser.on("track")
    def on_track(track):
        nonlocal audio_frames_received
        if track.kind != "audio":
            return

        async def drain():
            nonlocal audio_frames_received
            try:
                while True:
                    frame = await track.recv()
                    assert frame.sample_rate == RTC_RATE
                    audio_frames_received += 1
            except Exception:
                pass

        asyncio.ensure_future(drain())

    # Real offer/answer exchange (what face_preview.html + /rtc/offer do)
    await browser.setLocalDescription(await browser.createOffer())
    answer = await manager.handle_offer(
        browser.localDescription.sdp, browser.localDescription.type)
    await browser.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    # Let ICE/DTLS connect and tracks start flowing
    deadline = time.time() + 10
    while browser.connectionState != "connected" and time.time() < deadline:
        await asyncio.sleep(0.1)
    assert browser.connectionState == "connected", \
        f"peer not connected: {browser.connectionState}"
    print("✓ peer connected")

    # 1. Mic uplink: dummy sine → read_mic()
    deadline = time.time() + 5
    mic_data = None
    while mic_data is None and time.time() < deadline:
        mic_data = manager.audio.read_mic()
        await asyncio.sleep(0.05)
    assert mic_data is not None and len(mic_data) == MIC_READ_BYTES
    assert manager.audio.mic_level_rms > 100, "mic signal too quiet"
    print("✓ mic uplink (16kHz PCM flowing)")

    # 2. TTS downlink + 3. visemes: write_spk → browser audio track
    pcm24 = (np.sin(np.arange(4800) / 5.0) * 5000).astype(np.int16).tobytes()
    manager.audio.write_spk(pcm24, visemes=[{"viseme": "A", "mouth_open": 0.9}])
    deadline = time.time() + 5
    while (audio_frames_received < 5 or not visemes_fired) and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert audio_frames_received >= 5, f"browser got {audio_frames_received} frames"
    print(f"✓ TTS downlink ({audio_frames_received} frames @ 48kHz)")
    assert visemes_fired and visemes_fired[0]["viseme"] == "A"
    print("✓ viseme fired with outbound audio")

    # 4. Camera uplink: dummy frames → read_frame()
    deadline = time.time() + 5
    ok, frame = manager.camera.read_frame()
    while not ok and time.time() < deadline:
        await asyncio.sleep(0.1)
        ok, frame = manager.camera.read_frame()
    assert ok and frame is not None and frame.shape == (480, 640, 3)
    # Video goes through a lossy codec (VP8/H.264) — allow tolerance,
    # just check the green channel dominates.
    g = int(frame[0, 0, 1])
    assert 150 <= g <= 255 and g > int(frame[0, 0, 0]) + 50, \
        f"camera frame content wrong (B,G,R={frame[0,0].tolist()})"
    print("✓ camera uplink (BGR frames)")

    await browser.close()
    await manager.shutdown()
    manager.audio.stop()
    manager.camera.stop()
    print("\nWebRTC loopback: all checks passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
