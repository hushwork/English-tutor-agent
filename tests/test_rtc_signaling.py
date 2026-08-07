#!/usr/bin/env python3
"""Signaling endpoint check: POST /rtc/offer on the dashboard server.

Spins up the real FastAPI app on a test port with an RTCDeviceManager
registered, then runs a real aiortc offer/answer over HTTP.

Run:  python3 tests/test_rtc_signaling.py
"""

from __future__ import annotations

import asyncio
import fractions
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack  # noqa: E402

from camera_tutor.rtc_device import (  # noqa: E402
    RTC_RATE, RTC_FRAME_SAMPLES, RTCDeviceManager,
    get_rtc_manager, pcm16_to_frame, set_rtc_manager,
)

PORT = 8299
BASE = f"http://127.0.0.1:{PORT}"


class SilentMicTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._start = None
        self._pts = 0

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        else:
            self._pts += RTC_FRAME_SAMPLES
            wait = self._start + self._pts / RTC_RATE - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        return pcm16_to_frame(b"\x00" * RTC_FRAME_SAMPLES * 2, RTC_RATE, self._pts)


def start_server() -> None:
    t = threading.Thread(
        target=lambda: uvicorn.run(
            "camera_tutor.dashboard_server:app",
            host="127.0.0.1", port=PORT, log_level="error",
        ),
        daemon=True,
    )
    t.start()
    for _ in range(50):
        try:
            if httpx.get(f"{BASE}/api/health", timeout=0.5).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("dashboard did not start")


async def rtc_client() -> None:
    pc = RTCPeerConnection()
    pc.addTrack(SilentMicTrack())
    await pc.setLocalDescription(await pc.createOffer())
    offer = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    resp = await asyncio.to_thread(httpx.post, f"{BASE}/rtc/offer", json=offer, timeout=10)
    assert resp.status_code == 200, f"offer rejected: {resp.status_code} {resp.text}"
    answer = resp.json()
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    deadline = time.time() + 10
    while pc.connectionState != "connected" and time.time() < deadline:
        await asyncio.sleep(0.1)
    assert pc.connectionState == "connected", f"not connected: {pc.connectionState}"
    print("✓ /rtc/offer exchange → peer connected over HTTP signaling")
    await pc.close()


def main() -> None:
    # 409 before any manager is registered
    start_server()
    r = httpx.post(f"{BASE}/rtc/offer", json={"sdp": "x", "type": "offer"})
    assert r.status_code == 409, f"expected 409, got {r.status_code}"
    print("✓ /rtc/offer → 409 when WebRTC mode not enabled")

    # Register a manager (what agent._start_dashboard does) and retry
    # 多会话模型：audio seam 属于每个 RTCSession（offer 时创建），经 hook 启动
    manager = RTCDeviceManager()
    manager.set_session_hooks(on_created=lambda s: s.audio.start())
    set_rtc_manager(manager)
    asyncio.run(rtc_client())

    # 400 on malformed body
    r = httpx.post(f"{BASE}/rtc/offer", json={"nope": 1})
    assert r.status_code == 400, f"expected 400, got {r.status_code}"
    print("✓ /rtc/offer → 400 on malformed body")

    print("\nSignaling checks passed ✓")


if __name__ == "__main__":
    main()
