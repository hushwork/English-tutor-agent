#!/usr/bin/env python3
"""Real-browser end-to-end check for WebRTC device mode.

Drives the actual face_preview.html?device=1 page in headless Chrome
(fake mic/camera devices) against the real dashboard server, verifying
the full stack: page JS → HTTP signaling → aiortc → media both ways.

This is the closest automated approximation of a real remote device
(phone/tablet browser). Requires: playwright + system Chrome.

Run:  .venv/bin/python tests/test_rtc_browser.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402

from camera_tutor.rtc_device import (  # noqa: E402
    RTCDeviceManager, get_rtc_manager, set_rtc_manager,
)

PORT = 8298
BASE = f"http://127.0.0.1:{PORT}"


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


def wait_for(fn, what: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v:
            return v
        time.sleep(0.2)
    raise AssertionError(f"timeout waiting for: {what}")


def main() -> None:
    from playwright.sync_api import sync_playwright

    manager = RTCDeviceManager()
    # 多会话模型：audio/camera seam 属于每个 RTCSession（offer 时创建），
    # 经 session hook 启动（对应 agent._on_rtc_session_created）
    manager.set_session_hooks(
        on_created=lambda s: (s.audio.start(), s.camera.start()))
    set_rtc_manager(manager)
    start_server()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",  # fake mic tone + fake camera
                "--use-fake-ui-for-media-stream",      # auto-accept permission prompt
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page()
        page.goto(f"{BASE}/static/face_preview.html?device=1&user=tester")
        page.click("#startBtn")

        # 1. Page reports connected
        wait_for(
            lambda: "设备已连接" in (page.text_content("#status") or ""),
            "page status '🟢 设备已连接'",
        )
        assert page.get_attribute("#connDot", "class") == "dot on"
        print("✓ page connected (status + connection indicator)")

        # 2. Server side sees the peer
        session = wait_for(lambda: next(iter(manager.sessions.values()), None),
                           "RTC session registered")
        wait_for(lambda: session.audio._peer_connected, "server peer_connected")
        print("✓ server: peer connected")

        # 3. Mic uplink: fake device tone → read_mic() yields non-silent PCM
        def mic_loud():
            pcm = session.audio.read_mic()
            if not pcm:
                return False
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            return float(np.sqrt(np.mean(arr ** 2))) > 50.0
        wait_for(mic_loud, "mic uplink with audible signal")
        print(f"✓ mic uplink: browser → read_mic(), level "
              f"{session.audio.mic_level_dbfs:.1f} dBFS")

        # 4. Camera uplink: fake video → read_frame() yields frames
        ok, frame = wait_for(lambda: session.camera.read_frame()[0] and
                             session.camera.read_frame(),
                             "camera uplink frames")
        assert ok and frame is not None and frame.size > 0
        print(f"✓ camera uplink: browser → read_frame() {frame.shape}")

        # 5. TTS downlink: write_spk → browser receives audio RTP
        pcm24 = (np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)
                 * 8000).astype(np.int16).tobytes()
        for i in range(0, len(pcm24), 480 * 2 * 10):  # feed in ~200ms chunks
            session.audio.write_spk(pcm24[i:i + 480 * 2 * 10])
            time.sleep(0.05)

        stats = wait_for(lambda: page.evaluate("""async () => {
            const s = await window._rtcPC.getStats();
            const r = {inAudio: 0, outAudio: 0, outVideoFrames: 0};
            s.forEach(x => {
              if (x.type === 'inbound-rtp' && x.kind === 'audio') r.inAudio = x.bytesReceived || 0;
              if (x.type === 'outbound-rtp' && x.kind === 'audio') r.outAudio = x.bytesSent || 0;
              if (x.type === 'outbound-rtp' && x.kind === 'video') r.outVideoFrames = x.framesSent || 0;
            });
            return r.inAudio > 0 && r.outAudio > 0 && r.outVideoFrames > 0 ? r : null;
        }"""), "browser RTP stats (audio in/out + video out)")
        print(f"✓ browser RTP stats: {stats}")

        browser.close()

    print("\nReal-browser e2e checks passed ✓")


if __name__ == "__main__":
    main()
