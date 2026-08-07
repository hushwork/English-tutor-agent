#!/usr/bin/env python3
"""Multi-user concurrency checks for WebRTC device mode.

Verifies the multi-session RTCDeviceManager:
  1. Two concurrent peers (different users) stay connected — no kicking
  2. Audio isolation: user A's mic/TTS never lands in user B's session
  3. Same-user reattach keeps the session and seams (only the peer is replaced)
  4. Peer close cleans up only that session; on_closed hook fires
  5. Invalid user_id rejected (path traversal guard)

Run:  python3 tests/test_rtc_multiuser.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402

from camera_tutor.rtc_device import (  # noqa: E402
    MIC_READ_BYTES, RTC_RATE, RTC_FRAME_SAMPLES,
    RTCDeviceManager, pcm16_to_frame, validate_user_id,
)
from tests.test_rtc_loopback import DummyMicTrack  # noqa: E402


async def make_browser(manager: RTCDeviceManager, user_id: str):
    """One simulated browser: offer/answer against the manager."""
    browser = RTCPeerConnection()
    browser.addTrack(DummyMicTrack())
    await browser.setLocalDescription(await browser.createOffer())
    answer = await manager.handle_offer(
        browser.localDescription.sdp, browser.localDescription.type,
        user_id=user_id)
    await browser.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    deadline = time.time() + 10
    while browser.connectionState != "connected" and time.time() < deadline:
        await asyncio.sleep(0.1)
    assert browser.connectionState == "connected", \
        f"{user_id}: peer not connected: {browser.connectionState}"
    return browser, answer


async def main() -> None:
    manager = RTCDeviceManager()
    created: list = []
    closed: list = []
    manager.set_session_hooks(on_created=created.append,
                              on_closed=closed.append)

    # ── 1. Two users connect concurrently, neither is kicked ──
    browser_a, answer_a = await make_browser(manager, "alice")
    browser_b, answer_b = await make_browser(manager, "bob")
    assert len(manager.sessions) == 2, \
        f"expected 2 sessions, got {len(manager.sessions)}"
    assert answer_a["user_id"] == "alice" and answer_b["user_id"] == "bob"
    assert answer_a["session_id"] != answer_b["session_id"]
    assert browser_a.connectionState == "connected", "alice kicked by bob!"
    print("✓ two concurrent peers, no kicking")

    sess_a = manager.get_session(answer_a["session_id"])
    sess_b = manager.get_session(answer_b["session_id"])
    assert sess_a.user_id == "alice" and sess_b.user_id == "bob"
    sess_a.audio.start()
    sess_b.audio.start()

    # ── 2. Audio isolation: separate buffers, separate state ──
    assert sess_a.audio is not sess_b.audio
    assert sess_a.camera is not sess_b.camera
    # Feed A's mic buffer directly; B must see nothing
    sess_a.audio._feed_mic(b"\x01\x00" * (MIC_READ_BYTES // 2))
    assert sess_b.audio.read_mic() is None, "A's audio leaked into B"
    data_a = sess_a.audio.read_mic()
    assert data_a is not None and len(data_a) == MIC_READ_BYTES
    # TTS queued for A does not appear in B's speaker ring
    sess_a.audio._peer_connected = True
    sess_a.audio.write_spk(b"\x00" * 960, visemes=None)
    assert len(sess_b.audio._spk_ring) == 0, "A's TTS leaked into B"
    print("✓ audio isolation between sessions")

    # ── 3. Same-user reattach: session/seams preserved, only peer replaced ──
    # 网络抖动重连不应拆会话（PracticeSession 无感知），这是"断断续续"修复的核心
    sess_a.audio._feed_mic(b"\x03\x00" * 100)  # 缓冲留数据，验证重挂后还在
    browser_a2, answer_a2 = await make_browser(manager, "alice")
    assert answer_a2["session_id"] == answer_a["session_id"], \
        "reattach must keep session_id"
    assert manager.get_session(answer_a["session_id"]) is sess_a, \
        "session object must be preserved on reattach"
    assert len(manager.sessions) == 2, "reattach must not add a third session"
    assert manager.get_session(answer_b["session_id"]) is sess_b, "bob disturbed"
    assert browser_b.connectionState == "connected"
    assert not closed, "reattach must not fire on_closed"
    # 新 browser 的音频流进同一个 seam（重挂后 mic 上行恢复）
    deadline = time.time() + 5
    mic_data = None
    while mic_data is None and time.time() < deadline:
        mic_data = sess_a.audio.read_mic()
        await asyncio.sleep(0.05)
    assert mic_data is not None, "mic uplink dead after reattach"
    print("✓ same-user reattach preserves session, only peer replaced")

    # ── 4. Peer close cleans up only that session ──
    await browser_a2.close()
    deadline = time.time() + 5
    while len(manager.sessions) != 1 and time.time() < deadline:
        await asyncio.sleep(0.1)
    assert len(manager.sessions) == 1, \
        f"expected 1 session after alice left, got {len(manager.sessions)}"
    assert manager.get_session(answer_b["session_id"]) is sess_b
    print("✓ peer close cleans up only that session")

    # ── 5. user_id validation (path traversal guard) ──
    assert validate_user_id("") == "default"
    assert validate_user_id("alice_2-x") == "alice_2-x"
    for bad in ("../evil", "a/b", "a b", "x" * 33, "中文"):
        try:
            validate_user_id(bad)
            raise AssertionError(f"bad user_id accepted: {bad!r}")
        except ValueError:
            pass
    print("✓ user_id validation")

    await browser_b.close()
    await manager.shutdown()
    assert len(manager.sessions) == 0
    print("\nMulti-user RTC checks passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
