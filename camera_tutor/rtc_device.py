"""RTC device mode — WebRTC-based remote A/V source/sink.

Replaces local sounddevice/cv2 hardware access with a WebRTC peer
(typically face_preview.html?device=1 in a browser on another machine).

Seam classes (duck-typed replacements used by CameraTutorAgent):
- RTCAudioManager — same public interface as AudioManager:
  read_mic() -> 16kHz PCM16 bytes (~200ms) / write_spk(24kHz PCM16, visemes)
- RTCFrameSource  — same read_frame() interface as CameraPipeline:
  read_frame() -> (ok, BGR ndarray | None)

Threading model:
- The RTCPeerConnection and all track I/O live in the uvicorn event loop
  (the offer endpoint runs there). Agent threads only touch plain
  deques/queues protected by locks — no cross-loop calls.

Viseme timing: local mode fires visemes when PortAudio plays the chunk.
Here they fire when the audio is handed to aiortc for sending, plus a
fixed lead (viseme_lead_ms) to cover the browser jitter buffer.
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Optional

import av
import numpy as np
from aiortc import (
    RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription,
)
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

from camera_tutor.audio_manager import (
    RATE_MIC, RATE_SPK, MIC_READ_FRAMES, DEFAULT_MIC_GAIN,
)

logger = logging.getLogger(__name__)

# WebRTC audio is always 48kHz; aiortc expects 20ms frames.
RTC_RATE = 48000
RTC_FRAME_SAMPLES = 960                     # 20ms @ 48kHz
SPK_SAMPLES_PER_TICK = 480                  # 20ms @ 24kHz (resampled to 960)
MIC_READ_BYTES = MIC_READ_FRAMES * 2        # 200ms @ 16kHz int16

DEFAULT_VISME_LEAD_MS = 80


def ice_servers() -> list[dict]:
    """ICE server 配置（env 驱动），dict 形式，aiortc 和浏览器端（/rtc/config）
    共用同一份。LAN 模式默认空（仅 host candidate）；公网访问时服务器藏在
    NAT 后，host candidate 不可达，必须配 TURN 中转：

      RTC_TURN_URL=turn:1.2.3.4:3478  RTC_TURN_USER=...  RTC_TURN_PASS=...
      RTC_STUN_URL=stun:1.2.3.4:3478  （可选，一般不必）
    """
    servers: list[dict] = []
    stun = os.environ.get("RTC_STUN_URL", "")
    if stun:
        servers.append({"urls": [stun]})
    turn = os.environ.get("RTC_TURN_URL", "")
    if turn:
        servers.append({
            "urls": [turn],
            "username": os.environ.get("RTC_TURN_USER", ""),
            "credential": os.environ.get("RTC_TURN_PASS", ""),
        })
    return servers


def _rtc_configuration() -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[RTCIceServer(**s) for s in ice_servers()])


def browser_ice_servers() -> list[dict]:
    """给浏览器的 ICE 配置：TURN 若只配了 TCP（本机网络 UDP 被封的无奈之举），
    给浏览器补上 UDP 变体——对端网络通常不封 UDP，UDP 优先、TCP 兜底。"""
    servers = []
    for s in ice_servers():
        s = dict(s)
        s["urls"] = urls = list(s["urls"])
        for u in urls:
            if u.startswith("turn:") and "transport=tcp" in u:
                s["urls"] = [u.split("?")[0], u]
                break
        servers.append(s)
    return servers


# ── Resampling helpers ────────────────────────────────────────────

def make_resampler(dst_rate: int) -> av.AudioResampler:
    """Resampler to 16-bit mono PCM at dst_rate (stateful — keep per stream)."""
    return av.AudioResampler(format="s16", layout="mono", rate=dst_rate)


def resample_frame(resampler: av.AudioResampler, frame: av.AudioFrame) -> bytes:
    """Resample an incoming AudioFrame, return s16 mono PCM bytes."""
    return b"".join(f.to_ndarray().tobytes() for f in resampler.resample(frame))


def pcm16_to_frame(pcm: bytes, rate: int, pts: int) -> av.AudioFrame:
    """Wrap s16 mono PCM bytes in an AudioFrame with the given pts."""
    arr = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    frame.sample_rate = rate
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, rate)
    return frame


def upsample_24k_to_48k(pcm: bytes) -> bytes:
    """Deterministic 2x linear upsample (same trick as AudioManager).

    Used on the outbound path where every 20ms tick must yield exactly
    960 samples — av.AudioResampler has filter latency and can't
    guarantee per-call alignment.
    """
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    up = np.interp(np.linspace(0, len(x) - 1, len(x) * 2), np.arange(len(x)), x)
    return up.astype(np.int16).tobytes()


# ── RTCAudioManager ───────────────────────────────────────────────

class RTCAudioManager:
    """AudioManager-compatible audio I/O backed by a WebRTC peer.

    Mic:  remote audio track → resampled to 16kHz → read_mic()
    Spk:  write_spk() → ring → _TTSOutTrack.recv() → resampled to 48kHz
    Visemes: fired at send time + viseme_lead_ms, via the registered handler.
    """

    def __init__(
        self,
        mic_gain: float = DEFAULT_MIC_GAIN,
        viseme_lead_ms: int = DEFAULT_VISME_LEAD_MS,
    ):
        self._started = False
        self._mic_gain = mic_gain
        self._viseme_lead_ms = viseme_lead_ms

        # Mic: remote track → pending bytes → read_mic()
        self._mic_buf = bytearray()
        self._mic_lock = threading.Lock()
        self._mic_buf_maxlen = MIC_READ_BYTES * 25   # ~5s
        self._mic_overflow = 0

        # Speaker: write_spk() → ring → out track recv()
        self._spk_ring: deque[tuple[bytes, list[dict] | None]] = deque()
        self._spk_ring_maxlen = 200
        self._spk_lock = threading.Lock()
        self._spk_underflow = 0
        self._spk_dropped_no_peer = 0

        # Viseme outbox: (due_epoch, payload) — drained by a thread
        self._viseme_outbox: deque[tuple[float, dict]] = deque()
        self._viseme_outbox_lock = threading.Lock()
        self._viseme_handler: Callable[[dict], None] | None = None
        self._viseme_drain_thread: threading.Thread | None = None
        self._viseme_drain_stop = threading.Event()

        # Peer plumbing
        self._mic_task: Optional[asyncio.Task] = None
        self._peer_connected = False

        # Level tracking
        self._level_samples: list[float] = []

        # Debug: dump exactly what read_mic() returns (raw s16 mono 16kHz)
        import os
        self._dump_path = os.environ.get("RTC_MIC_DUMP", "")
        self._dump_file = open(self._dump_path, "ab") if self._dump_path else None

    # ── Viseme handler ───────────────────────────────────────────

    def set_viseme_handler(self, handler: Callable[[dict], None]) -> None:
        """Register callback fired when a viseme is due (drain thread)."""
        self._viseme_handler = handler

    def _start_viseme_drain(self) -> None:
        if self._viseme_drain_thread is not None:
            return
        self._viseme_drain_stop.clear()
        self._viseme_drain_thread = threading.Thread(
            target=self._viseme_drain_loop, name="rtc-viseme-drain", daemon=True,
        )
        self._viseme_drain_thread.start()

    def _stop_viseme_drain(self) -> None:
        self._viseme_drain_stop.set()
        if self._viseme_drain_thread:
            self._viseme_drain_thread.join(timeout=1.0)
            self._viseme_drain_thread = None

    def _viseme_drain_loop(self) -> None:
        while not self._viseme_drain_stop.is_set():
            handler = self._viseme_handler
            if handler is None:
                time.sleep(0.1)
                continue
            now = time.time()
            due: list[dict] = []
            with self._viseme_outbox_lock:
                while self._viseme_outbox and self._viseme_outbox[0][0] <= now:
                    due.append(self._viseme_outbox.popleft()[1])
            for p in due:
                try:
                    handler(p)
                except Exception:
                    pass
            time.sleep(0.02)

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        with self._mic_lock:
            self._mic_buf.clear()
            self._mic_overflow = 0
        with self._spk_lock:
            self._spk_ring.clear()
            self._spk_underflow = 0
        with self._viseme_outbox_lock:
            self._viseme_outbox.clear()
        self._start_viseme_drain()
        self._started = True
        logger.info("RTC audio ready (mic=%dHz, spk=%dHz over WebRTC %dHz)",
                    RATE_MIC, RATE_SPK, RTC_RATE)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_viseme_drain()
        if self._dump_file is not None:
            self._dump_file.close()
            self._dump_file = None
        task = self._mic_task
        self._mic_task = None
        if task is not None:
            task.cancel()

    # ── Mic path (remote track → read_mic) ───────────────────────

    def attach_mic_track(self, track) -> None:
        """Start consuming the peer's audio track (call from the uvicorn loop)."""
        old = self._mic_task
        if old is not None:
            old.cancel()
        self._mic_task = asyncio.ensure_future(self._consume_mic(track))
        self.notify_peer_connected()
        logger.info("RTC mic track attached")

    async def _consume_mic(self, track) -> None:
        resampler = make_resampler(RATE_MIC)
        try:
            while True:
                frame = await track.recv()
                self._feed_mic(resample_frame(resampler, frame))
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.warning("RTC mic consume error: %s", e)
        finally:
            # 仅当本任务仍是当前 mic 消费者时才清除连接标志——
            # 否则旧 peer 的收尾会覆盖新连接的状态（双重 offer 竞态，
            # 症状：mic 正常但 write_spk 全部丢弃，对端无声）
            if self._mic_task is asyncio.current_task():
                self._peer_connected = False
            logger.info("RTC mic track ended")

    def _feed_mic(self, pcm: bytes) -> None:
        """Append 16kHz PCM16 bytes to the mic buffer (thread-safe)."""
        if not pcm:
            return
        with self._mic_lock:
            if len(self._mic_buf) >= self._mic_buf_maxlen:
                self._mic_overflow += 1
                del self._mic_buf[:len(pcm)]
            self._mic_buf.extend(pcm)

    def read_mic(self) -> Optional[bytes]:
        """Return ~200ms of 16kHz PCM16, or None if not enough buffered."""
        if not self._started:
            return None
        try:
            with self._mic_lock:
                if len(self._mic_buf) < MIC_READ_BYTES:
                    return None
                raw = bytes(self._mic_buf[:MIC_READ_BYTES])
                del self._mic_buf[:MIC_READ_BYTES]
                overflow = self._mic_overflow
                self._mic_overflow = 0

            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if self._mic_gain != 1.0:
                arr *= self._mic_gain
                np.clip(arr, -32767, 32767, out=arr)
            self._track_level(arr)
            if overflow:
                logger.warning("RTC mic buffer overflow x%d", overflow)
            out = arr.astype(np.int16).tobytes()
            if self._dump_file is not None:
                self._dump_file.write(out)
                self._dump_file.flush()
            return out
        except Exception as e:
            logger.warning("RTC mic read error: %s", e)
            return None

    def _track_level(self, samples: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self._level_samples.append(rms)
        max_samples = int(5.0 / (MIC_READ_FRAMES / RATE_MIC))
        if len(self._level_samples) > max_samples:
            self._level_samples = self._level_samples[-max_samples:]

    # ── Speaker path (write_spk → out track) ─────────────────────

    def write_spk(self, data: bytes, visemes: list[dict] | None = None) -> None:
        """Queue 24kHz PCM16 (+ visemes) for the WebRTC out track."""
        if not self._started:
            return
        if not self._peer_connected:
            # No browser attached — drop, but don't flood the log.
            self._spk_dropped_no_peer += 1
            if self._spk_dropped_no_peer == 1:
                logger.warning("RTC: no peer connected — dropping outbound audio")
            return
        with self._spk_lock:
            if len(self._spk_ring) >= self._spk_ring_maxlen:
                self._spk_ring.popleft()
            self._spk_ring.append((data, visemes))

    def _pop_spk(self, needed: int) -> tuple[bytes, list[dict]]:
        """Pull exactly `needed` bytes of 24kHz PCM for the out track.

        Zero-pads on underflow (keeps the WebRTC clock steady).
        Returns (audio_bytes, visemes_due). Called from the uvicorn loop.
        """
        parts: list[bytes] = []
        total = 0
        visemes_due: list[dict] = []
        with self._spk_lock:
            while self._spk_ring and total < needed:
                chunk, visemes = self._spk_ring.popleft()
                parts.append(chunk)
                total += len(chunk)
                if visemes:
                    visemes_due.extend(visemes)
            buf = b"".join(parts)
            if len(buf) > needed:
                self._spk_ring.appendleft((buf[needed:], None))
                buf = buf[:needed]
            if len(buf) < needed:
                self._spk_underflow += 1
                buf = buf + b"\x00" * (needed - len(buf))
        return buf, visemes_due

    def _schedule_visemes(self, visemes: list[dict]) -> None:
        """Queue visemes to fire after the lead time (browser playout delay)."""
        if not visemes or self._viseme_handler is None:
            return
        due = time.time() + self._viseme_lead_ms / 1000.0
        with self._viseme_outbox_lock:
            for p in visemes:
                self._viseme_outbox.append((due, p))

    def create_out_track(self) -> "_TTSOutTrack":
        """New outbound audio track for a peer (one per peer connection)."""
        return _TTSOutTrack(self)

    def notify_peer_connected(self) -> None:
        self._peer_connected = True
        self._spk_dropped_no_peer = 0

    def notify_peer_disconnected(self) -> None:
        self._peer_connected = False
        self._spk_dropped_no_peer = 0

    # ── Properties (mirror AudioManager) ─────────────────────────

    @property
    def mic_level_rms(self) -> float:
        if not self._level_samples:
            return 0.0
        return sum(self._level_samples) / len(self._level_samples)

    @property
    def mic_level_dbfs(self) -> float:
        rms = self.mic_level_rms
        if rms < 1e-6:
            return -96.0
        return 20 * np.log10(rms / 32767.0)

    @property
    def mic_gain(self) -> float:
        return self._mic_gain

    @mic_gain.setter
    def mic_gain(self, value: float) -> None:
        self._mic_gain = max(0.1, min(10.0, value))

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def mic_attached(self) -> bool:
        """是否有远端 mic track 在消费（供 dashboard 设备状态用）。"""
        return self._mic_task is not None

    @property
    def peer_connected(self) -> bool:
        return self._peer_connected


class _TTSOutTrack(AudioStreamTrack):
    """aiortc outbound audio track: pulls TTS PCM from RTCAudioManager.

    recv() is paced by aiortc in real time (20ms ticks); each tick pops
    480 samples of 24kHz PCM and upsamples to exactly 960 @ 48kHz.
    """

    kind = "audio"

    def __init__(self, manager: RTCAudioManager):
        super().__init__()
        self._manager = manager
        self._start: Optional[float] = None
        self._pts = 0

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        # Real-time pacing (standard aiortc custom-track pattern)
        if self._start is None:
            self._start = time.time()
            self._pts = 0
        else:
            self._pts += RTC_FRAME_SAMPLES
            wait = self._start + (self._pts / RTC_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        pcm24, visemes = self._manager._pop_spk(SPK_SAMPLES_PER_TICK * 2)
        self._manager._schedule_visemes(visemes)

        pcm48 = upsample_24k_to_48k(pcm24)
        return pcm16_to_frame(pcm48, RTC_RATE, self._pts)


# ── RTCFrameSource ────────────────────────────────────────────────

class RTCFrameSource:
    """CameraPipeline-compatible frame source backed by a WebRTC video track.

    A consumer task (uvicorn loop) keeps only the latest BGR frame;
    VisionManager pulls it via read_frame().
    """

    # Min interval between frame conversions (recv always runs to drain
    # the decoder; to_ndarray is throttled to save CPU).
    CONVERT_INTERVAL = 0.2

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # CameraPipeline-compatible lifecycle (no-op hardware to open)
    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
        with self._lock:
            self._latest = None

    def attach_track(self, track) -> None:
        """Start consuming the peer's video track (call from the uvicorn loop)."""
        old = self._task
        if old is not None:
            old.cancel()
        self._task = asyncio.ensure_future(self._consume(track))
        logger.info("RTC camera track attached")

    async def _consume(self, track) -> None:
        last_convert = 0.0
        try:
            while True:
                frame = await track.recv()
                now = time.time()
                if now - last_convert < self.CONVERT_INTERVAL:
                    continue
                last_convert = now
                arr = frame.to_ndarray(format="bgr24")
                with self._lock:
                    self._latest = arr
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.warning("RTC camera consume error: %s", e)
        finally:
            # 清掉残留的最后一帧：否则 peer 断开后 read_frame 仍返回陈旧帧，
            # vision reader 会对着静止画面全速空转编码（CPU 满载）
            with self._lock:
                self._latest = None
            logger.info("RTC camera track ended")

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """CameraPipeline.read_frame-compatible: (ok, latest BGR frame)."""
        if not self._running:
            return False, None
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    @property
    def camera_attached(self) -> bool:
        """是否有远端视频轨在消费（供 dashboard 设备状态用）。"""
        return self._task is not None


# ── RTCSession / RTCDeviceManager ─────────────────────────────────

import re
import uuid

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
DEFAULT_USER_ID = "default"


def validate_user_id(user_id: str) -> str:
    """校验浏览器传来的 user_id（用作数据目录名，必须防路径穿越）。

    合法: 1-32 位字母/数字/下划线/连字符。空串回退为 "default"。
    非法时抛 ValueError，由信令端点转成 400。
    """
    if not user_id:
        return DEFAULT_USER_ID
    if not USER_ID_RE.match(user_id):
        raise ValueError(f"invalid user_id: {user_id!r}")
    return user_id


class RTCSession:
    """一路 WebRTC peer 连接及其独立的 A/V seam 对象。

    每个并发用户（浏览器）一个实例：自己的 RTCAudioManager（麦克风缓冲、
    扬声器环形队列、viseme 队列）和 RTCFrameSource（最新帧），互不共享。
    """

    def __init__(self, session_id: str, user_id: str,
                 mic_gain: float = DEFAULT_MIC_GAIN,
                 viseme_lead_ms: int = DEFAULT_VISME_LEAD_MS):
        self.session_id = session_id
        self.user_id = user_id
        self.audio = RTCAudioManager(mic_gain=mic_gain,
                                     viseme_lead_ms=viseme_lead_ms)
        self.camera = RTCFrameSource()
        self._pc: Optional[RTCPeerConnection] = None


class RTCDeviceManager:
    """并发 WebRTC 会话注册表 —— 每路浏览器连接一个 RTCSession。

    多用户并发：新用户的 offer 创建新 session，互不影响。
    同一 user_id 重复 offer 走重挂路径（见 handle_offer）：复用该用户的
    seam 对象，仅替换 peer connection，上层 PracticeSession 无感知。
    """

    def __init__(self, mic_gain: float = DEFAULT_MIC_GAIN,
                 viseme_lead_ms: int = DEFAULT_VISME_LEAD_MS):
        self._mic_gain = mic_gain
        self._viseme_lead_ms = viseme_lead_ms
        self._sessions: dict[str, RTCSession] = {}
        self._offer_lock: Optional[asyncio.Lock] = None
        # 会话生命周期钩子（agent 层挂接，同步调用，运行在 uvicorn 线程）
        self._on_session_created: Optional[Callable[[RTCSession], None]] = None
        self._on_session_closed: Optional[Callable[[RTCSession], None]] = None

    def set_session_hooks(
        self,
        on_created: Optional[Callable[[RTCSession], None]] = None,
        on_closed: Optional[Callable[[RTCSession], None]] = None,
    ) -> None:
        """注册会话创建/关闭回调（每路 peer 各触发一次）。"""
        self._on_session_created = on_created
        self._on_session_closed = on_closed

    @property
    def sessions(self) -> dict[str, RTCSession]:
        """当前活跃会话的快照 {session_id: RTCSession}。"""
        return dict(self._sessions)

    def get_session(self, session_id: str) -> Optional[RTCSession]:
        return self._sessions.get(session_id)

    async def handle_offer(self, sdp: str, offer_type: str,
                           user_id: str = "") -> dict:
        """Handle a browser SDP offer; return the answer as a dict.

        Runs in the uvicorn event loop (called from the FastAPI endpoint).
        新用户创建独立 RTCSession；同一 user_id 重复 offer 走重挂路径
        （复用 seam 对象，只换 peer connection）。返回值带 session_id/
        user_id（浏览器可忽略多余字段，兼容旧版前端）。
        """
        user_id = validate_user_id(user_id)
        if self._offer_lock is None:
            self._offer_lock = asyncio.Lock()
        async with self._offer_lock:
            # 同一用户重复连接（网络抖动重连/页面刷新/第二个标签页）：
            # 保留 seam 对象与上层 PracticeSession，只替换 peer connection——
            # 对话上下文、模型 WS、TTS 队列、viseme 状态全部不受影响。
            # （旧行为是整体拆掉重建，公网 ICE 抖动时表现为"对话断断续续"）
            session = next(
                (s for s in self._sessions.values() if s.user_id == user_id),
                None)
            is_reattach = session is not None
            if is_reattach:
                logger.info("RTC: user %s re-attaching session %s (new peer)",
                            user_id, session.session_id)
                old_pc, session._pc = session._pc, None
                if old_pc is not None:
                    try:
                        await old_pc.close()
                    except Exception as e:
                        logger.debug("RTC old peer close error: %s", e)
            else:
                session = RTCSession(
                    session_id=uuid.uuid4().hex[:12],
                    user_id=user_id,
                    mic_gain=self._mic_gain,
                    viseme_lead_ms=self._viseme_lead_ms,
                )
                self._sessions[session.session_id] = session

            pc = RTCPeerConnection(configuration=_rtc_configuration())
            session._pc = pc

            @pc.on("track")
            def on_track(track):
                if track.kind == "audio":
                    session.audio.attach_mic_track(track)
                elif track.kind == "video":
                    session.camera.attach_track(track)

            @pc.on("connectionstatechange")
            async def on_state():
                logger.info("RTC [%s/%s] state: %s",
                            session.user_id, session.session_id,
                            pc.connectionState)
                if session._pc is not pc:
                    return  # 被替换掉的旧 peer 的迟到事件，不影响当前连接
                if self._sessions.get(session.session_id) is not session:
                    return  # 旧 session 的迟到事件，不影响新 session 状态
                if pc.connectionState == "connected":
                    session.audio.notify_peer_connected()
                elif pc.connectionState in ("failed", "closed", "disconnected"):
                    session.audio.notify_peer_disconnected()
                    if pc.connectionState in ("failed", "closed"):
                        await self._close_session(session)

            pc.addTrack(session.audio.create_out_track())

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            logger.info("RTC peer established: user=%s session=%s (%d active)%s",
                        user_id, session.session_id, len(self._sessions),
                        " [reattach]" if is_reattach else "")

        if not is_reattach and self._on_session_created is not None:
            try:
                self._on_session_created(session)
            except Exception:
                logger.exception("on_session_created hook failed")
        return {"sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "session_id": session.session_id,
                "user_id": user_id}

    async def _close_session(self, session: RTCSession) -> None:
        """关闭并注销一个 session（幂等）。"""
        if self._sessions.pop(session.session_id, None) is None:
            return  # 已关闭
        pc, session._pc = session._pc, None
        if pc is not None:
            try:
                await pc.close()
            except Exception as e:
                logger.debug("RTC close error: %s", e)
        session.audio.stop()
        session.camera.stop()
        logger.info("RTC session closed: user=%s session=%s (%d active)",
                    session.user_id, session.session_id, len(self._sessions))
        if self._on_session_closed is not None:
            try:
                self._on_session_closed(session)
            except Exception:
                logger.exception("on_session_closed hook failed")

    async def close_user_session(self, user_id: str) -> bool:
        """按 user_id 关闭会话（如 agent 层要求下线）。返回是否关到了。"""
        for s in list(self._sessions.values()):
            if s.user_id == user_id:
                await self._close_session(s)
                return True
        return False

    async def shutdown(self) -> None:
        for s in list(self._sessions.values()):
            await self._close_session(s)


# ── Module-level registry (dashboard ↔ agent, same process) ──────

_manager: Optional[RTCDeviceManager] = None


def set_rtc_manager(manager: Optional[RTCDeviceManager]) -> None:
    global _manager
    _manager = manager


def get_rtc_manager() -> Optional[RTCDeviceManager]:
    return _manager
