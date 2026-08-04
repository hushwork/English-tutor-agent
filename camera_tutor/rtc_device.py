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
import threading
import time
from collections import deque
from typing import Callable, Optional

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
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
        self._peer_connected = True
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
            return arr.astype(np.int16).tobytes()
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
            logger.info("RTC camera track ended")

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """CameraPipeline.read_frame-compatible: (ok, latest BGR frame)."""
        if not self._running:
            return False, None
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()


# ── RTCDeviceManager ──────────────────────────────────────────────

class RTCDeviceManager:
    """Owns the WebRTC peer connection and the A/V seam objects.

    One active peer at a time: a new offer replaces the old connection.
    """

    def __init__(self, mic_gain: float = DEFAULT_MIC_GAIN,
                 viseme_lead_ms: int = DEFAULT_VISME_LEAD_MS):
        self.audio = RTCAudioManager(mic_gain=mic_gain, viseme_lead_ms=viseme_lead_ms)
        self.camera = RTCFrameSource()
        self._pc: Optional[RTCPeerConnection] = None
        self._offer_lock: Optional[asyncio.Lock] = None

    async def handle_offer(self, sdp: str, offer_type: str) -> dict:
        """Handle a browser SDP offer; return the answer as a dict.

        Runs in the uvicorn event loop (called from the FastAPI endpoint).
        """
        if self._offer_lock is None:
            self._offer_lock = asyncio.Lock()
        async with self._offer_lock:
            if self._pc is not None:
                logger.info("RTC: new offer — closing previous peer")
                await self._close_pc()

            pc = RTCPeerConnection()
            self._pc = pc

            @pc.on("track")
            def on_track(track):
                if track.kind == "audio":
                    self.audio.attach_mic_track(track)
                elif track.kind == "video":
                    self.camera.attach_track(track)

            @pc.on("connectionstatechange")
            async def on_state():
                logger.info("RTC connection state: %s", pc.connectionState)
                if pc.connectionState in ("failed", "closed", "disconnected"):
                    self.audio.notify_peer_disconnected()
                    if pc.connectionState in ("failed", "closed"):
                        await self._close_pc()

            pc.addTrack(self.audio.create_out_track())

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            logger.info("RTC peer established")
            return {"sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type}

    async def _close_pc(self) -> None:
        pc, self._pc = self._pc, None
        if pc is not None:
            try:
                await pc.close()
            except Exception as e:
                logger.debug("RTC close error: %s", e)

    async def shutdown(self) -> None:
        await self._close_pc()


# ── Module-level registry (dashboard ↔ agent, same process) ──────

_manager: Optional[RTCDeviceManager] = None


def set_rtc_manager(manager: Optional[RTCDeviceManager]) -> None:
    global _manager
    _manager = manager


def get_rtc_manager() -> Optional[RTCDeviceManager]:
    return _manager
