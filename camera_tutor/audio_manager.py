"""AudioManager — mic capture and speaker playback lifecycle.

Both mic input AND speaker output use callback-based streams
because macOS CoreAudio + blocking RawStream causes dropouts.

Mic:     InputStream callback → ring buffer → read_mic()
Speaker: write_spk() → ring buffer → OutputStream callback
Visemes: paired with audio in ring buffer → pushed at play time

Usage:
    mgr = AudioManager()
    mgr.set_viseme_handler(my_handler)   # optional: for lip-sync
    mgr.start()
    data = mgr.read_mic()                # bytes, ~0.2s @ 16kHz
    mgr.write_spk(data, visemes=[...])   # bytes + viseme payloads
    mgr.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional, Callable

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────

RATE_MIC = 16000    # Mic sample rate (server expects 16kHz PCM)
RATE_SPK = 24000    # Speaker sample rate (server outputs 24kHz PCM)

MIC_CALLBACK_FRAMES = 800    # 50ms @ 16kHz
MIC_READ_FRAMES = 3200       # 200ms per read_mic() call

SPK_CALLBACK_FRAMES = 1200   # 50ms @ 24kHz

DEFAULT_MIC_GAIN = 1.0


# ── AudioManager ────────────────────────────────────────────────


class AudioManager:
    """Manages microphone input and speaker output streams.

    Both directions use callback-based streams. Viseme payloads
    can be paired with audio chunks for lip-sync — they are
    held until the corresponding audio actually plays, then
    pushed via the registered viseme_handler.
    """

    def __init__(
        self,
        mic_gain: float = DEFAULT_MIC_GAIN,
        mic_device_index: Optional[int] = None,
    ):
        self._mic_stream: Optional[sd.InputStream] = None
        self._spk_stream: Optional[sd.OutputStream] = None
        self._started = False
        self._mic_index: Optional[int] = mic_device_index
        self._mic_gain = mic_gain

        # Mic: callback → ring → read_mic()
        self._mic_ring: deque[bytes] = deque()
        self._mic_ring_maxlen = 100
        self._mic_lock = threading.Lock()
        self._mic_ring_overflow = 0

        # Speaker: write_spk() → ring → callback
        # Each entry: (audio_bytes, viseme_payloads_or_None)
        self._spk_ring: deque[tuple[bytes, list[dict] | None]] = deque()
        self._spk_ring_maxlen = 200
        self._spk_lock = threading.Lock()
        self._spk_ring_underflow = 0

        # Viseme outbox: callback pushes here; drain thread sends
        self._viseme_outbox: deque[dict] = deque()
        self._viseme_outbox_lock = threading.Lock()
        self._viseme_handler: Callable[[dict], None] | None = None
        self._viseme_drain_thread: threading.Thread | None = None
        self._viseme_drain_stop = threading.Event()

        # Level tracking
        self._level_samples: list[float] = []

    # ── Viseme handler (for lip-sync) ────────────────────────────

    def set_viseme_handler(self, handler: Callable[[dict], None]) -> None:
        """Register a callback for viseme payloads.

        Called from the viseme drain thread (NOT the audio callback)
        whenever a queued viseme is due for display.

        Args:
            handler: fn(dict) — typically FaceSyncManager.push_payload
        """
        self._viseme_handler = handler

    def _start_viseme_drain(self) -> None:
        """Background thread that drains the viseme outbox."""
        if self._viseme_drain_thread is not None:
            return
        self._viseme_drain_stop.clear()
        self._viseme_drain_thread = threading.Thread(
            target=self._viseme_drain_loop,
            name="viseme-drain",
            daemon=True,
        )
        self._viseme_drain_thread.start()

    def _stop_viseme_drain(self) -> None:
        self._viseme_drain_stop.set()
        if self._viseme_drain_thread:
            self._viseme_drain_thread.join(timeout=1.0)
            self._viseme_drain_thread = None

    def _viseme_drain_loop(self) -> None:
        """Continuously drain viseme outbox, ~50Hz."""
        while not self._viseme_drain_stop.is_set():
            handler = self._viseme_handler
            if handler is None:
                time.sleep(0.1)
                continue

            # Drain all pending visemes
            payloads: list[dict] = []
            with self._viseme_outbox_lock:
                while self._viseme_outbox:
                    payloads.append(self._viseme_outbox.popleft())

            for p in payloads:
                try:
                    handler(p)
                except Exception:
                    pass

            time.sleep(0.02)  # 50Hz

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return

        if self._mic_index is None:
            self._mic_index = self._find_mic_device()
        mic_name = (
            sd.query_devices()[self._mic_index]["name"]
            if self._mic_index is not None
            else "system default"
        )
        logger.info("Mic device: %s (index=%s)", mic_name, self._mic_index)
        if self._mic_gain != 1.0:
            logger.info("Mic gain: %.1fx (%.1f dB)", self._mic_gain,
                        20 * np.log10(max(self._mic_gain, 1e-6)))

        self._mic_ring.clear()
        self._mic_ring_overflow = 0
        self._spk_ring.clear()
        self._spk_ring_underflow = 0
        self._viseme_outbox.clear()

        self._mic_stream = sd.InputStream(
            samplerate=RATE_MIC, channels=1, dtype="int16",
            blocksize=MIC_CALLBACK_FRAMES, device=self._mic_index,
            callback=self._mic_callback,
        )
        self._spk_stream = sd.OutputStream(
            samplerate=RATE_SPK, channels=1, dtype="int16",
            blocksize=SPK_CALLBACK_FRAMES,
            callback=self._spk_callback,
        )
        self._mic_stream.start()
        self._spk_stream.start()

        self._start_viseme_drain()
        self._started = True
        logger.info("Audio ready (mic=%dHz, spk=%dHz)", RATE_MIC, RATE_SPK)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_viseme_drain()
        for stream, name in [(self._mic_stream, "Mic"), (self._spk_stream, "Speaker")]:
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception as e:
                    logger.warning("%s close error: %s", name, e)
        self._mic_stream = None
        self._spk_stream = None

    # ── Mic callback ─────────────────────────────────────────────

    def _mic_callback(self, indata: np.ndarray, frames: int,
                      time_info, status) -> None:
        raw = indata.tobytes()
        with self._mic_lock:
            if len(self._mic_ring) >= self._mic_ring_maxlen:
                self._mic_ring_overflow += 1
                self._mic_ring.popleft()
            self._mic_ring.append(raw)

    # ── Speaker callback ─────────────────────────────────────────

    def _spk_callback(self, outdata: np.ndarray, frames: int,
                      time_info, status) -> None:
        """Pull audio from spk ring, fill outdata. Queue visemes."""
        needed = frames * 2  # int16 = 2 bytes/sample
        viseme_batch: list[dict] = []

        with self._spk_lock:
            if self._spk_ring:
                # Gather audio + visemes until we have enough
                audio_parts: list[bytes] = []
                total = 0
                while self._spk_ring and total < needed:
                    chunk_bytes, visemes = self._spk_ring.popleft()
                    audio_parts.append(chunk_bytes)
                    total += len(chunk_bytes)
                    if visemes:
                        viseme_batch.extend(visemes)

                buf = b"".join(audio_parts)

                if len(buf) >= needed:
                    outdata[:] = np.frombuffer(buf[:needed], dtype=np.int16).reshape((frames, 1))
                    leftover = buf[needed:]
                    if leftover:
                        # Push leftover back — visemes already consumed
                        self._spk_ring.appendleft((leftover, None))
                else:
                    outdata.fill(0)
                    out_flat = outdata.ravel()
                    arr = np.frombuffer(buf, dtype=np.int16)
                    out_flat[:len(arr)] = arr
                    self._spk_ring_underflow += 1
            else:
                outdata.fill(0)
                self._spk_ring_underflow += 1

        # Push visemes to outbox (outside lock, safe for PortAudio thread)
        if viseme_batch and self._viseme_handler is not None:
            with self._viseme_outbox_lock:
                self._viseme_outbox.extend(viseme_batch)

    # ── Mic I/O ──────────────────────────────────────────────────

    def read_mic(self) -> Optional[bytes]:
        if not self._started or self._mic_stream is None:
            return None
        try:
            collected = 0
            parts: list[bytes] = []
            with self._mic_lock:
                while collected < MIC_READ_FRAMES and self._mic_ring:
                    chunk = self._mic_ring.popleft()
                    parts.append(chunk)
                    collected += MIC_CALLBACK_FRAMES
            if collected == 0:
                return None

            raw_bytes = b"".join(parts)
            if self._mic_gain != 1.0:
                arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                arr *= self._mic_gain
                np.clip(arr, -32767, 32767, out=arr)
                boosted = arr.astype(np.int16).tobytes()
            else:
                boosted = raw_bytes
                arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

            self._track_level(arr)
            if self._mic_ring_overflow > 0:
                count = self._mic_ring_overflow
                self._mic_ring_overflow = 0
                logger.warning("Mic ring overflow x%d", count)
            return boosted
        except Exception as e:
            logger.warning("Mic read error: %s", e)
            return None

    def _track_level(self, samples: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self._level_samples.append(rms)
        max_samples = int(5.0 / (MIC_READ_FRAMES / RATE_MIC))
        if len(self._level_samples) > max_samples:
            self._level_samples = self._level_samples[-max_samples:]

    # ── Speaker I/O ──────────────────────────────────────────────

    def write_spk(self, data: bytes, visemes: list[dict] | None = None) -> None:
        """Push audio + optional viseme payloads to speaker ring. Non-blocking.

        Visemes are held until the corresponding audio chunk actually
        plays, then pushed via the registered viseme_handler.

        Args:
            data: 16-bit PCM bytes @ 24kHz.
            visemes: Pre-computed viseme payload dicts (from FaceSync).
        """
        if not self._started or self._spk_stream is None:
            return
        with self._spk_lock:
            if len(self._spk_ring) >= self._spk_ring_maxlen:
                self._spk_ring.popleft()
            self._spk_ring.append((data, visemes))

    # ── Properties ───────────────────────────────────────────────

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
        logger.info("Mic gain changed to %.1fx", self._mic_gain)

    @property
    def ring_buffer_fill(self) -> int:
        return len(self._mic_ring)

    @property
    def spk_buffer_fill(self) -> int:
        return len(self._spk_ring)

    @property
    def is_running(self) -> bool:
        return self._started

    # ── Device detection ─────────────────────────────────────────

    @staticmethod
    def list_input_devices() -> list[dict]:
        import sounddevice as _sd
        result = []
        for i, dev in enumerate(_sd.query_devices()):
            if dev["max_input_channels"] > 0:
                result.append({
                    "index": i,
                    "name": dev["name"],
                    "samplerate": dev["default_samplerate"],
                })
        return result

    @staticmethod
    def _find_mic_device() -> Optional[int]:
        import sounddevice as _sd
        devices = _sd.query_devices()
        keywords_list = [
            ["Jabra", "USB Audio"],
            ["Jabra"],
            ["Brio", "mono"],
            ["Poly"],
        ]
        for keywords in keywords_list:
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0 and all(k in dev["name"] for k in keywords):
                    return i
        return None
