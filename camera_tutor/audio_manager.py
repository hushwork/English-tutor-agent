"""AudioManager — mic capture and speaker playback lifecycle.

Both mic input AND speaker output use callback-based streams
because macOS CoreAudio + blocking RawStream causes dropouts.

Mic:     InputStream callback → ring buffer → read_mic()
Speaker: write_spk() → ring buffer → OutputStream callback

Usage:
    mgr = AudioManager()
    mgr.start()
    data = mgr.read_mic()       # bytes, ~0.2s @ 16kHz
    mgr.write_spk(data)         # bytes, 24kHz PCM (non-blocking!)
    mgr.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────

RATE_MIC = 16000    # Mic sample rate (server expects 16kHz PCM)
RATE_SPK = 24000    # Speaker sample rate (server outputs 24kHz PCM)

# Mic: callback every 50ms
MIC_CALLBACK_FRAMES = 800    # 50ms @ 16kHz
MIC_READ_FRAMES = 3200       # 200ms — how much read_mic() returns per call

# Speaker: callback every 50ms
SPK_CALLBACK_FRAMES = 1200   # 50ms @ 24kHz

DEFAULT_MIC_GAIN = 1.0


# ── AudioManager ────────────────────────────────────────────────


class AudioManager:
    """Manages microphone input and speaker output streams.

    Both directions use callback-based streams to avoid
    macOS CoreAudio blocking-read buffer overflow issues.
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
        self._mic_ring_maxlen = 100  # 100 × 50ms = 5s
        self._mic_lock = threading.Lock()
        self._mic_ring_overflow = 0

        # Speaker: write_spk() → ring → callback
        self._spk_ring: deque[bytes] = deque()
        self._spk_ring_maxlen = 200  # 200 × 50ms = 10s
        self._spk_lock = threading.Lock()
        self._spk_ring_underflow = 0

        # Level tracking
        self._level_samples: list[float] = []

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

        # Clear rings
        self._mic_ring.clear()
        self._mic_ring_overflow = 0
        self._spk_ring.clear()
        self._spk_ring_underflow = 0

        # Input: callback-based InputStream
        self._mic_stream = sd.InputStream(
            samplerate=RATE_MIC,
            channels=1,
            dtype="int16",
            blocksize=MIC_CALLBACK_FRAMES,
            device=self._mic_index,
            callback=self._mic_callback,
        )
        # Output: callback-based OutputStream
        self._spk_stream = sd.OutputStream(
            samplerate=RATE_SPK,
            channels=1,
            dtype="int16",
            blocksize=SPK_CALLBACK_FRAMES,
            callback=self._spk_callback,
        )
        self._mic_stream.start()
        self._spk_stream.start()
        self._started = True
        logger.info("Audio ready (mic=%dHz cb=%d, spk=%dHz cb=%d)",
                    RATE_MIC, MIC_CALLBACK_FRAMES,
                    RATE_SPK, SPK_CALLBACK_FRAMES)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
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
        """Push mic audio into ring buffer."""
        raw = indata.tobytes()
        with self._mic_lock:
            if len(self._mic_ring) >= self._mic_ring_maxlen:
                self._mic_ring_overflow += 1
                self._mic_ring.popleft()
            self._mic_ring.append(raw)

    # ── Speaker callback ─────────────────────────────────────────

    def _spk_callback(self, outdata: np.ndarray, frames: int,
                      time_info, status) -> None:
        """Pull audio from spk ring buffer, fill outdata. Silence if empty."""
        needed = frames * 2  # int16 = 2 bytes per sample
        with self._spk_lock:
            if self._spk_ring:
                # Concatenate available chunks
                parts: list[bytes] = []
                total = 0
                while self._spk_ring and total < needed:
                    chunk = self._spk_ring.popleft()
                    parts.append(chunk)
                    total += len(chunk)

                buf = b"".join(parts)

                if len(buf) >= needed:
                    # We have enough — use exactly needed bytes
                    outdata[:] = np.frombuffer(buf[:needed], dtype=np.int16).reshape((frames, 1))
                    # Push leftover back to front of ring
                    leftover = buf[needed:]
                    if leftover:
                        self._spk_ring.appendleft(leftover)
                else:
                    # Not enough — pad with silence
                    outdata.fill(0)
                    out_flat = outdata.ravel()
                    arr = np.frombuffer(buf, dtype=np.int16)
                    out_flat[:len(arr)] = arr
                    self._spk_ring_underflow += 1
            else:
                outdata.fill(0)
                self._spk_ring_underflow += 1

    # ── Mic I/O ──────────────────────────────────────────────────

    def read_mic(self) -> Optional[bytes]:
        """Read ~200ms of audio from the mic ring buffer."""
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
                logger.warning("Mic ring overflow x%d — main loop too slow?", count)

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

    def write_spk(self, data: bytes) -> None:
        """Push audio data to speaker ring buffer. Non-blocking.

        Args:
            data: 16-bit PCM bytes @ 24kHz (from Qwen-Omni server).
        """
        if not self._started or self._spk_stream is None:
            return
        with self._spk_lock:
            if len(self._spk_ring) >= self._spk_ring_maxlen:
                # Ring full — drop oldest. This means playback is falling behind.
                logger.debug("Spk ring full — dropping oldest chunk")
                self._spk_ring.popleft()
            self._spk_ring.append(data)

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
        """How many callback chunks are queued for playback (debug)."""
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
