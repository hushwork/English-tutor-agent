"""AudioManager — mic capture and speaker playback lifecycle.

Wraps sounddevice RawInputStream (mic) and RawOutputStream (speaker)
with thread-safe start/stop, automatic device detection, and
convenience read/write methods.

Usage:
    mgr = AudioManager()
    mgr.start()
    data = mgr.read_mic()       # bytes, 0.2s @ 16kHz
    mgr.write_spk(data)         # bytes, 24kHz PCM
    mgr.stop()
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────

CHUNK = 3200        # 0.2s @ 16kHz — WebSocket audio chunk size
RATE_MIC = 16000    # Mic sample rate (server expects 16kHz PCM)
RATE_SPK = 24000    # Speaker sample rate (server outputs 24kHz PCM)


# ── AudioManager ────────────────────────────────────────────────


class AudioManager:
    """Manages microphone input and speaker output streams.

    Provides:
    - Automatic USB microphone device detection
    - Thread-safe start/stop
    - Non-blocking mic reads and speaker writes
    - Graceful cleanup on stop / error
    """

    def __init__(self):
        self._mic: Optional[sd.RawInputStream] = None
        self._spk: Optional[sd.RawOutputStream] = None
        self._lock = threading.Lock()
        self._started = False
        self._mic_index: Optional[int] = None

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Open mic and speaker streams. Idempotent."""
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            self._mic_index = self._find_mic_device()
            mic_name = (
                sd.query_devices()[self._mic_index]["name"]
                if self._mic_index is not None
                else "system default"
            )
            logger.info("Mic device: %s (index=%s)", mic_name, self._mic_index)

            self._mic = sd.RawInputStream(
                samplerate=RATE_MIC, channels=1, dtype="int16",
                blocksize=CHUNK, device=self._mic_index,
            )
            self._spk = sd.RawOutputStream(
                samplerate=RATE_SPK, channels=1, dtype="int16",
                blocksize=CHUNK,
            )
            self._mic.start()
            self._spk.start()
            self._started = True

    def stop(self) -> None:
        """Close audio streams. Idempotent."""
        with self._lock:
            if not self._started:
                return
            self._started = False
            try:
                if self._mic:
                    self._mic.stop()
                    self._mic.close()
            except Exception as e:
                logger.warning("Mic close error: %s", e)
            self._mic = None
            try:
                if self._spk:
                    self._spk.stop()
                    self._spk.close()
            except Exception as e:
                logger.warning("Speaker close error: %s", e)
            self._spk = None

    # ── I/O ──────────────────────────────────────────────────────

    def read_mic(self) -> Optional[bytes]:
        """Read one chunk from the microphone (blocking, ~0.2s).

        Returns:
            bytes: 16-bit PCM audio data, or None if not started.
        """
        if not self._started or self._mic is None:
            return None
        try:
            data, _overflow = self._mic.read(CHUNK)
            return bytes(data)
        except Exception as e:
            logger.warning("Mic read error: %s", e)
            return None

    def write_spk(self, data: bytes) -> None:
        """Write audio data to the speaker.

        Args:
            data: 16-bit PCM audio bytes at 24kHz.
        """
        if not self._started or self._spk is None:
            return
        try:
            self._spk.write(data)
        except Exception as e:
            logger.warning("Speaker write error: %s", e)

    @property
    def is_running(self) -> bool:
        return self._started

    # ── Device detection ─────────────────────────────────────────

    @staticmethod
    def _find_mic_device() -> Optional[int]:
        """Find the first USB microphone by common keyword patterns.

        Returns device index, or None for system default.
        """
        import sounddevice as _sd

        devices = _sd.query_devices()
        keywords_list = [
            ["Jabra", "USB Audio"],
            ["Brio", "mono"],
            ["Poly"],
        ]
        for keywords in keywords_list:
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0 and all(k in dev["name"] for k in keywords):
                    return i
        return None
