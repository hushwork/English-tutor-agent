"""Camera pipeline — frame capture, scene change detection, key frame selection.

Captures frames from a USB camera, detects meaningful scene changes,
and selects key frames for multimodal LLM analysis.
"""

from __future__ import annotations

import base64
import io
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class CapturedFrame:
    """A single frame captured from the camera."""
    image: np.ndarray           # Raw BGR image (numpy array)
    timestamp: float            # Unix timestamp
    frame_index: int            # Monotonic frame counter
    base64_jpeg: str = ""       # Cached JPEG base64 encoding
    is_key_frame: bool = False  # Selected for LLM analysis


class CameraPipeline:
    """Manages camera capture, scene change detection, and key frame selection.

    Features:
    - Continuous frame capture at configurable FPS
    - Scene change detection (meaningful vs. trivial changes)
    - Key frame selection (only send novel scenes to LLM)
    - Base64 JPEG encoding for API transmission
    - Rolling scene buffer for context
    """

    def __init__(
        self,
        camera_id: int = 0,
        fps: int = 5,
        resolution: tuple[int, int] = (640, 480),
        jpeg_quality: int = 75,
        scene_change_threshold: float = 0.15,
        key_frame_min_interval: float = 3.0,  # seconds
        history_size: int = 30,  # frames
    ):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.jpeg_quality = jpeg_quality
        self.scene_change_threshold = scene_change_threshold
        self.key_frame_min_interval = key_frame_min_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter = 0
        self._last_key_frame_time = 0.0
        self._last_frame_hash: Optional[int] = None
        self._history: deque[CapturedFrame] = deque(maxlen=history_size)
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self):
        """Open the camera and begin capturing."""
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._running = True

    def stop(self):
        """Release the camera."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── Frame capture ────────────────────────────────────────────

    def capture(self) -> Optional[CapturedFrame]:
        """Capture a single frame. Returns None if capture fails."""
        if not self._running or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._frame_counter += 1

        # Resize if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)

        cf = CapturedFrame(
            image=frame,
            timestamp=time.time(),
            frame_index=self._frame_counter,
        )

        # Scene change detection
        cf.is_key_frame = self._detect_scene_change(cf)

        self._history.append(cf)
        return cf

    def capture_key_frame(self) -> Optional[CapturedFrame]:
        """Block until a key frame is captured (scene has changed meaningfully).

        Returns None if no key frame found within a reasonable timeout.
        """
        timeout = time.time() + 10.0
        while time.time() < timeout:
            frame = self.capture()
            if frame and frame.is_key_frame:
                return frame
            time.sleep(1.0 / max(self.fps, 1))
        return None

    def get_latest_frame(self) -> Optional[CapturedFrame]:
        """Return the most recent captured frame (could be any frame)."""
        if self._history:
            return self._history[-1]
        return self.capture()

    # ── Encoding ─────────────────────────────────────────────────

    @staticmethod
    def to_base64(frame: CapturedFrame, quality: int = 75) -> str:
        """Encode a frame to base64 JPEG string. Caches result on the frame."""
        if frame.base64_jpeg:
            return frame.base64_jpeg

        _, buffer = cv2.imencode(
            '.jpg', frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        frame.base64_jpeg = base64.b64encode(buffer).decode('ascii')
        return frame.base64_jpeg

    # ── Scene change detection ───────────────────────────────────

    def _detect_scene_change(self, frame: CapturedFrame) -> bool:
        """Determine if this frame represents a meaningful scene change.

        Uses perceptual hash comparison: resize → DCT → compare.
        Fast enough for real-time on Jetson Orin (< 5ms per frame).
        """
        # Always mark as key frame if this is the first frame
        if self._last_frame_hash is None:
            self._last_frame_hash = self._frame_hash(frame.image)
            self._last_key_frame_time = frame.timestamp
            return True

        # Rate-limit key frames
        if frame.timestamp - self._last_key_frame_time < self.key_frame_min_interval:
            return False

        # Compare perceptual hashes
        current_hash = self._frame_hash(frame.image)
        hamming = bin(self._last_frame_hash ^ current_hash).count('1')

        # Threshold: if hash differs enough, it's a scene change
        if hamming >= self.scene_change_threshold * 64:  # 64-bit hash
            self._last_frame_hash = current_hash
            self._last_key_frame_time = frame.timestamp
            return True

        return False

    @staticmethod
    def _frame_hash(image: np.ndarray) -> int:
        """Compute a perceptual hash (pHash) of an image.

        Resize → grayscale → DCT → compare to median.
        Returns a 64-bit integer hash.
        """
        # Resize to 32x32 for pHash
        resized = cv2.resize(image, (32, 32))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # DCT
        dct = cv2.dct(np.float32(gray))
        # Take top-left 8x8 of DCT (low frequencies)
        dct_low = dct[:8, :8]

        # Compare to median
        median = np.median(dct_low)
        bits = (dct_low > median).flatten()

        # Pack into 64-bit integer
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val
