"""VisionManager — camera frame capture and preview streaming.

Manages two background threads:
1. Camera reader — continuously captures frames from CameraPipeline
   and keeps the latest JPEG base64 in a thread-safe buffer.
2. Camera preview — periodically sends the latest frame to the
   Qwen-Omni WebSocket (input_image_buffer.append) and to the
   dashboard HTTP endpoint.

Thread lifecycle is managed with a stop Event, ensuring clean
shutdown and safe re-initialisation on WebSocket reconnect.

Usage:
    mgr = VisionManager(camera, ws)
    mgr.start()
    # ... agent loop ...
    mgr.stop()
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Optional

import cv2
import httpx

from camera_tutor.camera import CameraPipeline

logger = logging.getLogger(__name__)


class VisionManager:
    """Manages camera frame capture and preview streaming threads.

    Provides:
    - Continuous frame reading from CameraPipeline
    - Periodic frame push to WebSocket (input_image_buffer.append)
    - Periodic frame push to Dashboard HTTP endpoint
    - Thread-safe stop/start for clean reconnect
    """

    def __init__(
        self,
        camera: CameraPipeline,
        ws_getter,
        audio_ready: Optional[threading.Event] = None,
        session_ready: Optional[threading.Event] = None,
    ):
        """Initialize VisionManager.

        Args:
            camera: CameraPipeline instance.
            ws_getter: Callable returning current WebSocketApp (or None).
            audio_ready: Event set when first mic audio chunk is sent.
            session_ready: Event set when session.updated is received.
                Both must be set before WS image push begins.
        """
        self._camera = camera
        self._ws_getter = ws_getter
        self._audio_ready = audio_ready
        self._session_ready = session_ready
        self.ws_interval: float = 2.0

        # Thread management
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

        # Frame buffer (thread-safe: latest JPEG base64)
        self._frame_lock = threading.Lock()
        self._latest_b64: Optional[str] = None

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Start camera reader and preview threads. Idempotent."""
        if self._started:
            return
        self._stop_event.clear()
        self._latest_b64 = None

        if self._camera is None:
            logger.warning("No camera available — vision manager disabled")
            self._started = True
            return

        reader = threading.Thread(
            target=self._reader_loop,
            name="camera-reader",
            daemon=True,
        )
        preview = threading.Thread(
            target=self._preview_loop,
            name="camera-preview",
            daemon=True,
        )
        self._threads = [reader, preview]
        reader.start()
        preview.start()
        self._started = True
        logger.info("Vision manager started")

    def stop(self) -> None:
        """Signal threads to stop and wait for them. Idempotent."""
        if not self._started:
            return
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._started = False
        self._latest_b64 = None
        with self._frame_lock:
            self._latest_b64 = None
        logger.info("Vision manager stopped")

    @property
    def is_running(self) -> bool:
        return self._started

    # ── Internal: reader loop ───────────────────────────────────

    def _reader_loop(self) -> None:
        """Continuously capture frames and update the shared buffer."""
        cap = self._camera._cap
        if cap is None:
            logger.warning("Camera capture not available — reader disabled")
            return

        while not self._stop_event.is_set():
            try:
                ret, img = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                img = cv2.resize(img, (360, 360))
                jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
                b64 = base64.b64encode(jpg).decode()
                with self._frame_lock:
                    self._latest_b64 = b64
            except Exception as e:
                logger.warning("Camera reader error: %s", e)
                time.sleep(0.05)

    # ── Internal: preview loop ──────────────────────────────────

    def _preview_loop(self) -> None:
        """Periodically send frames to WebSocket + Dashboard HTTP.

        WS images wait for audio_ready (mic has produced first chunk)
        to honour Omni API ordering requirements."""
        last_keyframe_time = 0.0
        preview_interval = 0.2

        # Wait for BOTH session ready AND audio ready before sending WS images.
        # Without this, images can arrive before session is configured,
        # causing "Error append image before append audio" from Omni API.
        if self._session_ready is not None:
            self._session_ready.wait()
        if self._audio_ready is not None:
            self._audio_ready.wait()
        while not self._stop_event.is_set():
            try:
                b64 = self._latest_frame()
                if b64 is None:
                    time.sleep(0.1)
                    continue

                now = time.time()

                # Push to dashboard HTTP (always)
                self._push_to_dashboard(b64)

                # Push to WebSocket (rate-limited, starts after audio)
                if now - last_keyframe_time >= self.ws_interval:
                    last_keyframe_time = now
                    self._push_to_websocket(b64)

                time.sleep(preview_interval)
            except Exception as e:
                logger.warning("Camera preview error: %s", e)
                time.sleep(0.3)

    # ── Internal: helpers ────────────────────────────────────────

    def _latest_frame(self) -> Optional[str]:
        """Get the latest frame base64. Thread-safe."""
        with self._frame_lock:
            return self._latest_b64

    def _push_to_dashboard(self, b64: str) -> None:
        """HTTP POST the frame to the dashboard server."""
        try:
            httpx.post(
                "http://localhost:8200/api/emma/camera",
                json={"camera_frame": b64},
                timeout=2,
            )
        except httpx.RequestError:
            pass  # Dashboard not running — not critical
        except Exception:
            pass

    def _push_to_websocket(self, b64: str) -> None:
        """Send input_image_buffer.append via WebSocket."""
        ws = self._ws_getter()
        if ws is None:
            return
        try:
            ws.send(json_encode({"type": "input_image_buffer.append", "image": b64}))
        except Exception as e:
            logger.warning("WS camera push error: %s", e)


def json_encode(obj: dict) -> str:
    """Fast JSON encode for WebSocket messages."""
    import json as _json
    return _json.dumps(obj)
