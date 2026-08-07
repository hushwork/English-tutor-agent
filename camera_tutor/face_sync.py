"""FaceSyncManager — viseme-driven face animation push to dashboard.

Manages the connection to the dashboard server for face animation:
- WebSocket connection (primary, low-latency)
- HTTP POST fallback (for old dashboard servers)
- Auto-reconnect with limited retries
- Viseme deduplication (only push when viseme changes)
- Viseme → Live2D parameter translation via VisemeParams

Usage:
    sync = FaceSyncManager()
    sync.start()
    sync.push_viseme(viseme, transcript)
    sync.stop()
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import httpx

from camera_tutor.avatar import Viseme
from camera_tutor.live2d_bridge import VisemeParams

logger = logging.getLogger(__name__)

# dashboard 开 TLS 时（WebRTC 远程设备模式）必须走 wss/https，
# 否则 WS 握手失败、HTTP fallback 也被静默吞掉——唇形完全不动
_DASHBOARD_TLS = bool(os.environ.get("DASHBOARD_TLS_CERT")
                      and os.environ.get("DASHBOARD_TLS_KEY"))
_HTTP = "https" if _DASHBOARD_TLS else "http"
_WS = "wss" if _DASHBOARD_TLS else "ws"
DASHBOARD_WS_URL = f"{_WS}://localhost:8200/ws/emma/source"
DASHBOARD_HTTP_URL = f"{_HTTP}://localhost:8200/api/emma/face"
DASHBOARD_CAMERA_URL = f"{_HTTP}://localhost:8200/api/emma/camera"
# 本机回环 + 自签证书（mkcert）：跳过证书校验
_SSL_OPT = {"cert_reqs": 0} if _DASHBOARD_TLS else None   # ssl.CERT_NONE
_HTTP_VERIFY = not _DASHBOARD_TLS

WS_RETRY_ATTEMPTS = 5
WS_RETRY_DELAY = 0.5


class FaceSyncManager:
    """Manages face animation data push to the dashboard server.

    Provides:
    - WebSocket push with automatic reconnect + HTTP fallback
    - Viseme deduplication (skips identical consecutive visemes)
    - Viseme → Live2D parameter translation
    - Clean lifecycle (start/stop)
    """

    def __init__(self, user_id: str = ""):
        self.user_id = user_id
        self._ws = None
        self._http_fallback = False
        self._last_viseme_label: Optional[str] = None
        self._started = False
        self._ws_mod = None

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Connect to the dashboard. Tries WebSocket first, falls back to HTTP."""
        if self._started:
            return

        import websocket as _ws_mod
        self._ws_mod = _ws_mod

        for attempt in range(WS_RETRY_ATTEMPTS):
            try:
                kw = {"timeout": 2}
                if _SSL_OPT is not None:
                    kw["sslopt"] = _SSL_OPT
                self._ws = _ws_mod.create_connection(DASHBOARD_WS_URL, **kw)
                self._http_fallback = False
                logger.info("Face WS connected to dashboard")
                self._started = True

                # Reset viseme state on dashboard
                self._send_reset()
                return
            except Exception as e:
                if attempt == 0:
                    logger.warning("Face WS connect failed (%s), retrying...", e)
                time.sleep(WS_RETRY_DELAY)

        # Fallback: HTTP POST
        self._http_fallback = True
        self._started = True
        logger.warning("Face WS unavailable, using HTTP fallback")

    def stop(self) -> None:
        """Close the WebSocket connection."""
        self._started = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── Public API ──────────────────────────────────────────────

    def push_viseme(self, viseme: Viseme, transcript: str) -> None:
        """Push a viseme to the dashboard (deduped).

        Args:
            viseme: Viseme enum value to display.
            transcript: Current full transcript text for display.
        """
        if not isinstance(viseme, Viseme):
            return
        if viseme.label == self._last_viseme_label:
            return
        self._last_viseme_label = viseme.label

        try:
            params = VisemeParams.from_viseme(viseme)
            payload = {
                "type": "viseme",
                "viseme": viseme.label,
                "mouth_open": params.mouth_open,
                "mouth_width": params.mouth_width,
                "tongue_visible": params.tongue_visible,
                "transcript": transcript,
            }
            self._send_payload(payload)
        except Exception as e:
            logger.warning("Viseme push error: %s", e)

    def push_payload(self, payload: dict) -> None:
        """Push a pre-built payload dict (no dedup). For AudioManager lip-sync."""
        payload.setdefault("type", "viseme")
        try:
            self._send_payload(payload)
        except Exception as e:
            logger.warning("Viseme payload error: %s", e)

    def reset_viseme(self) -> None:
        """Reset to silence/rest state."""
        self._last_viseme_label = None
        self._send_reset()

    def push_camera_frame(self, b64: str) -> None:
        """Push a camera frame to the dashboard (HTTP only)."""
        try:
            httpx.post(
                DASHBOARD_CAMERA_URL,
                json={"camera_frame": b64, "user_id": self.user_id},
                timeout=2,
                verify=_HTTP_VERIFY,
            )
        except httpx.RequestError:
            pass
        except Exception:
            pass

    # ── Internal ────────────────────────────────────────────────

    def _send_payload(self, payload: dict) -> None:
        """Send a JSON payload via WS or HTTP fallback."""
        # 多用户路由：所有事件携带 user_id（dashboard 按此分发）
        payload.setdefault("user_id", self.user_id)
        # WebSocket path
        if self._ws is not None and not self._http_fallback:
            try:
                self._ws.send(json.dumps(payload))
                return
            except Exception:
                logger.debug("Face WS send failed, attempting reconnect")
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

        # Try reconnect once
        if self._ws is None and not self._http_fallback:
            self._try_reconnect_once(payload)

        # HTTP fallback
        if self._http_fallback:
            self._send_http(payload)

    def _try_reconnect_once(self, payload: dict) -> None:
        """Attempt a single WS reconnect."""
        try:
            import websocket as _ws_mod
            kw = {"timeout": 1}
            if _SSL_OPT is not None:
                kw["sslopt"] = _SSL_OPT
            self._ws = _ws_mod.create_connection(DASHBOARD_WS_URL, **kw)
            self._ws.send(json.dumps(payload))
            logger.info("Face WS reconnected")
        except Exception:
            self._ws = None

    def _send_http(self, payload: dict) -> None:
        """Send payload via HTTP POST."""
        try:
            httpx.post(DASHBOARD_HTTP_URL, json=payload, timeout=1,
                       verify=_HTTP_VERIFY)
        except httpx.RequestError:
            pass
        except Exception:
            pass

    def _send_reset(self) -> None:
        """Send a reset/rest viseme state."""
        payload = {
            "type": "viseme",
            "viseme": "rest",
            "mouth_open": 0.0,
            "mouth_width": 0.0,
            "tongue_visible": 0.0,
            "transcript": "",
            "user_id": self.user_id,
        }
        if self._ws is not None and not self._http_fallback:
            try:
                self._ws.send(json.dumps(payload))
            except Exception:
                pass
