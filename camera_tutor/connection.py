"""RealtimeConnection — WebSocket lifecycle for Qwen-Omni Realtime API.

Manages connection, disconnection, and exponential-backoff reconnection
for the Qwen-Omni WebSocket-based real-time multimodal API.

Usage:
    conn = RealtimeConnection(
        url="wss://...",
        api_key="sk-...",
        on_open=my_on_open,
        on_message=my_on_message,
        on_error=my_on_error,
        on_close=my_on_close,
    )
    conn.start()   # blocking — returns when connection is established
    conn.stop()    # graceful shutdown
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ── Default callbacks (no-ops) ──────────────────────────────────

def _noop(*args, **kwargs):
    pass


# ── Connection config ────────────────────────────────────────────


@dataclass
class ReconnectConfig:
    """Reconnection strategy configuration."""
    max_attempts: int = 0           # 0 = unlimited
    base_delay: float = 1.0         # First retry after 1s
    max_delay: float = 30.0         # Cap at 30s
    jitter: float = 0.1             # ±10% random jitter
    ping_interval: int = 120        # WebSocket ping interval (seconds)


# ── Connection class ─────────────────────────────────────────────


class RealtimeConnection:
    """Manages a WebSocket connection to Qwen-Omni Realtime API.

    Provides:
    - Clean lifecycle (connect / disconnect / reconnect)
    - Exponential backoff with jitter
    - Thread-safe stop signal that interrupts reconnection waits
    - Callback registration for open/message/error/close events
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        on_open: Callable = _noop,
        on_message: Callable = _noop,
        on_error: Callable = _noop,
        on_close: Callable = _noop,
        config: Optional[ReconnectConfig] = None,
    ):
        self.url = url
        self.api_key = api_key
        self._on_open = on_open
        self._on_message = on_message
        self._on_error = on_error
        self._on_close = on_close
        self.config = config or ReconnectConfig()

        # State
        self.ws: object = None       # websocket.WebSocketApp
        self.running = True
        self._connected = False
        self._reconnect_attempt = 0
        self._last_ws_created: float = 0.0

        # WebSocket module (lazy import)
        self._ws_mod = None

    # ── Public API ───────────────────────────────────────────────

    def start(self) -> None:
        """Enter the connection loop. Blocks until stop() is called."""
        import websocket as _ws

        self._ws_mod = _ws
        self.running = True
        self._reconnect_attempt = 0

        while self.running:
            self._connect_once()

            # If we stopped cleanly, break
            if not self.running:
                break

            # Reconnect delay with exponential backoff
            delay = self._backoff_delay()
            logger.info(
                "Reconnecting in %.1fs (attempt %d)...",
                delay, self._reconnect_attempt,
            )
            self._sleep_interruptible(delay)

    def stop(self) -> None:
        """Signal the connection loop to stop and close the WebSocket."""
        self.running = False
        self._connected = False
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def reconnect_attempt(self) -> int:
        return self._reconnect_attempt

    # ── Internal ─────────────────────────────────────────────────

    def _connect_once(self) -> None:
        """Create a WebSocketApp and run it (blocks until disconnect)."""
        ws = self._ws_mod.WebSocketApp(
            self.url,
            header=[f"Authorization: Bearer {self.api_key}"],
            on_open=self._wrap_on_open(),
            on_message=self._wrap_on_message(),
            on_error=self._wrap_on_error(),
            on_close=self._wrap_on_close(),
        )
        self.ws = ws
        self._last_ws_created = time.time()
        self._reconnect_attempt += 1

        # Blocks until connection closes or stop() is called
        ws.run_forever(ping_interval=self.config.ping_interval)

    def _backoff_delay(self) -> float:
        """Compute exponential backoff with jitter."""
        if self.config.max_attempts > 0 and self._reconnect_attempt >= self.config.max_attempts:
            # Give up
            return float("inf")

        import random
        delay = self.config.base_delay * (2 ** min(self._reconnect_attempt - 1, 8))
        delay = min(delay, self.config.max_delay)
        jitter = delay * self.config.jitter * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in short increments so stop() is responsive."""
        interval = 0.1
        elapsed = 0.0
        while elapsed < seconds and self.running:
            time.sleep(interval)
            elapsed += interval

    # ── Callback wrappers ────────────────────────────────────────

    def _wrap_on_open(self):
        _on_open = self._on_open
        _self = self

        def _on_open_wrapper(ws):
            _self._connected = True
            _self._reconnect_attempt = 0
            logger.info("WebSocket connected")
            _on_open(ws)

        return _on_open_wrapper

    def _wrap_on_message(self):
        _on_message = self._on_message
        _self = self

        def _on_message_wrapper(ws, message):
            if not _self.running:
                return
            _on_message(ws, message)

        return _on_message_wrapper

    def _wrap_on_error(self):
        _on_error = self._on_error
        _self = self

        def _on_error_wrapper(ws, error):
            _self._connected = False
            logger.warning("WebSocket error: %s", error)
            _on_error(ws, error)

        return _on_error_wrapper

    def _wrap_on_close(self):
        _on_close = self._on_close
        _self = self

        def _on_close_wrapper(ws, status, msg):
            _self._connected = False
            logger.info("WebSocket closed (code=%s)", status)
            _on_close(ws, status, msg)

        return _on_close_wrapper
