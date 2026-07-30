"""SceneProber — periodic scene analysis via Omni REST API.

Runs in a background thread. Every N seconds:
1. Snaps the latest camera frame from the shared VisionManager buffer
2. Calls Omni REST API (/chat/completions) to analyze the scene
3. Parses the structured JSON response
4. Injects significant state changes into the conversation via WS session.update

Keeps analysis fully independent from the voice conversation pipeline
(no audio output, no child interruption, no token pollution on the WS).

Usage:
    prober = SceneProber(
        frame_getter=vision_mgr._latest_frame,
        ws_getter=lambda: agent_ws,
        api_key="sk-...",
    )
    prober.start()
    ...
    prober.stop()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Default interval —～4 probes per minute; low enough for token budget ──
DEFAULT_PROBE_INTERVAL = 15.0  # seconds

# ── Structured scene analysis from Omni ──────────────────────────


@dataclass
class SceneSnapshot:
    """Parsed result of one scene probe."""
    activity: str = ""          # e.g. "drawing", "playing", "reading", "idle"
    mood: str = ""              # "focused", "happy", "bored", "frustrated"
    person_count: int = 1
    objects: list[str] = field(default_factory=list)
    should_engage: bool = False
    engagement_reason: str = ""
    teaching_opportunity: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "SceneSnapshot":
        return cls(
            activity=data.get("activity", ""),
            mood=data.get("mood", ""),
            person_count=data.get("person_count", 1),
            objects=data.get("objects", []),
            should_engage=data.get("should_engage", False),
            engagement_reason=data.get("engagement_reason", ""),
            teaching_opportunity=data.get("teaching_opportunity", ""),
        )

    def meaningfully_different_from(self, other: "SceneSnapshot") -> bool:
        """Check if two snapshots are different enough to warrant an update."""
        return (
            self.activity != other.activity
            or self.mood != other.mood
            or self.person_count != other.person_count
            or self.should_engage != other.should_engage
        )

    def to_context_string(self) -> str:
        """Format as a short context line for the conversation prompt."""
        parts = [f"Child is {self.activity or 'nearby'}"]
        if self.mood:
            parts.append(f"| mood: {self.mood}")
        parts.append(f"| people: {self.person_count}")
        if self.objects:
            parts.append(f"| objects: {', '.join(self.objects[:5])}")
        if self.should_engage:
            parts.append(
                f"| ENGAGE: {self.engagement_reason or self.teaching_opportunity}"
            )
        else:
            parts.append("| STAY QUIET: protect focus")
        return " ".join(parts)


# ── SceneProber ──────────────────────────────────────────────────


class SceneProber:
    """Periodic scene analysis probe using Omni REST API.

    Runs in a background daemon thread. Analysis results are stored
    thread-safely and injected into the conversation via session.update
    when the scene changes meaningfully.
    """

    def __init__(
        self,
        frame_getter: Callable[[], Optional[str]],
        ws_getter: Callable[[], Optional[object]],
        api_key: Optional[str] = None,
        instructions_builder: Optional[Callable[[], str]] = None,
        interval: float = DEFAULT_PROBE_INTERVAL,
    ):
        self._frame_getter = frame_getter
        self._ws_getter = ws_getter
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._instructions_builder = instructions_builder
        self._interval = interval

        # API config
        self._api_url = os.environ.get(
            "LLM_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/") + "/chat/completions"
        self._model = os.environ.get("OMNI_CLOUD_MODEL", "qwen-omni-turbo")

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        # Latest analysis (thread-safe)
        self._lock = threading.Lock()
        self._latest: Optional[SceneSnapshot] = None
        self._probe_count: int = 0

        # HTTP client (created lazily in the probe thread)
        self._http: Optional[httpx.Client] = None

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the background probe thread. Idempotent."""
        if self._started:
            return
        self._stop_event.clear()
        self._started = True
        self._thread = threading.Thread(
            target=self._probe_loop,
            name="scene-prober",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SceneProber started (interval=%.0fs, model=%s)", self._interval, self._model,
        )

    def stop(self) -> None:
        """Signal the probe thread to stop and wait."""
        if not self._started:
            return
        self._stop_event.set()
        self._started = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._http:
            self._http.close()
            self._http = None
        logger.info("SceneProber stopped")

    # ── State queries ────────────────────────────────────────────

    @property
    def latest(self) -> Optional[SceneSnapshot]:
        with self._lock:
            return self._latest

    @property
    def probe_count(self) -> int:
        with self._lock:
            return self._probe_count

    def scene_context(self) -> str:
        """Get the latest scene analysis as a context string for prompts.

        Returns empty string if no analysis has been completed yet.
        """
        snap = self.latest
        if snap is None:
            return ""
        return snap.to_context_string()

    # ── Probe loop ───────────────────────────────────────────────

    def _probe_loop(self) -> None:
        """Main loop: sleep → snap → analyze → inject → repeat."""
        while not self._stop_event.is_set():
            self._sleep_interruptible(self._interval)
            if self._stop_event.is_set():
                break

            frame = self._frame_getter()
            if frame is None:
                logger.debug("Scene probe skipped — no camera frame available")
                continue

            try:
                snapshot = self._analyze_frame(frame)
                if snapshot is None:
                    continue
                self._on_analysis(snapshot)
            except Exception as e:
                logger.warning("Scene probe failed: %s", e)

    def _analyze_frame(self, frame_b64: str) -> Optional[SceneSnapshot]:
        """Send frame to Omni REST API, parse JSON response."""
        if self._http is None:
            self._http = httpx.Client(
                base_url="",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(15.0),
            )

        prompt = (
            "You are a child activity observer. Look at this image from a camera "
            "overlooking a child's play/study area. Respond with ONLY a JSON object "
            "(no markdown, no explanation) containing:\n"
            '- "activity": what the child is doing (one of: playing, drawing, '
            "reading, studying, moving, idle, unknown)\n"
            '- "mood": apparent emotional state (one of: focused, happy, bored, '
            "frustrated, tired, neutral)\n"
            '- "person_count": number of people visible (integer)\n'
            '- "objects": list of notable visible objects (toys, books, furniture, etc.)\n'
            '- "should_engage": true ONLY if the child is looking toward camera, '
            "holding something up, appears bored, or newly arrived in the room. "
            "false otherwise (focused, doing homework, multiple people, sleeping)\n"
            '- "engagement_reason": short reason if should_engage is true\n'
            '- "teaching_opportunity": a brief teaching angle if relevant (e.g. '
            '"new toy — teach colors", "book — reading time")\n\n'
            "IMPORTANT: The camera is for a child's room. Be conservative — "
            "prefer should_engage: false unless clearly indicated."
        )

        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 200,
            "temperature": 0.3,  # low temp for structured analysis
        }

        try:
            resp = self._http.post(self._api_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return self._parse_response(text)
        except (httpx.RequestError, KeyError, json.JSONDecodeError) as e:
            logger.warning("Scene analysis API error: %s", e)
            return None

    def _parse_response(self, text: str) -> Optional[SceneSnapshot]:
        """Parse Omni response into a SceneSnapshot."""
        # Try direct JSON parse
        try:
            data = json.loads(text)
            return SceneSnapshot.from_json(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block from markdown
        import re
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                return SceneSnapshot.from_json(data)
            except json.JSONDecodeError:
                pass

        logger.warning("Scene probe: could not parse JSON from response: %s", text[:120])
        return None

    # ── State management ─────────────────────────────────────────

    def _on_analysis(self, snapshot: SceneSnapshot) -> None:
        """Handle a new scene analysis — update state, inject if changed."""
        with self._lock:
            self._probe_count += 1
            prev = self._latest
            self._latest = snapshot

        changed = prev is None or snapshot.meaningfully_different_from(prev)
        logger.debug(
            "Scene: %s | mood=%s | people=%d | engage=%s%s",
            snapshot.activity, snapshot.mood, snapshot.person_count,
            snapshot.should_engage,
            " (changed)" if changed else "",
        )

        if changed:
            self._inject_state()

    def _inject_state(self) -> None:
        """Push updated scene context into the conversation via session.update.

        Rebuilds the full instructions (using the instructions_builder callback)
        and sends a session.update event. This tells Emma about the child's
        current state without interrupting or producing audio.
        """
        ws = self._ws_getter()
        if ws is None:
            return
        builder = self._instructions_builder
        if builder is None:
            return

        try:
            instructions = builder()
            ws.send(json.dumps({
                "type": "session.update",
                "session": {"instructions": instructions},
            }))
            logger.debug("Scene state injected into conversation")
        except Exception as e:
            logger.warning("Scene inject error: %s", e)

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in short increments so stop() is responsive."""
        interval = 0.5
        elapsed = 0.0
        while elapsed < seconds and not self._stop_event.is_set():
            time.sleep(interval)
            elapsed += interval
