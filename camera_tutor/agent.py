"""CameraTutorAgent — orchestrator for the Camera Tutor real-time system.

The Agent owns all sub-managers (audio, vision, face sync, connection)
and coordinates them via WebSocket event handlers.

Lifecycle:
    agent = CameraTutorAgent()
    agent.setup()          # Init all managers
    agent.start()          # Start dashboard, connect to Omni API
    # ... runs until Ctrl+C or stop() ...
    agent.stop()           # Graceful shutdown

This replaces the monolithic realtime_demo.py with a class-based
architecture that properly manages threads, state, and reconnection.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from camera_tutor.audio_manager import AudioManager
from camera_tutor.connection import RealtimeConnection, ReconnectConfig
from camera_tutor.face_sync import FaceSyncManager
from camera_tutor.tutor_personas import get_active_tutor
from camera_tutor.camera import CameraPipeline
from camera_tutor.avatar import Viseme
from camera_tutor.vision_manager import VisionManager
from camera_tutor.parent_report import ParentReportEngine

# Shared memory & SR (reuse english_tutor modules)
from english_tutor.memory import ConversationMemory
from english_tutor.spaced_repetition import SpacedRepetition

from camera_tutor.scene_prober import SceneProber

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for CameraTutorAgent."""

    api_key: str = ""
    workspace_id: str = ""
    model: str = "qwen3.5-omni-flash-realtime"
    dashboard_port: int = 8200

    # Camera
    camera_enabled: bool = True
    camera_id: int = 0
    camera_fps: int = 5
    camera_resolution: tuple[int, int] = (360, 360)
    camera_scene_threshold: float = 0.15
    camera_key_interval: float = 1.0

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.workspace_id:
            self.workspace_id = os.environ.get("WORKSPACE_ID",
                "llm-xo2ff9jhvnvgvu6b")  # fallback — matches original realtime_demo.py
        if not self.model:
            self.model = os.environ.get("OMNI_MODEL", "qwen3.5-omni-flash-realtime")

        if not self.workspace_id:
            logger.error(
                "WORKSPACE_ID is empty — WS URL will be malformed. "
                "Set WORKSPACE_ID in .env or pass workspace_id to AgentConfig."
            )

    @property
    def ws_url(self) -> str:
        return (
            f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            f"/api-ws/v1/realtime?model={self.model}&language=en"
        )


# ── Agent State ──────────────────────────────────────────────────

@dataclass
class AgentState:
    """Thread-safe agent state snapshot."""
    running: bool = True
    session_ready: threading.Event = field(default_factory=threading.Event)
    audio_started: threading.Event = field(default_factory=threading.Event)
    current_transcript: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_transcript(self, delta: str) -> None:
        with self._lock:
            self.current_transcript += delta

    def set_transcript(self, text: str) -> None:
        with self._lock:
            self.current_transcript = text

    def get_transcript(self) -> str:
        with self._lock:
            return self.current_transcript


# ── CameraTutorAgent ─────────────────────────────────────────────

class CameraTutorAgent:
    """Main agent that orchestrates Camera Tutor's real-time interaction.

    Owns sub-managers and handles WebSocket events. Designed for
    clean lifecycle, safe reconnect, and testability.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = AgentState()

        # Callback references (set by setup)
        self._on_child_speech = None
        self._on_tutor_response = None

        # Sub-managers (initialised in setup/start)
        self.audio: Optional[AudioManager] = None
        self.camera: Optional[CameraPipeline] = None
        self.vision: Optional[VisionManager] = None
        self.face_sync: Optional[FaceSyncManager] = None
        self.connection: Optional[RealtimeConnection] = None

        # Tutor
        self.tutor = get_active_tutor()

        # Mic audio sender thread (managed separately — tied to WS lifecycle)
        self._mic_thread: Optional[threading.Thread] = None
        self._mic_stop = threading.Event()

        # Dashboard thread
        self._dashboard_thread: Optional[threading.Thread] = None

        # Memory & learning tracking
        self.memory: Optional[ConversationMemory] = None
        self.sr: Optional[SpacedRepetition] = None
        self.reporter: Optional[ParentReportEngine] = None
        self.scene_prober: Optional[SceneProber] = None
        self._storage_dir: Path = Path(
            __file__).resolve().parent.parent / ".camera-tutor-data"

        # Track recent utterances for context
        self._last_child_utterance: str = ""
        self._last_emma_utterance: str = ""
        self._utterances_this_session: int = 0
        self._recent_emma_phrases: list[str] = []

        # Signal handling
        self._signal_setup_done = False

    # ── Setup (called once before start) ─────────────────────────

    def setup(self) -> None:
        """Initialise sub-managers (camera, audio, face sync).

        Does NOT start connections or threads — that happens in start().
        This lets callers override config before the agent goes live.
        """
        # Camera
        if self.config.camera_enabled:
            self._setup_camera()

        # Audio
        self.audio = AudioManager()

        # Face sync (dashboard connection)
        self.face_sync = FaceSyncManager()

        # Memory & learning systems
        self.memory = ConversationMemory(storage_dir=self._storage_dir, user_id="camera_tutor")
        self.memory.new_session()
        logger.info("Conversation memory: %s", self.memory._data_dir)

        self.sr = SpacedRepetition(storage_dir=self._storage_dir, user_id="camera_tutor")
        logger.info("Spaced repetition: %s", self.sr._path)

        self.reporter = ParentReportEngine(storage_dir=self._storage_dir)

        # Scene prober (periodic scene analysis via Omni REST)
        self.scene_prober = SceneProber(
            frame_getter=self._get_latest_camera_frame,
            ws_getter=self._get_current_ws,
            api_key=self.config.api_key,
            instructions_builder=self._build_instructions,
        )

        # Register signal handler
        if not self._signal_setup_done:
            signal.signal(signal.SIGINT, self._handle_sigint)
            signal.signal(signal.SIGTERM, self._handle_sigint)
            self._signal_setup_done = True

        logger.info("Agent setup complete")

    def _get_latest_camera_frame(self) -> Optional[str]:
        """Get the latest camera frame b64 for the scene prober."""
        if self.vision is None or not self.vision.is_running:
            return None
        return self.vision._latest_frame()

    def _get_current_ws(self) -> Optional[object]:
        """Get the current WebSocket connection for scene injection."""
        if self.connection is None or self.connection.ws is None:
            return None
        return self.connection.ws

    def _setup_camera(self) -> None:
        """Try to initialise the camera pipeline."""
        for cam_id in [1, 0, 2]:
            try:
                cam = CameraPipeline(
                    camera_id=cam_id,
                    fps=self.config.camera_fps,
                    resolution=self.config.camera_resolution,
                    scene_change_threshold=self.config.camera_scene_threshold,
                    key_frame_min_interval=self.config.camera_key_interval,
                )
                cam.start()
                self.camera = cam
                logger.info("Camera: /dev/video%d", cam_id)
                return
            except Exception as e:
                logger.debug("Camera %d: %s", cam_id, e)
        logger.warning("No camera device found — running without visuals")

    # ── Start (begin operation) ──────────────────────────────────

    def start(self) -> None:
        """Start all managers and connect to the Omni API.

        Starts dashboard server, audio, vision, face sync, and
        the main WebSocket connection loop.
        """
        logger.info("Starting Camera Tutor Agent...")

        # Start dashboard server
        self._start_dashboard()

        # Start audio
        self.audio.start()

        # Start face sync
        self.face_sync.start()
        self.face_sync.reset_viseme()

        # Build connection
        ws_url = self.config.ws_url
        api_key = self.config.api_key

        if not api_key:
            logger.error("DASHSCOPE_API_KEY not set — set it in .env or pass to AgentConfig")
            sys.exit(1)

        if not self.config.workspace_id:
            logger.error(
                "WORKSPACE_ID is empty — cannot construct WS URL. "
                "Set WORKSPACE_ID in .env or pass workspace_id to AgentConfig."
            )
            sys.exit(1)

        self.connection = RealtimeConnection(
            url=ws_url,
            api_key=api_key,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            config=ReconnectConfig(
                base_delay=1.0,
                max_delay=30.0,
                ping_interval=120,
            ),
        )

        # Run connection loop (blocks until stop)
        logger.info("Connecting to %s ...", self.config.model)
        self.connection.start()

    def stop(self) -> None:
        """Graceful shutdown of all components."""
        logger.info("Stopping Camera Tutor Agent...")
        self.state.running = False

        # Stop connection
        if self.connection:
            self.connection.stop()

        # Stop mic thread
        self._mic_stop.set()

        # Stop vision
        if self.vision:
            self.vision.stop()

        # Stop face sync
        if self.face_sync:
            self.face_sync.stop()

        # Stop audio
        if self.audio:
            self.audio.stop()

        # Stop camera
        if self.camera:
            self.camera.stop()

        # Stop scene prober
        if self.scene_prober:
            self.scene_prober.stop()

        # Save memory state
        if self.memory:
            try:
                self.memory._save_stats()
            except Exception:
                pass

        logger.info("Camera Tutor Agent stopped")

    # ── WebSocket event handlers ─────────────────────────────────

    def _on_ws_open(self, ws) -> None:
        """Handle WebSocket open — configure session and start threads."""
        logger.info("WebSocket opened, configuring session...")

        # Configure session
        ws.send(json.dumps({
            "event_id": "session_init",
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "Tina",
                "instructions": self._build_instructions(),
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "input_audio_transcription": {
                    "language": "en",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                    "silence_duration_ms": 800,
                },
            },
        }))

        # Reset session events
        self.state.session_ready.clear()
        self.state.audio_started.clear()
        self.state.set_transcript("")

        # Start mic sending thread (stop old one first if any)
        self._mic_stop.set()
        self._mic_stop.clear()
        self._mic_thread = threading.Thread(
            target=self._mic_send_loop,
            args=(ws,),
            name="mic-send",
            daemon=True,
        )
        self._mic_thread.start()

        # Start vision manager (stop old one first if any)
        if self.vision:
            self.vision.stop()
        if self.camera:
            self.vision = VisionManager(camera=self.camera, ws_getter=lambda: ws)
            self.vision.start()

        # Start scene prober (periodic scene analysis via REST)
        if self.scene_prober:
            self.scene_prober.stop()
        self.scene_prober = SceneProber(
            frame_getter=self._get_latest_camera_frame,
            ws_getter=self._get_current_ws,
            api_key=self.config.api_key,
            instructions_builder=self._build_instructions,
        )
        self.scene_prober.start()

        self._print_welcome()

    def _on_ws_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket messages (audio, text, events)."""
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return

        event_type = event.get("type", "")

        # Log non-audio events at debug level
        if event_type not in ("response.audio.delta", "input_audio_buffer.speech_started"):
            logger.debug("[event] %s", event_type)

        if event_type == "session.updated":
            self.state.session_ready.set()
            self._notify_browser_ready()

        elif event_type == "input_audio_buffer.speech_started":
            self.state.audio_started.set()

        # ── Audio playback + viseme sync ──
        elif event_type == "response.audio.delta":
            chunk = base64.b64decode(event["delta"])
            self.audio.write_spk(chunk)
            self._process_viseme_chunk(chunk)

        elif event_type == "response.audio_transcript.delta":
            delta = event.get("delta", "")
            if delta:
                self.state.update_transcript(delta)

        elif event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info("🤖 %s: %s", self.tutor.name, transcript)
                self.state.set_transcript(transcript)
                self._last_emma_utterance = transcript
                self._utterances_this_session += 1
                self._recent_emma_phrases.append(transcript)
                if len(self._recent_emma_phrases) > 10:
                    self._recent_emma_phrases.pop(0)
                # Log to memory
                if self.memory:
                    self.memory.save_message("assistant", transcript)
                if self.reporter:
                    self.reporter.log_event("emma_spoke", {"text": transcript[:200]})

        elif event_type == "response.audio.done":
            self.face_sync.reset_viseme()
            self.state.set_transcript("")
            # Extract vocabulary from the completed utterance
            if self._last_emma_utterance:
                self._check_vocabulary(self._last_emma_utterance, is_emma=True)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info("👧 Child: %s", transcript)
                self._last_child_utterance = transcript
                # Log to memory
                if self.memory:
                    self.memory.save_message("user", transcript)
                if self.reporter:
                    self.reporter.log_event("child_spoke_english", {"text": transcript[:200]})
                # Extract vocabulary & errors
                self._check_vocabulary(transcript, is_emma=False)

        elif event_type == "error":
            err = event.get("error", {})
            logger.error(
                "API Error: %s (code=%s)",
                err.get("message", "unknown"),
                err.get("code", ""),
            )

    def _on_ws_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        logger.error("WebSocket error: %s", error)

    def _on_ws_close(self, ws, status, msg) -> None:
        """Handle WebSocket close — stop vision manager."""
        logger.info("WebSocket closed (code=%s)", status)
        if self.vision:
            self.vision.stop()
        self._mic_stop.set()

    # ── Internal: threads ────────────────────────────────────────

    def _mic_send_loop(self, ws) -> None:
        """Continuously read mic audio and send to WebSocket."""
        first = True
        while not self._mic_stop.is_set() and self.state.running:
            data = self.audio.read_mic()
            if data is None:
                time.sleep(0.01)
                continue
            try:
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode(),
                }))
                if first:
                    first = False
                    self.state.audio_started.set()
            except Exception as e:
                logger.warning("Mic send error: %s", e)
                break

    def _process_viseme_chunk(self, chunk: bytes) -> None:
        """Extract visemes from an audio chunk and push to face sync.

        Uses spectral analysis for zero-latency viseme extraction.
        """
        try:
            from camera_tutor.spectral_viseme import chunk_to_visemes
            for v in chunk_to_visemes(chunk, 24000):
                self.face_sync.push_viseme(v, self.state.get_transcript())
        except Exception as e:
            logger.debug("Viseme processing error: %s", e)

    # ── Dashboard lifecycle ──────────────────────────────────────

    def _start_dashboard(self, port: Optional[int] = None) -> None:
        """Start the dashboard server (if not already running)."""
        port = port or self.config.dashboard_port

        # Check if already running
        try:
            import httpx
            r = httpx.get(f"http://localhost:{port}/api/health", timeout=0.5)
            if r.status_code == 200:
                logger.info("Dashboard already running on port %d", port)
                return
        except Exception:
            pass

        logger.info("Starting dashboard on port %d...", port)
        import uvicorn

        def _run_dashboard():
            uvicorn.run(
                "camera_tutor.dashboard_server:app",
                host="0.0.0.0",
                port=port,
                log_level="warning",
            )

        self._dashboard_thread = threading.Thread(
            target=_run_dashboard, daemon=True, name="dashboard",
        )
        self._dashboard_thread.start()

        # Wait for dashboard to be ready
        for i in range(30):
            try:
                import httpx
                r = httpx.get(f"http://localhost:{port}/api/health", timeout=0.5)
                if r.status_code == 200:
                    logger.info("Dashboard ready on http://localhost:%d", port)
                    return
            except Exception:
                pass
            time.sleep(0.2)

        logger.warning("Dashboard start timeout — continuing without dashboard")

    # ── Rich instruction builder ──────────────────────────────────

    def _build_instructions(self) -> str:
        """Build a rich system prompt with child profile and vocabulary."""
        child_age = self._get_child_age()
        max_words = {3: 5, 5: 8, 7: 10, 9: 12, 12: 15}
        closest = min(max_words.keys(), key=lambda k: abs(k - child_age))
        w = max_words[closest]

        known_vocab = self._get_known_vocab()
        common_errors = self._get_common_errors()
        due_words = self._get_due_words()
        session_info = self._get_session_info()

        vocab_line = ""
        if known_vocab:
            vocab_line = (
                f"\nCHILD'S KNOWN VOCABULARY ({len(known_vocab)} words):\n"
                f"{', '.join(known_vocab)}\n"
            )
        errors_line = ""
        if common_errors:
            errors_line = f"\nCHILD'S COMMON ERRORS:\n{common_errors}\n"
        due_line = ""
        if due_words:
            due_line = (
                f"\nWORDS TO PRACTICE TODAY:\n"
                f"Try to naturally use these words: {', '.join(due_words)}\n"
            )

        # Recent phrases Emma used (avoid repetition)
        recent_phrases = self._recent_emma_phrases[-4:]
        recent_line = ""
        if recent_phrases:
            recent_line = (
                f"\nYOUR RECENT RESPONSES (do NOT repeat these):\n"
            )
            for i, p in enumerate(recent_phrases, 1):
                recent_line += f"  {i}. \"{p[:60]}...\"\n"
            recent_line += (
                f"\nUse completely different words and sentence structures.\n"
            )

        # Scene context (from SceneProber)
        scene_line = ""
        if self.scene_prober:
            ctx = self.scene_prober.scene_context()
            if ctx:
                scene_line = f"\nCURRENT SCENE (real-time analysis):\n{ctx}\n"

        return (
            f"You are {self.tutor.name}, a {self.tutor.teaching_style} English tutor "
            f"for a {child_age}-year-old child.\n"
            f"Personality: {', '.join(self.tutor.personality_traits[:3])}.\n"
            f"Appearance: a {self.tutor.age_appearance}-year-old woman.\n\n"
            f"{self.tutor._tutor_rules()}\n\n"
            f"IMPORTANT: The child speaks ENGLISH. Transcribe as English.\n\n"
            f"VISION: You receive real-time camera images — use what you SEE.\n\n"
            f"{session_info}{vocab_line}{errors_line}{due_line}{recent_line}{scene_line}"
            f"CRITICAL RULES:\n"
            f"1. MAXIMUM {w} words per sentence. ONE sentence only.\n"
            f"2. Use only simple words a {child_age}-year-old can understand.\n"
            f"3. NEVER repeat the same sentence structure you used recently.\n"
            f"4. Praise every attempt to speak English.\n"
            f"5. If you SEE something interesting in the camera, mention it.\n"
            f"6. This is turn #{self._utterances_this_session + 1} in this conversation.\n"
            f"7. If the child struggles, model the correct form patiently.\n"
            f"8. Use at least one word from the child's known vocabulary.\n"
        )

    def _get_child_age(self) -> int:
        from camera_tutor.tutor_personas import get_child_age as _get_age
        return _get_age()

    def _get_known_vocab(self) -> list[str]:
        if not self.memory:
            return []
        vocab = self.memory.get_vocabulary()
        return [v["word"] for v in vocab[-20:]] if vocab else []

    def _get_common_errors(self) -> str:
        if not self.memory:
            return ""
        errors = self.memory.stats.get("common_errors", {})
        if not errors:
            return ""
        parts = []
        for err_type, info in errors.items():
            ex = info.get("examples", [])
            ex_str = f' (e.g. "{ex[-1]}")' if ex else ""
            parts.append(f"- {err_type}: {info['count']}x{ex_str}")
        return "\n".join(parts)

    def _get_due_words(self) -> list[str]:
        if not self.sr:
            return []
        try:
            return [c.word for c in self.sr.get_due_cards(limit=5)]
        except Exception:
            return []

    def _get_session_info(self) -> str:
        if not self.memory:
            return ""
        s = self.memory.stats
        sr_total = len(self.sr.cards) if self.sr else 0
        return (
            f"LEARNER PROFILE:\n"
            f"- Session #{max(s.get('total_sessions', 0), 1)} for this child\n"
            f"- Messages so far: ~{max(s.get('total_messages', 0), 1)}\n"
            f"- Vocabulary tracked: {len(s.get('vocabulary', []))} words\n"
            f"- Spaced repetition cards: {sr_total} cards\n\n"
        )

    def _check_vocabulary(self, text: str, is_emma: bool) -> None:
        if not text.strip():
            return
        words = text.strip().strip(".,!?").split()
        if len(words) < 2:
            return
        if is_emma:
            self._extract_new_words(text)
        else:
            self._check_child_errors(text)

    def _extract_new_words(self, text: str) -> None:
        import re
        for _, word in re.findall(
            r"\b(this is|that is|it's a|it's an|here is|there is|"
            r"see the|look at the|that's a|that's an)\s+(\w+)",
            text, re.IGNORECASE,
        ):
            word = word.strip(".,!?'\"")
            if len(word) > 2 and word[0].isalpha() and self.sr and self.memory:
                self.sr.add_card(word, "", text[:100])
                self.memory.add_new_word(word, "", text[:100])
                logger.info("📝 New word tracked: %s", word)
        for (word,) in re.findall(
            r"\b(\w{3,})\b.*(?:word|say|repeat|sound|letter)", text, re.IGNORECASE,
        ):
            word = word.strip(".,!?'\"")
            if len(word) > 2 and word[0].isalpha() and self.sr and self.memory:
                self.sr.add_card(word, "", text[:100])
                self.memory.add_new_word(word, "", text[:100])
                logger.info("📝 New word tracked (emphatic): %s", word)

    def _check_child_errors(self, text: str) -> None:
        import re
        text_lower = text.lower().strip()
        for pattern, error_type in [
            (r"\bhe go\b|\bshe go\b|\bhe run\b|\bshe run\b|\bhe want\b|\bshe want\b",
             "subject-verb agreement (3rd person)"),
            (r"\ba apple\b|\ban cat\b|\ba elephant\b", "article (a/an confusion)"),
            (r"\bgoed\b|\brunned\b|\bswimmed\b|\bcatched\b", "irregular past tense"),
            (r"\bdon't have no\b|\bcan't find no\b", "double negative"),
            (r"\bhim go\b|\bher go\b|\bthem go\b", "pronoun (subject vs object)"),
            (r"\btwo book\b|\bthree dog\b|\bten cat\b", "plural -s missing"),
        ]:
            if re.search(pattern, text_lower) and self.memory:
                self.memory.record_error(error_type, text[:100])
                logger.info("📝 Error tracked: %s — \"%s\"", error_type, text[:60])

    # ── Helpers ──────────────────────────────────────────────────

    def _print_welcome(self) -> None:
        """Print the welcome banner."""
        print(f"   Tutor: {self.tutor.emoji} {self.tutor.name} ({self.tutor.voice})")
        print("   [Ctrl+C 退出]\n")

    def _notify_browser_ready(self) -> None:
        """Pre-warm browser static files for face preview."""
        try:
            import httpx as _hx
            paths = [
                "/static/face_preview.html",
                "/static/live2d/bundle.js",
                "/static/live2d/core/live2dcubismcore.min.js",
            ]
            for p in paths:
                try:
                    _hx.get(f"http://localhost:{self.config.dashboard_port}{p}", timeout=2)
                except Exception:
                    pass
            print(f"   🔗 http://localhost:{self.config.dashboard_port}/static/face_preview.html")
        except Exception:
            pass

    def _handle_sigint(self, sig, frame) -> None:
        """Handle Ctrl+C — initiate graceful shutdown."""
        logger.info("Received signal %s, shutting down...", sig)
        self.stop()
