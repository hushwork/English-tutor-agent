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
    camera_resolution: tuple[int, int] = (1280, 720)
    camera_scene_threshold: float = 0.15
    camera_key_interval: float = 1.0

    # Audio
    mic_gain: float = 1.0         # Linear — leave at 1.0 unless diagnostic says otherwise
    mic_device_index: int | None = None  # Force a specific mic, or None for auto-detect
    spk_device_index: int | None = None  # Force a specific speaker, or None for system default
    agc_enabled: bool = True     # Digital AGC on the mic path
    server_vad_threshold: float = 0.5   # Server VAD sensitivity (lower = more sensitive)
    tts_speed: float = 1.0        # Speech rate: 0.25-4.0 (1.0 = normal)

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.workspace_id:
            self.workspace_id = os.environ.get("WORKSPACE_ID",
                "llm-xo2ff9jhvnvgvu6b")  # fallback — matches original realtime_demo.py
        if not self.model:
            self.model = os.environ.get("OMNI_MODEL", "qwen3.5-omni-flash-realtime")
        # Allow env override for mic gain
        env_gain = os.environ.get("MIC_GAIN")
        if env_gain:
            try:
                self.mic_gain = float(env_gain)
            except ValueError:
                pass

        if not self.workspace_id:
            logger.error(
                "WORKSPACE_ID is empty — WS URL will be malformed. "
                "Set WORKSPACE_ID in .env or pass workspace_id to AgentConfig."
            )

    @property
    def ws_url(self) -> str:
        """WebSocket URL。本地模式（OMNI_WS_URL 已设置）→ 本地 speech-to-speech
        （OpenAI Realtime 兼容，ws://localhost:8765/v1/realtime）；
        否则 → 阿里 MaaS 云端实时接口。"""
        override = os.environ.get("OMNI_WS_URL", "")
        if override:
            return override
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
        self._storage_dir: Path = Path(
            __file__).resolve().parent.parent / ".camera-tutor-data"

        # Track recent utterances for context
        self._last_child_utterance: str = ""
        self._last_emma_utterance: str = ""
        self._utterances_this_session: int = 0
        self._recent_emma_phrases: list[str] = []
        self._last_speech_time: float = 0.0

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
        self.audio = AudioManager(
            mic_gain=self.config.mic_gain,
            mic_device_index=self.config.mic_device_index,
            spk_device_index=self.config.spk_device_index,
            agc_enabled=self.config.agc_enabled,
        )

        # Face sync (dashboard connection)
        self.face_sync = FaceSyncManager()

        # Wire viseme handler: AudioManager will call this when audio plays
        self.audio.set_viseme_handler(self._on_viseme_play)

        # Memory & learning systems
        self.memory = ConversationMemory(storage_dir=self._storage_dir, user_id="camera_tutor")
        self.memory.new_session()
        logger.info("Conversation memory: %s", self.memory._data_dir)

        self.sr = SpacedRepetition(storage_dir=self._storage_dir, user_id="camera_tutor")
        logger.info("Spaced repetition: %s", self.sr._data_dir)

        self.reporter = ParentReportEngine(storage_dir=self._storage_dir)

        # Register signal handler
        if not self._signal_setup_done:
            signal.signal(signal.SIGINT, self._handle_sigint)
            signal.signal(signal.SIGTERM, self._handle_sigint)
            self._signal_setup_done = True

        logger.info("Agent setup complete")

    def _setup_camera(self) -> None:
        """Try to initialise the camera pipeline.

        The configured camera_id (from --camera or saved device config) is
        tried first; the rest of 0/1/2 are fallbacks.
        """
        preferred = self.config.camera_id
        candidates = [preferred] + [i for i in (0, 1, 2) if i != preferred]
        for cam_id in candidates:
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

        local_mode = bool(os.environ.get("OMNI_WS_URL", ""))
        if not api_key and not local_mode:
            logger.error("DASHSCOPE_API_KEY not set — set it in .env or pass to AgentConfig")
            sys.exit(1)

        if not self.config.workspace_id and not local_mode:
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

        # Save memory state
        if self.memory:
            try:
                self.memory._save_stats()
            except Exception:
                pass

        if hasattr(self, "_calib_chunks") and self._calib_chunks:
            self._save_calibration_wav()

        logger.info("Camera Tutor Agent stopped")

    def _save_calibration_wav(self) -> None:
        """Save accumulated audio chunks as WAV files for MFCC calibration."""
        import wave
        out_dir = self._storage_dir / "calibration"
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        while (out_dir / f"emma_{n:02d}.wav").exists():
            n += 1
        path = out_dir / f"emma_{n:02d}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(24000)
            wf.writeframes(b"".join(self._calib_chunks))
        logger.info("Calibration audio saved: %s (%d bytes)", path, sum(len(c) for c in self._calib_chunks))

    # ── WebSocket event handlers ─────────────────────────────────

    def _on_ws_open(self, ws) -> None:
        """Handle WebSocket open — configure session and start threads."""
        logger.info("WebSocket opened, configuring session...")

        # Configure session（本地 s2s 只需基础字段：voice/speed/transcription/silence 会报错）
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "modalities": ["text", "audio"],
                "instructions": self._build_instructions(),
                "audio": {"output": {"voice": "af_heart"}},
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.config.server_vad_threshold,
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
            # audio_ready=None: 本地 local_pipe 没有 Omni 云端那种
            # "append image before append audio" 的顺序要求，不需要等
            # 麦克风首发再推图，避免预览线程被无谓阻塞。
            self.vision = VisionManager(camera=self.camera, ws_getter=lambda: ws,
                                         audio_ready=None,
                                         session_ready=self.state.session_ready)
            self.vision.start()

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

        if event_type == "session.updated" or event_type == "session.created":
            self.state.session_ready.set()
            if not getattr(self, "_link_printed", False):
                self._link_printed = True
                self._notify_browser_ready()

        elif event_type == "input_audio_buffer.speech_started":
            self.state.audio_started.set()
            self._last_speech_time = time.time()

        # ── Audio playback + viseme sync ──
        elif event_type == "response.audio.delta":
            chunk = base64.b64decode(event["delta"])
            viseme_payloads = self._extract_viseme_payloads(chunk)
            self.audio.write_spk(chunk, visemes=viseme_payloads)
            # Calibration: save audio to file if env var is set
            if os.environ.get("SAVE_CALIBRATION_AUDIO"):
                if not hasattr(self, "_calib_chunks"):
                    self._calib_chunks = []
                self._calib_chunks.append(chunk)

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
                # Inject updated instructions so model knows what it recently said
                self._update_session_instructions()
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
                # NOTE: child transcript from ASR is unreliable (accent, noise).
                # Omni's semantic understanding is correct, but the text output
                # often has errors. We log/save it for reference but do NOT run
                # error detection or vocabulary extraction on it.
                if self.memory:
                    self.memory.save_message("user", transcript)

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
        # Wait for session.updated before sending anything — avoids
        # "Error append image before append audio" race on reconnect.
        logger.debug("Mic thread waiting for session.updated...")
        self.state.session_ready.wait(timeout=10.0)
        if not self.state.session_ready.is_set():
            logger.error("Session not ready after 10s — mic send aborted")
            return

        first = True
        level_check_at = time.time() + 5.0  # First level check after 5s
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

            # Periodic mic level monitoring — only warn if persistently dead
            now = time.time()
            if now >= level_check_at:
                level_check_at = now + 30.0  # Every 30s — less noisy
                rms = self.audio.mic_level_rms
                dbfs = self.audio.mic_level_dbfs
                # Only warn if the mic has been near-silent for 30s AND
                # server VAD hasn't triggered recently (no one is talking).
                # Silence during conversation pauses is normal.
                if rms < 5 and self._seconds_since_last_speech() > 30:
                    logger.warning(
                        "⚠️  Mic level VERY LOW for 30s+ (RMS=%.1f, %.1f dBFS). "
                        "Check mic connection, System Preferences > Sound > Input volume, "
                        "or increase mic_gain (currently %.1fx).",
                        rms, dbfs, self.audio.mic_gain,
                    )
                elif rms < 15:
                    logger.debug("Mic level: RMS=%.1f, %.1f dBFS (quiet — normal when silent)", rms, dbfs)
                else:
                    logger.debug("Mic level: RMS=%.1f, %.1f dBFS", rms, dbfs)

    def _extract_viseme_payloads(self, chunk: bytes) -> list[dict] | None:
        """Extract viseme payloads from an audio chunk.
        
        Returns pre-computed payload dicts. They are queued alongside
        the audio in AudioManager and pushed to FaceSync at play time.
        """
        try:
            from camera_tutor.spectral_viseme import chunk_to_visemes
            from camera_tutor.live2d_bridge import VisemeParams
            transcript = self.state.get_transcript()
            payloads = []
            for v in chunk_to_visemes(chunk, 24000):
                params = VisemeParams.from_viseme(v)
                payloads.append({
                    "viseme": v.label,
                    "mouth_open": params.mouth_open,
                    "mouth_width": params.mouth_width,
                    "tongue_visible": params.tongue_visible,
                    "transcript": transcript,
                })
            return payloads if payloads else None
        except Exception as e:
            logger.debug("Viseme extraction error: %s", e)
            return None

    def _on_viseme_play(self, payload: dict) -> None:
        """Called by AudioManager's drain thread when a viseme is due."""
        if self.face_sync:
            self.face_sync.push_payload(payload)

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

    def _update_session_instructions(self) -> None:
        """Send updated instructions to the model so it knows what it recently said."""
        if self.connection is None or self.connection.ws is None:
            return
        try:
            self.connection.ws.send(json.dumps({
                "type": "session.update",
                "session": {"instructions": self._build_instructions()},
            }))
        except Exception as e:
            logger.debug("Session update error: %s", e)

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

        # Recent phrases (use memory for full history, cap at 8 for prompt)
        recent_phrases = []
        repeated_warning = ""
        if self.memory:
            ctx = self.memory.get_context(max_messages=24)
            recent_phrases = [
                m["content"][:80]
                for m in ctx[-12:] if m.get("role") == "assistant"
            ][-8:]

            # — Same-text repetition guard —
            # If Emma said the exact same thing 2+ consecutive times, the
            # model is stuck (usually garbled audio). Force a topic change.
            if len(recent_phrases) >= 2:
                last = recent_phrases[-1]
                same_count = 0
                for p in reversed(recent_phrases):
                    if p == last:
                        same_count += 1
                    else:
                        break
                if same_count >= 2:
                    repeated_warning = (
                        f"\n⚠️  CRITICAL: You just said the EXACT same thing "
                        f"{same_count} times in a row! "
                        f"The child is trying to talk but the audio is unclear. "
                        f"CHANGE THE SUBJECT. Ask a DIFFERENT question. "
                        f"Or say: 'I didn't quite catch that — say it again?'\n"
                    )
        recent_line = ""
        if recent_phrases:
            recent_line = (
                f"\n⚠️  YOU ALREADY SAID THESE (SAY SOMETHING DIFFERENT):\n"
            )
            for i, p in enumerate(recent_phrases, 1):
                recent_line += f"  {i}. \"{p}\"\n"
            # Also check frequently-used words
            if self.memory:
                vocab = self.memory.get_vocabulary()
                if len(vocab) > 10:
                    top_words = [v["word"] for v in vocab[-5:]]
                    recent_line += (
                        f"\nWords you use a lot (try DIFFERENT ones): "
                        f"{', '.join(top_words)}\n"
                    )

        return (
            f"You are {self.tutor.name}, a {self.tutor.teaching_style} English tutor "
            f"for a {child_age}-year-old child.\n"
            f"Personality: {', '.join(self.tutor.personality_traits[:3])}.\n"
            f"Appearance: a {self.tutor.age_appearance}-year-old woman.\n\n"
            f"{self.tutor._tutor_rules()}\n\n"
            f"IMPORTANT: The child speaks ENGLISH. Transcribe as English.\n\n"
            f"VISION: You receive real-time camera images — use what you SEE.\n\n"
            f"YOUR MISSION: Be a friendly companion who happens to speak English.\n"
            f"Follow the child's lead. Chat, play, wonder — don't teach or quiz.\n\n"
            f"{session_info}{vocab_line}{errors_line}{due_line}{repeated_warning}{recent_line}"
            f"YOUR STYLE:\n"
            f"1. Short and natural — 1-2 sentences, like talking to a friend.\n"
            f"2. Switch it up: sometimes playful 🎨, sometimes curious 🔍,\n"
            f"   sometimes a storyteller 📖, sometimes quietly present 🤫.\n"
            f"3. Wonder out loud: 'I wonder what that does...' 'That looks fun!'\n"
            f"4. Start simple games: I-Spy, counting, finding colors.\n"
            f"5. NEVER repeat the same phrase twice. Be unpredictable in a good way.\n"
            f"6. Read the room: quiet child → be gentle, excited child → match energy.\n"
            f"7. Talk about the camera view like you're both looking out a window.\n"
            f"8. Sound like a friend — never like a textbook or a quiz.\n"
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
        """Extract new vocabulary from Emma's speech.

        Catches:
        1. Explicitly introduced words: 'This is a X', 'See the X'
        2. Words Emma asks the child to say: 'Can you say X?'
        3. All content words (nouns/adjectives/verbs) not already known
        """
        import re
        if not self.sr or not self.memory:
            return

        clean = text.strip().lower()
        existing = {v["word"].lower() for v in self.memory.get_vocabulary()}
        found: set[str] = set()

        # 1. Explicit introduction patterns
        intro = re.findall(
            r"\b(this is|that is|it's a|it's an|here is|there is|"
            r"see the|look at the|that's a|that's an|say the word|"
            r"can you say|try saying)\s+(\w+)",
            clean, re.IGNORECASE,
        )
        for _, word in intro:
            w = word.strip(".,!?'\"")
            if len(w) > 2 and w.isalpha() and w not in existing:
                found.add(w)

        # 2. Content nouns/adjectives only — skip common verbs/adverbs
        stop_words = {
            "the","and","for","that","you","are","this","all","not","but",
            "its","can","see","big","red","blue","like","good","great",
            "yes","now","one","two","too","let","did","get","has","had","have",
            "was","she","her","him","his","our","out","how","who",
            "what","when","where","why","here","there","come","want","going",
            "doing","been","very","just","about","your","from","with","will",
            "well","some","more","than","then","them","they","into","over",
            "hello","hi","hey","bye","please","thank","thanks","okay","yeah",
            "love","loves","make","made","look","looks","looking","say","says",
            "saying","ask","asking","tell","told","try","trying","know","think",
        }
        words = re.findall(r"\b([a-z]{3,})\b", clean)
        for w in words:
            if w not in stop_words and w not in existing:
                found.add(w)

        # Save all new words
        for w in found:
            self.sr.add_card(w, "", text[:100])
            self.memory.add_new_word(w, "", text[:100])
            logger.info("📝 New word: %s", w)

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

    def _seconds_since_last_speech(self) -> float:
        if self._last_speech_time == 0.0:
            return float("inf")
        return time.time() - self._last_speech_time

    def _handle_sigint(self, sig, frame) -> None:
        """Handle Ctrl+C — initiate graceful shutdown."""
        logger.info("Received signal %s, shutting down...", sig)
        self.stop()
