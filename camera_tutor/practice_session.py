"""PracticeSession — 单个练习会话（一路 WebRTC peer，或本地模式的唯一会话）。

从原 CameraTutorAgent 单体平移的每会话逻辑：WebSocket 事件处理、麦克风
发送线程、指令构建、学习追踪、viseme 同步。每个会话持有自己的 AgentState、
FaceSyncManager、ConversationMemory、SpacedRepetition 和 RealtimeConnection。

audio/camera 由调用方（webrtc 模式下来自 RTCSession，本地模式下来自运行时）
创建并拥有生命周期 —— PracticeSession 不 stop 它们。

多会话并发：reporter（ParentReportEngine）由运行时共享传入，log_event
调用一律经 reporter_lock 加锁。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from camera_tutor.connection import RealtimeConnection, ReconnectConfig
from camera_tutor.face_sync import FaceSyncManager
from camera_tutor.tutor_personas import get_active_tutor
from camera_tutor.vision_manager import VisionManager
from camera_tutor.parent_report import ParentReportEngine
from camera_tutor.paths import data_dir

# Memory & SR (self-contained camera_tutor modules)
from camera_tutor.memory import ConversationMemory
from camera_tutor.session_recorder import SessionRecorder
from camera_tutor.spaced_repetition import SpacedRepetition

if TYPE_CHECKING:
    from camera_tutor.agent import AgentConfig

logger = logging.getLogger(__name__)


# ── Session State ────────────────────────────────────────────────

@dataclass
class AgentState:
    """Thread-safe session state snapshot."""
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


# ── PracticeSession ──────────────────────────────────────────────

class PracticeSession:
    """一个用户的练习会话：自己的模型连接、记忆、学习追踪。

    生命周期由运行时（CameraTutorAgent）管理：start() / start_blocking()
    开始，stop() 幂等停止。audio/camera 的所有权在调用方。
    """

    def __init__(self, config: "AgentConfig", user_id: str, session_id: str,
                 audio, camera=None,
                 reporter: Optional[ParentReportEngine] = None,
                 reporter_lock: Optional[threading.Lock] = None,
                 fresh: bool = False, resume_from: str = ""):
        self.config = config
        self.user_id = user_id
        self.session_id = session_id

        # A/V seam（调用方拥有生命周期，本会话不 stop）
        self.audio = audio
        self.camera = camera

        # 共享的家长报告引擎（多会话并发写，需加锁）
        self.reporter = reporter
        self._reporter_lock = reporter_lock or threading.Lock()

        # Per-session state & sub-managers
        self.state = AgentState()
        self.vision: Optional[VisionManager] = None
        self.face_sync = FaceSyncManager(user_id=self.user_id)
        self.connection: Optional[RealtimeConnection] = None

        # Tutor（创建时快照；_maybe_switch_tutor 保留全局轮询热切换）
        self.tutor = get_active_tutor()

        # Mic audio sender thread (managed separately — tied to WS lifecycle)
        self._mic_thread: Optional[threading.Thread] = None
        self._mic_stop = threading.Event()

        # Memory & learning tracking（按用户隔离存储）
        self._storage_dir: Path = data_dir()
        self.memory = ConversationMemory(storage_dir=self._storage_dir, user_id=user_id)
        self.memory.new_session()
        logger.info("[%s] Conversation memory: %s", user_id, self.memory._data_dir)
        # 上一次会话的最近几轮（注入 instructions，让导师能"接着上次聊"；
        # 必须在 new_session 之后、本会话写入任何消息之前抓取）。
        # 优先级：fresh=True（"开始新对话"）→ 不注入；
        # resume_from 非空（"接着聊"某条历史对话）→ 注入该会话；
        # 否则延续上一次会话。
        if fresh:
            self._prev_session_tail: list[dict] = []
            logger.info("[%s] 新对话（不延续上次上下文）", user_id)
        elif resume_from:
            self._prev_session_tail = self.memory.get_session_messages(resume_from)
            logger.info("[%s] 续聊指定会话 %s（%d 条消息）",
                        user_id, resume_from, len(self._prev_session_tail))
        else:
            self._prev_session_tail = self.memory.get_previous_session_messages()
            if self._prev_session_tail:
                logger.info("[%s] 延续上次会话上下文（%d 条消息）",
                            user_id, len(self._prev_session_tail))
        self.sr = SpacedRepetition(storage_dir=self._storage_dir, user_id=user_id)
        logger.info("[%s] Spaced repetition: %s", user_id, self.sr._data_dir)

        # 会话录音：双向音频合成单个 16kHz WAV，文件名与历史面板的会话 id
        # 同名，dashboard 音频回放按此查找。RECORD_AUDIO=0 时完全不创建。
        self._recorder: Optional[SessionRecorder] = None
        if os.environ.get("RECORD_AUDIO", "1") != "0":
            try:
                rec_dir = self.memory._data_dir / "recordings"
                rec_path = rec_dir / f"session_{self.memory.session_id}.wav"
                self._recorder = SessionRecorder(rec_path)
                logger.info("[%s] Session recording: %s", user_id, rec_path)
            except Exception as e:
                logger.warning("[%s] 会话录音初始化失败（不录音）: %s", user_id, e)

        # Track recent utterances for context

        self._last_child_utterance: str = ""
        self._last_emma_utterance: str = ""
        self._utterances_this_session: int = 0
        self._recent_emma_phrases: list[str] = []
        self._last_speech_time: float = 0.0

        # 会话线程（start() 非阻塞模式）与停止标志
        self._practice_thread: Optional[threading.Thread] = None
        self._stopped = False

        # Wire viseme handler: audio 播放时回调（本地 PortAudio / RTC 发送时）
        self.audio.set_viseme_handler(self._on_viseme_play)

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """非阻塞启动：连接循环跑在 daemon 线程 practice-{user_id} 里。"""
        if not self._build_connection():
            return
        self.face_sync.start()
        self.face_sync.reset_viseme()
        self._practice_thread = threading.Thread(
            target=self.connection.start,
            name=f"practice-{self.user_id}",
            daemon=True,
        )
        self._practice_thread.start()
        logger.info("[%s] Practice session started (session=%s)",
                    self.user_id, self.session_id)

    def start_blocking(self) -> None:
        """阻塞启动：在当前线程跑连接循环（本地模式保持原行为）。"""
        if not self._build_connection():
            return
        self.face_sync.start()
        self.face_sync.reset_viseme()
        self.connection.start()

    def stop(self) -> None:
        """幂等停止：connection / mic 线程 / vision / face_sync，保存记忆。"""
        if self._stopped:
            return
        self._stopped = True
        logger.info("[%s] Stopping practice session...", self.user_id)
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

        # Save memory state
        if self.memory:
            try:
                self.memory._save_stats()
            except Exception:
                pass

        # 关闭会话录音（幂等；置 None 后 write 路径自然短路，
        # 残余线程即使拿到旧引用，close 后的写入也会被忽略）
        if self._recorder:
            self._recorder.close()
            self._recorder = None

        if hasattr(self, "_calib_chunks") and self._calib_chunks:
            self._save_calibration_wav()

        logger.info("[%s] Practice session stopped", self.user_id)

    def _build_connection(self) -> bool:
        """构建 RealtimeConnection；API key / workspace 缺失时记日志并返回 False
        （多会话共存，不能 sys.exit 杀掉整个进程）。"""
        ws_url = self.config.ws_url
        api_key = self.config.api_key

        local_mode = bool(os.environ.get("OMNI_WS_URL", ""))
        if not api_key and not local_mode:
            logger.error("[%s] DASHSCOPE_API_KEY not set — set it in .env or pass to AgentConfig",
                         self.user_id)
            return False

        if not self.config.workspace_id and not local_mode:
            logger.error(
                "[%s] WORKSPACE_ID is empty — cannot construct WS URL. "
                "Set WORKSPACE_ID in .env or pass workspace_id to AgentConfig.",
                self.user_id,
            )
            return False

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

        mode = "本地 local_pipe" if local_mode else "云端 MaaS"
        logger.info("[%s] Connecting to %s [%s] ...", self.user_id, ws_url, mode)
        return True

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

    def _log_report_event(self, event_type: str, data: dict | None = None) -> None:
        """写共享报告引擎（多会话并发，加锁）。"""
        if self.reporter is None:
            return
        with self._reporter_lock:
            self.reporter.log_event(event_type, data)

    # ── WebSocket event handlers ─────────────────────────────────

    def _on_ws_open(self, ws) -> None:
        """Handle WebSocket open — configure session and start threads."""
        logger.info("[%s] WebSocket opened, configuring session...", self.user_id)

        # Configure session（本地 s2s 只需基础字段：voice/speed/transcription/silence 会报错）
        audio_output = {"voice": self.tutor.voice}
        session = {
            "type": "realtime",
            "modalities": ["text", "audio"],
            "instructions": self._build_instructions(),
            "audio": {"output": audio_output},
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.config.server_vad_threshold,
            },
        }
        if os.environ.get("OMNI_WS_URL", ""):
            # 本地 local_pipe 专有字段（云端 Omni 不认，不发）：按角色传语速
            audio_output["speed"] = getattr(self.tutor, "speed", 1.0)
        ws.send(json.dumps({
            "type": "session.update",
            "session": session,
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
            name=f"mic-send-{self.user_id}",
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
            # 会话录音：TTS 下行（24kHz→16kHz 重采样在 recorder 内完成）
            rec = self._recorder
            if rec:
                rec.write_tts(chunk)
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
                self._log_report_event("emma_spoke", {"text": transcript[:200]})

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
                self._log_report_event("child_spoke",
                                       {"transcript": transcript[:200]})

        elif event_type == "error":
            err = event.get("error", {})
            logger.error(
                "API Error: %s (code=%s)",
                err.get("message", "unknown"),
                err.get("code", ""),
            )

    def _on_ws_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        logger.error("[%s] WebSocket error: %s", self.user_id, error)

    def _on_ws_close(self, ws, status, msg) -> None:
        """Handle WebSocket close — stop vision manager."""
        logger.info("[%s] WebSocket closed (code=%s)", self.user_id, status)
        if self.vision:
            self.vision.stop()
        self._mic_stop.set()

    # ── Internal: threads ────────────────────────────────────────

    def _mic_send_loop(self, ws) -> None:
        """Continuously read mic audio and send to WebSocket."""
        # Wait for session.updated before sending anything — avoids
        # "Error append image before append audio" race on reconnect.
        logger.debug("[%s] Mic thread waiting for session.updated...", self.user_id)
        self.state.session_ready.wait(timeout=10.0)
        if not self.state.session_ready.is_set():
            logger.error("[%s] Session not ready after 10s — mic send aborted", self.user_id)
            return

        first = True
        level_check_at = time.time() + 5.0  # First level check after 5s
        while not self._mic_stop.is_set() and self.state.running:
            data = self.audio.read_mic()
            if data is None:
                time.sleep(0.01)
                continue
            # 会话录音：mic 上行直接落盘（close 后写入被忽略，stop 后安全）
            rec = self._recorder
            if rec:
                rec.write_mic(data)
            try:
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode(),
                }))
                if first:
                    first = False
                    self.state.audio_started.set()
            except Exception as e:
                logger.warning("[%s] Mic send error: %s", self.user_id, e)
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
        """Called by the audio drain thread when a viseme is due."""
        if self.face_sync:
            self.face_sync.push_payload(payload)

    # ── Rich instruction builder ──────────────────────────────────

    def _update_session_instructions(self) -> None:
        """Send updated instructions to the model so it knows what it recently said."""
        self._maybe_switch_tutor()
        if self.connection is None or self.connection.ws is None:
            return
        try:
            self.connection.ws.send(json.dumps({
                "type": "session.update",
                "session": {"instructions": self._build_instructions()},
            }))
        except Exception as e:
            logger.debug("Session update error: %s", e)

    def _maybe_switch_tutor(self) -> None:
        """Hot-swap the tutor persona when the dashboard selection changes.

        tutor_prefs.json 是唯一事实来源（dashboard 切换导师时写它）。
        每轮对话后检查一次：新人设的指令下一轮即生效；本地管道
        （local_pipe）同时热切换音色/语速，云端 Omni 不支持会话中
        换音色，重连后生效。无需重启 agent。
        """
        try:
            current = get_active_tutor()
        except Exception:
            return
        if current.id == self.tutor.id:
            return
        logger.info("[%s] Tutor switched: %s → %s", self.user_id, self.tutor.name, current.name)
        self.tutor = current
        if not (self.connection and self.connection.ws):
            return
        if os.environ.get("OMNI_WS_URL", ""):
            # 本地 local_pipe 专有：会话中直接换音色/语速
            try:
                self.connection.ws.send(json.dumps({
                    "type": "session.update",
                    "session": {"audio": {"output": {
                        "voice": current.voice,
                        "speed": getattr(current, "speed", 1.0),
                    }}},
                }))
            except Exception as e:
                logger.debug("Tutor voice switch error: %s", e)

    def _build_instructions(self) -> str:
        """Build a rich system prompt with child profile and vocabulary."""
        if getattr(self.tutor, "audience", "child") == "adult":
            return self._build_adult_instructions()
        child_age = self._get_child_age()

        common_errors = self._get_common_errors()
        session_info = self._get_session_info()

        errors_line = ""
        if common_errors:
            errors_line = f"\nCHILD'S COMMON ERRORS:\n{common_errors}\n"

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
            f"{session_info}{self._get_prev_session_line()}{errors_line}{repeated_warning}{recent_line}"
            f"YOUR STYLE:\n"
            f"1. Short and natural, like talking to a friend.\n"
            f"2. Switch it up: sometimes playful 🎨, sometimes curious 🔍,\n"
            f"   sometimes a storyteller 📖, sometimes quietly present 🤫.\n"
            f"3. Wonder out loud: 'I wonder what that does...' 'That looks fun!'\n"
            f"4. Start simple games: I-Spy, counting, finding colors.\n"
            f"5. NEVER repeat the same phrase twice. Be unpredictable in a good way.\n"
            f"6. Read the room: quiet child → be gentle, excited child → match energy.\n"
            f"7. Talk about the camera view like you're both looking out a window.\n"
            f"8. Sound like a friend — never like a textbook or a quiz.\n"
        )

    def _build_adult_instructions(self) -> str:
        """Lean system prompt for adult-facing personas (e.g. interview coach).

        成人不需要儿童那套约束（句长上限、词汇表、夸奖规则），
        prompt 保持精简，小模型的指令遵循更可靠。
        """
        # 复用防复读机制：把最近说过的话列出来，避免重复提问
        recent_line = ""
        if self.memory:
            ctx = self.memory.get_context(max_messages=24)
            recent = [
                m["content"][:80]
                for m in ctx[-12:] if m.get("role") == "assistant"
            ][-6:]
            if recent:
                recent_line = "\nYOU ALREADY ASKED/SAID THESE (don't repeat):\n"
                for i, p in enumerate(recent, 1):
                    recent_line += f"  {i}. \"{p}\"\n"

        return (
            f"You are {self.tutor.name}, a {self.tutor.teaching_style} "
            f"English interview coach for adults.\n"
            f"Personality: {', '.join(self.tutor.personality_traits[:3])}.\n\n"
            f"{self.tutor._tutor_rules()}\n\n"
            f"IMPORTANT: The user speaks ENGLISH. Transcribe as English.\n"
            f"{self._get_prev_session_line()}"
            f"{recent_line}\n"
            f"YOUR STYLE:\n"
            f"1. Speak naturally — as long as the point needs, no filler.\n"
            f"2. One question at a time; let the user finish before feedback.\n"
            f"3. Balance encouragement with specific, actionable corrections.\n"
            f"4. NEVER repeat the same question twice in a session.\n"
            f"5. This is a LIVE VOICE conversation — no stage directions, "
            f"no parenthesized actions. Just talk.\n"
        )

    def _get_child_age(self) -> int:
        from camera_tutor.tutor_personas import get_child_age as _get_age
        return _get_age()

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

    def _get_prev_session_line(self) -> str:
        """上次会话的最近几轮对话（重启/重连后延续上下文）。"""
        if not self._prev_session_tail:
            return ""
        lines = ["LAST SESSION (continue naturally from where you left off):"]
        for m in self._prev_session_tail:
            who = "You" if m.get("role") == "assistant" else "Learner"
            lines.append(f"  {who}: {m.get('content', '')[:120]}")
        return "\n".join(lines) + "\n\n"

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
        print(f"   [{self.user_id}] Tutor: {self.tutor.emoji} {self.tutor.name} ({self.tutor.voice})")
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
            scheme = "https" if os.environ.get("DASHBOARD_TLS_CERT") else "http"
            print(f"   🔗 {scheme}://localhost:{self.config.dashboard_port}/static/face_preview.html")
            if self.config.av_source == "webrtc":
                print(f"   📱 设备端（远程采集）: {scheme}://{self._lan_ip()}:"
                      f"{self.config.dashboard_port}/static/face_preview.html?device=1")
        except Exception:
            pass

    @staticmethod
    def _lan_ip() -> str:
        """Best-effort LAN IP for the device-mode URL (no traffic sent)."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "<本机IP>"

    def _seconds_since_last_speech(self) -> float:
        if self._last_speech_time == 0.0:
            return float("inf")
        return time.time() - self._last_speech_time
