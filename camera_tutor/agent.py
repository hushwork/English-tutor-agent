"""CameraTutorAgent — 进程级运行时外壳（多用户并发练习）。

架构：Agent 只负责进程级资源 —— dashboard、信号处理、共享的报告引擎，
以及 A/V 来源（local 模式：本机 CameraPipeline + AudioManager；
webrtc 模式：RTCDeviceManager 多会话注册表）。每路练习会话
（一路 WebRTC peer，或本地模式的唯一会话）由 PracticeSession 承载，
见 camera_tutor/practice_session.py。

Lifecycle（对外签名不变，realtime_demo.py 直接调用）:
    agent = CameraTutorAgent()
    agent.setup()          # Init 进程级资源
    agent.start()          # 启动 dashboard；webrtc 模式阻塞等待浏览器接入，
                           # local 模式在主线程跑唯一会话的连接循环
    # ... runs until Ctrl+C or stop() ...
    agent.stop()           # Graceful shutdown（停所有 PracticeSession）
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from camera_tutor.audio_manager import AudioManager
from camera_tutor.camera import CameraPipeline
from camera_tutor.parent_report import ParentReportEngine
from camera_tutor.paths import data_dir
from camera_tutor.practice_session import AgentState, PracticeSession

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

    # A/V source: "local" = sounddevice/cv2 hardware on this machine;
    # "webrtc" = remote browser device (face_preview.html?device=1)
    av_source: str = "local"
    viseme_lead_ms: int = 80      # WebRTC mode: viseme delay to cover browser playout

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
        # A/V source override
        env_av = os.environ.get("AV_SOURCE")
        if env_av:
            self.av_source = env_av.strip().lower()
        env_lead = os.environ.get("VISME_LEAD_MS")
        if env_lead:
            try:
                self.viseme_lead_ms = int(env_lead)
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


# ── CameraTutorAgent ─────────────────────────────────────────────

class CameraTutorAgent:
    """进程级运行时：dashboard、信号、A/V 来源、PracticeSession 注册表。

    每个练习会话（WebRTC peer 或本地唯一会话）一个 PracticeSession；
    Agent 自身不再持有模型连接和每会话状态。
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.rtc = None  # RTCDeviceManager, only in av_source="webrtc" mode

        # 活跃练习会话 {session_id: PracticeSession}
        self._sessions: dict[str, PracticeSession] = {}
        self._sessions_lock = threading.Lock()

        # Local-mode hardware (None in webrtc mode)
        self.audio: Optional[AudioManager] = None
        self.camera: Optional[CameraPipeline] = None

        # 共享的家长报告引擎（各 PracticeSession 并发写，配锁）
        self.reporter: Optional[ParentReportEngine] = None
        self._reporter_lock = threading.Lock()
        self._storage_dir = data_dir()

        # Dashboard thread
        self._dashboard_thread: Optional[threading.Thread] = None

        # Signal handling
        self._signal_setup_done = False

    # ── Setup (called once before start) ─────────────────────────

    def setup(self) -> None:
        """Initialise 进程级资源（A/V 来源、共享 reporter、信号处理）。

        Does NOT start connections or threads — that happens in start()
        或浏览器 peer 接入时（webrtc 模式）。
        """
        # A/V source: local hardware or remote WebRTC device registry
        if self.config.av_source == "webrtc":
            from camera_tutor.rtc_device import RTCDeviceManager
            self.rtc = RTCDeviceManager(
                mic_gain=self.config.mic_gain,
                viseme_lead_ms=self.config.viseme_lead_ms,
            )
            self.rtc.set_session_hooks(
                on_created=self._on_rtc_session_created,
                on_closed=self._on_rtc_session_closed,
            )
            logger.info("A/V source: WebRTC remote device (multi-session)")
        else:
            if self.config.camera_enabled:
                self._setup_camera()
            self.audio = AudioManager(
                mic_gain=self.config.mic_gain,
                mic_device_index=self.config.mic_device_index,
                spk_device_index=self.config.spk_device_index,
                agc_enabled=self.config.agc_enabled,
            )

        # 共享的家长报告引擎（memory/sr 已改为每会话自建，见 PracticeSession）
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
        """Start dashboard，然后按模式运行：

        - webrtc：不主动连模型，阻塞等待浏览器 peer 接入（每路 peer 由
          _on_rtc_session_created 拉起独立 PracticeSession）。
        - local：创建唯一 PracticeSession，主线程跑连接循环（原行为）。
        """
        logger.info("Starting Camera Tutor Agent...")

        # Start dashboard server
        self._start_dashboard()

        if self.config.av_source == "webrtc":
            scheme = "https" if os.environ.get("DASHBOARD_TLS_CERT") else "http"
            print(f"   🌐 WebRTC 多用户模式：等待浏览器接入 "
                  f"{scheme}://localhost:{self.config.dashboard_port}/?device=1")
            print("   [Ctrl+C 退出]\n")
            while self.state.running:
                time.sleep(0.5)
            return

        # Local 模式：唯一会话，主线程阻塞连接循环
        self.audio.start()
        session = PracticeSession(
            config=self.config,
            user_id="camera_tutor",
            session_id="local",
            audio=self.audio,
            camera=self.camera,
            reporter=self.reporter,
            reporter_lock=self._reporter_lock,
        )
        with self._sessions_lock:
            self._sessions["local"] = session
        session.start_blocking()

    def stop(self) -> None:
        """Graceful shutdown of all components."""
        logger.info("Stopping Camera Tutor Agent...")
        self.state.running = False

        # Stop all practice sessions
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.stop()
            except Exception:
                logger.exception("Practice session stop error")

        # Stop local hardware (local mode only; RTC A/V 归各 RTCSession)
        if self.audio:
            self.audio.stop()
        if self.camera:
            self.camera.stop()

        logger.info("Camera Tutor Agent stopped")

    # ── RTC session hooks（uvicorn 事件循环线程，同步调用）────────

    def _on_rtc_session_created(self, rtc_session) -> None:
        """一路新 WebRTC peer：拉起独立的 PracticeSession。

        reporter 用 dashboard_server 的 per-user 注册表——WebRTC 模式下
        dashboard 与本进程同生共死，共享实例后 /api/report/* 才能看到
        当天的事件（report 数据在引擎内存里，隔天 rollover 才落盘）。
        """
        from camera_tutor.dashboard_server import report_engine_for
        rtc_session.audio.start()
        camera = None
        if self.config.camera_enabled:
            rtc_session.camera.start()
            camera = rtc_session.camera
        session = PracticeSession(
            config=self.config,
            user_id=rtc_session.user_id,
            session_id=rtc_session.session_id,
            audio=rtc_session.audio,
            camera=camera,
            reporter=report_engine_for(rtc_session.user_id),
            reporter_lock=self._reporter_lock,
        )
        with self._sessions_lock:
            self._sessions[rtc_session.session_id] = session
        # start() 必须在独立线程跑：本钩子运行在 uvicorn 事件循环上，
        # 而 PracticeSession.start() 里的 face_sync 会同步阻塞连 dashboard
        # 的 WS——同循环自连会互相等待直到超时（表现为 viseme/字幕全丢）。
        threading.Thread(
            target=session.start, daemon=True,
            name=f"session-start-{rtc_session.user_id}",
        ).start()
        logger.info("Practice session up: user=%s session=%s (%d active)",
                    rtc_session.user_id, rtc_session.session_id,
                    len(self._sessions))

    def _on_rtc_session_closed(self, rtc_session) -> None:
        """WebRTC peer 断开：停止并注销对应 PracticeSession。"""
        with self._sessions_lock:
            session = self._sessions.pop(rtc_session.session_id, None)
        if session is not None:
            # stop() 会 join 若干线程（秒级），同样不能阻塞事件循环
            threading.Thread(
                target=session.stop, daemon=True,
                name=f"session-stop-{rtc_session.user_id}",
            ).start()
            logger.info("Practice session down: user=%s session=%s (%d active)",
                        rtc_session.user_id, rtc_session.session_id,
                        len(self._sessions))

    # ── Dashboard lifecycle ──────────────────────────────────────

    def _start_dashboard(self, port: Optional[int] = None) -> None:
        """Start the dashboard server (if not already running)."""
        port = port or self.config.dashboard_port

        # Check if already running
        try:
            import httpx
            r = httpx.get(f"http://localhost:{port}/api/health", timeout=0.5)
            if r.status_code == 200:
                if self.config.av_source == "webrtc":
                    logger.error(
                        "Port %d already has a dashboard, but WebRTC device mode "
                        "requires the dashboard to run inside THIS process "
                        "(/rtc/offer signaling). Stop the other dashboard first.",
                        port,
                    )
                    sys.exit(1)
                logger.info("Dashboard already running on port %d", port)
                return
        except httpx.RequestError:
            pass

        # Register the RTC manager before uvicorn starts so /rtc/offer
        # is live as soon as the port accepts connections.
        if self.rtc is not None:
            from camera_tutor.rtc_device import set_rtc_manager
            set_rtc_manager(self.rtc)

        logger.info("Starting dashboard on port %d...", port)
        import uvicorn

        # TLS: required for getUserMedia from remote browsers
        # (getUserMedia needs a secure context; localhost is exempt).
        ssl_kwargs: dict = {}
        cert = os.environ.get("DASHBOARD_TLS_CERT", "")
        key = os.environ.get("DASHBOARD_TLS_KEY", "")
        if cert and key:
            ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
            logger.info("Dashboard TLS enabled (cert=%s)", cert)

        def _run_dashboard():
            uvicorn.run(
                "camera_tutor.dashboard_server:app",
                host="0.0.0.0",
                port=port,
                log_level="warning",
                **ssl_kwargs,
            )

        self._dashboard_thread = threading.Thread(
            target=_run_dashboard, daemon=True, name="dashboard",
        )
        self._dashboard_thread.start()

        # Wait for dashboard to be ready
        scheme = "https" if ssl_kwargs else "http"
        for i in range(30):
            try:
                import httpx
                r = httpx.get(f"{scheme}://localhost:{port}/api/health",
                              timeout=0.5, verify=False)
                if r.status_code == 200:
                    logger.info("Dashboard ready on %s://localhost:%d", scheme, port)
                    return
            except Exception:
                pass
            time.sleep(0.2)

        logger.warning("Dashboard start timeout — continuing without dashboard")

    # ── Signal handling ──────────────────────────────────────────

    def _handle_sigint(self, sig, frame) -> None:
        """Handle Ctrl+C — initiate graceful shutdown."""
        logger.info("Received signal %s, shutting down...", sig)
        self.stop()
