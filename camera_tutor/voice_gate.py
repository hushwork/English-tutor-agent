"""Voice Gate — 统一语音门禁。

把方案 A（文本拒识）和方案 B（KWS 唤醒词）收敛成一个门禁对象，
配置文件 `.camera-tutor-data/voice_gate.json` 里用 `mode` 选择：

- "off"      ：不放行任何门禁（现状，默认）
- "text"     ：方案 A——音频自由进入，STT 后按指令表拒识（command_filter）
- "kws"      ：方案 B——声学唤醒词门禁；未唤醒时音频根本不进 VAD/STT/LLM，
               唤醒后开门 N 秒自由对话
- "kws+text" ：叠加——唤醒词开门，每句话仍过文本指令表

KWS 后端默认用 openWakeWord（需 `pip install openwakeword`，模型需自备
或按 docs/WAKEWORD_REJECTION_PLAN.md 方案 B 训练）。检测器通过构造参数
注入，便于测试和替换后端。

集成方式（local_pipe.py）：全局 GATE 作配置模板并按 mtime 热重载；
每个 WebSocket 连接用 `GATE.new_session()` 派生独立实例：
    gate = GATE.new_session()
    # handler 收音频时：
    if gate.feed_audio(pcm_int16):   # kws 模式下未唤醒返回 False，直接丢弃
        buf.add(pcm_bytes)
    # process() 里 STT 之后：
    decision = gate.check_text(text)
    if not decision.allowed:
        ...播报 decision.canned_reply 或静默丢弃
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from camera_tutor.command_filter import (
    CommandFilter, CommandFilterConfig, RejectDecision,
)

# 门禁模式
MODE_OFF = "off"
MODE_TEXT = "text"
MODE_KWS = "kws"
MODE_KWS_TEXT = "kws+text"

SR = 16000
KWS_FRAME = 1280          # openWakeWord 单帧 80ms @16kHz


# ── 配置 ────────────────────────────────────────────────────────

@dataclass
class KwsConfig:
    model_paths: list[str] = field(default_factory=list)  # .onnx 模型路径（空=内置模型）
    threshold: float = 0.5          # 唤醒判定阈值（sensitivity 的反义：越高越严）
    activation_window_s: float = 10.0   # 唤醒后开门时长
    score_cooldown_s: float = 1.0   # 触发后的去抖间隔，防止同一声唤醒重复刷新


@dataclass
class VoiceGateConfig:
    mode: str = MODE_OFF
    text: CommandFilterConfig = field(default_factory=CommandFilterConfig)
    kws: KwsConfig = field(default_factory=KwsConfig)


# ── KWS 检测器抽象 ──────────────────────────────────────────────

class KwsDetector(Protocol):
    """声学唤醒词检测器。输入一帧 80ms int16 PCM，返回 0~1 的唤醒分数。"""

    def score(self, frame: np.ndarray) -> float: ...


class OpenWakeWordDetector:
    """openWakeWord 后端（延迟导入，未安装时给出明确报错）。"""

    def __init__(self, cfg: KwsConfig):
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise RuntimeError(
                "kws 模式需要 openwakeword：pip install openwakeword "
                "（模型文件见 docs/WAKEWORD_REJECTION_PLAN.md 方案 B）"
            ) from e
        kwargs = {"inference_framework": "onnx"}
        if cfg.model_paths:
            kwargs["wakeword_models"] = cfg.model_paths
        self._model = Model(**kwargs)
        self._last_scores: dict[str, float] = {}

    def score(self, frame: np.ndarray) -> float:
        pred = self._model.predict(frame)
        self._last_scores = dict(pred)
        return max(pred.values()) if pred else 0.0


# ── 统一门禁 ────────────────────────────────────────────────────

class VoiceGate:
    """A/B 方案统一入口。音频侧 feed_audio()，文本侧 check_text()。"""

    def __init__(self, config: VoiceGateConfig, detector: Optional[KwsDetector] = None):
        self.config = config
        self.text_filter = CommandFilter(config.text)
        self._detector = detector
        self._door_until: float = 0.0      # KWS 开门截止时间
        self._last_trigger: float = 0.0    # 去抖
        self._frame_buf = np.zeros(0, dtype=np.int16)  # 凑 80ms 帧的残 buffer

    # ── 配置读写 ──

    @staticmethod
    def default_path() -> Path:
        from camera_tutor.paths import data_dir
        return data_dir() / "voice_gate.json"

    @classmethod
    def load(cls, path: Optional[Path] = None,
             detector: Optional[KwsDetector] = None) -> "VoiceGate":
        path = path or cls.default_path()
        cfg = VoiceGateConfig()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                cfg = cls._config_from_dict(data)
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        return cls(cfg, detector=detector)

    @staticmethod
    def _config_from_dict(data: dict) -> VoiceGateConfig:
        text_fields = set(CommandFilterConfig.__dataclass_fields__)
        kws_fields = set(KwsConfig.__dataclass_fields__)
        text_data = {k: v for k, v in (data.get("text") or {}).items() if k in text_fields}
        kws_data = {k: v for k, v in (data.get("kws") or {}).items() if k in kws_fields}
        return VoiceGateConfig(
            mode=data.get("mode", MODE_OFF),
            text=CommandFilterConfig(**text_data),
            kws=KwsConfig(**kws_data),
        )

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self.config)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ── 音频侧（KWS 门） ──

    def new_session(self) -> "VoiceGate":
        """派生每连接独立的门禁实例（多用户并发用）。

        共享配置与 KWS 检测器（底层模型只加载一份），但开门状态、
        80ms 帧缓冲、文本指令表窗口各自独立——否则 A 用户的唤醒会给
        B 用户开门，两路音频的帧还会交错进同一打分缓冲，互相污染。
        """
        if self.config.mode in (MODE_KWS, MODE_KWS_TEXT):
            # 确保模型只加载一次、各会话共享。加载失败绝不能抛出——
            # 否则 local_pipe 的 WS handler 建连即死（1011 重连循环）；
            # detector 保持 None，后续 feed_audio 的惰性加载由
            # _gate_feeds 捕获异常并 fail-open 放行
            try:
                self._get_detector()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "KWS 检测器加载失败，本进程门禁 fail-open: %s", e)
        return VoiceGate(self.config, detector=self._detector)

    def _get_detector(self) -> KwsDetector:
        if self._detector is None:
            self._detector = OpenWakeWordDetector(self.config.kws)
        return self._detector

    def feed_audio(self, pcm_int16: np.ndarray, now: Optional[float] = None) -> bool:
        """喂入一块 int16 PCM（任意长度），返回当前是否放行音频。

        off/text 模式恒 True；kws 模式未唤醒时 False（音频应被丢弃）。
        """
        mode = self.config.mode
        if mode in (MODE_OFF, MODE_TEXT):
            return True

        now = time.time() if now is None else now
        cfg = self.config.kws

        # 凑 80ms 帧逐帧打分
        self._frame_buf = np.concatenate([self._frame_buf, pcm_int16])
        detector = self._get_detector()
        while len(self._frame_buf) >= KWS_FRAME:
            frame, self._frame_buf = self._frame_buf[:KWS_FRAME], self._frame_buf[KWS_FRAME:]
            score = detector.score(frame)
            if score >= cfg.threshold and now - self._last_trigger >= cfg.score_cooldown_s:
                self._last_trigger = now
                self._door_until = now + cfg.activation_window_s

        return now < self._door_until

    @property
    def door_open(self) -> bool:
        return time.time() < self._door_until

    # ── 文本侧（指令表门禁） ──

    def check_text(self, text: str, now: Optional[float] = None) -> RejectDecision:
        """STT 之后调用。off/kws 模式直接放行；text/kws+text 过指令表。"""
        mode = self.config.mode
        if mode in (MODE_OFF, MODE_KWS):
            return RejectDecision(True, reason=f"mode_{mode}_passthrough")
        return self.text_filter.check(text, now=now)

    def reset(self) -> None:
        self._door_until = 0.0
        self._last_trigger = 0.0
        self._frame_buf = np.zeros(0, dtype=np.int16)
        self.text_filter.reset()
