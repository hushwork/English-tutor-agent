"""Command Filter — 拒识门禁（独立模块，尚未接入管线）。

设计方案见 docs/WAKEWORD_REJECTION_PLAN.md（方案 A：STT 后文本门禁）。

作用：在 STT 之后、LLM 之前拦截——转写文本不在指令表内就拒识，
不调用 LLM。指令表、模式、激活窗口、未命中行为均可配置。

使用方式（集成时在 local_pipe.process() 里）：

    filt = CommandFilter.load()
    decision = filt.check(text)
    if not decision.allowed:
        # decision.canned_reply 有值则直接 TTS 播报（不经 LLM），否则静默丢弃
        return

配置持久化：.camera-tutor-data/command_filter.json（见 config 字段注释）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── 配置 ────────────────────────────────────────────────────────

# on_reject 策略
REJECT_SILENT = "silent"            # 静默丢弃（车载式）
REJECT_CANNED = "canned_reply"      # 播固定话术（不经 LLM，儿童场景建议）
REJECT_ESCALATE = "escalate"        # 连续被拒 N 次后才播引导话术

# 匹配模式
MODE_COMMAND = "command"            # 整句须命中指令表
MODE_PREFIX = "prefix"              # 以唤醒前缀开头即放行（后面可接自由对话）

DEFAULT_CANNED_REPLY = "Please use one of my special commands!"
DEFAULT_ESCALATE_REPLY = "Remember — start with a special command, then I can talk!"


@dataclass
class CommandFilterConfig:
    enabled: bool = False            # 默认关闭：未配置前不影响现有行为
    mode: str = MODE_COMMAND
    commands: list[str] = field(default_factory=list)        # command 模式指令表
    wake_prefixes: list[str] = field(default_factory=list)   # prefix 模式前缀表
    activation_window_s: float = 10.0   # 命中后的放行窗口；0 = 纯单次（无状态）
    on_reject: str = REJECT_CANNED
    canned_reply: str = DEFAULT_CANNED_REPLY
    escalate_reply: str = DEFAULT_ESCALATE_REPLY
    escalate_after: int = 3          # escalate 策略下，连续被拒几次后才播引导
    fuzzy_tolerance: int = 1         # 编辑距离容差（0=精确匹配）


# ── 判定结果 ────────────────────────────────────────────────────

@dataclass
class RejectDecision:
    allowed: bool
    matched: Optional[str] = None    # 命中的指令/前缀（未命中为 None）
    reason: str = ""                 # 判定原因（日志/埋点用）
    canned_reply: Optional[str] = None   # 被拒时需播报的话术（silent 时为 None）


# ── 文本归一化与匹配 ────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s']+")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """小写、去标点、压缩空白。'What's this?!' -> \"what's this\""""
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def edit_distance(a: str, b: str, max_dist: int) -> int:
    """Levenshtein 距离，超过 max_dist 提前退出（返回 max_dist+1）。"""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def fuzzy_contains(words: list[str], phrase_words: list[str], tolerance: int) -> bool:
    """phrase 是否以"近似形态"出现在 words 中（滑动等长窗口 + 编辑距离）。"""
    if not phrase_words or len(words) < len(phrase_words):
        return False
    n = len(phrase_words)
    target = " ".join(phrase_words)
    for i in range(len(words) - n + 1):
        if edit_distance(" ".join(words[i:i + n]), target, tolerance) <= tolerance:
            return True
    return False


# ── 门禁本体 ────────────────────────────────────────────────────

class CommandFilter:
    """拒识门禁：check() 纯函数式判定 + 激活窗口/连拒计数内部状态。"""

    def __init__(self, config: CommandFilterConfig):
        self.config = config
        self._active_until: float = 0.0      # 激活窗口截止时间（time.time() 域）
        self._consecutive_rejects: int = 0   # escalate 策略计数

    # ── 配置读写 ──

    @staticmethod
    def default_path() -> Path:
        from camera_tutor.paths import data_dir
        return data_dir() / "command_filter.json"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CommandFilter":
        path = path or cls.default_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                valid = {f for f in CommandFilterConfig.__dataclass_fields__}
                return cls(CommandFilterConfig(**{k: v for k, v in data.items() if k in valid}))
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        return cls(CommandFilterConfig())

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.config), ensure_ascii=False, indent=2))

    # ── 判定 ──

    def check(self, text: str, now: Optional[float] = None) -> RejectDecision:
        """对一条 STT 转写做拒识判定。now 可注入便于测试。"""
        now = time.time() if now is None else now
        cfg = self.config

        if not cfg.enabled:
            return RejectDecision(True, reason="disabled")

        norm = normalize(text)
        if not norm:
            return self._reject(now, "empty")

        # 激活窗口内直接放行（窗口随每句放行刷新）
        if cfg.activation_window_s > 0 and now < self._active_until:
            self._active_until = now + cfg.activation_window_s
            self._consecutive_rejects = 0
            return RejectDecision(True, reason="activation_window")

        matched = self._match(norm)
        if matched is not None:
            if cfg.activation_window_s > 0:
                self._active_until = now + cfg.activation_window_s
            self._consecutive_rejects = 0
            return RejectDecision(True, matched=matched, reason="matched")

        return self._reject(now, "no_match")

    def _match(self, norm: str) -> Optional[str]:
        """返回命中的指令/前缀，未命中返回 None。"""
        cfg = self.config
        words = norm.split()
        tol = max(0, cfg.fuzzy_tolerance)

        if cfg.mode == MODE_PREFIX:
            for p in cfg.wake_prefixes:
                pn = normalize(p)
                if not pn:
                    continue
                pw = pn.split()
                if norm == pn or norm.startswith(pn + " "):
                    return p
                # 模糊：首词容差（如 emma → ema/amma）
                if tol and fuzzy_contains(words[: len(pw)], pw, tol):
                    return p
            return None

        # command 模式（默认）
        for c in cfg.commands:
            cn = normalize(c)
            if not cn:
                continue
            if cn in norm:
                return c
            if tol and fuzzy_contains(words, cn.split(), tol):
                return c
        return None

    def _reject(self, now: float, reason: str) -> RejectDecision:
        cfg = self.config
        self._consecutive_rejects += 1

        if cfg.on_reject == REJECT_SILENT:
            return RejectDecision(False, reason=reason)
        if cfg.on_reject == REJECT_ESCALATE:
            if self._consecutive_rejects >= cfg.escalate_after:
                self._consecutive_rejects = 0   # 播报后重新计数
                return RejectDecision(False, reason=f"{reason}_escalated",
                                      canned_reply=cfg.escalate_reply)
            return RejectDecision(False, reason=reason)
        # 默认 canned_reply
        return RejectDecision(False, reason=reason, canned_reply=cfg.canned_reply)

    # ── 测试/调试辅助 ──

    @property
    def is_active(self) -> bool:
        return time.time() < self._active_until

    def reset(self) -> None:
        """清空运行时状态（窗口、连拒计数）。"""
        self._active_until = 0.0
        self._consecutive_rejects = 0
