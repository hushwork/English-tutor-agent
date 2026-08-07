#!/usr/bin/env python3
"""voice_gate 统一门禁单元测试（独立脚本，直接 python3 运行，非 pytest）。

KWS 检测器用 FakeDetector 注入（不依赖 openwakeword 和真实模型），
覆盖：四种模式、开门/关门/去抖、帧拼接、文本侧联动、配置读写。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_tutor.voice_gate import (
    VoiceGate, VoiceGateConfig, KwsConfig, KWS_FRAME,
    MODE_OFF, MODE_TEXT, MODE_KWS, MODE_KWS_TEXT,
)
from camera_tutor.command_filter import CommandFilterConfig

passed = failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


class FakeDetector:
    """脚本化打分的假 KWS 检测器：逐帧消费 scores 列表（循环）。"""
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def score(self, frame):
        assert len(frame) == KWS_FRAME, f"帧长应为 {KWS_FRAME}，实际 {len(frame)}"
        s = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return s


def kws_gate(scores, window=10.0, cooldown=1.0, mode=MODE_KWS, text_cfg=None):
    cfg = VoiceGateConfig(
        mode=mode,
        kws=KwsConfig(threshold=0.5, activation_window_s=window,
                      score_cooldown_s=cooldown),
        text=text_cfg or CommandFilterConfig(),
    )
    return VoiceGate(cfg, detector=FakeDetector(scores))


def silent_chunk():
    return np.zeros(KWS_FRAME, dtype=np.int16)


# ── off 模式 ──
print("== off ==")
g = kws_gate([0.0], mode=MODE_OFF)
check("off 音频全放行", g.feed_audio(silent_chunk(), now=1000))
check("off 文本全放行", g.check_text("anything", now=1000).allowed)

# ── text 模式：音频不设卡，文本过指令表 ──
print("== text ==")
text_cfg = CommandFilterConfig(enabled=True, commands=["what's this"],
                               activation_window_s=0, fuzzy_tolerance=0)
g = kws_gate([0.0], mode=MODE_TEXT, text_cfg=text_cfg)
check("text 音频恒放行（不消耗检测器）", g.feed_audio(silent_chunk(), now=1000)
      and g._detector.calls == 0)
check("text 命中放行", g.check_text("what's this", now=1000).allowed)
check("text 未命中拒识", not g.check_text("blah", now=1000).allowed)

# ── kws 模式：唤醒开门 ──
print("== kws ==")
g = kws_gate([0.1], window=10.0)
check("未唤醒拒音频", not g.feed_audio(silent_chunk(), now=1000))

g = kws_gate([0.9] + [0.1] * 50, window=10.0)
check("唤醒帧开门", g.feed_audio(silent_chunk(), now=1000))
check("窗口内放行", g.feed_audio(silent_chunk(), now=1005))
check("窗口过期关门", not g.feed_audio(silent_chunk(), now=1012))
check("kws 文本侧直通", g.check_text("free talk", now=1000).allowed)

# 去抖：1 秒内重复高分不刷新截止时间
g = kws_gate([0.9] * 100, window=10.0, cooldown=1.0)
g.feed_audio(silent_chunk(), now=1000)       # 触发 → door=1010
g.feed_audio(silent_chunk(), now=1000.5)     # 冷却期内，不刷新
check("去抖期不刷新", abs(g._door_until - 1010) < 1e-9)
g.feed_audio(silent_chunk(), now=1001.5)     # 冷却期外，刷新
check("冷却期外刷新", abs(g._door_until - 1011.5) < 1e-9)

# 帧拼接：不足一帧的余量应缓存到下次
g = kws_gate([0.1, 0.9], window=10.0)
half = KWS_FRAME // 2
check("半帧不开门", not g.feed_audio(np.zeros(half, dtype=np.int16), now=1000))
check("补齐后按帧打分开门", g.feed_audio(np.zeros(half + KWS_FRAME, dtype=np.int16), now=1001))

# ── kws+text 叠加 ──
print("== kws+text ==")
text_cfg = CommandFilterConfig(enabled=True, commands=["what's this"],
                               activation_window_s=0, fuzzy_tolerance=0)
g = kws_gate([0.9] + [0.1] * 50, mode=MODE_KWS_TEXT, text_cfg=text_cfg)
check("叠加：唤醒开门", g.feed_audio(silent_chunk(), now=1000))
check("叠加：文本命中放行", g.check_text("what's this", now=1001).allowed)
check("叠加：文本未命中仍拒", not g.check_text("blah", now=1001).allowed)

# ── 配置读写 ──
print("== 持久化 ==")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "voice_gate.json"
    g = kws_gate([0.0], mode=MODE_KWS_TEXT, text_cfg=text_cfg)
    g.config.kws.threshold = 0.7
    g.save(p)
    g2 = VoiceGate.load(p, detector=FakeDetector([0.0]))
    check("mode 往返", g2.config.mode == MODE_KWS_TEXT)
    check("kws 配置往返", abs(g2.config.kws.threshold - 0.7) < 1e-9)
    check("text 配置往返", g2.config.text.commands == ["what's this"])
    g3 = VoiceGate.load(Path(td) / "none.json", detector=FakeDetector([0.0]))
    check("文件缺失回退 off", g3.config.mode == MODE_OFF)

print(f"\n{'=' * 40}\n结果: {passed} 通过, {failed} 失败")
sys.exit(0 if failed == 0 else 1)
