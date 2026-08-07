#!/usr/bin/env python3
"""command_filter 拒识门禁单元测试（独立脚本，直接 python3 运行，非 pytest）。

覆盖：归一化、精确/模糊匹配、command/prefix 两种模式、激活窗口、
未命中三种策略、配置读写往返。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_tutor.command_filter import (
    CommandFilter, CommandFilterConfig, normalize, edit_distance, fuzzy_contains,
    MODE_COMMAND, MODE_PREFIX, REJECT_SILENT, REJECT_CANNED, REJECT_ESCALATE,
)

passed = failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


# ── 基础函数 ──
print("== normalize / edit_distance / fuzzy_contains ==")
check("normalize 去标点小写", normalize("What's THIS?!  Red car.") == "what's this red car")
check("normalize 空白压缩", normalize("  hello   world ") == "hello world")
check("edit_distance 基本", edit_distance("look", "book", 1) == 1)
check("edit_distance 超限早退", edit_distance("look", "xxxx", 1) == 2)
check("fuzzy_contains 命中", fuzzy_contains("emma look at this".split(), ["look"], 0))
check("fuzzy_contains 模糊命中", fuzzy_contains("what is book".split(), ["look"], 1))
check("fuzzy_contains 超容差不中", not fuzzy_contains("what is book".split(), ["look"], 0))

# ── 关闭状态：全部放行 ──
print("== disabled ==")
f = CommandFilter(CommandFilterConfig(enabled=False))
check("disabled 放行一切", f.check("random noise", now=1000).allowed)

# ── command 模式 ──
print("== command 模式 ==")
cfg = CommandFilterConfig(
    enabled=True, mode=MODE_COMMAND,
    commands=["what's this", "sing a song"],
    activation_window_s=10, fuzzy_tolerance=1,
)
f = CommandFilter(cfg)

d = f.check("What's this?", now=1000)
check("精确命中", d.allowed and d.matched == "what's this")

f.reset()
d = f.check("whats this", now=1000)          # STT 常丢撇号
check("归一化后命中", d.allowed)

f.reset()
d = f.check("what's thi", now=1000)          # 编辑距离 1（少一个 s）
check("模糊命中(容差1)", d.allowed and d.matched == "what's this")

f.reset()
d = f.check("what's those", now=1000)        # 距离 2
check("超容差拒识", not d.allowed and d.canned_reply)

f.reset()
d = f.check("blah blah", now=1000)
check("无关语音拒识", not d.allowed and d.reason == "no_match")

# ── 激活窗口 ──
print("== 激活窗口 ==")
f = CommandFilter(cfg)
f.check("what's this", now=1000)             # 命中，开窗 [1000,1010)
d = f.check("why is it red", now=1005)       # 窗口内自由追问
check("窗口内放行", d.allowed and d.reason == "activation_window")
d = f.check("tell me more", now=1012)        # 窗口随上次放行刷新 [1012,1022)
check("窗口刷新后仍放行", d.allowed)
d = f.check("anything", now=1023)            # 已过期
check("窗口过期后拒识", not d.allowed)

cfg_nowindow = CommandFilterConfig(
    enabled=True, mode=MODE_COMMAND, commands=["what's this"],
    activation_window_s=0, fuzzy_tolerance=0,
)
f = CommandFilter(cfg_nowindow)
f.check("what's this", now=1000)
d = f.check("why", now=1001)
check("窗口=0 为纯单次模式", not d.allowed)

# ── prefix 模式 ──
print("== prefix 模式 ==")
cfg_p = CommandFilterConfig(
    enabled=True, mode=MODE_PREFIX, wake_prefixes=["emma"],
    activation_window_s=0, fuzzy_tolerance=1,
)
f = CommandFilter(cfg_p)
d = f.check("Emma, look at my car!", now=1000)
check("前缀放行自由对话", d.allowed and d.matched == "emma")
d = f.check("emmaline is nice", now=1000)    # 不能误中词内子串
check("词边界保护", not d.allowed)
d = f.check("ema look at this", now=1000)    # 首词模糊
check("前缀模糊命中", d.allowed)
d = f.check("look at this", now=1000)
check("无前缀拒识", not d.allowed)

# ── 未命中策略 ──
print("== on_reject 策略 ==")
f = CommandFilter(CommandFilterConfig(enabled=True, commands=["x"], on_reject=REJECT_SILENT))
d = f.check("hello there", now=1000)
check("silent 无话术", not d.allowed and d.canned_reply is None)

f = CommandFilter(CommandFilterConfig(
    enabled=True, commands=["xyz"], on_reject=REJECT_ESCALATE, escalate_after=3,
    fuzzy_tolerance=0))   # 精确匹配，避免超短指令+容差1把一切误判为命中
r1 = f.check("a", now=1000)
r2 = f.check("b", now=1001)
r3 = f.check("c", now=1002)
check("escalate 前两次静默", r1.canned_reply is None and r2.canned_reply is None)
check("escalate 第三次播引导", r3.canned_reply is not None)
r4 = f.check("d", now=1003)
check("escalate 播报后重新计数", r4.canned_reply is None)

# ── 配置读写往返 ──
print("== 持久化 ==")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "command_filter.json"
    f = CommandFilter(cfg)
    f.save(p)
    f2 = CommandFilter.load(p)
    check("读写往返一致", f2.config.commands == cfg.commands
          and f2.config.activation_window_s == cfg.activation_window_s
          and f2.config.enabled)
    f3 = CommandFilter.load(Path(td) / "nonexistent.json")
    check("文件缺失回退默认(关闭)", not f3.config.enabled)

print(f"\n{'=' * 40}\n结果: {passed} 通过, {failed} 失败")
sys.exit(0 if failed == 0 else 1)
