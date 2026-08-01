#!/usr/bin/env python3
"""Camera Tutor — 实时语音对话 (WebSocket).

Thin entry point for CameraTutorAgent. The agent class handles
all lifecycle, connection management, and sub-manager orchestration.

运行:
  python3 camera_tutor/realtime_demo.py                     # 默认（mac 原版行为）
  python3 camera_tutor/realtime_demo.py --select-devices    # 启动时菜单式选择 麦克风/扬声器/摄像头
  python3 camera_tutor/realtime_demo.py --mic 7 --spk 7 --camera 0
  python3 camera_tutor/realtime_demo.py --agc               # 开启麦克风 AGC 自动增益

依赖:
  pip install websocket-client sounddevice numpy
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from camera_tutor.agent import CameraTutorAgent, AgentConfig


def _probe_cameras(max_index: int = 8) -> list[int]:
    """Probe camera indexes; stop after 2 consecutive misses (no noise).

    OpenCV's C++ logger writes straight to fd 2, so silence it at the
    file-descriptor level for the probe window.
    """
    import cv2
    cams: list[int] = []
    misses = 0
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)  # fd 2 -> /dev/null
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened()
            cap.release()
            if ok:
                cams.append(i)
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    break
    finally:
        os.dup2(saved_fd, 2)  # restore stderr
        os.close(devnull_fd)
        os.close(saved_fd)
    return cams


def select_menu(title: str, items: list[tuple[int, str]]) -> int | None:
    """Show a numbered menu; return the picked device id (None = skip).

    items: list of (device_id, human-readable label).
    """
    print(f"\n{title}", flush=True)
    for i, (_dev_id, label) in enumerate(items):
        print(f"  [{i}] {label}", flush=True)
    while True:
        ans = input(f"请输入编号 [0-{len(items) - 1}，直接回车跳过]: ").strip()
        if ans == "":
            return None
        try:
            v = int(ans)
        except ValueError:
            print("  ⚠️  请输入数字编号，或直接回车跳过", flush=True)
            continue
        if 0 <= v < len(items):
            return items[v][0]  # return the real device id
        print(f"  ⚠️  无效编号，请输入 0-{len(items) - 1}", flush=True)


def _camera_name(cam_id: int) -> str | None:
    """Read the camera's friendly name from sysfs (Linux)."""
    try:
        name = Path(f"/sys/class/video4linux/video{cam_id}/name").read_text().strip()
        return name or None
    except OSError:
        return None


def _camera_kind(cam_id: int) -> str | None:
    """Classify a video node: '画面流' / '元数据流(非画面)' / None (unknown)."""
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{cam_id}", "--all"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        if "Metadata Capture" in out:
            return "元数据流(非画面)"
        if "Video Capture" in out:
            return "画面流"
    except Exception:
        pass
    return None


def list_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
    """Enumerate audio input/output devices and probe cameras.

    Returns (mics, speakers, cameras), each a list of (device_id, label).
    """
    import sounddevice as sd
    devices = sd.query_devices()
    mics = [(i, f"{d['name']}  (in:{d['max_input_channels']})")
            for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    spks = [(i, f"{d['name']}  (out:{d['max_output_channels']})")
            for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    cams = []
    for i in _probe_cameras():
        name = _camera_name(i)
        kind = _camera_kind(i)
        label = f"/dev/video{i}"
        if name:
            label += f"  ({name}"
            label += f", {kind})" if kind else ")"
        elif kind:
            label += f"  ({kind})"
        cams.append((i, label))
    return mics, spks, cams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera Tutor — 实时语音对话")
    p.add_argument("--select-devices", action="store_true",
                   help="启动时菜单式选择 麦克风/扬声器/摄像头（结果自动保存，下次复用）")
    p.add_argument("--mic", type=int, default=None,
                   help="麦克风设备编号 (sounddevice index)，临时覆盖保存的配置")
    p.add_argument("--spk", type=int, default=None,
                   help="扬声器设备编号 (sounddevice index)，临时覆盖保存的配置")
    p.add_argument("--camera", type=int, default=None,
                   help="摄像头编号 (0/1/2...)，临时覆盖保存的配置")
    p.add_argument("--agc", action="store_true",
                   help="开启麦克风 AGC 自动增益")
    p.add_argument("--reset-devices", action="store_true",
                   help="清除已保存的设备配置")
    return p.parse_args(argv)


# ── Saved device config (auto-reuse last selection) ────────────
# Stored inside the runtime data dir so it's git-ignored and travels
# with the project checkout.

CONFIG_PATH = os.path.join(
    Path(__file__).resolve().parent.parent, ".camera-tutor-data", "devices.json"
)
_LEGACY_CONFIG_PATH = os.path.join(Path.home(), ".camera-tutor-devices.json")


def _migrate_legacy_config() -> None:
    """Move the old ~/.camera-tutor-devices.json into .camera-tutor-data/."""
    if os.path.exists(_LEGACY_CONFIG_PATH) and not os.path.exists(CONFIG_PATH):
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            os.replace(_LEGACY_CONFIG_PATH, CONFIG_PATH)
            print(f"♻️  设备配置已迁移到: {CONFIG_PATH}", flush=True)
        except OSError:
            pass


_migrate_legacy_config()


def save_config(config: AgentConfig) -> None:
    """Persist the current device selection for the next launch."""
    data = {
        "mic_device_index": config.mic_device_index,
        "spk_device_index": config.spk_device_index,
        "camera_id": config.camera_id,
        "agc_enabled": config.agc_enabled,
    }
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 设备选择已保存: {CONFIG_PATH}", flush=True)
    except OSError as e:
        print(f"⚠️  保存配置失败: {e}", flush=True)


def load_config() -> dict:
    """Load saved device config, or {} if none/invalid."""
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k: data.get(k) for k in
                ("mic_device_index", "spk_device_index", "camera_id", "agc_enabled")}
    except (OSError, ValueError):
        return {}


def reset_config() -> None:
    try:
        os.remove(CONFIG_PATH)
        print("🗑️  已清除保存的设备配置", flush=True)
    except FileNotFoundError:
        print("（没有已保存的配置）", flush=True)


def _device_valid(dev_id: int, want_input: bool) -> bool:
    """Check a saved sounddevice index still exists with the right direction."""
    import sounddevice as sd
    try:
        d = sd.query_devices(dev_id)
    except Exception:
        return False
    channels = d["max_input_channels"] if want_input else d["max_output_channels"]
    return channels > 0


def _camera_valid(cam_id: int) -> bool:
    return cam_id in _probe_cameras()


def apply_device_config(config: AgentConfig, args: argparse.Namespace) -> None:
    """Apply device config: explicit args > saved config > defaults."""
    if args.reset_devices:
        reset_config()

    explicit = (args.mic is not None or args.spk is not None
                or args.camera is not None or args.agc)

    if args.select_devices:
        # Interactive menus, then save for next launch
        mics, spks, cams = list_devices()
        print("\n—— 手动选择设备（直接回车 = 跳过该项）——", flush=True)
        if mics:
            v = select_menu("🎤 麦克风：", mics)
            if v is not None:
                config.mic_device_index = v
        if spks:
            v = select_menu("🔊 扬声器：", spks)
            if v is not None:
                config.spk_device_index = v
        if cams:
            v = select_menu("📷 摄像头：", cams)
            if v is not None:
                config.camera_id = v
        config.agc_enabled = bool(args.agc)
        print(f"\n📋 本次选择 → 麦克风: {config.mic_device_index}  "
              f"扬声器: {config.spk_device_index}  "
              f"摄像头: {config.camera_id}  AGC: {config.agc_enabled}", flush=True)
        save_config(config)
        return

    if explicit:
        # Explicit CLI args override everything (not persisted)
        if args.mic is not None:
            config.mic_device_index = args.mic
        if args.spk is not None:
            config.spk_device_index = args.spk
        if args.camera is not None:
            config.camera_id = args.camera
        config.agc_enabled = bool(args.agc)
        return

    # No explicit args: reuse last saved selection (if still valid)
    saved = load_config()
    if not saved:
        return  # defaults = mac original behaviour

    mic = saved.get("mic_device_index")
    spk = saved.get("spk_device_index")
    cam = saved.get("camera_id")
    agc = saved.get("agc_enabled")

    if mic is not None and _device_valid(mic, want_input=True):
        config.mic_device_index = mic
    if spk is not None and _device_valid(spk, want_input=False):
        config.spk_device_index = spk
    if cam is not None and _camera_valid(cam):
        config.camera_id = cam
    if agc:
        config.agc_enabled = True

    print("♻️  已自动应用上次保存的设备配置:"
          f"  麦克风={config.mic_device_index} 扬声器={config.spk_device_index} "
          f"摄像头={config.camera_id} AGC={config.agc_enabled}"
          f"  (如需重选: --select-devices；清除: --reset-devices)", flush=True)


def main():
    """Entry point: configure, setup, and run the agent."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

    config = AgentConfig()
    apply_device_config(config, args)
    agent = CameraTutorAgent(config=config)

    print("=" * 55)
    print("  Camera Tutor — 实时语音对话")
    print("  (WebSocket · 服务端 VAD · 实时打断)")
    print("=" * 55)
    print()

    try:
        agent.setup()
        agent.start()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        print("\n\n👋 再见！")


if __name__ == "__main__":
    main()
