#!/usr/bin/env python3
"""Camera Tutor — 实时语音对话 Demo (WebSocket).

真正的实时对话：打开麦克风就能跟 Emma 聊天，像电话一样。
服务端 VAD 自动检测语音起止，模型实时生成语音回应。

运行:
  python3 camera_tutor/realtime_demo.py

依赖:
  pip install websocket-client pyaudio numpy
"""

import json
import os
import threading
import time
import base64
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_tutor.tutor_personas import get_active_tutor
from camera_tutor.camera import CameraPipeline
from _thread import interrupt_main

# ── Audio setup ──────────────────────────────────────────────────

import pyaudio
import numpy as np

CHUNK = 3200  # 0.2s @ 16kHz — WebSocket 音频块大小
RATE_MIC = 16000   # 麦克风采样率（服务端要求 16kHz PCM）
RATE_SPK = 24000   # 音箱采样率（服务端输出 24kHz PCM）

# ── Configuration ────────────────────────────────────────────────

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not API_KEY:
    print("❌ 请设置 DASHSCOPE_API_KEY")
    sys.exit(1)

WORKSPACE_ID = "llm-xo2ff9jhvnvgvu6b"
MODEL = "qwen-omni-turbo-realtime"  # 也可用 qwen3.5-omni-plus-realtime

WS_URL = f"wss://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model={MODEL}"

# ── Audio I/O ───────────────────────────────────────────────────

pa = pyaudio.PyAudio()

# 找 Jabra 或 Brio
mic_index = None
for i in range(pa.get_device_count()):
    name = pa.get_device_info_by_index(i)["name"]
    ch = pa.get_device_info_by_index(i)["maxInputChannels"]
    if ch > 0:
        if "Jabra" in name and "USB Audio" in name:
            mic_index = i
            break
if mic_index is None:
    for i in range(pa.get_device_count()):
        name = pa.get_device_info_by_index(i)["name"]
        ch = pa.get_device_info_by_index(i)["maxInputChannels"]
        if ch > 0 and "Brio" in name and "mono" in name:
            mic_index = i
            break

mic = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE_MIC,
              input=True, input_device_index=mic_index, frames_per_buffer=CHUNK)
spk = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE_SPK, output=True)

print(f"   Mic: device {mic_index}")
print(f"   Model: {MODEL}")

# 启动摄像头（1 fps，场景变化时才上传）
camera = CameraPipeline(camera_id=0, fps=1, resolution=(224, 224),
                        scene_change_threshold=0.40, key_frame_min_interval=1.0)
try:
    camera.start()
    print("   Camera: ✅")
except Exception as e:
    print(f"   Camera: ❌ ({e})")
    camera = None
print()

# ── WebSocket callbacks ─────────────────────────────────────────

tutor = get_active_tutor()
_ws = None
_running = True

def on_open(ws):
    global _ws
    _ws = ws
    print("✅ 连接已建立，可以说话了！")
    print(f"   Tutor: {tutor.emoji} {tutor.name} ({tutor.voice})")
    print("   [Ctrl+C 退出]\n")

    # 配置会话
    ws.send(json.dumps({
        "event_id": "session_init",
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "voice": tutor.voice if tutor.voice in ("Cherry", "Serena") else "Cherry",
            "instructions": tutor.system_prompt_guidance(),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 600,
            },
        }
    }))

    # 启动麦克风发送线程
    def send_audio():
        while _running:
            try:
                data = mic.read(CHUNK, exception_on_overflow=False)
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode()
                }))
            except Exception:
                break
    threading.Thread(target=send_audio, daemon=True).start()

    # 启动摄像头发送线程（自适应帧率：动态场景 3fps，静态场景 0.1fps）
    if camera:
        def send_camera():
            last_frame = None
            still_count = 0
            while _running:
                try:
                    frame = camera.get_latest_frame()
                    if frame:
                        if frame.is_key_frame:
                            still_count = 0
                            jpg = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
                            b64 = base64.b64encode(jpg).decode()
                            ws.send(json.dumps({
                                "type": "input_image_buffer.append",
                                "image": b64
                            }))
                            last_frame = frame
                            time.sleep(0.3)  # 动态场景：3fps
                        else:
                            still_count += 1
                            # 每10秒发一次保活帧
                            if still_count % 10 == 0:
                                jpg = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
                                b64 = base64.b64encode(jpg).decode()
                                ws.send(json.dumps({
                                    "type": "input_image_buffer.append",
                                    "image": b64
                                }))
                            time.sleep(1.0)  # 静态场景：1fps
                except Exception:
                    pass
        threading.Thread(target=send_camera, daemon=True).start()

def on_message(ws, message):
    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    event_type = event.get("type", "")

    if event_type == "response.audio.delta":
        # Emma 说话的音频流 — 立即播放
        audio_bytes = base64.b64decode(event["delta"])
        spk.write(audio_bytes)

    elif event_type == "response.audio.done":
        pass  # 音频段结束

    elif event_type == "response.audio_transcript.done":
        # Emma 说完了 — 显示文字
        transcript = event.get("transcript", "")
        if transcript:
            print(f"  🤖 {tutor.name}: {transcript}")

    elif event_type == "conversation.item.input_audio_transcription.completed":
        # 孩子说了什么 — 显示文字
        child_text = event.get("transcript", "")
        if child_text:
            print(f"  👧 Child: {child_text}")

    elif event_type == "error":
        err = event.get("error", {})
        print(f"  ❌ Error: {err.get('message', 'unknown')} (code={err.get('code','')})")

def on_error(ws, error):
    print(f"  ❌ Connection error: {error}")

def on_close(ws, status, msg):
    print(f"\n  连接已关闭 (code={status})")

# ── Main ────────────────────────────────────────────────────────

import websocket

print("=" * 55)
print("  Camera Tutor — 实时语音对话")
print("  (WebSocket · 服务端 VAD · 实时打断)")
print("=" * 55)
print()

ws = websocket.WebSocketApp(
    WS_URL,
    header=[f"Authorization: Bearer {API_KEY}"],
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

try:
    ws.run_forever()
except KeyboardInterrupt:
    print("\n\n👋 再见！")
finally:
    _running = False
    if camera:
        camera.stop()
    mic.close()
    spk.close()
    pa.terminate()
