#!/usr/bin/env python3
"""Camera Tutor — 实时语音对话 Demo (WebSocket).

真正的实时对话：打开麦克风就能跟 Emma 聊天，像电话一样。
服务端 VAD 自动检测语音起止，模型实时生成语音回应。

运行:
  python3 camera_tutor/realtime_demo.py

依赖:
  pip install websocket-client sounddevice numpy
"""

from dotenv import load_dotenv
import json
import os
import threading
import time
import base64
import sys
import cv2

load_dotenv()
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_tutor.tutor_personas import get_active_tutor
from camera_tutor.camera import CameraPipeline
from camera_tutor.avatar import EmmaAvatar, Viseme
from _thread import interrupt_main

# ── Audio setup ──────────────────────────────────────────────────

import sounddevice as sd
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
MODEL = "qwen3.5-omni-flash-realtime"

WS_URL = f"wss://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model={MODEL}&language=en"

# ── Audio I/O ───────────────────────────────────────────────────

# 找麦克风设备 — Jabra / Brio / Poly Sync 20 / 任意可用设备
mic_index = None
devices = sd.query_devices()
for keywords in [
    ["Jabra", "USB Audio"],
    ["Brio", "mono"],
    ["Poly"],
]:
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and all(k in dev["name"] for k in keywords):
            mic_index = i
            break
    if mic_index is not None:
        break

mic_name = devices[mic_index]["name"] if mic_index is not None else "系统默认"
print(f"   Mic: {mic_name} (device {mic_index})")

def _open_mic():
    s = sd.RawInputStream(samplerate=RATE_MIC, channels=1, dtype='int16',
                           blocksize=CHUNK, device=mic_index)
    s.start()
    return s

def _open_spk():
    s = sd.RawOutputStream(samplerate=RATE_SPK, channels=1, dtype='int16',
                            blocksize=CHUNK)
    s.start()
    return s

mic = _open_mic()
spk = _open_spk()

print(f"   Model: {MODEL}")

# 启动摄像头 — 尝试多个 camera_id
camera = None
for cam_id in [1, 0, 2]:  # macOS: 优先内置摄像头，跳过虚拟设备
    try:
        cam = CameraPipeline(camera_id=cam_id, fps=5, resolution=(360, 360),
                             scene_change_threshold=0.15, key_frame_min_interval=1.0)
        cam.start()
        camera = cam
        print(f"   Camera: ✅ (device /dev/video{cam_id})")
        break
    except Exception as e:
        print(f"   Camera {cam_id}: ❌ ({e})")
if camera is None:
    print("   Camera: ❌ (no device found)")
print()

# ── WebSocket callbacks ─────────────────────────────────────────

tutor = get_active_tutor()
emma_avatar = EmmaAvatar()

from types import SimpleNamespace as _NS
state = _NS(
    ws=None,
    running=True,
    last_audio_time=[0.0],
    audio_started=threading.Event(),
    session_ready=threading.Event(),
    browser_opened=False,
    camera_started=False,
    current_transcript="",
    face_ws=None,
    last_pushed_viseme=None,
    face_fallback_http=False,
)

def on_open(ws):
    state.ws = ws
    print("✅ 连接已建立，可以说话了！")
    print(f"   Tutor: {tutor.emoji} {tutor.name} ({tutor.voice})")
    print("   [Ctrl+C 退出]\n")

    # 配置会话
    ws.send(json.dumps({
        "event_id": "session_init",
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "voice": "Tina",
            "instructions": tutor.system_prompt_guidance(),
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
        }
    }))

    # 启动麦克风发送线程
    def send_audio():
        first = True
        while state.running:
            try:
                data = bytes(mic.read(CHUNK)[0])  # sounddevice returns (data, overflow)
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode()
                }))
                if first:
                    first = False
                    state.audio_started.set()  # signal camera: audio flowing
            except Exception as e:
                print(f"  ⚠️ mic error: {e}")
                break
    threading.Thread(target=send_audio, daemon=True).start()

    # 启动摄像头发送线程（等 session 确认 + 音频就绪后再发图像）
    if camera:
        # Camera reader: singleton — started once, not recreated on reconnect
        if not state.camera_started:
            state.camera_started = True
            _camera_latest_b64 = [None]
            _camera_frame_lock = threading.Lock()

            def _camera_reader():
                cap = camera._cap
                if cap is None:
                    return
                while state.running:
                    try:
                        ret, img = cap.read()
                        if ret:
                            img = cv2.resize(img, (360, 360))
                            jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
                            b64 = base64.b64encode(jpg).decode()
                            with _camera_frame_lock:
                                _camera_latest_b64[0] = b64
                    except Exception:
                        pass
                    # Only throttle on miss — keep buffer drained on hit
                    if not ret:
                        time.sleep(0.05)
            threading.Thread(target=_camera_reader, daemon=True).start()

            def _camera_preview():
                state.session_ready.wait()
                state.audio_started.wait()
                print("   📷 Camera streaming started")
                last_keyframe_time = 0
                while state.running:
                    try:
                        with _camera_frame_lock:
                            b64 = _camera_latest_b64[0]
                        if b64 is None:
                            time.sleep(0.1)
                            continue
                        try:
                            import httpx
                            httpx.post("http://localhost:8200/api/emma/camera",
                                json={"camera_frame": b64}, timeout=2)
                        except: pass
                        now = time.time()
                        if now - last_keyframe_time > 2:
                            last_keyframe_time = now
                            ws.send(json.dumps({
                                "type": "input_image_buffer.append", "image": b64
                            }))
                        time.sleep(0.2)
                    except Exception:
                        time.sleep(0.3)
            threading.Thread(target=_camera_preview, daemon=True).start()



# ── Formant-based viseme sync (inline, zero-latency) ──────────
# Viseme is computed directly from each audio delta chunk as it
# arrives — no ring buffer, no polling thread, no timing drift.
# Audio plays and mouth moves in perfect sync by construction.



def _init_face_ws():
    """Connect to dashboard WebSocket for low-latency face + camera push.
    Falls back to HTTP POST if WebSocket is unavailable (old dashboard)."""
    import websocket as _wslib
    import time as _t
    for attempt in range(5):  # reduced from 10 — fail fast if old server
        try:
            state.face_ws = _wslib.create_connection(
                "ws://localhost:8200/ws/emma/source", timeout=2)
            print("   ✅ Face WS connected to dashboard")
            state.face_fallback_http = False
            return
        except Exception as e:
            if attempt == 0:
                print(f"   ⚠️ WS connect failed ({e}), retrying...")
            _t.sleep(0.5)
    # Fallback: use HTTP POST (works with old dashboard too)
    state.face_fallback_http = True
    print("   ⚠️ WS unavailable, falling back to HTTP POST for face sync")

def _send_viseme_payload(payload: dict):
    """Send viseme via WS, with auto-reconnect + HTTP fallback."""

    # WebSocket path
    if state.face_ws is not None and not state.face_fallback_http:
        try:
            state.face_ws.send(json.dumps(payload))
            return
        except Exception:
            try: state.face_ws.close()
            except: pass
            state.face_ws = None

    # Try reconnect once
    if state.face_ws is None and not state.face_fallback_http:
        try:
            import websocket as _wslib
            state.face_ws = _wslib.create_connection(
                "ws://localhost:8200/ws/emma/source", timeout=1)
            state.face_ws.send(json.dumps(payload))
            return
        except Exception:
            state.face_ws = None

    # HTTP fallback
    try:
        import httpx
        httpx.post("http://localhost:8200/api/emma/face", json=payload, timeout=1)
    except Exception:
        pass

def _push_face_viseme(viseme, full_transcript: str):
    """Push viseme (must be Viseme object). Dedups unchanged labels."""

    if not isinstance(viseme, Viseme):
        return
    if viseme.label == state.last_pushed_viseme:
        return
    state.last_pushed_viseme = viseme.label

    try:
        from camera_tutor.live2d_bridge import VisemeParams
        p = VisemeParams.from_viseme(viseme)
        _send_viseme_payload({
            "type": "viseme",
            "viseme": viseme.label,
            "mouth_open": p.mouth_open,
            "mouth_width": p.mouth_width,
            "tongue_visible": p.tongue_visible,
            "transcript": full_transcript,
        })
    except Exception:
        pass




def on_message(ws, message):

    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    event_type = event.get("type", "")

    if event_type != "response.audio.delta":
        print(f"  [event] {event_type}", flush=True)

    if event_type == "session.updated":
        state.session_ready.set()
        if not state.browser_opened:
            state.browser_opened = True
            try:
                import webbrowser, httpx as _hx
                for p in ["/static/face_preview.html", "/static/live2d/bundle.js",
                          "/static/live2d/core/live2dcubismcore.min.js"]:
                    try: _hx.get(f"http://localhost:8200{p}", timeout=2)
                    except: pass
                print("   🔗 http://localhost:8200/static/face_preview.html")
            except Exception:
                pass

    # ── AUDIO: play + spectral viseme (pure audio-driven, zero latency) ──
    if event_type == "response.audio.delta":
        chunk = base64.b64decode(event["delta"])
        spk.write(chunk)  # Play immediately

        try:
            from camera_tutor.spectral_viseme import chunk_to_visemes
            for v in chunk_to_visemes(chunk, RATE_SPK):
                _push_face_viseme(v, state.current_transcript)
        except Exception:
            pass

    elif event_type == "response.audio_transcript.delta":
        delta = event.get("delta", "")
        if delta:
            state.current_transcript += delta

    elif event_type == "response.audio_transcript.done":
        transcript = event.get("transcript", "")
        if transcript:
            print(f"  🤖 {tutor.name}: {transcript}")
            state.current_transcript = transcript

    elif event_type == "response.audio.done":
        state.last_pushed_viseme = None
        _push_face_viseme(Viseme.V00_SIL, "")
        state.current_transcript = ""

    elif event_type == "conversation.item.input_audio_transcription.completed":
        child_text = event.get("transcript", "")
        if child_text:
            print(f"  👧 Child: {child_text}")
            import time as _tm
            state.last_audio_time[0] = _tm.time()

    elif event_type == "error":
        err = event.get("error", {})
        print(f"  ❌ Error: {err.get('message', 'unknown')} (code={err.get('code','')})")

def on_error(ws, error):
    print(f"  ❌ Connection error: {error}")

def on_close(ws, status, msg):
    print(f"\n  连接已关闭 (code={status})")

# ── Dashboard auto-start ──────────────────────────────────────

def _start_dashboard(port: int = 8200):
    """Start the dashboard server in a background thread.

    If port is already occupied (e.g. dashboard started separately),
    we skip startup — the HTTP/WS fallback in _init_face_ws handles it."""
    import httpx
    try:
        r = httpx.get(f"http://localhost:{port}/api/health", timeout=0.5)
        if r.status_code == 200:
            print(f"   Dashboard: ✅ already running on port {port}")
            return
    except Exception:
        pass

    print(f"   Dashboard: ⏳ starting on port {port}...")
    import uvicorn
    def _run():
        uvicorn.run(
            "camera_tutor.dashboard_server:app",
            host="0.0.0.0", port=port,
            log_level="warning",
        )
    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait until ready
    for _ in range(30):
        try:
            r = httpx.get(f"http://localhost:{port}/api/health", timeout=0.5)
            if r.status_code == 200:
                print(f"   Dashboard: ✅ ready on http://localhost:{port}")
                return
        except Exception:
            pass
        time.sleep(0.2)
    print(f"   Dashboard: ⚠️ 启动超时，请手动运行")
    print(f"     python3 -m uvicorn camera_tutor.dashboard_server:app --host 0.0.0.0 --port {port}")


# ── Main ────────────────────────────────────────────────────────

import websocket
import signal

def _handle_sigint(sig, frame):
    state.running = False
    try: ws.close()
    except: pass

signal.signal(signal.SIGINT, _handle_sigint)

print("=" * 55)
print("  Camera Tutor — 实时语音对话")
print("  (WebSocket · 服务端 VAD · 实时打断)")
print("=" * 55)
print()

# 自动启动 Dashboard（提供面部动画 WebSocket 广播 + 静态页面）
_start_dashboard(8200)

_init_face_ws()  # connect dashboard WebSocket for face/camera push

# 清空上次的嘴型状态 (via WebSocket)
if state.face_ws:
    try:
        state.face_ws.send(json.dumps({"type": "viseme", "viseme": "rest",
            "mouth_open": 0.0, "mouth_width": 0.0, "tongue_visible": 0.0,
            "transcript": ""}))
    except: pass

print()

ws = websocket.WebSocketApp(
    WS_URL,
    header=[f"Authorization: Bearer {API_KEY}"],
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

while state.running:
    try:
        ws.run_forever(ping_interval=120)
    except KeyboardInterrupt:
        state.running = False
        break
    if not state.running:
        break
    print("   ⏳ Omni API 重连中...")
    time.sleep(2)
    state.audio_started.clear()
    state.session_ready.clear()
    ws = websocket.WebSocketApp(
        WS_URL,
        header=[f"Authorization: Bearer {API_KEY}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

print("\n\n👋 再见！")
if camera:
    camera.stop()
mic.stop()
spk.stop()
