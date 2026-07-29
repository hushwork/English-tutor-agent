#!/usr/bin/env python3
"""Camera Tutor — 实时语音对话 Demo (WebSocket).

真正的实时对话：打开麦克风就能跟 Emma 聊天，像电话一样。
服务端 VAD 自动检测语音起止，模型实时生成语音回应。

运行:
  python3 camera_tutor/realtime_demo.py

依赖:
  pip install websocket-client pyaudio numpy
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
MODEL = "qwen3.5-omni-flash-realtime"

WS_URL = f"wss://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model={MODEL}&language=en"

# ── Audio I/O ───────────────────────────────────────────────────

pa = pyaudio.PyAudio()

# 找麦克风设备 — Jabra / Brio / Poly Sync 20 / 任意可用设备
mic_index = None
for keywords in [
    ["Jabra", "USB Audio"],
    ["Brio", "mono"],
    ["Poly"],
]:
    for i in range(pa.get_device_count()):
        name = pa.get_device_info_by_index(i)["name"]
        ch = pa.get_device_info_by_index(i)["maxInputChannels"]
        if ch > 0 and all(k in name for k in keywords):
            mic_index = i
            break
    if mic_index is not None:
        break

mic_name = pa.get_device_info_by_index(mic_index)["name"] if mic_index is not None else "系统默认"
print(f"   Mic: {mic_name} (device {mic_index})")

mic = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE_MIC,
              input=True, input_device_index=mic_index, frames_per_buffer=CHUNK)
spk = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE_SPK, output=True)

print(f"   Model: {MODEL}")

# 启动摄像头（1 fps，场景变化时才上传）
camera = CameraPipeline(camera_id=0, fps=1, resolution=(224, 224),
                        scene_change_threshold=0.20, key_frame_min_interval=1.0)
try:
    camera.start()
    print("   Camera: ✅")
except Exception as e:
    print(f"   Camera: ❌ ({e})")
    camera = None
print()

# ── WebSocket callbacks ─────────────────────────────────────────

tutor = get_active_tutor()
emma_avatar = EmmaAvatar()   # instance — drives face viseme lookup
_ws = None
_running = True
_last_audio_time = [0.0]  # for response.create fallback
_audio_started = threading.Event()  # gate: audio before camera image

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
            "voice": "Tina",
            "instructions": tutor.system_prompt_guidance(),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_audio_transcription": {
                "language": "en",
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 400,
            },
        }
    }))

    # 启动麦克风发送线程（全双工，服务端 VAD 处理回授）
    def send_audio():
        first = True
        while _running:
            try:
                data = mic.read(CHUNK, exception_on_overflow=False)
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode()
                }))
                if first:
                    first = False
                    _audio_started.set()  # signal camera: audio flowing
            except Exception:
                break
    threading.Thread(target=send_audio, daemon=True).start()

    # 启动摄像头发送线程（场景变化时 1fps，静止时每 15 秒 1 帧保活）
    if camera:
        def send_camera():
            _audio_started.wait()  # wait until first audio is sent
            still_seconds = 0
            while _running:
                try:
                    frame = camera.get_latest_frame()
                    if frame:
                        # 场景变化 → 立即发
                        if frame.is_key_frame:
                            still_seconds = 0
                            jpg = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
                            b64 = base64.b64encode(jpg).decode()
                            ws.send(json.dumps({
                                "type": "input_image_buffer.append",
                                "image": b64
                            }))
                            time.sleep(1.0)
                        else:
                            # 静止 → 每 15 秒发一帧保活
                            still_seconds += 1
                            if still_seconds >= 15:
                                still_seconds = 0
                                jpg = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
                                b64 = base64.b64encode(jpg).decode()
                                ws.send(json.dumps({
                                    "type": "input_image_buffer.append",
                                    "image": b64
                                }))
                            time.sleep(1.0)
                except Exception:
                    pass
        threading.Thread(target=send_camera, daemon=True).start()

# ── Whisper-aligned face sync ───────────────────────────────
# Buffers audio, runs faster-whisper on transcript.done for word-level
# timestamps, then plays audio + viseme in perfect sync.
# Adds only ~0.3s whisper inference delay (tiny model, CPU).
_buffer_audio: list[bytes] = []
_buffer_transcript: list[str] = []
_buffer_seq_id = 0
_face_timer = None
_face_seq_id = 0
_whisper_model = None  # lazy init

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # 离线加载，避免 HuggingFace 连接超时
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("   ⏳ Loading whisper tiny model...", flush=True)
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("   ✅ Whisper ready", flush=True)
    return _whisper_model

def _push_face_viseme(word: str, full_transcript: str):
    try:
        if not word or not word.strip():
            mouth_open, tongue_visible, mouth_width, viseme_label = 0.05, 0.0, 0.3, "sil"
        else:
            viseme = emma_avatar._word_to_viseme(word)
            from camera_tutor.live2d_bridge import VisemeParams
            p = VisemeParams.from_viseme(viseme)
            mouth_open, tongue_visible, mouth_width, viseme_label = \
                p.mouth_open, p.tongue_visible, p.mouth_width, viseme.label
        import httpx
        httpx.post("http://localhost:8200/api/emma/face", json={
            "viseme": viseme_label, "mouth_open": mouth_open,
            "mouth_width": mouth_width, "tongue_visible": tongue_visible,
            "transcript": full_transcript,
        }, timeout=1)
    except Exception as e:
        print(f"  ⚠️ face push error: {e}")

def _build_wav(samples: bytes) -> bytes:
    """Wrap raw 24kHz 16bit mono PCM in a minimal WAV header."""
    import struct
    data_len = len(samples)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_len, b'WAVE',
        b'fmt ', 16, 1, 1, 24000, 24000 * 2, 2, 16,
        b'data', data_len
    )
    return header + samples

def _flush_whisper_sync():
    """Run whisper alignment, then play audio + viseme in sync."""
    global _buffer_audio, _buffer_transcript, _buffer_seq_id, _face_timer, _face_seq_id
    if not _buffer_audio:
        return
    _buffer_seq_id += 1
    seq = _buffer_seq_id

    audio_chunks = _buffer_audio[:]
    transcript = " ".join(_buffer_transcript).strip()
    _buffer_audio = []
    _buffer_transcript = []

    words = transcript.split()
    if not words or len(words) < 2:
        # Too short for alignment — just play
        for chunk in audio_chunks:
            spk.write(chunk)
        return

    # Build WAV in memory, run whisper
    try:
        wav = _build_wav(b"".join(audio_chunks))
        model = _get_whisper()
        import io
        segments, _ = model.transcribe(
            io.BytesIO(wav), word_timestamps=True,
            language="en", beam_size=1,
            vad_filter=False, initial_prompt=transcript,
        )
        word_times = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    word_times.append((w.word.strip(',.!?\'"'), w.start, w.end))
    except Exception as e:
        print(f"  ⚠️ whisper failed: {e}, fallback to direct play")
        for chunk in audio_chunks:
            spk.write(chunk)
        return

    if not word_times:
        for chunk in audio_chunks:
            spk.write(chunk)
        return

    # Cancel previous face timer
    if _face_timer:
        _face_timer.cancel()
    _face_seq_id += 1
    fseq = _face_seq_id

    # Schedule face pushes at whisper timestamps
    for w, start_s, end_s in word_times:
        if not w:
            continue
        delay = start_s
        def _push_at_time(word=w, d=delay, s=fseq):
            if s != _face_seq_id or seq != _buffer_seq_id:
                return
            if d > 0:
                t = threading.Timer(d, lambda: _push_face_viseme(word, transcript))
                t.daemon = True
                t.start()
            else:
                _push_face_viseme(word, transcript)
        _push_at_time()

    # Schedule reset at end
    last_end = word_times[-1][2] if word_times else 0
    def _reset_later():
        if fseq == _face_seq_id:
            _push_face_viseme("", transcript)
    _face_timer = threading.Timer(last_end + 0.3, _reset_later)
    _face_timer.daemon = True
    _face_timer.start()

    # Play audio in background
    def _play():
        import time as _t
        for chunk in audio_chunks:
            if seq != _buffer_seq_id:
                break
            spk.write(chunk)
    threading.Thread(target=_play, daemon=True).start()


def on_message(ws, message):
    global _buffer_audio, _buffer_transcript
    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    event_type = event.get("type", "")
    # Debug: log all non-audio-delta events
    if event_type != "response.audio.delta":
        print(f"  [event] {event_type}", flush=True)

    if event_type == "session.updated":
        # API session confirmed — open browser now
        try:
            import webbrowser, httpx as _hx
            for p in ["/static/face_preview.html", "/static/live2d/bundle.js",
                      "/static/live2d/core/live2dcubismcore.min.js"]:
                try: _hx.get(f"http://localhost:8200{p}", timeout=2)
                except: pass
            webbrowser.open("http://localhost:8200/static/face_preview.html")
            print("   🔗 Emma 面部页面已打开")
        except Exception:
            pass

    if event_type == "response.audio.delta":
        # Buffer audio for whisper-aligned playback
        _buffer_audio.append(base64.b64decode(event["delta"]))

    elif event_type == "response.audio_transcript.delta":
        delta = event.get("delta", "")
        if delta:
            _buffer_transcript.append(delta)

    elif event_type == "response.audio_transcript.done":
        transcript = event.get("transcript", "")
        if transcript:
            print(f"  🤖 {tutor.name}: {transcript}")
            _buffer_transcript = [transcript]
        _flush_whisper_sync()

    elif event_type == "response.audio.done":
        _flush_whisper_sync()

    elif event_type == "conversation.item.input_audio_transcription.completed":
        child_text = event.get("transcript", "")
        if child_text:
            print(f"  👧 Child: {child_text}")
            import time as _tm
            _last_audio_time[0] = _tm.time()

    elif event_type == "error":
        err = event.get("error", {})
        print(f"  ❌ Error: {err.get('message', 'unknown')} (code={err.get('code','')})")

def on_error(ws, error):
    print(f"  ❌ Connection error: {error}")

def on_close(ws, status, msg):
    print(f"\n  连接已关闭 (code={status})")

# ── Dashboard auto-start ──────────────────────────────────────

def _start_dashboard(port: int = 8200):
    """Start the dashboard server in a background thread if not already running."""
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

print("=" * 55)
print("  Camera Tutor — 实时语音对话")
print("  (WebSocket · 服务端 VAD · 实时打断)")
print("=" * 55)
print()

# 自动启动 Dashboard（提供面部动画 WebSocket 广播 + 静态页面）
_start_dashboard(8200)

# 清空上次的嘴型状态
import httpx
try:
    httpx.post("http://localhost:8200/api/emma/face/reset", timeout=1)
except Exception:
    pass

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
