"""Camera Tutor — Web Dashboard for Parents.

FastAPI-based web server providing:
- Daily/Weekly learning reports
- Vocabulary tracking & SR card management
- Child activity timeline
- Device settings and controls

Self-contained: all imports come from the camera_tutor package.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Import camera_tutor modules
from camera_tutor.parent_report import ParentReportEngine
from camera_tutor.decision_engine import DecisionEngine, TutorState, ChildState, ChildActivity, ChildMood
from camera_tutor.paths import data_dir

# Memory & SR (self-contained camera_tutor modules)
from camera_tutor.spaced_repetition import SpacedRepetition
from camera_tutor.memory import ConversationMemory

# ── App setup ───────────────────────────────────────────────────

app = FastAPI(
    title="Camera Tutor Dashboard",
    description="Parent dashboard for Camera Tutor — AI English tutor for kids",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state
report_engine = ParentReportEngine()
sr = SpacedRepetition(storage_dir=data_dir())
memory = ConversationMemory(storage_dir=data_dir())
decision_engine = DecisionEngine()

# Per-user 实例（惰性构造）；空 user 走上面的全局单例（legacy 全局视图）
_sr_by_user: dict[str, SpacedRepetition] = {}
_memory_by_user: dict[str, ConversationMemory] = {}
_report_by_user: dict[str, ParentReportEngine] = {}


def _resolve_user(user: str) -> str:
    """校验 ?user= 参数（空串 = legacy 全局视图）；非法 → 400。"""
    if not user:
        return ""
    from camera_tutor.rtc_device import validate_user_id
    try:
        return validate_user_id(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _sr_for(user: str) -> SpacedRepetition:
    uid = _resolve_user(user)
    if not uid:
        return sr
    if uid not in _sr_by_user:
        _sr_by_user[uid] = SpacedRepetition(storage_dir=data_dir(), user_id=uid)
    return _sr_by_user[uid]


def _memory_for(user: str) -> ConversationMemory:
    uid = _resolve_user(user)
    if not uid:
        return memory
    if uid not in _memory_by_user:
        _memory_by_user[uid] = ConversationMemory(storage_dir=data_dir(), user_id=uid)
    return _memory_by_user[uid]


def _report_for(user: str) -> ParentReportEngine:
    uid = _resolve_user(user)
    if not uid:
        return report_engine
    if uid not in _report_by_user:
        _report_by_user[uid] = ParentReportEngine(user_id=uid)
    return _report_by_user[uid]

# Tutor personas
from camera_tutor.tutor_personas import (
    list_tutors, get_active_tutor, set_active_tutor, get_child_age, set_child_age,
)

# Device state (would be connected to real device in production)
_device_state = {
    "camera_connected": False,
    "microphone_connected": False,
    "tutor_state": "observing",
    "led_color": "blue",
    "lens_cover_closed": False,
    "volume": 70,
    "wifi_connected": True,
    "model_mode": "local",
    "uptime_seconds": 0,
}

# ── Mic capture processing toggles (WebRTC getUserMedia constraints) ──
# 默认值与 local_pipe 的 RMS VAD 调参一致：
# AEC 开（外放防回采）、NS 关（会吃掉小声语音）、AGC 关（放大噪音底破坏 VAD）
AUDIO_SETTINGS_DEFAULTS = {
    "echoCancellation": True,
    "noiseSuppression": False,
    "autoGainControl": False,
}


def _audio_settings_path() -> Path:
    return data_dir() / "audio_settings.json"


def _load_audio_settings() -> dict:
    path = _audio_settings_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return {k: bool(data.get(k, v)) for k, v in AUDIO_SETTINGS_DEFAULTS.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(AUDIO_SETTINGS_DEFAULTS)


def _save_audio_settings(settings: dict) -> None:
    path = _audio_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))


# ── Static files ────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent / "static_parent"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files (face_preview.html, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_dashboard():
    """Serve the parent dashboard SPA."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text())
async def serve_dashboard():
    """Serve the parent dashboard SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Camera Tutor Dashboard</h1><p>Static files not built yet.</p>")


# ── Report API ──────────────────────────────────────────────────

@app.get("/api/report/daily")
async def get_daily_report(date: Optional[str] = None, user: str = ""):
    """Get the daily report for a specific date (default: today).

    Returns: DailyReport as JSON
    """
    report = _report_for(user).generate_daily_report()
    return report.__dict__


@app.get("/api/report/weekly")
async def get_weekly_summary(user: str = ""):
    """Get the weekly summary."""
    return _report_for(user).generate_weekly_summary()


@app.get("/api/report/history")
async def get_report_history(days: int = 7):
    """Get report history for the last N days."""
    reports_dir = Path(report_engine.storage_dir)
    reports = []
    for f in sorted(reports_dir.glob("report_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            reports.append(data)
        except (json.JSONDecodeError, OSError):
            continue
        if len(reports) >= days:
            break
    return {"reports": reports, "count": len(reports)}


# ── Vocabulary API (reuses SM-2 from english-tutor) ─────────────

@app.get("/api/vocabulary")
async def get_vocabulary(user: str = ""):
    """Get all vocabulary cards with SR stats."""
    user_sr = _sr_for(user)
    cards = user_sr.get_all_cards()
    return {
        "cards": [c.to_dict() for c in cards],
        "stats": user_sr.get_stats(),
    }


@app.get("/api/vocabulary/due")
async def get_due_vocabulary(limit: int = 10, user: str = ""):
    """Get vocabulary cards due for review."""
    cards = _sr_for(user).get_due_cards(limit=limit)
    return {"cards": [c.to_dict() for c in cards], "count": len(cards)}


@app.post("/api/vocabulary/review")
async def submit_vocabulary_review(word: str, quality: int = Query(ge=0, le=5), user: str = ""):
    """Submit a vocabulary review (SM-2 quality 0-5)."""
    try:
        card = _sr_for(user).review_card(word, quality)
        return {"success": True, "card": card.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/vocabulary/add")
async def add_vocabulary(word: str, definition: str = "", context: str = "", user: str = ""):
    """Add a new vocabulary card."""
    card = _sr_for(user).add_card(word, definition, context)
    return {"success": True, "card": card.to_dict()}


# ── Activity Timeline API ───────────────────────────────────────

@app.get("/api/timeline")
async def get_timeline(date: Optional[str] = None, user: str = ""):
    """Get the activity timeline for today (or specified date).

    Returns list of events with timestamps.
    """
    # For now, return the raw log (in production: query by date)
    engine = _report_for(user)
    log = engine._log
    return {
        "date": date or engine._today,
        "events": log[-50:],  # Last 50 events
        "total": len(log),
    }


# ── Device State API ────────────────────────────────────────────

@app.get("/api/device/status")
async def get_device_status():
    """Get current device status."""
    return _device_state


@app.post("/api/device/settings")
async def update_device_settings(
    volume: Optional[int] = Query(None, ge=0, le=100),
    max_interventions: Optional[int] = Query(None, ge=0, le=20),
    model_mode: Optional[str] = None,
    disable_hours_start: Optional[int] = Query(None, ge=0, le=23),
    disable_hours_end: Optional[int] = Query(None, ge=0, le=23),
):
    """Update device settings."""
    if volume is not None:
        _device_state["volume"] = volume
    if max_interventions is not None:
        decision_engine.max_interventions_per_hour = max_interventions
    if model_mode is not None:
        _device_state["model_mode"] = model_mode
    if disable_hours_start is not None:
        _device_state["disable_hours_start"] = disable_hours_start
    if disable_hours_end is not None:
        _device_state["disable_hours_end"] = disable_hours_end

    return {"success": True, "settings": _device_state}


@app.post("/api/device/rest")
async def set_rest_mode():
    """Put the device in rest mode (sleep/night)."""
    decision_engine.transition(TutorState.RESTING)
    _device_state["tutor_state"] = "resting"
    _device_state["led_color"] = "purple"
    return {"success": True, "state": "resting"}


@app.post("/api/device/wake")
async def wake_device():
    """Wake the device from rest mode."""
    decision_engine.return_to_observing()
    _device_state["tutor_state"] = "observing"
    _device_state["led_color"] = "blue"
    return {"success": True, "state": "observing"}


# ── Highlights API ──────────────────────────────────────────────

@app.get("/api/highlights")
async def get_highlights(limit: int = 5, user: str = ""):
    """Get recent 'wow moments' — times the child spoke English."""
    report = _report_for(user).generate_daily_report()
    return {"highlights": report.highlights[:limit]}



# ── Tutor Persona API ─────────────────────────────────────────

@app.get("/api/device/audio-settings")
async def get_audio_settings():
    """Get mic capture processing toggles (consumed by the WebRTC device page)."""
    return _load_audio_settings()


@app.post("/api/device/audio-settings")
async def update_audio_settings(
    echoCancellation: Optional[bool] = Query(None),
    noiseSuppression: Optional[bool] = Query(None),
    autoGainControl: Optional[bool] = Query(None),
):
    """Update mic capture processing toggles (applied on next device capture)."""
    settings = _load_audio_settings()
    for key, val in {
        "echoCancellation": echoCancellation,
        "noiseSuppression": noiseSuppression,
        "autoGainControl": autoGainControl,
    }.items():
        if val is not None:
            settings[key] = val
    _save_audio_settings(settings)
    return {"success": True, "settings": settings}


# ── Voice Gate（拒识/唤醒词门禁） ──────────────────────────────
# local_pipe 按文件 mtime 热重载 voice_gate.json，保存后无需重启

@app.get("/api/voice-gate")
async def get_voice_gate():
    """Get voice gate config (mode + text filter + kws settings)."""
    from camera_tutor.voice_gate import VoiceGate
    return asdict(VoiceGate.load().config)


@app.post("/api/voice-gate")
async def update_voice_gate(request: Request):
    """Update voice gate config (partial merge; body = JSON)."""
    from camera_tutor.voice_gate import VoiceGate
    data = await request.json()
    gate = VoiceGate.load()
    current = asdict(gate.config)
    merged = dict(current)
    if "mode" in data:
        merged["mode"] = data["mode"]
    for section in ("text", "kws"):
        if isinstance(data.get(section), dict):
            merged[section] = {**current.get(section, {}), **data[section]}
    gate.config = VoiceGate._config_from_dict(merged)
    gate.save()
    return {"success": True, "config": asdict(gate.config)}

@app.get("/api/tutor/list")
async def api_list_tutors():
    """Get all available tutor personas."""
    tutors = []
    active = get_active_tutor()
    for t in list_tutors():
        tutors.append({
            "id": t.id,
            "name": t.name,
            "emoji": t.emoji,
            "voice": t.voice,
            "description_cn": t.description_cn,
            "description_en": t.description_en,
            "teaching_style": t.teaching_style,
            "child_age_min": t.child_age_min,
            "child_age_max": t.child_age_max,
            "personality_traits": t.personality_traits,
            "age_appearance": t.age_appearance,
            "active": t.id == active.id,
        })
    return {"tutors": tutors}


@app.get("/api/tutor/active")
async def api_get_active_tutor():
    """Get the currently active tutor."""
    t = get_active_tutor()
    return {
        "id": t.id, "name": t.name, "emoji": t.emoji,
        "voice": t.voice, "teaching_style": t.teaching_style,
        "description_cn": t.description_cn,
    }


@app.post("/api/tutor/select")
async def api_select_tutor(tutor_id: str = Query(...)):
    """Switch to a different tutor persona."""
    from camera_tutor.tutor_personas import TUTOR_LIBRARY
    if tutor_id not in TUTOR_LIBRARY:
        raise HTTPException(status_code=404, detail=f"Unknown tutor: {tutor_id}")
    set_active_tutor(tutor_id)
    t = get_active_tutor()
    return {"success": True, "tutor": {"id": t.id, "name": t.name, "emoji": t.emoji}}


@app.get("/api/tutor/child-age")
async def api_get_child_age():
    """Get configured child age."""
    return {"age": get_child_age()}


@app.post("/api/tutor/child-age")
async def api_set_child_age(age: int = Query(ge=1, le=18)):
    """Set child's age."""
    set_child_age(age)
    return {"success": True, "age": age}



# Emma's real-time face state (written by realtime_demo.py)
_emma_face = {"viseme": "rest", "mouth_open": 0.0, "tongue_visible": 0.0, "transcript": ""}
# ws → user_id（空串 = 未标识的旧版客户端，接收所有用户的事件）
_face_clients: dict[WebSocket, str] = {}

@app.post("/api/emma/face/reset")
async def reset_emma_face():
    """Reset face state (call at realtime_demo startup)."""
    _emma_face.update({"viseme": "rest", "mouth_open": 0.0, "tongue_visible": 0.0, "transcript": ""})
    import asyncio
    asyncio.create_task(_broadcast_event({
        "type": "init", "viseme": "rest", "mouth_open": 0.0,
        "mouth_width": 0.0, "tongue_visible": 0.0,
    }))
    return {"ok": True}

@app.websocket("/ws/emma/face")
async def ws_emma_face(websocket: WebSocket):
    """Browser clients — receives typed events: {"type":"viseme",...} {"type":"camera",...}

    ?user=<id> 标识该客户端只收对应用户的事件（未标识则全收）。
    """
    await websocket.accept()
    _face_clients[websocket] = websocket.query_params.get("user", "")
    try:
        await websocket.send_json({"type": "init", "viseme": "rest",
                                    "mouth_open": 0.0, "mouth_width": 0.0,
                                    "tongue_visible": 0.0})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _face_clients.pop(websocket, None)

@app.websocket("/ws/emma/source")
async def ws_emma_source(websocket: WebSocket):
    """realtime_demo.py pushes typed events — relayed directly to browsers."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            import asyncio
            asyncio.create_task(_broadcast_event(data))
    except WebSocketDisconnect:
        pass

async def _broadcast_event(data: dict):
    """Relay a typed event to browser WS clients.

    事件带 user_id 时只发给同 user 的客户端 + 未标识的旧版客户端；
    不带 user_id 时广播所有客户端。
    """
    uid = data.get("user_id", "")
    dead = []
    for ws, client_uid in _face_clients.items():
        if uid and client_uid not in (uid, ""):
            continue
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _face_clients.pop(ws, None)

# Camera frames arrive via HTTP POST (1fps) — broadcast as typed event
@app.post("/api/emma/camera")
async def set_camera_frame(data: dict):
    frame = data.get("camera_frame", "")
    if frame:
        import asyncio
        asyncio.create_task(_broadcast_event({
            "type": "camera",
            "camera_frame": frame,
            "user_id": data.get("user_id", ""),
        }))
    return {"ok": True}


# ── Users API ───────────────────────────────────────────────────

@app.get("/api/users")
async def list_users():
    """列出有练习数据的用户（data_dir 下含 sessions/ 或 stats.json 的子目录）。

    legacy 全局数据（直接在 data_dir 下）存在时始终包含 "default"。
    """
    root = data_dir()
    users: list[str] = []
    if (root / "stats.json").exists() or (root / "sessions").is_dir():
        users.append("default")
    try:
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in users:
                continue
            if (d / "sessions").is_dir() or (d / "stats.json").exists():
                users.append(d.name)
    except OSError:
        pass
    return {"users": users}


# ── Health Check ────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }


# ── WebRTC device signaling ─────────────────────────────────────

@app.post("/rtc/offer")
async def rtc_offer(request: Request):
    """WebRTC signaling: browser sends an SDP offer, gets an answer.

    Only available when the agent runs with av_source=webrtc (the RTC
    manager lives in the agent process). Optional bearer token via
    RTC_TOKEN env var.
    """
    token = os.environ.get("RTC_TOKEN", "")
    if token:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid RTC token")

    from camera_tutor.rtc_device import get_rtc_manager
    manager = get_rtc_manager()
    if manager is None:
        raise HTTPException(
            status_code=409,
            detail="WebRTC device mode not enabled (start with --av-source webrtc)",
        )

    try:
        payload = await request.json()
        sdp, offer_type = payload["sdp"], payload["type"]
    except Exception:
        raise HTTPException(status_code=400, detail="expected {sdp, type} JSON body")

    # 多用户并发：浏览器可在 offer 里带 user_id（缺省 → "default"）；
    # validate_user_id 拒绝非法 ID（路径穿越防护）时返回 400。
    user_id = payload.get("user_id", "")
    try:
        return await manager.handle_offer(sdp, offer_type, user_id=user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/rtc/config")
async def rtc_config(request: Request):
    """返回浏览器端 RTCPeerConnection 的 ICE 配置。

    手机等在运营商 CGNAT 后的浏览器只有内网 host candidate，必须配
    STUN/TURN 才能打通。与 /rtc/offer 共用 RTC_TOKEN 鉴权。
    """
    token = os.environ.get("RTC_TOKEN", "")
    if token:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid RTC token")

    from camera_tutor.rtc_device import browser_ice_servers
    return {"iceServers": browser_ice_servers()}


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8200,
        reload=True,
    )
