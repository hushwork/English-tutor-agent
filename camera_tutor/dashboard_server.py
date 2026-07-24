"""Camera Tutor — Web Dashboard for Parents.

FastAPI-based web server providing:
- Daily/Weekly learning reports
- Vocabulary tracking & SR card management
- Child activity timeline
- Device settings and controls

Reuses english_tutor's web_server.py pattern (FastAPI + SSE + static serving).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Import camera_tutor modules
from camera_tutor.parent_report import ParentReportEngine
from camera_tutor.decision_engine import DecisionEngine, TutorState, ChildState, ChildActivity, ChildMood

# Import english_tutor modules (reuse)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from english_tutor.spaced_repetition import SpacedRepetition
from english_tutor.memory import ConversationMemory

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
sr = SpacedRepetition(storage_dir=Path(__file__).resolve().parent.parent / ".camera-tutor-data")
memory = ConversationMemory(storage_dir=Path(__file__).resolve().parent.parent / ".camera-tutor-data")
decision_engine = DecisionEngine()

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


# ── Static files ────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent / "static_parent"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def serve_dashboard():
    """Serve the parent dashboard SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Camera Tutor Dashboard</h1><p>Static files not built yet.</p>")


# ── Report API ──────────────────────────────────────────────────

@app.get("/api/report/daily")
async def get_daily_report(date: Optional[str] = None):
    """Get the daily report for a specific date (default: today).

    Returns: DailyReport as JSON
    """
    report = report_engine.generate_daily_report()
    return report.__dict__


@app.get("/api/report/weekly")
async def get_weekly_summary():
    """Get the weekly summary."""
    return report_engine.generate_weekly_summary()


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
async def get_vocabulary():
    """Get all vocabulary cards with SR stats."""
    cards = sr.get_all_cards()
    return {
        "cards": [c.to_dict() for c in cards],
        "stats": sr.get_stats(),
    }


@app.get("/api/vocabulary/due")
async def get_due_vocabulary(limit: int = 10):
    """Get vocabulary cards due for review."""
    cards = sr.get_due_cards(limit=limit)
    return {"cards": [c.to_dict() for c in cards], "count": len(cards)}


@app.post("/api/vocabulary/review")
async def submit_vocabulary_review(word: str, quality: int = Query(ge=0, le=5)):
    """Submit a vocabulary review (SM-2 quality 0-5)."""
    try:
        card = sr.review_card(word, quality)
        return {"success": True, "card": card.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/vocabulary/add")
async def add_vocabulary(word: str, definition: str = "", context: str = ""):
    """Add a new vocabulary card."""
    card = sr.add_card(word, definition, context)
    return {"success": True, "card": card.to_dict()}


# ── Activity Timeline API ───────────────────────────────────────

@app.get("/api/timeline")
async def get_timeline(date: Optional[str] = None):
    """Get the activity timeline for today (or specified date).

    Returns list of events with timestamps.
    """
    # For now, return the raw log (in production: query by date)
    log = report_engine._log
    return {
        "date": date or report_engine._today,
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
async def get_highlights(limit: int = 5):
    """Get recent 'wow moments' — times the child spoke English."""
    report = report_engine.generate_daily_report()
    return {"highlights": report.highlights[:limit]}


# ── Health Check ────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8200,
        reload=True,
    )
