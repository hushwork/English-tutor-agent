"""Web server for English Tutor — FastAPI + mobile-first SPA.

Run with: uvicorn english_tutor.web_server:app --host 0.0.0.0 --port 8080
Or:       python3 run.py --web
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from english_tutor.immersion_mode import (
    CONTENT_TYPES,
    CONTENT_TYPE_LABELS,
    DIFFICULTY_DESCRIPTIONS,
    DIFFICULTY_LABELS,
    IMMERSION_SYSTEM_PROMPT,
    TOPIC_SEEDS,
    TYPE_GUIDES,
    ImmersionContent,
    ImmersionSession,
)
from english_tutor.llm_client import LLMClient
from english_tutor.memory import ConversationMemory
from english_tutor.spaced_repetition import SpacedRepetition
from english_tutor.tutor_prompt import (
    CONVERSATION_SUMMARY_PROMPT,
    build_system_message,
    build_topic_suggestion,
)
from english_tutor.user_manager import UserManager, UserProfile
from english_tutor.face_utils import (
    is_camera_available,
    capture_face_for_registration,
    find_matching_user,
    is_face_recognition_available,
    face_image_to_base64,
)

load_dotenv()

# ── Init ────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
AUDIO_DIR = Path(tempfile.gettempdir()) / "english-tutor-audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="English Tutor", version="2.0")

# Shared state — initialized on startup
client: LLMClient | None = None
memory: ConversationMemory | None = None
sr: SpacedRepetition | None = None
um: UserManager | None = None


def get_client() -> LLMClient:
    if client is None:
        raise HTTPException(500, "Server not initialized")
    return client


def get_um() -> UserManager:
    if um is None:
        raise HTTPException(500, "Server not initialized")
    return um


def _init_per_user():
    """(Re)create memory and SR for the current active user."""
    global memory, sr
    u = get_um()
    profile = u.ensure_active_user()
    memory = ConversationMemory(user_id=profile.user_id)
    sr = SpacedRepetition(user_id=profile.user_id)
    # Load or create a session
    sessions = memory.list_sessions()
    for s in sessions:
        if s["message_count"] > 1:
            memory.load_session(s["id"])
            break
    else:
        memory.new_session()


def get_memory() -> ConversationMemory:
    if memory is None:
        raise HTTPException(500, "Server not initialized")
    return memory


def get_sr() -> SpacedRepetition:
    if sr is None:
        raise HTTPException(500, "Server not initialized")
    return sr


# ── Startup ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global client, memory, sr, um
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    client = LLMClient(api_key=api_key)
    um = UserManager()

    # Initialize per-user memory/SR for the last active user
    _init_per_user()


# ── Static files ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>English Tutor</h1><p>Frontend not found.</p>")


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated MP3 audio file."""
    path = AUDIO_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(path, media_type="audio/mpeg")


# ── Chat API ────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Stream a chat completion via SSE."""
    cl = get_client()
    mem = get_memory()

    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        raise HTTPException(400, "message required")

    # Build messages, enhanced with current user context
    system_msg = build_system_message()
    u = get_um()
    if u.active_user:
        extra = f"\n\n## Current Learner\n{u.active_user.prompt_context()}"
        system_msg["content"] += extra
    messages = [system_msg]

    for msg in mem.get_context(max_messages=20):
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})
    mem.save_message("user", user_message)

    async def event_stream():
        collected = []
        try:
            async for token in cl.chat(messages):
                collected.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        response = "".join(collected)
        mem.save_message("assistant", response)
        yield f"data: {json.dumps({'done': True, 'full': response})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── TTS API ─────────────────────────────────────────────────────────

@app.post("/api/speak")
async def speak(request: Request):
    """Generate TTS audio and return the URL."""
    body = await request.json()
    text = body.get("text", "")[:2000]
    if not text:
        raise HTTPException(400, "text required")
    voice = body.get("voice", "en-US-JennyNeural")

    try:
        import edge_tts

        filename = f"tts_{hash(text) % 1000000:06d}.mp3"
        output_path = AUDIO_DIR / filename

        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(str(output_path))

        return JSONResponse({
            "url": f"/api/audio/{filename}",
            "size": output_path.stat().st_size,
        })
    except ImportError:
        raise HTTPException(500, "edge-tts not installed")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── User Management API ───────────────────────────────────────────

@app.get("/api/users")
async def list_users():
    """List all registered learners."""
    u = get_um()
    users = u.list_users()
    return JSONResponse({
        "users": [
            {
                "user_id": p.user_id,
                "name": p.name,
                "target_cefr": p.target_cefr,
                "focus_areas": p.focus_areas,
                "daily_goal_minutes": p.daily_goal_minutes,
                "is_active": u.active_user is not None and p.user_id == u.active_user.user_id,
                "last_active": p.last_active[:10] if p.last_active else "",
            }
            for p in users
        ]
    })


@app.get("/api/users/current")
async def current_user():
    """Get the currently active user profile."""
    u = get_um()
    profile = u.active_user
    if not profile:
        raise HTTPException(404, "No active user")
    return JSONResponse({
        "user_id": profile.user_id,
        "name": profile.name,
        "target_cefr": profile.target_cefr,
        "focus_areas": profile.focus_areas,
        "daily_goal_minutes": profile.daily_goal_minutes,
        "created_at": profile.created_at[:10] if profile.created_at else "",
        "last_active": profile.last_active[:10] if profile.last_active else "",
    })


@app.post("/api/users")
async def create_user(request: Request):
    """Create a new learner profile."""
    u = get_um()
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name required")

    try:
        profile = u.create_user(
            name=name,
            target_cefr=body.get("target_cefr", "B1"),
            focus_areas=body.get("focus_areas", ["speaking", "reading"]),
            daily_goal_minutes=body.get("daily_goal_minutes", 20),
            face_embedding=body.get("face_embedding"),
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return JSONResponse({
        "user_id": profile.user_id,
        "name": profile.name,
        "target_cefr": profile.target_cefr,
        "focus_areas": profile.focus_areas,
        "daily_goal_minutes": profile.daily_goal_minutes,
    })


@app.post("/api/users/switch")
async def switch_user(request: Request):
    """Switch to a different active user."""
    u = get_um()
    body = await request.json()
    user_id = body.get("user_id", "").strip()
    if not user_id:
        raise HTTPException(400, "user_id required")

    try:
        profile = u.set_active_user(user_id)
    except ValueError:
        raise HTTPException(404, f"User '{user_id}' not found")

    # Re-initialize per-user memory and SR
    _init_per_user()

    return JSONResponse({
        "ok": True,
        "user_id": profile.user_id,
        "name": profile.name,
        "target_cefr": profile.target_cefr,
        "focus_areas": profile.focus_areas,
        "daily_goal_minutes": profile.daily_goal_minutes,
    })


@app.put("/api/users/{user_id}")
async def update_user(user_id: str, request: Request):
    """Update a user profile."""
    u = get_um()
    body = await request.json()
    allowed = {"name", "target_cefr", "focus_areas", "daily_goal_minutes"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(400, "No valid fields to update")
    try:
        profile = u.update_user(user_id, **updates)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return JSONResponse({
        "user_id": profile.user_id,
        "name": profile.name,
        "target_cefr": profile.target_cefr,
        "focus_areas": profile.focus_areas,
        "daily_goal_minutes": profile.daily_goal_minutes,
    })


# ── Face Recognition API ─────────────────────────────────────────

@app.post("/api/face/capture")
async def face_capture():
    """Capture a face from camera and return embedding + base64 image."""
    if not is_camera_available():
        raise HTTPException(503, "Camera not available")

    face_img, embedding = capture_face_for_registration(num_attempts=2)
    if face_img is None:
        raise HTTPException(404, "No face detected. Make sure you're facing the camera.")

    result = {"detected": True}
    if embedding:
        result["embedding"] = embedding
    try:
        result["image_base64"] = face_image_to_base64(face_img)
    except Exception:
        pass
    return JSONResponse(result)


@app.post("/api/face/recognize")
async def face_recognize():
    """Try to recognize the current face and auto-switch user."""
    u = get_um()
    if not is_camera_available() or not is_face_recognition_available():
        return JSONResponse({"recognized": False, "reason": "face_recognition not available"})

    users = u.list_users()
    known = [(p.user_id, p.face_embedding) for p in users if p.face_embedding]
    if not known:
        return JSONResponse({"recognized": False, "reason": "no registered faces"})

    face_img, embedding = capture_face_for_registration(num_attempts=1)
    if not embedding:
        return JSONResponse({"recognized": False, "reason": "no face detected"})

    match_id = find_matching_user(embedding, known)
    if match_id:
        profile = u.set_active_user(match_id)
        _init_per_user()
        return JSONResponse({
            "recognized": True,
            "user_id": profile.user_id,
            "name": profile.name,
            "target_cefr": profile.target_cefr,
            "focus_areas": profile.focus_areas,
            "daily_goal_minutes": profile.daily_goal_minutes,
        })

    return JSONResponse({"recognized": False, "reason": "face not matched"})


# ── Immersion API ───────────────────────────────────────────────────

@app.post("/api/immerse/generate")
async def immerse_generate(request: Request):
    """Generate a new immersion content piece."""
    cl = get_client()

    body = await request.json()
    difficulty = body.get("difficulty", "medium")
    content_type = body.get("content_type") or random.choice(CONTENT_TYPES)
    topic = body.get("topic") or random.choice(TOPIC_SEEDS)

    if difficulty not in DIFFICULTY_LABELS:
        difficulty = "medium"

    prompt = IMMERSION_SYSTEM_PROMPT.format(
        difficulty=difficulty,
        level_guide=DIFFICULTY_DESCRIPTIONS.get(difficulty, ""),
        topic=topic,
        content_type=content_type,
        type_guide=TYPE_GUIDES.get(content_type, ""),
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": (
            f"Generate a {CONTENT_TYPE_LABELS.get(content_type, content_type)} "
            f"immersion passage at {difficulty} level. "
            f"Topic: {topic}. Output ONLY the JSON object, nothing else."
        )},
    ]

    try:
        response = await cl.chat_sync(
            messages, temperature=0.7, max_tokens=512,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    data = ImmersionSession._parse_json_response(response)
    passage = ImmersionSession._clean_passage(data.get("passage", response))

    return JSONResponse({
        "title": data.get("title", "Listening Passage"),
        "content_type": content_type,
        "difficulty": difficulty,
        "key_words": data.get("key_words", []),
        "passage": passage,
        "closing_thought": data.get(
            "closing_thought", "What did you notice in what you heard?"
        ),
        "type_label": CONTENT_TYPE_LABELS.get(content_type, content_type),
        "difficulty_label": DIFFICULTY_LABELS.get(difficulty, difficulty),
    })


# ── Stats API ───────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """Return learning statistics."""
    mem = get_memory()
    srs = get_sr()

    sr_stats = srs.get_stats()
    vocab = mem.get_vocabulary()
    errors = mem.stats.get("common_errors", {})

    return JSONResponse({
        "sessions": mem.stats.get("total_sessions", 0),
        "messages": mem.stats.get("total_messages", 0),
        "vocabulary_count": len(vocab),
        "vocabulary": vocab[-20:],
        "errors": {
            k: {"count": v["count"], "examples": v.get("examples", [])[-3:]}
            for k, v in errors.items()
        },
        "error_count": sum(v["count"] for v in errors.values()),
        "sr_total": sr_stats.get("total", 0),
        "sr_learned": sr_stats.get("learned", 0),
        "sr_due": sr_stats.get("due", 0),
        "sr_learning": sr_stats.get("learning", 0),
    })


@app.get("/api/vocabulary")
async def get_vocabulary():
    """Return vocabulary list."""
    mem = get_memory()
    return JSONResponse({"vocabulary": mem.get_vocabulary()})


@app.get("/api/errors")
async def get_errors():
    """Return error analysis."""
    mem = get_memory()
    errors = mem.stats.get("common_errors", {})
    return JSONResponse({
        "errors": {
            k: {"count": v["count"], "examples": v.get("examples", [])[-5:]}
            for k, v in errors.items()
        }
    })


# ── Review API ──────────────────────────────────────────────────────

@app.get("/api/review/due")
async def review_due():
    """Get due review cards."""
    srs = get_sr()
    cards = srs.get_due_cards()
    return JSONResponse({
        "cards": [
            {
                "word": c.word,
                "definition": c.definition,
                "context": c.context,
                "interval_days": c.interval_days,
                "ease_factor": round(c.ease_factor, 2),
                "repetitions": c.repetitions,
            }
            for c in cards
        ]
    })


@app.post("/api/review/submit")
async def review_submit(request: Request):
    """Submit a review result."""
    srs = get_sr()
    body = await request.json()
    word = body.get("word", "")
    quality = body.get("quality", 0)

    card = srs.find_card(word)
    if not card:
        raise HTTPException(404, f"Card not found: {word}")

    card.review(quality)
    srs._save()
    return JSONResponse({"ok": True, "next_review": card.next_review})


# ── Topic API ───────────────────────────────────────────────────────

@app.get("/api/topic")
async def suggest_topic():
    """Suggest a conversation topic."""
    mem = get_memory()
    topic = build_topic_suggestion(set(mem.topic_history))
    mem.topic_history.append(topic)
    return JSONResponse({"topic": topic})


# ── Health ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return JSONResponse({"status": "ok", "model": client.model if client else "N/A"})


# ── Main ────────────────────────────────────────────────────────────

def run_web(host: str = "0.0.0.0", port: int = 8080):
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_web()
