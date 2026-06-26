"""Conversation memory — JSON-based persistence with session history."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default storage directory — use env var or fallback to writable workspace path
DEFAULT_STORAGE_DIR = os.environ.get(
    "ENGLISH_TUTOR_DATA_DIR",
    str(Path(__file__).resolve().parent.parent / ".english-tutor-data"),
)


class ConversationMemory:
    """Manages conversation history and user stats, persisted as JSON files."""

    def __init__(self, storage_dir: str | Path | None = None):
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Current session
        self.session_id: str = ""
        self.messages: list[dict] = []
        self.topic_history: list[str] = []
        self.new_words: list[dict] = []  # {word, definition, context, timestamp}

        # User stats (loaded from profile)
        self.stats: dict[str, Any] = self._load_stats()

    # ── Session management ────────────────────────────────────────

    def new_session(self) -> str:
        """Start a new conversation session. Returns the session ID."""
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.messages = []
        self.new_words = []
        return self.session_id

    def save_message(self, role: str, content: str):
        """Add a message to the current session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._auto_save()

    def get_context(self, max_messages: int = 20) -> list[dict]:
        """Get messages suitable for the LLM context window (system + recent)."""
        # Return last N messages for context window management
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages

    # ── Vocabulary ────────────────────────────────────────────────

    def add_new_word(self, word: str, definition: str, context: str):
        """Record a new word learned during conversation."""
        entry = {
            "word": word,
            "definition": definition,
            "context": context,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Avoid duplicates
        if not any(w["word"].lower() == word.lower() for w in self.stats.get("vocabulary", [])):
            self.stats.setdefault("vocabulary", []).append(entry)
        self.new_words.append(entry)
        self._save_stats()

    def get_vocabulary(self) -> list[dict]:
        """Return all saved vocabulary."""
        return self.stats.get("vocabulary", [])

    # ── Session persistence ────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """List all past sessions sorted by date (newest first)."""
        sessions = []
        for f in self.storage_dir.glob("session_*.json"):
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "id": data.get("session_id", f.stem.replace("session_", "")),
                    "date": data.get("created_at", ""),
                    "message_count": len(data.get("messages", [])),
                    "word_count": len(data.get("new_words", [])),
                })
            except (json.JSONDecodeError, OSError):
                continue
        sessions.sort(key=lambda s: s["date"], reverse=True)
        return sessions

    def load_session(self, session_id: str) -> bool:
        """Load a specific session from disk. Returns True if found."""
        path = self.storage_dir / f"session_{session_id}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self.session_id = data.get("session_id", session_id)
            self.messages = data.get("messages", [])
            self.new_words = data.get("new_words", [])
            self.topic_history = data.get("topics", [])
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def _auto_save(self):
        """Auto-save the current session."""
        if not self.session_id:
            return
        path = self.storage_dir / f"session_{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "created_at": self.messages[0]["timestamp"] if self.messages else "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": self.messages,
            "new_words": self.new_words,
            "topics": self.topic_history,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ── Stats ─────────────────────────────────────────────────────

    def _load_stats(self) -> dict:
        """Load user stats from disk."""
        path = self.storage_dir / "stats.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "total_sessions": 0,
            "total_messages": 0,
            "vocabulary": [],
            "common_errors": {},
            "streak_days": 0,
            "last_session_date": "",
        }

    def _save_stats(self):
        """Save user stats to disk."""
        self.stats["total_sessions"] = len(self.list_sessions())
        self.stats["total_messages"] = sum(s["message_count"] for s in self.list_sessions())
        self.stats["last_session_date"] = datetime.now(timezone.utc).isoformat()
        path = self.storage_dir / "stats.json"
        path.write_text(json.dumps(self.stats, ensure_ascii=False, indent=2))

    def record_error(self, error_type: str, example: str):
        """Track a common error pattern."""
        errors = self.stats.setdefault("common_errors", {})
        if error_type not in errors:
            errors[error_type] = {"count": 0, "examples": []}
        errors[error_type]["count"] += 1
        errors[error_type]["examples"].append(example)
        # Keep only last 5 examples
        errors[error_type]["examples"] = errors[error_type]["examples"][-5:]

    def get_stats_summary(self, sr_stats: dict | None = None) -> str:
        """Return a detailed stats summary with SR integration and progress bars."""
        vocab_count = len(self.stats.get("vocabulary", []))
        session_count = self.stats.get("total_sessions", 0)
        msg_count = self.stats.get("total_messages", 0)
        errors = self.stats.get("common_errors", {})
        error_count = sum(v["count"] for v in errors.values())
        error_summary = "; ".join(f"{k}: {v['count']}x" for k, v in errors.items()) or "none"

        sr_line = ""
        if sr_stats:
            learned_pct = (
                (sr_stats['learned'] / sr_stats['total'] * 100)
                if sr_stats['total'] > 0 else 0
            )
            sr_line = (
                f"\n   📚 Cards: {sr_stats['total']} total | "
                f"{sr_stats['learned']} learned ({learned_pct:.0f}%) | "
                f"{sr_stats['due']} due"
            )

        streak = self._calc_streak()
        vocab_bar = "█" * min(vocab_count // 5, 20) + "░" * (20 - min(vocab_count // 5, 20))
        streak_bar = "█" * min(streak, 20) + "░" * max(0, 20 - min(streak, 20))

        return (
            f"## 📊 Your Learning Progress\n\n"
            f"**Activity**\n"
            f"   Sessions: **{session_count}** | Messages: **{msg_count}**\n"
            f"   Current streak: **{streak} days** `{streak_bar}`\n"
            f"{sr_line}"
            f"\n\n**Vocabulary**\n"
            f"   Words saved: **{vocab_count}** `{vocab_bar}`\n"
            f"   Target: 100 words\n\n"
            f"**Error Tracking**\n"
            f"   Patterns tracked: **{error_count}** ({error_summary})\n"
            f"   Use `/errors` for detailed analysis with practice tips.\n\n"
            f"💡 Tip: Practice daily with `/read` for input and chat with Emma for output!"
        )

    def _calc_streak(self) -> int:
        """Calculate consecutive days with learning sessions."""
        sessions = self.list_sessions()
        if not sessions:
            return 0
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).date()
        dates = set()
        for s in sessions:
            try:
                d = datetime.fromisoformat(s["date"]).date()
                dates.add(d)
            except (ValueError, TypeError):
                pass
        streak = 0
        check = today
        while check in dates:
            streak += 1
            check -= timedelta(days=1)
        return streak
