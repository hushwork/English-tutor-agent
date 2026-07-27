"""Multi-user profile management for English Tutor.

Each family member gets a separate profile with their own:
- Learning goals (CEFR target, focus areas, daily minutes)
- Conversation memory (sessions, vocabulary, errors)
- Spaced repetition cards
- Face embedding (optional, for camera-based recognition)

Storage layout:
    .english-tutor-data/
        users.json              ← index of all profiles
        {user_id}/
            stats.json
            spaced_repetition.json
            sessions/
                session_*.json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Constants ───────────────────────────────────────────────────────

VALID_CEFR = ("A1", "A2", "B1", "B2", "C1", "C2")
VALID_FOCUS = ("speaking", "reading", "listening", "writing", "vocabulary", "grammar")
VALID_AGE_GROUPS = ("preschool", "elementary", "middle")

# Predefined learning targets per age group
AGE_GROUP_CONFIGS = {
    "preschool": {
        "label": "Preschool (3-6)",
        "default_cefr": "A1",
        "default_focus": ["listening", "speaking"],
        "default_daily_minutes": 10,
        "target_vocabulary": 200,
        "description": "Play-based English exposure. No forced output. Songs, stories, simple commands.",
        "dialogue_style": "warm, playful, use simple 3-5 word sentences, lots of repetition and praise",
    },
    "elementary": {
        "label": "Elementary (7-12)",
        "default_cefr": "A2",
        "default_focus": ["speaking", "reading", "listening"],
        "default_daily_minutes": 20,
        "target_vocabulary": 500,
        "description": "Interest-driven conversation. Picture books, simple dialogue, basic phonics.",
        "dialogue_style": "encouraging, use 5-8 word sentences, ask open questions, gently correct errors",
    },
    "middle": {
        "label": "Middle School (13-15)",
        "default_cefr": "B1",
        "default_focus": ["speaking", "reading", "writing", "grammar"],
        "default_daily_minutes": 30,
        "target_vocabulary": 1500,
        "description": "Academic + real-world English. Discussions, debates, reading comprehension.",
        "dialogue_style": "respectful peer-like tone, use natural sentences, discuss abstract topics, subtle error correction",
    },
}
DEFAULT_STORAGE_DIR = os.environ.get(
    "ENGLISH_TUTOR_DATA_DIR",
    str(Path(__file__).resolve().parent.parent / ".english-tutor-data"),
)


# ── UserProfile ─────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """A single learner's profile with fixed learning goals."""

    user_id: str  # URL-safe slug, e.g. "alice"
    name: str  # Display name, e.g. "Alice"
    age_group: str = "elementary"  # preschool, elementary, middle
    target_cefr: str = "B1"  # A1-C2
    focus_areas: list[str] = field(default_factory=lambda: ["speaking", "reading"])
    daily_goal_minutes: int = 20
    face_embedding: list[float] | None = None
    created_at: str = ""
    last_active: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.last_active:
            self.last_active = self.created_at
        # Normalize
        self.target_cefr = self.target_cefr.upper()
        if self.target_cefr not in VALID_CEFR:
            self.target_cefr = "B1"
        self.focus_areas = [
            f for f in self.focus_areas if f in VALID_FOCUS
        ] or ["speaking"]
        self.daily_goal_minutes = max(5, min(120, self.daily_goal_minutes))
        # Normalize age_group
        if self.age_group not in VALID_AGE_GROUPS:
            self.age_group = "elementary"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "age_group": self.age_group,
            "target_cefr": self.target_cefr,
            "focus_areas": self.focus_areas,
            "daily_goal_minutes": self.daily_goal_minutes,
            "face_embedding": self.face_embedding,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def slugify(name: str) -> str:
        """Convert a display name to a URL-safe user_id."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9_\u4e00-\u9fff-]", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug.strip("-") or "user"

    def goal_summary(self) -> str:
        """One-line goal summary for display."""
        focus_str = ", ".join(self.focus_areas)
        age_label = AGE_GROUP_CONFIGS.get(self.age_group, {}).get("label", self.age_group)
        return (
            f"{age_label} | "
            f"Target: {self.target_cefr} | "
            f"Focus: {focus_str} | "
            f"Daily: {self.daily_goal_minutes} min"
        )

    def prompt_context(self) -> str:
        """A paragraph to inject into the tutor system prompt."""
        focus = ", ".join(self.focus_areas)
        config = AGE_GROUP_CONFIGS.get(self.age_group, {})
        age_label = config.get("label", self.age_group)
        dialogue_style = config.get("dialogue_style", "encouraging")
        target_vocab = config.get("target_vocabulary", 500)
        return (
            f"The student's name is {self.name}. "
            f"Age group: {age_label}. "
            f"Target CEFR: {self.target_cefr}. "
            f"Target vocabulary: {target_vocab} words. "
            f"Focus areas: {focus}. "
            f"Dialogue style: {dialogue_style}. "
            f"Daily goal: {self.daily_goal_minutes} minutes."
        )

    def age_config(self) -> dict:
        """Return the age-group configuration dict."""
        return AGE_GROUP_CONFIGS.get(self.age_group, AGE_GROUP_CONFIGS["elementary"])


# ── UserManager ─────────────────────────────────────────────────────

class UserManager:
    """Manages all user profiles and the active user."""

    def __init__(self, storage_dir: str | Path | None = None):
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._users: dict[str, UserProfile] = {}  # user_id -> profile
        self._active_user_id: str | None = None
        self._load()

    # ── CRUD ─────────────────────────────────────────────────────

    def create_user(
        self,
        name: str,
        age_group: str = "elementary",
        target_cefr: str = "B1",
        focus_areas: list[str] | None = None,
        daily_goal_minutes: int = 20,
        face_embedding: list[float] | None = None,
    ) -> UserProfile:
        """Create a new user profile. Raises ValueError if name conflicts."""
        user_id = UserProfile.slugify(name)
        if user_id in self._users:
            raise ValueError(f"User '{name}' already exists (id: {user_id})")

        # Apply age-group defaults
        config = AGE_GROUP_CONFIGS.get(age_group, AGE_GROUP_CONFIGS["elementary"])
        resolved_cefr = target_cefr if target_cefr != "B1" else config["default_cefr"]
        resolved_focus = focus_areas if focus_areas else config["default_focus"]
        resolved_goal = daily_goal_minutes if daily_goal_minutes != 20 else config["default_daily_minutes"]

        profile = UserProfile(
            user_id=user_id,
            name=name,
            age_group=age_group,
            target_cefr=resolved_cefr,
            focus_areas=resolved_focus,
            daily_goal_minutes=resolved_goal,
            face_embedding=face_embedding,
        )
        self._users[user_id] = profile
        # Ensure per-user data directory
        (self.storage_dir / user_id / "sessions").mkdir(parents=True, exist_ok=True)
        self._save()
        return profile

    def get_user(self, user_id: str) -> UserProfile | None:
        return self._users.get(user_id)

    def list_users(self) -> list[UserProfile]:
        """Return all users sorted by last active (newest first)."""
        users = list(self._users.values())
        users.sort(key=lambda u: u.last_active or "", reverse=True)
        return users

    def update_user(self, user_id: str, **kwargs) -> UserProfile:
        """Update fields on an existing profile."""
        profile = self._users.get(user_id)
        if not profile:
            raise ValueError(f"User '{user_id}' not found")

        allowed = {"name", "target_cefr", "focus_areas", "daily_goal_minutes",
                    "face_embedding"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(profile, k, v)

        # Re-slugify if name changed
        new_id = UserProfile.slugify(profile.name)
        if new_id != user_id and new_id not in self._users:
            old_dir = self.storage_dir / user_id
            new_dir = self.storage_dir / new_id
            if old_dir.exists():
                old_dir.rename(new_dir)
            profile.user_id = new_id
            del self._users[user_id]
            self._users[new_id] = profile

        profile.__post_init__()  # re-normalize
        self._save()
        return profile

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all their data. Returns True if found."""
        if user_id not in self._users:
            return False
        del self._users[user_id]
        # Remove data directory
        import shutil
        user_dir = self.storage_dir / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)
        self._save()
        return True

    # ── Active user ──────────────────────────────────────────────

    @property
    def active_user(self) -> UserProfile | None:
        if self._active_user_id:
            return self._users.get(self._active_user_id)
        return None

    def set_active_user(self, user_id: str) -> UserProfile:
        """Switch to a different active user."""
        profile = self._users.get(user_id)
        if not profile:
            raise ValueError(f"User '{user_id}' not found")
        self._active_user_id = user_id
        profile.last_active = datetime.now(timezone.utc).isoformat()
        self._save()
        return profile

    def ensure_active_user(self) -> UserProfile:
        """Get active user, or auto-select/prompt for one."""
        if self.active_user:
            return self.active_user
        users = self.list_users()
        if users:
            self.set_active_user(users[0].user_id)
            return users[0]
        # Create a default user
        return self.create_user(name="Me", target_cefr="B1")

    # ── Persistence ─────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self.storage_dir / "users.json"

    def _load(self):
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text())
                for item in data.get("users", []):
                    profile = UserProfile.from_dict(item)
                    self._users[profile.user_id] = profile
                self._active_user_id = data.get("active_user_id")
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self):
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "active_user_id": self._active_user_id,
            "user_count": len(self._users),
            "users": [u.to_dict() for u in self._users.values()],
        }
        self._index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ── Migration ────────────────────────────────────────────────

    def migrate_legacy_data(self, user_name: str = "Me") -> UserProfile | None:
        """Detect and migrate legacy global data into a named user profile.

        Legacy layout: .english-tutor-data/stats.json, spaced_repetition.json,
                        session_*.json (flat, no per-user subdirs)

        Migrates to:    .english-tutor-data/{user_id}/...

        Returns the created user profile, or None if no legacy data found
        or users already exist.
        """
        import shutil

        # Only migrate if we have no users and legacy data exists
        if self._users:
            return None

        legacy_stats = self.storage_dir / "stats.json"
        legacy_sr = self.storage_dir / "spaced_repetition.json"
        legacy_files = [legacy_stats, legacy_sr]
        legacy_sessions = list(self.storage_dir.glob("session_*.json"))

        has_legacy = any(f.exists() for f in legacy_files) or legacy_sessions
        if not has_legacy:
            return None

        # Create a user to hold the migrated data
        profile = self.create_user(name=user_name, target_cefr="B1")
        user_dir = self.storage_dir / profile.user_id
        sessions_dir = user_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Move legacy files into user directory
        for src in legacy_files:
            if src.exists():
                dst = user_dir / src.name
                shutil.move(str(src), str(dst))

        # Move legacy sessions
        for src in legacy_sessions:
            dst = sessions_dir / src.name
            shutil.move(str(src), str(dst))

        self.set_active_user(profile.user_id)
        return profile
