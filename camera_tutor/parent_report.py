"""Parent Report — daily learning summary generation.

Aggregates activity logs, vocabulary tracking, and interaction data
into a parent-friendly daily report. Can be generated daily or on-demand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_REPORT_DIR = os.environ.get(
    "CAMERA_TUTOR_DATA_DIR",
    str(Path(__file__).resolve().parent.parent / ".camera-tutor-data"),
)


@dataclass
class DailyReport:
    """A single day's learning report."""
    date: str
    total_english_input: int          # Sentences spoken by Emma
    child_utterances: int             # Times child spoke English
    new_vocabulary: list[str]         # New words introduced
    vocabulary_reviewed: int          # SM-2 cards reviewed
    total_interaction_minutes: float  # Active conversation time
    total_english_exposure_minutes: float  # Total time English was spoken
    dominant_mood: str                # happiest/focused/etc
    focus_minutes: float              # Longest focused period
    books_read: list[str]             # Books identified
    activities: list[str]             # Activities observed
    highlights: list[str]             # Notable moments (child speaking English)
    parent_notes: str = ""


class ParentReportEngine:
    """Collects, aggregates, and generates parent-facing reports.

    Writes JSON data per day and provides methods to generate
    human-readable summaries for the parent dashboard.
    """

    def __init__(self, storage_dir: str | Path | None = None):
        self.storage_dir = Path(storage_dir or DEFAULT_REPORT_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Current day's running data
        self._today: str = self._today_str()
        self._log: list[dict] = []  # Activity log entries

    # ── Data collection ──────────────────────────────────────────

    def log_event(self, event_type: str, data: dict | None = None):
        """Log a single event.

        Event types:
        - emma_spoke: Emma generated speech
        - child_spoke: Child vocalized (possibly English)
        - child_spoke_english: Child definitely spoke English
        - new_word_introduced: New vocabulary word
        - book_detected: A book was identified
        - activity_detected: Activity changed
        - focus_start / focus_end: Focus period boundary
        - mood_detected: Mood classification
        - interaction_start / interaction_end: Conversation boundary
        - game_start / game_end: Game session
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
        }
        if data:
            entry.update(data)

        # Handle day rollover
        today = self._today_str()
        if today != self._today:
            self._flush_daily_report()
            self._today = today
            self._log = []

        self._log.append(entry)

    # ── Report generation ────────────────────────────────────────

    def generate_daily_report(self) -> DailyReport:
        """Generate today's report from the activity log."""
        log = self._log

        # Count English input (Emma speaking)
        english_input = sum(1 for e in log if e["type"] == "emma_spoke")

        # Count child utterances
        child_spoke = sum(1 for e in log if e["type"] in ("child_spoke", "child_spoke_english"))
        child_spoke_english = sum(1 for e in log if e["type"] == "child_spoke_english")

        # New vocabulary
        new_words = [
            e.get("word", "")
            for e in log
            if e["type"] == "new_word_introduced" and e.get("word")
        ]

        # Vocabulary reviewed
        vocab_reviewed = sum(1 for e in log if e["type"] == "vocabulary_reviewed")

        # Books
        books = list(set(
            e.get("title", "")
            for e in log
            if e["type"] == "book_detected" and e.get("title")
        ))

        # Activities
        activities = list(set(
            e.get("activity", "")
            for e in log
            if e["type"] == "activity_detected" and e.get("activity")
        ))

        # Focus
        focus_starts = [e for e in log if e["type"] == "focus_start"]
        focus_ends = [e for e in log if e["type"] == "focus_end"]
        focus_minutes = 0.0
        if focus_starts and focus_ends:
            # Simplistic: sum of focus periods
            for i in range(min(len(focus_starts), len(focus_ends))):
                try:
                    t0 = datetime.fromisoformat(focus_starts[i]["timestamp"])
                    t1 = datetime.fromisoformat(focus_ends[i]["timestamp"])
                    focus_minutes += (t1 - t0).total_seconds() / 60
                except (ValueError, KeyError):
                    pass

        # Mood
        moods = [e.get("mood", "") for e in log if e["type"] == "mood_detected"]
        dominant_mood = max(set(moods), key=moods.count) if moods else "unknown"

        # Interaction time
        interaction_starts = [e for e in log if e["type"] == "interaction_start"]
        interaction_ends = [e for e in log if e["type"] == "interaction_end"]
        interaction_minutes = 0.0
        if interaction_starts and interaction_ends:
            for i in range(min(len(interaction_starts), len(interaction_ends))):
                try:
                    t0 = datetime.fromisoformat(interaction_starts[i]["timestamp"])
                    t1 = datetime.fromisoformat(interaction_ends[i]["timestamp"])
                    interaction_minutes += (t1 - t0).total_seconds() / 60
                except (ValueError, KeyError):
                    pass

        # English exposure (rough approximation: each Emma utterance ~3s)
        exposure_minutes = english_input * 3 / 60

        # Highlights
        highlights = [
            e.get("transcript", e.get("word", ""))
            for e in log
            if e["type"] in ("child_spoke_english", "new_word_introduced")
            and (e.get("transcript") or e.get("word"))
        ][:5]

        return DailyReport(
            date=self._today,
            total_english_input=english_input,
            child_utterances=child_spoke,
            new_vocabulary=new_words,
            vocabulary_reviewed=vocab_reviewed,
            total_interaction_minutes=round(interaction_minutes, 1),
            total_english_exposure_minutes=round(exposure_minutes, 1),
            dominant_mood=dominant_mood,
            focus_minutes=round(focus_minutes, 1),
            books_read=books,
            activities=activities,
            highlights=highlights,
        )

    def generate_weekly_summary(self) -> dict:
        """Generate a weekly summary from daily reports."""
        # Load all reports from this week
        reports = []
        for f in sorted(self.storage_dir.glob("report_*.json")):
            try:
                data = json.loads(f.read_text())
                reports.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        if not reports:
            return {"message": "No reports yet this week"}

        return {
            "days_active": len(reports),
            "total_english_input": sum(r.get("total_english_input", 0) for r in reports),
            "total_child_utterances": sum(r.get("child_utterances", 0) for r in reports),
            "total_new_words": sum(len(r.get("new_vocabulary", [])) for r in reports),
            "total_interaction_minutes": round(
                sum(r.get("total_interaction_minutes", 0) for r in reports), 1
            ),
            "weekly_streak": len(reports),
            "top_words": self._top_words(reports),
            "mood_trend": self._mood_trend(reports),
        }

    # ── Persistence ──────────────────────────────────────────────

    def _flush_daily_report(self):
        """Save today's report to disk before rolling over."""
        if not self._log:
            return
        report = self.generate_daily_report()
        path = self.storage_dir / f"report_{self._today}.json"
        path.write_text(json.dumps(
            report.__dict__,
            ensure_ascii=False,
            indent=2,
        ))

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _top_words(reports: list[dict], n: int = 10) -> list[str]:
        from collections import Counter
        counter = Counter()
        for r in reports:
            for w in r.get("new_vocabulary", []):
                counter[w] += 1
        return [w for w, _ in counter.most_common(n)]

    @staticmethod
    def _mood_trend(reports: list[dict]) -> list[dict]:
        trend = []
        for r in reports:
            trend.append({
                "date": r.get("date", ""),
                "mood": r.get("dominant_mood", "unknown"),
            })
        return trend
