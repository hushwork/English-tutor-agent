"""Spaced Repetition System — SM-2 algorithm for vocabulary review scheduling.

Implements the SuperMemo SM-2 algorithm (the algorithm behind Anki) for
optimizing vocabulary review intervals based on user recall performance.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── SM-2 Algorithm ──────────────────────────────────────────────────

MIN_INTERVAL_DAYS = 1
MAX_INTERVAL_DAYS = 365
EASE_FACTOR_MIN = 1.3


class SpacedRepetition:
    """Manages spaced repetition scheduling for vocabulary items."""

    def __init__(self, storage_dir: str | Path | None = None):
        base = os.environ.get(
            "ENGLISH_TUTOR_DATA_DIR",
            str(Path(__file__).resolve().parent.parent / ".english-tutor-data"),
        )
        self.storage_dir = Path(storage_dir or base)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.cards: dict[str, Card] = {}  # word -> Card
        self._load()

    # ── Card operations ──────────────────────────────────────────

    def add_card(
        self,
        word: str,
        definition: str = "",
        context: str = "",
    ) -> "Card":
        """Add a new vocabulary card."""
        word_key = word.lower().strip()
        if word_key in self.cards:
            return self.cards[word_key]

        card = Card(
            word=word,
            definition=definition,
            context=context,
        )
        self.cards[word_key] = card
        self._save()
        return card

    def review_card(self, word: str, quality: int) -> "Card":
        """Review a card with quality score 0-5.

        Quality scale (SM-2):
            0-1: Complete blackout / forgotten
            2-3: Recalled with serious difficulty / hesitation
            4: Recalled with some hesitation
            5: Perfect recall
        """
        word_key = word.lower().strip()
        if word_key not in self.cards:
            raise ValueError(f"Card '{word}' not found")

        card = self.cards[word_key]
        card.review(quality)
        self._save()
        return card

    def get_due_cards(self, limit: int = 20) -> list["Card"]:
        """Get cards due for review, sorted by due date (oldest first)."""
        now = time.time()
        due = [
            c for c in self.cards.values()
            if c.next_review <= now
        ]
        due.sort(key=lambda c: c.next_review)
        return due[:limit]

    def get_all_cards(self) -> list["Card"]:
        """Return all cards, sorted by ease factor (hardest first)."""
        cards = list(self.cards.values())
        cards.sort(key=lambda c: c.ease_factor)
        return cards

    def get_stats(self) -> dict:
        """Get summary statistics."""
        cards = list(self.cards.values())
        if not cards:
            return {"total": 0, "learned": 0, "due": 0, "avg_ease": 0}

        now = time.time()
        return {
            "total": len(cards),
            "learned": sum(1 for c in cards if c.interval_days >= 21),
            "learning": sum(1 for c in cards if 0 < c.interval_days < 21),
            "new": sum(1 for c in cards if c.interval_days == 0),
            "due": sum(1 for c in cards if c.next_review <= now),
            "avg_ease": round(sum(c.ease_factor for c in cards) / len(cards), 2),
            "avg_interval": round(sum(c.interval_days for c in cards) / len(cards), 1),
        }

    # ── Persistence ──────────────────────────────────────────────

    def _load(self):
        path = self.storage_dir / "spaced_repetition.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for item in data.get("cards", []):
                    card = Card.from_dict(item)
                    self.cards[card.word.lower().strip()] = card
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self):
        path = self.storage_dir / "spaced_repetition.json"
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "card_count": len(self.cards),
            "cards": [c.to_dict() for c in self.cards.values()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ── Pretty printing ──────────────────────────────────────────

    def format_due_cards(self) -> str:
        """Format due cards for display."""
        due = self.get_due_cards(limit=15)
        if not due:
            return "🎉 No cards due for review! You're all caught up."

        lines = ["📚 **Cards due for review:**\n"]
        for i, card in enumerate(due, 1):
            interval_str = self._format_interval(card.interval_days)
            ease_str = f"EF={card.ease_factor:.2f}"
            lines.append(
                f"  {i}. **{card.word}** — {card.definition[:40]} "
                f"[{interval_str}, {ease_str}]"
            )
        lines.append(
            "\nType `/review` to start reviewing, or use `/quiz` for a test."
        )
        return "\n".join(lines)

    def format_all_cards(self) -> str:
        """Format all cards for display."""
        cards = self.get_all_cards()
        if not cards:
            return "No vocabulary cards yet. Start learning new words!"

        lines = [f"📖 **Vocabulary: {len(cards)} cards**\n"]
        for card in cards[:20]:
            status = "🟢" if card.interval_days >= 21 else "🟡" if card.interval_days > 0 else "🆕"
            lines.append(
                f"  {status} **{card.word}** — {card.definition[:40]} "
                f"(interval: {self._format_interval(card.interval_days)})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_interval(days: float) -> str:
        if days == 0:
            return "new"
        if days < 1:
            return f"{int(days * 24)}h"
        if days < 30:
            return f"{int(days)}d"
        return f"{int(days / 30)}mo"


# ── Card data class ────────────────────────────────────────────────

class Card:
    """A single vocabulary card with SM-2 scheduling data."""

    def __init__(
        self,
        word: str,
        definition: str = "",
        context: str = "",
        interval_days: float = 0.0,
        ease_factor: float = 2.5,
        repetitions: int = 0,
        next_review: float | None = None,
        created_at: str = "",
        last_reviewed: str = "",
    ):
        self.word = word
        self.definition = definition
        self.context = context
        self.interval_days = interval_days
        self.ease_factor = ease_factor
        self.repetitions = repetitions
        self.next_review = next_review or time.time()
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.last_reviewed = last_reviewed

    def review(self, quality: int):
        """Apply SM-2 algorithm on review.

        Args:
            quality: 0 (forgotten) to 5 (perfect)
        """
        # Clamp quality
        quality = max(0, min(5, quality))

        # Update ease factor
        new_ef = self.ease_factor + (
            0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )
        self.ease_factor = max(EASE_FACTOR_MIN, new_ef)

        # Calculate new interval
        if quality < 3:
            # Forgotten — reset
            self.repetitions = 0
            self.interval_days = MIN_INTERVAL_DAYS
        else:
            self.repetitions += 1
            if self.repetitions == 1:
                self.interval_days = 1
            elif self.repetitions == 2:
                self.interval_days = 6
            else:
                self.interval_days = min(
                    MAX_INTERVAL_DAYS,
                    self.interval_days * self.ease_factor,
                )

        # Set next review
        self.next_review = time.time() + self.interval_days * 86400
        self.last_reviewed = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "definition": self.definition,
            "context": self.context,
            "interval_days": self.interval_days,
            "ease_factor": self.ease_factor,
            "repetitions": self.repetitions,
            "next_review": self.next_review,
            "created_at": self.created_at,
            "last_reviewed": self.last_reviewed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Card":
        return cls(**data)
