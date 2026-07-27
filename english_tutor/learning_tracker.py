"""Silent learning tracker — interest-driven vocabulary & grammar exposure tracking.

Tracks what English content each child is exposed to during natural
interactions, without the child feeling like they're being "tested."

Key concepts:
- Vocabulary exposure: words the child heard/read in context
- Grammar exposure: grammatical patterns embedded in natural conversation
- Interest tags: topics/themes the child gravitates toward
- Coverage tracking: what % of age-group target vocabulary has been covered
- Gap detection: what target words haven't appeared naturally yet
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from english_tutor.user_manager import AGE_GROUP_CONFIGS

# ── Data classes ──────────────────────────────────────────────────

@dataclass
class InteractionRecord:
    """A single interaction between child and Emma."""
    timestamp: str
    trigger: str  # "child_showing_object", "child_called", "reading_moment", etc.
    child_age_group: str
    objects_detected: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    english_sentences_spoken: int = 0
    child_english_utterances: int = 0
    new_vocabulary_exposed: list[str] = field(default_factory=list)
    grammar_points_exposed: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    child_mood: str = "neutral"
    child_activity: str = "unknown"
    emma_response_summary: str = ""


@dataclass
class DailyLearningSummary:
    """End-of-day summary for internal use (next-session context) + parent report."""
    date: str
    user_id: str
    user_name: str
    age_group: str

    # Exposure stats
    total_interactions: int = 0
    total_english_sentences: int = 0
    total_child_utterances: int = 0
    total_duration_minutes: float = 0.0

    # Vocabulary
    new_vocabulary_today: list[str] = field(default_factory=list)
    total_vocabulary_known: int = 0
    vocabulary_coverage_pct: float = 0.0  # % of age-group target covered

    # Interests
    top_interests: list[str] = field(default_factory=list)
    dominant_activity: str = "unknown"
    dominant_mood: str = "neutral"

    # Grammar
    grammar_points_covered: list[str] = field(default_factory=list)

    # Highlights
    highlights: list[str] = field(default_factory=list)

    # Recommendations for tomorrow
    suggested_topics: list[str] = field(default_factory=list)
    suggested_vocabulary: list[str] = field(default_factory=list)
    suggested_grammar: list[str] = field(default_factory=list)


# ── Tracker ───────────────────────────────────────────────────────

class LearningTracker:
    """Silently tracks vocabulary/grammar exposure and interest patterns.

    Usage:
        tracker = LearningTracker(user_id="alice")
        tracker.record_interaction(
            trigger="child_showing_object",
            objects=["dinosaur", "crayon"],
            emma_sentences=5,
            ...
        )
        summary = tracker.generate_daily_summary()
    """

    def __init__(self, user_id: str, user_name: str = "",
                 age_group: str = "elementary",
                 storage_dir: str | Path | None = None):
        self.user_id = user_id
        self.user_name = user_name
        self.age_group = age_group

        base = os.environ.get(
            "ENGLISH_TUTOR_DATA_DIR",
            str(Path(__file__).resolve().parent.parent / ".english-tutor-data"),
        )
        self.storage_dir = Path(storage_dir or base) / user_id / "tracking"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Today's data
        self._today = self._today_str()
        self._interactions: list[InteractionRecord] = []
        self._load_today()

    # ── Recording ─────────────────────────────────────────────────

    def record_interaction(
        self,
        trigger: str = "unknown",
        objects_detected: list[str] | None = None,
        topics: list[str] | None = None,
        emma_sentences: int = 0,
        child_utterances: int = 0,
        new_vocabulary: list[str] | None = None,
        grammar_points: list[str] | None = None,
        duration_seconds: float = 0.0,
        child_mood: str = "neutral",
        child_activity: str = "unknown",
        emma_summary: str = "",
    ) -> InteractionRecord:
        """Record a single interaction event."""
        record = InteractionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            child_age_group=self.age_group,
            objects_detected=objects_detected or [],
            topics=topics or [],
            english_sentences_spoken=emma_sentences,
            child_english_utterances=child_utterances,
            new_vocabulary_exposed=new_vocabulary or [],
            grammar_points_exposed=grammar_points or [],
            duration_seconds=duration_seconds,
            child_mood=child_mood,
            child_activity=child_activity,
            emma_response_summary=emma_summary,
        )
        self._interactions.append(record)
        self._save_today()
        return record

    # ── Analysis ──────────────────────────────────────────────────

    def get_known_vocabulary(self) -> list[str]:
        """Get all vocabulary ever exposed to this child across all days."""
        all_words = set()
        for f in sorted(self.storage_dir.glob("tracking_*.json")):
            try:
                data = json.loads(f.read_text())
                for inter in data.get("interactions", []):
                    for w in inter.get("new_vocabulary_exposed", []):
                        all_words.add(w.lower().strip())
            except (json.JSONDecodeError, OSError):
                pass
        # Also include today's
        for inter in self._interactions:
            for w in inter.new_vocabulary_exposed:
                all_words.add(w.lower().strip())
        return sorted(all_words)

    def get_vocabulary_coverage(self) -> float:
        """What % of the age-group target vocabulary has been covered."""
        config = AGE_GROUP_CONFIGS.get(self.age_group, AGE_GROUP_CONFIGS["elementary"])
        target = config.get("target_vocabulary", 500)
        known = len(self.get_known_vocabulary())
        return round(min(100.0, known / target * 100), 1) if target > 0 else 0.0

    def get_vocabulary_gaps(self, limit: int = 10) -> list[str]:
        """Suggest vocabulary that hasn't appeared yet but should.

        Uses a built-in seed list per age group. In production, this would
        be replaced with a proper curriculum-aligned vocabulary list.
        """
        known = set(self.get_known_vocabulary())

        # Age-group seed vocabulary (simplified — expand in production)
        seed_lists = {
            "preschool": [
                "red", "blue", "green", "yellow", "big", "small", "cat", "dog",
                "bird", "fish", "apple", "banana", "milk", "water", "hello",
                "bye", "please", "thank", "yes", "no", "one", "two", "three",
                "mom", "dad", "baby", "happy", "sad", "run", "jump", "eat",
                "drink", "sleep", "book", "ball", "car", "sun", "moon", "star",
                "hand", "foot", "nose", "eye", "ear", "mouth", "hot", "cold",
            ],
            "elementary": [
                "dinosaur", "planet", "ocean", "mountain", "robot", "castle",
                "dragon", "pirate", "astronaut", "detective", "kitchen", "garden",
                "weather", "season", "spring", "summer", "autumn", "winter",
                "family", "friend", "school", "teacher", "homework", "playground",
                "bicycle", "computer", "camera", "piano", "guitar", "paint",
                "cook", "bake", "travel", "explore", "discover", "imagine",
                "yesterday", "tomorrow", "always", "never", "sometimes", "often",
                "because", "favorite", "beautiful", "interesting", "important",
                "elephant", "penguin", "dolphin", "butterfly", "volcano",
                "rainbow", "island", "forest", "desert", "jungle",
            ],
            "middle": [
                "environment", "technology", "democracy", "culture", "economy",
                "psychology", "philosophy", "literature", "architecture", "genetics",
                "artificial", "sustainable", "innovation", "globalization", "diversity",
                "perspective", "hypothesis", "evidence", "argument", "conclusion",
                "analyze", "evaluate", "synthesize", "compare", "contrast",
                "significant", "fundamental", "inevitable", "controversial", "ambiguous",
                "negotiate", "collaborate", "advocate", "compromise", "prioritize",
                "consequence", "responsibility", "opportunity", "challenge", "achievement",
                "identity", "community", "tradition", "prejudice", "compassion",
                "universe", "evolution", "civilization", "revolution", "renaissance",
            ],
        }

        seeds = seed_lists.get(self.age_group, seed_lists["elementary"])
        gaps = [w for w in seeds if w.lower() not in known]
        return gaps[:limit]

    def get_interest_profile(self) -> dict[str, int]:
        """Return topic/interest frequency across all interactions."""
        topic_counter = Counter()
        for inter in self._interactions:
            for t in inter.topics:
                topic_counter[t.lower()] += 1
            for o in inter.objects_detected:
                topic_counter[o.lower()] += 1
        return dict(topic_counter.most_common(15))

    def get_grammar_coverage(self) -> list[str]:
        """Return grammar points covered today."""
        points = set()
        for inter in self._interactions:
            for g in inter.grammar_points_exposed:
                points.add(g)
        return sorted(points)

    # ── Daily summary ─────────────────────────────────────────────

    def generate_daily_summary(self) -> DailyLearningSummary:
        """Generate end-of-day learning summary."""
        interactions = self._interactions

        total_sentences = sum(i.english_sentences_spoken for i in interactions)
        total_child = sum(i.child_english_utterances for i in interactions)
        total_duration = sum(i.duration_seconds for i in interactions) / 60.0

        # New vocabulary today
        today_words = []
        for i in interactions:
            today_words.extend(i.new_vocabulary_exposed)

        # Interests
        interests = self.get_interest_profile()
        top_interests = list(interests.keys())[:5]

        # Dominant activity
        activities = [i.child_activity for i in interactions if i.child_activity != "unknown"]
        dominant_activity = max(set(activities), key=activities.count) if activities else "unknown"

        # Dominant mood
        moods = [i.child_mood for i in interactions if i.child_mood != "neutral"]
        dominant_mood = max(set(moods), key=moods.count) if moods else "neutral"

        # Highlights
        highlights = []
        for i in interactions:
            if i.child_english_utterances > 0:
                highlights.append(
                    f"Child spoke English {i.child_english_utterances}x during "
                    f"{i.trigger.replace('_', ' ')}"
                )
            if i.new_vocabulary_exposed:
                highlights.append(
                    f"New words: {', '.join(i.new_vocabulary_exposed[:5])}"
                )
        highlights = highlights[:5]

        # Suggestions for tomorrow
        gaps = self.get_vocabulary_gaps(limit=5)
        suggested_topics = top_interests[:3] if top_interests else ["animals", "colors", "family"]

        return DailyLearningSummary(
            date=self._today,
            user_id=self.user_id,
            user_name=self.user_name,
            age_group=self.age_group,
            total_interactions=len(interactions),
            total_english_sentences=total_sentences,
            total_child_utterances=total_child,
            total_duration_minutes=round(total_duration, 1),
            new_vocabulary_today=list(set(today_words)),
            total_vocabulary_known=len(self.get_known_vocabulary()),
            vocabulary_coverage_pct=self.get_vocabulary_coverage(),
            top_interests=top_interests,
            dominant_activity=dominant_activity,
            dominant_mood=dominant_mood,
            grammar_points_covered=self.get_grammar_coverage(),
            highlights=highlights,
            suggested_topics=suggested_topics,
            suggested_vocabulary=gaps,
            suggested_grammar=[],
        )

    # ── Parent-facing report ──────────────────────────────────────

    def generate_parent_report(self) -> str:
        """Generate a Chinese-language parent report for today."""
        s = self.generate_daily_summary()

        age_label = AGE_GROUP_CONFIGS.get(
            self.age_group, {}
        ).get("label", self.age_group)

        lines = [
            f"## 📊 {s.user_name} 的今日学习报告",
            f"**日期**: {s.date} | **年龄段**: {age_label}",
            "",
            "### 📈 今日数据",
            f"- 互动次数: **{s.total_interactions}** 次",
            f"- Emma 英语输入: **{s.total_english_sentences}** 句",
            f"- 孩子英语输出: **{s.total_child_utterances}** 次",
            f"- 互动总时长: **{s.total_duration_minutes}** 分钟",
            "",
            "### 📚 词汇进度",
            f"- 今日新接触词汇: {', '.join(s.new_vocabulary_today[:10]) if s.new_vocabulary_today else '无'}",
            f"- 累计词汇量: **{s.total_vocabulary_known}** / {AGE_GROUP_CONFIGS.get(self.age_group, {}).get('target_vocabulary', 500)} ({s.vocabulary_coverage_pct}%)",
        ]

        if s.suggested_vocabulary:
            lines.append(f"- 建议明天引入: {', '.join(s.suggested_vocabulary[:5])}")

        lines.extend([
            "",
            "### 🎯 兴趣追踪",
            f"- 主要兴趣点: {', '.join(s.top_interests) if s.top_interests else '待观察'}",
            f"- 主要活动: {s.dominant_activity}",
            f"- 情绪状态: {s.dominant_mood}",
            "",
            "### 🌟 亮点时刻",
        ])
        if s.highlights:
            for h in s.highlights:
                lines.append(f"- {h}")
        else:
            lines.append("- 今天主要是英语输入，孩子还在吸收阶段")

        lines.extend([
            "",
            "### 💡 明天建议",
            f"- 推荐话题: {', '.join(s.suggested_topics)}",
        ])
        if s.suggested_vocabulary:
            lines.append(
                f"- 可自然引入的词汇: {', '.join(s.suggested_vocabulary[:5])}"
            )

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _today_path(self) -> Path:
        return self.storage_dir / f"tracking_{self._today}.json"

    def _load_today(self):
        path = self._today_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for item in data.get("interactions", []):
                    self._interactions.append(InteractionRecord(**item))
            except (json.JSONDecodeError, OSError):
                pass

    def _save_today(self):
        data = {
            "date": self._today,
            "user_id": self.user_id,
            "age_group": self.age_group,
            "interaction_count": len(self._interactions),
            "interactions": [
                {
                    "timestamp": i.timestamp,
                    "trigger": i.trigger,
                    "child_age_group": i.child_age_group,
                    "objects_detected": i.objects_detected,
                    "topics": i.topics,
                    "english_sentences_spoken": i.english_sentences_spoken,
                    "child_english_utterances": i.child_english_utterances,
                    "new_vocabulary_exposed": i.new_vocabulary_exposed,
                    "grammar_points_exposed": i.grammar_points_exposed,
                    "duration_seconds": i.duration_seconds,
                    "child_mood": i.child_mood,
                    "child_activity": i.child_activity,
                    "emma_response_summary": i.emma_response_summary,
                }
                for i in self._interactions
            ],
        }
        self._today_path().write_text(json.dumps(data, ensure_ascii=False, indent=2))
