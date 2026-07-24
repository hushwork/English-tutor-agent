"""Scene Analyzer — understanding what's happening in the child's environment.

Caches scene analyses to avoid redundant LLM calls. Tracks objects,
activities, and provides child state inference without LLM when possible.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .decision_engine import ChildActivity, ChildMood, ChildState


@dataclass
class SceneAnalysis:
    """Cached analysis of a scene."""
    objects: list[str] = field(default_factory=list)
    activity: str = ""
    description: str = ""
    child_state: ChildState = field(default_factory=ChildState)
    timestamp: float = field(default_factory=time.time)
    frame_hash: str = ""  # Perceptual hash to identify same scene


class SceneAnalyzer:
    """Manages scene understanding with caching and heuristics.

    Three levels of analysis:
    1. Cache hit (same scene, no LLM call)
    2. Heuristic (rapid rules-based inference without LLM)
    3. LLM (full multimodal analysis for novel scenes)
    """

    def __init__(
        self,
        cache_ttl: float = 10.0,  # seconds before cache expires
        max_cache_size: int = 50,
    ):
        self.cache_ttl = cache_ttl
        self._cache: dict[str, SceneAnalysis] = {}
        self._recent_analyses: deque[SceneAnalysis] = deque(maxlen=max_cache_size)

    # ── Public API ───────────────────────────────────────────────

    def get_cached(
        self,
        frame_hash: str,
    ) -> Optional[SceneAnalysis]:
        """Return cached analysis if still valid."""
        if frame_hash in self._cache:
            analysis = self._cache[frame_hash]
            if time.time() - analysis.timestamp < self.cache_ttl:
                return analysis
            # Expired — remove
            del self._cache[frame_hash]
        return None

    def cache(self, frame_hash: str, analysis: SceneAnalysis):
        """Store a scene analysis in the cache."""
        analysis.timestamp = time.time()
        analysis.frame_hash = frame_hash
        self._cache[frame_hash] = analysis
        self._recent_analyses.append(analysis)

        # Evict oldest if cache is too big
        if len(self._cache) > self.max_cache_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp,
            )
            del self._cache[oldest_key]

    def infer_child_activity(
        self,
        objects: list[str],
        motion_level: float = 0.0,
    ) -> ChildActivity:
        """Quick heuristic: what is the child doing?

        This avoids an LLM call for obvious cases.
        """
        obj_lower = [o.lower() for o in objects]

        # Books → reading
        book_keywords = ["book", "picture", "page", "绘本", "story"]
        if any(k in ' '.join(obj_lower) for k in book_keywords):
            return ChildActivity.READING

        # Drawing materials → drawing
        draw_keywords = ["crayon", "pencil", "pen", "paper", "drawing",
                         "color", "paint", "brush", "marker", "笔", "画"]
        if any(k in ' '.join(obj_lower) for k in draw_keywords):
            return ChildActivity.DRAWING

        # Toys → playing
        toy_keywords = ["toy", "block", "doll", "car", "lego",
                        "puzzle", "ball", "teddy", "robot", "玩具"]
        if any(k in ' '.join(obj_lower) for k in toy_keywords):
            return ChildActivity.PLAYING

        # School supplies + no toys → studying
        school_keywords = ["notebook", "textbook", "homework", "worksheet",
                           "作业", "课本", "练习"]
        if any(k in ' '.join(obj_lower) for k in school_keywords):
            return ChildActivity.STUDYING

        # High motion → moving
        if motion_level > 0.5:
            return ChildActivity.MOVING

        return ChildActivity.UNKNOWN

    def infer_child_mood(
        self,
        activity: ChildActivity,
        focus_duration: float,
        idle_duration: float,
        vocal_energy: float = 0.0,
    ) -> ChildMood:
        """Quick heuristic: what mood is the child in?"""
        # Focused for a while → focused
        if focus_duration > 120:  # 2 minutes
            return ChildMood.FOCUSED

        # Idle for too long → bored
        if idle_duration > 300:  # 5 minutes
            return ChildMood.BORED

        # High vocal energy → happy
        if vocal_energy > 0.7:
            return ChildMood.HAPPY

        # Low vocal energy + idle → tired
        if vocal_energy < 0.1 and idle_duration > 180:
            return ChildMood.TIRED

        return ChildMood.NEUTRAL

    def get_recent_context(self, n: int = 3) -> str:
        """Get a text summary of recent scenes for LLM context."""
        if not self._recent_analyses:
            return ""

        recent = list(self._recent_analyses)[-n:]
        lines = []
        for a in recent:
            lines.append(
                f"- Activity: {a.activity or 'unknown'}, "
                f"Objects: {', '.join(a.objects[:3]) or 'none'}"
            )
        return "\n".join(lines)

    def compute_motion_level(
        self,
        current_hash: int,
        previous_hash: int,
    ) -> float:
        """Estimate motion level from hash distance (0.0 - 1.0).

        Higher = more motion between frames.
        """
        if previous_hash == 0:
            return 0.0
        hamming = bin(current_hash ^ previous_hash).count('1')
        return min(1.0, hamming / 32.0)  # Normalize: 32 bits of diff = max motion
