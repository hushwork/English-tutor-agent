"""Decision Engine — the "when to speak" brain of Camera Tutor.

This is the hardest component. It determines:
1. Whether now is a good time to engage
2. What role to adopt (recorder/tutor/playmate)
3. When to stay silent (protecting the child's focus)

State machine: OBSERVING → ENGAGING → TEACHING → GAMING → RESTING

Design principle: "宁可错过10次教学机会，也不打断1次深度专注"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TutorState(Enum):
    OBSERVING = "observing"    # Default: watching, recording, silent
    ENGAGING = "engaging"      # Child initiated interaction
    TEACHING = "teaching"      # Proactive teaching moment detected
    GAMING = "gaming"          # Play mode
    RESTING = "resting"        # Sleep/night mode or explicitly disabled


class ChildActivity(Enum):
    PLAYING = "playing"          # Free play with toys
    DRAWING = "drawing"          # Drawing/coloring
    READING = "reading"          # Looking at books
    STUDYING = "studying"        # Doing homework/schoolwork
    IDLE = "idle"                # Not doing anything obvious
    MOVING = "moving"            # Walking/running around
    UNKNOWN = "unknown"


class ChildMood(Enum):
    HAPPY = "happy"
    FOCUSED = "focused"
    BORED = "bored"
    FRUSTRATED = "frustrated"
    TIRED = "tired"
    NEUTRAL = "neutral"


@dataclass
class ChildState:
    """Snapshot of the child's current state."""
    activity: ChildActivity = ChildActivity.UNKNOWN
    mood: ChildMood = ChildMood.NEUTRAL
    focus_duration: float = 0.0       # Seconds of continuous focus
    idle_duration: float = 0.0        # Seconds of no clear activity
    last_speech_time: float = 0.0     # When child last spoke
    looking_at_camera: bool = False   # Is child looking toward camera?
    holding_object: bool = False      # Holding something up to show?
    holding_book: bool = False        # Holding a book?
    person_count: int = 1             # How many people detected?
    timestamp: float = field(default_factory=time.time)


class DecisionEngine:
    """Core decision engine for when and how to interact.

    Manages a state machine, intervention frequency limits,
    and context-appropriate role selection.
    """

    def __init__(
        self,
        max_interventions_per_hour: int = 5,
        min_interval_between_interventions: float = 60.0,  # seconds
        focus_protect_threshold: float = 180.0,   # 3 min before considering intervention
        idle_engage_threshold: float = 300.0,      # 5 min idle before proactive check
        bedtime_start: int = 20,                   # 8 PM
        bedtime_end: int = 7,                      # 7 AM
    ):
        self.max_interventions_per_hour = max_interventions_per_hour
        self.min_interval = min_interval_between_interventions
        self.focus_protect_threshold = focus_protect_threshold
        self.idle_engage_threshold = idle_engage_threshold
        self.bedtime_start = bedtime_start
        self.bedtime_end = bedtime_end

        # State
        self.current_state: TutorState = TutorState.OBSERVING
        self._last_intervention_time: float = 0.0
        self._intervention_count_this_hour: int = 0
        self._hour_start: float = time.time()
        self._child_state: ChildState = ChildState()

        # Observing history
        self._activity_start_time: float = time.time()
        self._current_activity: ChildActivity = ChildActivity.UNKNOWN

    # ── Main decision method ─────────────────────────────────────

    def decide(
        self,
        child_state: ChildState,
        child_spoke: str = "",
        scene_changed: bool = False,
        new_objects: list[str] | None = None,
    ) -> "InterventionDecision":
        """The core decision: should we say something, and if so, what?

        Args:
            child_state: Current child state snapshot
            child_spoke: What the child said (empty if silent)
            scene_changed: Whether the scene has changed meaningfully
            new_objects: New objects detected in scene

        Returns:
            InterventionDecision with action, role, and priority
        """
        self._update_child_state(child_state)
        self._reset_hourly_counter_if_needed()

        # Priority 0: Child explicitly called us
        if child_spoke and self._is_calling_tutor(child_spoke):
            return InterventionDecision(
                should_speak=True,
                state=TutorState.ENGAGING,
                priority=0,
                reason="child_called",
            )

        # Priority 1: Child is showing something
        if child_state.looking_at_camera and child_state.holding_object:
            return InterventionDecision(
                should_speak=True,
                state=TutorState.ENGAGING,
                priority=1,
                reason="child_showing_object",
            )

        # Priority 2: Child holding a book (reading time)
        if child_state.holding_book and not child_state.focus_duration > 30:
            # Only engage if they just picked up the book (not mid-reading)
            return InterventionDecision(
                should_speak=True,
                state=TutorState.TEACHING,
                priority=2,
                reason="reading_moment",
            )

        # Protection: DON'T interrupt deep focus
        if child_state.focus_duration > self.focus_protect_threshold:
            return InterventionDecision(
                should_speak=False,
                state=TutorState.OBSERVING,
                priority=99,
                reason="protecting_focus",
            )

        # Protection: DON'T interrupt homework
        if child_state.activity == ChildActivity.STUDYING:
            return InterventionDecision(
                should_speak=False,
                state=TutorState.OBSERVING,
                priority=99,
                reason="studying",
            )

        # Protection: DON'T interrupt when multiple people (social time)
        if child_state.person_count > 1:
            return InterventionDecision(
                should_speak=False,
                state=TutorState.OBSERVING,
                priority=99,
                reason="social_time",
            )

        # Protection: BEDTIME — don't speak
        if self._is_bedtime():
            return InterventionDecision(
                should_speak=False,
                state=TutorState.RESTING,
                priority=99,
                reason="bedtime",
            )

        # Opportunity: Child is idle/bored
        if child_state.idle_duration > self.idle_engage_threshold:
            if self._can_intervene():
                return InterventionDecision(
                    should_speak=True,
                    state=TutorState.GAMING,
                    priority=3,
                    reason="child_idle",
                )

        # Opportunity: Scene changed with new objects
        if scene_changed and new_objects:
            if self._can_intervene():
                return InterventionDecision(
                    should_speak=True,
                    state=TutorState.TEACHING,
                    priority=4,
                    reason="new_objects",
                    objects=new_objects,
                )

        # Opportunity: Child spoke but not to us (narration opportunity)
        if child_spoke and not self._is_calling_tutor(child_spoke):
            if self._can_intervene():
                return InterventionDecision(
                    should_speak=True,
                    state=TutorState.ENGAGING,
                    priority=5,
                    reason="child_spoke",
                )

        # Default: stay silent, keep observing
        return InterventionDecision(
            should_speak=False,
            state=TutorState.OBSERVING,
            priority=10,
            reason="observing",
        )

    # ── State transitions ────────────────────────────────────────

    def transition(self, new_state: TutorState):
        """Transition to a new state."""
        old = self.current_state
        self.current_state = new_state
        if new_state in (TutorState.ENGAGING, TutorState.TEACHING, TutorState.GAMING):
            self._last_intervention_time = time.time()
            self._intervention_count_this_hour += 1

    def return_to_observing(self):
        """Return to observing after an interaction ends."""
        self.current_state = TutorState.OBSERVING

    # ── Helpers ──────────────────────────────────────────────────

    def _can_intervene(self) -> bool:
        """Check if we're allowed to intervene (rate limits)."""
        now = time.time()
        if self._intervention_count_this_hour >= self.max_interventions_per_hour:
            return False
        if now - self._last_intervention_time < self.min_interval:
            return False
        return True

    def _update_child_state(self, state: ChildState):
        """Track activity changes for focus/idle duration."""
        if state.activity != self._current_activity:
            self._activity_start_time = time.time()
            self._current_activity = state.activity

        if state.activity in (ChildActivity.PLAYING, ChildActivity.DRAWING,
                               ChildActivity.READING, ChildActivity.STUDYING):
            state.focus_duration = time.time() - self._activity_start_time
            state.idle_duration = 0.0
        elif state.activity == ChildActivity.IDLE:
            state.idle_duration = time.time() - self._activity_start_time
            state.focus_duration = 0.0

        self._child_state = state

    @staticmethod
    def _is_calling_tutor(text: str) -> bool:
        """Check if child is explicitly calling the tutor."""
        text_lower = text.lower().strip()
        call_phrases = [
            "emma", "hello", "hi", "hey",
            "what's this", "what is this",
            "look", "see", "watch",
            "老师", "过来", "看看",
        ]
        return any(phrase in text_lower for phrase in call_phrases)

    def _is_bedtime(self) -> bool:
        """Check if current time is in bedtime window."""
        from datetime import datetime
        hour = datetime.now().hour
        if self.bedtime_start <= 23:
            return hour >= self.bedtime_start or hour < self.bedtime_end
        return hour >= self.bedtime_start

    def _reset_hourly_counter_if_needed(self):
        """Reset the hourly intervention counter if an hour has passed."""
        if time.time() - self._hour_start > 3600:
            self._intervention_count_this_hour = 0
            self._hour_start = time.time()


@dataclass
class InterventionDecision:
    """Result of the decision engine."""
    should_speak: bool
    state: TutorState
    priority: int        # 0 = highest priority, 99 = never
    reason: str
    objects: list[str] = field(default_factory=list)
    suggested_role: str = ""  # "tutor", "playmate", "narrator"

    def __post_init__(self):
        if self.state == TutorState.ENGAGING:
            self.suggested_role = "tutor"
        elif self.state == TutorState.TEACHING:
            self.suggested_role = "narrator"
        elif self.state == TutorState.GAMING:
            self.suggested_role = "playmate"
