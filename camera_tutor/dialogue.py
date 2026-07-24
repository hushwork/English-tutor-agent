"""Dialogue Manager — child-friendly conversation generation.

Manages the conversation flow between the child and Emma (the tutor).
Adapts language complexity based on child age and observed proficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DialogueTurn:
    """A single turn in the conversation."""
    role: str  # "emma" or "child"
    content: str
    timestamp: float


class EmmaDialogue:
    """Manages dialogue generation for child interaction.

    Handles:
    - Age-appropriate language adaptation
    - Error correction (recast, not explicit)
    - Conversation flow (greeting → engagement → teaching → closing)
    - Game state management
    """

    # ── Age-specific language parameters ─────────────────────────

    AGE_CONFIG = {
        3: {
            "max_words_per_sentence": 5,
            "vocabulary_level": "A0 (colors, animals, family, body parts)",
            "sentence_types": "statements and yes/no questions only",
            "repetition": "high — repeat key words 2-3 times",
            "tone": "very high energy, sing-song, exaggerated",
            "example": "Red car! Vroom vroom! Can you say 'car'?",
        },
        5: {
            "max_words_per_sentence": 8,
            "vocabulary_level": "A1 (daily objects, simple actions, feelings)",
            "sentence_types": "statements, yes/no, simple wh- questions",
            "repetition": "medium — repeat new words in different contexts",
            "tone": "warm and encouraging, clear enunciation",
            "example": "Wow! You found a red car! Is it fast?",
        },
        7: {
            "max_words_per_sentence": 10,
            "vocabulary_level": "A2 (describe features, tell simple stories)",
            "sentence_types": "all question types, comparisons, because/when",
            "repetition": "low — only for brand new vocabulary",
            "tone": "natural and conversational, peer-like",
            "example": "That's a cool red car! What makes it go so fast?",
        },
        9: {
            "max_words_per_sentence": 12,
            "vocabulary_level": "B1 (explain reasons, share opinions)",
            "sentence_types": "all types, conditional, hypothetical",
            "repetition": "minimal",
            "tone": "natural peer conversation",
            "example": "I like how you described the red car. Where would you drive it?",
        },
    }

    # ── Error correction templates ───────────────────────────────

    CORRECTION_TEMPLATES = {
        "missing_article": [
            "Yes! {corrected}! Good job!",
            "I see {corrected}! Can you say that?",
        ],
        "wrong_plural": [
            "Wow, {corrected}! One {singular}, two {plural}!",
            "You have {corrected}! That's right!",
        ],
        "wrong_tense": [
            "That happened! {corrected}!",
            "Oh, {corrected}! Tell me more!",
        ],
        "word_order": [
            "I think you mean: {corrected}. Is that right?",
            "Let me try: {corrected}. Does that sound better?",
        ],
        "wrong_word": [
            "Close! It's called {corrected}. Can you say {corrected}?",
            "Almost! We say {corrected}. Your turn!",
        ],
    }

    def __init__(self, child_age: int = 5):
        self.child_age = child_age
        self.history: list[DialogueTurn] = []
        self._game_active = False
        self._game_type = ""

    # ── Public API ───────────────────────────────────────────────

    def format_age_guidance(self) -> str:
        """Get age-specific language guidance for LLM prompts."""
        config = self.AGE_CONFIG.get(
            self.child_age,
            self.AGE_CONFIG[5],
        )
        return (
            f"Child age: {self.child_age}\n"
            f"Max words per sentence: {config['max_words_per_sentence']}\n"
            f"Vocabulary: {config['vocabulary_level']}\n"
            f"Sentences: {config['sentence_types']}\n"
            f"Repetition: {config['repetition']}\n"
            f"Tone: {config['tone']}\n"
            f"Example: {config['example']}"
        )

    def generate_correction(
        self,
        child_said: str,
        corrected_form: str,
        error_type: str = "general",
    ) -> str:
        """Generate a gentle correction using recast (not explicit).

        Follows the principle: acknowledge → model correct version → encourage.
        Never says "that's wrong" or "you made a mistake."
        """
        import random

        templates = self.CORRECTION_TEMPLATES.get(
            error_type,
            ["I hear you! {corrected}! Let's say it together: {corrected}!"],
        )
        template = random.choice(templates)
        return template.format(corrected=corrected_form)

    def start_game(self, game_type: str) -> str:
        """Start a game session."""
        self._game_active = True
        self._game_type = game_type

        openers = {
            "i_spy": "Let's play I Spy! I spy with my little eye... something in the room! Can you guess?",
            "simon_says": "Let's play Simon Says! Simon says... stand up! Your turn!",
            "counting": "Let's count together! Ready? One, two, three! What do you see?",
            "movement": "Let's move! Jump up high! Jump, jump, jump!",
        }
        return openers.get(game_type, openers["i_spy"])

    def end_game(self) -> str:
        """End a game session gracefully."""
        self._game_active = False
        closers = [
            "That was fun! Let's play again later!",
            "Great game! You did so well!",
            "Time to rest! Good game!",
        ]
        import random
        return random.choice(closers)

    @property
    def is_in_game(self) -> bool:
        return self._game_active
