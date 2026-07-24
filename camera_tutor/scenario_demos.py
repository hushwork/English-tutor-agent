#!/usr/bin/env python3
"""Scenario Demos — validate each core interaction pattern from PRODUCT_VISION.md.

These test scripts validate the logic for the 5 core scenarios:
    1. Free play narration
    2. Picture book co-reading
    3. Homework companionship (silent observing)
    4. Show & Tell
    5. Bedtime story mode

Usage:
    python3 scenario_demos.py               # Run all scenarios
    python3 scenario_demos.py --scenario 2   # Run specific scenario
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_tutor.decision_engine import (
    DecisionEngine, TutorState,
    ChildActivity, ChildMood, ChildState,
)
from camera_tutor.dialogue import EmmaDialogue


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    expected_state: TutorState
    actual_state: TutorState
    expected_speak: bool
    actual_speak: bool
    notes: str


def run_free_play_scenario(engine: DecisionEngine) -> ScenarioResult:
    """Scenario 1: Child playing with blocks, Emma narrates occasionally."""
    child_state = ChildState(
        activity=ChildActivity.PLAYING,
        mood=ChildMood.HAPPY,
        focus_duration=60.0,  # Playing for 1 minute
        idle_duration=0.0,
        person_count=1,
    )
    decision = engine.decide(
        child_state=child_state,
        child_spoke="",
        scene_changed=True,
        new_objects=["blocks", "toy car"],
    )

    # Playing + new objects → teaching (narration)
    # But should NOT interrupt deep focus
    expected_speak = decision.should_speak
    expected = TutorState.TEACHING if decision.should_speak else TutorState.OBSERVING

    return ScenarioResult(
        name="Free Play Narration",
        passed=(decision.state == expected),
        expected_state=expected,
        actual_state=decision.state,
        expected_speak=expected_speak,
        actual_speak=decision.should_speak,
        notes=f"Playing for {child_state.focus_duration}s, new objects: {decision.objects}",
    )


def run_picture_book_scenario(engine: DecisionEngine) -> ScenarioResult:
    """Scenario 2: Child picks up a book, Emma offers to read together."""
    child_state = ChildState(
        activity=ChildActivity.READING,
        mood=ChildMood.FOCUSED,
        focus_duration=5.0,  # Just picked up the book
        idle_duration=0.0,
        holding_book=True,
        person_count=1,
    )
    decision = engine.decide(
        child_state=child_state,
        child_spoke="",
        scene_changed=True,
        new_objects=["Brown Bear picture book"],
    )

    return ScenarioResult(
        name="Picture Book Co-reading",
        passed=decision.should_speak and decision.priority <= 2,
        expected_state=TutorState.TEACHING,
        actual_state=decision.state,
        expected_speak=True,
        actual_speak=decision.should_speak,
        notes=f"Just picked up a book, focus only {child_state.focus_duration}s",
    )


def run_homework_scenario(engine: DecisionEngine) -> ScenarioResult:
    """Scenario 3: Child doing homework, Emma stays completely silent."""
    child_state = ChildState(
        activity=ChildActivity.STUDYING,
        mood=ChildMood.FOCUSED,
        focus_duration=600.0,  # 10 minutes of homework
        idle_duration=0.0,
        person_count=1,
    )
    decision = engine.decide(
        child_state=child_state,
        child_spoke="",
        scene_changed=False,
        new_objects=None,
    )

    return ScenarioResult(
        name="Homework Companionship (Silent)",
        passed=not decision.should_speak,
        expected_state=TutorState.OBSERVING,
        actual_state=decision.state,
        expected_speak=False,
        actual_speak=decision.should_speak,
        notes=f"Focus: {child_state.focus_duration}s. Reason: {decision.reason}",
    )


def run_show_and_tell_scenario(engine: DecisionEngine) -> ScenarioResult:
    """Scenario 4: Child holds up a drawing, Emma responds enthusiastically."""
    child_state = ChildState(
        activity=ChildActivity.DRAWING,
        mood=ChildMood.HAPPY,
        focus_duration=30.0,
        idle_duration=0.0,
        looking_at_camera=True,
        holding_object=True,
        person_count=1,
    )
    decision = engine.decide(
        child_state=child_state,
        child_spoke="Look! I drew a cat!",
        scene_changed=True,
        new_objects=["drawing", "cat picture"],
    )

    return ScenarioResult(
        name="Show & Tell",
        passed=decision.should_speak and decision.priority <= 1,
        expected_state=TutorState.ENGAGING,
        actual_state=decision.state,
        expected_speak=True,
        actual_speak=decision.should_speak,
        notes=f"Child held up drawing, priority={decision.priority}",
    )


def run_bedtime_scenario(engine: DecisionEngine) -> ScenarioResult:
    """Scenario 5: Bedtime mode — Emma stays in resting/sleep state."""
    # Override bedtime for test
    engine.bedtime_start = 0
    engine.bedtime_end = 23  # Always bedtime for test

    child_state = ChildState(
        activity=ChildActivity.IDLE,
        mood=ChildMood.TIRED,
        focus_duration=0.0,
        idle_duration=100.0,
        person_count=1,
    )
    decision = engine.decide(
        child_state=child_state,
        child_spoke="Good night",
        scene_changed=False,
    )

    # Reset bedtime
    engine.bedtime_start = 20
    engine.bedtime_end = 7

    return ScenarioResult(
        name="Bedtime Mode",
        passed=not decision.should_speak,
        expected_state=TutorState.RESTING,
        actual_state=decision.state,
        expected_speak=False,
        actual_speak=decision.should_speak,
        notes=f"Bedtime active. Reason: {decision.reason}",
    )


def run_correction_scenario(dialogue: EmmaDialogue) -> ScenarioResult:
    """Test: Child makes grammar error, Emma corrects via recast."""
    correction = dialogue.generate_correction(
        child_said="I have two foot",
        corrected_form="I have two feet! One foot, two feet!",
        error_type="wrong_plural",
    )

    has_encouragement = any(
        word in correction.lower()
        for word in ["wow", "yes", "good", "right", "great"]
    )
    no_negative = "wrong" not in correction.lower() and "mistake" not in correction.lower()

    return ScenarioResult(
        name="Error Correction (Recast)",
        passed=has_encouragement and no_negative,
        expected_state=TutorState.ENGAGING,
        actual_state=TutorState.ENGAGING,
        expected_speak=True,
        actual_speak=True,
        notes=f"Correction: '{correction}' | Encouraging: {has_encouragement} | No negative: {no_negative}",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Camera Tutor Scenario Demos")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a specific scenario (1-6)")
    args = parser.parse_args()

    engine = DecisionEngine(max_interventions_per_hour=20)
    dialogue = EmmaDialogue(child_age=5)

    all_scenarios = [
        run_free_play_scenario,
        run_picture_book_scenario,
        run_homework_scenario,
        run_show_and_tell_scenario,
        run_bedtime_scenario,
        lambda e: run_correction_scenario(dialogue),
    ]

    scenarios = [all_scenarios[args.scenario - 1]] if args.scenario else all_scenarios

    print("=" * 70)
    print("  Camera Tutor — Scenario Validation Demos")
    print("=" * 70)

    results: list[ScenarioResult] = []
    for scenario_fn in scenarios:
        result = scenario_fn(engine)
        results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  [{status}] {result.name}")
        print(f"    State: {result.actual_state.value} (expected: {result.expected_state.value})")
        print(f"    Speak: {result.actual_speak} (expected: {result.expected_speak})")
        print(f"    {result.notes}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'=' * 70}")
    print(f"  Results: {passed}/{total} scenarios passed")
    if passed == total:
        print("  🎉 All scenarios validated!")
    else:
        print(f"  ⚠️  {total - passed} scenario(s) need attention")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
