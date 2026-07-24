#!/usr/bin/env python3
"""Camera Tutor — Real-time Interactive Demo.

Core loop:
  Camera watches → detects moment to speak → Emma streams audio
  → plays through speaker → VAD listens simultaneously
  → child speaks → playback interrupted → Emma responds → loop

Usage:
  python3 demo.py                          # Real hardware
  python3 demo.py --mock                   # No hardware (text-based interactive)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Mock demo (no hardware, text-based, validates logic) ────────

async def run_mock():
    """Text-based interactive demo — validates the logic without hardware.

    You type what the child says. Emma responds in text.
    Type 'stop' or 'bye' to simulate child interrupting.
    """
    from camera_tutor.decision_engine import (
        DecisionEngine, ChildActivity, ChildMood, ChildState,
    )
    from camera_tutor.dialogue import EmmaDialogue

    engine = DecisionEngine()
    dialogue = EmmaDialogue(child_age=5)

    print("=" * 60)
    print("  Camera Tutor — Mock Interactive Demo")
    print("=" * 60)
    print("  Type what the child says (or 'q' to quit)")
    print("  Type 'stop' to interrupt Emma mid-response")
    print("=" * 60)

    # Simulated activities and Emma responses
    conversation = []
    cycle = 0

    while True:
        user_input = input("\n👧 Child: ").strip()
        if user_input.lower() in ('q', 'quit', 'exit'):
            break

        # Simulate child state based on input
        if 'stop' in user_input.lower() or 'bye' in user_input.lower():
            child_state = ChildState(
                activity=ChildActivity.IDLE,
                mood=ChildMood.NEUTRAL,
                looking_at_camera=True,
            )
        elif 'look' in user_input.lower() or 'see' in user_input.lower():
            child_state = ChildState(
                activity=ChildActivity.PLAYING,
                mood=ChildMood.HAPPY,
                looking_at_camera=True,
                holding_object=True,
            )
        elif 'book' in user_input.lower() or 'read' in user_input.lower():
            child_state = ChildState(
                activity=ChildActivity.READING,
                mood=ChildMood.FOCUSED,
                holding_book=True,
            )
        elif 'homework' in user_input.lower() or 'study' in user_input.lower():
            child_state = ChildState(
                activity=ChildActivity.STUDYING,
                mood=ChildMood.FOCUSED,
                focus_duration=600,
            )
        elif 'bored' in user_input.lower():
            child_state = ChildState(
                activity=ChildActivity.IDLE,
                mood=ChildMood.BORED,
                idle_duration=400,
            )
        else:
            child_state = ChildState(
                activity=ChildActivity.PLAYING,
                mood=ChildMood.HAPPY,
            )

        # Decision
        decision = engine.decide(
            child_state=child_state,
            child_spoke=user_input,
            scene_changed=(cycle == 0),
        )

        print(f"  📊 State: {decision.state.value} | Speak: {decision.should_speak} | Reason: {decision.reason}")

        if decision.should_speak:
            # Simulate streaming response (normally from Qwen-Omni)
            print("  🤖 Emma (streaming): ", end="", flush=True)

            # Simulate chunks arriving with delay
            responses = {
                "engaging": [
                    "Wow! What do you have there?",
                    "I see! That looks amazing!",
                    "Tell me more about that!",
                ],
                "teaching": [
                    "That's a great red block! Can you say 'red block'?",
                    "Look at the picture book! What animal do you see?",
                    "One, two, three! Let's count together!",
                ],
                "gaming": [
                    "Let's play! I spy something in the room!",
                    "Simon says... touch your nose! Touch, touch, touch!",
                    "Jump up high! Jump, jump, jump!",
                ],
            }
            response = responses.get(decision.state.value, responses["engaging"])[cycle % 3]

            # Stream word by word
            words = response.split()
            for i, word in enumerate(words):
                # Simulate: child interrupts mid-response
                if user_input.lower() == 'stop' and i > 1:
                    print(" [INTERRUPTED]")
                    print("  🤫 Emma stopped — child spoke!")
                    break
                print(word, end=" ", flush=True)
                await asyncio.sleep(0.2)  # Simulate streaming delay

            if i == len(words) - 1:  # Completed without interruption
                print()
                conversation.append({"role": "emma", "content": response})

        elif decision.reason == "studying":
            print("  🤫 Emma stays silent — child is studying")
        elif decision.reason == "protecting_focus":
            print("  🤫 Emma stays silent — protecting focus")
        else:
            print("  🤫 Emma observes silently")

        conversation.append({"role": "child", "content": user_input})
        cycle += 1

    print(f"\n👋 Goodbye! {cycle} turns completed.")


# ── Real hardware demo ──────────────────────────────────────────

async def run_hardware():
    """Real hardware demo with camera + microphone + speaker.

    Flow:
      1. Camera watches → scene analysis
      2. Decision engine: should Emma speak?
      3. Qwen-Omni streams Emma's audio response
      4. PlaybackController plays each chunk as it arrives
      5. VAD runs simultaneously → child interrupts → Emma stops
      6. Capture child's speech → send to Qwen-Omni → goto 3
    """
    from camera_tutor.camera import CameraPipeline
    from camera_tutor.audio_io import AudioCapture, PlaybackController
    from camera_tutor.scene_analyzer import SceneAnalyzer
    from camera_tutor.decision_engine import DecisionEngine, ChildActivity, ChildMood, ChildState
    from camera_tutor.omni_client import OmniClient, ModelMode
    from camera_tutor.dialogue import EmmaDialogue
    from camera_tutor.parent_report import ParentReportEngine

    print("=" * 60)
    print("  Camera Tutor — Real-time Interactive Demo")
    print("=" * 60)

    # ── Init hardware ────────────────────────────────────────
    print("\n[1/6] Camera...")
    camera = CameraPipeline(camera_id=0, fps=3, resolution=(640, 480))
    camera.start()
    print("       ✅ Ready")

    print("[2/6] Microphone (VAD always on)...")
    capture = AudioCapture(sample_rate=16000, chunk_duration=0.05, vad_threshold=0.3)
    capture.start()
    print("       ✅ Ready — VAD listening")

    print("[3/6] Speaker...")
    playback = PlaybackController(sample_rate=24000, chunk_ms=50)
    playback.start()
    print("       ✅ Ready")

    print("[4/6] Qwen-Omni client...")
    omni = OmniClient(mode=ModelMode.AUTO)
    available = await omni.check_available()
    print(f"       ✅ local={available['local']}, cloud={available['cloud']}")

    print("[5/6] Decision engine...")
    engine = DecisionEngine(max_interventions_per_hour=10)
    analyzer = SceneAnalyzer()
    dialogue = EmmaDialogue(child_age=5)
    reporter = ParentReportEngine()
    print("       ✅ Ready")

    print("[6/6] Starting interactive loop...")
    print("\n" + "=" * 60)
    print("  Camera Tutor is watching. Press Ctrl+C to stop.")
    print("  Child can speak at any time — Emma will stop and listen.")
    print("=" * 60)

    history: list[dict] = []  # Conversation turns
    cycle = 0

    try:
        while True:
            # ── Step 1: Observe scene ─────────────────────────
            frame = camera.capture()
            if frame is None:
                await asyncio.sleep(0.5)
                continue

            # Only process key frames (scene changed meaningfully)
            if not frame.is_key_frame:
                # Still check VAD while waiting
                if capture.is_speaking and not playback.is_playing:
                    child_audio = capture.read_speech_segment(timeout=3.0)
                    if child_audio:
                        print("\n👧 Child is speaking...")
                        await handle_child_speech(child_audio, capture, playback, omni, history, reporter)
                await asyncio.sleep(0.5)
                continue

            # ── Step 2: Scene analysis ────────────────────────
            frame_b64 = CameraPipeline.to_base64(frame)
            try:
                result = await omni.analyze_scene(
                    image_b64=frame_b64,
                    context=analyzer.get_recent_context(),
                )
            except Exception as e:
                print(f"[WARN] Scene analysis failed: {e}")
                continue

            # ── Step 3: Child state ───────────────────────────
            activity = analyzer.infer_child_activity(result.objects)
            mood = analyzer.infer_child_mood(activity, 30, 0)
            child_state = ChildState(activity=activity, mood=mood)

            # ── Step 4: Decision ──────────────────────────────
            decision = engine.decide(
                child_state=child_state,
                child_spoke="",
                scene_changed=True,
                new_objects=result.objects,
            )

            if decision.should_speak:
                print(f"\n🎯 Emma speaks [{decision.reason}]: ", end="", flush=True)

                # ── Step 5: Stream Emma's response ────────────
                prompt = result.suggested_response or "Describe what the child is doing."
                transcript = ""

                async for chunk in omni.stream_response(text=prompt, image_b64=frame_b64):
                    if chunk["type"] == "text":
                        transcript += chunk["content"]
                        print(chunk["content"], end="", flush=True)

                    elif chunk["type"] == "audio":
                        # Push audio to speaker immediately
                        import numpy as np
                        audio_np = np.frombuffer(chunk["content"], dtype=np.int16).astype(np.float32) / 32767.0
                        playback.enqueue(audio_np)

                    elif chunk["type"] == "done":
                        break

                    # ── Step 6: Check for interruption ────────
                    chunk_data = capture.read_chunk()
                    if chunk_data and capture.is_speaking and playback.is_playing:
                        elapsed = playback.elapsed
                        if elapsed > 0.5:  # Don't interrupt in first 500ms
                            playback.interrupt(reason="vad")
                            print("\n  ⚡ INTERRUPTED — child spoke!")
                            break

                # Wait for playback to finish (unless interrupted)
                while playback.is_playing:
                    # Still checking for more child speech
                    chunk_data = capture.read_chunk()
                    if chunk_data and capture.is_speaking and playback.elapsed > 0.5:
                        playback.interrupt(reason="vad")
                        print("\n  ⚡ INTERRUPTED!")
                        break
                    await asyncio.sleep(0.05)

                # ── Step 7: Listen for child's response ────────
                if capture.is_speaking or playback.was_interrupted:
                    child_audio = capture.read_speech_segment(timeout=5.0)
                    if child_audio:
                        await handle_child_speech(child_audio, capture, playback, omni, history, reporter)

                if transcript:
                    print()  # New line after transcript
                    history.append({"role": "emma", "content": transcript})
                    reporter.log_event("emma_spoke", {"text": transcript[:200]})

            else:
                # Emma stays silent. Check if child is speaking anyway.
                if capture.is_speaking and not playback.is_playing:
                    child_audio = capture.read_speech_segment(timeout=3.0)
                    if child_audio:
                        print("\n👧 Child spoke (unsolicited)")
                        await handle_child_speech(child_audio, capture, playback, omni, history, reporter)

            cycle += 1

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        camera.stop()
        capture.stop()
        playback.stop()
        await omni.close()

    report = reporter.generate_daily_report()
    print(f"\nSession: {report.total_english_input} Emma utterances, {report.child_utterances} child responses")


async def handle_child_speech(
    child_audio, capture, playback, omni, history, reporter,
):
    """Process a child's speech segment and get Emma's response."""
    from camera_tutor.audio_io import AudioCapture
    audio_b64 = AudioCapture.to_wav_base64(child_audio)
    print("  🎤 Captured child speech, thinking...")

    # Build prompt with conversation context
    context = "\n".join(
        f"{'Emma' if h['role']=='emma' else 'Child'}: {h['content'][:80]}"
        for h in history[-4:]
    )

    prompt = (
        f"You are Emma, English tutor for a young child.\n"
        f"Recent conversation:\n{context}\n\n"
        f"The child just said something (see audio). "
        f"Respond with ONE short, encouraging English sentence (under 10 words)."
    )

    async for chunk in omni.stream_response(text=prompt, child_audio_b64=audio_b64):
        if chunk["type"] == "text":
            print(f"  🤖 Emma: {chunk['content']}", end="", flush=True)
        elif chunk["type"] == "audio":
            import numpy as np
            audio_np = np.frombuffer(chunk["content"], dtype=np.int16).astype(np.float32) / 32767.0
            playback.enqueue(audio_np)
        elif chunk["type"] == "done":
            print()
            break

    reporter.log_event("child_spoke")
    reporter.log_event("interaction_end")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Camera Tutor Real-time Demo")
    parser.add_argument("--mock", action="store_true",
                        help="Text-based mock (no hardware needed)")
    args = parser.parse_args()

    if args.mock:
        asyncio.run(run_mock())
    else:
        asyncio.run(run_hardware())


if __name__ == "__main__":
    main()
