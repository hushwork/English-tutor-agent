"""TTS (Text-to-Speech) module — speak English text aloud.

Two modes:
- Quick: uses speech-dispatcher (spd-say) for immediate playback
- High-quality: uses edge-tts (Microsoft neural voices) saved to file
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

# ── Voices ──────────────────────────────────────────────────────────

EDGE_VOICES = {
    "en-us-jenny": "en-US-JennyNeural",   # Female, American
    "en-us-guy": "en-US-GuyNeural",       # Male, American
    "en-gb-sonia": "en-GB-SoniaNeural",   # Female, British
    "en-gb-ryan": "en-GB-RyanNeural",     # Male, British
    "en-au-natasha": "en-AU-NatashaNeural", # Female, Australian
}

DEFAULT_EDGE_VOICE = "en-US-JennyNeural"
DEFAULT_EDGE_RATE = "+0%"  # -50% to +50%
DEFAULT_EDGE_VOLUME = "+0%"


# ── Speech-dispatcher (quick, robotic) ──────────────────────────────

def _speak_quick(text: str, rate: int = 0) -> bool:
    """Speak text using speech-dispatcher (spd-say). Returns success."""
    try:
        cmd = [
            "spd-say", "-o", "pulse",
            "-l", "en",
            "-r", str(max(-100, min(100, 180 + rate))),
            text,
        ]
        subprocess.run(cmd, timeout=30, capture_output=True)
        return True
    except Exception:
        return False


def _speak_sync(text: str, voice: str = "", rate: int = 0) -> bool:
    """Blocking TTS via speech-dispatcher. Returns True if played."""
    return _speak_quick(text, rate=rate)


# ── Edge-TTS (high-quality, async, saves file) ─────────────────────

async def _speak_edge(
    text: str,
    voice: str = DEFAULT_EDGE_VOICE,
    output_path: str | None = None,
) -> str | None:
    """Generate speech with edge-tts to an MP3 file. Returns the file path or None."""
    try:
        import edge_tts
    except ImportError:
        return None

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp3", prefix="english_tutor_")
        os.close(fd)

    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=DEFAULT_EDGE_RATE,
        volume=DEFAULT_EDGE_VOLUME,
    )
    await communicate.save(output_path)
    return output_path if os.path.getsize(output_path) > 0 else None


# ── Public API ──────────────────────────────────────────────────────

def speak_now(text: str, rate: int = 0) -> bool:
    """Quickly speak text aloud using system speech-dispatcher.

    Args:
        text: Text to speak (will be truncated to 2000 chars for SPD).
        rate: Speech rate offset from default (-50 to +50). Negative = slower.

    Returns:
        True if speech was played successfully.
    """
    # Truncate very long text
    if len(text) > 2000:
        text = text[:1997] + "..."
    return _speak_sync(text, rate=rate)


async def speak_to_file(
    text: str,
    voice: str = DEFAULT_EDGE_VOICE,
    output_path: str | None = None,
) -> str | None:
    """Generate high-quality speech using edge-tts and save to file.

    Args:
        text: Text to synthesize.
        voice: Edge TTS voice name.
        output_path: Optional path to save MP3. If None, uses temp file.

    Returns:
        Path to MP3 file, or None if failed.
    """
    return await _speak_edge(text, voice, output_path)


def list_voices() -> dict[str, str]:
    """Return available edge-tts voices."""
    return dict(EDGE_VOICES)


def is_speech_dispatcher_available() -> bool:
    """Check if speech-dispatcher is available on this system."""
    return subprocess.run(
        ["which", "spd-say"], capture_output=True
    ).returncode == 0


def is_edge_tts_available() -> bool:
    """Check if edge-tts package is installed."""
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False
