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


# ── Kokoro voices ──────────────────────────────────────────────────

KOKORO_VOICES = {
    "en-us-female": "af_heart",       # American female, warm
    "en-us-male": "am_adam",          # American male
    "en-gb-female": "bf_emma",        # British female
    "en-gb-male": "bm_george",        # British male
}
DEFAULT_KOKORO_VOICE = "af_heart"

# Edge-tts → Kokoro voice mapping for seamless switching
_EDGE_TO_KOKORO = {
    "en-US-JennyNeural": "af_heart",
    "en-US-GuyNeural": "am_adam",
    "en-GB-SoniaNeural": "bf_emma",
    "en-GB-RyanNeural": "bm_george",
    "en-AU-NatashaNeural": "af_heart",
}


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
    """Generate high-quality speech and save to file.

    Uses local Kokoro TTS by default (offline). Set USE_CLOUD_TTS=1 for
    edge-tts (Microsoft neural voices, requires network).

    Args:
        text: Text to synthesize.
        voice: Voice name (edge-tts voice for cloud, kokoro voice for local).
        output_path: Optional path to save audio. If None, uses temp file.

    Returns:
        Path to audio file, or None if failed.
    """
    if _use_local_tts():
        kokoro_voice = _EDGE_TO_KOKORO.get(voice, DEFAULT_KOKORO_VOICE)
        return await _speak_kokoro(text, kokoro_voice, output_path)
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


# ── Kokoro TTS (local, offline, high-quality English) ──────────────

_kokoro_pipeline = None


def _get_kokoro():
    """Get or create the Kokoro pipeline singleton (CPU)."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")  # American English
    return _kokoro_pipeline


async def _speak_kokoro(
    text: str,
    voice: str = "af_heart",
    output_path: str | None = None,
) -> str | None:
    """Generate speech with local Kokoro TTS to a WAV file.

    Returns the file path, or None on failure.
    """
    import numpy as np
    import soundfile as sf

    try:
        pipeline = _get_kokoro()
    except Exception:
        return None

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="kokoro_")
        os.close(fd)

    try:
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice):
            chunks.append(audio)
        if not chunks:
            return None
        combined = np.concatenate(chunks)
        sf.write(output_path, combined, 24000)
        return output_path if os.path.getsize(output_path) > 0 else None
    except Exception:
        return None


def is_kokoro_available() -> bool:
    """Check if kokoro is installed."""
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False


# ── TTS mode selection ─────────────────────────────────────────────

def _use_local_tts() -> bool:
    """Return True if local TTS should be used instead of cloud."""
    return (
        os.environ.get("USE_CLOUD_TTS", "").lower() not in ("1", "true", "yes")
        and is_kokoro_available()
    )
