"""STT (Speech-to-Text) module — record voice input and transcribe.

Uses arecord (ALSA) for recording and faster-whisper for local transcription.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# ── Recording ───────────────────────────────────────────────────────

SAMPLE_RATE = 16000
DEFAULT_DURATION = 10  # seconds


def is_recording_available() -> bool:
    """Check if arecord is available for recording."""
    return subprocess.run(["which", "arecord"], capture_output=True).returncode == 0


def record_audio(
    duration: int = DEFAULT_DURATION,
    sample_rate: int = SAMPLE_RATE,
    output_path: str | None = None,
) -> str | None:
    """Record audio from default microphone using arecord.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Sample rate in Hz (16000 recommended for Whisper).
        output_path: Optional path to save WAV file.

    Returns:
        Path to the recorded WAV file, or None on failure.
    """
    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="english_tutor_rec_"
        )
        os.close(fd)

    try:
        cmd = [
            "arecord",
            "-q",  # quiet mode
            "-f", "S16_LE",  # 16-bit signed little-endian PCM
            "-r", str(sample_rate),
            "-c", "1",  # mono
            "-d", str(duration),
            output_path,
        ]
        result = subprocess.run(cmd, timeout=duration + 5, capture_output=True)
        if result.returncode != 0:
            return None

        if os.path.getsize(output_path) < 1000:  # Too small = silence
            os.unlink(output_path)
            return None

        return output_path

    except Exception:
        return None


# ── Transcription ───────────────────────────────────────────────────

_whisper_model = None


def _get_model(model_size: str = "tiny") -> "WhisperModel":
    """Get or create the Whisper model singleton."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        # Use CPU by default, int8 for speed
        _whisper_model = WhisperModel(
            model_size_or_path=model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1,
        )
    return _whisper_model


def transcribe(
    audio_path: str,
    language: str = "en",
    model_size: str = "tiny",
) -> str:
    """Transcribe an audio file to text using faster-whisper.

    Args:
        audio_path: Path to the audio file (WAV/MP3/OGG).
        language: Language code ('en', 'zh', etc.). None for auto-detect.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium').

    Returns:
        Transcribed text, or empty string on failure.
    """
    try:
        model = _get_model(model_size)

        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=3,
            vad_filter=True,  # Filter out silence
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    except Exception:
        return ""


# ── High-level convenience ──────────────────────────────────────────

def record_and_transcribe(
    duration: int = DEFAULT_DURATION,
    language: str = "en",
    model_size: str = "tiny",
) -> str:
    """Record voice from microphone and transcribe it.

    Args:
        duration: Recording duration in seconds.
        language: Language for transcription.
        model_size: Whisper model size.

    Returns:
        Transcribed text, or empty string on failure/error.
    """
    if not is_recording_available():
        print("  [STT] arecord not available — cannot record audio.")
        return ""

    audio_path = record_audio(duration=duration)
    if not audio_path:
        print("  [STT] Recording failed or no audio detected.")
        return ""

    try:
        text = transcribe(audio_path, language=language, model_size=model_size)
        return text
    finally:
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except OSError:
            pass
