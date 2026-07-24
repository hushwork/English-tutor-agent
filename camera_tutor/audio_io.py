"""Audio I/O — stream capture, VAD, and interruptible playback.

## Architecture: How ChatGPT does it vs how we do it

ChatGPT's Advanced Voice Mode is "real-time interruptible" because:
  GPT-4o is an end-to-end multimodal model. Audio enters as continuous
  tokens, the model understands semantically that the user is
  interrupting, and it stops mid-generation. No separate VAD needed.

Qwen-Omni has the SAME architecture — Thinker-Talker, native audio
tokens in/out. It CAN do this too. Our current limitation is that
we call it in request-response mode, not streaming-duplex mode.

## Our three-level approach

  Level 1 [Current — VAD-based interrupt]:
    Play audio in chunks (PlaybackController), run VAD simultaneously.
    If VAD detects child voice → immediately stop playback + flush queue.
    Pros: Works now. Cons: VAD can false-trigger on coughs/noise.

  Level 2 [Short-term — Streaming response]:
    Stream Qwen-Omni's audio output token-by-token. Push each small
    audio chunk to speaker as it arrives. Can be interrupted mid-stream.
    Pros: Lower first-word latency. Cons: Still uses VAD for interrupt.

  Level 3 [Future — End-to-end duplex]:
    Qwen-Omni's audio input stream is continuously fed to the model.
    The model semantically detects interruption intent from audio
    context (not just VAD energy). Stops itself.
    Pros: ChatGPT-level experience. Requires Qwen-Omni API support
    for streaming-duplex mode (not yet in public API).
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import numpy as np


# ── Audio chunk ──────────────────────────────────────────────────


@dataclass
class AudioChunk:
    """A chunk of audio data."""
    data: np.ndarray          # Float32 audio samples [-1.0, 1.0]
    sample_rate: int          # e.g. 16000
    timestamp: float          # Unix timestamp
    is_speech: bool = False   # VAD result
    base64_wav: str = ""      # Cached WAV base64


# ── Playback state ───────────────────────────────────────────────


class PlaybackState(Enum):
    IDLE = "idle"
    PLAYING = "playing"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"


@dataclass
class PlaybackStatus:
    """Current status of audio playback, queryable from any thread."""
    state: PlaybackState = PlaybackState.IDLE
    total_chunks: int = 0
    played_chunks: int = 0
    started_at: float = 0.0
    interrupted_at: float = 0.0
    interrupted_by: str = ""  # "vad" | "explicit" | "timeout"


class PlaybackController:
    """Manages interruptible audio playback in a background thread.

    Usage:
        controller = PlaybackController(sample_rate=24000)
        controller.start()

        # Load audio to play
        controller.enqueue(audio_samples)

        # Main loop — check if child is interrupting
        while controller.is_playing:
            if audio_capture.is_speaking and controller.elapsed > 0.5:
                controller.interrupt(reason="vad")
                break
            time.sleep(0.05)

        controller.stop()
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_ms: int = 50,           # 50ms chunks = smooth playback
        interrupt_grace_ms: int = 500, # Minimum play before interrupt allowed
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.interrupt_grace_ms = interrupt_grace_ms
        self.device_index = device_index

        self._queue: deque[np.ndarray] = deque()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._interrupt_requested = False
        self._interrupt_reason = ""
        self._pyaudio = None
        self._stream = None
        self._bytes_per_sample = 2  # int16

        # Status (thread-safe via lock)
        self._lock = threading.Lock()
        self._status = PlaybackStatus()

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self):
        """Open audio output device and start background playback thread."""
        import pyaudio

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=self.chunk_samples,
        )

        self._running = True
        self._interrupt_requested = False
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop playback and release audio device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

    # ── Public API ───────────────────────────────────────────────

    def enqueue(self, audio: np.ndarray):
        """Queue audio samples for playback.

        Accepts float32 [-1,1] or int16. Converts to int16 for PyAudio.
        Call this as audio arrives from Qwen-Omni stream.
        """
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)

        # Break into chunk-sized pieces for responsive interruption
        for i in range(0, len(audio), self.chunk_samples):
            chunk = audio[i:i + self.chunk_samples]
            if len(chunk) > 0:
                with self._lock:
                    self._queue.append(chunk)
                    self._status.total_chunks += 1

    def enqueue_wav(self, wav_bytes: bytes):
        """Queue audio from WAV bytes (e.g., from edge-tts)."""
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.enqueue(audio)

    def interrupt(self, reason: str = "vad"):
        """Interrupt current playback immediately.

        Call this from the main loop when VAD detects child speech.
        Remaining queued audio is discarded.
        """
        self._interrupt_requested = True
        self._interrupt_reason = reason

    def clear(self):
        """Clear the playback queue without interrupting current chunk."""
        with self._lock:
            self._queue.clear()
            self._status = PlaybackStatus()

    # ── Status queries (thread-safe) ─────────────────────────────

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._status.state == PlaybackState.PLAYING

    @property
    def elapsed(self) -> float:
        """Seconds since playback started."""
        with self._lock:
            if self._status.started_at == 0:
                return 0.0
            return time.time() - self._status.started_at

    @property
    def progress(self) -> float:
        """Fraction of audio played (0.0 - 1.0)."""
        with self._lock:
            if self._status.total_chunks == 0:
                return 0.0
            return self._status.played_chunks / self._status.total_chunks

    @property
    def current_audio_position(self) -> float:
        """Current position in audio, in seconds (for lip-sync)."""
        with self._lock:
            return self._status.played_chunks * self.chunk_ms / 1000.0

    @property
    def was_interrupted(self) -> bool:
        with self._lock:
            return self._status.state == PlaybackState.INTERRUPTED

    def get_status(self) -> PlaybackStatus:
        with self._lock:
            return PlaybackStatus(
                state=self._status.state,
                total_chunks=self._status.total_chunks,
                played_chunks=self._status.played_chunks,
                started_at=self._status.started_at,
                interrupted_at=self._status.interrupted_at,
                interrupted_by=self._status.interrupted_by,
            )

    # ── Internal playback loop ────────────────────────────────────

    def _playback_loop(self):
        """Background thread: consume queue and play audio chunks.

        Checks for interrupt between every chunk (50ms granularity).
        """
        while self._running:
            # Get next chunk
            with self._lock:
                if len(self._queue) == 0:
                    if self._status.state == PlaybackState.PLAYING:
                        self._status.state = PlaybackState.FINISHED
                    self._status = PlaybackStatus()  # Reset
                    time.sleep(0.01)
                    continue

                chunk = self._queue.popleft()

                if self._status.state == PlaybackState.IDLE:
                    self._status.state = PlaybackState.PLAYING
                    self._status.started_at = time.time()

            # Check for interrupt
            if self._interrupt_requested:
                elapsed_ms = self._status.played_chunks * self.chunk_ms
                if elapsed_ms >= self.interrupt_grace_ms:
                    with self._lock:
                        self._status.state = PlaybackState.INTERRUPTED
                        self._status.interrupted_at = time.time()
                        self._status.interrupted_by = self._interrupt_reason
                        self._queue.clear()  # Discard remaining
                    self._interrupt_requested = False
                    continue

            # Play this chunk
            try:
                self._stream.write(chunk.tobytes())
            except Exception:
                pass  # Device may have been closed

            with self._lock:
                self._status.played_chunks += 1


# ── Audio Capture ────────────────────────────────────────────────


class AudioCapture:
    """Real-time audio capture with VAD.

    Uses PyAudio for capture and Silero VAD for speech detection.
    VAD is ALWAYS running — even during playback — so child can
    interrupt Emma at any time without pressing any button.

    Architecture:
        Microphone → PyAudio stream → 50ms chunks → Silero VAD
                                                      ↓
                                            is_speaking (bool)
                                                      ↓
                            Main loop checks this → if True + Emma playing
                            → PlaybackController.interrupt()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.05,  # 50ms — tight for responsive VAD
        device_index: Optional[int] = None,
        vad_threshold: float = 0.3,    # Lower for children's voices
        speech_buffer_duration: float = 5.0,
        silence_duration: float = 0.8,  # Shorter — child pauses are brief
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device_index = device_index
        self.vad_threshold = vad_threshold
        self.speech_buffer_duration = speech_buffer_duration
        self.silence_duration = silence_duration

        self._audio = None
        self._stream = None
        self._vad_model = None
        self._running = False

        # Speech detection state
        self._speech_buffer: deque[AudioChunk] = deque(
            maxlen=int(speech_buffer_duration / chunk_duration)
        )
        self._is_speaking = False
        self._silence_count = 0

        # Interruption guard: ignore very short VAD triggers
        self._min_speech_chunks = 2  # 100ms of speech before calling it speech
        self._speech_chunk_count = 0

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self):
        """Open microphone and initialize VAD."""
        import pyaudio

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=None,
        )
        self._load_vad()
        self._running = True

    def stop(self):
        """Close microphone."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio:
            self._audio.terminate()
            self._audio = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _load_vad(self):
        """Load Silero VAD model."""
        try:
            import torch
            model, utils = torch.hub.load(
                'snakers4/silero-vad',
                'silero_vad',
                force_reload=False,
                onnx=False,
            )
            self._vad_model = model
            self._vad_utils = utils
        except Exception as e:
            print(f"[WARN] Failed to load Silero VAD: {e}. VAD disabled.")
            self._vad_model = None

    # ── Audio capture ────────────────────────────────────────────

    def read_chunk(self) -> Optional[AudioChunk]:
        """Read one audio chunk from the microphone (non-blocking).

        Returns None if stream is not running.
        Call this at ~20Hz (50ms chunks) in your main loop.
        """
        if not self._running or self._stream is None:
            return None

        try:
            data = self._stream.read(self.chunk_size, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)

            chunk = AudioChunk(
                data=audio,
                sample_rate=self.sample_rate,
                timestamp=time.time(),
            )

            if self._vad_model is not None:
                chunk.is_speech = self._detect_speech(audio)

            self._update_speech_state(chunk)
            return chunk

        except Exception as e:
            print(f"[WARN] Audio read error: {e}")
            return None

    def read_speech_segment(self, timeout: float = 8.0) -> Optional[list[AudioChunk]]:
        """Block until a complete speech segment is captured.

        Returns list of audio chunks, or None if timeout.
        Use for capturing a full child utterance for Qwen-Omni.
        """
        segment: list[AudioChunk] = []
        silence_after_speech = 0
        started = False
        t0 = time.time()

        while time.time() - t0 < timeout:
            chunk = self.read_chunk()
            if chunk is None:
                continue

            if chunk.is_speech:
                started = True
                silence_after_speech = 0
                segment.append(chunk)
            elif started:
                silence_after_speech += self.chunk_duration
                segment.append(chunk)
                if silence_after_speech >= self.silence_duration:
                    break

        return segment if started and len(segment) > 0 else None

    # ── VAD ──────────────────────────────────────────────────────

    def _detect_speech(self, audio: np.ndarray) -> bool:
        """Run VAD on an audio chunk."""
        if self._vad_model is None:
            return True

        import torch
        tensor = torch.from_numpy(audio).float()
        if self.sample_rate != 16000:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                scale_factor=16000 / self.sample_rate,
                mode='linear',
            ).squeeze()

        try:
            prob = self._vad_model(tensor, 16000).item()
            return prob > self.vad_threshold
        except Exception:
            return True

    def _update_speech_state(self, chunk: AudioChunk):
        """Update speech detection with guard against false triggers."""
        self._speech_buffer.append(chunk)

        if chunk.is_speech:
            self._speech_chunk_count += 1
            if self._speech_chunk_count >= self._min_speech_chunks:
                self._is_speaking = True
            self._silence_count = 0
        elif self._is_speaking:
            self._speech_chunk_count = 0
            self._silence_count += 1
            silence_sec = self._silence_count * self.chunk_duration
            if silence_sec >= self.silence_duration:
                self._is_speaking = False
                self._silence_count = 0

    @property
    def is_speaking(self) -> bool:
        """Is the child currently vocalizing? Check this every tick.

        When True + Emma is playing → call PlaybackController.interrupt()
        """
        return self._is_speaking

    @property
    def speech_confidence(self) -> float:
        """Average VAD confidence over recent speech chunks (0-1).

        Higher = more likely actual speech (not cough/noise).
        Use to gate interruption: only interrupt if confidence > 0.6.
        """
        recent = [c for c in list(self._speech_buffer)[-5:] if c.is_speech]
        if not recent:
            return 0.0
        return min(1.0, len(recent) / 3)  # Simple heuristic

    # ── Encoding ─────────────────────────────────────────────────

    @staticmethod
    def to_wav_base64(chunks: list[AudioChunk]) -> str:
        """Encode audio chunks to base64 WAV for API transmission."""
        if not chunks:
            return ""

        full_audio = np.concatenate([c.data for c in chunks])
        sample_rate = chunks[0].sample_rate

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int16_audio = (full_audio * 32767).astype(np.int16)
            wf.writeframes(int16_audio.tobytes())

        return base64.b64encode(buffer.getvalue()).decode('ascii')


# ── Main loop helper ────────────────────────────────────────────


async def interactive_loop(
    capture: AudioCapture,
    playback: PlaybackController,
    on_child_speech: Callable,
    on_emma_response: Callable,
):
    """Reference implementation of the interruptible conversation loop.

    This is how the main loop drives the responsive interaction:
    - VAD always running
    - Emma plays → VAD checks simultaneously
    - Child speaks → playback interrupted immediately
    - No record button needed

    Args:
        capture: AudioCapture instance (microphone + VAD)
        playback: PlaybackController instance (speaker output)
        on_child_speech: Called when child speech segment is captured
        on_emma_response: Called to generate Emma's response (text)
    """
    while True:
        chunk = capture.read_chunk()
        if chunk is None:
            await asyncio.sleep(0.05)
            continue

        # ── Emma is playing ──
        if playback.is_playing:
            # Check for interruption
            if capture.is_speaking and capture.speech_confidence > 0.5:
                elapsed = playback.elapsed
                if elapsed > 0.3:  # Grace period: don't interrupt first 300ms
                    playback.interrupt(reason="vad")
                    print(f"[INTERRUPT] Emma stopped after {elapsed:.2f}s — child spoke")

            await asyncio.sleep(0.05)
            continue

        # ── Emma is silent, child started speaking ──
        if capture.is_speaking and not playback.is_playing:
            segment = capture.read_speech_segment(timeout=8.0)
            if segment:
                # Got a complete utterance → process
                response_text = await on_child_speech(segment)

                if response_text:
                    # Generate Emma's audio (Qwen-Omni or Kokoro)
                    audio = await on_emma_response(response_text)
                    if audio is not None:
                        playback.enqueue(audio)

        await asyncio.sleep(0.05)
