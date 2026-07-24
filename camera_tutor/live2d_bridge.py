"""Live2D Bridge — Python ↔ Cubism Native SDK integration.

Architecture:
  avatar.py (viseme calculation)
       │
       ▼
  live2d_bridge.py (this file) — parameter translation
       │
       ▼
  Cubism Native SDK (C++) — model rendering
       │
       ▼
  Display output (LCD / projector)

Setup (one-time):
  1. git clone https://github.com/Live2D/CubismNativeSamples.git ~/Live2D/
  2. cd ~/Live2D && ./build_linux.sh  (see build_live2d_linux.sh)
  3. Place .moc3 model file in camera_tutor/models/

Usage:
  from camera_tutor.live2d_bridge import Live2DBackend

  backend = Live2DBackend(model_path="models/Haru.moc3")
  backend.start()

  # Each frame (60fps):
  params = avatar.get_viseme_params()  # from avatar.py
  backend.set_parameters(params)
  frame = backend.render()
  # display frame on LCD/projector

License: Cubism SDK for Native is FREE for development and for
  commercial use by individuals/small enterprises (revenue < threshold).
  See: https://www.live2d.com/en/download/cubism-sdk/
"""

from __future__ import annotations

import json
import os
import subprocess
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Viseme → Live2D parameter mapping ────────────────────────────


@dataclass
class VisemeParams:
    """Parameters for a single frame of Live2D animation.

    These are computed by avatar.py from phoneme alignment data,
    then fed to Live2D's CubismModel.SetParameterValue() each frame.
    """
    mouth_open: float = 0.0       # ParamMouthOpenY  (0=closed, 1=wide open)
    mouth_width: float = 0.0      # Custom: mouth width (0=narrow, 1=wide smile)
    mouth_form: float = 0.0       # ParamMouthForm (0=pursed, 1=spread)
    tongue_visible: float = 0.0   # Custom: tongue visibility (0=hidden, 1=fully out)
    lip_bite: float = 0.0        # Custom: lower lip on upper teeth (/f/ /v/)
    eye_open: float = 1.0        # ParamEyeLOpen / ParamEyeROpen
    brow_height: float = 0.0     # ParamBrowLY / ParamBrowRY
    emotion: str = "neutral"     # "happy", "surprised", "thinking", "neutral"

    @classmethod
    def from_viseme(cls, viseme, emotion: str = "neutral") -> "VisemeParams":
        """Convert our Viseme enum to Live2D-compatible parameters.

        This is the key translation layer — maps our 10 viseme types
        to Live2D's parameter space (~30+ float parameters on a typical model).

        Note: Exact parameter names depend on the specific Live2D model.
        The values below target the standard Haru sample model parameters.
        Adapt for your Emma model.
        """
        from camera_tutor.avatar import Viseme

        from camera_tutor.avatar import Viseme
        params = {
            # V00 — silence / neutral
            Viseme.V00_SIL:   cls(mouth_open=0.05, mouth_width=0.3, mouth_form=0.5),
            # V01 — /æ/ cat, /ʌ/ cup — wide open
            Viseme.V01_AE_AH: cls(mouth_open=0.85, mouth_width=0.65, mouth_form=0.55),
            # V02 — /ɑ/ car — widest open (NOT in Chinese) ☆☆☆
            Viseme.V02_AA:    cls(mouth_open=0.95, mouth_width=0.55, mouth_form=0.5),
            # V03 — /ɔ/ dog — medium open, lips rounded
            Viseme.V03_AO:    cls(mouth_open=0.55, mouth_width=0.25, mouth_form=0.3),
            # V04 — /ɛ/ bed — neutral-moderate open
            Viseme.V04_EH_EY: cls(mouth_open=0.35, mouth_width=0.45, mouth_form=0.5),
            # V05 — /ɝ/ bird — unique: tongue curled + lips rounded
            Viseme.V05_ER:    cls(mouth_open=0.3,  mouth_width=0.3, mouth_form=0.35),
            # V06 — /i/ bee, /ɪ/ ship — wide smile
            Viseme.V06_IY_IH: cls(mouth_open=0.12, mouth_width=0.9, mouth_form=0.9),
            # V07 — /u/ blue, /w/ wet — rounded + pursed
            Viseme.V07_UW_W:  cls(mouth_open=0.15, mouth_width=0.15, mouth_form=0.15),
            # V08 — /oʊ/ boat — medium round
            Viseme.V08_OW:    cls(mouth_open=0.35, mouth_width=0.2, mouth_form=0.2),
            # V09 — /aʊ/ cow — wide → round (start of diphthong)
            Viseme.V09_AW:    cls(mouth_open=0.8,  mouth_width=0.4, mouth_form=0.4),
            # V10 — /ɔɪ/ boy — round → smile
            Viseme.V10_OY:    cls(mouth_open=0.4,  mouth_width=0.25, mouth_form=0.3),
            # V11 — /aɪ/ eye — wide → smile
            Viseme.V11_AY:    cls(mouth_open=0.75, mouth_width=0.5, mouth_form=0.45),
            # V12 — /h/ hot — open, breath
            Viseme.V12_H:     cls(mouth_open=0.5,  mouth_width=0.4, mouth_form=0.5),
            # V13 — /ɹ/ red — rounded lips (very different from Chinese r)
            Viseme.V13_R:     cls(mouth_open=0.1,  mouth_width=0.15, mouth_form=0.12),
            # V14 — /l/ like — tongue tip on ridge
            Viseme.V14_L:     cls(mouth_open=0.15, mouth_width=0.35, mouth_form=0.5, tongue_visible=0.5),
            # V15 — /s/ see, /z/ zoo — teeth together
            Viseme.V15_S_Z:   cls(mouth_open=0.05, mouth_width=0.5, mouth_form=0.5, tongue_visible=0.3),
            # V16 — /ʃ/ she, /tʃ/ chip — lips pursed forward
            Viseme.V16_SH_ZH: cls(mouth_open=0.08, mouth_width=0.2, mouth_form=0.1),
            # V17 — /θ/ think, /ð/ this — TONGUE VISIBLE ☆☆☆☆☆
            Viseme.V17_TH_DH: cls(mouth_open=0.1,  mouth_width=0.35, mouth_form=0.45, tongue_visible=1.0),
            # V18 — /f/ five, /v/ very — LIP BITE ☆☆☆☆☆
            Viseme.V18_F_V:   cls(mouth_open=0.03, mouth_width=0.35, mouth_form=0.4, lip_bite=0.95),
            # V19 — /t/ top, /d/ dog, /n/ no — tongue on ridge
            Viseme.V19_T_D_N: cls(mouth_open=0.08, mouth_width=0.3, mouth_form=0.5, tongue_visible=0.3),
            # V20 — /k/ cat, /g/ go — back of mouth
            Viseme.V20_K_G_NG:cls(mouth_open=0.2,  mouth_width=0.35, mouth_form=0.5),
            # V21 — /p/ pop, /b/ big, /m/ mom — LIPS CLOSED
            Viseme.V21_P_B_M: cls(mouth_open=0.0,  mouth_width=0.3, mouth_form=0.5),
        }

        base = params.get(viseme, params[Viseme.V00_SIL])

        # Blend emotion modifiers
        emotion_mods = {
            "happy":        {"mouth_width": 0.1, "brow_height": 0.2},
            "surprised":    {"mouth_open": 0.1, "brow_height": 0.5},
            "thinking":     {"mouth_width": -0.05, "brow_height": -0.1},
            "encouraging":  {"mouth_width": 0.05, "brow_height": 0.1},
            "curious":      {"brow_height": 0.3},
        }
        mod = emotion_mods.get(emotion, {})

        return cls(
            mouth_open=base.mouth_open + mod.get("mouth_open", 0),
            mouth_width=base.mouth_width + mod.get("mouth_width", 0),
            mouth_form=base.mouth_form + mod.get("mouth_form", 0),
            tongue_visible=base.tongue_visible + mod.get("tongue_visible", 0),
            lip_bite=base.lip_bite + mod.get("lip_bite", 0),
            eye_open=1.0,
            brow_height=base.brow_height + mod.get("brow_height", 0),
            emotion=emotion,
        )


# ── Live2D C++ process bridge ────────────────────────────────────


class Live2DBackend:
    """Manages a Live2D Cubism Native rendering process.

    Communication protocol (binary, over pipe):
      Python → C++:  4-byte param_count (uint32) + N × (name_len, name, value)
      C++ → Python:  4-byte frame_size (uint32) + frame_size bytes (RGBA pixels)

    For MVP, we use a subprocess approach: a small C++ program compiled
    from the Cubism Native SDK samples, modified to accept parameter
    input via stdin and output rendered frames via stdout.

    For production, replace with ctypes/Cython direct DLL calls.
    """

    # Standard Live2D parameter names (Haru model)
    PARAM_MOUTH_OPEN_Y = "ParamMouthOpenY"
    PARAM_MOUTH_FORM = "ParamMouthForm"
    PARAM_EYE_L_OPEN = "ParamEyeLOpen"
    PARAM_EYE_R_OPEN = "ParamEyeROpen"
    PARAM_BROW_L_Y = "ParamBrowLY"
    PARAM_BROW_R_Y = "ParamBrowRY"
    PARAM_ANGLE_X = "ParamAngleX"
    PARAM_ANGLE_Y = "ParamAngleY"
    PARAM_ANGLE_Z = "ParamAngleZ"
    PARAM_BODY_ANGLE_X = "ParamBodyAngleX"

    # Custom parameters (added to our Emma model)
    PARAM_MOUTH_WIDTH = "ParamMouthWidth"
    PARAM_TONGUE_VISIBLE = "ParamTongueVisible"
    PARAM_LIP_BITE = "ParamLipBite"

    def __init__(
        self,
        model_path: str | None = None,
        executable: str | None = None,
        width: int = 512,
        height: int = 512,
    ):
        self.model_path = model_path or os.environ.get(
            "LIVE2D_MODEL_PATH",
            str(Path(__file__).resolve().parent / "models" / "Haru.moc3"),
        )
        self.executable = executable or os.environ.get(
            "LIVE2D_RENDERER",
            str(Path(__file__).resolve().parent.parent / "build" / "live2d_renderer"),
        )
        self.width = width
        self.height = height
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._last_frame: Optional[bytes] = None
        self._lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self):
        """Launch the Live2D C++ rendering process."""
        if not Path(self.executable).exists():
            raise FileNotFoundError(
                f"Live2D renderer not found at {self.executable}. "
                f"Run build_live2d_linux.sh first, or set LIVE2D_RENDERER env var."
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Live2D model not found at {self.model_path}. "
                f"Place a .moc3 file there, or set LIVE2D_MODEL_PATH env var."
            )

        self._process = subprocess.Popen(
            [self.executable, "--model", self.model_path,
             "--width", str(self.width), "--height", str(self.height),
             "--pipe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._running = True
        threading.Thread(target=self._read_frames, daemon=True).start()

    def stop(self):
        """Stop the rendering process."""
        self._running = False
        if self._process:
            self._process.stdin.close()
            self._process.terminate()
            self._process.wait(timeout=2.0)
            self._process = None

    # ── Per-frame parameter setting ──────────────────────────────

    def set_parameters(self, params: VisemeParams):
        """Send viseme parameters to Live2D for the current frame.

        Called at 60fps from the main loop.
        """
        if not self._running or self._process is None:
            return

        param_list = [
            (self.PARAM_MOUTH_OPEN_Y, params.mouth_open),
            (self.PARAM_MOUTH_FORM,  params.mouth_form),
            (self.PARAM_EYE_L_OPEN,  params.eye_open),
            (self.PARAM_EYE_R_OPEN,  params.eye_open),
            (self.PARAM_BROW_L_Y,    params.brow_height),
            (self.PARAM_BROW_R_Y,    params.brow_height),
            (self.PARAM_MOUTH_WIDTH, params.mouth_width),
            (self.PARAM_TONGUE_VISIBLE, params.tongue_visible),
            (self.PARAM_LIP_BITE,    params.lip_bite),
        ]

        # Binary protocol: param_count + (name_len, name_bytes, value_float)
        data = struct.pack("I", len(param_list))
        for name, value in param_list:
            name_bytes = name.encode('utf-8')
            data += struct.pack("I", len(name_bytes))
            data += name_bytes
            data += struct.pack("f", value)

        try:
            self._process.stdin.write(data)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._running = False

    def get_frame(self) -> Optional[bytes]:
        """Get the latest rendered frame (RGBA bytes).

        Returns None if no frame available yet.
        """
        with self._lock:
            return self._last_frame

    def render_and_get(self, params: VisemeParams) -> Optional[bytes]:
        """Convenience: set params + wait for frame.

        Use only if latency is acceptable (adds ~16ms).
        For 60fps rendering, call set_parameters() + get_frame() separately.
        """
        self.set_parameters(params)
        time.sleep(0.016)  # Wait one frame
        return self.get_frame()

    # ── Internal ─────────────────────────────────────────────────

    def _read_frames(self):
        """Background thread: read rendered frames from C++ process stdout."""
        while self._running and self._process:
            try:
                # Read frame size (uint32)
                size_bytes = self._process.stdout.read(4)
                if len(size_bytes) < 4:
                    break
                size = struct.unpack("I", size_bytes)[0]

                # Read frame data (RGBA pixels)
                frame_data = self._process.stdout.read(size)
                if len(frame_data) < size:
                    break

                with self._lock:
                    self._last_frame = frame_data

            except (BrokenPipeError, OSError, struct.error):
                break

        self._running = False


# ── Fallback: SVG renderer (when Live2D not available) ───────────


class SVGBackend:
    """SVG-based rendering fallback when Live2D is not available.

    Used automatically by get_backend() when Live2D binary is not found.
    Shares the same VisemeParams interface — swap backends without
    changing the viseme calculation code.
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self._current_svg = ""
        self._avatar = None  # EmmaAvatar instance (set by factory)

    def set_parameters(self, params: VisemeParams):
        """No-op — SVG renders differently (via EmmaAvatar.render_svg)."""
        pass

    def get_frame(self) -> Optional[str]:
        """Return the current SVG string."""
        return self._current_svg

    def render_svg(self, svg_text: str):
        """Store SVG rendered by EmmaAvatar."""
        self._current_svg = svg_text


# ── Backend factory ──────────────────────────────────────────────


def get_backend(
    model_path: str | None = None,
    preferred: str = "auto",
) -> Live2DBackend | SVGBackend:
    """Factory: return the best available rendering backend.

    Args:
        model_path: Path to .moc3 model file (for Live2D)
        preferred: "live2d", "svg", or "auto" (try Live2D first)
    """
    if preferred == "svg":
        return SVGBackend()

    if preferred in ("live2d", "auto"):
        try:
            backend = Live2DBackend(model_path=model_path)
            backend.start()
            return backend
        except FileNotFoundError:
            if preferred == "live2d":
                raise
            print("[INFO] Live2D not found, using SVG fallback")

    return SVGBackend()
