"""Projection — desktop AR overlay and gesture tracking for Pro configuration.

Manages the pico projector output and hand gesture interaction:
1. Projection calibration (camera-to-projector coordinate mapping)
2. Desktop overlay rendering (Emma face, word cards, game elements)
3. Hand gesture tracking via MediaPipe Hands
4. Touch/hit detection (finger tip → projected element interaction)

For MVP, uses the existing top-down camera (no additional depth sensor).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Projected elements ──────────────────────────────────────────


class ElementType(Enum):
    EMMA_FACE = "emma_face"
    WORD_CARD = "word_card"
    GAME_OPTION = "game_option"
    HIGHLIGHT_RING = "highlight_ring"
    PARTICLE = "particle"
    PROGRESS_BAR = "progress_bar"


@dataclass
class ProjectedElement:
    """A UI element projected onto the desktop."""
    id: str
    type: ElementType
    x: float          # Center X in projection space (0-1, normalized)
    y: float          # Center Y in projection space (0-1)
    width: float      # Width in projection space (0-1)
    height: float     # Height in projection space (0-1)
    label: str = ""   # Text label (for word cards)
    image_id: str = ""  # Image/emoji identifier
    highlighted: bool = False
    hit: bool = False  # Currently being touched/pointed at
    opacity: float = 1.0
    animation: str = ""  # Animation state name
    created_at: float = field(default_factory=time.time)


@dataclass
class GestureEvent:
    """A detected hand gesture event."""
    type: str           # "point", "hold", "wave", "spread", "peace"
    x: float            # Finger tip position in projection space (0-1)
    y: float
    duration: float     # How long the gesture has been held
    finger_count: int   # Number of visible fingers
    confidence: float   # Detection confidence (0-1)


class ProjectionEngine:
    """Manages desktop AR projection and gesture interaction.

    Responsibilities:
    - Projection calibration (one-time setup)
    - Desktop overlay layout (face position, game elements)
    - Gesture-to-element hit testing
    - Animation state management for projected elements
    """

    def __init__(
        self,
        projection_width: int = 1920,
        projection_height: int = 1080,
        camera_to_projector_offset: tuple[float, float] = (0.0, 0.0),
        hit_threshold: float = 0.08,  # Distance threshold for "touch"
        hold_duration: float = 1.0,   # Seconds to hold for "confirm"
    ):
        self.projection_width = projection_width
        self.projection_height = projection_height
        self.camera_offset = camera_to_projector_offset
        self.hit_threshold = hit_threshold
        self.hold_duration = hold_duration

        self.elements: dict[str, ProjectedElement] = {}
        self._gesture_history: deque[GestureEvent] = deque(maxlen=30)
        self._last_gesture: Optional[GestureEvent] = None

        # MediaPipe Hands (lazy init)
        self._hands = None
        self._mp_drawing = None

    # ── Projection layout ───────────────────────────────────────

    def layout_emma_face(self) -> ProjectedElement:
        """Place Emma's face at the top-center of the projection area."""
        element = ProjectedElement(
            id="emma_face",
            type=ElementType.EMMA_FACE,
            x=0.5, y=0.25,   # Top center
            width=0.25, height=0.25,
        )
        self.elements["emma_face"] = element
        return element

    def layout_word_cards(
        self, words: list[str], images: list[str] | None = None
    ) -> list[ProjectedElement]:
        """Layout word cards in a horizontal row for matching games.

        Args:
            words: List of words to display
            images: Optional emoji/image identifiers matching words
        """
        n = len(words)
        if n == 0:
            return []

        # Clear old word cards
        for key in list(self.elements.keys()):
            if key.startswith("word_"):
                del self.elements[key]

        cards = []
        spacing = 0.8 / max(n, 1)  # 80% of width divided by N
        start_x = 0.5 - (spacing * (n - 1)) / 2

        for i, word in enumerate(words):
            card = ProjectedElement(
                id=f"word_{i}",
                type=ElementType.WORD_CARD,
                x=start_x + spacing * i,
                y=0.65,   # Lower portion of projection
                width=0.18,
                height=0.12,
                label=word,
                image_id=images[i] if images and i < len(images) else "",
            )
            self.elements[card.id] = card
            cards.append(card)

        return cards

    def layout_progress_bar(self, progress: float) -> ProjectedElement:
        """Show a progress bar (e.g., stars collected)."""
        element = ProjectedElement(
            id="progress_bar",
            type=ElementType.PROGRESS_BAR,
            x=0.5, y=0.9,
            width=min(progress, 1.0) * 0.6,
            height=0.03,
            opacity=min(1.0, progress),
        )
        self.elements["progress_bar"] = element
        return element

    def clear_elements(self, prefix: str = ""):
        """Remove elements matching prefix."""
        if prefix:
            for key in list(self.elements.keys()):
                if key.startswith(prefix):
                    del self.elements[key]
        else:
            self.elements.clear()

    # ── Gesture tracking ────────────────────────────────────────

    def init_hands(self):
        """Initialize MediaPipe Hands (call once before tracking)."""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,       # Child typically uses one hand
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        except ImportError:
            print("[WARN] MediaPipe not installed. Gesture tracking disabled.")
            print("       Install: pip install mediapipe")
            return False
        except Exception as e:
            print(f"[WARN] Failed to init MediaPipe Hands: {e}")
            return False

    def process_frame(self, frame):
        """Process a camera frame for hand landmarks.

        Args:
            frame: BGR image (numpy array) from camera

        Returns:
            GestureEvent or None
        """
        if self._hands is None:
            return None

        import numpy as np

        # Convert BGR to RGB
        rgb = frame[:, :, ::-1] if frame.shape[-1] == 3 else frame

        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            self._last_gesture = None
            return None

        # Take first hand
        landmarks = results.multi_hand_landmarks[0]

        # Extract key points
        index_tip = landmarks.landmark[8]  # Index finger tip
        thumb_tip = landmarks.landmark[4]  # Thumb tip
        wrist = landmarks.landmark[0]

        # Map camera coordinates to projection space
        px = self._camera_to_projection(index_tip.x, index_tip.y)
        py = self._camera_to_projection(index_tip.x, index_tip.y)[1]

        # Detect gesture type
        gesture_type = self._classify_gesture(landmarks.landmark)

        # Calculate duration
        now = time.time()
        if self._last_gesture and self._last_gesture.type == gesture_type:
            duration = now - (self._last_gesture.duration or now)
        else:
            duration = 0.0

        event = GestureEvent(
            type=gesture_type,
            x=px,
            y=py,
            duration=duration,
            finger_count=self._count_extended_fingers(landmarks.landmark),
            confidence=0.9,  # MediaPipe confidence
        )

        self._gesture_history.append(event)
        self._last_gesture = event
        return event

    def release(self):
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._hands = None

    # ── Hit testing ─────────────────────────────────────────────

    def hit_test(self, gesture: GestureEvent) -> Optional[ProjectedElement]:
        """Check if a gesture hits any projected element.

        Returns the hit element, or None if no hit.
        """
        for element in self.elements.values():
            if self._point_in_rect(
                gesture.x, gesture.y,
                element.x, element.y, element.width, element.height,
            ):
                return element
        return None

    def check_hold_confirm(self, element: ProjectedElement) -> bool:
        """Check if an element has been held long enough to confirm."""
        if not self._last_gesture:
            return False
        if self._last_gesture.type != "point":
            return False
        return self._last_gesture.duration >= self.hold_duration

    # ── Render output ───────────────────────────────────────────

    def render_overlay(self, output_path: str | None = None) -> str:
        """Generate the projection overlay as HTML/SVG for display.

        In production, this renders directly to the projector display
        via PyGame/OpenGL. For MVP, generates an HTML page that can
        be displayed in a fullscreen browser on the projector output.

        Args:
            output_path: Optional path to save the HTML file

        Returns:
            HTML string for the projection overlay
        """
        # Collect active elements as positioned HTML divs
        element_html_parts = []
        for element in self.elements.values():
            if element.opacity <= 0.01:
                continue

            left = (element.x - element.width / 2) * 100
            top = (element.y - element.height / 2) * 100
            width = element.width * 100
            height = element.height * 100

            highlight = '3px solid #4ecf8d' if element.highlighted else 'none'
            hit_glow = '0 0 20px #ffdd00' if element.hit else 'none'

            if element.type == ElementType.WORD_CARD:
                html = (
                    f'<div style="position:absolute;left:{left}%;top:{top}%;'
                    f'width:{width}%;height:{height}%;'
                    f'border-radius:12px;background:rgba(255,255,255,0.9);'
                    f'border:{highlight};box-shadow:{hit_glow};'
                    f'display:flex;flex-direction:column;align-items:center;'
                    f'justify-content:center;font-family:sans-serif;'
                    f'font-size:2.5vw;color:#333;font-weight:bold;'
                    f'opacity:{element.opacity};transition:all 0.2s;">'
                    f'{element.image_id}<br/>{element.label}</div>'
                )
                element_html_parts.append(html)

            elif element.type == ElementType.EMMA_FACE:
                # Emma face is rendered separately by avatar.py
                # Placeholder for the avatar renderer
                html = (
                    f'<div id="emma-face" style="position:absolute;'
                    f'left:{left}%;top:{top}%;'
                    f'width:{width}%;height:{height}%;'
                    f'border-radius:50%;opacity:{element.opacity};'
                    f'box-shadow:{hit_glow};'
                    f'display:flex;align-items:center;justify-content:center;">'
                    f'<!-- Emma SVG avatar inserted here --></div>'
                )
                element_html_parts.append(html)

            elif element.type == ElementType.PROGRESS_BAR:
                html = (
                    f'<div style="position:absolute;left:{left}%;top:{top}%;'
                    f'width:{width}%;height:{height}%;'
                    f'border-radius:4px;background:linear-gradient(90deg,#4ecf8d,#6c8cff);'
                    f'opacity:{element.opacity};transition:width 0.5s;"></div>'
                )
                element_html_parts.append(html)

        return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; width:100vw; height:100vh; overflow:hidden; }}
  #overlay {{ position:relative; width:100%; height:100%; }}
</style></head>
<body><div id="overlay">{"".join(element_html_parts)}</div></body></html>'''

    # ── Internal helpers ─────────────────────────────────────────

    def _camera_to_projection(self, cam_x: float, cam_y: float) -> tuple[float, float]:
        """Map camera coordinates (0-1) to projection coordinates (0-1).

        Accounts for camera-projector offset and perspective.
        """
        # Simple linear mapping with offset
        proj_x = cam_x + self.camera_offset[0]
        proj_y = cam_y + self.camera_offset[1]
        return (
            max(0.0, min(1.0, proj_x)),
            max(0.0, min(1.0, proj_y)),
        )

    @staticmethod
    def _point_in_rect(
        px: float, py: float,
        rx: float, ry: float, rw: float, rh: float,
    ) -> bool:
        """Check if point (px,py) is inside rectangle centered at (rx,ry)."""
        half_w = rw / 2
        half_h = rh / 2
        return (rx - half_w <= px <= rx + half_w and
                ry - half_h <= py <= ry + half_h)

    def _classify_gesture(self, landmarks) -> str:
        """Classify hand gesture from 21 MediaPipe landmarks.

        Simplified classification for child interaction:
        - "point": index finger extended, others curled
        - "spread": all fingers extended (show me)
        - "wave": horizontal hand movement (detected across frames)
        - "peace": index + middle extended, others curled
        - "unknown": can't classify
        """
        # Check which fingers are extended
        extended = self._count_extended_fingers(landmarks)

        if extended == 1:
            return "point"
        elif extended >= 4:
            return "spread"
        elif extended == 2:
            # Could be peace or point-thumb
            # Check if index(8) and middle(12) are up
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            if index_up and middle_up:
                return "peace"
            return "point"
        else:
            return "unknown"

    @staticmethod
    def _count_extended_fingers(landmarks) -> int:
        """Count how many fingers are extended (tip above PIP joint)."""
        # Fingertip indices: thumb(4), index(8), middle(12), ring(16), pinky(20)
        # PIP joint indices: thumb(3), index(6), middle(10), ring(14), pinky(18)
        finger_pairs = [
            (4, 3),   # Thumb
            (8, 6),   # Index
            (12, 10), # Middle
            (16, 14), # Ring
            (20, 18), # Pinky
        ]

        count = 0
        for tip_idx, pip_idx in finger_pairs:
            # Finger is extended if tip is above PIP (smaller Y)
            if landmarks[tip_idx].y < landmarks[pip_idx].y:
                count += 1
        return count

    # ── Calibration ─────────────────────────────────────────────

    @staticmethod
    def calibrate_offset(camera_frame, projection_landmarks) -> tuple[float, float]:
        """Calculate camera-to-projector offset from calibration points.

        Place known markers in camera view and their corresponding
        positions in projection space. Returns (offset_x, offset_y).

        Simplified for MVP: assume centered alignment, manual offset.
        """
        # In production: use homography matrix from 4+ calibration points
        # For MVP: return zero offset (assumes camera and projector aligned)
        return (0.0, 0.0)
