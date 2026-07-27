"""Face detection and recognition utilities for English Tutor.

Uses OpenCV for face detection (built-in, no extra deps).
Optionally uses face_recognition library for embedding-based matching.
Gracefully degrades if libraries are not installed.

All functions return safe defaults (None/False) when dependencies are missing.
"""

from __future__ import annotations

import base64

# ── Lazy imports ──────────────────────────────────────────────────

_np = None
_cv2 = None
_face_recognition = None
_deps_checked = False
_cv2_available = False
_np_available = False


def _check_deps():
    global _np, _cv2, _face_recognition, _deps_checked
    global _cv2_available, _np_available
    if _deps_checked:
        return
    _deps_checked = True
    try:
        import numpy as _np_mod
        _np = _np_mod
        _np_available = True
    except ImportError:
        _np_available = False
    try:
        import cv2 as _cv2_mod
        _cv2 = _cv2_mod
        _cv2_available = True
    except ImportError:
        _cv2_available = False
    try:
        import face_recognition as _fr
        _face_recognition = _fr
    except ImportError:
        _face_recognition = None


# ── Face detection ────────────────────────────────────────────────

def detect_face(image) -> object | None:
    """Detect a single face in an image and return the cropped face region.

    Returns None if OpenCV not available or no face detected.
    """
    _check_deps()
    if not _cv2_available or not _np_available:
        return None

    cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
    face_cascade = _cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    if len(faces) > 1:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)

    x, y, w, h = faces[0]
    margin = int(w * 0.2)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    return image[y:y + h, x:x + w]


def is_camera_available() -> bool:
    """Check if a camera is connected and accessible."""
    _check_deps()
    if not _cv2_available:
        return False
    try:
        cap = _cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception:
        return False


# ── Face recognition ─────────────────────────────────────────────

def is_face_recognition_available() -> bool:
    """Check if the face_recognition library is installed."""
    _check_deps()
    return _face_recognition is not None


def get_face_embedding(face_image) -> list[float] | None:
    """Extract a 128-dim face embedding from a cropped face image.

    Requires face_recognition + OpenCV. Returns None if unavailable.
    """
    _check_deps()
    if not _face_recognition or not _cv2_available:
        return None
    rgb = _cv2.cvtColor(face_image, _cv2.COLOR_BGR2RGB)
    embeddings = _face_recognition.face_encodings(rgb)
    if not embeddings:
        return None
    return embeddings[0].tolist()


def compare_faces(
    known_embedding: list[float],
    candidate_embedding: list[float],
    tolerance: float = 0.6,
) -> bool:
    """Compare two face embeddings via Euclidean distance."""
    if not known_embedding or not candidate_embedding or not _np_available:
        return False
    _check_deps()
    dist = _np.linalg.norm(
        _np.array(known_embedding) - _np.array(candidate_embedding)
    )
    return dist <= tolerance


def find_matching_user(
    candidate_embedding: list[float],
    known_users: list[tuple[str, list[float]]],
    tolerance: float = 0.6,
) -> str | None:
    """Find which known user matches a face embedding.

    Args:
        candidate_embedding: The embedding to match.
        known_users: List of (user_id, embedding) tuples.
        tolerance: Maximum Euclidean distance for a match.

    Returns:
        user_id of the best match, or None if no match found.
    """
    if not candidate_embedding or not _np_available:
        return None
    _check_deps()

    best_user = None
    best_dist = float("inf")
    candidate = _np.array(candidate_embedding)

    for user_id, emb in known_users:
        if not emb:
            continue
        dist = _np.linalg.norm(_np.array(emb) - candidate)
        if dist < best_dist and dist <= tolerance:
            best_dist = dist
            best_user = user_id

    return best_user


# ── Camera capture ────────────────────────────────────────────────

def capture_face_image(camera_id: int = 0) -> object | None:
    """Capture a frame from camera and detect a face. Returns face image or None."""
    _check_deps()
    if not _cv2_available:
        return None
    cap = _cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None
    try:
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return detect_face(frame)
    finally:
        cap.release()


def capture_face_for_registration(
    num_attempts: int = 3,
) -> tuple[object | None, list[float] | None]:
    """Capture a face and compute its embedding for user registration.

    Returns (face_image, embedding) — both None if capture fails.
    """
    best_face = None
    best_embedding = None

    for _ in range(num_attempts):
        face = capture_face_image()
        if face is not None:
            best_face = face
            if is_face_recognition_available():
                emb = get_face_embedding(face)
                if emb:
                    best_embedding = emb
                    break
            else:
                break

    return best_face, best_embedding


# ── Image encoding ────────────────────────────────────────────────

def face_image_to_base64(face_image) -> str:
    """Convert a face image (numpy array) to a base64 JPEG string."""
    _check_deps()
    _, buffer = _cv2.imencode(".jpg", face_image)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_face_image(b64_str: str) -> object | None:
    """Convert a base64 JPEG string back to a numpy array."""
    _check_deps()
    if not _np_available or not _cv2_available:
        return None
    try:
        data = base64.b64decode(b64_str)
        arr = _np.frombuffer(data, _np.uint8)
        return _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    except Exception:
        return None
