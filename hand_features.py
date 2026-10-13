"""Turns a camera picture into a list of 63 numbers describing the hand shape.

MediaPipe finds 21 points on the hand. We move the wrist to the centre, make
the hand a standard size, and (for left hands) mirror it, so the same sign
gives similar numbers wherever the hand is and whichever hand is used.
"""
import cv2
import numpy as np
import mediapipe as mp

_mp_hands = mp.solutions.hands
MAX_SIDE = 1024  # very large photos are shrunk first, it makes no difference to accuracy


def create_detector():
    return _mp_hands.Hands(
        static_image_mode=True,   # every request is an independent picture
        max_num_hands=1,
        min_detection_confidence=0.5,
    )


def normalize_landmarks(points, mirror=False):
    """points: 21 rows of (x, y, z). Returns 63 floats, or None if the hand is degenerate."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts -= pts[0]                 # wrist becomes the origin
    if mirror:
        pts[:, 0] *= -1           # left hand now looks like a right hand
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        return None
    pts /= scale                  # hand size no longer matters
    return pts.flatten()


def extract_features(detector, frame_bgr):
    """Returns the 63 numbers for the hand in the picture, or None if there is no hand."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        factor = MAX_SIDE / longest
        frame_bgr = cv2.resize(frame_bgr, (int(w * factor), int(h * factor)))
        h, w = frame_bgr.shape[:2]

    result = detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    # x and y are fractions of the picture, so scale them to pixels to keep the real proportions
    points = [[p.x * w, p.y * h, p.z * w] for p in landmarks]

    mirror = False
    if result.multi_handedness:
        mirror = result.multi_handedness[0].classification[0].label == "Left"
    return normalize_landmarks(points, mirror)
