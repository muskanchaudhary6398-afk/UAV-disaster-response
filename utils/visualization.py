"""
utils/visualization.py

Draws YOLOv5 detections on a frame for on-screen overlay or saved-to-disk
debugging during SITL runs. Kept dependency-light (just OpenCV) since this
runs in the hot path of the real-time inference pipeline.
"""

from typing import Iterable

import cv2

# Fixed colour-per-class so overlays stay visually consistent across frames
CLASS_COLORS = {
    "human": (0, 0, 255),     # red   — highest-priority class
    "debris": (0, 165, 255),  # orange
    "vehicle": (255, 200, 0), # cyan-ish
}


def draw_detections(frame, detections: Iterable[dict], class_names=None):
    """detections: iterable of dicts with keys
        {'xyxy': (x1,y1,x2,y2), 'conf': float, 'cls': int or str}
    Returns the frame with boxes/labels drawn in place (and returned for chaining).
    """
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        cls = det["cls"]
        label_name = class_names[cls] if class_names and isinstance(cls, int) else str(cls)
        color = CLASS_COLORS.get(label_name, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{label_name} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def save_annotated_frame(frame, path: str):
    cv2.imwrite(path, frame)
