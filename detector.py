"""
detection/detector.py

Thin wrapper around a fine-tuned YOLOv5 checkpoint that exposes a simple
`.detect(frame) -> list[dict]` interface, so the rest of the pipeline
(pipeline/realtime_pipeline.py) doesn't need to know anything about
torch.hub, autoshape, or YOLOv5 internals.

Loading priority:
  1. Local clone (third_party/yolov5) + local .pt weights via torch.hub
     (works fully offline once the repo + weights are present).
  2. Falls back to `torch.hub.load("ultralytics/yolov5", ...)` which fetches
     the repo code from GitHub on first run (needs internet once, then caches).
"""

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = REPO_ROOT / "third_party" / "yolov5"

DEFAULT_CLASS_NAMES = ["human", "debris", "vehicle"]


class YOLOv5Detector:
    def __init__(
        self,
        weights: str,
        class_names: Optional[List[str]] = None,
        conf_thres: float = 0.4,
        iou_thres: float = 0.45,
        img_size: int = 640,
        device: str = "",
    ):
        import torch  # local import: keeps this module importable without torch for unit tests

        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size

        if (YOLOV5_DIR / "hubconf.py").exists():
            self.model = torch.hub.load(str(YOLOV5_DIR), "custom", path=weights,
                                         source="local")
        else:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)

        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres
        if device:
            self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Runs inference on a single BGR (OpenCV-style) frame.

        Returns a list of dicts: {'xyxy': (x1,y1,x2,y2), 'conf': float, 'cls': int, 'name': str}
        """
        # YOLOv5's autoshape wrapper accepts numpy arrays directly (BGR is fine,
        # it handles the channel order internally for cv2-sourced frames).
        results = self.model(frame, size=self.img_size)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            cls = int(cls)
            detections.append({
                "xyxy": tuple(xyxy),
                "conf": float(conf),
                "cls": cls,
                "name": self.class_names[cls] if cls < len(self.class_names) else str(cls),
            })
        return detections


class MockDetector:
    """Drop-in replacement for YOLOv5Detector with no torch/model dependency.

    Used by tests/test_pipeline_mock.py and for dry-running the SITL
    integration logic before a trained checkpoint exists.
    """

    def __init__(self, scripted_detections: Optional[List[List[Dict]]] = None,
                 class_names: Optional[List[str]] = None):
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self._script = scripted_detections or []
        self._call_count = 0

    def detect(self, frame: np.ndarray) -> List[Dict]:
        if self._script:
            result = self._script[min(self._call_count, len(self._script) - 1)]
        else:
            result = []
        self._call_count += 1
        return result
