"""
pipeline/realtime_pipeline.py

The integration point of the whole project:

    Gazebo camera frame -> YOLOv5Detector -> decision logic -> ArduPilotController

Decision logic implemented here (kept intentionally simple/explainable for a
hackathon-scope perception-driven autonomy demo):
  - HUMAN detected with confidence above `human_conf_threshold`:
        hover in place + log a "victim found" event (lat/lon/alt at detection time)
  - DEBRIS detected directly ahead with confidence above `debris_conf_threshold`:
        sidestep (small lateral reroute) rather than flying straight through it
  - VEHICLE detected:
        logged only (informational — vehicles are a navigation landmark /
        context class, not an action trigger)

Each detection event is timestamped and appended to a session log
(JSON Lines) so a post-mission report can reconstruct what the UAV "saw"
and how it reacted, alongside the live precision/recall tally from
utils.metrics.DetectionTally.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from detection.detector import YOLOv5Detector, MockDetector
from sitl.mavlink_controller import ArduPilotController, MockController
from utils.visualization import draw_detections
from utils.metrics import DetectionTally


@dataclass
class PipelineConfig:
    human_conf_threshold: float = 0.5
    debris_conf_threshold: float = 0.45
    hover_seconds: float = 3.0
    sidestep_distance_m: float = 2.0
    debris_center_band: float = 0.25   # fraction of frame width considered "directly ahead"
    cooldown_seconds: float = 5.0      # avoid re-triggering on the same victim every frame
    log_path: str = "runs/session_log.jsonl"


class RealtimePerceptionPipeline:
    def __init__(self, detector, controller, config: Optional[PipelineConfig] = None):
        self.detector = detector
        self.controller = controller
        self.config = config or PipelineConfig()
        self.tally = DetectionTally()
        self._last_human_trigger_at = 0.0
        self._frame_count = 0

        Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.config.log_path, "a")

    # ------------------------------------------------------------- core loop

    def on_frame(self, frame: np.ndarray):
        """Callback registered with a camera bridge (Gazebo or video file)."""
        self._frame_count += 1
        detections = self.detector.detect(frame)
        self._react(frame, detections)
        draw_detections(frame, detections, self.detector.class_names)
        return frame, detections

    def _react(self, frame: np.ndarray, detections):
        frame_width = frame.shape[1] if frame is not None else None
        now = time.time()

        for det in detections:
            name = det.get("name", str(det["cls"]))
            conf = det["conf"]

            if name == "human" and conf >= self.config.human_conf_threshold:
                if now - self._last_human_trigger_at >= self.config.cooldown_seconds:
                    self._last_human_trigger_at = now
                    self._log_event("victim_detected", det)
                    self.controller.hover(self.config.hover_seconds)

            elif name == "debris" and conf >= self.config.debris_conf_threshold:
                if frame_width and self._is_in_path(det["xyxy"], frame_width):
                    self._log_event("debris_in_path", det)
                    self.controller.goto_relative(dx=0, dy=self.config.sidestep_distance_m, dz=0)

            elif name == "vehicle":
                self._log_event("vehicle_observed", det)

    def _is_in_path(self, xyxy, frame_width: int) -> bool:
        x1, _, x2, _ = xyxy
        box_center_x = (x1 + x2) / 2
        band_half_width = self.config.debris_center_band * frame_width / 2
        frame_center = frame_width / 2
        return abs(box_center_x - frame_center) <= band_half_width

    def _log_event(self, event_type: str, det: dict):
        record = {
            "t": time.time(),
            "frame": self._frame_count,
            "event": event_type,
            "class": det.get("name"),
            "conf": det.get("conf"),
            "xyxy": det.get("xyxy"),
        }
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()
        print(f"[pipeline] {event_type}: {det.get('name')} ({det.get('conf'):.2f})")

    def update_tally(self, tp: int = 0, fp: int = 0, fn: int = 0):
        """Optional: called by an evaluation harness running ground-truthed
        footage through the live pipeline, to track precision/recall online."""
        self.tally.update(tp=tp, fp=fp, fn=fn)

    def close(self):
        self._log_file.close()


def build_pipeline_from_args(args) -> RealtimePerceptionPipeline:
    config = PipelineConfig(
        human_conf_threshold=args.human_conf,
        debris_conf_threshold=args.debris_conf,
        hover_seconds=args.hover_seconds,
        log_path=args.log_path,
    )

    if args.mock:
        detector = MockDetector()
        controller = MockController()
    else:
        detector = YOLOv5Detector(weights=args.weights, conf_thres=min(args.human_conf, args.debris_conf))
        controller = ArduPilotController(connection_string=args.mavlink)
        controller.arm_and_takeoff(args.takeoff_alt)

    return RealtimePerceptionPipeline(detector, controller, config)


def main():
    parser = argparse.ArgumentParser(description="Run the perception-driven autonomous navigation pipeline")
    parser.add_argument("--weights", default="runs/train/disaster_uav_full/weights/best.pt")
    parser.add_argument("--mavlink", default="udp:127.0.0.1:14550")
    parser.add_argument("--camera-topic", default="/iris/camera/image_raw")
    parser.add_argument("--video-source", default=None,
                         help="use a video file/webcam instead of a ROS/Gazebo camera topic")
    parser.add_argument("--human-conf", type=float, default=0.5)
    parser.add_argument("--debris-conf", type=float, default=0.45)
    parser.add_argument("--hover-seconds", type=float, default=3.0)
    parser.add_argument("--takeoff-alt", type=float, default=10.0)
    parser.add_argument("--log-path", default="runs/session_log.jsonl")
    parser.add_argument("--mock", action="store_true",
                         help="run with MockDetector/MockController (no torch, no SITL needed)")
    args = parser.parse_args()

    pipeline = build_pipeline_from_args(args)

    if args.video_source is not None:
        from sitl.gazebo_bridge import VideoFileBridge
        bridge = VideoFileBridge(source=args.video_source)
    else:
        from sitl.gazebo_bridge import GazeboCameraBridge
        bridge = GazeboCameraBridge(topic=args.camera_topic)

    bridge.register_callback(pipeline.on_frame)
    try:
        bridge.spin()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
