"""
tests/test_pipeline_mock.py

Validates the perception -> decision -> action wiring in
pipeline/realtime_pipeline.py WITHOUT needing a trained YOLOv5 checkpoint,
GPU, ROS, Gazebo, or ArduPilot SITL running. This is the test we run in CI /
on a laptop to prove the integration logic is correct; the actual model
performance numbers (precision/recall) are validated separately via
scripts/evaluate.py against real footage.

Run with:
    pytest tests/test_pipeline_mock.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.detector import MockDetector
from sitl.mavlink_controller import MockController
from pipeline.realtime_pipeline import RealtimePerceptionPipeline, PipelineConfig


def make_frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_human_detection_triggers_hover(tmp_path):
    detections = [[{"xyxy": (100, 100, 200, 300), "conf": 0.91, "cls": 0, "name": "human"}]]
    detector = MockDetector(scripted_detections=detections)
    controller = MockController()
    config = PipelineConfig(log_path=str(tmp_path / "log.jsonl"))
    pipeline = RealtimePerceptionPipeline(detector, controller, config)

    pipeline.on_frame(make_frame())

    assert ("hover", config.hover_seconds) in controller.log
    pipeline.close()


def test_low_confidence_human_does_not_trigger(tmp_path):
    detections = [[{"xyxy": (100, 100, 200, 300), "conf": 0.2, "cls": 0, "name": "human"}]]
    detector = MockDetector(scripted_detections=detections)
    controller = MockController()
    config = PipelineConfig(log_path=str(tmp_path / "log.jsonl"))
    pipeline = RealtimePerceptionPipeline(detector, controller, config)

    pipeline.on_frame(make_frame())

    assert all(entry[0] != "hover" for entry in controller.log)
    pipeline.close()


def test_debris_directly_ahead_triggers_sidestep(tmp_path):
    # frame width 640 -> center band is +/-80px around x=320 by default config
    detections = [[{"xyxy": (290, 200, 350, 280), "conf": 0.7, "cls": 1, "name": "debris"}]]
    detector = MockDetector(scripted_detections=detections)
    controller = MockController()
    config = PipelineConfig(log_path=str(tmp_path / "log.jsonl"))
    pipeline = RealtimePerceptionPipeline(detector, controller, config)

    pipeline.on_frame(make_frame())

    sidestep_calls = [c for c in controller.log if c[0] == "goto_relative"]
    assert len(sidestep_calls) == 1
    pipeline.close()


def test_debris_off_to_the_side_does_not_trigger_sidestep(tmp_path):
    # box far to the left edge of a 640px-wide frame -> not "directly ahead"
    detections = [[{"xyxy": (0, 200, 40, 280), "conf": 0.7, "cls": 1, "name": "debris"}]]
    detector = MockDetector(scripted_detections=detections)
    controller = MockController()
    config = PipelineConfig(log_path=str(tmp_path / "log.jsonl"))
    pipeline = RealtimePerceptionPipeline(detector, controller, config)

    pipeline.on_frame(make_frame())

    assert all(c[0] != "goto_relative" for c in controller.log)
    pipeline.close()


def test_vehicle_detection_is_logged_but_no_action_taken(tmp_path):
    detections = [[{"xyxy": (10, 10, 60, 60), "conf": 0.8, "cls": 2, "name": "vehicle"}]]
    detector = MockDetector(scripted_detections=detections)
    controller = MockController()
    config = PipelineConfig(log_path=str(tmp_path / "log.jsonl"))
    pipeline = RealtimePerceptionPipeline(detector, controller, config)

    pipeline.on_frame(make_frame())

    assert controller.log == []  # no flight action for vehicles, only a log entry
    log_contents = (tmp_path / "log.jsonl").read_text()
    assert "vehicle_observed" in log_contents
    pipeline.close()


def test_human_cooldown_prevents_repeated_hover_every_frame(tmp_path):
    # Same high-confidence human detected for 3 consecutive frames; cooldown
    # should keep us from spamming hover() on every single frame.
    one_frame = [{"xyxy": (100, 100, 200, 300), "conf": 0.9, "cls": 0, "name": "human"}]
    detector = MockDetector(scripted_detections=[one_frame, one_frame, one_frame])
    controller = MockController()
    config = PipelineConfig(log_path=str(tmp_path / "log.jsonl"), cooldown_seconds=999)
    pipeline = RealtimePerceptionPipeline(detector, controller, config)

    for _ in range(3):
        pipeline.on_frame(make_frame())

    hover_calls = [c for c in controller.log if c[0] == "hover"]
    assert len(hover_calls) == 1
    pipeline.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
