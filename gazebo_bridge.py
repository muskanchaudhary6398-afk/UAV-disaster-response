"""
sitl/gazebo_bridge.py

ROS node that subscribes to the simulated UAV's camera topic (published by
the ardupilot_gazebo plugin / gazebo_ros camera plugin) and forwards each
frame, as an OpenCV BGR numpy array, to a registered callback.

This isolates all ROS-specific code in one place: the rest of the pipeline
(pipeline/realtime_pipeline.py) only ever deals with plain numpy frames, so
it can also be driven from a recorded video file or a webcam for development
without ROS installed at all.

Requires a sourced ROS environment (ROS Noetic or ROS 2 + ros1_bridge):
    source /opt/ros/noetic/setup.bash
    rosrun gazebo_ros gazebo worlds/iris_arducopter_runway.world &
"""

from typing import Callable

import numpy as np


class GazeboCameraBridge:
    """
    Usage:
        bridge = GazeboCameraBridge(topic="/webcam/image_raw")
        bridge.register_callback(my_frame_handler)
        bridge.spin()
    """

    def __init__(self, topic: str = "/iris/camera/image_raw", queue_size: int = 1):
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        self._rospy = rospy
        self._bridge = CvBridge()
        self._callbacks: list[Callable[[np.ndarray], None]] = []

        if not rospy.core.is_initialized():
            rospy.init_node("uav_perception_bridge", anonymous=True)

        self._sub = rospy.Subscriber(topic, Image, self._on_image, queue_size=queue_size)
        print(f"[gazebo_bridge] subscribed to {topic}")

    def register_callback(self, fn: Callable[[np.ndarray], None]):
        self._callbacks.append(fn)

    def _on_image(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        for cb in self._callbacks:
            cb(frame)

    def spin(self):
        self._rospy.spin()


class VideoFileBridge:
    """Non-ROS stand-in: streams frames from a video file or webcam index.
    Useful for developing/testing the perception pipeline without a full
    ROS + Gazebo + ArduPilot SITL stack running.
    """

    def __init__(self, source=0):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    def register_callback(self, fn: Callable[[np.ndarray], None]):
        self._callbacks.append(fn)

    def spin(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            for cb in self._callbacks:
                cb(frame)
            if self._cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
