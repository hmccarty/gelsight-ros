#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rospy import Message as ROSMsg
from sensor_msgs.msg import Image
from typing import Dict, Tuple, Any, Optional

from .util import *

class ProcExecutionError(Exception):
    """Raised when non-fatal error occurs in execution"""
    pass

class GelsightOp:
    def __call__(self):
        return NotImplementedError()

class GelsightProc:
    def execute(self) -> bool:
        raise NotImplementedError()

    def get_ros_type(self) -> ROSMsg:
        raise NotImplementedError()

    def get_ros_msg(self) -> ROSMsg:
        raise NotImplementedError()

class ImageProc(GelsightProc):
    """
    Collects image frame from stream.
    """

    # Parameter defaults
    encoding: str = "bgr8"
    size: Tuple[int, int] = (120, 160) # width, height 

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        if "url" not in cfg:
            raise RuntimeError("Missing stream url.")
        self._dev = cv2.VideoCapture(cfg["url"])
        self.size = (
            int(cfg["width"]) if "width" in cfg else self.size[0],
            int(cfg["height"]) if "height" in cfg else self.size[1],
        )

        self.output_coords = [(0, 0), (self.size[0], 0), self.size, (0, self.size[1])]
        self._roi = cfg["roi"] if "roi" in cfg else None
        self._frame = None

    def is_running(self) -> bool:
        return self._dev.isOpened()

    def execute(self):
        ret, frame = self._dev.read()
        if not ret:
            raise RuntimeError("Unable to grab image from stream. Check GelSight feed.")

        # Warp to match ROI
        if self._roi is not None:
            M = cv2.getPerspectiveTransform(
                np.float32(self._roi), np.float32(self.output_coords))
            frame = cv2.warpPerspective(frame, M, self.size)

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        if self._frame is not None:
            return np.copy(self._frame)

    def get_ros_type(self) -> Image:
        return Image

    def get_ros_msg(self) -> Optional[Image]:
        return CvBridge().cv2_to_imgmsg(self._frame, self.encoding)

class ImageDiffProc(GelsightProc):
    """
    Computes pixel-diff from initial frame in stream.
    """

    # Parameter defaults
    intensity: float = 3.0
    encoding: str = "8UC3"

    def __init__(self, stream: ImageProc):
        super().__init__()
        self._stream: ImageProc = stream
        self._init_frame: Optional[np.ndarray] = None
        self._diff_frame: Optional[np.ndarray] = None

    def execute(self):
        frame = self._stream.get_frame()
        if self._init_frame is None:
            self._init_frame = frame

        diff = ((frame * 1.0) - self._init_frame) * self.intensity
        diff[diff > 255] = 255
        diff[diff < 0] = 0
        self._diff_frame = diff

    def get_frame(self) -> Optional[np.ndarray]:
        if self._diff_frame is not None:
            return np.copy(self._diff_frame)

    def get_ros_type(self) -> Image:
        return Image

    def get_ros_msg(self) -> Optional[Image]:
        if self._diff_frame is not None:
            return CvBridge().cv2_to_imgmsg(np.uint8(self._diff_frame), self.encoding)
