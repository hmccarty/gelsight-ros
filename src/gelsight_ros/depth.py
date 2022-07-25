#!/usr/bin/env python3

from collections import deque
import cv2
from find_marker import Matching
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
from rospy import AnyMsg
from sensor_msgs.msg import PointCloud2
import torch
from typing import Dict, Tuple, Any, Optional

from .proc import GelsightProc, ImageProc, ImageDiffProc
from .proc import ProcExecutionError
from .data import GelsightDepth, GelsightPose
from .grad_model import RGB2Grad
from .util import *

class DepthProc(GelsightProc):
    """Converts image streams into depth maps."""

    def get_depth(self) -> GelsightDepth:
        raise NotImplementedError()

    def get_mpp(self) -> float:
        raise NotImplementedError()

    def get_ros_type(self) -> PointCloud2:
        return PointCloud2

class DepthFromModelProc(DepthProc):
    """
    Uses trained NN to convert image streams into depth maps.
    
    execute() -> PointCloud2 msg 

    Params:
      - model_path (Required)
      - compute_type (default: 'cpu')
      - model_output_width (default: 120)
      - model_output_height (default: 160)
    """

    # Parameter defaults
    compute_type: str = "cuda"
    image_mpp: float = 0.005
    model_output_width: int = 120
    model_output_height: int = 160
    depth_min: float = 3.0
    depth_ratio: float = 0.5

    def __init__(self, stream: ImageProc, diff_stream: ImageDiffProc, cfg: Dict[str, Any]):
        super().__init__()
        self._stream: ImageProc = stream
        self._diff_stream: ImageDiffProc = diff_stream

        if "model_path" not in cfg:
            raise RuntimeError("Missing model path.")

        if "compute_type" in cfg:
            self.compute_type = cfg["compute_type"]
        if "image_mpp" in cfg:
            self.image_mpp = cfg["image_mpp"]
        if "model_output_width" in cfg:
            self.model_output_width = cfg["model_output_width"]
        if "model_output_height" in cfg:
            self.model_output_height = cfg["model_output_height"]

        self._model = RGB2Grad()
        self._model.load_state_dict(torch.load(cfg["model_path"]))
        self._model.eval()

        self._init_dm: Optional[GelsightDepth] = None
        self._dm: Optional[GelsightDepth] = None

    def execute(self):
        diff_frame = self._diff_stream.get_frame()
        if diff_frame is None:
            ProcExecutionError("Diff frame returned 'None'") 
        
        # Transform frame into model input
        batch_len = self.model_output_height * self.model_output_width
        X = np.reshape(diff_frame, (batch_len, 3))
        xv, yv = np.meshgrid(np.arange(diff_frame.shape[1]), np.arange(diff_frame.shape[0]))
        X = np.concatenate((X, np.reshape(xv, (batch_len, 1))), axis=1)
        X = np.concatenate((X, np.reshape(yv, (batch_len, 1))), axis=1)

        # Collect gradients from model and reshape
        grad = self._model(torch.from_numpy(X.astype(np.float32)))
        gx = grad.detach().numpy()[:, 1].reshape((self.model_output_height, self.model_output_width))
        gy = grad.detach().numpy()[:, 0].reshape((self.model_output_height, self.model_output_width))

        # Interpolate gradients over markers
        frame = self._stream.get_frame()
        if frame is None:
            raise ProcExecutionError("Stream frame returned 'None'")
        gx, gy = demark(self._stream.get_frame(), gx, gy)

        # Construct depth map using poisson reconstruction
        boundary = np.zeros((self.model_output_height, self.model_output_width))
        dm = poisson_reconstruct(gx, gy, boundary)
        dm = np.reshape(dm, (self.model_output_height, self.model_output_width))

        # Process depth map and return
        dm *= -1
        if self._init_dm is None:
            self._init_dm = dm
        dm -= self._init_dm
        dm[dm < self.depth_min] = 0.0
        dm[dm < self.depth_ratio * np.max(dm)] = 0.0
        self._dm = GelsightDepth(self.model_output_width, self.model_output_height, dm)
    
    def get_depth(self) -> GelsightDepth:
        return self._dm

    def get_mpp(self) -> float:
        return self.image_mpp

    def get_ros_msg(self) -> Optional[PointCloud2]:
        if self._dm is not None:
            return self._dm.get_ros_msg(self.image_mpp)


class DepthFromCoeffProc(DepthProc):
    """
    Approximates depth maps from stream using poisson reconstruction.
    See more: https://hhoppe.com/poissonrecon.pdf

    execute() -> PointCloud2 msg

    Params:
      - image_mpp (default: 0.005)
      - image_width (default: 120)
      - image_height (default: 160)
    """

    # Parameter defaults
    image_mpp: float = 0.005
    image_width: int = 120
    image_height: int = 160

    def __init__(self, stream: ImageProc, diff_stream: ImageDiffProc, cfg: Dict[str, Any]):
        super().__init__()
        self._stream: ImageProc = stream
        self._diff_stream: ImageDiffProc = diff_stream

        if "image_mpp" in cfg:
            self.image_mpp = cfg["image_mpp"]

        self._init_frame = None
        self._init_dm = None

    def execute(self):
        # Compute depth map using solver method
        diff = self._diff_stream.get_frame() 
        dx = diff[:, :, 1] / 255.0
        dx = dx / (1.0 - dx ** 2) ** 0.5 / 32.0
        dy = (diff[:, :, 0] - diff[:, :, 2]) / 255.0
        dy = dy / (1.0 - dy ** 2) ** 0.5 / 32.0

        diff /= 255.0
        frame = self._stream.get_frame()
        dx, dy = demark(frame, dx, dy)

        zeros = np.zeros_like(dx)
        dm = poisson_reconstruct(dy, dx, zeros)

        if self._init_dm is None:
            self._init_dm = dm
        
        dm -= self._init_dm
        dm[dm < 0] = 0

        self._dm = GelsightDepth(self.image_width, self.image_height, dm)
    
    def get_depth(self) -> GelsightDepth:
        return self._dm

    def get_mpp(self) -> float:
        return self.image_mpp

    def get_ros_msg(self) -> PointCloud2:
        return self._dm.get_ros_msg(self.image_mpp)

class PoseFromDepthProc(GelsightProc):
    """
    Approximates contact pose from depth using PCA. 

    execute() -> PoseStamped msg

    Params:
      - buffer_size (default: 5)
    """

    # Parameter defaults
    buffer_size: int = 5

    def __init__(self, depth: DepthProc, cfg: Dict[str, Any]):
        super().__init__()
        self._depth: DepthProc = depth

        if "buffer_size" in cfg:
            self.buffer_size = cfg["buffer_size"]

        self._buffer: deque = deque([], maxlen=self.buffer_size)
        self._pose: Optional[GelsightPose] = None

    def execute(self):
        gsdepth = self._depth.get_depth()

        if gsdepth: 
            dm = gsdepth.depth

            pnts = np.where(dm > 0)
            X = pnts[1].reshape(-1, 1)
            Y = pnts[0].reshape(-1, 1)
            pnts = np.concatenate([X, Y], axis=1)
            pnts = pnts.reshape(-1, 2).astype(np.float64)
            if pnts.shape[0] == 0:
                return None

            mv = np.mean(pnts, 0).reshape(2, 1)
            pnts -= mv.T
            w, v = np.linalg.eig(np.dot(pnts.T, pnts))
            w_max = np.max(w)

            col = np.where(w == w_max)[0]
            if len(col) > 1:
                col = col[-1]

            V_max = v[:, col]
            if V_max[0] > 0 and V_max[1] > 0:
                V_max *= -1

            V_max = V_max.reshape(-1) * (w_max**0.3 / 1)
            theta = math.atan2(V_max[1], V_max[0])

            if len(self._buffer) > 0:
                self._buffer.popleft()
            self._buffer.append((mv[0], mv[1], theta))

            x_bar = 0.0
            y_bar = 0.0
            theta_bar = 0.0
            for a in list(self._buffer):
                x, y, theta = a
                x_bar += x
                y_bar += y
                theta_bar += theta

            if len(self._buffer) > 0:
                x_bar /= len(self._buffer)
                y_bar /= len(self._buffer)
                theta_bar /= len(self._buffer)

            x_bar = (x_bar - (dm.shape[1]//2)) * self._depth.get_mpp()
            y_bar = (y_bar - (dm.shape[0]//2)) * self._depth.get_mpp()
            self._pose = GelsightPose(x_bar, y_bar, theta)

    def get_pose(self) -> GelsightPose:
        return self._pose

    def get_ros_type(self) -> PoseStamped:
        return PoseStamped

    def get_ros_msg(self) -> Optional[PoseStamped]:
        if self._pose is not None:
            return self._pose.get_ros_msg()
        return PoseStamped()