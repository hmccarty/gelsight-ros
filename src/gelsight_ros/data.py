#!/usr/bin/env python3

from dataclasses import dataclass
from gelsight_ros.msg import GelsightMarkers as GelsightMarkersMsg, \
    GelsightMarkersStamped as GelsightMarkersStampedMsg, GelsightFlowStamped as GelsightFlowStampedMsg
from geometry_msgs.msg import PoseStamped
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler



@dataclass
class GelsightFrame:
    """Stores a single frame captured by sensor"""
    image: np.ndarray
    encoding: str

    def get_ros_msg(self) -> Image:
        return CvBridge().cv2_to_imgmsg(self.image, self.encoding)

@dataclass
class GelsightMarkers:
    """Stores position of each marker within a frame"""
    rows: int
    cols: int 
    markers: np.ndarray # List of markers coordinates (n_markers, 2)

    def __post_init__(self):
        if self.rows <= 0:
            raise ValueError(f"GelsightMarkers: rows cannot be less than or equal to 0, given: {self.rows}")
        elif self.cols <= 0:
            raise ValueError(f"GelsightMarkers: cols cannot be less than or equal to 0, given: {self.cols}")
        elif len(self.markers.shape) != 2 or self.markers.shape[1] != 2:
            raise ValueError(f"GelsightMarkers: markers must have shape (n_markers, 2), given shape: {self.markers.shape}")

    def get_ros_msg(self) -> GelsightMarkersStampedMsg:
        msg = GelsightMarkersStampedMsg()    
        msg.n = self.rows
        msg.m = self.cols
        msg.data = self.markers.astype('float32').flatten(order='C') # Row major
        return msg

@dataclass
class GelsightFlow:
    """Measures marker displacement"""
    ref: GelsightMarkers
    cur: GelsightMarkers

    def __post_init__(self):
        if len(self.ref.markers) != len(self.cur.markers):
            raise ValueError(f"Reference and current markers have different sizes")

    def get_ros_msg_stamped(self) -> GelsightFlowStampedMsg:
        msg = GelsightFlowStampedMsg()
        msg.ref_markers = self.ref.get_ros_msg().markers
        msg.cur_markers = self.cur.get_ros_msg().markers
        return msg 

@dataclass
class GelsightDepth:
    """Stores depth at each pixel"""
    depth: np.ndarray # (im_height, im_width, dtype=float32)

    def get_ros_msg(self, mpp: float) -> PointCloud2:
        height, width = self.depth.shape
        points = [] 
        for i in range(width):
            for j in range(height):
                points.append(
                    ((i - (width//2)) * mpp,
                     (j - (height//2)) * mpp,
                     self.depth[j, i] / 1000.0, int(0))
                )

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("r/home/yu/Documents/catkin_ws/src/gelsight-ros/data/panpan_datagb", 12, PointField.UINT32, 1),
        ]

        header = Header()
        return point_cloud2.create_cloud(header, fields, points)

@dataclass
class GelsightPCA:
    """Stores principal axis"""
    x: float
    y: float
    theta: float

    def get_ros_msg(self) -> PoseStamped:
        pose = PoseStamped()
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = 0.0

        x, y, w, z = quaternion_from_euler(0.0, 0.0, self.theta)
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = w
        pose.pose.orientation.w = z

        return pose