#!/usr/bin/env python3

"""
Runs main pipeline for gelsight inference.

Conifgured in config/gelsight.yml
"""

import gelsight_ros as gsr
from gelsight_ros.msg import GelsightFlowStamped, GelsightMarkersStamped
from geometry_msgs.msg import PoseStamped
import rospy
from rospy import AnyMsg
from sensor_msgs.msg import PointCloud2, Image
from typing import List, Dict, Any

# Default parameters
DEFAULT_RATE = 30
DEFAULT_QUEUE_SIZE = 4
DEFAULT_DEPTH_METHOD = "poisson"

class Gelsight:
    def __init__(self):
        self.processes: List[gsr.GelsightProc]  = []

    def add_process(self, proc: gsr.GelsightProc, topic_name: str = ""): 
        pub = None
        if topic_name != "":
            # Set queue size by ROS type 
            ros_type = proc.get_ros_type()
            queue_size = DEFAULT_QUEUE_SIZE
            if ros_type == Image:
                queue_size = 2

            pub = rospy.Publisher(topic_name, ros_type, queue_size=queue_size) 
        else:
            rospy.logwarn(f"{proc.__class__.__name__}: topic_name not set, publisher disabled.")
        self.processes.append((proc, pub))

    def execute(self):
        for process, pub in self.processes:
            try:
                process.execute()
                if pub:
                    msg = process.get_ros_msg() 
                    if hasattr(msg, "header"):
                        # TODO: Set actual frame ID
                        msg.header.frame_id = "map"
                    pub.publish(msg)
            except Exception as e:
                import traceback
                rospy.logerr(f"{process.__class__.__name__}: Exception occured: {traceback.format_exc()}")

def get_topic_name(cfg: Dict[str, Any]):
    return cfg["topic_name"] if "topic_name" in cfg else ""

if __name__ == "__main__":
    rospy.init_node("gelsight")
    rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))

    # Create and add basic image streaming processes
    gelsight = Gelsight()
    image_cfg = rospy.get_param("~http_stream")
    image_proc = gsr.ImageProc(image_cfg)
    gelsight.add_process(image_proc, get_topic_name(image_cfg))

    # Load image diff process 
    if rospy.get_param("~diff/enable", False):
        diff_cfg = rospy.get_param("~diff")
        diff_proc = gsr.ImageDiffProc(image_proc)
        gelsight.add_process(gsr.ImageDiffProc(image_proc), get_topic_name(diff_cfg))

    # Load depth reconstruction process
    if rospy.get_param("~depth/enable", False):
        depth_cfg = rospy.get_param("~depth")
        depth_method = rospy.get_param("~depth/method", DEFAULT_DEPTH_METHOD)
        depth_topic = get_topic_name(depth_cfg)

        if depth_method == "functional":
            # Compute depth using functional approx
            depth_proc = gsr.DepthFromCoeffProc(stream, depth_cfg)
            gelsight.add_process(depth_proc, depth_topic)
        elif depth_method == "model":
            # Compute depth using trained model
            depth_proc = gsr.DepthFromModelProc(stream, depth_cfg)
            gelsight.add_process(depth_proc, depth_topic)
        else:
            rospy.signal_shutdown(f"Depth method not recognized: {depth_method}")

        # Load pose process
        if rospy.get_param("~pose/enable", False):
            pose_cfg = rospy.get_param("~pose")
            pose_proc = gsr.PoseFromDepthProc(depth_proc, pose_cfg)
            gelsight.add_process(pose_proc, get_topic_name(pose_cfg))
    
    elif rospy.get_param("~pose/enable", False):
        rospy.logwarn("Pose detection is enabled, but depth computing is disabled. Pose will be ignored.")

    # Load marker process
    if rospy.get_param("~markers/enable", False):
        marker_cfg = rospy.get_param("~markers")
        marker_proc = gsr.MarkersProc(image_proc, marker_cfg)
        gelsight.add_process(marker_proc, get_topic_name(marker_cfg))

        if rospy.get_param("~markers/publish_image", False):
            marker_im_proc = gsr.DrawMarkersProc(image_proc, marker_proc)
            gelsight.add_process(marker_im_proc, get_topic_name(marker_cfg) + "_image")

        # Load flow process
        if rospy.get_param("~flow/enable", False):
            flow_cfg = rospy.get_param("~flow")
            flow_proc = gsr.FlowProc(marker_proc, flow_cfg)
            gelsight.add_process(flow_proc, get_topic_name(flow_cfg))

            if rospy.get_param("~flow/publish_image", False):
                flow_im_proc = gsr.DrawFlowProc(image_proc, flow_proc)
                gelsight.add_process(flow_im_proc, get_topic_name(flow_cfg) + "_image")
    
    elif rospy.get_param("~flow/enable", False):
        rospy.logwarn("Flow detection is enabled, but marker tracking is disabled. Flow will be ignored.")

    # Main loop
    while not rospy.is_shutdown() and image_proc.is_running():
        try:
            gelsight.execute() 
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
