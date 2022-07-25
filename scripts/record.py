#!/usr/bin/env python3

"""
Collects a sequence of images from the sensor.

roslaunch gelsight_ros record.launch output_path:=(...) num_sec:=(Optional 30s def) num_imgs:=(Optional none def)

Stream configured in config/gelsight.yml
"""

import cv2
import gelsight_ros as gsr
import os
import rospy

# Default parameters
DEFAULT_RATE = 30
DEFAULT_DURATION = 30
DEFAULT_NUM_IMGS = 5000

if __name__ == "__main__":
    rospy.init_node("record")
    rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))
    end_time = rospy.Time.now() + rospy.Duration(rospy.get_param("~num_secs", DEFAULT_DURATION))
    num_imgs = rospy.get_param("~num_imgs")
    if num_imgs == -1:
        num_imgs = DEFAULT_NUM_IMGS

    if not rospy.has_param("~output_path"):
        rospy.signal_shutdown("No output path provided. Please set output_path/.")
    output_path = rospy.get_param("~output_path")
    if output_path[-1] == "/":
        output_path = output_path[:len(output_path)-1]

    if not os.path.exists(output_path):
        rospy.logwarn("Output folder doesn't exist, will create it.")
        os.makedirs(output_path)
        
        if not os.path.exists(output_path):
            rospy.signal_shutdown(f"Failed to create output folder: {output_path}")

    cfg = rospy.get_param("stream")
    if not cfg:
        rospy.signal_shutdown("No config provided for HTTP stream. Please set http_stream/.")
    stream = gsr.ImageProc(cfg)
    
    # Main loop
    i = 0
    while not rospy.is_shutdown() and stream.is_running() and \
        rospy.Time.now() < end_time and i < num_imgs:
        try:
            stream.execute()
            frame = stream.get_frame()
            if not cv2.imwrite(f"{output_path}/{i}.jpg", frame):
                rospy.logwarn(f"Failed to write file to {output_path}")
            else:
                i += 1
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
