#!/usr/bin/env python3

"""
Allows you to quickly check the meter-per-pixel
of a Gelsight sensor.

Directions:
Press a pair of calipers on the finger of the gelsight and take a screenshot.
Run this script with the length and image path as arguments:

rosrun gelsight_ros get_mpp.py _dist_mm:=(dist in mm) _img_path:=(path to img)

Click the two points of calipers and check the command-line for the value.
"""

import csv
from csv import writer
import cv2
import gelsight_ros as gsr
import math
import numpy as np
import os
import rospy

# Global variables
dist = None
click_a = None

def click_cb(event, x, y, flags, param):
    global dist, click_a
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_a is None:
            click_a = (x, y)
        else:
            px_dist = math.sqrt((x-click_a[0])**2 + (y-click_a[1])**2)
            print(f"MPP: {dist/px_dist}")
            exit()

if __name__ == "__main__":
    rospy.init_node("get_mpp")

    if not rospy.has_param("~dist_mm"):
        rospy.signal_shutdown("No dist provided. Please set dist_mm.")
    dist = rospy.get_param("~dist_mm") / 1000.0

    # Retrieve path where image is stored
    if not rospy.has_param("~img_path"):
        rospy.signal_shutdown("No input path provided. Please set img_path.")
    input_path = rospy.get_param("~img_path")
    im = cv2.imread(input_path)

    # Configure cv window
    cv2.namedWindow("get_mpp")
    cv2.setMouseCallback("get_mpp", click_cb)

    cv2.imshow("get_mpp", im) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()