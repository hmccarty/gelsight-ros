#!/usr/bin/env python3

"""
Allows you to quickly check the meter-per-pixel
of a Gelsight sensor.

Directions:
Press a pair of calipers on the finger of the gelsight and take a screenshot.
Run this script with the length and image path as arguments.
Click the two points of calipers and check the command-line for the value.
"""

import cv2
import gelsight_ros as gsr
import rospy

if __name__ == "__main__":
    rospy.init_node("cv_window")

    # Setup stream
    cfg = rospy.get_param("stream")
    if "roi" in cfg:
        del cfg["roi"]
    stream = gsr.ImageProc(cfg)

    while not rospy.is_shutdown() and stream.is_running():
        stream.execute()
        cv2.imshow("cv_window", stream.get_frame()) 
        
        if cv2.waitKey(2) == ord('q'):
            break
    
    cv2.destroyAllWindows()