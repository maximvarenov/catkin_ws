

import sys
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

count =0

def callback(data):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data,"bgr8") 
    for count in range(1,500):
        cv2.imwrite("/media/ruixuan/Volume/ruixuan/Pictures/student/4/"+ str(count)+".png", cv_image) 
        count += 1


def showImage():

    rospy.init_node('subImage',anonymous = True)
    rospy.Subscriber('/usb_cam/image_raw', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    showImage()
