import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(img):
    pub = rospy.Publisher('/IVUSimg_gray', Image, queue_size = 10)
    rate = rospy.Rate(1)
    bridge = CvBridge() 
    image_gray = Image()
    image = bridge.imgmsg_to_cv2(img,"bgr8")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("image_window",gray)
    #cv2.waitKey(50)
    image_gray = bridge.cv2_to_imgmsg(gray,"mono8")
    image_gray.header.stamp = rospy.Time.now() 
    print(image_gray.header.stamp)
    image_gray.header.frame_id = 'img_gray'

    pub.publish(image_gray) 
    rate.sleep()


def trans():
    rospy.init_node('img_gray_node')
    rospy.Subscriber('/IVUSimg', Image,callback)
    rospy.spin()


if __name__ == '__main__':
    trans()
