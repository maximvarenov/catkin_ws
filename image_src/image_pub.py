
import sys
import cv2
import rospy
import os
import time
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

 
def pubImage():
   rospy.init_node('pubImage',anonymous = True)
   pub = rospy.Publisher('IVUSimg', Image, queue_size = 10)
   rate = rospy.Rate(10)
   bridge = CvBridge()  

   #imagepath = "/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/em_us_paper/image_sub/3_predict.png"
   imagepath = "/media/ruixuan/Volume/ruixuan/Documents/us_image/1/86.png"
   while not rospy.is_shutdown():
       image = cv2.imread(imagepath) 
       cv2.imshow("image_window",image)
       cv2.waitKey(500)

       image = bridge.cv2_to_imgmsg(image,"bgr8")
       image.header.stamp = rospy.Time.now() 
       image.header.frame_id = 'image'
       print(image.header)

       pub.publish(image)
       rate.sleep()



if __name__ == '__main__':
    try:
        pubImage()
    except rospy.ROSInterruptException:
        pass



