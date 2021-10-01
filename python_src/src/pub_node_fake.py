import numpy as np
import rospy
import time
import string
import glob 
import sys
import cv2 
import os  
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
   

def pub(files,line):

    image = cv2.imread(files) 
    cv2.imshow("image_window",image)
    cv2.waitKey(50)
    image = bridge.cv2_to_imgmsg(image, 'bgr8' )
    image.header.stamp = rospy.Time.now() 
    image.header.frame_id = 'image'
  
    aurora.header.stamp = image.header.stamp 
    aurora.header.frame_id = 'pose' 
    split_string = string.split(line,' ') 
    print(split_string)
    aurora.pose.position.x = float(split_string[0])
    aurora.pose.position.y = float(split_string[1])
    aurora.pose.position.z = float(split_string[2])
    aurora.pose.orientation.x = float(split_string[3])
    aurora.pose.orientation.y = float(split_string[4])
    aurora.pose.orientation.z = float(split_string[5])
    aurora.pose.orientation.w = float(split_string[6])
    print(aurora)

    pub_image.publish(image)
    pub_pose.publish(aurora)

    rate.sleep()


rospy.init_node('pub_fake_node') 
pub_pose = rospy.Publisher('/Fusion_track', PoseStamped, queue_size = 10)
pub_image = rospy.Publisher('/IVUSimg', Image, queue_size = 10)
rate = rospy.Rate(50)
bridge = CvBridge() 
aurora = PoseStamped()

filename = "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_16/cal1.txt"
file=open(filename)
lines=file.readlines()
imagepath = os.listdir('/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_16/cal_1/') 
sort_num_list = []
for file in imagepath:
    sort_num_list.append(int(file.split('.png')[0]))      
    sort_num_list.sort() 

sorted_file = []
for sort_num in sort_num_list:
    for file in imagepath:
        if str(sort_num) == file.split('.png')[0]:
            sorted_file.append(file) 

for i,line in zip(sorted_file,lines):
    img=str('/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_16/cal_1/'+i) 
    if not rospy.is_shutdown():
       pub(img,line)

