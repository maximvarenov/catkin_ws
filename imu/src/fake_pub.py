#!/usr/bin/env python
#coding:utf-8

import numpy as np
import rospy
import time
import string
import glob 
import sys
import cv2 
import os  
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu,MagneticField
from std_msgs.msg import String
from datetime import datetime


def talker():

    rospy.init_node('pub_simulation_node')
    pub_pose = rospy.Publisher('imu_data', Imu, queue_size = 1) 
    rate = rospy.Rate(1) 
    myImu = Imu()
    myMag = MagneticField()

    filename = "/media/ruixuan/Volume/ruixuan/Documents/database/us_image/26-06/pose3.txt"
    file=open(filename)
    lines=file.readlines()

    for line in lines: 

        split_string = string.split(line,' ')

        myImu.header.stamp = image.header.stamp
        myImu.header.frame_id = 'myImu_pose' 
        split_string = string.split(line,' ') 
        myImu.linear_acceleration.x = (float(split_string[1]) ) 
        myImu.linear_acceleration.y = (float(split_string[3]) )
        myImu.linear_acceleration.z = (float(split_string[5]) )  
        ##  unit m/s2  without gravity  filtered
        
        myImu.angular_velocity.x = float(split_string[7]) 
        myImu.angular_velocity.y = float(split_string[9]) 
        myImu.angular_velocity.z = float(split_string[11])  
        ## angular velocity unit rad/s filtered

        myMag.magnetic_field.x = float(split_string[13])
        myMag.magnetic_field.y = float(split_string[15])
        myMag.magnetic_field.z = float(split_string[17])   
        ##mag milliGauss

        myImu.orientation.w =  float(split_string[19])
        myImu.orientation.x =  float(split_string[21])
        myImu.orientation.y =  float(split_string[23])
        myImu.orientation.z =  float(split_string[25]) 

        print(myImu)

        pub_pose.publish(myImu)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
