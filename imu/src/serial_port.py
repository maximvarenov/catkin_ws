#! /usr/bin/env python

import serial
import math
import time
import rospy
import string
from sensor_msgs.msg import Imu,MagneticField
from std_msgs.msg import String
from datetime import datetime
import numpy as np

def talker():

    pub = rospy.Publisher('/imu_data', Imu)
    pub2 = rospy.Publisher('/mag_data', MagneticField)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(400) # 10-10hz  400-400hz
    myImu = Imu()
    myMag = MagneticField() 
    ser = serial.Serial('/dev/ttyACM0',115200,timeout=1)  # open first serial port
    print(ser.name)          # check which port was really used
    count=0 

    while not rospy.is_shutdown():
        line = ser.readline()
        rospy.loginfo(line)
        split_string = string.split(line,' ')
        myImu.header.stamp = rospy.get_rostime()
        myImu.header.frame_id = 'base_link'

        if    count> 30 :

            myImu.linear_acceleration.x = (float(split_string[0]) ) 
            myImu.linear_acceleration.y = (float(split_string[1]) )
            myImu.linear_acceleration.z = (float(split_string[2]) )  
            ##  unit m/s2  without gravity  filtered
            
            myImu.angular_velocity.x = float(split_string[3]) 
            myImu.angular_velocity.y = float(split_string[4]) 
            myImu.angular_velocity.z = float(split_string[5])  
            ## angular velocity unit rad/s filtered

            myMag.magnetic_field.x = float(split_string[6])
            myMag.magnetic_field.y = float(split_string[7])
            myMag.magnetic_field.z = float(split_string[8])   
            ##mag milliGauss

            myImu.orientation.w =  float(split_string[9])
            myImu.orientation.x =  float(split_string[10])
            myImu.orientation.y =  float(split_string[11])
            myImu.orientation.z =  float(split_string[12]) 


            pub.publish(myImu)
            pub2.publish(myMag)
            rospy.loginfo(myImu)

        count =count+1 
        r.sleep()

    ser.close()    
        
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
