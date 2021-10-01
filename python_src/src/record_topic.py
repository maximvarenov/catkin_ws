import cv2
import numpy as np
import rospy
from sensor_msgs.msg import * 
import message_filters
from std_msgs.msg import *
import os
  

def multi_callback(sub_force, sub_distance):
    file_path = '/media/ruixuan/Volume/ruixuan/Documents/database/force_sensor/F_D_test1.txt'
    while not rospy.is_shutdown():
        print('writing into txt...........')
        with open(file_path,'a') as f:   
            f.write(str(sub_force.data))
            f.write('   ')  
            f.write(sub_distance.data) 
            f.write('\n') 




if __name__ == '__main__':
    rospy.init_node('record_topic_node', anonymous=True)
    sub_force = message_filters.Subscriber('/wrench_raw', Float64MultiArray)
    sub_distance = message_filters.Subscriber("/distance", String)
 
    sync = message_filters.ApproximateTimeSynchronizer([sub_force, sub_distance], 10,0.1, allow_headerless=True)
    sync.registerCallback(multi_callback) 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("over!")
        cv2.destroyAllWindows()