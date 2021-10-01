 
import numpy as np
import rospy
import time
import string
from ndi_aurora_msgs.msg import AuroraDataVector
from ndi_aurora_msgs.msg import AuroraData

from sensor_msgs.msg import LaserScan

def talker(): 

    pub = rospy.Publisher('/aurora_data', AuroraDataVector) 
    rospy.init_node('talker', anonymous=True)  
    pose = AuroraDataVector()
    data = AuroraData()
    

    while not rospy.is_shutdown():
        filename = '/media/ruixuan/Volume/ruixuan/Documents/master_program/rosbag/poes_bag/test.txt'
        file=open(filename)
        lines=file.readlines()

        for line in lines:

            split_string = string.split(line,'\t') 
            data.position.x = float(split_string[2])
            data.position.y = float(split_string[3])
            data.position.z = float(split_string[4])
            data.orientation.x = float(split_string[5])
            data.orientation.y = float(split_string[6])
            data.orientation.z = float(split_string[7])
            data.orientation.w = float(split_string[8])
            data.visible = 1
            data.portHandle = 0

            pose.data = data
            pose.timestamp = rospy.get_rostime() 

            pub.publish(pose)
            print(pose)

       
if __name__ == '__main__':
   try:
       talker()
   except rospy.ROSInterruptException: pass

