#!/usr/bin/env python
#coding:utf-8

import numpy as np
import rospy
from  ndi_aurora_msgs.msg import AuroraDataVector
from ndi_aurora_msgs.msg import AuroraData 

from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_matrix


def callback(pose):
      
    ox = pose.data[0].position.x
    oy = pose.data[0].position.y
    oz = pose.data[0].position.z
    qx1 = pose.data[0].orientation.x
    qy1 = pose.data[0].orientation.y
    qz1 = pose.data[0].orientation.z
    qw1 = pose.data[0].orientation.w
    print (ox, oy,oz,qx1,qy1,qz1,qw1)

def trans():
    rospy.init_node('tf_node')
    rospy.Subscriber('/aurora_data',AuroraDataVector,callback)
    rospy.spin()


if __name__ == '__main__':
    trans()
