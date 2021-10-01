
import numpy as np
import rospy
from ndi_aurora_msgs.msg import AuroraDataVector
from ndi_aurora_msgs.msg import AuroraData  
from geometry_msgs.msg import PoseStamped


def callback(pose):
    pub = rospy.Publisher('/aurora_data_header', PoseStamped, queue_size = 10) 
    aurora = PoseStamped()
    
    aurora.pose.position.x = pose.data[0].position.x
    aurora.pose.position.y = pose.data[0].position.y
    aurora.pose.position.z = pose.data[0].position.z
    aurora.pose.orientation.x = pose.data[0].orientation.x
    aurora.pose.orientation.y = pose.data[0].orientation.y
    aurora.pose.orientation.z = pose.data[0].orientation.z
    aurora.pose.orientation.w = pose.data[0].orientation.w
    visibility1 = pose.data[0].visible
    portHandle1 = pose.data[0].portHandle 
   

    print(aurora.header.stamp)
    aurora.header.frame_id = 'aurora_laserscan'
    aurora.header.stamp = rospy.Time.now()

    pub.publish(aurora) 


def auroratrans():
    rospy.init_node('tf_node')
    rospy.Subscriber('/aurora_data',AuroraDataVector,callback)
    rospy.spin()


if __name__ == '__main__':
    auroratrans()
