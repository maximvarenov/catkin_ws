
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_matrix
import pyquaternion


def callback(pose):
    pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size = 10)
    rate = rospy.Rate(10) 
    ox = pose.data[12]
    oy = pose.data[13]
    oz = pose.data[14]
    R_dcm=np.array([[pose.data[0], pose.data[1],pose.data[2],0.0],[pose.data[4],pose.data[5],pose.data[6],0.0],[pose.data[8],pose.data[9],pose.data[10],0.0],[0,0,0,1]])
    R_quat = quaternion_from_matrix(R_dcm)
    R_quat= np.array(R_quat)  
   
    aurora = PoseStamped()
    aurora.header.frame_id = 'robot_pose'
    aurora.header.stamp = rospy.Time.now()
    aurora.pose.position.x= pose.data[12]
    aurora.pose.position.y= pose.data[13]
    aurora.pose.position.z= pose.data[14]
    aurora.pose.orientation.x= R_quat[0]
    aurora.pose.orientation.y= R_quat[1]
    aurora.pose.orientation.z= R_quat[2]
    aurora.pose.orientation.w= R_quat[3]
    
    position=[aurora.pose.position.x,aurora.pose.position.y,aurora.pose.position.z,aurora.pose.orientation.x,aurora.pose.orientation.y,aurora.pose.orientation.z,aurora.pose.orientation.w]
    f = open('/media/ruixuan/Volume/ruixuan/Documents/database/robot_data/plane.txt','a')
    f.write( str(position))
    f.write('\n')

    
    pub.publish(aurora) 


def trans():
    rospy.init_node('pose_trans')
    rospy.Subscriber('/virtuose/pose',Float64MultiArray,callback)
    rospy.spin()


if __name__ == '__main__':
    trans()
