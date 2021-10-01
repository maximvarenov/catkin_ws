import rospy
from std_msgs.msg import String, Bool,  Float64MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from random import *
import time
# seed random number generator

x = np.linspace(0,180,200)
y=np.sin(x) 
count=0.0
seed(1)
 

def main():
    rospy.init_node('fake_pub_node', anonymous=True) 
    pub  = rospy.Publisher('/Fusion_track', PoseStamped, queue_size=10) 
    fusion  = PoseStamped()  
    rate = rospy.Rate(10) # 10hz 
    
    while not rospy.is_shutdown():
        global count
        randN = y[count]

        if randN > 0.8:
            send_msg = True
        else:
            send_msg = False

        count=count +1.0
        if count ==100.0:
            count =0.0
        #rospy.loginfo(send_msg) 
        #pub1.publish(send_msg)
        fusion.header.frame_id = 'fusion_track_pose1'
        fusion.header.stamp = rospy.Time.now() 
        fusion.pose.position.x= random()*60
        fusion.pose.position.y= random()*40
        fusion.pose.position.z= random()*60
        fusion.pose.orientation.x= random()*20
        fusion.pose.orientation.y= random()*60
        fusion.pose.orientation.z= random()*30
        fusion.pose.orientation.w= random()*60

 
        rospy.loginfo(fusion)
        pub.publish(fusion) 
        

 
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


