import rospy
from std_msgs.msg import String, Bool,  Float64MultiArray
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
    sub = rospy.Subscriber('/cal_status', Bool, callback)
    rospy.spin()


def callback(msg): 
    pub2 = rospy.Publisher('/distance', String, queue_size=10)
    pub3 = rospy.Publisher('/new_pose',  Float64MultiArray, queue_size=10)
    pub4 = rospy.Publisher('/if_new_pose', Bool, queue_size=10)
    # pub5 = rospy.Publisher('/Fusion_tracker', Float64MultiArray, queue_size=10)
    # pub6 = rospy.Publisher('/wrench_raw', Float64MultiArray, queue_size=10)
    pub7 = rospy.Publisher('/pivot',  Float64MultiArray, queue_size=10)
    new_pose = Float64MultiArray()
    # fusion = Float64MultiArray()
    # raw_W = Float64MultiArray()
    pivot = Float64MultiArray()
    send_msg4 = False
    rate = rospy.Rate(0.1) # 10hz 

    
    while not rospy.is_shutdown():
        global count
        count=count +1
        if count ==500:
            count =0 


        send_msg2 = str(random())
        #rospy.loginfo(send_msg2)
        pub2.publish(send_msg2) 
        rate.sleep()


        # fusion.data = [random()*60,random()*60,random()*60+1,count/100,count/100,count/100]
        # rospy.loginfo(fusion)
        # pub5.publish(fusion) 

        # raw_W.data = [random(),random()+1.1,random()+1.3,random(),random(),random()]
        # rospy.loginfo(raw_W)
        # pub6.publish(raw_W) 
         
        send_msg4 = True
        new_pose.data = [-0.5, random()/5, 0.4, 3.14, 0, 3.14]
        pivot.data = [0.1, 0.1, 0, 0.1, 0.1, 0 ,0.1, 0.1, 0, -0.1, -0.1, 0,0.1, 0.1, 0, 0.14, 0,0.14, 0]
        

        rospy.loginfo(send_msg4)
        pub4.publish(send_msg4)  
        rospy.loginfo(new_pose)
        pub3.publish(new_pose) 
        rospy.loginfo(pivot)
        pub7.publish(pivot) 


 
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


