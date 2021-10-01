import rospy
from std_msgs.msg import String, Bool,  Float64MultiArray
import numpy as np
from random import seed
from random import random
import time
# seed random number generator

x = np.linspace(0,180,200)
y=np.sin(x) 
count=0
seed(1)

 
 

def main():
    rospy.init_node('fake_pub_node', anonymous=True)
    pub1 = rospy.Publisher('/cal_status', Bool, queue_size=10)
    pub2 = rospy.Publisher('/distance', String, queue_size=10)
    pub3 = rospy.Publisher('/new_pose',  Float64MultiArray, queue_size=10)
    pub4 = rospy.Publisher('/if_new_pose', Bool, queue_size=10)
    pub5 = rospy.Publisher('/fusion_tracker', Float64MultiArray, queue_size=10)
    pub6 = rospy.Publisher('/wrench_raw', Float64MultiArray, queue_size=10)
    new_pose = Float64MultiArray()
    net_W = Float64MultiArray()
    raw_W = Float64MultiArray()
    rate = rospy.Rate(2) # 10hz 
    
    while not rospy.is_shutdown():
        global count 

        if count > 40:
            send_msg = True
        else:
            send_msg = False

        count=count +1
        if count == 50:
            count =0
        rospy.loginfo(send_msg) 
        pub1.publish(send_msg)


        # send_msg2 = str(random())
        # #rospy.loginfo(send_msg2)
        # pub2.publish(send_msg2) 
        # rate.sleep()
 
        # new_pose.data = [random(),random(),random()+1,0,0,0]
        # #rospy.loginfo(new_pose)
        # pub3.publish(new_pose) 
 

        # if count%90 == 0:
        #     send_msg4 = True
        # else:
        #     send_msg4 = False

        # #rospy.loginfo(send_msg4)
        # pub4.publish(send_msg4) 

        # net_W.data = [random(),random(),random()+1,random(),random(),random()]
        # rospy.loginfo(net_W)
        # pub5.publish(net_W) 

        # raw_W.data = [random(),random()+1.9,random()+0.9,random(),random(),random()]
        # rospy.loginfo(raw_W)
        # pub6.publish(raw_W) 
        rate.sleep()
        

 
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


