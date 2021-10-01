import rospy
from std_msgs.msg import String, Bool,  Float64MultiArray
import numpy as np
from random import seed
from random import random
import time
# seed random number generator

 
count=0 

def main():
    rospy.init_node('fake_pub_node', anonymous=True)
    pub1 = rospy.Publisher('/contact_water', Bool, queue_size=10) 
    pub2 = rospy.Publisher('/collect_data', Bool, queue_size=10) 

    contact = False
    collect = True
    rate = rospy.Rate(1) # 10hz 
    
    while not rospy.is_shutdown():
        global count 

        if count > 40:
            contact = True
        else:
            contact = False

        count=count +1
        if count == 50:
            count =0
        rospy.loginfo(contact) 
        pub1.publish(contact)
        rospy.loginfo(collect) 
        pub2.publish(collect)
        rate.sleep()
        

 
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


