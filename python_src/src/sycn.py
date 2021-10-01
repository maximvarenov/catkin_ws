
import rospy 
import string  
import numpy as np  
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from geometry_msgs.msg import  PoseStamped 
import cv2 as cv
import message_filters 


class Real_Time():
    def __init__(self): 
    
        self.bridge = CvBridge() 
        self.pos_sub = message_filters.Subscriber('/Fusion_track', PoseStamped )
        self.image_sub = message_filters.Subscriber('/IVUSimg', Image)  
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pos_sub], queue_size=10,slop=0.1, allow_headerless=True) 
        self.ts.registerCallback(self.callback)
        rospy.loginfo("initialization has finished........") 



    def callback(self,cv_img,pos): 
        img_index = 1
        calculation_list = []  
        
        while not rospy.is_shutdown():
            # from ros topic to translation matrix 
            
            print(type(data)) 
            img = self.bridge.imgmsg_to_cv2(cv_img, "bgr8" )
            print(type(img))   
            print("get image ", str(img_index)) 
             



def main(): 
    rospy.init_node('real_time_reconstruction')
    Real_Time()
    try:
        rospy.spin()
    except Exception as e:
        print("Exception:\n", e,"\n")

if __name__ == "__main__":
    main()


 