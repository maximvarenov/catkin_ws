
import rospy
import time
import os
import string
import struct
import glob
import numpy as np  
import atracsys.ftk as tracker_sdk
from sensor_msgs.msg import Image, PointCloud
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point32, PoseStamped
from scipy.spatial.transform import Rotation as R
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 as cv
from cv_bridge.boost.cv_bridge_boost import getCvType
from cv_bridge import CvBridge, CvBridgeError




# # Fusion Track python wrapper, now you HAVE TO put in global space. It doesn't work with others. 
#------------------------------------------------------------------------------------
def exit_with_error(error, tracking_system):
    print(error)
    errors_dict = {}
    if tracking_system.get_last_error(errors_dict) == tracker_sdk.Status.Ok:
        for level in ['errors', 'warnings', 'messages']:
            if level in errors_dict:
                print(errors_dict[level])
    exit(1)

def exit_with_error(error, tracking_system):
    print(error)
    errors_dict = {}
    if tracking_system.get_last_error(errors_dict) == tracker_sdk.Status.Ok:
        for level in ['errors', 'warnings', 'messages']:
            if level in errors_dict:
                print(errors_dict[level])
    exit(1)
print("fusionTrack Start initialization......")
path= '/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/fusion_track/fusionTrack_SDK-v4.5.2-linux64/data'
tracking_system = tracker_sdk.TrackingSystem()
frame = tracker_sdk.FrameData()

if tracking_system.initialise() != tracker_sdk.Status.Ok:
    exit_with_error("Error, can't initialise the atracsys SDK api.", tracking_system)
if tracking_system.enumerate_devices() != tracker_sdk.Status.Ok:
    exit_with_error("Error, can't enumerate devices.", tracking_system)
if tracking_system.create_frame(False, 10, 20, 20, 10) != tracker_sdk.Status.Ok:
    exit_with_error("Error, can't create frame object.", tracking_system)
geometry_path = tracking_system.get_data_option("Data Directory")

for geometry in ['geometry001.ini', 'geometry002.ini', 'geometry003.ini', 'geometry004.ini', 'geometry005.ini', 'geometry006.ini']:
    if tracking_system.set_geometry(os.path.join(path, geometry)) != tracker_sdk.Status.Ok:
        exit_with_error("Error, can't create frame object.", tracking_system)
print ("initial finish!")
#------------------------------------------------------------------------------------




class Real_Time():
    def __init__(self): 
        self.bridge = CvBridge() 
        self.header = Header() 
        self.new_pose_marker_6 =  PoseStamped()  
        self.ICount = 1
        self.time_start=time.time()
        # self.image_sub = rospy.Subscriber('/IVUSimg', Image, self.callback)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback) 
        rospy.loginfo("initialization has finished........") 

 
    def callback(self,cv_img):  
        cvimage = self.bridge.imgmsg_to_cv2(cv_img,"bgr8") 

        self.updateFrame()
        x = self.new_pose_marker_6.pose.position.x
        y = self.new_pose_marker_6.pose.position.y
        z = self.new_pose_marker_6.pose.position.z
        qx = self.new_pose_marker_6.pose.orientation.x
        qy = self.new_pose_marker_6.pose.orientation.y
        qz = self.new_pose_marker_6.pose.orientation.z
        qw = self.new_pose_marker_6.pose.orientation.w  


        path = "/media/ruixuan/Volume/ruixuan/Pictures/sawbone/manual_new_frame/raw10/"
        filename = path + str(self.ICount) + ".png"
        cv.imwrite(filename, cvimage)

        print(self.ICount)
        s1 = " "
        seq =(str(x),str(y),str(z),str(qx),str(qy),str(qz),str(qw))
        res =s1.join(seq)
        with open("/media/ruixuan/Volume/ruixuan/Pictures/sawbone/manual_new_frame/raw10.txt","a") as fw:    
            fw.write( res)    
            fw.write('\n')    

        self.ICount +=1  
        # if  self.ICount > 1300:
        #     time_end=time.time()
        #     print('time cost',time_end - self.time_start,'s')
    





    def updateFrame(self):
        self.current_marker_index = 0
        self.markers_indexes = {}  
        self.new_pose_marker_6 = PoseStamped() 
        self.ifVisible_marker_6 = False
        self.rot_6 = None
        global tracking_system
        global frame
        tracking_system.get_last_frame(frame) 

        for marker in frame.markers: 
            if not marker.geometry_id in self.markers_indexes:
                self.markers_indexes[marker.geometry_id] = self.current_marker_index
                self.current_marker_index += 1 
            if marker.geometry_id == 6 :  
                self.ifVisible_marker_6 = True
                self.rot_6 = marker.rotation
                R_quat_6 = R.from_matrix(self.rot_6)
                R_quat_6 = R_quat_6.as_quat()
                self.new_pose_marker_6.pose.position.x= marker.position[0]
                self.new_pose_marker_6.pose.position.y= marker.position[1]
                self.new_pose_marker_6.pose.position.z= marker.position[2]
                self.new_pose_marker_6.pose.orientation.x= R_quat_6[0]
                self.new_pose_marker_6.pose.orientation.y= R_quat_6[1]
                self.new_pose_marker_6.pose.orientation.z= R_quat_6[2]
                self.new_pose_marker_6.pose.orientation.w= R_quat_6[3] 
            else:
                self.ifVisible_marker_6 = False 
                print("===========  marker 6 is not visible ===========")



def main(): 
    rospy.init_node('real_time_reconstruction')
    Real_Time()
    try:
        rospy.spin()
    except Exception as e:
        print("Exception:\n", e,"\n")

if __name__ == "__main__":
    main()

