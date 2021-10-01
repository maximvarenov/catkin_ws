
import rospy
import os
import time
import string
import struct
import glob
import numpy as np  
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point32, PoseStamped
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 as cv
import message_filters
import ctypes 



def quaternion_to_rotation_matrix(Q): 
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
      
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2) 
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1) 
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix



def pub(files,line,tsi,s_x,s_y):
    image = cv.imread(files)  
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)  
    ret, binary = cv.threshold(gray, 220,  255, cv.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)  
    closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel) 
    ROI=np.zeros([480,640],dtype=np.uint8)
    ROI[70:380,220:460]=255 
    masked=cv.add(closing, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
    canny = cv.Canny(masked,50,220)
    image_arr = np.array(canny)  
    cvyrange,cvxrange = image_arr.shape


    split_string = str.split(line,' ')  
    x = float(split_string[0])
    y = float(split_string[1])
    z = float(split_string[2])
    qx = float(split_string[3])
    qy = float(split_string[4])
    qz = float(split_string[5])
    qw = float(split_string[6]) 
    quat = np.array([qw,qx,qy,qz])
    trans = np.array([x,y,z])
    rot = quaternion_to_rotation_matrix(quat) 
    mat = np.hstack((rot, trans.reshape(-1,1)))
    homo = np.vstack((mat, np.array([0,0,0,1])))


    with open("/home/ruixuan/result.txt",'a')  as f1: 
        res = []
        s1 = "," 
        for x in range( cvxrange ): 
            for y in range( cvyrange ) :
                if image_arr[y][x]-image_arr[y-1][x] >=250   :  
                    y_s = s_y*(y - 33)     # - cv y
                    x_s = s_x*(x - 217)     # - 217 # - cv x
                    uvw = np.array([x_s,y_s,0,1]) 
                    res1 = np.dot(tsi, uvw)  
                    res2 = np.dot(homo, res1) 
                    if len(res2) != 0:
                        res.append(list(res2[0:3]))  
                    break 

           
    # with open("/media/ruixuan/Volume/ruixuan/Pictures/auto_cali3/result1.txt",'a')  as f1: 
    #     # s1 = ","
    #     # seq1 =(str(x),str(y),str(z))
    #     # res_str =s1.join(seq1)  
    #     # f1.write(res_str)      
    #     # f1.write('\n')
    #     for j in range(0,240): 
    #         for i in range(0,350) :
    #             if image_arr[i][j]-image_arr[i-1][j] >250 : #  or (i == 0 and j == 0 )  :  
    #                 i_s = s_x*i
    #                 j_s = s_y*j  
    #                 uvw = np.array([i_s,j_s,0,1]) 
    #                 res1 = np.dot(tsi, uvw)  
    #                 res2 = np.dot(homo, res1) 
    #                 s1 = ","
    #                 seq1 =(str(res2[0]),str(res2[1]),str(res2[2]))
    #                 res_str =s1.join(seq1)  
    #                 f1.write(res_str)      
    #                 f1.write('\n') 
    #                 break 



tsi = np.load('/home/ruixuan/calibration_matrix.npy')
print(tsi)
s = np.load('/home/ruixuan/scale.npy')
s_x =  s[1]
s_y =  s[0]


filename = "/media/ruixuan/Volume/ruixuan/Pictures/sawbone/raw4.txt"
file=open(filename)
lines=file.readlines()
imagepath = os.listdir('/media/ruixuan/Volume/ruixuan/Pictures/sawbone/raw4/') 
sort_num_list = []
for file in imagepath:
    sort_num_list.append(int(file.split('.png')[0]))      
    sort_num_list.sort() 

sorted_file = []
for sort_num in sort_num_list:
    for file in imagepath:
        if str(sort_num) == file.split('.png')[0]:
            sorted_file.append(file) 

count = 0
for i,line in zip(sorted_file,lines):
    img=str('/media/ruixuan/Volume/ruixuan/Pictures/sawbone/raw4/'+i) 
    count +=1 
    if  count != 0 :  #(count <380 or (count>410 and count <960) or (count>1100 and count<1430)  ) and
        print("wotking",count)
        pub(img,line,tsi,s_x,s_y)
    



 