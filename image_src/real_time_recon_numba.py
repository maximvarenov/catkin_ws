import os
import time
import string
import struct
import glob
import numpy as np  
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point32, PoseStamped
import sys   
import message_filters
import ctypes 
from numba import jit
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 as cv



def quaternion_to_rotation_matrix(Q): 
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
      
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2) 
    r7 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1) 
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
      
    rot_matrix = np.array([[r00, r01, r02],
                           [r7, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

@jit(nopython=True)
def for_array(image_arr,homo,tsi,s_x,s_y): 
    cvyrange,cvxrange = image_arr.shape 
    res = []
    for x in range( cvxrange ): 
        for y in range( cvyrange ) :
            if  image_arr[y][x]-image_arr[y-1][x]  >= 255   :  
                y_s = s_y*(y - 33)     # - cv y
                x_s = s_x*(x - 217)     # - 217 # - cv x
                uvw = np.array([x_s,y_s,0,1]) 
                res1 = np.dot(tsi, uvw)  
                res2 = np.dot(homo, res1) 
                if len(res2) != 0:
                    res.append(list(res2[0:3]))  
                # break 
    return res


def onType(a): 
    pass
   


def pub(files,line,tsi,s_x,s_y):
    img = cv.imread(files)  
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
    cv.imshow("gray", gray) 
    cv.waitKey(5) 
    img_gamma = np.power(gray, 1.00).clip(0,255).astype(np.uint8)

 
    cv.namedWindow("image")
    cv.imshow("image",gray)  
    cv.createTrackbar("threshold","image", 50 ,255,onType) 
    mythreshold = cv.getTrackbarPos("threshold", "image")  
    ret, image_bin = cv.threshold(img_gamma, mythreshold, 255,  cv.THRESH_BINARY)    
    # image_bin = cv.adaptiveThreshold(image_bin,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,0)
    cv.imshow("image",image_bin) 
    cv.waitKey(5) 


    kernel = np.ones((5,5),np.uint8)  
    closing = cv.morphologyEx(image_bin, cv.MORPH_CLOSE, kernel) 
    ROI=np.zeros([480,640],dtype=np.uint8)
    ROI[100:400,220:460]=255 
    masked=cv.add(closing, np.zeros(np.shape(image_bin), dtype=np.uint8), mask=ROI) 
    canny = cv.Canny(masked,50,255)
    cv.imshow("canny", canny) 
    cv.waitKey(5)
    # binary,contours,hierarchy = cv.findContours(masked,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # res = np.zeros(img.shape[:2], np.uint8)
    # cv.drawContours(res,contours,-1,255,1,8,hierarchy)
     


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

    if x != 0 and y !=0:
        image_arr = np.array(canny)
        file2 = open("/media/ruixuan/Volume/ruixuan/Pictures/sawbone/manual_new_frame/3.txt","a")   
        res= for_array(image_arr,homo,tsi,s_x,s_y)
        for i in range(len(res)):
            count =0
            for j in range(len(res[i])):
                file2.write(str(res[i][j]))   
                if count <2  :          
                    file2.write(',') 
                    count +=1                          
            file2.write('\n')                               
        file2.close()

 

 




tsi = np.load('/media/ruixuan/Volume/ruixuan/Pictures/icar/calibration_matrix_icar.npy')
print(tsi)
s = np.load('/media/ruixuan/Volume/ruixuan/Pictures/icar/scale_icar.npy')
s_x =  s[1]
s_y =  s[0]
print(s)


filename = "/media/ruixuan/Volume/ruixuan/Pictures/sawbone/manual_new_frame/raw3.txt"
file=open(filename)
lines=file.readlines()
imagepath = os.listdir('/media/ruixuan/Volume/ruixuan/Pictures/sawbone/manual_new_frame/raw3/') 
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
    img=str('/media/ruixuan/Volume/ruixuan/Pictures/sawbone/manual_new_frame/raw3/'+i) 
    count +=1 
    if  count ==  2581 :  
        print("wotking",count)
        pub(img,line,tsi,s_x,s_y)
    


    



 