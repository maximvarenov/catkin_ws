import rospy
import time
import string 
import glob
import numpy as np 
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from geometry_msgs.msg import  PoseStamped

from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
import sys 
import os
import cv2 as cv

 
calculation_list = []

def callback(cv_img,line,img_index):  
    roi_image = make_roi() 
    coordinatedict = {}
    object_info = []  

 
    split_string = string.split(line,' ')  
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


    img = cv.imread(cv_img) 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,img = cv.threshold(img,230,255,cv.THRESH_BINARY)
     
    img = gamma_correction(0.25,img)  
    object_info = object_detection(img, img_index)
    # cv.imshow('img', img) 
    # cv.waitKey(50)   
    if object_info[2] == False:
        coordinatedict[img_index] = [object_info[3],object_info[4]]
        coor_list = coordinatedict[img_index]
        coor_list.extend(list(homo.flatten())) 
        calculation_list.append(coor_list)  


        if img_index >1000 or len(calculation_list) >200:
            print("working")
            cross_point = np.array([-217.72477624,186.02931252,1762.05716577])
            calculation(cross_point,calculation_list)

    print('this is image ', img_index, 'good number',len(calculation_list))  




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


def make_roi ():
    roi_image = np.zeros((480,640,3), np.uint8)
    start_point = (200,40)
    end_point = (500,400)
    color = (255,255,255)
    thickness = -1
    roi_image = cv.rectangle(roi_image, start_point, end_point, color, thickness)
    return roi_image


def apply_roi(img,roi_img):
    roi  = ((img/255) * roi_img) 
    return roi


def show_roi(img,index):
    start_point = (200,40)
    end_point = (500,400)
    color = (0,0,255)
    thickness = 1
    img = cv.rectangle(img, start_point, end_point, color, thickness) 


def gamma_correction(gamma,img):
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8') 
    return gamma_corrected


def mask_gamma_correction(img):
    threshold = 240
    mask = img <= threshold 
    img[mask] = (0)
    return img


def show_coordinates(img,cX,cY):
    x = cX
    y = cY
    start_point1 = (y,0)
    end_point1 = (y,x)
    start_point2 = (0,x)
    end_point2 = (y,x)
    color = (0,0,255)
    thickness = -1
    coord_image = cv.rectangle(img, start_point1, end_point1, color, thickness)
    coord_image = cv.rectangle(coord_image,start_point2,end_point2,color,thickness) 
    return coord_image


def object_detection(img,index):
    ## First object detection
    objects_area = []
    objects_coord = []
    objects_xcoord = []
    objects_ycoord = []
    objects_aspect_ratio = []
    objects_width = []
    coords_of_bb = []           # bb = bounding boxes
    #objects_info = []

    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    binary,contours,hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    object_counter = len(contours)
    # cv.imshow('binary', binary) 
    # cv.waitKey(100)
    i = 0
    area_counter = 0
    area_threshold = 20
    objects_amount = 0
    point_AR_threshold = 4

    for c in contours:     
        area = cv.contourArea(contours[i])
        #print(area)
        objects_area.append(area)
        if area > area_threshold:                                ## Important parameter, defines the minimum size of small to dots to be ignored !!
            area_counter += 1
        i += 1
        if object_counter > area_counter:           ## Ignore small dots
            objects_amount = area_counter
        else:
            objects_amount = object_counter 


    ## Second we check the objects 
    a = 0 
    objects_area= [] ## remove too small values from area list by clearing it, and adding values > threshold back
    for c in contours:
        area = cv.contourArea(contours[a]) 
        if area > area_threshold:                               #### DOEN MET BOOLEAN IPV NOG IS AREA TE BEREKENEN ZIE VORIGE STAP
            objects_area.append(area)
        # calculate moments for each contour
            M = cv.moments(c)
            x,y,w,h = cv.boundingRect(c)

            coords_of_bb.extend([x,y,x+w,y+h])

        # calculate x,y coordinate of center
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv.circle(img, (cX, cY), 2, (0, 0, 255), -1)
                cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

                '''
                #For rotated rectangle
                cv.drawContours(img,[box],0,(0,0,255),1)
                '''
                aspect_ratio = float(w)/h
                objects_coord.extend([cX,cY])
                objects_xcoord.append(cX)
                objects_ycoord.append(cY)
                objects_aspect_ratio.append(aspect_ratio)
                objects_width.append(w)
            except:
                pass
        a += 1
    
    
    ## Thrid we work with the information of possible multiple objects     
    one_obj = False
    two_obj = False
    bad_img = False

    if objects_amount == 0:
        bad_img = True
        return [one_obj, two_obj, bad_img]
      

    if objects_amount > 2:
        amount_of_edges = 0
        for area,width,ar,xco,yco in zip(objects_area,objects_width,objects_aspect_ratio,objects_xcoord,objects_ycoord):
            #print(area, width, ar, xco, yco)
            if area > 500 and width > 100:
                objects_area.remove(area)
                objects_width.remove(width)
                objects_aspect_ratio.remove(ar)
                objects_xcoord.remove(xco)
                objects_ycoord.remove(yco)
                amount_of_edges += 1
        
        objects_amount = objects_amount - amount_of_edges

        if objects_amount == 2:
            objects_coord=[]
            objects_coord.append(objects_xcoord[0])
            objects_coord.append(objects_ycoord[0])
            objects_coord.append(objects_xcoord[1])
            objects_coord.append(objects_ycoord[1])

        #print(objects_amount)
        if objects_amount > 2:
            bad_img = True
            return [one_obj, two_obj, bad_img]

    if objects_amount == 1:
        one_obj = True
        if objects_aspect_ratio[0] <= point_AR_threshold and objects_area[0] >= 20:                     ## Voorwaarde aspect ratio 1 object
            final_cX = objects_coord[1]
            final_cY = objects_coord[0]
            img = show_coordinates(img,final_cX,final_cY)
            print(str(index) + 'one object good.png',final_cX,final_cY)
            return [one_obj, two_obj, bad_img, final_cX, final_cY]
        else:
            bad_img = True
            return [one_obj, two_obj, bad_img]

    if objects_amount == 2:
        two_obj = True
        if objects_aspect_ratio[0] >= objects_aspect_ratio[1]:
            small_obj_x = 3
            small_obj_y = 2
            small_obj_ar = objects_aspect_ratio[1]                                  ## ar = aspect ratio
            big_obj_ar = objects_aspect_ratio[0]
            big_obj_width = objects_width[0]
        else:
            small_obj_x = 1
            small_obj_y = 0
            small_obj_ar = objects_aspect_ratio[0]
            big_obj_ar = objects_aspect_ratio[1]
            big_obj_width = objects_width[1]

        delta_x = abs(objects_coord[1] - objects_coord[3])
        #print('Delta x: ' + str(delta_x))
        #print(str(big_obj_width))
        if delta_x < 50:                                   ## Als dit waar is, is delta x vrij klein en liggen beide objects ongeveer op 1 lijn
            if big_obj_width > 20:
                if big_obj_ar >= 3.5 and small_obj_ar <= point_AR_threshold:         ## Voorwarde aspect ratio's 2 objects
                    final_cX = objects_coord[small_obj_x]
                    final_cY = objects_coord[small_obj_y]
                    img = show_coordinates(img,final_cX,final_cY)
                    print(str(index) + 'teo object good.png',final_cX,final_cY)
                    return [one_obj, two_obj, bad_img, final_cX, final_cY]
                else:
                    bad_img = True
                    return [one_obj, two_obj, bad_img]
            else:
                bad_img = True
                return [one_obj, two_obj, bad_img]
        else:
            bad_img = True
            return [one_obj, two_obj, bad_img]



def calculation(cross_point,data): 
    numPoints = np.asarray(data).shape[0]
    A = np.zeros((3* numPoints,9), dtype=float)
    b = np.zeros((3* numPoints,1), dtype=float)
    x =  np.array([0.0,0,0,0,0,0,0,0,0])
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    a =  np.array([0.0,0,0,0,0,0,0,0,0])
    x =  np.array([0.0,0,0,0,0,0,0,0,0])
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    a =  np.array([0.0,0,0,0,0,0,0,0,0])
    t =  np.array([0.0,0,0])
    s =  np.array([0.0,0])


    for i in range(numPoints):
        ui = data[i][1] - 217
        vi = data[i][0] - 33 #image size 480*640

        R2[0][0] = data[i][2]
        R2[0][1] = data[i][3]
        R2[0][2] = data[i][4]
        R2[1][0] = data[i][6]
        R2[1][1] = data[i][7]
        R2[1][2] = data[i][8]
        R2[2][0] = data[i][10]
        R2[2][1] = data[i][11]
        R2[2][2] = data[i][12]
        t2x = data[i][5]
        t2y = data[i][9]
        t2z = data[i][13]
    
 
        pi_x =  cross_point[0]
        pi_y =  cross_point[1]
        pi_z =  cross_point[2]
    #############################

        index = 3*i
        A[index][0] = R2[0][0]*(ui)
        A[index][1] = R2[0][1]*(ui)
        A[index][2] = R2[0][2]*(ui)
        A[index][3] = R2[0][0]*(vi) 
        A[index][4] = R2[0][1]*(vi)  
        A[index][5] = R2[0][2]*(vi)  
        A[index][6] = R2[0][0]
        A[index][7] = R2[0][1]
        A[index][8] = R2[0][2]
        b[index] = pi_x-t2x
        index = index +1
        A[index][0] = R2[1][0]*(ui)
        A[index][1] = R2[1][1]*(ui)
        A[index][2] = R2[1][2]*(ui)
        A[index][3] = R2[1][0]*(vi)  
        A[index][4] = R2[1][1]*(vi) 
        A[index][5] = R2[1][2]*(vi)  
        A[index][6] = R2[1][0] 
        A[index][7] = R2[1][1]
        A[index][8] = R2[1][2]
        b[index] = pi_y-t2y
        index = index +1
        A[index][0] = R2[2][0]*(ui)
        A[index][1] = R2[2][1]*(ui)
        A[index][2] = R2[2][2]*(ui)
        A[index][3] = R2[2][0]*(vi) 
        A[index][4] = R2[2][1]*(vi)  
        A[index][5] = R2[2][2]*(vi) 
        A[index][6] = R2[2][0] 
        A[index][7] = R2[2][1]
        A[index][8] = R2[2][2]
        b[index] = pi_z-t2z 

    n=np.dot(np.transpose(A),A)
    m=np.dot(np.transpose(A),b)  
    x = np.dot(np.linalg.inv(n), m ) 

    r1 = np.array([0.0,0,0])    
    r2 = np.array([0.0,0,0])
    R3 = np.array([[0.0,0,0],[0.0,0,0],[0.0,0,0]])
    r1[0] = x[0]   
    r1[1] = x[1]   
    r1[2] = x[2]
    m_x = np.sqrt(r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2])
    norm = np.linalg.norm(r1)
    r1 = r1/norm
    r2[0] = x[3]  
    r2[1] = x[4]  
    r2[2] = x[5]
    m_y = np.sqrt(r2[0]*r2[0]+r2[1]*r2[1]+r2[2]*r2[2])
    norm = np.linalg.norm(r2)
    r2 = r2/norm
    r3 = np.cross(r1,r2)
    R3[0] = r1
    R3[1] = r2
    R3[2] = r3
    m_r3 = np.linalg.norm(r3)
    r3 = r3/m_r3
    R33 = np.c_[r1,r2, r3] # insert r1, r2, r3 as columns

    # get the closest (Frobenius norm) rotation matrix via SVD
    U, S, Vh = np.linalg.svd(R33)
    out_rotMat = np.dot(U, Vh) 

    t[0] = x[6]
    t[1] = x[7]
    t[2] = x[8] 

    s[0] = m_x
    s[1] = m_y
 
    mat = np.hstack((out_rotMat, t.reshape(-1,1)))
    mat = np.vstack((mat, np.array([0,0,0,1])))
    
    print ('trans matrix')
    print (mat)
    print ('scale')
    print (m_x,m_y) 

    # path = '/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_22/auto_detect_'
    # np.save(path + "calibration.npy",mat)
    # np.save(path + "scale.npy",s)




filename = "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_22/cal3.txt"
file=open(filename)
lines=file.readlines()
imagepath = os.listdir('/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_22/cal_3/') 
sort_num_list = []
for file in imagepath:
    sort_num_list.append(int(file.split('.png')[0]))      
    sort_num_list.sort() 

sorted_file = []
for sort_num in sort_num_list:
    for file in imagepath:
        if str(sort_num) == file.split('.png')[0]:
            sorted_file.append(file) 

img_index =0
count =0 
for i,line in zip(sorted_file,lines):
    if count >100:
        cv_img=str('/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_22/cal_3/'+i) 
        callback(cv_img,line,img_index)
        img_index += 1
    count += 1


    



 