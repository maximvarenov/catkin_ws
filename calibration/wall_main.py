import numpy as np  
import csv
import random
import scipy.linalg
import math 
from numba import jit
import time
import math
import os 
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 


def show_coordinates(img,cX,cY):        # The lines in comments are for coordinates to top and left on screen, not a rectangle
    x = cX
    y = cY
    
    start_point1 = (y,0)
    end_point1 = (y,x)
    start_point2 = (0,x)
    end_point2 = (y,x)

    color = (0,0,255)
    thickness = 1
    coord_image = cv2.rectangle(img, start_point1, end_point1, color, thickness)
    coord_image = cv2.rectangle(coord_image,start_point2,end_point2,color,thickness)

    return coord_image




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



def extract_data_from_txtfile(file):
    data = open(file)
    data_content = data.read().splitlines()
    data.close()

    return data_content

  
def get_parameters(filename):
    data = extract_data_from_txtfile(filename)
    list_of_parameters = []
    for item in data:
        item = item.split()
        item = [float(i) for i in item]
        list_of_parameters.append(item)

    return list_of_parameters

def get_inverse_transformation(R, t):
    inv_rot = np.linalg.inv(R)
    inv_transl = -inv_rot.dot(t)

    return inv_rot, inv_transl


def points_to_vector(p1, p2):
    vector = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
    return vector
 


def order_coordinates(x_coord, y_coord):
    error_margin = 10
    count_x = -6
    count_y = -6
    for x in x_coord:
        for xx in x_coord:
            if abs(x - xx) < error_margin:
                count_x += 1
    for y in y_coord:
        for yy in y_coord:
            if abs(y - yy) < error_margin:
                count_y += 1
    count_x /= 2
    count_y /= 2

    if count_x > 0 and count_y == 6:
        # P1
        d_list = []
        for j in range(len(x_coord)):
            d = np.sqrt(x_coord[j] ** 2 + y_coord[j] ** 2)
            d_list.append(d)
        index_P1 = d_list.index(min(d_list))
        P1 = [x_coord[index_P1], y_coord[index_P1]]

        # P2 & P3
        p12 = []
        for j in range(len(x_coord)):
            if abs(y_coord[j] - P1[1]) < error_margin and x_coord[j] > P1[0]:
                p12.append([x_coord[j], y_coord[j]])
        if p12[0][0] < p12[1][0]:
            P2 = p12[0]
            P3 = p12[1]
        else:
            P2 = p12[1]
            P3 = p12[0]

        # P4
        P4 = [0, 0]
        for j in range(len(x_coord)):
            if abs(x_coord[j] - P1[0]) < error_margin and y_coord[j] > P1[1]:
                P4 = [x_coord[j], y_coord[j]]

        # P5 & P6
        p56 = []
        for j in range(len(x_coord)):
            if abs(y_coord[j] - P4[1]) < error_margin and x_coord[j] > P4[0]:
                p56.append([x_coord[j], y_coord[j]])
        if p56[0][0] < p56[1][0]:
            P5 = p56[0]
            P6 = p56[1]
        else:
            P5 = p56[1]
            P6 = p56[0]

    else:
        x_max = x_min = x_coord[0]
        y_max = y_min = y_coord[0]          # y-axis of image is downwards so y_max (top) is the smallest y-value!!
        index_x_max = index_x_min = index_y_max = index_y_min = 0
        for j in range(len(x_coord)):
            if x_coord[j] > x_max:
                x_max = x_coord[j]
                index_x_max = j
            if x_coord[j] < x_min:
                x_min = x_coord[j]
                index_x_min = j
        for j in range(len(y_coord)):
            if y_coord[j] < y_max:
                y_max = y_coord[j]
                index_y_max = j
            if y_coord[j] > y_min:
                y_min = y_coord[j]
                index_y_min = j

        P_top = [x_coord[index_y_max], y_coord[index_y_max]]
        P_bottom = [x_coord[index_y_min], y_coord[index_y_min]]
        P_left = [x_coord[index_x_min], y_coord[index_x_min]]
        P_right = [x_coord[index_x_max], y_coord[index_x_max]]

        d_left = np.sqrt((P_top[0] - P_left[0]) ** 2 + (P_top[1] - P_left[1]) ** 2)
        d_right = np.sqrt((P_top[0] - P_right[0]) ** 2 + (P_top[1] - P_right[1]) ** 2)

        if d_left < d_right:
            P1 = P_top
            P3 = P_right
            P4 = P_left
            P6 = P_bottom
        else:
            P1 = P_left
            P3 = P_top
            P4 = P_bottom
            P6 = P_right

        y_P5 = 0
        index_P5 = 0
        sum_of_indices = 0
        for j in range(len(y_coord)):
            if y_coord[j] > y_P5 and j not in (index_x_max, index_x_min, index_y_max, index_y_min):
                y_P5 = y_coord[j]
                index_P5 = j
            sum_of_indices += j
        index_P2 = sum_of_indices - index_P5 - index_x_max - index_x_min - index_y_max - index_y_min

        P2 = [x_coord[index_P2], y_coord[index_P2]]
        P5 = [x_coord[index_P5], y_coord[index_P5]]

    return P1, P2, P3, P4, P5, P6


def get_ratio(P1, P2, P3):
    d_12 = np.sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P1[1]) ** 2)
    d_13 = np.sqrt((P3[0] - P1[0]) ** 2 + (P3[1] - P1[1]) ** 2)
    ratio = d_12 / d_13

    return ratio




@jit  
def count_contour(background):
    cvyrange,cvxrange = background.shape
    upper_contour = []
    for i in range( cvxrange ): 
        for j in range(cvyrange ) :
            if background[j][i]-background[j-1][i] >250 : # or (x == 0 and y == 0 )  :  
                a = np.array([i,j])
                upper_contour.append(a)
                break 
    return upper_contour



def calibration(imagepoint, track_pre):
    numPoints = len(imagepoint)  # number of lines in file 
    data = imagepoint
    A = np.zeros((3* numPoints,12), dtype=float)
    b = np.zeros((3* numPoints,1), dtype=float)
    x =  np.array([0.0,0,0,0,0,0,0,0,0])
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    a =  np.array([0.0,0,0,0,0,0,0,0,0,0,0,0])
    t =  np.array([0.0,0,0])
    s =  np.array([0.0,0])
    
    for i in range(numPoints): 
        ui = data[i][1] -217   #-217 - data[i][1]   # height - cv -x -
        vi = data[i][0] -33   # width - cv -y-


        quat = np.array([track_pre[i][6],track_pre[i][3],track_pre[i][4],track_pre[i][5]]) 
        rot = quaternion_to_rotation_matrix(quat) 

        R2[0][0] = rot[0][0]
        R2[0][1] = rot[0][1]
        R2[0][2] = rot[0][2]
        R2[1][0] = rot[1][0]
        R2[1][1] = rot[1][1]
        R2[1][2] = rot[1][2]
        R2[2][0] = rot[2][0]
        R2[2][1] = rot[2][1]
        R2[2][2] = rot[2][2]
        t2x = track_pre[i][0]
        t2y = track_pre[i][1]
        t2z = track_pre[i][2]

        A[i][0] = R2[0][0]*ui 
        A[i][1] = R2[0][1]*ui
        A[i][2] = R2[0][2]*ui
        A[i][3] = R2[1][0]*ui 
        A[i][4] = R2[1][1]*ui 
        A[i][5] = R2[1][2]*ui 
        A[i][6] = R2[2][0]*ui 
        A[i][7] = R2[2][1]*ui 
        A[i][8] = R2[2][2]*ui
        A[i][9] = R2[0][0]*vi 
        A[i][10] = R2[0][1]*vi
        A[i][11] = R2[0][2]*vi
        A[i][12] = R2[1][0]*vi 
        A[i][13] = R2[1][1]*vi 
        A[i][14] = R2[1][2]*vi 
        A[i][15] = R2[2][0]*vi 
        A[i][16] = R2[2][1]*vi 
        A[i][17] = R2[2][2]*vi
        A[i][18] = R2[0][0] 
        A[i][19] = R2[0][1]
        A[i][20] = R2[0][2]
        A[i][21] = R2[1][0] 
        A[i][22] = R2[1][1] 
        A[i][23] = R2[1][2] 
        A[i][24] = R2[2][0] 
        A[i][25] = R2[2][1] 
        A[i][26] = R2[2][2]
        A[i][27] = t2x 
        A[i][28] = t2y 
        A[i][29] = t2z
        A[i][30] = 1    



def get_plane_parameter(scaling):
    B = np.array([0, 100])
    C = np.array([15, 0])
    list_of_distances = []              # list with vertical distances (in pixels) between the wires in the image
    for i in range(len(scaling)):
        x_coord = []
        y_coord = []
        for j in range(len(scaling[i])):
            if j % 2:
                y_coord.append(scaling[i][j])
            else:
                x_coord.append(scaling[i][j])
        points = order_coordinates(x_coord, y_coord)

        # middle points of top, middle and bottom wires in image coordinate system
        uv1 = np.array(points[1])
        uv2 = np.array(points[4])
        # ratios
        r1 = get_ratio(points[0], points[1], points[2])
        r2 = get_ratio(points[3], points[4], points[5])
        # x & y coordinates of middle points in phantom coordinate system
        x1_phantom = B[0] + r1 * (C[0] - B[0])
        y1_phantom = B[1] + r1 * (C[1] - B[1])
        x2_phantom = B[0] + r2 * (C[0] - B[0])
        y2_phantom = B[1] + r2 * (C[1] - B[1])
        # middle points in phantom coordinate system
        P1 = np.array([x1_phantom, y1_phantom, 0])
        P2 = np.array([x2_phantom, y2_phantom, -10])

        d_phantom = np.sqrt(np.sum((P1-P2)**2, axis=0))
        factor = 10 / d_phantom

        d = np.sqrt(np.sum((uv1-uv2)**2, axis=0))
        d *= factor
        list_of_distances.append(d)

    return list_of_distances



def object_detection(img,index):
    ## First object detection
    objects_area = []
    objects_coord = []
    objects_xcoord = []
    objects_ycoord = []
    objects_aspect_ratio = []
    objects_width = []
    objects_height = []
    coords_of_bb = []           # bb = bounding boxes 
 
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    object_counter = len(contours)

    i = 0
    area_counter = 0
    area_threshold = 15
    objects_amount = 0
    point_AR_threshold = 4

    for c in contours:     
        area = cv2.contourArea(contours[i])
        #print(area)
        objects_area.append(area)
        if area > area_threshold:                   ## Important parameter, defines the minimum size of small to dots to be ignored !!
            area_counter += 1
        i += 1
        if object_counter > area_counter:           ## Ignore small dots
            objects_amount = area_counter
        else:
            objects_amount = object_counter
    
    ## Second we check the objects 
    a = 0 
    objects_area.clear() ## remove too small values from area list by clearing it, and adding values > threshold back
    for c in contours:
        area = cv2.contourArea(contours[a]) 
        if area > area_threshold:                               
            objects_area.append(area)
        # calculate moments for each contour
            M = cv2.moments(c)
            x,y,w,h = cv2.boundingRect(c)

            coords_of_bb.extend([x,y,x+w,y+h])

        # calculate x,y coordinate of center
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(img, (cX, cY), 2, (0, 0, 255), -1)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

                
                aspect_ratio = float(w)/h
                objects_coord.extend([cX,cY])
                objects_xcoord.append(cX)
                objects_ycoord.append(cY)
                objects_aspect_ratio.append(aspect_ratio)
                objects_width.append(w)
                objects_height.append(h)
            except:
                pass
        a += 1

    
    ## Thrid we work with the information of possible multiple objects     
    one_obj = False
    two_obj = False
    bad_img = False 

    if objects_amount != 1:
        bad_img = True
        return [one_obj, two_obj, bad_img]

    else :
        one_obj = True
        if objects_area[0] >= 25:                   # Condition to ignore small dots 
            if h > 15:                              # If this condition, then object is not horizontally and we must check the orientation, we do this by checking 1 line in the boundingbox
                img_array = np.array(img)
                height_index = 1
                line_values = []
                for i in range(y+1,y+h):
                    for j in range(len(img_array[i])):
                        pixel_value = img_array[i][j] 
                         
    
                        if pixel_value  != 0:
                            line_values.append(height_index)

                        height_index += 1
                
                try:
                    avg = sum(line_values) / len(line_values)
                except:
                    # print('No avg line value detected')
                    bad_img = True
                    return [one_obj, two_obj, bad_img]

                if avg > h/2:
                    # Calculation of 2 points 
                    x1 = int(x + (0.25 * w))
                    y1 = int(y + (0.75 * h))

                    x2 = int(x + (0.75 * w))
                    y2 = int(y + (0.25 * h))

                elif avg < h/2:
                    x1 = int(x + (0.25 * w))
                    y1 = int(y + (0.25 * h))

                    x2 = int(x + (0.75 * w))
                    y2 = int(y + (0.75 * h))
                
                final_x1 = x1 - 217
                final_y1 = y1 - 33
                final_x2 = x2 - 217
                final_y2 = y2 - 33

                # y1 and x1 are reversed, different xy coordinate frame for both functions!!
                img = show_coordinates(img,y1,x1)
                img = show_coordinates(img,y2,x2)

                return [one_obj, two_obj, bad_img, final_x1, final_y1, final_x2, final_y2]

            elif h <= 15:
                x1 = int(x + (0.25 * w))
                y1 = int(y + (h/2))

                x2 = int(x + (0.75 * w))
                y2 = int(y + (h/2))

                final_x1 = x1 - 217
                final_y1 = y1 - 33
                final_x2 = x2 - 217
                final_y2 = y2 - 33


                img = show_coordinates(img,y1,x1)
                img = show_coordinates(img,y2,x2)

                return [one_obj, two_obj, bad_img, final_x1, final_y1, final_x2, final_y2]
        else:
            bad_img = True
            return [one_obj, two_obj, bad_img]
    


@jit       
def image_processing(img, image_count):
    goodImages = []
    image_point = []  
    obj = 0 
    print('image number ' ,image_count) 
 
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    blur = cv2.blur(gray,(3,3)) 
    img_gamma = np.power(blur, 1.1).clip(0,255).astype(np.uint8) 
    ret, binary = cv2.threshold(img_gamma, 180,  255, cv2.THRESH_BINARY) 
    ROI=np.zeros([480,640],dtype=np.uint8)
    ROI[150:450,217:460]=255 
    masked=cv2.add(binary, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 



    lines = cv2.HoughLinesP(masked, 1,  np.pi / 180, 10, np.array([]), 10, 50)
    background=np.zeros([480,640],dtype=np.uint8)
    background[:,:]=0
    # print(lines)
    line_count = 0
    if len(lines) != 0:
        for line in lines: 
            for x1,y1,x2,y2 in line:
                cv2.line(background,(x1,y1),(x2,y2),(255,255,255),1)
                # if line_count == 0 :
                #     image_point = [x1-217,y1-33,x2-217, y2-33] 
                # line_count += 1

    kernel = np.ones((2,2),np.uint8)  
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)
    # upper_contour = count_contour(background)
    # image_point = [upper_contour[0][0]-217,upper_contour[0][1]-33,upper_contour[len(upper_contour)-1][0]-217, upper_contour[len(upper_contour)-1][1]-33] 
    

    object_info = object_detection(background,image_count) 
    if object_info[0] == True and object_info[2] == False :
        goodImages.append(image_count) 
        # cv2.imshow('Final image',img)
        # cv2.waitKey(5)
        image_point = (object_info[3],object_info[4], object_info[5],object_info[6]) 

    return  image_point

     


time_start=time.time()
filename = "/media/ruixuan/Volume/ruixuan/Pictures/icar/wall2/cal3.txt"
file=open(filename)
lines=file.readlines()
path= '/media/ruixuan/Volume/ruixuan/Pictures/icar/wall2/cal3/'
filelist = os.listdir(path) 
sort_num_list = [] 
for file in filelist:
    sort_num_list.append(int(file.split('.png')[0]))      
    sort_num_list.sort() 
sorted_file = []
for sort_num in sort_num_list:
    for file in filelist:
        if str(sort_num) == file.split('.png')[0]:
            sorted_file.append(file)
track_pre =[]
imagepoint =[]
cal_pre =[]
image_count = 0

for i,line in zip(sorted_file,lines):  
    if  image_count> 2000:
        split_string = str.split(line,' ') 
        tx = float(split_string[0]) 
        ty = float(split_string[1])
        tz = float(split_string[2])
        qx = float(split_string[3])
        qy = float(split_string[4])
        qz = float(split_string[5])
        qw = float(split_string[6])  

        img=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR) 
        image_point = image_processing(img, image_count) 
        print(image_point)
        if image_point != None:
            imagepoint.append(image_point)
            track_pre.append([tx,ty,tz,qx,qy,qz,qw]) 
            cal_pre.append([image_point,tx,ty,tz,qx,qy,qz,qw]) 
    image_count +=1 
print('track_pre',len(track_pre))
print('imagepoint',len(imagepoint))       



wall_parameters = get_parameters('./wall_phantom/ft_data_for_second_marker_planeV2.txt')
scaling = get_parameters('./wall_phantom/point_coordinates_scale_factorsV2.txt')

list_of_distances = get_plane_parameter(scaling)

t_PW = np.array([wall_parameters[0][0], wall_parameters[0][1], wall_parameters[0][2]])
R_PW = np.array([
    [wall_parameters[0][3], wall_parameters[0][4], wall_parameters[0][5]],
    [wall_parameters[0][6], wall_parameters[0][7], wall_parameters[0][8]],
    [wall_parameters[0][9], wall_parameters[0][10], wall_parameters[0][11]]
]) 
# 3 random points on plane in phantom frame (plane equation: z = 0):
Q_P = np.array([0, 0, 0])
R_P = np.array([0, 0, 1])
S_P = np.array([0, 1, 0])
pi = math.pi
# the same points with correction because of the marker position:
rot_marker = np.array([ [math.cos(pi/2), -math.sin(pi/2),0], [math.sin(pi/2), math.cos(pi/2),0],[0,0,1]])  # rotation about the x-axis
t_marker = np.array([-100,0,-140])  #[0,0,-140]
Q_P = np.dot(rot_marker,Q_P) + t_marker
R_P = np.dot(rot_marker,R_P) + t_marker
S_P = np.dot(rot_marker,S_P) + t_marker
# same points in world frame:
Q = np.dot(R_PW,Q_P) + t_PW
R = np.dot(R_PW,R_P) + t_PW
S = np.dot(R_PW,S_P) + t_PW

t_marker1 = np.array([0, 0, 0])
t_marker2 = np.array([1, 0, 0])
vec_plane1 = np.dot(R_PW,t_marker1) 
vec_plane2 = np.dot(R_PW,t_marker2) 
vec_plane = vec_plane1 -vec_plane2
norm_vec = np.linalg.norm(vec_plane)
vec_plane = vec_plane/norm_vec
print('vec_plane',vec_plane)


row = []
row1 = []
d1 = []
 
for i in range(len(imagepoint)): 
    print(imagepoint[i])
    # slope of line on image: m = (y2-y1)/(x2-x1)
    if imagepoint[i]:
        print("list is null")
    else:
        m = ((imagepoint[i][3]) - (imagepoint[i][1])) / \
            ((imagepoint[i][2]) - (imagepoint[i][0]))


        quat = np.array([track_pre[i][6],track_pre[i][3], track_pre[i][4],track_pre[i][5]])
        t_SW = np.array([track_pre[i][0], track_pre[i][1],track_pre[i][2]])
        r_SW = quaternion_to_rotation_matrix(quat) 
        mat = np.hstack((r_SW, t_SW.reshape(-1,1)))
        homo = np.vstack((mat, np.array([0,0,0,1])))  
        r_WS, t_WS = get_inverse_transformation(r_SW, t_SW) 
       

        n1 = vec_plane[0]
        n2 = vec_plane[1]
        n3 = vec_plane[2]
        row.append([n1, n2, n3, m*n1, m*n2, m*n3])
        row1.append(vec_plane)

        Q_S = np.dot(homo,np.array([150,0,0,1])) 
        Q_S = np.array([150,0,0,1])

        d = (Q_S[0]*vec_plane[0] + Q_S[1]*vec_plane[1] + Q_S[2]*vec_plane[2])
        d1.append(d)

# A from Ax = 0, ||Ax|| should be minimized so svd is used
A = np.vstack([row])
U_svd, s, V_svd = np.linalg.svd(A)
index = min(range(len(s)), key=s.__getitem__)
X = V_svd[:, index]


Xn = X / np.linalg.norm(X[0:3])
U = Xn[0:3]
k = np.linalg.norm(Xn[3:6])         # scaling factor ratio
V = Xn[3:6] / k

Z = np.cross(U, V)
Z = Z / np.linalg.norm(Z)
Y = np.cross(Z, U)
Y = Y / np.linalg.norm(Y)

rotation = np.array([
    [U[0], Y[0], Z[0]],
    [U[1], Y[1], Z[1]],
    [U[2], Y[2], Z[2]]
])
y_im = 0
for item in list_of_distances:
    y_im += item

y_im = y_im/len(list_of_distances)      # delta y in image (vertical distance (in pixels) between the two fiducials)
y_ph = 10                               # delta y in phantom (known geometry)

# this is the axial pixel-to-millimeter ratio s_y
s_y = y_ph/y_im
s_x = s_y / k


d = np.sum(d1) / len(d1)

row2 = []
for i in range(len(imagepoint)):
    # coordinates of a (random) point on the line in the image:
    x = (imagepoint[i][0])
    y = (imagepoint[i][1])
    # normal vector of plane:
    n = row1[i]
    row2.append([d - s_x*x*(U.dot(n)) - s_y*y*(V.dot(n))])


# A & b from Ax = b, ||Ax-b|| should be minimized so the least squares method is used
A = np.vstack([row1])
b = np.vstack([row2])

translation = np.linalg.lstsq(A, b, rcond=None)[0] 
transformation = np.vstack([np.append(rotation, translation, axis=1), [0, 0, 0, 1]])


np.set_printoptions(suppress=True)
print()
print('Scaling factor x (lateral pixel-to-millimeter ratio): ', s_x)
print('Scaling factor y (axial pixel-to-millimeter ratio):   ', s_y, '\n')
print('This is the transformation matrix:\n', transformation, '\n')



# calibration(imagepoint,track_pre)

time_end=time.time()
print('time cost',time_end-time_start,'s')