import os 
import sys
import string
import time
import csv
import datetime
import numpy as np 
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 



def distance2npPts(point1, point2):
    diff = point1 -point2
    dist = np.linalg.norm(diff)
    return dist


@jit
def calibration(cal_pre_list):
    print('len(cal_pre_list)   ',len(cal_pre_list))
    numPoints = len(cal_pre_list)  # number of lines in file 
    data = cal_pre_list
    A = np.zeros((3* numPoints,12), dtype=float)
    b = np.zeros((3* numPoints,1), dtype=float)
    x =  np.array([0.0,0,0,0,0,0,0,0,0])
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    a =  np.array([0.0,0,0,0,0,0,0,0,0,0,0,0])
    t =  np.array([0.0,0,0])
    s =  np.array([0.0,0])
    
    for i in range(numPoints): 
        if data[i] != None:
            ui = data[i][1] -217   #-217 - data[i][1]   # height - cv -x -
            vi = data[i][0] -33   # width - cv -y-

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

            index = 3*i
            A[index][0] = R2[0][0] *ui
            A[index][1] = R2[0][1] *ui
            A[index][2] = R2[0][2] *ui
            A[index][3] = R2[0][0] *vi 
            A[index][4] = R2[0][1] *vi  
            A[index][5] = R2[0][2] *vi  
            A[index][6] = R2[0][0]
            A[index][7] = R2[0][1]
            A[index][8] = R2[0][2]
            A[index][9] = -1.0
            A[index][10] = 0.0
            A[index][11] = 0.0
            b[index] = -t2x
            index = index +1

            A[index][0] = R2[1][0] *ui
            A[index][1] = R2[1][1] *ui
            A[index][2] = R2[1][2] *ui
            A[index][3] = R2[1][0] *vi 
            A[index][4] = R2[1][1] *vi 
            A[index][5] = R2[1][2] *vi 
            A[index][6] = R2[1][0] 
            A[index][7] = R2[1][1]
            A[index][8] = R2[1][2]
            A[index][9] = 0.0
            A[index][10] = -1.0
            A[index][11] = 0.0
            b[index] = -t2y
            index = index +1

            A[index][0] = R2[2][0]*ui
            A[index][1] = R2[2][1]*ui
            A[index][2] = R2[2][2]*ui
            A[index][3] = R2[2][0]*vi 
            A[index][4] = R2[2][1]*vi 
            A[index][5] = R2[2][2]*vi  
            A[index][6] = R2[2][0] 
            A[index][7] = R2[2][1]
            A[index][8] = R2[2][2]
            A[index][9] = 0.0
            A[index][10] = 0.0
            A[index][11] = -1.0
            b[index] = -t2z
  
  


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
    np.save("./singlepoint_python/calibration2.npy",mat)
    np.save("./singlepoint_python/scale2.npy",s)
    
    print(x[9],x[10],x[11])
    print ('trans matrix')
    print (mat)
    print ('scale')
    print (m_x,m_y)
    r = R.from_matrix(out_rotMat) 
    r.as_quat()
    print ('quaternion')
    print(r.as_quat())   # xyzw
    r.as_euler('zyx', degrees=True)
    print("as euler",r.as_euler('zyx', degrees=True))
    return mat, s




def validation(cal_pre_list,mat, s):
    numPoints = len(cal_pre_list)
    data = cal_pre_list
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    t2 = np.array([0,0,0])
    fixed_point=[] 

    error_list =[] 
    point = [69.84294933 , 291.75401864 ,1634.02993749]
    tsi = mat 
    s_x = s[1]
    s_y = s[0]
 
    for i in range(numPoints):
        ui = data[i][1]  -217   #  x -u -217
        vi = data[i][0]  -33

        R2[0][0] = data[i][2]
        R2[0][1] = data[i][3]
        R2[0][2] = data[i][4]
        R2[1][0] = data[i][6]
        R2[1][1] = data[i][7]
        R2[1][2] = data[i][8]
        R2[2][0] = data[i][10]
        R2[2][1] = data[i][11]
        R2[2][2] = data[i][12]
        t2[0] = data[i][5]
        t2[1] = data[i][9]
        t2[2] = data[i][13]
        mat = np.hstack((R2, t2.reshape(-1,1)))
        mat = np.vstack((mat, np.array([0,0,0,1])))

        i_s = s_x*ui
        j_s = s_y*vi
        uvw = np.array([i_s,j_s,0,1]) 
        res1 = np.dot(tsi, uvw)  
        res2 = np.dot(mat, res1)
        error =  np.sqrt(pow((res2[0]-point[0]),2)+pow((res2[1]-point[1]),2)+pow((res2[2]-point[2]),2))
        error_list.append(error)
        fixed_point.append(res2.tolist()) 
    fixed_point=np.asarray(fixed_point)  
    print(error_list.index(max(error_list))) 
    print(np.mean(np.asarray(error_list)))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sct, = ax.plot(fixed_point[:,0], fixed_point[:,1], fixed_point[:,2], "o", markersize=2)
    sct, = ax.plot(point[0],point[1],point[2], "o", markersize=2)
    # plt.show()

    #---------analysis----------------
    xmin = np.min(fixed_point[:,0])
    xmax = np.max(fixed_point[:,0])
    ymin = np.min(fixed_point[:,1])
    ymax = np.max(fixed_point[:,1])
    zmin = np.min(fixed_point[:,2])
    zmax = np.max(fixed_point[:,2])

    print("x-range: ", xmax-xmin)
    print("y-range: ", ymax-ymin)
    print("z-range: ", zmax-zmin)

    dislist=[]
    for i in fixed_point:
        td = distance2npPts(i[:3], point)
        dislist.append(td)
    dislist  = np.asarray(dislist)
    print(dislist.shape)
    print("max distance: ", np.max(dislist))
    print("min distance: ", np.min(dislist))
    print("mean distance: ", np.mean(dislist))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(dislist)
    plt.show()



@jit
def image_process(img,rot): 
    str_center = []
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    blur = cv2.blur(gray,(3,3))  
    img_gamma = np.power(blur, 1.05).clip(0,255).astype(np.uint8)  
    ret, binary = cv2.threshold(img_gamma, 200,  255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)  
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
    dilation = cv2.dilate(closing,kernel,iterations = 1)
    ROI=np.zeros([480,640],dtype=np.uint8)
    ROI[50:280,220:460]=255 
    masked=cv2.add(dilation, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
    contours = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    object_counter = len(contours)
    # cv2.imshow("masked", masked) 
    # cv2.waitKey(5)
    cv2.drawContours(img,contours,-1,(0,0,255),1)  
    cv2.imwrite("./img.png", img)  


    if len(contours) ==1:
        for contour in contours:
            M = cv2.moments(contour)  
            if M["m00"] == 0:
                continue
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            center_point = np.array((center_x,center_y),dtype=np.uint8)  
            cv2.circle(masked, (center_x, center_y), 1, (0,0,255), -1)
            str_center=[center_y,center_x,rot[0][0],rot[0][1],rot[0][2],x,rot[1][0],rot[1][1],rot[1][2],y,rot[2][0],rot[2][1],rot[2][2],z]

    
        return str_center  



a = []
b = []
str_list=[]
count = 0
# re_path ='/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_sphere/cal_1/'

starttime = datetime.datetime.now() 

filename = "/media/ruixuan/Volume/ruixuan/Pictures/icar/cross/cal2.txt"
file=open(filename)
lines=file.readlines()

path='/media/ruixuan/Volume/ruixuan/Pictures/icar/cross/cal2/'
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


for i,line in zip(sorted_file,lines):
    if count==1062: 
        point = False
        split_string = str.split(line,' ') 
        x = float(split_string[0]) 
        y = float(split_string[1])
        z = float(split_string[2])
        qx = float(split_string[3])
        qy = float(split_string[4])
        qz = float(split_string[5])
        qw = float(split_string[6]) 
        r = R.from_quat([qx, qy, qz, qw])
        rot = r.as_matrix()

        img=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR) 

        str_center = image_process(img,rot)
        str_list.append(str_center)

        
        # 
        # cv2.imshow("img", img) 
        # cv2.waitKey(5) 
        # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        # blur = cv2.blur(gray,(3,3)) 
        # img_gamma = np.power(blur, 1.05).clip(0,255).astype(np.uint8) 
        # ret, binary = cv2.threshold(img_gamma, 180,  255, cv2.THRESH_BINARY)
        # kernel = np.ones((3,3),np.uint8)  
        # closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
        # dilation = cv2.dilate(closing,kernel,iterations = 1)
        # ROI=np.zeros([480,640],dtype=np.uint8)
        # ROI[50:280,220:460]=255 
        # masked=cv2.add(dilation, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
        # contours = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # object_counter = len(contours)
        # cv2.imshow("masked", masked) 
        # cv2.waitKey(5)


        # if len(contours) ==1:
        #     for contour in contours:
        #         M = cv2.moments(contour)  
        #         if M["m00"] == 0:
        #             continue
        #         center_x = int(M["m10"] / M["m00"])
        #         center_y = int(M["m01"] / M["m00"])
        #         center_point = np.array((center_x,center_y),dtype=np.uint8)  
        #         cv2.circle(masked, (center_x, center_y), 1, (0,0,255), -1)
        #         str_list.append([center_y,center_x,rot[0][0],rot[0][1],rot[0][2],x,rot[1][0],rot[1][1],rot[1][2],y,rot[2][0],rot[2][1],rot[2][2],z]) 
       

    print('image number',count)
    count +=1








mat,s= calibration(str_list)
endtime = datetime.datetime.now()
print (endtime - starttime) 

validation(str_list,mat,s)