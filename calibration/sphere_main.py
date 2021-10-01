import os
import string
import math
import datetime
import numpy as np 
from scipy import optimize
import time
from matplotlib.patches import Ellipse, Circle
from numba import jit
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 


def calc_R(xc, yc):
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

def f_2(c):
    Ri = calc_R(*c)
    return Ri - Ri.mean()
 
def ransac(data,  n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf 
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]  
        test_points = data[test_idxs]  
        maybemodel = fit(maybe_inliers) 
        test_err = get_error(test_points, maybemodel) 
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if debug:
            print ('test_err.min()', test_err.min())
            print ('test_err.max()', test_err.max())
            print ('numpy.mean(test_err)', np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        if len(also_inliers > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  
            bettermodel = fit(betterdata)
            better_errs = get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs)) 
        iterations += 1
        if bestfit is None:
            return np.array([0,0,0])
        if return_all:
            return bestfit, {'inliers': best_inlier_idxs}
        else:
            return bestfit
 
 
def random_partition(n, n_data):
    all_idxs = np.arange(n_data) 
    np.random.shuffle(all_idxs)  
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

 
def fit( data):
    A = []
    B = []
    for d in data:
        A.append([-d[0], -d[1], -1])
        B.append(d[0] ** 2 + d[1] ** 2)
    A_matrix = np.array(A)
    B_matrix = np.array(B)
    C_matrix = A_matrix.T.dot(A_matrix)
    result = np.linalg.inv(C_matrix).dot(A_matrix.T.dot(B_matrix))    
    return result 


def get_error( data, model):
    err_per_point = []
    for d in data:
        B = d[0] ** 2 + d[1] ** 2
        B_fit = model[0]* d[0] + model[1] * d[1] + model[2]
        err_per_point.append((B + B_fit) ** 2)  # sum squared error per row
    return np.array(err_per_point)

def gamma_correction(img, c, g):
    out = img.copy()
    out /= 255
    out = (1/c * out) ** (1/g)

    out *= 255
    out = out.astype(np.uint8)

    return out

def getCircle(input): 
    x21 = input[1][0] - input[0][0]
    y21 = input[1][1] - input[0][1]
    x32 = input[2][0] - input[1][0]
    y32 = input[2][1] - input[1][1]
    # three colinear
    if (x21 * y32 - x32 * y21 == 0):
        return None
    xy21 = input[1][0] * input[1][0] - input[0][0] * input[0][0] + input[1][1] * input[1][1] - input[0][1] * input[0][1]
    xy32 = input[2][0] * input[2][0] - input[1][0] * input[1][0] + input[2][1] * input[2][1] - input[1][1] * input[1][1]
    y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21))
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    R = ((input[0][0] - x0) ** 2 + (input[0][1] - y0) ** 2) ** 0.5
    return x0, y0, R


def canny_demo(image):
    t = 100
    canny_output = cv2.Canny(image, t, t * 2)  
    return canny_output

@jit(nopython=True)
def for_array(image_arr):
    cvyrange,cvxrange = image_arr.shape
    upper_contour = []
    for i in range( cvxrange ): 
        for j in range(cvyrange ) :
            if image_arr[j][i]-image_arr[j-1][i] >250 : # or (x == 0 and y == 0 )  :  
                a = np.array([i,j])
                upper_contour.append(a)
                break 
    return upper_contour



 
def calibration(cal_pre_list):
    numPoints = len(cal_pre_list)  # number of lines in file 
    data = cal_pre_list
    A = np.zeros((3* numPoints,12), dtype=float)
    b = np.zeros((3* numPoints,1), dtype=float)
    x =  np.array([0.0,0,0,0,0,0,0,0,0])
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    a =  np.array([0.0,0,0,0,0,0,0,0,0,0,0,0])
    t =  np.array([0.0,0,0])
    s =  np.array([0.0,0])
    print(len(cal_pre_list))
    
    for i in range(numPoints): 
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
    t[2] = x[8] + 4

    s[0] = m_x
    s[1] = m_y
 
    mat = np.hstack((out_rotMat, t.reshape(-1,1)))
    mat = np.vstack((mat, np.array([0,0,0,1])))
    np.save("/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/calibration/sphere_related/calibration9.npy",mat)
    np.save("/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/calibration/sphere_related/scale9.npy",s)
    
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

 
def distance2npPts(point1, point2):
    diff = point1 -point2
    dist = np.linalg.norm(diff)
    return dist


def validation(cal_pre_list,mat, s):
    numPoints = len(cal_pre_list)  # number of lines in file 
    data = cal_pre_list
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    t2 = np.array([0,0,0])
    fixed_point=[] 

    error_list =[] 
    point = [259.84803777 , 285.22698618, 1653.97476708]
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

 

starttime = datetime.datetime.now() 
filename = "/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere_old/cal2.txt"
path= '/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere_old/cal_2/'
file=open(filename)
lines=file.readlines()
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


count = 0
cal_pre_list =[]
for i,line in zip(sorted_file,lines): 
    if count == 351:
        split_string = str.split(line,' ') 

        tx = float(split_string[0]) 
        ty = float(split_string[1])
        tz = float(split_string[2])
        qx = float(split_string[3])
        qy = float(split_string[4])
        qz = float(split_string[5])
        qw = float(split_string[6])  
        
        img=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        blur = cv2.blur(gray,(3,3))  
        img_gamma = np.power(blur, 1.03).clip(0,255).astype(np.uint8)
        # cv2.imshow("img_gamma", img_gamma) 
        # cv2.waitKey(20)
        ret, binary = cv2.threshold(img_gamma, 220,  255, cv2.THRESH_BINARY) 
        kernel = np.ones((3,3),np.uint8)  
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
        dilation = cv2.dilate(closing,kernel,iterations = 1)
        ROI=np.zeros([480,640],dtype=np.uint8)
        ROI[50:350,220:460]=255 
        masked=cv2.add(dilation, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
        canny = cv2.Canny(masked,50,220)
        # out, contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        cv2.imshow("canny", canny) 
        cv2.waitKey(5)  
        cv2.imwrite("./canny.png", canny)  

        print("image number", count)
        if count >5 and tx !=0:
            image_arr = np.array(canny)  
            upper_contour = for_array(image_arr)

            x =[]
            y =[]
            for i in range(len(upper_contour)):            
                x.append(upper_contour[i][0])
                y.append(upper_contour[i][1])
            x_m = np.mean(x)
            y_m = np.mean(y) 
            center_estimate = x_m, y_m
            center_2, _ = optimize.leastsq(f_2, center_estimate)
            xc_2, yc_2 = center_2
            Ri_2       = calc_R(xc_2, yc_2)
            R_2        = Ri_2.mean() 
            print(R_2) 

            if  R_2 > 95 and R_2 <105:
                cal_pre = [yc_2, xc_2, 1-2*qy*qy-2*qz*qz,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qx,tx,2*qx*qy+2*qz*qw,1-2*qx*qx-2*qz*qz,2*qy*qz-2*qx*qw,ty,2*qx*qz-2*qy*qw,2*qy*qz+2*qx*qw,1-2*qx*qx-2*qy*qy,tz]
                cal_pre_list.append(cal_pre)
    count +=1



mat, s = calibration(cal_pre_list)
endtime = datetime.datetime.now()
print (endtime - starttime)  
validation(cal_pre_list,mat, s)