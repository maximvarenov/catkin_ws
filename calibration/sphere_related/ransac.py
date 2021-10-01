
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np
import scipy.linalg as sl
import scipy as sp
import os
import cv2
import string
import numpy as np 
from scipy import optimize
from matplotlib.patches import Ellipse, Circle

 


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
            raise ValueError("did't meet fit acceptance criteria")
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
 
 
if __name__ == "__main__": 
    # filename = "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_sphere8/cal2.txt"
    # file=open(filename)
    # lines=file.readlines()
    # path= '/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_sphere8/cal_2/'
    # filelist = os.listdir(path) 
    # sort_num_list = [] 
    # for file in filelist:
    #     sort_num_list.append(int(file.split('.png')[0]))      
    #     sort_num_list.sort() 
    # sorted_file = []
    # for sort_num in sort_num_list:
    #     for file in filelist:
    #         if str(sort_num) == file.split('.png')[0]:
    #             sorted_file.append(file) 


    # kernel = np.ones((2,2),np.uint8)   
    # for i,line in zip(sorted_file,lines):  
    #     split_string = str.split(line,' ') 
    #     tx = float(split_string[0]) 
    #     ty = float(split_string[1])
    #     tz = float(split_string[2])
    #     qx = float(split_string[3])
    #     qy = float(split_string[4])
    #     qz = float(split_string[5])
    #     qw = float(split_string[6])  
        
    img=cv2.imread('./10.png',cv2.IMREAD_COLOR)  
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret, binary = cv2.threshold(gray, 210,  255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8) 
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing,kernel,iterations = 1)
    ROI=np.zeros([480,640],dtype=np.uint8)
    ROI[50:360,220:460]=255 
    masked=cv2.add(dilation, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
    out, contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    # cv2.imshow("masked", masked) 
    # cv2.waitKey(30)

    image_arr = np.array(masked)  
    cvyrange,cvxrange = image_arr.shape
    upper_contour = []
    for x in range( cvxrange ): 
        for y in range(cvyrange ) :
            if image_arr[y][x]-image_arr[y-1][x] >250 : # or (x == 0 and y == 0 )  :  
                a = np.array([x,y])
                upper_contour.append(a)
                break  

    if len(upper_contour) > 20: 
        points_x =[]
        points_y =[]
        for i in range(len(upper_contour)):            
            points_x.append(upper_contour[i][0])
            points_y.append(upper_contour[i][1])
        x_m = np.mean(points_x)
        y_m = np.mean(points_y) 
        # center_estimate = x_m, y_m
        # print(center_estimate)
        # print(f_2)
        # center_2, _ = optimize.leastsq(f_2, center_estimate)
        # xc_2, yc_2 = center_2
        # Ri_2       = calc_R(xc_2, yc_2)
        # R_2        = Ri_2.mean()
        # print(R_2)  
        # plt.plot(points_x, points_y, "ro", label="data points")
        # plt.axis('equal') 
        data = np.vstack([points_x, points_y]).T
        result = fit(data)
        x0 = result[0] * -0.5
        y0 = result[1] * -0.5
        r = 0.5 * math.sqrt(result[0] ** 2 + result[1] ** 2 - 4 * result[2])
        circle2 = Circle(xy=(x0, y0), radius=r, alpha=0.5, fill=False, label="least square fit circle")
        print("2222   circle x is %f, y is %f, r is %f" % (x0, y0, r)) 

        ransac_fit, ransac_data = ransac(data,  20, 1000, 20, 50, debug=False, return_all=True)
        x1 = ransac_fit[0] * -0.5
        y1 = ransac_fit[1] * -0.5
        r_1 = 0.5 * math.sqrt(ransac_fit[0] ** 2 + ransac_fit[1] ** 2 - 4 * ransac_fit[2])
        circle3 = Circle(xy=(x1, y1), radius=r_1, alpha=0.5, fill=False, label="least square ransac fit circle")
        print("333    circle x is %f, y is %f, r is %f" % (x1, y1, r_1))