import os
import cv2
import string
import math
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



 
    
img=cv2.imread('./351.png',cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
blur = cv2.blur(gray,(3,3)) 
img_gamma = np.power(blur, 1.03).clip(0,255).astype(np.uint8)
# cv2.imshow("img_gamma", img_gamma) 
# cv2.waitKey(20)
cv2.imwrite('img_gamma.png', img_gamma)
ret, binary = cv2.threshold(img_gamma, 220,  255, cv2.THRESH_BINARY)
cv2.imwrite('binary.png', binary)
kernel = np.ones((3,3),np.uint8)  
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
dilation = cv2.dilate(closing,kernel,iterations = 1)
ROI=np.zeros([480,640],dtype=np.uint8)
ROI[50:250,220:460]=255 
masked=cv2.add(dilation, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
cv2.imwrite('masked.png', masked)
canny = cv2.Canny(masked,50,220)
out, contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cv2.imwrite('canny.png', canny) 
cv2.drawContours(img,contours,-1,(0,0,255),2)  
cv2.imwrite("./img_n.png", img)



 
image_arr = np.array(canny)  
cvyrange,cvxrange = image_arr.shape
upper_contour = []
for i in range( cvxrange ): 
    for j in range(cvyrange ) :
        if image_arr[j][i]-image_arr[j-1][i] >250 : # or (x == 0 and y == 0 )  :  
            a = np.array([i,j])
            upper_contour.append(a)
            break   

x =[]
y =[]
for i in range(3, len(upper_contour)-3):            
    x.append(upper_contour[i][0])
    y.append(upper_contour[i][1])
x_m = np.mean(x)
y_m = np.mean(y) 
center_estimate = x_m, y_m
center_2, _ = optimize.leastsq(f_2, center_estimate)
xc_2, yc_2 = center_2
Ri_2       = calc_R(xc_2, yc_2)
R_2        = Ri_2.mean()   


circle_draw=cv2.circle(img,(int(xc_2), int(yc_2)),int(R_2),(0,0,255),2)

cv2.imwrite('circle_draw_color.png', circle_draw)

# if  R_2 > 94 and R_2 <100:
#     print(yc_2 , xc_2, 1-2*qy*qy-2*qz*qz,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qx,tx,2*qx*qy+2*qz*qw,1-2*qx*qx-2*qz*qz,2*qy*qz-2*qx*qw,ty,2*qx*qz-2*qy*qw,2*qy*qz+2*qx*qw,1-2*qx*qx-2*qy*qy,tz,0,0,0,1)
# print(center_2[1] +80, center_2[0]+200, 1-2*qy*qy-2*qz*qz,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qx,tx,2*qx*qy+2*qz*qw,1-2*qx*qx-2*qz*qz,2*qy*qz-2*qx*qw,ty,2*qx*qz-2*qy*qw,2*qy*qz+2*qx*qw,1-2*qx*qx-2*qy*qy,tz,0,0,0,1)


# data = np.vstack([x, y]).T 
# ransac_fit, ransac_data = ransac(data, 20, 10, 20, 20, debug=False, return_all=True)
# if ransac_fit[0] != 0:
#     x1 = ransac_fit[0] * -0.5
#     y1 = ransac_fit[1] * -0.5
#     r_1 = 0.5 * math.sqrt(ransac_fit[0] ** 2 + ransac_fit[1] ** 2 - 4 * ransac_fit[2])
#     circle3 = Circle(xy=(x1, y1), radius=r_1, alpha=0.5, fill=False, label="least square ransac fit circle") 
#     if  R_2 > 95 and R_2 <102:
#         print(y1 , x1, 1-2*qy*qy-2*qz*qz,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qx,tx,2*qx*qy+2*qz*qw,1-2*qx*qx-2*qz*qz,2*qy*qz-2*qx*qw,ty,2*qx*qz-2*qy*qw,2*qy*qz+2*qx*qw,1-2*qx*qx-2*qy*qy,tz,0,0,0,1)

