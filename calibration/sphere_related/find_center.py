import cv2  as cv
import numpy as np
import os
import math
from scipy import optimize


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


def calc_R(xc, yc):
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

def f_2(c):
    Ri = calc_R(*c)
    return Ri - Ri.mean()


def GetTheAngleOfTwoPoints(x1,y1,x2,y2):
    return math.atan2(y2-y1,x2-x1)


def GetDistOfTwoPoints(x1,y1,x2,y2):
    return math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
 

def GetClosestID(p_x,p_y,pt_set):
    id = 0
    min = 10000000
    for i in range(pt_set.shape[1]):
        dist = GetDistOfTwoPoints(p_x,p_y,pt_set[0][i],pt_set[1][i])
        if dist < min:
            id = i
            min = dist
    return id

def DistOfTwoSet(set1,set2):
    loss = 0;
    for i in range(set1.shape[1]):
        id = GetClosestID(set1[0][i],set1[1][i],set2)
        dist = GetDistOfTwoPoints(set1[0][i],set1[1][i],set2[0][id],set2[1][id])
        loss = loss + dist
    return loss/set1.shape[1]
 
def ICP(sourcePoints,targetPoints,center):
    A = targetPoints 
    B = sourcePoints 
    A_ =A
    B_ =B
 
    iteration_times = 0 
    dist_now = 1 
    dist_improve = 1 
    dist_before = DistOfTwoSet(A,B) 
    while iteration_times < 10 and dist_improve > 0.001: 
        x_mean_target = A[0].mean()
        y_mean_target = A[1].mean() 
        x_mean_source = B[0].mean() 
        y_mean_source = B[1].mean() 

        for i in range(len(A)): 
            A_[i][0] = A[i][0] - x_mean_target
            A_[i][1] = A[i][1] - y_mean_target
        for j in range(len(B)):
            B_[j][0] = B[j][0] - x_mean_target
            B_[j][1] = B[j][1] - y_mean_target 
 
        w_up = 0 
        w_down = 0 
        for i in range(A_.shape[1]): 
            j = GetClosestID(A_[0][i],A_[1][i],B) 
            w_up_i = A_[0][i]*B_[1][j] - A_[1][i]*B_[0][j] 
            w_down_i = A_[0][i]*B_[0][j] + A_[1][i]*B_[1][j]
            w_up = w_up + w_up_i
            w_down = w_down + w_down_i
 
        TheRadian = math.atan2(w_up,w_down) 
        x = x_mean_target - math.cos(TheRadian)*x_mean_source - math.sin(TheRadian)*y_mean_source 
        y = x_mean_target + math.cos(TheRadian)*x_mean_source - math.sin(TheRadian)*y_mean_source 
        R = np.array([[math.cos(TheRadian),math.sin(TheRadian)],[-math.sin(TheRadian),math.cos(TheRadian)]]) 
            


        # B_matmul = np.matmul(R,B)
        for k in range(len(B)): 
            B_matmul = np.matmul(R,B[k])
            B[k][0] = B_matmul[0] + x
            B[k][1] = B_matmul[1] + y  
        print("here",np.matmul( R, center)) 
        center[0] =  np.matmul(center, R)[0] +x 
        center[1] =  np.matmul(center, R)[1] +y 

 
        iteration_times = iteration_times + 1  
        dist_now = DistOfTwoSet(A,B)  
        dist_improve = dist_before - dist_now 
        dist_before = dist_now 
 
    return B , center



def canny_demo(image):
    t = 100
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output) 
    return canny_output
 
 
img = cv.imread("./26.png")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
ROI=np.zeros([480,640],dtype=np.uint8)
ROI[50:350,220:400]=255 
masked=cv.add(binary, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
out, contours, hierarchy = cv.findContours(masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("masked", masked)
cv.waitKey(5000)
image_arr = np.array(masked) 
cvyrange,cvxrange = image_arr.shape
upper_contour = [] 
for x in range( cvxrange ): 
    for y in range(cvyrange ) :
        if image_arr[y][x]-image_arr[y-1][x] >250 : # or (x == 0 and y == 0 )  :  
            a = np.array([x,y])
            upper_contour.append(a) 
upper_contour=np.asarray(upper_contour)



upper_contour = np.asarray(upper_contour)   
input = np.asarray([upper_contour[20],upper_contour[40],upper_contour[60]]) 
print(getCircle(input)) 



x =[]
y =[]
for i in range(len(upper_contour)):            
    x.append(upper_contour[i][0])
    y.append(upper_contour[i][1])
x_m = np.mean(x)
y_m = np.mean(y) 
center_estimate = x_m, y_m
center_2, _ = optimize.leastsq(f_2, center_estimate)
print(center_2) 





target =[]
radius = 92/2
center = np.array([radius,radius])
theta = np.linspace(0, 2*np.pi,100) 
for i in range(len(theta)):
    b = np.array([radius*np.cos(theta[i])+radius, radius*np.sin(theta[i])+radius])
    target.append(b)
target = np.asarray(target)
new_circle, center = ICP(upper_contour,target, center)
print(center)

x =[]
y =[]
for i in range(len(new_circle)):            
    x.append(new_circle[i][0])
    y.append(new_circle[i][1])
x_m = np.mean(x)
y_m = np.mean(y) 
center_estimate = x_m, y_m
center_2, _ = optimize.leastsq(f_2, center_estimate)
print(center_2) 


# for c in range(len(contours)):
#     (cx, cy), (a, b), angle = cv.fitEllipse(contours[c])
#     cv.ellipse(src, (np.int32(cx), np.int32(cy)),(np.int32(a/2), np.int32(b/2)), angle, 0, 360, (0, 0, 255), 2, 8, 0)
#     print((np.int32(cx), np.int32(cy)))
 
# cv.imshow("contours_analysis", src) 
# cv.waitKey(1000)
# # cv.destroyAllWindows() 
# image_arr = np.array(binary)  
# cvyrange,cvxrange = image_arr.shape
# upper_contour = []
# for x in range(  cvxrange ): 
#     for y in range(cvyrange ) :
#         if image_arr[y][x]-image_arr[y-1][x] >250 : # or (x == 0 and y == 0 )  :  
#             a = np.array([x,y])
#             upper_contour.append(a) 
#             break
