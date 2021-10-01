import sys
import os
import numpy as np
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 



img=cv2.imread(os.path.join('./812.png'),cv2.IMREAD_COLOR) 
cv2.imshow("img", img) 
cv2.waitKey(5) 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
blur = cv2.blur(gray,(3,3)) 
img_gamma = np.power(blur, 1.05).clip(0,255).astype(np.uint8) 
ret, binary = cv2.threshold(img_gamma, 180,  255, cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)  
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
dilation = cv2.dilate(closing,kernel,iterations = 1)
ROI=np.zeros([480,640],dtype=np.uint8)
ROI[50:280,220:460]=255 
masked=cv2.add(dilation, np.zeros(np.shape(binary), dtype=np.uint8), mask=ROI) 
contours = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
object_counter = len(contours)
cv2.imshow("masked", masked) 
cv2.waitKey(5000)


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


 

