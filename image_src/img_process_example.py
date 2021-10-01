import cv2
import numpy as np 
import os
 

image=cv2.imread('./Picture1.png')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
ret, binary = cv2.threshold(gray, 120,  255, cv2.THRESH_BINARY) 
# cv2.imshow("raw",image)
# cv2.waitKey(100)
rectangle = np.zeros(binary.shape[:2], dtype = "uint8")
mask = cv2.rectangle(rectangle, (30,30), (200, 200), (255,255,255), -1)
masked = cv2.bitwise_and(binary ,binary, mask =mask)

# ret, binary = cv2.threshold(masked,40,255,cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) 
closing = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)
# res =cv2.bitwise_not(closing,closing)
cv2.imwrite('./out.png', closing) 
# ret, binary = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
# # binary = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,10)
# cv2.imshow("demo",binary)
# cv2.waitKey(100)
canny = cv2.Canny(closing,50,220)

cv2.imwrite('./out2.png', canny) 