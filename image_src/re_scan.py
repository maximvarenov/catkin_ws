import cv2 as cv
import numpy as np


img = cv.imread("/media/ruixuan/Volume/ruixuan/Documents/database/us_image/29-10/img1/50.png")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
rectangle = np.zeros(gray_img.shape[0:2], dtype = "uint8")
cv.rectangle(rectangle, (220, 30), (500, 400), 255, -1)
masked = cv.bitwise_and(gray_img, gray_img, mask=rectangle)



ret, thresh = cv.threshold(masked, 50, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print(len(contours[0])) 

cv.drawContours(img, contours, -1, (0,255,0), 3)
cv.imshow("draw", img)
cv.waitKey(0)
cv.destroyAllWindows()