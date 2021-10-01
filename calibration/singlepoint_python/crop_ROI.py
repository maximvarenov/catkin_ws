import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 

img_file = '/home/yuyu/rosbag/5-6/processed/9_4_image/'

images = os.listdir(img_file)
for image in images:
    if image[-3:] != 'jpg':
        continue
    img = cv2.imread(img_file+image)
    cropped = img[50:385, 80:596]
    cv2.namedWindow('image')
    cv2.imshow('image',cropped)
    cv2.imwrite(img_file+image,cropped)
    cv2.waitKey(1)


# image = '1617900784893.jpg'
# img = cv2.imread(img_file+image)
# cv2.namedWindow('image')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyWindow("image")