#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import csv
import cv2
import numpy as np

def segmentation(dir,image_name):
    img = cv2.imread(dir+image_name)
    #将图片转为灰度图
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_gray",img_gray)
    cropped = img_gray[70:280, 240:420]
    ret,thresh = cv2.threshold(cropped,140,255,cv2.THRESH_BINARY)
    thresh, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,offset=(240,70))
    cv2.drawContours(img,contours,-1,255,1,8,hierarchy)
    cv2.imshow("image after segment", img)
    cv2.waitKey(0)



if __name__ == "__main__":
    dir = '/home/yuyu/Documents/reconstruction/2-23/9_image/'
    images = os.listdir(dir)
    images.sort()
    imagepoint = []
    for image in images:
        if image[-3:] == 'jpg':
            print(image)
            imageinline = segmentation(dir,image)
    cv2.destroyAllWindows()
