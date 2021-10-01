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
    cropped = img_gray[60:120, 190:320]
    ret,thresh = cv2.threshold(cropped,110,255,cv2.THRESH_BINARY)
    thresh, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,offset=(190,60))
    cv2.drawContours(img,contours,-1,255,1,8,hierarchy)
    i = 0
    center_list = []
    dtype = [('center_x', int), ('center_y', int)]
    for contour in contours:
        M = cv2.moments(contour)  # 计算第一条轮廓的各阶矩,字典形式
        if M["m00"] == 0:
            continue
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center_point = np.array((center_x,center_y),dtype=dtype)
        
        center_list.append(center_point)
        cv2.circle(img, (center_x, center_y), 1, (0,0,255), -1)
    center_list = np.asarray(center_list)
    if center_list.shape[0] == 9:
        center_list = np.sort(center_list,order='center_x')
        column_0 = center_list[:3]
        column_1 = center_list[3:6]
        column_2 = center_list[6:]
        column_0 = np.sort(column_0,order='center_y')
        column_1 = np.sort(column_1,order='center_y')
        column_2 = np.sort(column_2,order='center_y')
        point_matrix = np.array([[[column_0[0][0],column_0[0][1]],[column_1[0][0],column_1[0][1]],[column_2[0][0],column_2[0][1]]],
                                [[column_0[1][0],column_0[1][1]],[column_1[1][0],column_1[1][1]],[column_2[1][0],column_2[1][1]]],
                                [[column_0[2][0],column_0[2][1]],[column_1[2][0],column_1[2][1]],[column_2[2][0],column_2[2][1]]]])

        # print(point_matrix)
        row_index = 0
        column_index = 0
        imageinline = []
        for m in point_matrix:
            for n in m:
                cv2.putText(img, str(row_index)+', '+str(column_index), (n[0],n[1]), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255), thickness=1)
                point = [image_name,row_index,column_index,n[0],n[1]]
                imageinline.append(point)
                column_index += 1
                if column_index==3:
                    # print(imageinline)
                    column_index = 0
                    row_index += 1
        cv2.imshow("image after segment", img)
        cv2.waitKey(0)
        return imageinline


if __name__ == "__main__":
    dir = '/home/yuyu/rosbag/5-6/processed/9_5_image/'
    file_name = "imagepixel"
    csvfilepath = '{}{}.csv'.format(dir,file_name)
    images = os.listdir(dir)
    images.sort()
    imagepoint = []
    for image in images:
        if image[-3:] == 'jpg':
            print(image)
            imageinline = segmentation(dir,image)
            if imageinline == None:
                continue
            imagepoint.append(imageinline)
    with open(csvfilepath, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ',')
        fileHeader = ["name", "row", "column", "x", "y"]
        filewriter.writerow(fileHeader)
        for line in imagepoint:
            for data in line:
                # print(line)
                filewriter.writerow(data)
    cv2.destroyAllWindows()

