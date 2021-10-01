#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import sys
import csv
import numpy as np
import os
import time
import copy
import Tkinter
from Tkinter import *
import Tkinter
from tkSimpleDialog import askinteger
from tkMessageBox import askyesno
import codecs

class Getpixel():
    def __init__(self,dir):
        self.dir = dir
        self.pointinline = []
        self.imagepoint = []
        self.row = 0
        self.column = 0
        file_name = "imagepixel"
        self.csvfilepath = '{}{}.csv'.format(dir,file_name)
        if os.path.exists(self.csvfilepath):
            print(self.csvfilepath, " is exists!")
            sz = os.path.getsize(self.csvfilepath)
            if not sz:
                print(self.csvfilepath, " is empty!")
                self.create_file = True
            else:
                print(self.csvfilepath, " is not empty, size is ", sz)
                self.create_file = False
        else:
            print(self.csvfilepath, " is not exists!")
            self.create_file = True
        if self.create_file:
            fileHeader = ["name", "row", "column", "x", "y"]
            csvfile = open(self.csvfilepath, "w")
            filewriter = csv.writer(csvfile, delimiter = ',')
            filewriter.writerow(fileHeader)
            csvfile.close()



    def on_EVENT_LBUTTONDOWN(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.column == 0:
                self.gui.withdraw() #show dialog interface
                self.row = askinteger(title = "please enter the row index", prompt = "int x (0-4)：")
            xy = "%d,%d" % (x,y)
            cv2.circle(self.img,(x,y),1,(255,0,0),thickness = -1)
            if self.row == None:
                print("\n error \n error_log: row is empty.")
                sys.exit(0)
            cv2.putText(self.img, str(self.row)+', '+str(self.column), (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
            print("\n row: "+ str(self.row) + " column: "+str(self.column))
            print('\n')
            print(x,y)
            point = [self.image_name,self.row,self.column,x,y]
            self.imageinline.append(point)
            self.column += 1

    def getpixelindex(self):
        dir = self.dir
        images = os.listdir(dir)
        if self.create_file == False:
            with open(self.csvfilepath,'r') as csvfile:
                reader = csv.DictReader(csvfile)
                Processed_image = []
                for row in reader:
                    image_name = str(row['name'])
                    Processed_image.append(image_name[:-4])
            Processed_image_list = list(set(Processed_image))
            Processed_image_list.sort(key=Processed_image.index)
            print("\n already processed image : %d" %(len(Processed_image_list)))
            images_remove = copy.deepcopy(images)
            for n in Processed_image_list:
                for i in images:
                    if i[:-4]==n:
                        images_remove.remove(i)
            images = copy.deepcopy(images_remove)
        count = 0
        total = len(images)
        for image in images:
            if image[-3:] != 'jpg':
                continue
            self.image_name = image
            print("\nprocessed image : %d/%d" %(count,total))
            count += 1
            print('\nfile_name: %s.jpg'%(image[:-4]))
            self.imagepoint = []
            self.imageinline = []
            self.row = 0
            self.column = 0
            self.gui = Tkinter.Tk()  #initial GUI
            self.img = cv2.imread(dir+self.image_name)
            self.cache = copy.deepcopy(self.img)
            cv2.namedWindow('image')
            cv2.imshow('image',self.img)
            
            cv2.setMouseCallback("image",self.on_EVENT_LBUTTONDOWN)
            

            while(1):
                cv2.imshow('image',self.img)
                k = cv2.waitKey(2) & 0xFF
                if k == 120: # 'x' to change image
                    break
                if k == 27: #Escape Key
                    sys.exit(0)
                if len(self.imageinline) == 3:
                    time.sleep(0.3)
                    cv2.imshow('image',self.img)
                    self.gui.withdraw() #show dialog interface
                    re = askyesno(title='Save Row?', message='yes or no ?')
                    if re == True:
                        self.imagepoint.append(self.imageinline)
                        self.cache = copy.deepcopy(self.img)
                    else:
                        self.img = copy.deepcopy(self.cache)
                    self.column = 0
                    self.imageinline = []
    
            self.gui.withdraw() #show dialog interface
            save_file = askyesno(title='Save Data?', message='yes or no ?')
            if save_file and len(self.imagepoint):
                with open(self.csvfilepath, 'a+') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter = ',')
                    for line in self.imagepoint:
                        for data in line:
                            filewriter.writerow(data)
                    self.imagepoint = []
            else:
                self.imagepoint = []
            # close all windows
            cv2.destroyWindow("image")

if __name__ == '__main__' :
    dir = '/home/yuyu/rosbag/4-19/processed/9_4_new_image/'
    px = Getpixel(dir)
    px.getpixelindex()
    px.gui.destroy() #close GUI
