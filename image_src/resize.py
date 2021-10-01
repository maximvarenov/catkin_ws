#!/usr/bin/env python
#coding:utf-8


from PIL import Image
#one pic

#img=Image.open("/home/ruixuan/Documents/unet/data/spine/test_label/4_predict.png")
#try:
#    new_img=img.resize((640,480),I mage.BILINEAR)   
#    new_img.save(os.path.join("/home/ruixuan/Documents/unet/data/spine/test_label/",os.path.basename("4_predict.png")))
#except Exception as e:
#    print(e)



import cv2
import os
import glob

# files
 
path = r'/media/ruixuan/Volume/ruixuan/Downloads/test_for_paper/*.png'
for i in glob.glob(path):
    im1 = cv2.imread(i)
    im2 = cv2.resize(im1,(640,480))
    ret,thresh = cv2.threshold(im2,50,255,cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(r'/media/ruixuan/Volume/ruixuan/Downloads/test_for_paper/',os.path.basename(i)),im2)
