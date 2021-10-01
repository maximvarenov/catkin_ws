#!/usr/bin/env python
#coding:utf-8

import cv2
import numpy as np

from PIL import Image
import os
 
file_dir = '/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_oldversion/data/spine/test/'
out_dir = '/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_oldversion/data/spine/test/'
a = os.listdir(file_dir)
 
for i in a:
    print(i)
    I = Image.open(file_dir+i)
    L = I.convert('L')
    L.save(out_dir+i)
 
