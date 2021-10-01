#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np


img_path = '/home/yuyu/rosbag/5-6/processed/9_image/1620294921485.jpg'
img = cv2.imread(img_path)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()