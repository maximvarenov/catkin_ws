import numpy as np
from scipy import *
import math  
import string 

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def distance2npPts(point1, point2):
    diff = point1 -point2
    dist = np.linalg.norm(diff)
    return dist

filename ='/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_sphere4/cal2.txt'  # number of lines in file
 
point = [ -168.92381065, 412.96552353 ,2519.87912441]

file=open(filename)
lines=file.readlines()

distance =[]
for line in lines:  
    split_string = string.split(line,' ') 
    tx = float(split_string[0]) 
    ty = float(split_string[1])
    tz = float(split_string[2])
    this_dis = np.sqrt((tx-point[0])*(tx-point[0])+(ty-point[1])*(ty-point[1])+(tz-point[2])*(tz-point[2]))
    distance.append(this_dis) 



print(distance) 
print(np.mean(np.asarray(distance)))
