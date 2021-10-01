import numpy as np
import math


#  n phantom
tsi = np.load('/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/calibration/sphere_related/calibration9.npy')
print(tsi)
theta = -math.pi/2

rotation = np.array([[1,0,0,0],[0,math.cos(theta),-math.sin(theta),0],[0,math.sin(theta),math.cos(theta),0],[0,0,0,1]])
res =np.dot(rotation,tsi)
print(res)
np.save("./calibration2.npy",res)