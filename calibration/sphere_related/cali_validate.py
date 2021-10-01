import numpy as np
from scipy import *
import math  
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def distance2npPts(point1, point2):
    diff = point1 -point2
    dist = np.linalg.norm(diff)
    return dist

numPoints = len(open('/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere2/cal_pre1.txt','rb').readlines())  # number of lines in file
data = np.loadtxt('/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere2/cal_pre1.txt',delimiter=' ')
R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
t2 = np.array([0,0,0])
fixed_point=[] 
error_list =[] 
point = [-44.86817885,  306.99926517, 1757.95752548]
tsi = np.load('/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere2/calibration1.npy')
s = np.load('/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere2/scale1.npy')
s_x = s[1]
s_y = s[0]

print(tsi)
print(s_x,s_y) 
for i in range(numPoints):
    ui = data[i][1]  -217
    vi = data[i][0]  -33

    R2[0][0] = data[i][2]
    R2[0][1] = data[i][3]
    R2[0][2] = data[i][4]
    R2[1][0] = data[i][6]
    R2[1][1] = data[i][7]
    R2[1][2] = data[i][8]
    R2[2][0] = data[i][10]
    R2[2][1] = data[i][11]
    R2[2][2] = data[i][12]
    t2[0] = data[i][5]
    t2[1] = data[i][9]
    t2[2] = data[i][13]
    mat = np.hstack((R2, t2.reshape(-1,1)))
    mat = np.vstack((mat, np.array([0,0,0,1])))

    i_s = s_x*ui
    j_s = s_y*vi
    uvw = np.array([i_s,j_s,0,1]) 
    res1 = np.dot(tsi, uvw)  
    res2 = np.dot(mat, res1)
    error =  np.sqrt(pow((res2[0]-point[0]),2)+pow((res2[1]-point[1]),2)+pow((res2[2]-point[2]),2))
    error_list.append(error)
    fixed_point.append(res2.tolist()) 
fixed_point=np.asarray(fixed_point)  

with open("./data.txt","a") as f:
    f.writelines(str(error_list))
    f.writelines('\n')
print(error_list.index(max(error_list))) 
print(np.mean(np.asarray(error_list)))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sct, = ax.plot(fixed_point[:,0], fixed_point[:,1], fixed_point[:,2], "o", markersize=2)
sct, = ax.plot(point[0],point[1],point[2], "o", markersize=2)
# plt.show()

#---------analysis----------------
xmin = np.min(fixed_point[:,0])
xmax = np.max(fixed_point[:,0])
ymin = np.min(fixed_point[:,1])
ymax = np.max(fixed_point[:,1])
zmin = np.min(fixed_point[:,2])
zmax = np.max(fixed_point[:,2])

print("x-range: ", xmax-xmin)
print("y-range: ", ymax-ymin)
print("z-range: ", zmax-zmin)

dislist=[]
for i in fixed_point:
    td = distance2npPts(i[:3], point)
    dislist.append(td)
dislist  = np.asarray(dislist)
print(dislist.shape)
print("max distance: ", np.max(dislist))
print("min distance: ", np.min(dislist))
print("mean distance: ", np.mean(dislist))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(dislist)
plt.show()


    
