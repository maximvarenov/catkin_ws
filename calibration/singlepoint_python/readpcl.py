#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    \n
    '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


fig = plt.figure()
ax1 = plt.axes(projection='3d')
filename = '/home/yuyu/rosbag/5-6/scan/point.txt'
PC = []

f=open(filename,'r')
lines=f.readlines()
for i in lines:
    R = np.zeros(3)
    readline1 = i.strip().split(',')
    R[0] = float(readline1[0])
    R[1] = float(readline1[1])
    R[2] = float(readline1[2])

    PC.append(R)
PC = np.asarray(PC)

f.close()
output_file = '/home/yuyu/rosbag/5-6/scan/point.ply'
one = np.ones((PC.shape[0],3))
one = np.float32(one)*255
create_output(PC, one, output_file)
# ax1.scatter3D(PC[:,0],PC[:,1],PC[:,2], cmap='Blues')
# ax1.set_xlabel('X Label')
# ax1.set_ylabel('Y Label')
# ax1.set_zlabel('Z label')
# plt.show()

