#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import rosbag
import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

AuroraTopic = '/aurora_data'
bag_file = '2021-05-06-16-30-19.bag'

fig = plt.figure()
ax1 = plt.axes(projection='3d')

x_3d = []
y_3d = []
z_3d = []

with rosbag.Bag(bag_file) as bag:
    for subtopic, msg, t in bag.read_messages(AuroraTopic):
        msg_aurora = msg.data
        # timestamp = Subcriber_aurora.timestamp
        length = len(msg_aurora)
        # print(msg_aurora.data)
        for i in range(length):
            handle = msg_aurora[i].portHandle
            x = msg_aurora[i].position.x
            y = msg_aurora[i].position.y
            z = msg_aurora[i].position.z
            qx = msg_aurora[i].orientation.x
            qy = msg_aurora[i].orientation.y
            qz = msg_aurora[i].orientation.z
            qw = msg_aurora[i].orientation.w
            if x != 0 and handle == 10:
                x_3d.append(x)
                y_3d.append(y)
                z_3d.append(z)


ax1.scatter3D(x_3d,y_3d,z_3d)  #绘制散点图

dis_x = x_3d[0]-x_3d[-1]
dis_y = y_3d[0]-y_3d[-1]
dis_z = z_3d[0]-z_3d[-1]
point_x = np.linspace(x_3d[0],x_3d[-1],1000)
point_y = np.linspace(y_3d[0],y_3d[-1],1000)
point_z = np.linspace(z_3d[0],z_3d[-1],1000)

dis = np.sqrt(dis_x**2 + dis_y**2)
ax1.plot(point_x,point_y,point_z,linewidth=1)
print(dis)
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z label')

plt.show()