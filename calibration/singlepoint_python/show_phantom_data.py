#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import csv
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
xd = []
yd = []
zd = []
point_index = []
points = []
phantom_param_data_path = '/home/yuyu/rosbag/processed/phantom_param_data_backup2.csv'
new_phantom_param_data_path = '/home/yuyu/Documents/singlepoint_python/phantom_param_data_small.csv'
'''
[[ 0.56213989 -0.82704217  0.          0.        ]
 [ 0.77127821  0.52423717 -0.36097828  0.        ]
 [ 0.29854426  0.20292029  0.93257422  0.        ]
 [ 0.          0.          0.          1.        ]]

[[-0.81489647  0.56390167 -0.13400989  0.        ]
 [-0.55857574 -0.82576611 -0.07812476  0.        ]
 [-0.15471551  0.01119108  0.98789568  0.        ]
 [ 0.          0.          0.          1.        ]]

'''
with open(phantom_param_data_path,"r") as csvfile:
    filereader = csv.DictReader(csvfile)
    i = 0
    for row in filereader:
        point = []
        index = str(row['Points index'])
        x = float(row['px'])
        y = float(row['py'])
        z = float(row['pz'])
        xd.append(x)
        yd.append(y)
        zd.append(z)
        if index == '7':
            index_7 = i
            x_7 = x
            y_7 = y
        if index == '4':
            index_4 = i
            x_4 = x
            y_4 = y
        if index == '5':
            index_5 = i

        i += 1
        point = [index,x,y,z]
        point_for_rotation = np.array([[x],[y],[z],[1]])
        points.append(point_for_rotation)
        point_index.append(point)

points = np.asarray(points)
# ax1.scatter3D(points[:,0],points[:,1],points[:,2], cmap='Blues')


x_r = x_4-x_7
y_r = y_4-y_7



alpha = np.arctan2(x_r,y_r)
transform_z = np.array([[np.cos(alpha),-np.sin(alpha),0,0],[np.sin(alpha),np.cos(alpha),0,0],[0,0,1,0],[0,0,0,1]])
new_points = []
for n in points:
    new_p = np.dot(transform_z,n)
    new_points.append(new_p)
new_points = np.asarray(new_points)
# print(new_points[:,0])
# ax1.scatter3D(new_points[:,0],new_points[:,1],new_points[:,2], cmap='Greens')  #plot

y_4 = new_points[index_4][1][0]
# print(y_4)
z_4 = new_points[index_4][2][0]
y_7 = new_points[index_7][1][0]
z_7 = new_points[index_7][2][0]
z_r = z_4-z_7
gamma = -np.arctan2(z_r,(y_4-y_7))
transform_x = np.array([[1,0,0,0],[0,np.cos(gamma),-np.sin(gamma),0],[0,np.sin(gamma),np.cos(gamma),0],[0,0,0,1]])
new_points_1 = []
for n in new_points:
    new_p_1 = np.dot(transform_x,n)
    new_points_1.append(new_p_1)
new_points_1 = np.asarray(new_points_1)

x_4 = new_points_1[index_4][0][0]
z_4 = new_points_1[index_4][2][0]
x_5 = new_points_1[index_5][0][0]
z_5 = new_points_1[index_5][2][0]
x_r = x_5 - x_4
z_r = z_5 - z_4
beta = -np.arctan2(x_r,z_r)
transform_y = np.array([[np.cos(beta),0,np.sin(beta),0],[0,1,0,0],[-np.sin(beta),0 ,np.cos(beta),0],[0,0,0,1]])

# transformation = np.dot(transform_z,transform_y)
transformation = np.dot(transform_y,np.dot(transform_x,transform_z))
print(transformation)
inv_transformation = np.linalg.inv(transformation)
# print(inv_transformation)

# print(np.dot(transformation,inv_transformation))

y_point = []
save_data = []
a = 0
for m in points:
    new_p = np.dot(transformation,m)
    save_point = [point_index[a][0],new_p[0][0],new_p[1][0],new_p[2][0]]
    if point_index[a][0][0] == 'f' or point_index[a][0][0] == 'b':
        save_data.append(save_point)
        if point_index[a][0] == 'be2':
            be2_x = new_p[0][0]
            be2_y = new_p[1][0]
            be2_z = new_p[2][0]
            ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
        if point_index[a][0] == 'fk2':
            fk2_x = new_p[0][0]
            fk2_y = new_p[1][0]
            fk2_z = new_p[2][0]
            ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
        if point_index[a][0] == 'fe3':
            fe3_x = new_p[0][0]
            fe3_y = new_p[1][0]
            fe3_z = new_p[2][0]
            ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
        if point_index[a][0] == 'bk3':
            bk3_x = new_p[0][0]
            bk3_y = new_p[1][0]
            bk3_z = new_p[2][0]
            ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
        if point_index[a][0] == 'be4':
            be4_x = new_p[0][0]
            be4_y = new_p[1][0]
            be4_z = new_p[2][0]
            ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
        if point_index[a][0] == 'fk4':
            fk4_x = new_p[0][0]
            fk4_y = new_p[1][0]
            fk4_z = new_p[2][0]
            ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
    else:
        ax1.text(new_p[0][0],new_p[1][0],new_p[2][0],point_index[a][0])
    y_point.append(new_p)
    a +=1

inv_points = []
for p in y_point:
    inv_p = np.dot(inv_transformation,p)
    inv_points.append(inv_p)


phantom_point = ['fe2','ff2','fk2','fe3','fj3','fk3','fe4','ff4','fk4','be2','bj2','bk2','be3','bf3','bk3','be4','bj4','bk4']

bn2_x = be2_x
bn2_y = be2_y+5
bn2_z = be2_z
bn2 = ['bn2',bn2_x,bn2_y,bn2_z]
save_data.append(bn2)

fn2_x = fk2_x-5
fn2_y = fk2_y
fn2_z = fk2_z
fn2 = ['fn2',fn2_x,fn2_y,fn2_z]
save_data.append(fn2)

fn3_x = fe3_x
fn3_y = fe3_y+5
fn3_z = fe3_z
fn3 = ['fn3',fn3_x,fn3_y,fn3_z]
save_data.append(fn3)

bn3_x = bk3_x+5
bn3_y = bk3_y
bn3_z = bk3_z
bn3 = ['bn3',bn3_x,bn3_y,bn3_z]
save_data.append(bn3)

bn4_x = be4_x
bn4_y = be4_y+5
bn4_z = be4_z
bn4 = ['bn4',bn4_x,bn4_y,bn4_z]
save_data.append(bn4)

fn4_x = fk4_x-5
fn4_y = fk4_y
fn4_z = fk4_z
fn4 = ['fn4',fn4_x,fn4_y,fn4_z]
save_data.append(fn4)

y_point= np.asarray(y_point)
ax1.scatter3D(y_point[:,0],y_point[:,1],y_point[:,2], cmap='Reds')  #plot
inv_points = np.asarray(inv_points)
# ax1.scatter3D(inv_points[:,0],inv_points[:,1],inv_points[:,2], cmap='Greens')  #plot
ax1.scatter3D(fn2_x,fn2_y,fn2_z)
ax1.text(fn2_x,fn2_y,fn2_z,'fn')
ax1.scatter3D(fn2_x,fn2_y,fn2_z)
ax1.scatter3D(bn2_x,bn2_y,bn2_z)
ax1.scatter3D(fn3_x,fn3_y,fn3_z)
ax1.scatter3D(bn3_x,bn3_y,bn3_z)
ax1.scatter3D(fn4_x,fn4_y,fn4_z)
ax1.scatter3D(bn4_x,bn4_y,bn4_z)

# print(fe2_x,fe2_y,fe2_z)
# print(ff2_x,ff2_y,ff2_z)
# print(fe3_x,fe3_y,fe3_z)
# print(ff3_x,ff3_y,ff3_z)
# print(fe4_x,fe4_y,fe4_z)
# print(ff4_x,ff4_y,ff4_z)

fileHeader = ["Points index","px", "py", "pz"]
with open(new_phantom_param_data_path, "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter = ',')
    filewriter.writerow(fileHeader)
    for data in save_data:
        print(data[0])
        filewriter.writerow(data)

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z label')
# for i in range(len(point_index)):
#
plt.show()
