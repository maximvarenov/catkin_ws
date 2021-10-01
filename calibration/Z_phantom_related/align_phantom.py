#! /usr/bin/env python 
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = plt.axes(projection='3d')
points = []

fn4_x = -284.79
fn4_y = 228.474
fn4_z = 1752
fn4 = [[fn4_x],[fn4_y],[fn4_z],[1]]
fn4 = np.asarray(fn4)
points.append(fn4)

bn3_x = -237.803
bn3_y = 217.932
bn3_z = 1659.024
bn3 = [[bn3_x],[bn3_y],[bn3_z],[1]]
bn3 = np.asarray(bn3)
points.append(bn3)



bn4_x = -175.784
bn4_y = 226.5049
bn4_z = 1689.7628
bn4 = [[bn4_x],[bn4_y],[bn4_z],[1]]
bn4 = np.asarray(bn4)
points.append(bn4)



x_r = bn3_x-bn4_x
y_r = bn3_y-bn4_y



alpha = np.arctan2(x_r,y_r)
transform_z = np.array([[np.cos(alpha),-np.sin(alpha),0,0],[np.sin(alpha),np.cos(alpha),0,0],[0,0,1,0],[0,0,0,1]])
new_points = []
for n in points:
    new_p = np.dot(transform_z,n)
    new_points.append(new_p)
# new_points = np.asarray(new_points)
# ax1.scatter3D(new_points[:,0],new_points[:,1],new_points[:,2], cmap='Greens')  #plot
# ax1.text(new_points[0][0][0],new_points[0][1][0],new_points[0][2][0],'b_1_2')
# ax1.text(new_points[1][0][0],new_points[1][1][0],new_points[1][2][0],'f_1_2')
# ax1.text(new_points[2][0][0],new_points[2][1][0],new_points[2][2][0],'f_2_2')

y_4 = new_points[1][1][0]
z_4 = new_points[1][2][0]
y_7 = new_points[2][1][0]
z_7 = new_points[2][2][0]
z_r = z_4-z_7
gamma = -np.arctan2(z_r,(y_4-y_7))
transform_x = np.array([[1,0,0,0],[0,np.cos(gamma),-np.sin(gamma),0],[0,np.sin(gamma),np.cos(gamma),0],[0,0,0,1]])
new_points_1 = []
for n in new_points:
    new_p_1 = np.dot(transform_x,n)
    new_points_1.append(new_p_1)
new_points_1 = np.asarray(new_points_1)



x_4 = new_points_1[1][0][0]
z_4 = new_points_1[1][2][0]
x_5 = new_points_1[0][0][0]
z_5 = new_points_1[0][2][0]
x_r = x_5 - x_4
z_r = z_5 - z_4
beta = -np.arctan2(x_r,z_r)
transform_y = np.array([[np.cos(beta),0,np.sin(beta),0],[0,1,0,0],[-np.sin(beta),0 ,np.cos(beta),0],[0,0,0,1]])
translation = np.dot(np.dot(transform_y,np.dot(transform_x,transform_z)),points[0])

translation = np.array([[1,0,0,-translation[0][0]],[0,1,0,-translation[1][0]],[0,0,1,-translation[2][0]],[0,0,0,1]])
transformation = np.dot(translation,np.dot(transform_y,np.dot(transform_x,transform_z)))

print("\n transformation: \n[[%.8f,%.8f,%.8f,%.8f],\n[%.8f,%.8f,%.8f,%.8f],\n[%.8f,%.8f,%.8f,%.8f],\n[0         ,0         ,0         ,1         ]]\n" %
        (transformation[0,0],transformation[0,1],transformation[0,2],transformation[0,3],
        transformation[1,0],transformation[1,1],transformation[1,2],transformation[1,3],
        transformation[2,0],transformation[2,1],transformation[2,2],transformation[2,3]))
inv_transformation = np.linalg.inv(transformation)


y_point = []
a = 0
for m in points:
    new_p = np.dot(transformation,m)
    y_point.append(new_p)
    a +=1
y_point= np.asarray(y_point)
print(y_point)
ax1.scatter(y_point[:,0],y_point[:,1],y_point[:,2], cmap='Blues')  #plot
ax1.text(y_point[0][0][0],y_point[0][1][0],y_point[0][2][0],'b_1_2')
ax1.text(y_point[1][0][0],y_point[1][1][0],y_point[1][2][0],'f_1_2')
ax1.text(y_point[2][0][0],y_point[2][1][0],y_point[2][2][0],'f_2_2')


# ax1.scatter(bn3_x,bn3_y,bn3_z)
# ax1.scatter(fn4_x,fn4_y,fn4_z)
# ax1.scatter(bn4_x,bn4_y,bn4_z)

# ax1.text(bn3_x,bn3_y,bn3_z,'b_1_2')
# ax1.text(fn4_x,fn4_y,fn4_z,'f_1_2')
# ax1.text(bn4_x,bn4_y,bn4_z,'f_2_2')
ax1.set_xlabel('X Label')
ax1.set_xlim(-10,10)
ax1.set_ylabel('Y Label')
# ax1.set_ylim(-100,100)
ax1.set_zlabel('Z label')
# ax1.set_zlim(-200,-100)

plt.show()