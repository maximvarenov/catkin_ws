import numpy as np 
import csv
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



x =[]
y =[]
z =[] 
x_new =[]
y_new =[]
z_new =[] 

with open('/media/ruixuan/Volume/ruixuan/Documents/etasl/ws/my_new_workspace/src/etasl_application_template/scripts/etasl/motion_models/test.txt', 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            T = np.eye(4) 
            data = np.loadtxt(row, delimiter=',')
            x.append(data[0])
            y.append(data[1])
            z.append(data[2])

number = 6
for i in range(0,len(x) -1 ):
    inc_x = (x[i+1]-x[i])/number
    inc_y = (y[i+1]-y[i])/number
    inc_z = (z[i+1]-z[i])/number
    for a in range(0,number):
        x_new.append(round(x[i]+a*inc_x ,3))
        y_new.append(round(y[i]+a*inc_y,3))
        z_new.append(round(z[i]+a*inc_z,3))


with open('/media/ruixuan/Volume/ruixuan/Documents/etasl/ws/my_new_workspace/src/etasl_application_template/scripts/etasl/motion_models/data.txt','a') as f:
    for i in range(0, len(x_new)):
        f.write(str(x_new[i]))
        f.writelines(',')
        f.write(str(y_new[i ]))
        f.writelines(',')
        f.write(str(z_new[i ]))
        f.writelines('\n')




fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
ax.scatter(x_new,y_new,z_new)
plt.show()
 
 

