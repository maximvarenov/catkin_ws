import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

X3,Y3,Z3= [],[],[]
filename = '/media/ruixuan/Volume/ruixuan/Pictures/patient1/pose11_1.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        value = [float(s) for s in line.split(' ')]
        X3.append(value[0])#5
        Y3.append(value[1])
        Z3.append(value[2])




fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X3,Y3,Z3)

ax.set_zlabel('X', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('Z', fontdict={'size': 15, 'color': 'red'})

plt.show()