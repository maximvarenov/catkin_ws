import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   



def rot2RPY(matrix):
    r = R.from_matrix(matrix)
    q = r.as_quat()
    euler = euler_from_quaternion(q)
    return euler


X4,Y4,Z4= [],[],[]
R,P,Y= [],[],[]
filename = '/media/ruixuan/Volume/ruixuan/Pictures/icar/sphere/cal3.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(' ')]
        X4.append(value[0])#5
        Y4.append(value[1])
        Z4.append(value[2]) 




fig = plt.figure()
ax = Axes3D(fig) 
ax.scatter(X4,Y4,Z4)
ax.set_zlabel('X', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('Z', fontdict={'size': 15, 'color': 'red'})
plt.show()






# x=np.arange(0,len(X4))
# l=plt.plot(x,X4,'r--',label='type1') 
# plt.plot(x,X4,'ro-') 
# plt.xlabel('row')
# plt.ylabel('column')
# plt.legend()
# plt.show()