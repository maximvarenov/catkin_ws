import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

X3,Y3,Z3= [],[],[]
x= []
count =1 
filename = '/media/ruixuan/Volume/ruixuan/Documents/database/force_sensor/F_test14.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        x.append(count)
        value = [float(s) for s in line.split(',')]
        X3.append(value[0])#5
        Y3.append(value[1])
        Z3.append(value[2])
        count = count +1
 




fig = plt.figure()
plt.plot(x, X3,linewidth=1, label="x axis")  
plt.plot(x, Y3,linewidth=1, label="y axis")  
plt.plot(x, Z3,linewidth=1, label="z axis") 
plt.legend(loc='upper left', bbox_to_anchor=(0.85, 0.25)) 
plt.title("force measurements")
plt.xlabel("sample number")
plt.ylabel("force [N]")
 
plt.show() 