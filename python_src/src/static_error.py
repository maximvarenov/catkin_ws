import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

average = np.array([0,0,0])
error = 0
count = 0
x = []
y = []
z = []
distance = []

filename = '/media/ruixuan/Volume/ruixuan/Documents/database/us_image/25-11/accuracy/line3.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = line.split(' ')
        x.append(float(value[0]))
        y.append(float(value[1]))
        z.append(float(value[2]))
        count = count +1

x_mean = np.mean(x)
y_mean = np.mean(y)
z_mean = np.mean(z)

for i in range(0,count):
    delta =  np.array([x[i]-x_mean,y[i]-y_mean,z[i]-z_mean])
    distance.append(np.linalg.norm(delta))

distance_mean = np.mean(distance)
distance_var = np.std(distance, ddof=1)

print(distance_mean)
print(distance_var)


