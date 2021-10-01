import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

tip = np.array([ 65.2411,-155.42935,898.8202938])
error = []
count = 0

filename = '/media/ruixuan/Volume/ruixuan/Documents/database/us_image/25-11/accuracy/point3.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = line.split(' ')
        error_squre = pow((tip[0]-float(value[0])),2) + pow((tip[1]-float(value[1])),2 )+ pow((tip[2]-float(value[2])),2)
        error.append(np.sqrt(error_squre))
        count = count +1
    mean = np.mean(error)
    std = np.std(error, ddof=1)
    print(mean)
    print(std)


