#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

filename = '/home/ruixuan/Desktop/imu_data/imu_ekf_ypr6.txt'
#filename2 = '/home/ruixuan/Desktop/imu_data/imu_ekf.txt'
time, time2,ax, ay, az = [], [], [], [], []
gx, gy, gz = [], [], []
mx, my, mz = [], [], []
yaw, pitch, roll = [], [], []

with open(filename, 'r') as f:

    lines = f.readlines()

    for line in lines:
        t = line.split(',')
        time.append(float(t[0]))
        ax.append(float(t[1]))
        ay.append(float(t[2]))
        az.append(float(t[3]))
        #gx.append(float(t[7]))
        #gy.append(float(t[8]))
        #gz.append(float(t[9]))
        #mx.append(float(t[10]))
        #my.append(float(t[11]))
        #mz.append(float(t[12]))
        


plt.figure()
#plt.subplot(311)
plt.plot(time, ax, 'red', label='yaw')
plt.plot(time, ay, 'blue', label='pitch')
plt.plot(time, az, 'black', label='roll')
plt.xlabel('time ')
plt.ylabel('angle')
plt.ylim(-180, 185)
#plt.title('accelerometer')
plt.legend()
plt.show()

#plt.subplot(312)
#plt.plot(time, gx, 'red', label='x')
#plt.plot(time, gy, 'blue', label='y')
#plt.plot(time, gz, 'black', label='z')
#plt.xlabel('time ')
#plt.ylabel('velocity m/s')
#plt.legend()
#plt.show()

#plt.subplot(313)
#plt.plot(time, mx, 'red', label='vx')
#plt.plot(time, my, 'blue', label='vy')
#plt.plot(time, mz, 'black', label='vz')
#plt.title(' ')
#plt.xlabel('time ')
#plt.ylabel(' tranlation m ')
#plt.legend()
#plt.show()





#n, bins, patches = plt.hist(mx, bins='auto',   rwidth=0.85)
#plt.xlabel('ax ')
#plt.show()
