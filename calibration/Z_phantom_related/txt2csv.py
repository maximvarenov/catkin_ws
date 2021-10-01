#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys, csv
import numpy as np

dir = '/home/yuyu/Documents/conference work/n'
a = 3
txt_name = 'cal' + str(a)
txt_filename = '{}/{}.txt'.format(dir,txt_name)
transformations = []
Phantom_pos_x = -237.803
Phantom_pos_y = 217.932
Phantom_pos_z = 1659.024
with open(txt_filename,'r') as f:
    index = 0
    lines=f.readlines()
    for i in lines:
        readline1 = i.strip().split(' ')
        probe_x = float(readline1[0])
        probe_y = float(readline1[1])
        probe_z = float(readline1[2])
        probe_qx = float(readline1[3])
        probe_qy = float(readline1[4])
        probe_qz = float(readline1[5])
        probe_qw = float(readline1[6])
        values = [index, probe_x, probe_y, probe_z, probe_qx, probe_qy, probe_qz, probe_qw, Phantom_pos_x, Phantom_pos_y, Phantom_pos_z]
        index += 1
        transformations.append(values)

csv_name = 'Aurora' + str(a)
csv_filename = '{}/{}.csv'.format(dir,csv_name)
header = ["timestamp", "_10_x", "_10_y", "_10_z", "_10_qx", "_10_qy", "_10_qz", "_10_qw", "_11_x", "_11_y", "_11_z"]
with open(csv_filename,"w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter = ',')
    filewriter.writerow(header)
    for t in transformations:
        filewriter.writerow(t)