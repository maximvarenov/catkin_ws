import os
import sys
import numpy as np
import csv 

def pivot_calibration(transforms):
    p_t = np.zeros((3, 1))
    T = np.eye(4)

    A = []
    b = []

    for item in transforms:
        i = 1
        A.append(np.append(item[0, [0, 1, 2]], [-1, 0, 0]))
        A.append(np.append(item[1, [0, 1, 2]], [0, -1, 0]))
        A.append(np.append(item[2, [0, 1, 2]], [0, 0, -1]))
        b.append((item[0, [3]]))
        b.append((item[1, [3]]))
        b.append((item[2, [3]]))

    x = np.linalg.lstsq(A, b, rcond=None)
    result = (x[0][0:3]).flatten() * -1
    p_t = np.asarray(result).transpose()
    T[:3, 3] = p_t.T
    return p_t, T,x



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    transforms = list()

    with open('./marker_3_pointer_pivot.txt', 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            T = np.eye(4) 
            data = np.loadtxt(row, delimiter=',')
            data = data.reshape((3, 4))
            T[:3, :4] = data
            transforms.append(T) 

    p_t, T, x = pivot_calibration(transforms)
    print(x)
    print('Calibtration matrix T')
    print(T)
    print('position of tip')
    print(-x[0][3],-x[0][4],-x[0][5])

