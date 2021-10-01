# /**
#  * Solve the equation system Ax=b, where A is a 3nX9 matrix (n>=3), x is a
#  * 9x1 vector and b is a 3nx1 vector, with each data element providing the
#  * following three equations:
#  * 
#  *                               [m_x*R3(:,1)]         
#  * [u_i*R2_i  v_i*R2_i  R2_i]    [m_y*R3(:,2)]    =  [p_i-t2_i]
#  *                               [    t3     ]
#  */

import numpy as np
from scipy import *
import math  
from scipy.spatial.transform import Rotation as R



def calculation(): 
    path = './'

    numPoints = len(open(path + 'cal_pre1.txt','rb').readlines())  # number of lines in file
    data = np.loadtxt(path + 'cal_pre1.txt',delimiter=',')
    A = np.zeros((3* numPoints,12), dtype=float)
    b = np.zeros((3* numPoints,1), dtype=float)
    x =  np.array([0.0,0,0,0,0,0,0,0,0])
    R2 = np.array([[0.0,0.0,0.0],[0,0,0],[0,0,0]])
    a =  np.array([0.0,0,0,0,0,0,0,0,0,0,0,0])
    t =  np.array([0.0,0,0])
    s =  np.array([0.0,0])
    
    for i in range(numPoints):
        ui = data[i][1]  - 217   # height - cv -x -
        vi = data[i][0]  - 33    # width - cv -y-

        R2[0][0] = data[i][2]
        R2[0][1] = data[i][3]
        R2[0][2] = data[i][4]
        R2[1][0] = data[i][6]
        R2[1][1] = data[i][7]
        R2[1][2] = data[i][8]
        R2[2][0] = data[i][10]
        R2[2][1] = data[i][11]
        R2[2][2] = data[i][12]
        t2x = data[i][5]
        t2y = data[i][9]
        t2z = data[i][13]
  
 
 

        index = 3*i
        A[index][0] = R2[0][0] *ui
        A[index][1] = R2[0][1] *ui
        A[index][2] = R2[0][2] *ui
        A[index][3] = R2[0][0] *vi 
        A[index][4] = R2[0][1] *vi  
        A[index][5] = R2[0][2] *vi  
        A[index][6] = R2[0][0]
        A[index][7] = R2[0][1]
        A[index][8] = R2[0][2]
        A[index][9] = -1.0
        A[index][10] = 0.0
        A[index][11] = 0.0
        b[index] = -t2x
        index = index +1

        A[index][0] = R2[1][0] *ui
        A[index][1] = R2[1][1] *ui
        A[index][2] = R2[1][2] *ui
        A[index][3] = R2[1][0] *vi 
        A[index][4] = R2[1][1] *vi 
        A[index][5] = R2[1][2] *vi 
        A[index][6] = R2[1][0] 
        A[index][7] = R2[1][1]
        A[index][8] = R2[1][2]
        A[index][9] = 0.0
        A[index][10] = -1.0
        A[index][11] = 0.0
        b[index] = -t2y
        index = index +1

        A[index][0] = R2[2][0]*ui
        A[index][1] = R2[2][1]*ui
        A[index][2] = R2[2][2]*ui
        A[index][3] = R2[2][0]*vi 
        A[index][4] = R2[2][1]*vi 
        A[index][5] = R2[2][2]*vi  
        A[index][6] = R2[2][0] 
        A[index][7] = R2[2][1]
        A[index][8] = R2[2][2]
        A[index][9] = 0.0
        A[index][10] = 0.0
        A[index][11] = -1.0
        b[index] = -t2z
  

    n=np.dot(np.transpose(A),A) 
    m=np.dot(np.transpose(A),b) 
    x = np.dot(np.linalg.inv(n), m ) 

    r1 = np.array([0.0,0,0])    
    r2 = np.array([0.0,0,0])
    R3 = np.array([[0.0,0,0],[0.0,0,0],[0.0,0,0]])
    r1[0] = x[0]   
    r1[1] = x[1]   
    r1[2] = x[2]
    m_x = np.sqrt(r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2])
    norm = np.linalg.norm(r1) 
    r1 = r1/norm
    r2[0] = x[3]  
    r2[1] = x[4]  
    r2[2] = x[5]
    m_y = np.sqrt(r2[0]*r2[0]+r2[1]*r2[1]+r2[2]*r2[2])
    norm = np.linalg.norm(r2) 
    r2 = r2/norm
    r3 = np.cross(r1,r2)
    R3[0] = r1
    R3[1] = r2
    R3[2] = r3
    m_r3 = np.linalg.norm(r3)
    r3 = r3/m_r3
    R33 = np.c_[r1,r2, r3] # insert r1, r2, r3 as columns

    # get the closest (Frobenius norm) rotation matrix via SVD
    U, S, Vh = np.linalg.svd(R33)
    out_rotMat = np.dot(U, Vh) 

    t[0] = x[6]
    t[1] = x[7]
    t[2] = x[8] 

    s[0] = m_x
    s[1] = m_y
 
    mat = np.hstack((out_rotMat, t.reshape(-1,1)))
    mat = np.vstack((mat, np.array([0,0,0,1])))
    np.save(path + "calibration1.npy",mat)
    np.save(path + "scale1.npy",s)
    
    print(x[9],x[10],x[11])
    print ('trans matrix')
    print (mat)
    print ('scale')
    print (m_x,m_y)
    r = R.from_matrix(out_rotMat) 
    r.as_quat()
    print ('quaternion')
    print(r.as_quat())   # xyzw
    r.as_euler('zyx', degrees=True)
    print("as euler",r.as_euler('zyx', degrees=True))
 



if __name__ == "__main__":
    calculation() 
