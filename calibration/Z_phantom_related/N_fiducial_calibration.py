#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import csv
import sys
import os
import random
import scipy.linalg
import math
import operator
import copy
from scipy.spatial.transform import Rotation
import Levenberg_marquardt as lm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SinglePointTargetUSCalibrationParametersEstimator():
    def __init__(self, transformationsFilePath, ImagePointsFilePath, lsType = False , minForEstimate = 32):
        self.transformationsFileName = transformationsFilePath
        self.ImagePointsFileName = ImagePointsFilePath
        self.RTT_Transformations = []
        self.ImagePoints = []
        self.phantom_param_data = []
        self.minForEstimate = minForEstimate
        self.LS_init_flag = False
        self.epsilon = 1.192092896e-07
        self.lsType = lsType
        self.Parameters = []
        self.initialParameters = np.zeros((8,1))
        self.initialParameters_more = np.zeros((9,1))
        self.finalParameters = np.zeros((12,1))
        self.distance= []
        self.distance_rms = []
        self.readfromcsv = True
        self.RANSAC_result = True
        self.phantom_origin_T = np.array([[-0.01673596,-0.99401481,0.10795583,11.50410147],
                                        [-0.98201079,-0.00396909,-0.18878306,515.22438609],
                                        [0.18808165,-0.10917326,-0.97606685,1709.94056225],
                                        [0         ,0         ,0         ,1         ]])  #phantom原点在世界坐标系中的旋转矩阵
        self.phantom_origin_invT = np.linalg.inv(self.phantom_origin_T)

    def getPointsInPhantom(self):
        # read phantom position w.r.t phantom coordinate
        phantom_param_data_path = './Zphantom_param.csv' #如果修改self.phantom_origin_T，需要注意Zphantom_param.csv中的
        phantom_area_1 = []
        phantom_area_2 = []
        with open(phantom_param_data_path,"r") as csvfile:
            filereader = csv.DictReader(csvfile)
            for row in filereader:
                index = str(row['Points_index'])
                x = float(row['px'])
                y = float(row['py'])
                z = float(row['pz'])
                phantom_list = [index,x,y,z]
                # print(phantom_list)
                if index[-2] == '1':
                    phantom_area_1.append(phantom_list)
                if index[-2] == '2':
                    phantom_area_2.append(phantom_list)

        self.phantom_param_data.append(phantom_area_1) #[b11,b12,b13,b14,b15,b16,f11,f12,f13,f14,f15,f16]
        self.phantom_param_data.append(phantom_area_2) #[b21,b22,b23,b24,b25,b26,f21,f22,f23,f24,f25,f26]

    def quaternion_to_rotation_matrix(self, quat,trans):
        q = quat.copy()
        n = np.dot(q, q)
        if n < np.finfo(q.dtype).eps:
            return np.identity(4)
        q = q * np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rot_matrix = np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], trans[0]],
            [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], trans[1]],
            [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], trans[2]],
            [0.0, 0.0, 0.0, 1.0]])
        return rot_matrix

    def loadTransformationsFromCSV(self):
        transformations = []
        with open(self.transformationsFileName,'r') as f:
            lines = csv.DictReader(f)
            for line in lines:
                timestamp = str(line['timestamp'])
                _10_x = float(line['_10_x'])
                _10_y = float(line['_10_y'])
                _10_z = float(line['_10_z'])
                _10_qx = float(line['_10_qx'])
                _10_qy = float(line['_10_qy'])
                _10_qz = float(line['_10_qz'])
                _10_qw = float(line['_10_qw'])
                _11_x = float(line['_11_x'])
                _11_y = float(line['_11_y'])
                _11_z = float(line['_11_z'])
                Trans = [timestamp, _10_x, _10_y, _10_z, _10_qx, _10_qy, _10_qz, _10_qw, _11_x, _11_y, _11_z]
                transformations.append(Trans)
        return transformations

    def loadImagePointsFromCSV(self):
        self.getPointsInPhantom()
        trans = self.loadTransformationsFromCSV()
        RTT = []
        Points = []
        with open(self.ImagePointsFileName,'r') as f:
            ImagePoints_in_column = []
            lines = csv.DictReader(f)
            i = 0
            count = 0
            for line in lines:
                i += 1
                image_name = str(line['name'])
                row = int(line['row'])
                column = int(line['column'])
                x = float(line['x'])-217
                y = float(line['y'])-33
                P = [image_name,row,column,x,y]
                ImagePoints_in_column.append(P)
                # calculate image point in world coordinate
                if i%3 == 0:
                    match = False
                    # transformation
                    for T in trans:
                        # print(image_name[:-4])
                        # print(T[0])
                        if image_name[:-4] == T[0]:
                            match = True
                            Trans_US = np.zeros((3,4))
                            Tran = np.zeros(3)
                            Quat = np.zeros(4)
                            NNN = np.zeros(9)
                            Tran[0] = T[1]
                            Tran[1] = T[2]
                            Tran[2] = T[3]
                            Quat[0] = T[4]
                            Quat[1] = T[5]
                            Quat[2] = T[6]
                            Quat[3] = T[7]
                            # Trans_US = self.quaternion_to_rotation_matrix(Quat,Tran)
                            R = Rotation.from_quat(Quat).as_matrix()
                            Trans_US[0][0] = R[0][0]
                            Trans_US[0][1] = R[0][1]
                            Trans_US[0][2] = R[0][2]
                            Trans_US[0][3] = Tran[0]
                            Trans_US[1][0] = R[1][0]
                            Trans_US[1][1] = R[1][1]
                            Trans_US[1][2] = R[1][2]
                            Trans_US[1][3] = Tran[1]
                            Trans_US[2][0] = R[2][0]
                            Trans_US[2][1] = R[2][1]
                            Trans_US[2][2] = R[2][2]
                            Trans_US[2][3] = Tran[2]

                    # point in image
                    if match == True:
                        k1 =  np.sqrt((ImagePoints_in_column[1][3] - ImagePoints_in_column[0][3])**2+ (ImagePoints_in_column[1][4] - ImagePoints_in_column[0][4])**2)
                        k2 =  np.sqrt((ImagePoints_in_column[2][3] - ImagePoints_in_column[0][3])**2+ (ImagePoints_in_column[2][4] - ImagePoints_in_column[0][4])**2)
                        a1 = k1/k2
                        if ImagePoints_in_column[0][1] == 0:
                            # point in world coordinate
                            x_kn = self.phantom_param_data[0][7][1] + a1*(self.phantom_param_data[0][0][1] - self.phantom_param_data[0][7][1]) #f12+k*(b11-f12)
                            y_kn = self.phantom_param_data[0][7][2] + a1*(self.phantom_param_data[0][0][2] - self.phantom_param_data[0][7][2])
                            z_kn = self.phantom_param_data[0][7][3] + a1*(self.phantom_param_data[0][0][3] - self.phantom_param_data[0][7][3])
                            Imagepoint_in_relative_coor = np.dot(self.phantom_origin_invT,np.array([[x_kn],[y_kn],[z_kn],[1]]))
                            point_in_real = [ImagePoints_in_column[1][3],ImagePoints_in_column[1][4],Imagepoint_in_relative_coor[0][0],Imagepoint_in_relative_coor[1][0],Imagepoint_in_relative_coor[2][0],x_kn,y_kn,z_kn]
                            Points.append(point_in_real)
                            RTT.append(Trans_US)

                        elif ImagePoints_in_column[0][1] == 1:
                            x_kn = self.phantom_param_data[0][3][1] + a1*(self.phantom_param_data[0][8][1] - self.phantom_param_data[0][3][1]) #b14+k*(f13-b14)
                            y_kn = self.phantom_param_data[0][3][2] + a1*(self.phantom_param_data[0][8][2] - self.phantom_param_data[0][3][2])
                            z_kn = self.phantom_param_data[0][3][3] + a1*(self.phantom_param_data[0][8][3] - self.phantom_param_data[0][3][3])
                            Imagepoint_in_relative_coor = np.dot(self.phantom_origin_invT,np.array([[x_kn],[y_kn],[z_kn],[1]]))
                            point_in_real = [ImagePoints_in_column[1][3],ImagePoints_in_column[1][4],Imagepoint_in_relative_coor[0][0],Imagepoint_in_relative_coor[1][0],Imagepoint_in_relative_coor[2][0],x_kn,y_kn,z_kn]
                            Points.append(point_in_real)
                            RTT.append(Trans_US)

                        elif ImagePoints_in_column[0][1] == 2:
                            x_kn = self.phantom_param_data[0][11][1] + a1*(self.phantom_param_data[0][4][1] - self.phantom_param_data[0][11][1]) #f16+k*(b15-f16)
                            y_kn = self.phantom_param_data[0][11][2] + a1*(self.phantom_param_data[0][4][2] - self.phantom_param_data[0][11][2])
                            z_kn = self.phantom_param_data[0][11][3] + a1*(self.phantom_param_data[0][4][3] - self.phantom_param_data[0][11][3])
                            Imagepoint_in_relative_coor = np.dot(self.phantom_origin_invT,np.array([[x_kn],[y_kn],[z_kn],[1]]))
                            point_in_real = [ImagePoints_in_column[1][3],ImagePoints_in_column[1][4],Imagepoint_in_relative_coor[0][0],Imagepoint_in_relative_coor[1][0],Imagepoint_in_relative_coor[2][0],x_kn,y_kn,z_kn]
                            Points.append(point_in_real)
                            RTT.append(Trans_US)

                        count += 1
                        ImagePoints_in_column = [] #clear Imagepoints_in_column
                        match = False
        self.ImagePoints = np.asarray(Points)
        print(count)
        self.RTT_Transformations = np.asarray(RTT)
        self.readfromcsv = True
        if(self.ImagePoints.size == 0): return False
        return True

    def leastSquaresEstimate(self,RTT_Transformations,ImagePoints):
        if self.lsType == False:
            self.analyticLeastSquaresEstimate(RTT_Transformations,ImagePoints)
        else:
            self.analyticLeastSquaresEstimate(RTT_Transformations,ImagePoints)
            if self.LS_init_flag == False:
                print("\n ERROR: LeastSquaresEstimate init fail \n")
                sys.exit(1)
            self.iterativeLeastSquaresEstimate(RTT_Transformations,ImagePoints)

    def analyticLeastSquaresEstimate(self,RTT_Transformations,ImagePoints):
        if RTT_Transformations.size < self.minForEstimate:
            print("\n ERROR: Transformation size is too small \n")
            sys.exit(1)

        numPoints = RTT_Transformations.shape[0]
        self.A = np.zeros((3*numPoints,9))
        x = np.zeros(9)
        self.b = np.zeros((3*numPoints,1))

        for i in range(numPoints):

            # 2D pixel coordinates
            ui = ImagePoints[i][0]
            vi = ImagePoints[i][1]
            if self.readfromcsv == True:
                self.x = ImagePoints[i][2]
                self.y = ImagePoints[i][3]
                self.z = ImagePoints[i][4]
            index = 3*i
            # first row for current data element
            self.A[index][0] = RTT_Transformations[i][0][0] * ui
            self.A[index][1] = RTT_Transformations[i][0][1] * ui
            self.A[index][2] = RTT_Transformations[i][0][2] * ui
            self.A[index][3] = RTT_Transformations[i][0][0] * vi
            self.A[index][4] = RTT_Transformations[i][0][1] * vi
            self.A[index][5] = RTT_Transformations[i][0][2] * vi
            self.A[index][6] = RTT_Transformations[i][0][0]
            self.A[index][7] = RTT_Transformations[i][0][1]
            self.A[index][8] = RTT_Transformations[i][0][2]
            self.b[index] =  self.x -RTT_Transformations[i][0][3]
            # print("\n A[index]: " + str(self.A[index]))
            index += 1
            # second row for current data element
            self.A[index][0] = RTT_Transformations[i][1][0] * ui
            self.A[index][1] = RTT_Transformations[i][1][1] * ui
            self.A[index][2] = RTT_Transformations[i][1][2] * ui
            self.A[index][3] = RTT_Transformations[i][1][0] * vi
            self.A[index][4] = RTT_Transformations[i][1][1] * vi
            self.A[index][5] = RTT_Transformations[i][1][2] * vi
            self.A[index][6] = RTT_Transformations[i][1][0]
            self.A[index][7] = RTT_Transformations[i][1][1]
            self.A[index][8] = RTT_Transformations[i][1][2]
            self.b[index] = self.y-RTT_Transformations[i][1][3]
            index += 1

            # third row for current data element
            self.A[index][0] = RTT_Transformations[i][2][0] * ui
            self.A[index][1] = RTT_Transformations[i][2][1] * ui
            self.A[index][2] = RTT_Transformations[i][2][2] * ui
            self.A[index][3] = RTT_Transformations[i][2][0] * vi
            self.A[index][4] = RTT_Transformations[i][2][1] * vi
            self.A[index][5] = RTT_Transformations[i][2][2] * vi
            self.A[index][6] = RTT_Transformations[i][2][0]
            self.A[index][7] = RTT_Transformations[i][2][1]
            self.A[index][8] = RTT_Transformations[i][2][2]
            self.b[index] = self.z-RTT_Transformations[i][2][3]


        # print("\n A.rank: " + str(np.linalg.matrix_rank(self.A)))
        Ainv = scipy.linalg.pinv2(self.A)
        # print("\n Ainv.matrix_rank: " + str(np.linalg.matrix_rank(Ainv)))
        if (np.linalg.matrix_rank(Ainv)<9): return
        x = np.dot(Ainv, self.b)

        r1 = np.zeros(3)
        r2 = np.zeros(3)
        r3 = np.zeros(3)

        r1[0] = x[0]
        r1[1] = x[1]
        r1[2] = x[2]
        mx = np.linalg.norm(r1)
        r1 /= mx

        r2[0] = x[3]
        r2[1] = x[4]
        r2[2] = x[5]
        my = np.linalg.norm(r2)
        r2 /= my

        r3 = np.cross(r1,r2)
        R3 = np.vstack((r1,r2))
        R3 = np.vstack((R3,r3))
        R3 = np.transpose(R3)
        # print("R3 before SVD: " + str(R3) + "\t\n")
        U,S,VT = np.linalg.svd(R3)
        S = np.diag(S)
        # print("S after SVD: " + str(S) + "\t\n")
        R3 = np.dot(U,VT)
        # print("R3 after SVD: " + str(R3) + "\t\n")
        smallAngle = 0.008726535498373935 #0.5 degrees
        halfPI = 1.5707963267948966192313216916398

        omega_y = np.arctan2(-R3[2][0], np.sqrt(R3[0][0]**2 + R3[1][0]**2))

        if(math.fabs(omega_y - halfPI) > smallAngle and math.fabs(omega_y + halfPI) > smallAngle):
            cy = np.cos(omega_y)
            omega_z = np.arctan2(R3[1][0]/cy, R3[0][0]/cy)
            omega_x = np.arctan2(R3[2][1]/cy, R3[2][2]/cy)
        else:
            omega_z = 0
            omega_x = np.arctan2(R3[0][1], R3[1][1])

        self.initialParameters[0] = x[6]    #t3_x
        self.initialParameters[1] = x[7]    #t3_y
        self.initialParameters[2] = x[8]    #t3_z
        self.initialParameters[3] = omega_z
        self.initialParameters[4] = omega_y
        self.initialParameters[5] = omega_x
        self.initialParameters[6] = mx
        self.initialParameters[7] = my
        if self.RANSAC_result==True:
            print("\n initialParameters: " + str(self.initialParameters))
        self.initialParameters_more[0] = mx*R3[0,0]
        self.initialParameters_more[1] = mx*R3[1,0]
        self.initialParameters_more[2] = mx*R3[2,0]
        self.initialParameters_more[3] = my*R3[0,1]
        self.initialParameters_more[4] = my*R3[1,1]
        self.initialParameters_more[5] = my*R3[2,1]
        self.initialParameters_more[6] = R3[0,2]
        self.initialParameters_more[7] = R3[1,2]
        self.initialParameters_more[8] = R3[2,2]
        # print("\n initialParameters_trans_maxtrix: " + str(R3))
        self.LS_init_flag = True

        if self.lsType == False:
            cz = np.cos(self.initialParameters[3])
            sz = np.sin(self.initialParameters[3])
            cy = np.cos(self.initialParameters[4])
            sy = np.sin(self.initialParameters[4])
            cx = np.cos(self.initialParameters[5])
            sx = np.sin(self.initialParameters[5])
            self.finalParameters[0,0] = self.initialParameters[6]*cz*cy
            self.finalParameters[1,0] = self.initialParameters[6]*sz*cy
            self.finalParameters[2,0] = self.initialParameters[6]*(-sy)
            self.finalParameters[3,0] = self.initialParameters[7]*(cz*sy*sx-sz*cx)
            self.finalParameters[4,0] = self.initialParameters[7]*(sz*sy*sx+cz*cx)
            self.finalParameters[5,0] = self.initialParameters[7]*cy*sx
            self.finalParameters[6,0] = cz*sy*cx + sz*sx
            self.finalParameters[7,0] = sz*sz*cx - cz*sx
            self.finalParameters[8,0] = cy*cx
            self.finalParameters[9,0] = self.initialParameters[0]
            self.finalParameters[10,0] = self.initialParameters[1]
            self.finalParameters[11,0] = self.initialParameters[2]
            if self.RANSAC_result==True:
                print("\n sx,sy : %f,%f" %(self.initialParameters[6],self.initialParameters[7]))
                print("\n finalParameters_trans_maxtrix: \n[[%f,%f,%f,%f],\n[%f,%f,%f,%f],\n[%f,%f,%f,%f],\n[0,0,0,1]]\n" %
                    (self.finalParameters[0,0],self.finalParameters[3,0],self.finalParameters[6,0],self.finalParameters[9,0],
                    self.finalParameters[1,0],self.finalParameters[4,0],self.finalParameters[7,0],self.finalParameters[10,0],
                    self.finalParameters[2,0],self.finalParameters[5,0],self.finalParameters[8,0],self.finalParameters[11,0]))


    def my_Func(self, params,A):
        # used in LM algorithm
        # print("\n input_data: "+ str(input_data.shape))
        t1_x = params[0,0]
        t1_y = params[1,0]
        t1_z = params[2,0]
        omega_z = params[3,0]
        omega_y = params[4,0]
        omega_x = params[5,0]
        mx = params[6,0]
        my = params[7,0]

        cz = np.cos(omega_z)
        sz = np.sin(omega_z)
        cy = np.cos(omega_y)
        sy = np.sin(omega_y)
        cx = np.cos(omega_x)
        sx = np.sin(omega_x)

        R11 = cz*cy
        R21 = sz*cy
        R31 = -sy

        R12 = cz*sy*sx-sz*cx
        R22 = sz*sy*sx+cz*cx
        R32 = cy*sx

        x = np.zeros((9,1))
        # params
        x[0][0] = R11*mx
        x[1][0] = R21*mx
        x[2][0] = R31*mx
        x[3][0] = R12*my
        x[4][0] = R22*my
        x[5][0] = R32*my
        x[6][0] = t1_x
        x[7][0] = t1_y
        x[8][0] = t1_z
        output =  np.dot(A, x)
        # print("\n output: "+ str(output.shape))
        return output

    def iterativeLeastSquaresEstimate(self,RTT_Transformations,ImagePoints):
        num_iter = 500
        # print("\n self.A: " + str(self.A.shape))
        # print("\n self.b: " + str(self.b.shape))
        numPoints = RTT_Transformations.shape[0]
        self.A = np.zeros((3*numPoints,9))
        x = np.zeros(9)
        self.b = np.zeros((3*numPoints,1))

        for i in range(numPoints):

            # 2D pixel coordinates
            ui = ImagePoints[i][0]
            vi = ImagePoints[i][1]
            if self.readfromcsv == True:
                self.x = ImagePoints[i][2]
                self.y = ImagePoints[i][3]
                self.z = ImagePoints[i][4]
            index = 3*i
            # first row for current data element
            self.A[index][0] = RTT_Transformations[i][0][0] * ui
            self.A[index][1] = RTT_Transformations[i][0][1] * ui
            self.A[index][2] = RTT_Transformations[i][0][2] * ui
            self.A[index][3] = RTT_Transformations[i][0][0] * vi
            self.A[index][4] = RTT_Transformations[i][0][1] * vi
            self.A[index][5] = RTT_Transformations[i][0][2] * vi
            self.A[index][6] = RTT_Transformations[i][0][0]
            self.A[index][7] = RTT_Transformations[i][0][1]
            self.A[index][8] = RTT_Transformations[i][0][2]
            self.b[index] =  self.x -RTT_Transformations[i][0][3]
            # print("\n A[index]: " + str(self.A[index]))
            index += 1
            # second row for current data element
            self.A[index][0] = RTT_Transformations[i][1][0] * ui
            self.A[index][1] = RTT_Transformations[i][1][1] * ui
            self.A[index][2] = RTT_Transformations[i][1][2] * ui
            self.A[index][3] = RTT_Transformations[i][1][0] * vi
            self.A[index][4] = RTT_Transformations[i][1][1] * vi
            self.A[index][5] = RTT_Transformations[i][1][2] * vi
            self.A[index][6] = RTT_Transformations[i][1][0]
            self.A[index][7] = RTT_Transformations[i][1][1]
            self.A[index][8] = RTT_Transformations[i][1][2]
            self.b[index] = self.y-RTT_Transformations[i][1][3]
            index += 1

            # third row for current data element
            self.A[index][0] = RTT_Transformations[i][2][0] * ui
            self.A[index][1] = RTT_Transformations[i][2][1] * ui
            self.A[index][2] = RTT_Transformations[i][2][2] * ui
            self.A[index][3] = RTT_Transformations[i][2][0] * vi
            self.A[index][4] = RTT_Transformations[i][2][1] * vi
            self.A[index][5] = RTT_Transformations[i][2][2] * vi
            self.A[index][6] = RTT_Transformations[i][2][0]
            self.A[index][7] = RTT_Transformations[i][2][1]
            self.A[index][8] = RTT_Transformations[i][2][2]
            self.b[index] = self.z-RTT_Transformations[i][2][3]

        LM = lm.Levenberg_marquardt(self.my_Func)
        est_params = LM.LM(num_iter,self.initialParameters,self.A,self.b)
        cz = np.cos(est_params[3])
        sz = np.sin(est_params[3])
        cy = np.cos(est_params[4])
        sy = np.sin(est_params[4])
        cx = np.cos(est_params[5])
        sx = np.sin(est_params[5])
        self.finalParameters[0,0] = est_params[6]*cz*cy
        self.finalParameters[1,0] = est_params[6]*sz*cy
        self.finalParameters[2,0] = est_params[6]*(-sy)
        self.finalParameters[3,0] = est_params[7]*(cz*sy*sx-sz*cx)
        self.finalParameters[4,0] = est_params[7]*(sz*sy*sx+cz*cx)
        self.finalParameters[5,0] = est_params[7]*cy*sx
        self.finalParameters[6,0] = cz*sy*cx + sz*sx
        self.finalParameters[7,0] = sz*sz*cx - cz*sx
        self.finalParameters[8,0] = cy*cx
        self.finalParameters[9,0] = est_params[0]
        self.finalParameters[10,0] = est_params[1]
        self.finalParameters[11,0] = est_params[2]
        if self.RANSAC_result==True:
            print("\n est_params: " + str(est_params))
            print("\n sx,sy : %f,%f" %(est_params[6],est_params[7]))
            print("\n finalParameters_trans_maxtrix: \n[[%f,%f,%f,%f],\n[%f,%f,%f,%f],\n[%f,%f,%f,%f],\n[0,0,0,1]]\n" %
                (cz*cy,(cz*sy*sx-sz*cx),self.finalParameters[6,0],self.finalParameters[9,0],
                sz*cy,(sz*sy*sx+cz*cx),self.finalParameters[7,0],self.finalParameters[10,0],
                -sy,cy*sx,self.finalParameters[8,0],self.finalParameters[11,0]))


    def getDistanceStatistics(self,RTT_Transformations,ImagePoints):
        T1 = np.zeros(4)
        T2 = np.zeros((4,4))
        T3_ini = np.zeros((4,4))
        T3_est = np.zeros((4,4))
        q = np.zeros((4,1))
        qInT_ini = np.zeros((4,1))
        qInT_est = np.zeros((4,1))
        T1_after = []
        T1_before = []
        T2_after = []
        real_world = []
        fig = plt.figure(1)
        ax1 = plt.axes(projection='3d')
        T3_ini[0,0] = self.initialParameters_more[0]
        T3_ini[1,0] = self.initialParameters_more[1]
        T3_ini[2,0] = self.initialParameters_more[2]
        T3_ini[3,0] = 0.0
        T3_ini[0,1] = self.initialParameters_more[3]
        T3_ini[1,1] = self.initialParameters_more[4]
        T3_ini[2,1] = self.initialParameters_more[5]
        T3_ini[3,1] = 0.0
        T3_ini[0,2] = self.initialParameters_more[6]
        T3_ini[1,2] = self.initialParameters_more[7]
        T3_ini[2,2] = self.initialParameters_more[8]
        T3_ini[3,2] = 0.0
        T3_ini[0,3] = self.initialParameters[0]
        T3_ini[1,3] = self.initialParameters[1]
        T3_ini[2,3] = self.initialParameters[2]
        T3_ini[3,3] = 1

        T3_est[0,0] = self.finalParameters[0]
        T3_est[1,0] = self.finalParameters[1]
        T3_est[2,0] = self.finalParameters[2]
        T3_est[3,0] = 0.0
        T3_est[0,1] = self.finalParameters[3]
        T3_est[1,1] = self.finalParameters[4]
        T3_est[2,1] = self.finalParameters[5]
        T3_est[3,1] = 0.0
        T3_est[0,2] = self.finalParameters[6]
        T3_est[1,2] = self.finalParameters[7]
        T3_est[2,2] = self.finalParameters[8]
        T3_est[3,2] = 0.0
        T3_est[0,3] = self.finalParameters[9]
        T3_est[1,3] = self.finalParameters[10]
        T3_est[2,3] = self.finalParameters[11]
        T3_est[3,3] = 1

        T2[0,0] = RTT_Transformations[0][0][0]
        T2[1,0] = RTT_Transformations[0][1][0]
        T2[2,0] = RTT_Transformations[0][2][0]
        T2[3,0] = 0.0
        T2[0,1] = RTT_Transformations[0][0][1]
        T2[1,1] = RTT_Transformations[0][1][1]
        T2[2,1] = RTT_Transformations[0][2][1]
        T2[3,1] = 0.0
        T2[0,2] = RTT_Transformations[0][0][2]
        T2[1,2] = RTT_Transformations[0][1][2]
        T2[2,2] = RTT_Transformations[0][2][2]
        T2[3,2] = 0.0
        T2[0,3] = RTT_Transformations[0][0][3]
        T2[1,3] = RTT_Transformations[0][1][3]
        T2[2,3] = RTT_Transformations[0][2][3]
        T2[3,3] = 1

        q[0] = ImagePoints[0][0]
        q[1] = ImagePoints[0][1]
        q[2] = 0.0
        q[3] = 1.0

        self.x = ImagePoints[0][2]
        self.y = ImagePoints[0][3]
        self.z = ImagePoints[0][4]
        offset = np.array((self.x,self.y,self.z))
        real_world.append(offset)
        T1[0] = ImagePoints[0][5]
        T1[1] = ImagePoints[0][6]
        T1[2] = ImagePoints[0][7]
        T1[3] = 1.0

        qInT_ini = np.dot(T2, np.dot(T3_ini, q))
        errX_ini = qInT_ini[0] -self.x
        errY_ini = qInT_ini[1] -self.y
        errZ_ini = qInT_ini[2] -self.z

        dist_ini = np.sqrt(errX_ini**2 + errY_ini**2 + errZ_ini**2)
        _min_ini = dist_ini
        _max_ini = dist_ini
        _mean_ini = dist_ini

        qInT_est = np.dot(T2, np.dot(T3_est, q))
        errX = qInT_est[0] -self.x
        errY = qInT_est[1] -self.y
        errZ = qInT_est[2] -self.z
        self.distance.append(np.sqrt(errX**2 + errY**2 + errZ**2))
        _mean_sqrt = errX**2 + errY**2 + errZ**2
        self.distance_rms.append(_mean_sqrt)
        validation_p1 = np.dot(self.phantom_origin_T,qInT_est)
        validation_p2 = validation_p1
        initial_point = np.array([T2[0,3],T2[1,3],T2[2,3]])
        T1_before.append(initial_point)
        validation_point1 = np.array([validation_p1[0][0],validation_p1[1][0],validation_p1[2][0]])
        validation_point2 = np.array([T1[0],T1[1],T1[2]])
        T1_after.append(validation_point1)
        T2_after.append(validation_point2)

        _min = self.distance[0]
        _max = self.distance[0]
        _mean = self.distance[0]
        n = RTT_Transformations.shape[0]
        for i in range(1,n):
            T2 = np.zeros((4,4))
            q = np.zeros((4,1))
            T2[0,0] = RTT_Transformations[i][0][0]
            T2[1,0] = RTT_Transformations[i][1][0]
            T2[2,0] = RTT_Transformations[i][2][0]
            T2[3,0] = 0.0
            T2[0,1] = RTT_Transformations[i][0][1]
            T2[1,1] = RTT_Transformations[i][1][1]
            T2[2,1] = RTT_Transformations[i][2][1]
            T2[3,1] = 0.0
            T2[0,2] = RTT_Transformations[i][0][2]
            T2[1,2] = RTT_Transformations[i][1][2]
            T2[2,2] = RTT_Transformations[i][2][2]
            T2[3,2] = 0.0
            T2[0,3] = RTT_Transformations[i][0][3]
            T2[1,3] = RTT_Transformations[i][1][3]
            T2[2,3] = RTT_Transformations[i][2][3]
            T2[3,3] = 1

            q[0] = ImagePoints[i][0]
            q[1] = ImagePoints[i][1]
            q[2] = 0.0
            q[3] = 1.0
            self.x = ImagePoints[i][2]
            self.y = ImagePoints[i][3]
            self.z = ImagePoints[i][4]
            offset = np.array((self.x,self.y,self.z))
            real_world.append(offset)
            T1[0] = ImagePoints[i][5]
            T1[1] = ImagePoints[i][6]
            T1[2] = ImagePoints[i][7]

            qInT_ini = np.dot(T2, np.dot(T3_ini, q))
            errX_ini = qInT_ini[0] -self.x
            errY_ini = qInT_ini[1] -self.y
            errZ_ini = qInT_ini[2] -self.z
            dist_ini = np.sqrt(errX_ini**2 + errY_ini**2 + errZ_ini**2)

            _mean_ini = _mean_ini + dist_ini
            if(dist_ini>_max_ini):
                _max_ini = dist_ini
            if(dist_ini<_min_ini):
                _min_ini = dist_ini

            qInT_est = np.dot(T2, np.dot(T3_est, q))
            errX = qInT_est[0] -self.x
            errY = qInT_est[1] -self.y
            errZ = qInT_est[2] -self.z
            validation_p1 = np.dot(self.phantom_origin_T,qInT_est)
            validation_p2 = validation_p1
            initial_point = np.array([T2[0,3],T2[1,3],T2[2,3]])
            T1_before.append(initial_point)
            validation_point1 = np.array([validation_p1[0][0],validation_p1[1][0],validation_p1[2][0]])
            validation_point2 = np.array([T1[0],T1[1],T1[2]])
            T1_after.append(validation_point1)
            T2_after.append(validation_point2)
            dist_sqrt = errX**2 + errY**2 + errZ**2
            dist = np.sqrt(errX**2 + errY**2 + errZ**2)
            # if dist>4:
                # print(i)
                # print(dist_ini)
                # print(dist)
            # print(i)

            _mean = _mean + dist
            _mean_sqrt = _mean_sqrt + dist_sqrt
            if(dist>_max):
                _max = dist
            if(dist<_min):
                _min = dist
            self.distance_rms.append(dist_sqrt)
            self.distance.append(dist)
        _mean /= n
        _rms = np.sqrt(_mean_sqrt/n)
        _mean_ini /= n
        print("\n ini: _mean, _min, _max: " + str(_mean_ini) + ', '+ str(_min_ini) + ', '+ str(_max_ini))
        T1_after = np.asarray(T1_after)
        T2_after = np.asarray(T2_after)
        T1_before = np.asarray(T1_before)
        real_world = np.asarray(real_world)
        xd = []
        yd = []
        zd = []
        phantom_param_data_path = './Zphantom_param.csv'
        with open(phantom_param_data_path,"r") as csvfile:
            filereader = csv.DictReader(csvfile)
            for row in filereader:
                index = str(row['Points_index'])
                x = float(row['px'])
                y = float(row['py'])
                z = float(row['pz'])
                xd.append(x)
                yd.append(y)
                zd.append(z)
        ax1.scatter(xd,yd,zd, cmap='Blues',label='Phantom')
        ax1.scatter(T1_after[:,0],T1_after[:,1],T1_after[:,2], cmap='Greens',label='Ground Truth')
        ax1.scatter(T2_after[:,0],T2_after[:,1],T2_after[:,2], cmap='Reds',label='Reconstruction Point')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        return _mean, _min, _max, _rms

    def N_fiducial_calibration(self):
        re = self.loadImagePointsFromCSV() #read pixel data and probe position data
        self.leastSquaresEstimate(self.RTT_Transformations,self.ImagePoints)
        _mean, _min, _max, _rms = self.getDistanceStatistics(self.RTT_Transformations,self.ImagePoints)
        print("\n Final: _mean, _min, _max, _rms: " + str(_mean) + ', '+ str(_min) + ', '+ str(_max)+ ', '+ str(_rms))
        plt.show()
        return _mean, _min, _max, _rms

if __name__=="__main__": 
    Filepath1 = "./1.csv"
    Filepath2 = "./imagepixel.csv"
    USCalibration = SinglePointTargetUSCalibrationParametersEstimator(Filepath1, Filepath2, True)
    _mean, _min, _max, _rms = USCalibration.N_fiducial_calibration()



