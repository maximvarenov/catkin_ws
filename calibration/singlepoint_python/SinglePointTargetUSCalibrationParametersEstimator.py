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
        self.maxinteration_num = 1000
        self.desiredProbabilityForNoOutliers = 0.99
        self.LS_init_flag = False
        self.epsilon = 1.192092896e-07
        self.lsType = lsType
        self.threshold_err = 1
        self.threshold_err_ini = 100
        self.Parameters = []
        self.initialParameters = np.zeros((8,1))
        self.initialParameters_more = np.zeros((9,1))
        self.finalParameters = np.zeros((12,1))
        self.distance= []
        self.readfromcsv = True
        self.RANSAC_result = True
        self.phantom_origin_T = np.array([[-0.81489647,  0.56390167,  -0.13400989,  0        ]
                                        ,[-0.55857574,   -0.82576611,  -0.07812476,  0        ]
                                        ,[-0.15471551, 0.01119108,   0.98789568,  0        ]
                                        ,[ 0,          0,          0,          1        ]])
        # self.phantom_origin_T = np.array([[-0.89834839, -0.41027207, -0.15699366,  0.        ]
        #                                 ,[ 0.4239268,  -0.90335783, -0.0650438,   0.        ]
        #                                 ,[-0.11513579, -0.12498581,  0.98545537,  0.        ]
        #                                 ,[ 0.,          0.,          0.,          1.        ]])
        self.phantom_origin_invT = np.linalg.inv(self.phantom_origin_T)
        if self.readfromcsv == False:
            self.x = 98.63086737114155
            self.y = 4.261085473508394
            self.z = 1051.1113655791607

    def getpointsinphantom(self):
        phantom_param_data_path = '/home/yuyu/Documents/singlepoint_python/phantom_param_data_small.csv'
        # phantom_param_data_path = '/home/yuyu/Documents/phantom_param_data_small.csv'
        phantom_c2 = []
        phantom_c3 = []
        phantom_c4 = []
        with open(phantom_param_data_path,"r") as csvfile:
            filereader = csv.DictReader(csvfile)
            for row in filereader:
                index = str(row['Points index'])
                x = float(row['px'])
                y = float(row['py'])
                z = float(row['pz'])
                phantom_list = [index,x,y,z]
                # print(phantom_list)
                if index[-1] == '2':
                    phantom_c2.append(phantom_list)
                if index[-1] == '3':
                    phantom_c3.append(phantom_list)
                if index[-1] == '4':
                    phantom_c4.append(phantom_list)

        self.phantom_param_data.append(phantom_c2) #[fe2,fj2,fk2,be2,bf2,bk2,fn2,bn2]
        self.phantom_param_data.append(phantom_c3) #[fe3,ff3,fk3,be3,bj3,bk3,fn3,bn3]
        self.phantom_param_data.append(phantom_c4) #[fe4,fj4,fk4,be4,bf4,bk4,fn4,bn4]


    def loadTransformations(self):
        transformations = []
        f=open(self.transformationsFileName,'r')
        lines=f.readlines()
        index = 0
        transformation_length = int(len(lines)/3)
        for i in range(transformation_length):
            index = i*3
            R = np.zeros((3,4))
            readline1 = lines[index].strip().split('\t')
            readline2 = lines[index+1].strip().split('\t')
            readline3 = lines[index+2].strip().split('\t')
            R[0][0] = float(readline1[0])
            R[0][1] = float(readline1[1])
            R[0][2] = float(readline1[2])
            R[0][3] = float(readline1[3])
            R[1][0] = float(readline2[0])
            R[1][1] = float(readline2[1])
            R[1][2] = float(readline2[2])
            R[1][3] = float(readline2[3])
            R[2][0] = float(readline3[0])
            R[2][1] = float(readline3[1])
            R[2][2] = float(readline3[2])
            R[2][3] = float(readline3[3])

            transformations.append(R)
        self.RTT_Transformations = np.asarray(transformations)
        f.close()
        # print("\n RTT_Transformations: "+ str(self.RTT_Transformations.shape))
        if(self.RTT_Transformations.size == 0): return False
        return True

    def loadImagePoints(self):
        Points = []
        f = open(self.ImagePointsFileName)
        lines = f.readlines()
        for line in lines:
            P = np.zeros(2)
            readpoint = line.strip().split('\t')
            P[0] = float(readpoint[0])
            P[1] = float(readpoint[1])
            Points.append(P)
        self.ImagePoints = np.asarray(Points)
        f.close()
        if(self.ImagePoints.size == 0): return False
        return True

    def loadTransformationsfromcsv(self):
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
                _11_qx = float(line['_11_qx'])
                _11_qy = float(line['_11_qy'])
                _11_qz = float(line['_11_qz'])
                _11_qw = float(line['_11_qw'])
                Trans = [timestamp, _10_x, _10_y, _10_z, _10_qx, _10_qy, _10_qz, _10_qw, _11_x, _11_y, _11_z, _11_qx, _11_qy, _11_qz, _11_qw]
                transformations.append(Trans)
        return transformations

    def loadImagePointsfromcsv(self):
        self.getpointsinphantom()
        trans = self.loadTransformationsfromcsv()
        RTT = []
        RTT_origin = []
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
                x = float(line['x'])
                y = float(line['y'])
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
                            # if count == 6:
                            #     print(image_name[:-4])
                            #     print(row)
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

                            # origin
                            Trans_origin = np.zeros((4,4))
                            Tran = np.zeros(3)
                            Qua = np.zeros(4)
                            Tran[0] = T[8]
                            Tran[1] = T[9]
                            Tran[2] = T[10]
                            Qua[0] = T[11]
                            Qua[1] = T[12]
                            Qua[2] = T[13]
                            Qua[3] = T[14]
                            R = Rotation.from_quat(Qua).as_matrix()
                            Trans_origin[0][0] = R[0][0]
                            Trans_origin[0][1] = R[0][1]
                            Trans_origin[0][2] = R[0][2]
                            Trans_origin[0][3] = Tran[0]
                            Trans_origin[1][0] = R[1][0]
                            Trans_origin[1][1] = R[1][1]
                            Trans_origin[1][2] = R[1][2]
                            Trans_origin[1][3] = Tran[1]
                            Trans_origin[2][0] = R[2][0]
                            Trans_origin[2][1] = R[2][1]
                            Trans_origin[2][2] = R[2][2]
                            Trans_origin[2][3] = Tran[2]
                            Trans_origin[3][0] = 0
                            Trans_origin[3][1] = 0
                            Trans_origin[3][2] = 0
                            Trans_origin[3][3] = 1

                    # point in image
                    if match == True:
                        k1 =  np.sqrt((ImagePoints_in_column[1][3] - ImagePoints_in_column[2][3])**2+ (ImagePoints_in_column[1][4] - ImagePoints_in_column[2][4])**2) #fe-ff: fe - ff
                        k2 =  np.sqrt((ImagePoints_in_column[2][3] - ImagePoints_in_column[0][3])**2+ (ImagePoints_in_column[2][4] - ImagePoints_in_column[0][4])**2) #ff-fk: ff - fk
                        a1 = k1/k2
                        if ImagePoints_in_column[0][1] == 1:
                            # point in world coordinate
                            pass
                            x_kn = self.phantom_param_data[row][1][1] + a1*(self.phantom_param_data[row][5][1] - self.phantom_param_data[row][1][1]) #x_fj = fe + a1(bk-fe)
                            y_kn = self.phantom_param_data[row][1][2] + a1*(self.phantom_param_data[row][5][2] - self.phantom_param_data[row][1][2]) #y_fj = fe + a1(bk-fe)
                            z_kn = self.phantom_param_data[row][1][3] + a1*(self.phantom_param_data[row][5][3] - self.phantom_param_data[row][1][3])#z_kn = fe
                            Imagepoint_in_relative_coor = np.dot(self.phantom_origin_invT,np.array([[x_kn],[y_kn],[z_kn],[1]]))
                            # Imagepoint_in_relative_coor = np.array([[x_kn],[y_kn],[z_kn],[1]])
                            Imagepoint_in_world_coor = np.dot(Trans_origin,Imagepoint_in_relative_coor)
                            point_in_real = [ImagePoints_in_column[1][3],ImagePoints_in_column[1][4],Imagepoint_in_world_coor[0][0],Imagepoint_in_world_coor[1][0],Imagepoint_in_world_coor[2][0],x_kn,y_kn,z_kn]
                            Points.append(point_in_real)
                            RTT.append(Trans_US)
                            RTT_origin.append(Trans_origin)
                        else:
                            x_kn = self.phantom_param_data[row][4][1] + a1*(self.phantom_param_data[row][2][1] - self.phantom_param_data[row][4][1]) #x_fj = be + a1(fk-be)
                            y_kn = self.phantom_param_data[row][4][2] + a1*(self.phantom_param_data[row][2][2] - self.phantom_param_data[row][4][2]) #y_fj = be + a1(fk-be)
                            z_kn = self.phantom_param_data[row][4][3] + a1*(self.phantom_param_data[row][2][3] - self.phantom_param_data[row][4][3])#z_kn = z_h
                            Imagepoint_in_relative_coor = np.dot(self.phantom_origin_invT,np.array([[x_kn],[y_kn],[z_kn],[1]]))
                            # Imagepoint_in_relative_coor = np.array([[x_kn],[y_kn],[z_kn],[1]])
                            Imagepoint_in_world_coor = np.dot(Trans_origin,Imagepoint_in_relative_coor)
                            point_in_real = [ImagePoints_in_column[1][3],ImagePoints_in_column[1][4],Imagepoint_in_world_coor[0][0],Imagepoint_in_world_coor[1][0],Imagepoint_in_world_coor[2][0],x_kn,y_kn,z_kn]
                            Points.append(point_in_real)
                            RTT.append(Trans_US)
                            RTT_origin.append(Trans_origin)
                        count += 1
                        ImagePoints_in_column = [] #clear Imagepoints_in_column
                        match = False
        self.ImagePoints = np.asarray(Points)
        print(count)
        self.RTT_Transformations = np.asarray(RTT)
        self.RTT_origin = np.asarray(RTT_origin)
        self.readfromcsv = True
        if(self.ImagePoints.size == 0): return False
        return True

    def loadTransformationsandImagePoints(self):
        transformations = []
        Points = []
        f=open(self.transformationsFileName,'r')
        lines=f.readlines()
        for line in lines:
            # read ImagePoints
            P = np.zeros(2)
            readfile = line.strip().split(' ')
            P[0] = float(readfile[0])
            P[1] = float(readfile[1])
            Points.append(P)
            # read Transformatioins
            R = np.zeros((3,4))
            R[0][0] = float(readfile[2])
            R[0][1] = float(readfile[3])
            R[0][2] = float(readfile[4])
            R[0][3] = float(readfile[5])
            R[1][0] = float(readfile[6])
            R[1][1] = float(readfile[7])
            R[1][2] = float(readfile[8])
            R[1][3] = float(readfile[9])
            R[2][0] = float(readfile[10])
            R[2][1] = float(readfile[11])
            R[2][2] = float(readfile[12])
            R[2][3] = float(readfile[13])
            transformations.append(R)
        self.RTT_Transformations = np.asarray(transformations)
        self.ImagePoints = np.asarray(Points)
        f.close()
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


    def my_Func(self, params,A):
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
        if self.RANSAC_result==True:
            print("\n est_params: " + str(est_params))
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
        print("\n finalParameters_trans_maxtrix: \n[[%f,%f,%f,%f],\n[%f,%f,%f,%f],\n[%f,%f,%f,%f],\n[0,0,0,1]]\n" %
        (self.finalParameters[0,0],self.finalParameters[3,0],self.finalParameters[6,0],self.finalParameters[9,0],
        self.finalParameters[1,0],self.finalParameters[4,0],self.finalParameters[7,0],self.finalParameters[10,0],
        self.finalParameters[2,0],self.finalParameters[5,0],self.finalParameters[8,0],self.finalParameters[11,0]))


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
        fig = plt.figure()
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
        validation_p1 = np.dot(np.linalg.inv(self.RTT_origin[0]),qInT_est)
        validation_p2 = np.dot(self.phantom_origin_T,validation_p1)
        initial_point = np.array([T2[0,3],T2[1,3],T2[2,3]])
        T1_before.append(initial_point)
        validation_point1 = np.array([validation_p2[0][0],validation_p2[1][0],validation_p2[2][0]])
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
            validation_p1 = np.dot(np.linalg.inv(self.RTT_origin[i]),qInT_est)
            validation_p2 = np.dot(self.phantom_origin_T,validation_p1)
            initial_point = np.array([T2[0,3],T2[1,3],T2[2,3]])
            T1_before.append(initial_point)
            validation_point1 = np.array([validation_p2[0][0],validation_p2[1][0],validation_p2[2][0]])
            validation_point2 = np.array([T1[0],T1[1],T1[2]])
            T1_after.append(validation_point1)
            T2_after.append(validation_point2)
            dist = np.sqrt(errX**2 + errY**2 + errZ**2)
            # if dist>4:
                # print(i)
                # print(dist_ini)
                # print(dist)
            # print(i)

            _mean = _mean + dist
            if(dist>_max):
                _max = dist
            if(dist<_min):
                _min = dist
            self.distance.append(dist)
        _mean /= n
        _mean_ini /= n
        print("\n ini: _mean, _min, _max: " + str(_mean_ini) + ', '+ str(_min_ini) + ', '+ str(_max_ini))
        T1_after = np.asarray(T1_after)
        T2_after = np.asarray(T2_after)
        T1_before = np.asarray(T1_before)
        xd = []
        yd = []
        zd = []
        phantom_param_data_path = '/home/yuyu/Documents/singlepoint_python/phantom_param_data_small.csv'
        # phantom_param_data_path = '/home/yuyu/Documents/phantom_param_data_small.csv'
        with open(phantom_param_data_path,"r") as csvfile:
            filereader = csv.DictReader(csvfile)
            i = 0
            for row in filereader:
                point = []
                index = str(row['Points index'])
                x = float(row['px'])
                y = float(row['py'])
                z = float(row['pz'])
                xd.append(x)
                yd.append(y)
                zd.append(z)
        ax1.scatter3D(xd,yd,zd, cmap='Blues')
        ax1.scatter3D(T1_after[:,0],T1_after[:,1],T1_after[:,2], cmap='Greens')
        ax1.scatter3D(T2_after[:,0],T2_after[:,1],T2_after[:,2], cmap='Reds')
        # ax1.scatter3D(T1_before[:,0],T1_before[:,1],T1_before[:,2], cmap='Blues')
        ax1.set_xlabel('X Label')
        ax1.set_ylabel('Y Label')
        ax1.set_zlabel('Z label')
        plt.show()
        # compute mse
        # print("\n distance: " + str(self.distance))
        # distance_mse = (_mean - self.distance[0]) ** 2
        # for i in range(1,n):
        #     distance_mse += (self.distance[i] - _mean) **2
        # distance_mse /= n
        # print("\n distance_mse: " + str(distance_mse))
        return _mean, _min, _max

    def agree(self,i):
        T1 = np.eye(4)
        T2 = np.zeros((4,4))
        T3_est = np.zeros((4,4))
        T3_ini = np.zeros((4,4))
        q = np.zeros((4,1))
        qInT_ini = np.zeros((4,1))
        qInT_est = np.zeros((4,1))

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

        T2[0,0] = self.RTT_Transformations[i][0][0]
        T2[1,0] = self.RTT_Transformations[i][1][0]
        T2[2,0] = self.RTT_Transformations[i][2][0]
        T2[3,0] = 0.0
        T2[0,1] = self.RTT_Transformations[i][0][1]
        T2[1,1] = self.RTT_Transformations[i][1][1]
        T2[2,1] = self.RTT_Transformations[i][2][1]
        T2[3,1] = 0.0
        T2[0,2] = self.RTT_Transformations[i][0][2]
        T2[1,2] = self.RTT_Transformations[i][1][2]
        T2[2,2] = self.RTT_Transformations[i][2][2]
        T2[3,2] = 0.0
        T2[0,3] = self.RTT_Transformations[i][0][3]
        T2[1,3] = self.RTT_Transformations[i][1][3]
        T2[2,3] = self.RTT_Transformations[i][2][3]
        T2[3,3] = 1

        q[0] = self.ImagePoints[i][0]
        q[1] = self.ImagePoints[i][1]
        q[2] = 0.0
        q[3] = 1.0
        self.x = self.ImagePoints[i][2]
        self.y = self.ImagePoints[i][3]
        self.z = self.ImagePoints[i][4]
        T1[0] = self.ImagePoints[i][5]
        T1[1] = self.ImagePoints[i][6]
        T1[2] = self.ImagePoints[i][7]

        qInT_ini = np.dot(T2, np.dot(T3_ini, q))
        errX_ini = qInT_ini[0] -self.x
        errY_ini = qInT_ini[1] -self.y
        errZ_ini = qInT_ini[2] -self.z
        dist_ini = np.sqrt(errX_ini**2 + errY_ini**2 + errZ_ini**2)

        qInT_est = np.dot(T2, np.dot(T3_est, q))
        errX = qInT_est[0] -self.x
        errY = qInT_est[1] -self.y
        errZ = qInT_est[2] -self.z
        dist = np.sqrt(errX**2 + errY**2 + errZ**2)

        if dist_ini < self.threshold_err_ini and dist < self.threshold_err:
            self.agree_distance.append(dist)
            return True
        else: return False

    def RANSAC_Compute(self):
        numDataObjects = self.RTT_Transformations.shape[0]
        numForEstimate = self.minForEstimate

        if(numDataObjects<numForEstimate or self.desiredProbabilityForNoOutliers >=1.0 or self.desiredProbabilityForNoOutliers <= 0.0):return 0 

        # subSetIndexComparator = []
        chosenSubSets = []

        numerator = math.log(1.0-self.desiredProbabilityForNoOutliers)
        allTries = self.choose(numDataObjects,numForEstimate)
        print('allTries: %d'%(allTries))
        numVotesForBest = 0
        numTries = allTries
        iterative_count = 0
        for i in range(numTries):
            print('iteration_num/allTries: %d/%d'%(i,allTries))
            notChosen = np.ones(numDataObjects)
            curSubSetIndexes = np.zeros(numForEstimate)
            exactEstimate_RTT = []
            exactEstiamte_ImagePoint = []
            maxIndex = numDataObjects -1

            for l in range(numForEstimate):
                selectedIndex = int(random.random()*maxIndex + 0.5)
                j = -1
                for k in range(numDataObjects):
                    if notChosen[k]:
                        j += 1
                    if j>=selectedIndex:
                        break
                k -= 1
                exactEstimate_RTT.append(self.RTT_Transformations[k])
                exactEstiamte_ImagePoint.append(self.ImagePoints[k])
                notChosen[k] = False
                maxIndex -= 1

            l = 0
            for m in range(numDataObjects):
                if notChosen[m] ==False:
                    curSubSetIndexes[l] = m+1
                    l += 1
            bool_insert = True
            for index in chosenSubSets:
                res = operator.is_(curSubSetIndexes,index)
                if res == True:
                    bool_insert = False

            if bool_insert == True:
                chosenSubSets.append(curSubSetIndexes)
                self.exactEstimate_RTT = np.asarray(exactEstimate_RTT)
                self.exactEstiamte_ImagePoint = np.asarray(exactEstiamte_ImagePoint)
                self.leastSquaresEstimate(self.exactEstimate_RTT,self.exactEstiamte_ImagePoint)
                if((self.initialParameters==np.zeros((8,1))).all()):
                    continue

                numVotesForCur = 0
                curVotes = np.zeros(numDataObjects)
                self.agree_distance = []
                for m in range(numDataObjects):
                    if (numVotesForBest - numVotesForCur)<(numDataObjects-m+1)==False:
                        break
                    if(self.agree(m)):
                        curVotes[m] = True
                        numVotesForCur += 1
                if(len(self.agree_distance)!=0):
                    print('agree_distance: %f' %(max(self.agree_distance)))
                print('numVotesForCur: %d' %(numVotesForCur))
                print('numDataObjects: %d' %(numDataObjects))
                if numVotesForCur > numVotesForBest:
                    numVotesForBest = numVotesForCur
                    bestVotes = copy.deepcopy(curVotes)
                    best_exactEstimate_RTT = copy.deepcopy(self.exactEstimate_RTT)
                    best_exactEstiamte_ImagePoint = copy.deepcopy(self.exactEstiamte_ImagePoint)

                    if(numVotesForBest == numDataObjects):
                        i = numTries
                        print("\n initialParameters: " + str(self.initialParameters))
                    else:
                        denominator = math.log(1.0 - math.pow((numVotesForCur/numDataObjects),numForEstimate))
                        if denominator==0.0:
                            denominator = self.epsilon
                        numTries = int(numerator/denominator + 0.5)
                        if numTries<allTries:
                            numTries = numTries
                        else:
                            numTries = allTries
            else:
                curSubSetIndexes = np.zeros(numForEstimate)

        leastSquaresEstimate_RTT = []
        leastSquaresEstimate_ImagePoint = []
        print(numVotesForBest)
        if(numVotesForBest > 0):
            for m in range(numDataObjects):
                if(bestVotes[m]==True):
                    leastSquaresEstimate_RTT.append(self.RTT_Transformations[m])
                    leastSquaresEstimate_ImagePoint.append(self.ImagePoints[m])
            self.leastSquaresEstimate_RTT = np.asarray(leastSquaresEstimate_RTT)
            print(self.leastSquaresEstimate_RTT.shape[0])
            self.leastSquaresEstimate_ImagePoint = np.asarray(leastSquaresEstimate_ImagePoint)
            self.LS_init_flag = False
            self.RANSAC_result = True
            self.leastSquaresEstimate(best_exactEstimate_RTT,best_exactEstiamte_ImagePoint)
            _mean, _min, _max = self.getDistanceStatistics(self.leastSquaresEstimate_RTT,self.leastSquaresEstimate_ImagePoint)
            return _mean, _min, _max
        else:
            return False,False,False
    def choose(self,n,m):
        if((n-m) > m):
            numeratorStart = n-m+1
            denominatorEnd = m
        else:
            numeratorStart = m+1
            denominatorEnd = n-m
        print('\n numDataObjects/numForEstimate : %d/%d'%(n,m))
        numerator = 1
        for i in range(numeratorStart,n):
            numerator *= i

        denominator = 1
        for i in range(1,denominatorEnd):
            denominator *= i

        result = numerator/denominator
        print('\n result = numerator/denominator : %d = %d/%d'%(result,numerator,denominator))
        if(denominator > self.maxinteration_num or numerator > self.maxinteration_num):
            return self.maxinteration_num
        else:
            return result

if __name__=="__main__":
    Filepath = "/home/yuyu/rosbag/4-20/processed/9_1_image/"
    Filepath1 = Filepath + "Aurora.csv"
    # Filepath2 = Filepath + "imagepixel_after_filter.csv"
    Filepath2 = Filepath + "imagepixel.csv"
    USCalibration = SinglePointTargetUSCalibrationParametersEstimator(Filepath1, Filepath2 , True)
    re = USCalibration.loadImagePointsfromcsv()

    # Filetest = "/home/yuyu/Documents/cal_pre1.txt"
    # USCalibration = SinglePointTargetUSCalibrationParametersEstimator(Filetest, Filetest , True)
    # re = USCalibration.loadTransformationsandImagePoints()
    if re == True:
        # _mean, _min, _max = USCalibration.RANSAC_Compute()
        USCalibration.leastSquaresEstimate(USCalibration.RTT_Transformations,USCalibration.ImagePoints)
        _mean, _min, _max = USCalibration.getDistanceStatistics(USCalibration.RTT_Transformations,USCalibration.ImagePoints)
        print("\n est: _mean, _min, _max: " + str(_mean) + ', '+ str(_min) + ', '+ str(_max))
        # print("\n distance: " + str(USCalibration.distance))



