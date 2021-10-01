#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import sys
import csv
import scipy.linalg
import math
import Levenberg_marquardt as lm
from scipy.spatial.transform import Rotation

class PivotCalibrationParametersEstimator():
    def __init__(self, transformationsFilePath, minForEstimate = 32):
        self.transformationsFileName = transformationsFilePath
        self.RTT_Transformations = []
        self.minForEstimate = 16
        self.X = np.zeros(6)
        self.est_X = np.zeros(6)
        self.knownExactTranslations = np.array((-17.7799, 1.1113, -156.865, 146.901, -62.9689, -1042.14))
        self.maxError = 1.0
        self.z_tran = 100
        self.y_tran = 10

    def loadTransformations(self):
        # load the data, each row in the file is expected to have
        # the following format x y z qx qy qz qs
        transformations = []
        f=open(self.transformationsFileName,'r')
        lines=f.readlines()
        for line in lines:
            T = np.zeros((3,4))
            Tran = np.zeros(3)
            Quat = np.zeros(4)
            readline = line.strip().split('\t')
            Tran[0] = float(readline[0])
            Tran[1] = float(readline[1])
            Tran[2] = float(readline[2])
            Quat[0] = float(readline[3])
            Quat[1] = float(readline[4])
            Quat[2] = float(readline[5])
            Quat[3] = float(readline[6])
            R = Rotation.from_quat(Quat).as_matrix()
            T[0][0] = R[0][0]
            T[0][1] = R[0][1]
            T[0][2] = R[0][2]
            T[0][3] = Tran[0]
            T[1][0] = R[1][0]
            T[1][1] = R[1][1]
            T[1][2] = R[1][2]
            T[1][3] = Tran[1]
            T[2][0] = R[2][0]
            T[2][1] = R[2][1]
            T[2][2] = R[2][2]
            T[2][3] = Tran[2]

            transformations.append(T)
        self.RTT_Transformations = np.asarray(transformations)
        # print("\n RTT_Transformations: "+ str(self.RTT_Transformations.shape))
        if(self.RTT_Transformations.size == 0): return False
        return True

    def loadTransformationsfromcsv(self):
        transformations = []
        with open(self.transformationsFileName,'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                T = np.zeros((3,4))
                Tran = np.zeros(3)
                Quat = np.zeros(4)
                Tran[0] = float(row['x'])
                Tran[1] = float(row['y'])
                Tran[2] = float(row['z'])
                Quat[0] = float(row['qx'])
                Quat[1] = float(row['qy'])
                Quat[2] = float(row['qz'])
                Quat[3] = float(row['qw'])
                R = Rotation.from_quat(Quat).as_matrix()
                T[0][0] = R[0][0]
                T[0][1] = R[0][1]
                T[0][2] = R[0][2]
                T[0][3] = Tran[0]
                T[1][0] = R[1][0]
                T[1][1] = R[1][1]
                T[1][2] = R[1][2]
                T[1][3] = Tran[1]
                T[2][0] = R[2][0]
                T[2][1] = R[2][1]
                T[2][2] = R[2][2]
                T[2][3] = Tran[2]

                transformations.append(T)
        self.RTT_Transformations = np.asarray(transformations)
        # print("\n RTT_Transformations: "+ str(self.RTT_Transformations.shape))
        if(self.RTT_Transformations.size == 0): return False
        return True

    def estimate(self):
        A = np.zeros((9,6))
        b = np.zeros(9)

        for i in range(3):
            index = 3*i
            A[index,0] = self.RTT_Transformations[i,0,0]
            A[index,1] = self.RTT_Transformations[i,0,1]
            A[index,2] = self.RTT_Transformations[i,0,2]
            A[index,3] = -1
            A[index,4] = 0
            A[index,5] = 0
            b[index] = -self.RTT_Transformations[i,3,0]
            index += 1

            A[index,0] = self.RTT_Transformations[i,1,0]
            A[index,1] = self.RTT_Transformations[i,1,1]
            A[index,2] = self.RTT_Transformations[i,1,2]
            A[index,3] = 0
            A[index,4] = -1
            A[index,5] = 0
            b[index] = -self.RTT_Transformations[i,3,1]
            index += 1

            A[index,0] = self.RTT_Transformations[i,2,0]
            A[index,1] = self.RTT_Transformations[i,2,1]
            A[index,2] = self.RTT_Transformations[i,2,2]
            A[index,3] = 0
            A[index,4] = 0
            A[index,5] = -1
            b[index] = -self.RTT_Transformations[i,3,2]

        Ainv = scipy.linalg.pinv2(A)
        if (np.linalg.matrix_rank(Ainv)<6): return
        self.est_X = np.dot(Ainv, b)



    def leastSquaresEstimate(self):
        if self.RTT_Transformations.shape[0] < self.minForEstimate:
            return
        n = self.RTT_Transformations.shape[0]
        A = np.zeros((3*n,6))
        b = np.zeros((3*n,1))
        for i in range(n):
            index = 3*i
            A[index,0] = self.RTT_Transformations[i,0,0]
            A[index,1] = self.RTT_Transformations[i,0,1]
            A[index,2] = self.RTT_Transformations[i,0,2]
            A[index,3] = -1
            A[index,4] = 0
            A[index,5] = 0
            b[index] = -self.RTT_Transformations[i,0,3]
            index += 1

            A[index,0] = self.RTT_Transformations[i,1,0]
            A[index,1] = self.RTT_Transformations[i,1,1]
            A[index,2] = self.RTT_Transformations[i,1,2]
            A[index,3] = 0
            A[index,4] = -1
            A[index,5] = 0
            b[index] = -self.RTT_Transformations[i,1,3]
            index += 1

            A[index,0] = self.RTT_Transformations[i,2,0]
            A[index,1] = self.RTT_Transformations[i,2,1]
            A[index,2] = self.RTT_Transformations[i,2,2]
            A[index,3] = 0
            A[index,4] = 0
            A[index,5] = -1
            b[index] = -self.RTT_Transformations[i,2,3]

        Ainv = scipy.linalg.pinv2(A)
        if (np.linalg.matrix_rank(Ainv)<6): return
        self.X = np.dot(Ainv, b)


        def getdistance(self):
            succeedLeastSquares = True
            for i in range(6):
                succeedLeastSquares = succeedLeastSquares and (math.fabs(self.x[i] - self.knownExactTranslations[i])<self.maxError)
            if(succeedLeastSquares):return True
            else: return False

if __name__=="__main__":

    Filetest = '/home/yuyu/rosbag/processed/phantom_points_data.csv'
    PiovtCalibration = PivotCalibrationParametersEstimator(Filetest)
    
    re = PiovtCalibration.loadTransformationsfromcsv()
    if re == True:
        # print("\n RTT_Transformations: " + str(PiovtCalibration.RTT_Transformations))
        PiovtCalibration.leastSquaresEstimate()
        print(PiovtCalibration.X)
        # _mean,_min,_max = PiovtCalibration.getDistanceStatistics()
        # print("\n _mean, _min, _max: " + str(_mean) + ', '+ str(_min) + ', '+ str(_max))
        # print("\n distance: " + str(PiovtCalibration.distance))
