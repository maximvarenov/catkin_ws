import numpy as np
import sys
import scipy.linalg
import math
import Levenberg_marquardt as lm


class SinglePointTargetUSCalibrationParametersEstimator():
    def __init__(self, transformationsFilePath, ImagePointsFilePath, lsType = False , minForEstimate = 32):
        self.transformationsFileName = transformationsFilePath
        self.ImagePointsFileName = ImagePointsFilePath
        self.RTT_Transformations = []
        self.ImagePoints = []
        self.minForEstimate = 32
        self.LS_init_flag = False
        self.epsilon = 1.192092896e-07
        self.lsType = lsType
        self.Parameters = []
        self.initialParameters = np.zeros((8,1))
        self.initialParameters_more = np.zeros((9,1))
        self.finalParameters = np.zeros((12,1))
        self.distance= []
        self.x = 98.63086737114155
        self.y = 4.261085473508394
        self.z = 1051.1113655791607



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
        if(self.ImagePoints.size == 0): return False
        return True

    def leastSquaresEstimate(self):
        if self.lsType == False:
            self.analyticLeastSquaresEstimate()
        else:
            self.analyticLeastSquaresEstimate()
            if self.LS_init_flag == False:
                print("\n ERROR: LeastSquaresEstimate init fail \n")
                sys.exit(1)
            self.iterativeLeastSquaresEstimate()

    def analyticLeastSquaresEstimate(self):
        if self.RTT_Transformations.size < self.minForEstimate:
            print("\n ERROR: Transformation size is too small \n")
            sys.exit(1)

        numPoints = self.RTT_Transformations.shape[0]
        self.A = np.zeros((3*numPoints,9))
        x = np.zeros(9)
        self.b = np.zeros((3*numPoints,1))

        for i in range(numPoints):
            # 2D pixel coordinates
            ui = self.ImagePoints[i][0] -33
            vi = self.ImagePoints[i][1] -217
            index = 3*i
            # first row for current data element
            self.A[index][0] = self.RTT_Transformations[i][0][0] * ui
            self.A[index][1] = self.RTT_Transformations[i][0][1] * ui
            self.A[index][2] = self.RTT_Transformations[i][0][2] * ui
            self.A[index][3] = self.RTT_Transformations[i][0][0] * vi
            self.A[index][4] = self.RTT_Transformations[i][0][1] * vi
            self.A[index][5] = self.RTT_Transformations[i][0][2] * vi
            self.A[index][6] = self.RTT_Transformations[i][0][0]
            self.A[index][7] = self.RTT_Transformations[i][0][1]
            self.A[index][8] = self.RTT_Transformations[i][0][2]
            self.b[index] =  self.x -self.RTT_Transformations[i][0][3]

            
            index += 1
            # second row for current data element
            self.A[index][0] = self.RTT_Transformations[i][1][0] * ui
            self.A[index][1] = self.RTT_Transformations[i][1][1] * ui
            self.A[index][2] = self.RTT_Transformations[i][1][2] * ui
            self.A[index][3] = self.RTT_Transformations[i][1][0] * vi
            self.A[index][4] = self.RTT_Transformations[i][1][1] * vi
            self.A[index][5] = self.RTT_Transformations[i][1][2] * vi
            self.A[index][6] = self.RTT_Transformations[i][1][0]
            self.A[index][7] = self.RTT_Transformations[i][1][1]
            self.A[index][8] = self.RTT_Transformations[i][1][2]
            self.b[index] = self.y-self.RTT_Transformations[i][1][3]
            index += 1

            # third row for current data element
            self.A[index][0] = self.RTT_Transformations[i][2][0] * ui
            self.A[index][1] = self.RTT_Transformations[i][2][1] * ui
            self.A[index][2] = self.RTT_Transformations[i][2][2] * ui
            self.A[index][3] = self.RTT_Transformations[i][2][0] * vi
            self.A[index][4] = self.RTT_Transformations[i][2][1] * vi
            self.A[index][5] = self.RTT_Transformations[i][2][2] * vi
            self.A[index][6] = self.RTT_Transformations[i][2][0]
            self.A[index][7] = self.RTT_Transformations[i][2][1]
            self.A[index][8] = self.RTT_Transformations[i][2][2]
            self.b[index] = self.z-self.RTT_Transformations[i][2][3]
            index += 1
        print("\n A.rank: " + str(np.linalg.matrix_rank(self.A)))
        Ainv = scipy.linalg.pinv2(self.A)
        # print("\n Ainv.shape: " + str(Ainv.shape))
        print("\n Ainv.rank: " + str(np.linalg.matrix_rank(Ainv)))
        # print("\n b.shape: " + str(b.shape))
        if (np.linalg.matrix_rank(Ainv)<9): return
        x = Ainv @ self.b
        
        
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
        R3 = U@VT
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
        output =  A @ x
        # print("\n output: "+ str(output.shape))
        return output

    def iterativeLeastSquaresEstimate(self):
        num_iter = 500
        # print("\n self.A: " + str(self.A.shape))
        # print("\n self.b: " + str(self.b.shape))
        LM = lm.Levenberg_marquardt(self.my_Func)
        est_params = LM.LM(num_iter,self.initialParameters,self.A,self.b)
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
        # print("\n finalParameters_trans_maxtrix: " + str(self.finalParameters))


    def getDistanceStatistics(self):
        T1 = np.eye(4)
        T2 = np.zeros((4,4))
        T3_ini = np.zeros((4,4))
        T3_est = np.zeros((4,4))
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

        T2[0,0] = self.RTT_Transformations[0][0][0]
        T2[1,0] = self.RTT_Transformations[0][1][0]
        T2[2,0] = self.RTT_Transformations[0][2][0]
        T2[3,0] = 0.0
        T2[0,1] = self.RTT_Transformations[0][0][1]
        T2[1,1] = self.RTT_Transformations[0][1][1]
        T2[2,1] = self.RTT_Transformations[0][2][1]
        T2[3,1] = 0.0
        T2[0,2] = self.RTT_Transformations[0][0][2]
        T2[1,2] = self.RTT_Transformations[0][1][2]
        T2[2,2] = self.RTT_Transformations[0][2][2]
        T2[3,2] = 0.0
        T2[0,3] = self.RTT_Transformations[0][0][3]
        T2[1,3] = self.RTT_Transformations[0][1][3]
        T2[2,3] = self.RTT_Transformations[0][2][3]
        T2[3,3] = 1

        q[0] = self.ImagePoints[0][0]-33
        q[1] = self.ImagePoints[0][1]-217
        q[2] = 0.0
        q[3] = 1.0

        T1[0,3] = -self.x
        T1[1,3] = -self.y
        T1[2,3] = -self.z
        T1[3,3] = 1.0
        
        qInT_ini = T1 @ T2 @ T3_ini @ q
        errX_ini = qInT_ini[0] 
        errY_ini = qInT_ini[1] 
        errZ_ini = qInT_ini[2] 
        dist_ini = np.sqrt(errX_ini**2 + errY_ini**2 + errZ_ini**2)
        _min_ini = dist_ini
        _max_ini = dist_ini
        _mean_ini = dist_ini

        qInT_est = T1 @ T2 @ T3_est @ q
        errX = qInT_est[0]
        errY = qInT_est[1]
        errZ = qInT_est[2]
        self.distance.append(np.sqrt(errX**2 + errY**2 + errZ**2))
        
        _min = self.distance[0]
        _max = self.distance[0]
        _mean = self.distance[0]
        n = self.RTT_Transformations.shape[0]

        for i in range(1,n):
            T2 = np.zeros((4,4))
            q = np.zeros((4,1))
            qInT = np.zeros((4,1))
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

            q[0] = self.ImagePoints[i][0] -33
            q[1] = self.ImagePoints[i][1] -217
            q[2] = 0.0
            q[3] = 1.0

            qInT_ini = T1 @ T2 @ T3_ini @ q
            errX_ini = qInT_ini[0] 
            errY_ini = qInT_ini[1] 
            errZ_ini = qInT_ini[2] 
            dist_ini = np.sqrt(errX_ini**2 + errY_ini**2 + errZ_ini**2)

            _mean_ini = _mean_ini + dist_ini
            if(dist_ini>_max_ini):
                _max_ini = dist_ini
            if(dist_ini<_min_ini):
                _min_ini = dist_ini

            qInT_est = T1 @ T2 @ T3_est @ q
            errX = qInT_est[0]
            errY = qInT_est[1]
            errZ = qInT_est[2]
            dist = np.sqrt(errX**2 + errY**2 + errZ**2)
            

            _mean = _mean + dist
            if(dist>_max):
                _max = dist
            if(dist<_min):
                _min = dist
            self.distance.append(dist)
        _mean /= n

        _mean_ini /= n
        print("\n ini: _mean, _min, _max: " + str(_mean_ini) + ', '+ str(_min_ini) + ', '+ str(_max_ini))
        
        # compute mse
        # print("\n distance: " + str(self.distance))
        # distance_mse = (_mean - self.distance[0]) ** 2
        # for i in range(1,n):
        #     distance_mse += (self.distance[i] - _mean) **2
        # distance_mse /= n
        # print("\n distance_mse: " + str(distance_mse))
        
        return _mean, _min, _max

if __name__=="__main__":
    # Filepath = "E:/KU Leuven/Master thesis/code/calibration/LSQRRecipes-master/testing/Data/"
    # Filepath1 = Filepath + "crossWirePhantomTransformations.txt"
    # Filepath2 = Filepath + "crossWirePhantom2DPoints.txt"
    Filetest = "/home/yuyu/Documents/cal_pre1.txt"
    USCalibration = SinglePointTargetUSCalibrationParametersEstimator(Filetest, Filetest , True)
    re = USCalibration.loadTransformationsandImagePoints()
    if re == True:
        USCalibration.leastSquaresEstimate()

        _mean, _min, _max = USCalibration.getDistanceStatistics()
        print("\n est: _mean, _min, _max: " + str(_mean) + ', '+ str(_min) + ', '+ str(_max))
        # print("\n distance: " + str(USCalibration.distance))


