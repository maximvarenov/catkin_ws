
import rospy
import scipy
import math
import numpy as np
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker
from filterpy.kalman import ExtendedKalmanFilter
from sympy import symbols, Matrix, pprint, init_printing, zeros
from pyquaternion.quaternion import Quaternion
from scipy.signal import butter, lfilter,filtfilt
from tf.transformations import quaternion_matrix
import numba as nb



init_printing(True)

class PixhawkEKF(ExtendedKalmanFilter):
    def __init__(self):
        super(PixhawkEKF, self).__init__(dim_x=7, dim_z=3)
        self.x[0, 0] = 1.
        self.Q = self.Q
        self.P[0:4, 0:4] = self.P[0:4, 0:4] * 0.01
        self.P[4:, 4:] = self.P[4:, 4:] * 0.001
        self.R = self.R * (0.05)
        self.timeStamp = None
        self.deltaT = 0
        self.axr=0
        self.ayr=0
        self.azr=0
        self.vx=0
        self.vy=0
        self.vz=0
        self.px=0
        self.py=0
        self.pz=0
        self.count =0 
        self.generateEquations()
        self.imuRotation = Quaternion(axis=[1., 0., 0.], degrees=180.0).rotation_matrix
        self.ekfNodeHandle = rospy.init_node('imu_path_node')
        self.imuSubscriber = rospy.Subscriber('/imu_data', Imu, self.omImuMessageReceived)
        self.cubePublisher = rospy.Publisher('/imu_path', Marker, queue_size=10)
        rospy.spin()

    
    @nb.jit()
    def generateEquations(self):
        self.qw, self.qx, self.qy, self.qz = symbols('qw qx qy qz')
        self.gx, self.gy, self.gz = symbols('gx, gy, gz')
        self.ax, self.ay, self.az = symbols('ax ay az')
        self.bx, self.by, self.bz = symbols('bwx bwy bwz')
        gyro = Matrix([[self.gx], [self.gy], [self.gz]])
        accel = Matrix([[self.ax], [self.ay], [self.az]])
        bias = Matrix([[self.bx], [self.by], [self.bz]])
        state = Matrix([[self.qw], [self.qx], [self.qy], [self.qz], 
                        [self.bx], [self.by], [self.bz]])
        omega2qdot = 0.5 * Matrix([[-self.qx, -self.qy, -self.qz],
                                    [self.qw, self.qz, -self.qy],
                                    [-self.qz, self.qw, self.qx],
                                    [self.qy, -self.qx, self.qw]])
        angVelJacobian = omega2qdot * (gyro - bias)
        self.xdot = zeros(7, 1)
        self.xdot[0, 0] = angVelJacobian[0]
        self.xdot[1, 0] = angVelJacobian[1]
        self.xdot[2, 0] = angVelJacobian[2]
        self.xdot[3, 0] = angVelJacobian[3]
        self.fSympy = self.xdot.jacobian(state)
        CNed2Body =  Matrix([[1-2*(self.qy**2+self.qz**2),2*(self.qx*self.qy+self.qz*self.qw),2*(self.qx*self.qz-self.qy*self.qw)],
                            [2*(self.qx*self.qy-self.qz*self.qw),1-2*(self.qx**2+self.qz**2),2*(self.qy*self.qz+self.qx*self.qw)],
                            [2*(self.qx*self.qz+self.qy*self.qw),2*(self.qy*self.qz-self.qx*self.qw),1-2*(self.qx**2+self.qy**2)]])
        self.gmps = 9.8
        trueGravity = Matrix([[0.0], [0.0], [self.gmps]])
        
        self.zhat = CNed2Body * trueGravity
        self.hSympy = self.zhat.jacobian(state)   

    @nb.jit()
    def getPredictSubs(self, gyro):
        gx = gyro[0, 0]
        gy = gyro[1, 0]
        gz = gyro[2, 0]
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        subs = {
            self.gx : gx,
            self.gy : gy,
            self.gz : gz,
            self.qw : qw,
            self.qx : qx,
            self.qy : qy,
            self.qz : qz,
            self.bx : self.x[4, 0],
            self.by : self.x[5, 0],
            self.bz : self.x[6, 0]
            }
        return subs


    @nb.jit()
    def getUpdateSubs(self, state):
        qw = state[0, 0]
        qx = state[1, 0]
        qy = state[2, 0]
        qz = state[3, 0]
        subs = {
            self.gmps : 1.0,
            self.qw : qw,
            self.qx : qx,
            self.qy : qy,
            self.qz : qz
        }
        return subs
    @nb.jit()
    def predict(self, u=0):
        self.predict_x(u)
        gx = u[0, 0]
        gy = u[1, 0]
        gz = u[2, 0]
        bwx = self.x[0, 0]
        bwy = self.x[1, 0]
        bwz = self.x[2, 0]
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        F = np.array(
        [[0, 0.5*bwx - 0.5*gx, 0.5*bwy - 0.5*gy, 0.5*bwz - 0.5*gz, 0.5*qx, 0.5*qy, 0.5*qz],
        [-0.5*bwx + 0.5*gx, 0, 0.5*bwz - 0.5*gz, -0.5*bwy + 0.5*gy, -0.5*qw, -0.5*qz, 0.5*qy],
        [-0.5*bwy + 0.5*gy, -0.5*bwz + 0.5*gz, 0, 0.5*bwx - 0.5*gx, 0.5*qz, -0.5*qw, -0.5*qx],
        [-0.5*bwz + 0.5*gz, 0.5*bwy - 0.5*gy, -0.5*bwx + 0.5*gx, 0, -0.5*qy, 0.5*qx, -0.5*qw],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
        Q = np.eye(7, 7)
        Q[4:, 4:] = Q[4:, 4:] * 0.001
        Q[0:4, 0:4] = Q[0:4, 0:4] * 0.05
        AA = np.zeros(shape=(14, 14))
        AA[0:7, 0:7] = -F
        AA[0:7, 7:] = Q
        AA[7:, 7:] = F.T
        AA_dt = AA * self.deltaT
        AA_dt_sq = np.dot(AA_dt, AA_dt)
        AA_dt_cu = np.dot(AA_dt_sq, AA_dt)
        AA_dt_qu = np.dot(AA_dt_cu, AA_dt)
        BB = np.eye(14) + AA_dt + 0.5 * AA_dt_sq + (1/6.0) * AA_dt_cu + (1/np.math.factorial(4)) * AA_dt_qu
        self.F = BB[7:, 7:].T
        self.Q = np.dot(self.F, BB[0:7, 7:])
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
    
    def predict_x(self, u=0):
        unbiasedGyro = u - self.x[4:]
        rotationVec = unbiasedGyro * self.deltaT
        angle = np.linalg.norm(rotationVec)
        if not np.isclose(angle, 0):
            quat = Quaternion(axis=rotationVec, angle=angle)
        else:
            quat = Quaternion([1., 0., 0., 0.])
        result = Quaternion(self.x[0:4, 0]) * quat
        if(self.x[0, 0] < 0):
            self.x[0, 0] = -self.x[0, 0]
            self.x[1, 0] = -self.x[1, 0]
            self.x[2, 0] = -self.x[2, 0]
            self.x[3, 0] = -self.x[3, 0]
        self.x[0:4, 0] = result.normalised.elements

    def rotateImuToPoseCoordSystem(self, imuMsg):
        gx = imuMsg.angular_velocity.x
        gy = imuMsg.angular_velocity.y
        gz = imuMsg.angular_velocity.z
        gyroMat = np.array([[gx], [gy], [gz]])
        rotatedGyro = np.dot(self.imuRotation, gyroMat)
################################################
        self.axr = imuMsg.linear_acceleration.x
        self.ayr = imuMsg.linear_acceleration.y
        self.azr = imuMsg.linear_acceleration.z
##################################################3
        ax = imuMsg.linear_acceleration.x
        ay = imuMsg.linear_acceleration.y
        az = imuMsg.linear_acceleration.z
        accelMat = np.array([[ax], [ay], [az]])
        rotatedAccel = np.dot(self.imuRotation, accelMat/np.linalg.norm(accelMat))
        return rotatedGyro, rotatedAccel
    
    def measurementEstimate(self, state):
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        estimatedAccel = np.array([[2*(qx * qz - qy * qw)],
                                   [2*(qy * qz + qx * qw)],
                                   [1 - 2 *(qx**2 + qy**2)]])
        return estimatedAccel

    
    def measurementJacobian(self, state):
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        H = np.array([[-2*qy, 2*qz, -2*qw, 2*qx, 0, 0, 0],
                            [2*qx, 2*qw, 2*qz, 2*qy, 0, 0, 0], 
                            [0, -4*qx, -4*qy, 0, 0, 0, 0]])
        return H

    def ekfUpdate(self, accelData):
        self.update(accelData, HJacobian=self.measurementJacobian, 
                    Hx=self.measurementEstimate)
        self.x[0:4, 0] = Quaternion(self.x[0:4, 0]).normalised.elements
    
    def omImuMessageReceived(self, imuMsg):
        if(self.timeStamp):
            currentTimestamp = imuMsg.header.stamp.to_sec()
            self.deltaT = currentTimestamp - self.timeStamp
            gyro, accel = self.rotateImuToPoseCoordSystem(imuMsg)
            self.predict(u=gyro)
            self.timeStamp = currentTimestamp
            self.ekfUpdate(accel)
            self.publishVisualizationMarker(accel)
        else:
            self.timeStamp = imuMsg.header.stamp.to_sec()
            #self.publishVisualizationMarker(accel)

    @nb.jit()
    def publishVisualizationMarker(self, accelData):
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.header.stamp = rospy.Time()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale.x = 0.7
        marker.scale.y = 1.
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        flip = Quaternion([0., 0., 0., 1.])
        markerOrient = flip * Quaternion(self.x[0:4, 0])
        marker.pose.orientation.w = markerOrient[0]
        marker.pose.orientation.x = markerOrient[1]
        marker.pose.orientation.y = markerOrient[2]
        marker.pose.orientation.z = markerOrient[3]
        ################################
       
        gx = 2 * (markerOrient[1] * markerOrient[3] - markerOrient[0]  * markerOrient[2]) *9.81
        gy = 2 * (markerOrient[0]  * markerOrient[1] + markerOrient[2] * markerOrient[3]) *9.81
        gz = -(markerOrient[0]* markerOrient[0] - markerOrient[1]* markerOrient[1] - markerOrient[2] * markerOrient[2] +markerOrient[3]* markerOrient[3])*9.81

        ax=round(self.axr-gx,2)
        ay=round(self.ayr-gy,2)
        az=round(self.azr-gz,2)

        roll = math.atan2(2*(marker.pose.orientation.w* marker.pose.orientation.x + marker.pose.orientation.y * marker.pose.orientation.z ),1-2*( marker.pose.orientation.x * marker.pose.orientation.x + marker.pose.orientation.y * marker.pose.orientation.y ))*180/math.pi
        pitch = math.asin(2*(marker.pose.orientation.w * marker.pose.orientation.y -marker.pose.orientation.z *marker.pose.orientation.x ))*180/math.pi
        yaw = math.atan2(2*(marker.pose.orientation.w * marker.pose.orientation.z + marker.pose.orientation.x * marker.pose.orientation.y ),1-2*(marker.pose.orientation.z *marker.pose.orientation.z + marker.pose.orientation.y * marker.pose.orientation.y ))*180/math.pi ####angle
        print yaw,pitch,roll
        # with open('/home/ruixuan/Desktop/imu_data/imu_ekf_ypr2.txt','a') as file_handle:   
        #     file_handle.write(str(marker.header.stamp)) 
        #     file_handle.write(str(yaw)) 
        #     file_handle.write(str(' ')) 
        #     file_handle.write(str(pitch)) 
        #     file_handle.write(str(' ')) 
        #     file_handle.write(str(roll))
        #     file_handle.write('\n') 
        if self.count >30:
            oldvx=self.vx
            oldvy=self.vy
            oldvz=self.vz
            self.vx= self.vx+ ax * self.deltaT
            self.vy= self.vy+ ay * self.deltaT
            self.vz= self.vz+ az * self.deltaT
            
            self.px  +=  self.vx *self.deltaT + 0.5* ax * self.deltaT *self.deltaT
            self.py  +=  self.vy *self.deltaT + 0.5* ay * self.deltaT *self.deltaT
            self.pz  +=  self.vz *self.deltaT + 0.5* az * self.deltaT *self.deltaT



            marker.pose.position.x = self.px
            marker.pose.position.y = self.py
            marker.pose.position.z = self.pz
            self.count = self.count +1
            #print self.deltaT, self.axr,self.ayr,self.azr,ax,ay,az, self.vx,self.vy,self.vz,self.px,self.py,self.pz

        t = []                    
        x = []                    
        y = []                    
        z = []                    
        plt.ion()                  
        for i in range(1000):       
            t.append(i)           
            x.append(self.px)        
            y.append(self.py)        
            z.append(self.pz)        
            plt.clf()              
            plt.plot(t,x)        
            plt.pause(0.1)         
            splt.ioff()        

        self.count = self.count +1
        self.cubePublisher.publish(marker)
        
        
if __name__ == "__main__":
    ekf = PixhawkEKF()
