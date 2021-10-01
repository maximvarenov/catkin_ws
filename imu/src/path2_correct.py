
import rospy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
 
import numpy as np
import time
import math
import numpy.linalg as lin 
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


def quat_mult(a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4):
        #quaternion multiplication
        q_0 = a_1*b_1 - a_2*b_2 - a_3*b_3 - a_4*b_4
        q_1 = a_1*b_2 + a_2*b_1 + a_3*b_4 - a_4*b_3
        q_2 = a_1*b_3 - a_2*b_4 + a_3*b_1 + a_4*b_2
        q_3 = a_1*b_4 + a_2*b_3 - a_3*b_2 + a_4*b_1
        q = np.matrix([q_0, q_1, q_2, q_3])
        q = q.T
        return q

def norm_quat(a_1, a_2, a_3, a_4):
        #making quaternion size 1
        q_0 = a_1/math.sqrt(a_1**2 + a_2**2 + a_3**2 + a_4**2)
        q_1 = a_2/math.sqrt(a_1**2 + a_2**2 + a_3**2 + a_4**2)
        q_2 = a_3/math.sqrt(a_1**2 + a_2**2 + a_3**2 + a_4**2)
        q_3 = a_4/math.sqrt(a_1**2 + a_2**2 + a_3**2 + a_4**2)
        q = np.matrix([q_0, q_1, q_2, q_3])
        q = q.T
        return q
def normalization(v1, v2, v3):
        #making the vector size 1
        norm = math.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2)
        v1 = v1 / norm
        v2 = v2 / norm
        v3=  v3 / norm
        return v1, v2, v3

def rotateVectorQuaternion(x, y, z, q0, q1, q2, q3):
        #rotate vector using quaternion
        vx = ((q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * (x) + 2 * (q1 * q2 - q0 * q3) * y + 2 * (q1 * q3 + q0 * q2) * z)
        vy = (2 * (q1 * q2 + q0 * q3) * x + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * y + 2 * (q2 * q3 - q0 * q1) * z)
        vz = (2 * (q1 * q3 - q0 * q2) * x + 2 * (q2 * q3 + q0 * q1) * y + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * z)
        return vx, vy, vz

class kalman_Filter:
    def __init__(self):
        self.X = np.matrix('1;0;0;0')
        self.P = np.identity(4)
        self.dt = float(1.0/78.5)
        self.H =  np.identity(4)

        # the gyro sensor standard variation
        self.Q = 10.0**(-10)*np.matrix([[1.0, 0, 0, 0],[0, 1.50628058**2, 0, 0],[0, 0, 1.4789602**2, 0],[0, 0, 0, 1.37315181**2]])

        # accelometer and magnetometer's standard variation
        self.R = 5*np.matrix([[0.00840483082215**2, 0, 0, 0],[0, 0.00100112198402**2, 0, 0], [0, 0, 0.00102210818946**2, 0], [0, 0, 0, 0.0114244938775**2]])

        # Subscriber created
        self.mag_x = 0.01
        self.mag_y = 0.01
        self.mag_z = 0.01
        self.acc_x = 0.01
        self.acc_y = 0.01
        self.acc_z = 0.01
        self.gyro_x = 0.01
        self.gyro_y = 0.01
        self.gyro_z = 0.01
        self.gyro_bias_x = 0
        self.gyro_bias_y = 0
        self.gyro_bias_z = 0
        self.vx=0
        self.vy=0
        self.vz=0
        self.px=0
        self.py=0
        self.pz=0 
        self.count = 1
        self.rate = rospy.Rate(20)
        rospy.Subscriber("/imu_data", Imu, self.imu_raw_data)
        rospy.Subscriber("/mag_data", MagneticField, self.mag_raw_data)
        print('finish initialized')
        # self.Kalman_cov_pub = rospy.Publisher("/pose_covariance",PoseWithCovarianceStamped, queue_size=1)


    def imu_raw_data(self, msg): 

        self.imu_data = msg
        self.imu_secs = self.imu_data.header.stamp.secs
        self.imu_nsecs = self.imu_data.header.stamp.nsecs
        self.acc_x = float(self.imu_data.linear_acceleration.x)*9.81
        self.acc_y = float(self.imu_data.linear_acceleration.y)*9.81
        self.acc_z = float(self.imu_data.linear_acceleration.z)*9.81

        self.gyro_x = float(self.imu_data.angular_velocity.x)
        self.gyro_y = float(self.imu_data.angular_velocity.y)
        self.gyro_z = float(self.imu_data.angular_velocity.z)

        self.qw = float(self.imu_data.orientation.w)
        self.qx = float(self.imu_data.orientation.x)
        self.qy = float(self.imu_data.orientation.y)
        self.qz = float(self.imu_data.orientation.z)

    def mag_raw_data(self, msg):
        self.mag_data = msg

        # bias when doing calibration at the motion capture lab
        self.mag_bias_x = -180
        self.mag_bias_y = -30
        self.mag_bias_z = 127

        self.mag_delta_x = 180
        self.mag_delta_y = 21
        self.mag_delta_z = 127
        
        self.mag_average = (self.mag_delta_x + self.mag_delta_y + self.mag_delta_z)/3

        #magnetometer sensor's axis is twisted so we have to change axis

        self.mag_x = (self.mag_data.magnetic_field.y -self.mag_bias_y) * (self.mag_average)/(self.mag_delta_y)
        self.mag_y = (self.mag_data.magnetic_field.x -self.mag_bias_x) * (self.mag_average)/(self.mag_delta_x)
        self.mag_z = -(self.mag_data.magnetic_field.z -self.mag_bias_z) * (self.mag_average)/(self.mag_delta_z)
        """
        self.mag_x = (self.mag_data.magnetic_field.y -self.mag_bias_y)
        self.mag_y = (self.mag_data.magnetic_field.x -self.mag_bias_x)
        self.mag_z = -(self.mag_data.magnetic_field.z -self.mag_bias_z)
            """
  
    def first_mag_cal(self):
        self.mag_cal_x = 0
        self.mag_cal_y = 0
        self.mag_cal_z = 0
        self.gyro_cal_x = 0
        self.gyro_cal_y = 0
        self.gyro_cal_z = 0
        self.acc_cal_x = 0
        self.acc_cal_y = 0
        self.acc_cal_z = 0
        self.cal_count = 0

        # waits until the value is available
        while self.cal_count <1000:
            time.sleep(0.01)

        # for 1.5 sec the average value is saved
        # the average value is calculated for calculating the yaw value
         
         
            self.mag_cal_x += self.mag_x
            self.mag_cal_y += self.mag_y
            self.mag_cal_z += self.mag_z
            self.gyro_cal_x += self.gyro_x
            self.gyro_cal_y += self.gyro_y
            self.gyro_cal_z += self.gyro_z
            self.acc_cal_x += self.acc_x
            self.acc_cal_y += self.acc_y
            self.acc_cal_z += self.acc_z
            self.cal_count += 1

        self.mag_cal_x /= self.cal_count
        self.mag_cal_y /= self.cal_count
        self.mag_cal_z /= self.cal_count
        self.gyro_cal_x /= self.cal_count
        self.gyro_cal_y /= self.cal_count
        self.gyro_cal_z /= self.cal_count
        self.acc_cal_x /= self.cal_count
        self.acc_cal_y /= self.cal_count
        self.acc_cal_z /= self.cal_count

        print('======== finish first mag calibration ========')



    def get_acc_quat(self):
        #normalize the accel value
        self.ax = self.acc_x / math.sqrt(self.acc_x**2 +self.acc_y**2 + self.acc_z**2)
        self.ay = self.acc_y / math.sqrt(self.acc_x**2 +self.acc_y**2 + self.acc_z**2)
        self.az = self.acc_z / math.sqrt(self.acc_x**2 +self.acc_y**2 + self.acc_z**2)

        if self.az >= 0:
            self.q_acc = np.matrix([math.sqrt(0.5*(self.az + 1)), -self.ay/(2*math.sqrt(0.5*(self.az+1))), self.ax/(2*math.sqrt(0.5*(self.az+1))), 0])
        else :
            self.q_acc_const = math.sqrt((1.0-self.az) * 0.5)
            self.q_acc = np.matrix([-self.ay/(2.0*self.q_acc_const), self.q_acc_const, 0.0, self.ax/(2.0*self.q_acc_const)])

    def state_check(self):
        # when gyro is almost 0 and accel's x and y axis is almost 0 then the system will assume the left over value is gyro bias
        gyro_scale = math.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2)
        acc_scale = math.sqrt(self.acc_x**2 + self.acc_y**2)
        if gyro_scale < 0.01 and acc_scale < 0.1 :
                return True
        else :
                return False

    def gyro_bias_update(self):
        # getting rid of gyro bias
        if self.state_check():
            self.gyro_bias_x += self.gyro_x
            self.gyro_bias_y += self.gyro_z
            self.gyro_bias_z += self.gyro_y

            self.gyro_x -= self.gyro_bias_x
            self.gyro_y -= self.gyro_bias_y
            self.gyro_z -= self.gyro_bias_z

    def get_mag_quat(self):
        #rotating the magnetometer's value using accelometer's quaternion
        lx, ly, lz = rotateVectorQuaternion(self.mag_x, self.mag_y, self.mag_z, self.q_acc[0,0], -self.q_acc[0,1], -self.q_acc[0,2], -self.q_acc[0,3])
        #calculating the yaw using rotated magnetometer value
        self.gamma = lx**2 + ly**2
        if lx >= 0:
            self.q0_mag = math.sqrt(self.gamma + lx * math.sqrt(self.gamma))/ math.sqrt(2 * self.gamma)
            self.q1_mag = 0
            self.q2_mag = 0
            self.q3_mag = ly / math.sqrt(2 * (self.gamma + lx * math.sqrt(self.gamma)))
            self.q_mag= norm_quat(self.q0_mag, self.q1_mag, self.q2_mag, self.q3_mag)
        if lx < 0:
            self.q0_mag = ly / math.sqrt(2 * (self.gamma - lx * math.sqrt(self.gamma)))
            self.q1_mag = 0
            self.q2_mag = 0
            self.q3_mag = math.sqrt(self.gamma - lx * math.sqrt(self.gamma))/ math.sqrt(2 * self.gamma)
            self.q_mag= norm_quat(self.q0_mag, self.q1_mag, self.q2_mag, self.q3_mag)



    def get_mag_acc_calibration(self):
        # calculating the starting yaw (the angle between the north)
        # and get rid of the starting yaw
        self.ax_cal = self.acc_cal_x / math.sqrt(self.acc_cal_x**2 +self.acc_cal_y**2 + self.acc_cal_z**2)
        self.ay_cal = self.acc_cal_y / math.sqrt(self.acc_cal_x**2 +self.acc_cal_y**2 + self.acc_cal_z**2)
        self.az_cal = self.acc_cal_z / math.sqrt(self.acc_cal_x**2 +self.acc_cal_y**2 + self.acc_cal_z**2)
        self.q_acc_cal = np.matrix([math.sqrt(0.5*(self.az_cal + 1)), -self.ay_cal/(2*math.sqrt(0.5*(self.az_cal+1))), self.ax_cal/(2*math.sqrt(0.5*(self.az_cal+1))), 0])

        self.mag_cal_x, self.mag_cal_y, self.mag_cal_z = normalization(self.mag_cal_x, self.mag_cal_y, self.mag_cal_z)
        lx_cal, ly_cal, lz_cal = rotateVectorQuaternion(self.mag_cal_x, self.mag_cal_y, self.mag_cal_z, -self.q_acc_cal[0,0], -self.q_acc_cal[0,1], -self.q_acc_cal[0,2], -self.q_acc_cal[0,3])
        self.gamma_cal = lx_cal ** 2 + ly_cal ** 2
        if lx_cal >= 0:
            self.q0_mag_cal = math.sqrt(self.gamma_cal + lx_cal * math.sqrt(self.gamma_cal))/ math.sqrt(2 * self.gamma_cal)
            self.q1_mag_cal = 0
            self.q2_mag_cal = 0
            self.q3_mag_cal = ly_cal / math.sqrt(2 * (self.gamma_cal + lx_cal * math.sqrt(self.gamma_cal)))
            self.q_mag_cal= norm_quat(self.q0_mag_cal, self.q1_mag_cal, self.q2_mag_cal, self.q3_mag_cal)
        if lx_cal < 0:
            self.q0_mag_cal = ly_cal / math.sqrt(2 * (self.gamma_cal - lx_cal * math.sqrt(self.gamma_cal)))
            self.q1_mag_cal = 0
            self.q2_mag_cal = 0
            self.q3_mag_cal = math.sqrt(self.gamma_cal - lx_cal * math.sqrt(self.gamma_cal))/ math.sqrt(2 * self.gamma_cal)
            self.q_mag_cal= norm_quat(self.q0_mag_cal, self.q1_mag_cal, self.q2_mag_cal, self.q3_mag_cal)
        print('finish get mag_acc calibration')


    def kalman(self):
        pose_topic = PoseStamped()
        self.gyro_bias_update()
        self.get_acc_quat()
        self.get_mag_quat()

        self.q_mag_calibrating = quat_mult(self.q_acc[0,0],self.q_acc[0,1],self.q_acc[0,2],self.q_acc[0,3],self.q_mag[0,0],self.q_mag[1,0],self.q_mag[2,0],self.q_mag[3,0])
        self.Z = quat_mult(self.q_mag_calibrating[0,0],self.q_mag_calibrating[1,0],self.q_mag_calibrating[2,0],self.q_mag_calibrating[3,0],self.q_mag_cal[0,0],-self.q_mag_cal[1,0],-self.q_mag_cal[2,0],-self.q_mag_cal[3,0])
        self.Z = norm_quat(self.Z[0,0],self.Z[1,0],self.Z[2,0],self.Z[3,0])
        #making the gyro matrix
        self.A = np.identity(4)-self.dt*0.5*np.matrix([[0,-self.gyro_x,-self.gyro_y,-self.gyro_z],[self.gyro_x,0,-self.gyro_z,self.gyro_y],[self.gyro_y,self.gyro_z,0,-self.gyro_x],[self.gyro_z,-self.gyro_y,self.gyro_x,0]])
        # Kalman Filter

        #calculating the predict value (gyro)
        self.Xp = self.A*self.X
        #normalize the quaternion
        self.Xp = norm_quat(self.Xp[0,0],self.Xp[1,0],self.Xp[2,0],self.Xp[3,0])
        #calculat predict covariance value
        self.Pp = self.A*self.P*self.A.T +self.Q
        #calculate kalman gain
        self.K = self.Pp*self.H.T*lin.inv(self.H*self.Pp*self.H.T + self.R)

        if (self.Xp[0,0] - self.Z[0,0])>1 or (self.Xp[0,0] - self.Z[0,0])< -1 or (self.Xp[1,0] - self.Z[1,0])>1 or (self.Xp[1,0] - self.Z[1,0])< -1 or (self.Xp[2,0] - self.Z[2,0])>1 or (self.Xp[2,0] - self.Z[2,0])< -1 or (self.Xp[3,0] - self.Z[3,0])>1 or (self.Xp[3,0] - self.Z[3,0])< -1 :
            # change the + -
            # + - multiplication is same quaternion
            self.Z = np.matrix([-self.Z[0,0],-self.Z[1,0],-self.Z[2,0],-self.Z[3,0]])
            self.Z = self.Z.T
        # calculating  the quaternion using the sensor fusion
        self.X = self.Xp + self.K*(self.Z - self.H*self.Xp)
        # normalize the quaternion
        self.X = norm_quat(self.X[0,0],self.X[1,0],self.X[2,0],self.X[3,0])
        # calculating the covariance
        self.P = self.Pp - self.K*self.H*self.Pp
        #print("new data ", self.acc_x,self.acc_y,self.acc_z,self.A,self.Xp)

        #　pose_topic.header.stamp.secs = self.imu_secs
        # pose_topic.header.stamp.nsecs = self.imu_nsecs
        pose_topic.header.frame_id = "world"
        pose_topic.pose.orientation.x = -self.X[1,0]
        pose_topic.pose.orientation.y = -self.X[2,0]
        pose_topic.pose.orientation.z = -self.X[3,0]
        pose_topic.pose.orientation.w = self.X[0,0]
        # pose_topic.pose.orientation.x = self.qx
        # pose_topic.pose.orientation.y = self.qy
        # pose_topic.pose.orientation.z = self.qz
        # pose_topic.pose.orientation.w = self.qw
        pose_topic.pose.position.x = 0
        pose_topic.pose.position.y = 0
        pose_topic.pose.position.z = 0
################################
        roll2 = math.atan2(2*(pose_topic.pose.orientation.w* pose_topic.pose.orientation.x+pose_topic.pose.orientation.y*pose_topic.pose.orientation.z),1-2*( pose_topic.pose.orientation.x* pose_topic.pose.orientation.x+pose_topic.pose.orientation.y*pose_topic.pose.orientation.y))*180
        pitch2 = math.asin(2*(pose_topic.pose.orientation.w*pose_topic.pose.orientation.y-pose_topic.pose.orientation.z*pose_topic.pose.orientation.x))*180
        yaw2 = math.atan2(2*(pose_topic.pose.orientation.w * pose_topic.pose.orientation.z+ pose_topic.pose.orientation.x*pose_topic.pose.orientation.y),1-2*(pose_topic.pose.orientation.z*pose_topic.pose.orientation.z+pose_topic.pose.orientation.y*pose_topic.pose.orientation.y))*180
        #print(yaw2,pitch2,roll2)
####################################

        gx = 2 * (self.X[1,0] * self.X[3,0] +self.X[0,0] * self.X[2,0]) * 9.81
        gy = 2 * (-self.X[0,0]  * self.X[1,0] + self.X[2,0] *self.X[3,0])* 9.81
        gz = (self.X[0,0] * self.X[0,0] - self.X[1,0]*self.X[1,0] - self.X[2,0] *self.X[2,0] + self.X[3,0]* self.X[3,0])* 9.81

        axr=self.acc_x-gx
        ayr=self.acc_y-gy
        azr=self.acc_z-gz
        
        self.px  +=  self.vx *self.dt + 0.5* axr * self.dt *self.dt
        self.py  +=  self.vy *self.dt + 0.5* ayr * self.dt *self.dt
        self.pz  +=  self.vz *self.dt + 0.5* azr * self.dt *self.dt
        self.vx= self.vx+ axr * self.dt
        self.vy= self.vy+ ayr * self.dt
        self.vz= self.vz+ azr * self.dt

        print(' gravity conpemsation a', axr,ayr,azr)







 # =============================== rotation correction ==========================================================
        if self.count < 5 :
            self.initial_orientation = np.array([roll2,pitch2,yaw2])


        roll_delta =  roll2 - self.initial_orientation[0]
        pitch_delta =  pitch2 - self.initial_orientation[1]
        yaw_delta =  yaw2 - self.initial_orientation[2]

        r = R.from_euler('xyz', np.array([roll_delta, pitch_delta, yaw_delta]), degrees=True)
        rotation_delta = r.as_matrix()

        a_correct =  np.dot(rotation_delta , np.array([axr,ayr,azr]))
        print(' gravity conpemsation acc after correction', a_correct )
        self.count +=1

 # =============================== rotation correction ==========================================================

        
        # t = []                    
        # x = []                    
        # y = []                    
        # z = []                    
        # plt.ion()                  
        # for i in range(1000):       
        #     t.append(i)           
        #     x.append(self.px)        
        #     y.append(self.py)        
        #     z.append(self.pz)        
        #     plt.clf()              
        #     plt.plot(t,x)        
        #     plt.pause(0.1)         
        #     plt.ioff()        

        #print(self.acc_x,self.acc_y,self.acc_z,axr,ayr,azr,self.vx,self.vy,self.vz,self.px,self.py,self.pz)
        #with open('/home/ruixuan/Desktop/imu_data/imu_ekf.txt','a') as file_handle:   
            #file_handle.write(str(self.time)) 
            #file_handle.write(str(' '))
            #file_handle.write(str(axr)) 
            #file_handle.write(str(' '))
            #file_handle.write(str(ayr)) 
            #file_handle.write(str(' '))
            #file_handle.write(str(azr))  
            #file_handle.write(str(' '))
            #file_handle.write(str(self.px)) 
            #file_handle.write(str(' '))
            #file_handle.write(str(self.py)) 
            #file_handle.write(str(' '))
            #file_handle.write(str(self.pz)) 
            #file_handle.write('\n')

    
        # self.Kalman_cov_pub.publish(pose_topic)
        # self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node("Kalman_Filter", anonymous=True) 
    try:  
        Filtering = kalman_Filter()
        # starting the calibration
        Filtering.first_mag_cal()
        # calculating the first yaw calculation
        Filtering.get_mag_acc_calibration()
        while not rospy.is_shutdown():
            #kalman filter starting 
            Filtering.kalman()
    except rospy.ROSInterruptException:
        print("ROS terminated")
        pass
