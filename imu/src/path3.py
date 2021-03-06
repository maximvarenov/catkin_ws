#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from numpy import sqrt, pi 
import time
import smbus
import numpy as np
import os, sys
import matplotlib.pyplot as plt

def quaternionMultiplication(p0, p1, p2, p3, q0, q1, q2, q3):
	r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
	r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
	r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
	r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
	return r0, r1, r2, r3

def normalizeQuaternion(q0, q1, q2, q3):
	norm = sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
	q0 = q0 / norm
	q1 = q1 / norm
	q2 = q2 / norm
	q3 = q3 / norm
	return q0, q1, q2, q3

def normalization(v1, v2, v3):
	norm = sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2)
	if norm == 0 :
		norm = 1 
	v1 = v1 / norm
	v2 = v2 / norm
 	v3=  v3 / norm
	return v1, v2, v3

def rotateVectorQuaternion(x, y, z, q0, q1, q2, q3):
	vx = (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * x + 2 * (q1 * q2 - q0 * q3) * y + 2 * (q1 * q3 + q0 * q2) * z
	vy = 2 * (q1 * q2 + q0 * q3) * x + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * y + 2 * (q2 * q3 - q0 * q1) * z
	vz = 2 * (q1 * q3 - q0 * q2) * x + 2 * (q2 * q3 + q0 * q1) * y + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * z
	return vx, vy, vz

class complement_Filter:
	def __init__(self):
		self.g_xBias, self.g_yBias, self.g_zBias = 0., 0., 0.
		self.m_xBias, self.m_yBias, self.m_zBias = -6, -229, -544
		self.m_xScale, self.m_yScale, self.m_zScale = 442.593423764, 397.070552885, 432.124639423
		self.Alpha, self.Beta, self.Gyro =0.04, 0.04, 0.2
		self.q0, self.q1, self.q2, self.q3 = 1., 0., 0., 0.
		self.q0_acc_mag, self.q1_acc_mag, self.q2_acc_mag, self.q3_acc_mag = 1., 0., 0., 0.
		self.a_x, self.a_y, self.a_z = 0., 0., 0.
		self.g_x, self.g_y, self.g_z = 0., 0., 0.
		self.m_x, self.m_y, self.m_z = 0., 0., 0.
		self.pub = rospy.Publisher("/quat", PoseWithCovarianceStamped, queue_size = 1)
		rospy.Subscriber("/imu_data", Imu, self.get_imu_data)
		rospy.Subscriber("/mag_data", MagneticField, self.get_mag_data)
		while self.m_x ** 2 + self.m_y ** 2 + self.m_z ** 2 == 0 :
			rospy.sleep(0.1)
		self.t_prev = time.time()
		self.q0_acc_mag, self.q1_acc_mag, self.q2_acc_mag, self.q3_acc_mag = self.initialize()
		self.rate = rospy.Rate(100)

	def calcDT(self):
		t_now = time.time()
		self.dt = t_now - self.t_prev
                print(self.dt)
		self.t_prev = t_now

	def get_imu_data(self, msg):
		self.a_x = msg.linear_acceleration.x
		self.a_y = msg.linear_acceleration.y
		self.a_z = msg.linear_acceleration.z	
		self.g_x = msg.angular_velocity.x - self.g_xBias
		self.g_y = msg.angular_velocity.y - self.g_yBias
		self.g_z = msg.angular_velocity.z - self.g_zBias
		
	def get_mag_data(self, msg): 
		self.m_x = (msg.magnetic_field.y - self.m_yBias) * (self.m_xScale + self.m_yScale + self.m_zScale) / (3 * self.m_yScale) 
		self.m_y = (msg.magnetic_field.x - self.m_xBias) * (self.m_xScale + self.m_yScale + self.m_zScale) / (3 * self.m_xScale)
		self.m_z = -(msg.magnetic_field.z - self.m_zBias) * (self.m_xScale + self.m_yScale + self.m_zScale) / (3 * self.m_zScale) 
	
	def initialize(self):
		q0_sum, q1_sum, q2_sum, q3_sum, count = 0., 0., 0., 0., 0.
		self.t_now = time.time()
		while self.t_now - self.t_prev < 1 :
			q0_acc, q1_acc, q2_acc, q3_acc, q0_mag, q1_mag, q2_mag, q3_mag = self.mag_predict()
			q0, q1, q2, q3 = quaternionMultiplication(q0_acc, q1_acc, q2_acc, q3_acc, q0_mag, q1_mag, q2_mag, q3_mag)
			q0_sum += q0
			q1_sum += q1
			q2_sum += q2
			q3_sum += q3
			count += 1
			self.t_now = time.time()
		q0 = (self.Alpha + self.Beta)/2 * q0_sum / count + (1 - (self.Alpha + self.Beta)/2)
		q1 = (self.Alpha + self.Beta)/2 * q1_sum / count
		q2 = (self.Alpha + self.Beta)/2 * q2_sum / count
		q3 = (self.Alpha + self.Beta)/2 * q3_sum / count
		#return q0, q1, q2, q3
		return q0_sum/count, q1_sum/count, q2_sum/count, q3_sum/count

	def steadyState(self):
		norm = sqrt(self.g_x ** 2 + self.g_y ** 2 + self.g_z ** 2)
		axy = sqrt(self.a_x ** 2 + self.a_y ** 2)
		if norm < 0.002 and axy < 0.01:
			return True
		return False

	def gyroUpdate(self):
		BiasIs = False
		self.g_xBias = self.g_xBias + (self.g_x - self.g_xBias) * self.Gyro
		self.g_yBias = self.g_yBias + (self.g_y - self.g_yBias) * self.Gyro
		self.g_zBias = self.g_zBias + (self.g_z - self.g_zBias) * self.Gyro

	def getPrediction(self):
		self.calcDT()
		#self.dt = 1./100.
		q0_gyro, q1_gyro, q2_gyro, q3_gyro = quaternionMultiplication(0, self.g_x, self.g_y, self.g_z, self.q0, self.q1, self.q2, self.q3)
		q0_gyro = self.q0 - 0.5 * q0_gyro * self.dt 
		q1_gyro = self.q1 - 0.5 * q1_gyro * self.dt 
		q2_gyro = self.q2 - 0.5 * q2_gyro * self.dt 
		q3_gyro = self.q3 - 0.5 * q3_gyro * self.dt 
		self.q0_gyro, self.q1_gyro, self.q2_gyro, self.q3_gyro = normalizeQuaternion(q0_gyro, q1_gyro, q2_gyro, q3_gyro)
		
	def gainFunction(self, a, b, c):
		norm = sqrt(a ** 2 + b ** 2 + c ** 2)
		error = abs(norm - 1)
		if error < 0.05: 
			return 1
		elif error < 0.1:
			return 1 - 5 * error
		else:
			return 0

	def acc_Correction(self):
		alpha = self.gainFunction(self.a_x, self.a_y, self.a_z) * self.Alpha
		print(alpha)
		#alpha = self.Alpha
		a_x, a_y, a_z = normalization(self.a_x, self.a_y, self.a_z)
		gx, gy, gz = rotateVectorQuaternion(a_x, a_y, a_z, self.q0_gyro, -self.q1_gyro, -self.q2_gyro, -self.q3_gyro)
		q0_acc = alpha * sqrt(0.5 * (gx + 1)) + (1 - alpha)
		q1_acc = alpha * ( - gy / sqrt(2 * (gz + 1)))
		q2_acc = alpha * ( gx / sqrt(2 * (gz + 1)))
		q3_acc = 0
		return q0_acc, q1_acc, q2_acc, q3_acc

	def mag_Correction(self):
		#beta = self.gainFunction(self.m_x, self.m_y, self.m_z) * self.Beta
		beta = self.Beta
		lx, ly, lz = rotateVectorQuaternion(self.m_x, self.m_y, self.m_z, self.q0, -self.q1, -self.q2, -self.q3)
		gamma = lx ** 2 + ly ** 2
		q0_mag = beta * sqrt(gamma + lx * sqrt(gamma))/ sqrt(2 * gamma) + (1 - beta)
		q1_mag = 0
		q2_mag = 0
		q3_mag = beta * ly / sqrt(2 * (gamma + lx * sqrt(gamma)))
		q0_mag, q1_mag, q2_mag, q3_mag = normalizeQuaternion(q0_mag, q1_mag, q2_mag, q3_mag)
		return q0_mag, q1_mag, q2_mag, q3_mag

	def accel_predict(self):
		a_x, a_y, a_z = normalization(self.a_x, self.a_y, self.a_z)
		if self.a_z >= 0:		
			q0_acc = sqrt(0.5 * (a_z + 1))
			q1_acc = - a_y / sqrt(2 * (a_z + 1))
			q2_acc = a_x / sqrt(2 * (a_z + 1))
			q3_acc = 0
		else :
			q0_acc = - a_y / sqrt(2 * (1 - a_z))
			q1_acc = sqrt(0.5 * (1 - a_z))
			q2_acc = 0
			q3_acc = a_x / sqrt(2 * (1 - a_z))
		q0_acc, q1_acc, q2_acc, q3_acc = normalizeQuaternion(q0_acc, q1_acc, q2_acc, q3_acc)
		return q0_acc, q1_acc, q2_acc, q3_acc

	def mag_predict(self):
		q0_acc, q1_acc, q2_acc, q3_acc = self.accel_predict()
		if self.m_x ** 2 + self.m_y ** 2 + self.m_z ** 2 == 0 :
			rospy.sleep(0.5)
		lx, ly, lz = rotateVectorQuaternion(self.m_x, self.m_y, self.m_z, q0_acc, -q1_acc, -q2_acc, -q3_acc)
		gamma = lx ** 2 + ly ** 2
		if lx >= 0 :
			q0_mag = sqrt((gamma + lx * sqrt(gamma))/ (2 * gamma))
			q1_mag = 0
			q2_mag = 0
			q3_mag = ly / sqrt( 2 * (gamma + lx * sqrt(gamma)))
		else:
			q0_mag = ly / sqrt( 2 * (gamma - lx * sqrt(gamma)))
			q1_mag = 0
			q2_mag = 0
			q3_mag = sqrt((gamma - lx * sqrt(gamma))/ (2 * gamma))
		return q0_acc, q1_acc, q2_acc, q3_acc, q0_mag, q1_mag, q2_mag, q3_mag
		
	def imu_Mag_Complementary(self):
		while not rospy.is_shutdown():
			if self.steadyState() == True:
				self.gyroUpdate()
			self.getPrediction()
			q0_acc, q1_acc, q2_acc, q3_acc = self.acc_Correction()
			q0, q1, q2, q3 = quaternionMultiplication(self.q0_gyro, self.q1_gyro, self.q2_gyro, self.q3_gyro, q0_acc, q1_acc, q2_acc, q3_acc)
			self.q0, self.q1, self.q2, self.q3 = normalizeQuaternion(q0, q1, q2, q3)
			q0_mag, q1_mag, q2_mag, q3_mag = self.mag_Correction()
			q0, q1, q2, q3 = quaternionMultiplication(self.q0, self.q1, self.q2, self.q3, q0_mag, q1_mag, q2_mag, q3_mag)
			self.q0, self.q1, self.q2, self.q3 = normalizeQuaternion(q0, q1, q2, q3)
			q0, q1, q2, q3 = quaternionMultiplication(self.q0, self.q1, self.q2, self.q3, self.q0_acc_mag, -self.q1_acc_mag, -self.q2_acc_mag, -self.q3_acc_mag )
			quat_topic = PoseWithCovarianceStamped()
			quat_topic.header.stamp = rospy.Time.now()
			quat_topic.header.frame_id = "world"
			quat_topic.pose.pose.position.x = 0
			quat_topic.pose.pose.position.y = 0
			quat_topic.pose.pose.position.z = 0
			quat_topic.pose.pose.orientation.x = - q1
			quat_topic.pose.pose.orientation.y = - q2
			quat_topic.pose.pose.orientation.z = - q3
			quat_topic.pose.pose.orientation.w = q0
			self.pub.publish(quat_topic)
			self.rate.sleep()

	def acc_Mag_Complementary(self):
		while not rospy.is_shutdown():
			q0_acc, q1_acc, q2_acc, q3_acc, q0_mag, q1_mag, q2_mag, q3_mag = self.mag_predict()
			q0, q1, q2, q3 = quaternionMultiplication(q0_acc, q1_acc, q2_acc, q3_acc, q0_mag, q1_mag, q2_mag, q3_mag)
			q0, q1, q2, q3 = quaternionMultiplication(q0, q1, q2, q3, self.q0_mag, -self.q1_mag, -self.q2_mag, -self.q3_mag)
			quat_topic = PoseWithCovarianceStamped()
			quat_topic.header.stamp = rospy.Time.now()
			quat_topic.header.frame_id = "world"
			quat_topic.pose.pose.position.x = 0
			quat_topic.pose.pose.position.y = 0
			quat_topic.pose.pose.position.z = 0
			quat_topic.pose.pose.orientation.x = - q1_mag
			quat_topic.pose.pose.orientation.y = - q2_mag
			quat_topic.pose.pose.orientation.z = - q3_mag
			quat_topic.pose.pose.orientation.w = q0_mag
			self.pub.publish(quat_topic)
			self.rate.sleep()

if __name__=="__main__":
	rospy.init_node("Complementary", anonymous = True)
	rospy.loginfo("starting Complementary Filter")
	complement = complement_Filter()
	try:
		rospy.loginfo("complementary filter start!")
		complement.imu_Mag_Complementary()
		#complement.acc_Mag_Complementary()
	except rospy.ROSInterruptException:
		print "ROS terminated"
		pass


