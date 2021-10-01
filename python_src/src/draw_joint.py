import rospy
import time 
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState


def callback(data):  
    plt.ion()  
    v_1.append(data.velocity[0]) 
    v_2.append(data.velocity[1]) 
    v_3.append(data.velocity[2]) 
    v_4.append(data.velocity[3]) 
    v_5.append(data.velocity[4]) 
    v_6.append(data.velocity[5]) 
    v_7.append(data.velocity[6]) 

    p_1.append(data.position[0]) 
    p_2.append(data.position[1]) 
    p_3.append(data.position[2]) 
    p_4.append(data.position[3]) 
    p_5.append(data.position[4]) 
    p_6.append(data.position[5]) 
    p_7.append(data.position[6])  
 


    ax.append(len(v_1))
    plt.title('joint velocity  [m/s]') 
    plt.clf()               
    plt.plot(ax,p_1, color='green', label='joint 1')     
    plt.plot(ax,p_2, color='red', label='joint 2') 
    plt.plot(ax,p_3,  label='joint 3') 
    plt.plot(ax,p_4,  label='joint 4') 
    plt.plot(ax,p_5,  label='joint 5') 
    plt.plot(ax,p_6,  label='joint 6') 
    plt.plot(ax,p_7,  label='joint 7') 
    plt.legend()
    plt.pause(0.001)          
    plt.ioff()     


def listener():
    rospy.init_node('kuka_joint_states', anonymous=True)
    global ax ,v_1, v_2, v_3, v_4, v_5, v_6, v_7
    global p_1, p_2, p_3, p_4, p_5, p_6, p_7
    ax = []
    v_1 = []
    v_2 = []
    v_3 = []
    v_4 = []
    v_5 = []
    v_6 = []
    v_7 = []  
    p_1 = []
    p_2 = []
    p_3 = []
    p_4 = []
    p_5 = []
    p_6 = []
    p_7 = []  
    rospy.Subscriber("/joint_states", JointState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()