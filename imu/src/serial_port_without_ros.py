#! /usr/bin/env python

import serial
import math
import time 
import string 
from datetime import datetime
import numpy as np

def talker():
  
    ser = serial.Serial('/dev/ttyACM0',115200,timeout=1)  # open first serial port
    print(ser.name)          # check which port was really used
    count=0 
 
    line = ser.readline() 
    split_string = string.split(line,' ') 

    if    count> 30 :

        linear_acceleration_x = (float(split_string[0]) ) 
        linear_acceleration_y = (float(split_string[1]) )
        linear_acceleration_z = (float(split_string[2]) )  
        ##  unit m/s2  without gravity  filtered
        
        angular_velocity_x = float(split_string[3]) 
        angular_velocity_y = float(split_string[4]) 
        angular_velocity_z = float(split_string[5])  
        ## angular velocity unit rad/s filtered

        magnetic_field_x = float(split_string[6])
        magnetic_field_y = float(split_string[7])
        magnetic_field_z = float(split_string[8])   
        ##mag milliGauss
 

        count =count+1  

    ser.close()    
        
if __name__ == '__main__':
    try:
        talker() 
