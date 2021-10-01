import os
import sys, csv
import time
import string
import rosbag
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np


# the aurora topic in lab and in OR are different
AuroraTopic_in_lab = '/aurora_data'
AuroraTopic_in_OR = '/aurora_pose'

def process_EM_bags(dir, out_dir,AuroraTopic):
    bags = os.listdir(dir)
    for bag_file in bags:  
        print(bag_file)
        with rosbag.Bag(dir+'/'+bag_file) as bag:
            # Create a new CSV file for Aurora Data
            filename = '{}/{}.csv'.format(out_dir,bag_file)
            with open(filename, 'w+') as csvfile:
                filewriter = csv.writer(csvfile, delimiter = ',')
                firstIteration = True   #allows header row
                for subtopic, msg, t in bag.read_messages(AuroraTopic):   # for each instant in time that has data for AuroraTopic
                    #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                    #   - put it in the form of a list of 2-element lists
                    msgString = str(msg)
                    msgList = string.split(msgString, '\n')
                    instantaneousListOfData = []
                    for nameValuePair in msgList:
                        splitPair = string.split(nameValuePair, ':')
                        for i in range(len(splitPair)): #should be 0 to 1
                            splitPair[i] = string.strip(splitPair[i])
                        instantaneousListOfData.append(splitPair)
                    #write the first row from the first element of each pair
                    if firstIteration:  # header
                        headers = ["rosbagTimestamp"]   #first column header
                        for pair in instantaneousListOfData:
                            if pair[0] != '-':             # correction needed only for Aurora data
                                headers.append(pair[0])
                        filewriter.writerow(headers)
                        firstIteration = False
                    # write the value from each pair to the file
                    values = [str(t)]   #first column will have rosbag timestamp
                    for pair in instantaneousListOfData:
                        if len(pair) > 1:
                            values.append(pair[1])
                    filewriter.writerow(values)

dir = '/home/yuyu/rosbag/pivotcalibration'
out_dir = '/home/yuyu/rosbag'

_out_dir = out_dir + '/pivot_processed'
# os.makedirs(_out_dir)
process_EM_bags(dir, _out_dir, AuroraTopic_in_lab)