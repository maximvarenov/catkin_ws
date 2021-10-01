import atracsys.stk as tracker_sdk
import numpy as np
import os
from collections import deque
import time


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# Replace atracsys.stk with atracsys.ftk for the fusionTrack.
def exit_with_error(error, tracking_system):
	print(error)
	errors_dict = {}
	if tracking_system.get_last_error(errors_dict) == tracker_sdk.Status.Ok:
		for level in ['errors', 'warnings', 'messages']:
			if level in errors_dict:
				print(errors_dict[level])
	exit(1)


# def getABMat(frame):
#         # print ("test", frame.fiducials[0].position[0])
#     p1= [frame.fiducials[0].position[0], frame.fiducials[0].position[1], frame.fiducials[0].position[2]]
#     p2= [frame.fiducials[1].position[0], frame.fiducials[1].position[1], frame.fiducials[1].position[2]]
#     p3= [frame.fiducials[2].position[0], frame.fiducials[2].position[1], frame.fiducials[2].position[2]]
#     print("-------------------")

#     if frame.fiducials[0].position[0] != 0  and frame.fiducials[1].position[0] !=0 and frame.fiducials[2].position[0]!=0:
#         p1 = np.asarray(p1)
#         p2 = np.asarray(p2)
#         p3 = np.asarray(p3)
#         # print (p1,p2,p3)
#         cenPt = np.mean(np.asarray([p1,p2,p3]),axis=0)
#         # print (cenPt)

#         vec1=p2-p1
#         vec2=p3-p1
#         vecNorm= np.cross(vec2, vec1)
#         Zaxis=normalize(vecNorm)
#         Yaxis=normalize(vec2)
#         Xaxis =normalize(np.cross(Yaxis,Zaxis))

#         Amat=np.asarray([Xaxis,Yaxis,Zaxis])
#         # print ("Amat", Amat)
#         RAmat =np.transpose(Amat)
#         # print("Tran Amat", RAmat)    

#         b=-cenPt
#     return RAmat, b


tracking_system = tracker_sdk.TrackingSystem()
if tracking_system.initialise() != tracker_sdk.Status.Ok:
    exit_with_error(
        "Error, can't initialise the atracsys SDK api.", tracking_system)
if tracking_system.enumerate_devices() != tracker_sdk.Status.Ok:
    exit_with_error("Error, can't enumerate devices.", tracking_system)
frame = tracker_sdk.FrameData()
if tracking_system.create_frame(False, 10, 20, 20, 10) != tracker_sdk.Status.Ok:
    exit_with_error("Error, can't create frame object.", tracking_system)
geometry_path = tracking_system.get_data_option("Data Directory")
path= '/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/fusion_track/fusionTrack_SDK-v4.5.2-linux64/data'
for geometry in ['geometry001.ini', 'geometry002.ini', 'geometry003.ini', 'geometry004.ini', 'geometry005.ini', 'geometry006.ini']:
    if tracking_system.set_geometry(os.path.join(path, geometry)) != tracker_sdk.Status.Ok:
        exit_with_error("Error, can't create frame object.", tracking_system)


x_list = []
y_list = []
z_list = []
filename = '/media/ruixuan/Volume/ruixuan/Pictures/gt2/truth4_2.txt'
f = open(filename,'a')
for x in range(100): 
    tracking_system.get_last_frame(frame)

    for marker in frame.markers:
        v_tip = np.array([ 2.78072234,  -167.59019318, -3.84547034])  #tip vextor
        curTrans  = marker.position
        curRots   = marker.rotation

        curVtip_rot = np.dot(curRots, v_tip)
        Tip_pos = curVtip_rot+ curTrans
        # rot = np.array([marker.rotation[0],marker.rotation[1],marker.rotation[2]])
        # res = np.dot(np.array([[rot[0][0],rot[0][1],rot[0][2],marker.position[0]],[rot[1][0],rot[1][1],rot[1][2],marker.position[1]],[rot[2][0],rot[2][1],rot[2][2],marker.position[2]],[0,0,0,1]]),v_tip)
        res = Tip_pos
        x_list.append(res[0])
        y_list.append(res[1])
        z_list.append(res[2])

    f.writelines(str(res[0]))
    f.writelines(',')
    f.writelines(str(res[1]))
    f.writelines(',')
    f.writelines(str(res[2]))
    f.writelines('\n')
    time.sleep(0.005)
    
print(np.mean(np.asarray(x_list)))
print(np.mean(np.asarray(y_list)))
print(np.mean(np.asarray(z_list)))

m= np.linalg.lstsq(A_full_mat,B_full_mat,rcond=None)[0]
print(m)


