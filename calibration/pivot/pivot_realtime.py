import atracsys.stk as tracker_sdk
import numpy as np
import os
from collections import deque
#record pose and compute pivot result with each fiducials


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# def getABMat(frame):
#     p1= [frame.fiducials[0].position[0], frame.fiducials[0].position[1], frame.fiducials[0].position[2]]
#     p2= [frame.fiducials[1].position[0], frame.fiducials[1].position[1], frame.fiducials[1].position[2]]
#     p3= [frame.fiducials[2].position[0], frame.fiducials[2].position[1], frame.fiducials[2].position[2]]
#     #p4= [frame.fiducials[3].position[0], frame.fiducials[3].position[1], frame.fiducials[3].position[2]]

#     p1 = np.asarray(p1)
#     p2 = np.asarray(p2)
#     p3 = np.asarray(p3)
#     #print (p1,p2,p3,p4)
#     cenPt = np.mean(np.asarray([p1,p2,p3]),axis=0)

#     vec1=p2-p1
#     vec2=p3-p1
#     vecNorm= np.cross(vec2, vec1)
#     Zaxis=normalize(vecNorm)
#     Yaxis=normalize(vec2)
#     Xaxis =normalize(np.cross(Yaxis,Zaxis))
#     Amat=np.asarray([Xaxis,Yaxis,Zaxis])
#     RAmat =np.transpose(Amat)  

#     b=-cenPt
#     return RAmat, b
 

def marker_pose(frame):
    for marker in frame.markers:
        rot = np.array([marker.rotation[0],marker.rotation[1],marker.rotation[2]])
        trans_matrix =np.array([[rot[0][0],rot[0][1],rot[0][2],marker.position[0]],[rot[1][0],rot[1][1],rot[1][2],marker.position[1]],[rot[2][0],rot[2][1],rot[2][2],marker.position[2]]]) 
        # print(marker.position)
        return trans_matrix


# Replace atracsys.stk with atracsys.ftk for the fusionTrack.
def exit_with_error(error, tracking_system):
    print(error)
    errors_dict = {}
    if tracking_system.get_last_error(errors_dict) == tracker_sdk.Status.Ok:
        for level in ['errors', 'warnings', 'messages']:
            if level in errors_dict:
                print(errors_dict[level])
    exit(1)


def pivot_calibration(transforms):
    p_t = np.zeros((3, 1))
    T = np.eye(4)

    A = []
    b = []

    for item in transforms:
        i = 1
        A.append(np.append(item[0, [0, 1, 2]], [-1, 0, 0]))
        A.append(np.append(item[1, [0, 1, 2]], [0, -1, 0]))
        A.append(np.append(item[2, [0, 1, 2]], [0, 0, -1]))
        b.append((item[0, [3]]))
        b.append((item[1, [3]]))
        b.append((item[2, [3]]))

    x = np.linalg.lstsq(A, b, rcond=None)
    result = (x[0][0:3]).flatten() * -1
    p_t = np.asarray(result).transpose()
    T[:3, 3] = p_t.T
    return p_t, T,x


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
for geometry in ['geometry001.ini', 'geometry002.ini', 'geometry003.ini', 'geometry004.ini', 'geometry006.ini']:
    if tracking_system.set_geometry(os.path.join(path, geometry)) != tracker_sdk.Status.Ok:
        exit_with_error("Error, can't create frame object.", tracking_system)



A_full_mat= []
B_full_mat =[] 
transforms = list()

for x in range(2000):
    tracking_system.get_last_frame(frame)
    trans_matrix = marker_pose(frame)
    T = np.eye(4)  
    data = trans_matrix.reshape((3, 4))
    T[:3, :4] = data
    transforms.append(T) 



p_t, T, x = pivot_calibration(transforms)
print(x)
print('Calibtration matrix T')
print(T)
print('position of tip')
print(-x[0][3],-x[0][4],-x[0][5])

