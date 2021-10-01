import os 
import sys
import string
import numpy as np 
from scipy.spatial.transform import Rotation as R
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 




a = []
b = []
count = 0
# re_path ='/media/ruixuan/Volume/ruixuan/Pictures/auto_cali_sphere/cal_1/'

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img) 
        global point, count  
        # cv2.imwrite(re_path+str(count)+'.png', img)
        point = True
        print(y,x,) 
        file2.write(str(y))
        file2.write(',') 
        file2.write(str(x))
        file2.write(',') 
    

filename = "/media/ruixuan/Volume/ruixuan/Pictures/icar/cross/cal1.txt"
file=open(filename)
lines=file.readlines()

path='/media/ruixuan/Volume/ruixuan/Pictures/icar/cross/cal1/'
filelist = os.listdir(path) 
sort_num_list = []
for file in filelist:
    sort_num_list.append(int(file.split('.png')[0]))      
    sort_num_list.sort() 
sorted_file = []
for sort_num in sort_num_list:
    for file in filelist:
        if str(sort_num) == file.split('.png')[0]:
            sorted_file.append(file) 
file2 = open("./cal_pre1.txt","a")  


for i,line in zip(sorted_file,lines): 
    point = False
    split_string = str.split(line,' ') 
    x = float(split_string[0]) 
    y = float(split_string[1])
    z = float(split_string[2])
    qx = float(split_string[3])
    qy = float(split_string[4])
    qz = float(split_string[5])
    qw = float(split_string[6]) 
    r = R.from_quat([qx, qy, qz, qw])
    rot = r.as_matrix()

    
    img=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR) 
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    # image_arr = np.array(gray)
    # if len(a) >0 :
    #     print( np.asarray(b)[-1],np.asarray(a)[-1]) 
    #     print(image_arr[ np.asarray(b)[-1]][np.asarray(a)[-1]])
    #     print(image_arr.shape)

    cv2.imshow("image", img)
    cv2.waitKey(50)   
    count +=1
    
    str_list = [rot[0][0],rot[0][1],rot[0][2],x,rot[1][0],rot[1][1],rot[1][2],y,rot[2][0],rot[2][1],rot[2][2],z] 
    if point:
        print(rot[0][0],rot[0][1],rot[0][2],x,rot[1][0],rot[1][1],rot[1][2],y,rot[2][0],rot[2][1],rot[2][2],z)
        file2.write(str(str_list))   
        file2.write('\n') 

