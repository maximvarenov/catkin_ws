import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import os

 
data_base_dir ='/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_cali/data/wire/train/image1/'
outfile_dir ='/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_cali/data/wire/train/test/'


filelist = os.listdir(data_base_dir) 

sort_num_list = []
for file in filelist:
    sort_num_list.append(int(file.split('.png')[0]))      
    sort_num_list.sort() 

sorted_file = []
for sort_num in sort_num_list:
    for file in filelist:
        if str(sort_num) == file.split('.png')[0]:
            sorted_file.append(file) 


print(sorted_file)
for file in sorted_file:  
    img=cv2.imread(os.path.join(data_base_dir, file)) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    cropped = gray[80:80+256, 200:456]

    out_img_name=os.path.join(outfile_dir, str(file) )
    print(str(file)) 
    cv2.imwrite(out_img_name, cropped) 