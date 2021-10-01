import cv2
import numpy as np
from skimage import *
import os
 

 
data_base_dir ="/media/ruixuan/Volume/ruixuan/Pictures/auto_cali2/raw_4/"
outfile_dir ='/media/ruixuan/Volume/ruixuan/Pictures/auto_cali2/test/'


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
    image=cv2.imread(os.path.join(data_base_dir, file))
    image = image[0:360, 0:240]
    # cv2.imshow("raw",image)
    # cv2.waitKey(100)
    # rectangle = np.zeros(image.shape[:2], dtype = "uint8")
    # mask = cv2.rectangle(rectangle, (150,50), (550, 350), (255,255,255), -1)
    # masked = cv2.bitwise_and(image ,image, mask =mask)

    # ret, binary = cv2.threshold(masked,40,255,cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15)) 
    # closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # res =cv2.bitwise_not(closing,closing)

    # ret, binary = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    # # binary = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,10)
    # cv2.imshow("demo",binary)
    # cv2.waitKey(100)
    out_img_name=os.path.join(outfile_dir, str(file) )
    print(str(file))
    cv2.imwrite(out_img_name, image) 