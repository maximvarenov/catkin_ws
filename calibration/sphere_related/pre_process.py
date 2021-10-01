import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# from skimage import *
import os

# img = cv2.imread('./data/wire/train/label/6.png')
 
# height,width,channel = img.shape
# for i in range(height):
#     for j in range(width):
#         b,g,r = img[i,j]
#         if b == r  or b >0: 
#             b=0
#             g=0
#             r=0
#         else:
#             b=255
#             g=255
#             r=255
                
#         img[i,j]=[r,g,b] 

# plt.figure()
# plt.imshow(img) 
# plt.show()


 

 
data_base_dir ='/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_cali/data/wire/test/'
outfile_dir ='/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_cali/data/wire/test1/'


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
    # cv2.imshow('res', img) 
    # cv2.waitKey(100)
    # height,width,channel = img.shape
    # for i in range(height):
    #     for j in range(width):
    #         b,g,r = img[i,j]
    #         if g == r  or g >10: 
    #             b=0
    #             g=0
    #             r=0
    #         else:
    #             b=255
    #             g=255
    #             r=255
                    
    #         img[i,j]=[b,g,r] 

    # im = Image.fromarray(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    data = cv2.resize(gray,dsize=(256,256),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)

    out_img_name=os.path.join(outfile_dir, str(file) )
    print(str(file))
    # im.save(out_img_name)
    cv2.imwrite(out_img_name, data) 