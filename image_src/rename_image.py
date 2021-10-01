import os


path = r'/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_sawbone/data/train/raw_full/'  


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

count= 0
for file in sorted_file:   
    Olddir=os.path.join(path,file)   
    if os.path.isdir(Olddir):  
        continue
    filename=os.path.splitext(file)[0]   
    filetype=os.path.splitext(file)[1]   
    Newdir=os.path.join(path,str(count) +filetype)  
    os.rename(Olddir,Newdir)
    count+=1

