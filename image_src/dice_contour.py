import cv2
import numpy as np

from matplotlib import pyplot as plt
 
 
def calDSI(binary_GT,binary_R):
    row, col = binary_GT.shape  
    DSI_s,DSI_t = 0.0,0.0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j]  == 255:
                DSI_t += 1
    DSI = 2*DSI_s/DSI_t
    # print(DSI)
    return DSI 
 

def calPrecision(binary_GT,binary_R):
    row, col = binary_GT.shape 
    P_s,P_t = 0.0,0.0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j]   == 255:
                P_t += 1
    Precision = P_s/P_t
    return Precision
 

def calRecall(binary_GT,binary_R):
    row, col = binary_GT.shape 
    R_s,R_t = 0.0,0.0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j]  == 255:
                R_t += 1
 
    Recall = R_s/R_t
    return Recall


def calAcc(binary_GT,binary_R):
    row, col = binary_GT.shape 
    R_s,R_t,P_s,P_t = 0.0,0.0,0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:  
                R_s += 1
            if binary_GT[i][j] == 255 :
                R_t += 1
            if binary_GT[i][j] == 0 and binary_R[i][j] == 0:
                P_s += 1
            if binary_GT[i][j] != 255 :
                P_t += 1
 
    ACC = (R_s+ P_s)/(R_t+P_t) 
    return ACC
 
 
 
if __name__ == '__main__': 
    count1 =[]
    count2 =[]
    count3 =[]
    count4 =[]
    image_number =0 
    total_number = 70
    for i in range(3,total_number):
        path1 = str('/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_new_phantom/data/test/result_480/')+str(i) + str( '_predict.png')
        path2 = str('/media/ruixuan/Volume/ruixuan/Documents/image_unet/unet_new_phantom/data/test/label_480/')+str(i) + str( '.png')
        img_GT = cv2.imread(path2,0)
        img_R  = cv2.imread(path1,0)
        #print(i)
        ret_GT, binary_GT = cv2.threshold(img_GT, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret_R, binary_R   = cv2.threshold(img_R, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
        #plt.figure()
        #plt.subplot(121),plt.imshow(binary_GT),plt.title('predict')
        #plt.axis('off')
        #plt.subplot(122),plt.imshow(binary_R),plt.title('manual ')
        #plt.axis('off')
        #plt.show()
        count1.append(calDSI(binary_GT,binary_R))
        count2.append(calPrecision(binary_GT,binary_R))
        count3.append(calRecall(binary_GT,binary_R))
        count4.append(calAcc(binary_GT,binary_R))

        print("current number:", image_number)
        print calDSI(binary_GT,binary_R)
        print calPrecision(binary_GT,binary_R)
        print calRecall(binary_GT,binary_R)
        print calAcc(binary_GT,binary_R)
        image_number+=1
    print ('image_number: ',image_number)
    print ('average DSI: ', np.mean(np.asarray(count1)))
    print ('average Precision: ', np.mean(np.asarray(count2))) 
    print ('average RECALL:  ', np.mean(np.asarray(count3)))
    print ('average ACC:  ', np.mean(np.asarray(count4)))