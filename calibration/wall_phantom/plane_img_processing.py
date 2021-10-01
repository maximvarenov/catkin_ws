import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np 
import time
import os 

## Remove previous result
try:
    os.remove('./point_coordinates_plane_TEST.txt')
except: 
    pass

start_time = time.time()


def make_roi ():
    roi_image = np.zeros((480,640,3), np.uint8)
    start_point = (210,50)
    end_point = (465,400)
    color = (255,255,255)
    thickness = -1
    roi_image = cv2.rectangle(roi_image, start_point, end_point, color, thickness)
    return roi_image


def apply_roi(img,roi_img):
    roi  = ((img/255) * roi_img) 
    return roi


def show_roi(img,index):
    start_point = (210,50)
    end_point = (465,400)
    color = (0,0,255)
    thickness = 1
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.imwrite(str(index) + '_ROI.png',img)


def gamma_correction(gamma,img):
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8') 
    return gamma_corrected


def mask_gamma_correction(img):
    threshold = 30
    mask = img <= threshold 
    img[mask] = (0)
    return img


def show_coordinates(img,cX,cY):        # The lines in comments are for coordinates to top and left on screen, not a rectangle
    x = cX
    y = cY
    
    start_point1 = (y,0)
    end_point1 = (y,x)
    start_point2 = (0,x)
    end_point2 = (y,x)

    color = (0,0,255)
    thickness = 1
    coord_image = cv2.rectangle(img, start_point1, end_point1, color, thickness)
    coord_image = cv2.rectangle(coord_image,start_point2,end_point2,color,thickness)

    return coord_image


def object_detection(img,index):
    ## First object detection
    objects_area = []
    objects_coord = []
    objects_xcoord = []
    objects_ycoord = []
    objects_aspect_ratio = []
    objects_width = []
    objects_height = []
    coords_of_bb = []           # bb = bounding boxes
    distance_bb = []

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    object_counter = len(contours)

    i = 0
    area_counter = 0
    area_threshold = 15
    objects_amount = 0
    point_AR_threshold = 4

    for c in contours:     
        area = cv2.contourArea(contours[i])
        #print(area)
        objects_area.append(area)
        if area > area_threshold:                   ## Important parameter, defines the minimum size of small to dots to be ignored !!
            area_counter += 1
        i += 1
        if object_counter > area_counter:           ## Ignore small dots
            objects_amount = area_counter
        else:
            objects_amount = object_counter
    
    ## Second we check the objects 
    a = 0 
    objects_area.clear() ## remove too small values from area list by clearing it, and adding values > threshold back
    for c in contours:
        area = cv2.contourArea(contours[a]) 
        if area > area_threshold:                               
            objects_area.append(area)
        # calculate moments for each contour
            M = cv2.moments(c)
            x,y,w,h = cv2.boundingRect(c)

            coords_of_bb.extend([x,y,x+w,y+h])

        # calculate x,y coordinate of center
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(img, (cX, cY), 2, (0, 0, 255), -1)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

                
                aspect_ratio = float(w)/h
                objects_coord.extend([cX,cY])
                objects_xcoord.append(cX)
                objects_ycoord.append(cY)
                objects_aspect_ratio.append(aspect_ratio)
                objects_width.append(w)
                objects_height.append(h)
            except:
                pass
        a += 1

    
    ## Thrid we work with the information of possible multiple objects     
    one_obj = False
    two_obj = False
    bad_img = False

    if objects_amount == 0:
        bad_img = True
        return [one_obj, two_obj, bad_img]

    if objects_amount == 1:
        one_obj = True
        if objects_area[0] >= 25:                   # Condition to ignore small dots 
            if h > 15:                              # If this condition, then object is not horizontally and we must check the orientation, we do this by checking 1 line in the boundingbox
                img_array = np.array(img)
                height_index = 1
                line_values = []
                for i in range(y+1,y+h):
                    for j in range(220,221):
                        pixel_value = img_array[i][j]
    
                        if pixel_value[0] != 0:
                            line_values.append(height_index)

                        height_index += 1
                
                try:
                    avg = sum(line_values) / len(line_values)
                except:
                    # print('No avg line value detected')
                    bad_img = True
                    return [one_obj, two_obj, bad_img]

                if avg > h/2:
                    # Calculation of 2 points 
                    x1 = int(x + (0.25 * w))
                    y1 = int(y + (0.75 * h))

                    x2 = int(x + (0.75 * w))
                    y2 = int(y + (0.25 * h))

                elif avg < h/2:
                    x1 = int(x + (0.25 * w))
                    y1 = int(y + (0.25 * h))

                    x2 = int(x + (0.75 * w))
                    y2 = int(y + (0.75 * h))
                
                final_x1 = x1 - 217
                final_y1 = y1 - 33
                final_x2 = x2 - 217
                final_y2 = y2 - 33

                # y1 and x1 are reversed, different xy coordinate frame for both functions!!
                img = show_coordinates(img,y1,x1)
                img = show_coordinates(img,y2,x2)

                return [one_obj, two_obj, bad_img, final_x1, final_y1, final_x2, final_y2]

            elif h <= 15:
                x1 = int(x + (0.25 * w))
                y1 = int(y + (h/2))

                x2 = int(x + (0.75 * w))
                y2 = int(y + (h/2))

                final_x1 = x1 - 217
                final_y1 = y1 - 33
                final_x2 = x2 - 217
                final_y2 = y2 - 33


                img = show_coordinates(img,y1,x1)
                img = show_coordinates(img,y2,x2)

                return [one_obj, two_obj, bad_img, final_x1, final_y1, final_x2, final_y2]
        else:
            bad_img = True
            return [one_obj, two_obj, bad_img]
    else:
        bad_img = True
        return [one_obj, two_obj, bad_img]

      

def image_processing():
    goodImages = []
    roi_image = make_roi()
    img_index = 1

    while img_index <= 2874:                   # define how many images you want to correct
        print('//// ' + str(img_index)) 
        obj = 0
        img_name ='./cal3/' +str(img_index)+'.png'
        img = cv2.imread(img_name) 
        cv2.imshow('Image',img)
        cv2.waitKey(5)
        img = gamma_correction(2,img)
        img = apply_roi(img,roi_image)
        img = mask_gamma_correction(img)
        img = gamma_correction(0.1,img) 
        object_info = object_detection(img,img_index)
        

        if object_info[0] == True and object_info[2] == False:
            goodImages.append(img_index)
            # cv2.imwrite(str(img_index) + '_processed.png', img)
            cv2.imshow('Final image',img)
            cv2.waitKey(5)
 
            file2write = open('./point_coordinates_plane_TEST.txt','a')
            file2write.write(str(object_info[3]) + ' ' + str(object_info[4]) + ' ' + str(object_info[5]) + ' ' + str(object_info[6]) + '\n')
            file2write.close()

        else:

            file2write = open('./point_coordinates_plane_TEST.txt','a')
            file2write.write(str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + '\n')
            file2write.close()
        
        img_index += 1


coords = image_processing()
print("--- %s seconds ---" % (time.time() - start_time))