


import cv2 as cv
import os


def crop_image(image_dir, output_path, size):    

    file_path_list = []
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        file_path_list.append(file_path)
 
    for counter, image_path in enumerate(file_path_list):
        image = cv.imread(image_path)
        h, w = image.shape[0:2]
        h_no = h // size
        w_no = w // size

        for row in range(0, h_no):
            for col in range(0, w_no):
                cropped_img = image[100 : 350, 220 :450, : ]
                cv.imwrite(output_path + str(counter)  + ".png",
                           cropped_img)


if __name__ == "__main__":
    image_dir = "/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/calibration/img/"
    output_path = "/media/ruixuan/Volume/ruixuan/Documents/catkin_b/src/calibration/img_crop/"
    size = 227
    crop_image(image_dir, output_path, size) 