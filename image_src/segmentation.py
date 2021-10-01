import os
from PIL import Image
from PIL import ImageEnhance

 
project_dir = os.path.dirname(os.path.abspath(__file__))
input = os.path.join(project_dir, '/media/ruixuan/Volume/ruixuan/Documents/database/us_image/6-1-2021/v_img1/')

 
output = os.path.join(project_dir, '/media/ruixuan/Volume/ruixuan/Documents/database/us_image/6-1-2021/v_seg/')
def modify(): 
    os.chdir(input)
 
    for image_name in os.listdir(os.getcwd()):
        print(image_name)
        im = Image.open(os.path.join(input, image_name))
        im = im.crop((220, 50, 480, 400))
        enh_bri = ImageEnhance.Brightness(im)
        brightness = 1.5
        im = enh_bri.enhance(brightness)
        enh_con = ImageEnhance.Contrast(im)
        contrast = 1.5
        im = enh_con.enhance(contrast)
        threshold = 180 
        im = im.point(lambda p: p > threshold and 255)  
        im.save(os.path.join(output, image_name))

if __name__ == '__main__':
    modify()