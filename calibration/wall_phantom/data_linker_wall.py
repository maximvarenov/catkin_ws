import os

try:
    os.remove('./final_data_wall.txt')
except: 
    pass


def extract_data_from_txtfile(file):
    data = open(file)
    data_content = data.read().splitlines()
    data.close()

    return data_content


def make_list(filename):
    data_content = extract_data_from_txtfile(filename)

    extracted_data = []

    for coordinates in data_content:
        coordinates = coordinates.split()
        extracted_data.append(coordinates)
        
    return extracted_data

image_data = make_list('./point_coordinates_plane_TEST.txt')
fusion_data = make_list('./cal3.txt')

for img_data, ft_data in zip(image_data, fusion_data):
    
    px1_img = int(img_data[0])
    py1_img = int(img_data[1])
    px2_img = int(img_data[2])
    py2_img = int(img_data[3])

    px = float(ft_data[0])
    py = float(ft_data[1])
    pz = float(ft_data[2])
    qx = float(ft_data[3])
    qy = float(ft_data[4])
    qz = float(ft_data[5])
    qw = float(ft_data[6])
    
    if px1_img != 0 and py1_img != 0:
        if px != 0 and py != 0 and pz != 0:             
            file2write = open('./final_data_wall.txt','a')
            file2write.write(str(px1_img) + ' ' + str(py1_img) + ' ' + str(px2_img) + ' ' + str(py2_img) + ' ' + str(px) + ' ' + str(py) + ' ' + str(pz) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(qw) + '\n')
            file2write.close()