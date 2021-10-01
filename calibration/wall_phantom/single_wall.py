import numpy as np
from math import sqrt, atan2, pi, sin, cos
import os


def extract_data_from_txtfile(file):
    lines = open(file)
    data = lines.read().splitlines()
    lines.close()

    return data


def get_parameters(filename):
    data = extract_data_from_txtfile(filename)
    list_of_parameters = []
    for item in data:
        item = item.split()
        item = [float(i) for i in item]
        list_of_parameters.append(item)

    return list_of_parameters


def get_transformation(x, y, z, qx, qy, qz, qw):
    norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm
    qw = qw / norm

    a = 1 - 2 * (qy ** 2 + qz ** 2)
    b = 2 * (qx * qy + qw * qz)
    c = 2 * (qx * qz - qw * qy)

    d = 2 * (qx * qy - qw * qz)
    e = 1 - 2 * (qx ** 2 + qz ** 2)
    f = 2 * (qy * qz + qw * qx)

    g = 2 * (qx * qz + qw * qy)
    h = 2 * (qy * qz - qw * qx)
    j = 1 - 2 * (qx ** 2 - qy ** 2)

    rot = np.array([[a, b, c], [d, e, f], [g, h, j]])
    translat = np.array([x, y, z])

    return rot, translat


def get_inverse_transformation(R, t):
    inv_rot = np.linalg.inv(R)
    inv_transl = -inv_rot.dot(t)

    return inv_rot, inv_transl


def points_to_vector(p1, p2):
    vector = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
    return vector


def order_coordinates(x_coord, y_coord):
    error_margin = 10
    count_x = -6
    count_y = -6
    for x in x_coord:
        for xx in x_coord:
            if abs(x - xx) < error_margin:
                count_x += 1
    for y in y_coord:
        for yy in y_coord:
            if abs(y - yy) < error_margin:
                count_y += 1
    count_x /= 2
    count_y /= 2

    if count_x > 0 and count_y == 6:
        # P1
        d_list = []
        for j in range(len(x_coord)):
            d = sqrt(x_coord[j] ** 2 + y_coord[j] ** 2)
            d_list.append(d)
        index_P1 = d_list.index(min(d_list))
        P1 = [x_coord[index_P1], y_coord[index_P1]]

        # P2 & P3
        p12 = []
        for j in range(len(x_coord)):
            if abs(y_coord[j] - P1[1]) < error_margin and x_coord[j] > P1[0]:
                p12.append([x_coord[j], y_coord[j]])
        if p12[0][0] < p12[1][0]:
            P2 = p12[0]
            P3 = p12[1]
        else:
            P2 = p12[1]
            P3 = p12[0]

        # P4
        P4 = [0, 0]
        for j in range(len(x_coord)):
            if abs(x_coord[j] - P1[0]) < error_margin and y_coord[j] > P1[1]:
                P4 = [x_coord[j], y_coord[j]]

        # P5 & P6
        p56 = []
        for j in range(len(x_coord)):
            if abs(y_coord[j] - P4[1]) < error_margin and x_coord[j] > P4[0]:
                p56.append([x_coord[j], y_coord[j]])
        if p56[0][0] < p56[1][0]:
            P5 = p56[0]
            P6 = p56[1]
        else:
            P5 = p56[1]
            P6 = p56[0]

    else:
        x_max = x_min = x_coord[0]
        y_max = y_min = y_coord[0]          # y-axis of image is downwards so y_max (top) is the smallest y-value!!
        index_x_max = index_x_min = index_y_max = index_y_min = 0
        for j in range(len(x_coord)):
            if x_coord[j] > x_max:
                x_max = x_coord[j]
                index_x_max = j
            if x_coord[j] < x_min:
                x_min = x_coord[j]
                index_x_min = j
        for j in range(len(y_coord)):
            if y_coord[j] < y_max:
                y_max = y_coord[j]
                index_y_max = j
            if y_coord[j] > y_min:
                y_min = y_coord[j]
                index_y_min = j

        P_top = [x_coord[index_y_max], y_coord[index_y_max]]
        P_bottom = [x_coord[index_y_min], y_coord[index_y_min]]
        P_left = [x_coord[index_x_min], y_coord[index_x_min]]
        P_right = [x_coord[index_x_max], y_coord[index_x_max]]

        d_left = sqrt((P_top[0] - P_left[0]) ** 2 + (P_top[1] - P_left[1]) ** 2)
        d_right = sqrt((P_top[0] - P_right[0]) ** 2 + (P_top[1] - P_right[1]) ** 2)

        if d_left < d_right:
            P1 = P_top
            P3 = P_right
            P4 = P_left
            P6 = P_bottom
        else:
            P1 = P_left
            P3 = P_top
            P4 = P_bottom
            P6 = P_right

        y_P5 = 0
        index_P5 = 0
        sum_of_indices = 0
        for j in range(len(y_coord)):
            if y_coord[j] > y_P5 and j not in (index_x_max, index_x_min, index_y_max, index_y_min):
                y_P5 = y_coord[j]
                index_P5 = j
            sum_of_indices += j
        index_P2 = sum_of_indices - index_P5 - index_x_max - index_x_min - index_y_max - index_y_min

        P2 = [x_coord[index_P2], y_coord[index_P2]]
        P5 = [x_coord[index_P5], y_coord[index_P5]]

    return P1, P2, P3, P4, P5, P6


def get_ratio(P1, P2, P3):
    d_12 = sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P1[1]) ** 2)
    d_13 = sqrt((P3[0] - P1[0]) ** 2 + (P3[1] - P1[1]) ** 2)
    ratio = d_12 / d_13

    return ratio


parameter_list = get_parameters('./final_data_wall.txt')
wall_parameters = get_parameters('./ft_data_for_second_marker_planeV2.txt')
scaling = get_parameters('./point_coordinates_scale_factorsV2.txt')


B = np.array([0, 100])
C = np.array([15, 0])
list_of_distances = []              # list with vertical distances (in pixels) between the wires in the image
for i in range(len(scaling)):
    x_coord = []
    y_coord = []
    for j in range(len(scaling[i])):
        if j % 2:
            y_coord.append(scaling[i][j])
        else:
            x_coord.append(scaling[i][j])
    points = order_coordinates(x_coord, y_coord)

    # middle points of top, middle and bottom wires in image coordinate system
    uv1 = np.array(points[1])
    uv2 = np.array(points[4])
    # ratios
    r1 = get_ratio(points[0], points[1], points[2])
    r2 = get_ratio(points[3], points[4], points[5])
    # x & y coordinates of middle points in phantom coordinate system
    x1_phantom = B[0] + r1 * (C[0] - B[0])
    y1_phantom = B[1] + r1 * (C[1] - B[1])
    x2_phantom = B[0] + r2 * (C[0] - B[0])
    y2_phantom = B[1] + r2 * (C[1] - B[1])
    # middle points in phantom coordinate system
    P1 = np.array([x1_phantom, y1_phantom, 0])
    P2 = np.array([x2_phantom, y2_phantom, -10])

    d_phantom = sqrt(np.sum((P1-P2)**2, axis=0))
    factor = 10 / d_phantom

    d = sqrt(np.sum((uv1-uv2)**2, axis=0))
    d *= factor
    list_of_distances.append(d)

y_im = 0
for item in list_of_distances:
    y_im += item

y_im = y_im/len(list_of_distances)      # delta y in image (vertical distance (in pixels) between the two fiducials)
y_ph = 10                               # delta y in phantom (known geometry)

# this is the axial pixel-to-millimeter ratio s_y
s_y = y_ph/y_im


# rotation and translation matrix from phantom to world frame
# R_PW, t_PW = get_transformation(wall_parameters[0][0], wall_parameters[0][1], wall_parameters[0][2],
#                                 wall_parameters[0][3], wall_parameters[0][4], wall_parameters[0][5],
#                                 wall_parameters[0][6])
t_PW = np.array([wall_parameters[0][0], wall_parameters[0][1], wall_parameters[0][2]])
R_PW = np.array([
    [wall_parameters[0][3], wall_parameters[0][4], wall_parameters[0][5]],
    [wall_parameters[0][6], wall_parameters[0][7], wall_parameters[0][8]],
    [wall_parameters[0][9], wall_parameters[0][10], wall_parameters[0][11]]
]) 
# 3 random points on plane in phantom frame (plane equation: z = 0):
Q_P = np.array([0, 0, 0])
R_P = np.array([1, 0, 0])
S_P = np.array([0, 1, 0])
# the same points with correction because of the marker position:
rot_marker = np.array([[1, 0, 0], [0, cos(pi/2), -sin(pi/2)], [0, sin(pi/2), cos(pi/2)]])  # rotation about the x-axis
# rot_marker = np.array([[cos(pi/2), 0, sin(pi/2)], [0, 1, 0], [-sin(pi/2), 0, cos(pi/2)]])  # rotation about the y-axis
# rot_marker = np.array([[cos(pi/2), -sin(pi/2), 0], [sin(pi/2), cos(pi/2), 0], [0, 0, 1]])  # rotation about the z-axis
# rot_marker = np.linalg.inv(rot_marker)
t_marker = np.array([0, 0, -140])
Q_P = rot_marker.dot(Q_P) + t_marker
R_P = rot_marker.dot(R_P) + t_marker
S_P = rot_marker.dot(S_P) + t_marker
# same points in world frame:
Q = R_PW.dot(Q_P) + t_PW
R = R_PW.dot(R_P) + t_PW
S = R_PW.dot(S_P) + t_PW


row = []
row1 = []
d1 = []
for i in range(len(parameter_list)):
    # slope of line on image: m = (y2-y1)/(x2-x1)
    m = ((parameter_list[i][3]) - (parameter_list[i][1])) / \
        ((parameter_list[i][2]) - (parameter_list[i][0]))
    # rotation and translation matrix from sensor to world frame:
    R_SW, t_SW = get_transformation((parameter_list[i][4]), (parameter_list[i][5]),
                                    (parameter_list[i][6]), (parameter_list[i][7]),
                                    (parameter_list[i][8]), (parameter_list[i][9]),
                                    (parameter_list[i][10]))

    # rotation and translation matrix from world to sensor frame:
    R_WS, t_WS = get_inverse_transformation(R_SW, t_SW)
    print(t_WS)

    # points in sensor frame:
    Q_S = R_WS.dot(Q) + t_WS
    R_S = R_WS.dot(R) + t_WS
    S_S = R_WS.dot(S) + t_WS
    # normal vector of plane in sensor frame:
    u_S = points_to_vector(Q_S, R_S)
    v_S = points_to_vector(Q_S, S_S)
    n_S = np.cross(u_S, v_S, axis=0)

    n1 = n_S[0]
    n2 = n_S[1]
    n3 = n_S[2]
    row.append([n1, n2, n3, m*n1, m*n2, m*n3])
    row1.append(n_S)

    d = -(Q_S[0]*n_S[0] + Q_S[1]*n_S[1] + Q_S[2]*n_S[2])
    d1.append(d)

# A from Ax = 0, ||Ax|| should be minimized so svd is used
A = np.vstack([row])

U_svd, s, V_svd = np.linalg.svd(A)
index = min(range(len(s)), key=s.__getitem__)
X = V_svd[:, index]


Xn = X / np.linalg.norm(X[0:3])
U = Xn[0:3]
k = np.linalg.norm(Xn[3:6])         # scaling factor ratio
V = Xn[3:6] / k

Z = np.cross(U, V)
Z = Z / np.linalg.norm(Z)
Y = np.cross(Z, U)
Y = Y / np.linalg.norm(Y)

rotation = np.array([
    [U[0], Y[0], Z[0]],
    [U[1], Y[1], Z[1]],
    [U[2], Y[2], Z[2]]
])

s_x = s_y / k


d = np.sum(d1) / len(d1)

row2 = []
for i in range(len(parameter_list)):
    # coordinates of a (random) point on the line in the image:
    x = (parameter_list[i][0])
    y = (parameter_list[i][1])
    # normal vector of plane:
    n = row1[i]

    row2.append([d - s_x*x*(U.dot(n)) - s_y*y*(V.dot(n))])


# A & b from Ax = b, ||Ax-b|| should be minimized so the least squares method is used
A = np.vstack([row1])
b = np.vstack([row2])

translation = np.linalg.lstsq(A, b, rcond=None)[0]
rotation = rotation.T

transformation = np.vstack([np.append(rotation, translation, axis=1), [0, 0, 0, 1]])


np.set_printoptions(suppress=True)
print()
print('Scaling factor x (lateral pixel-to-millimeter ratio): ', s_x)
print('Scaling factor y (axial pixel-to-millimeter ratio):   ', s_y, '\n')
print('This is the transformation matrix:\n', transformation, '\n')


#####################################################################################
# Euler angles (rotation about moving axes: z-x'-z")
#####################################################################################

b = atan2(sqrt(rotation[2][0] ** 2 + rotation[2][1] ** 2), rotation[2][2])          # atan2 -> (sin, cos)
a = atan2(rotation[0][2] / sin(b), -rotation[1][2] / sin(b))
g = atan2(rotation[2][0] / sin(b), rotation[2][1] / sin(b))

print('These are the Euler angles (with intrinsic rotations z-x\'-z\"): \n',
      'alfa = ', round(a/pi, 3), '* pi', '\n', 'beta = ', round(b/pi, 3), '* pi', '\n',
      'gamma = ', round(g/pi, 3), '* pi')


a = transformation[0][0]
b = transformation[0][1]
c = transformation[0][2]
d = transformation[0][3]
e = transformation[1][0]
f = transformation[1][1]
g = transformation[1][2]
h = transformation[1][3]
i = transformation[2][0]
j = transformation[2][1]
k = transformation[2][2]
l = transformation[2][3]

try:
    os.remove('./transformation_data.txt')
except FileNotFoundError:
    pass

file2write = open('./transformation_data.txt', 'a')
file2write.write(str(s_x) + ' ' + str(s_y) + ' ' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' +
                 str(d) + ' ' + str(e) + ' ' + str(f) + ' ' + str(g) + ' ' + str(h) + ' ' + str(i) + ' ' + str(j) + ' '
                 + str(k) + ' ' + str(l))
file2write.close()
