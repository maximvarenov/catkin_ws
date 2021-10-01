from scipy.spatial.transform import Rotation as R
import numpy as np


 

rotation = np.array([[0.934096, 0.00593702, 0.356973], 
    [-0.355991, 0.0914439, 0.930004],
    [-0.0271216 ,-0.995793 ,0.0875309 ]])

r = R.from_matrix(rotation) 
print(r.as_euler('zyx', degrees=True))


# point = np.array([ -22.70811267 ,100.11531197, 1867.78497097, 1.0])
# point = np.transpose(point)
# result = np.dot(rotation, point)
# print(result)

 
