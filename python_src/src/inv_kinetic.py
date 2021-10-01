import numpy as np
import scipy.linalg as linalg
import math
from scipy.spatial.transform import Rotation as R

### rotation about fixed axis
r4 = R.from_euler('zxy', [0, math.pi*(0.25),  math.pi*(0.25)], degrees=True)
rm = r4.as_matrix()
trans = np.array([0.1, -0.1, 0.1])
T_rm = np.transpose(rm)
r3 =R.from_matrix(T_rm )
euler = r3.as_euler('zxy', degrees=True)
inv_trans=np.dot((-T_rm),trans)
print(euler/3.1415)
print(inv_trans)
