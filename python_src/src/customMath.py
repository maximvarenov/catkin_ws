import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import heapq
import math
import time, datetime
import os

# Numpy array to vtk points
def numpyArr2vtkPoints(npArray):
    vtkpoints = vtk.vtkPoints()    
    for i in range(npArray.shape[0]):
        #print npArray[i]
        vtkpoints.InsertNextPoint(npArray[i])       
    return vtkpoints

# fast SVD
def fastSVDlongRowshortCol(inputY):
    tdivide =min(inputY.shape)
    readyData = np.transpose(inputY)
    # covariance
    tmpCovM  = np.matmul(readyData,inputY)/tdivide
    
    # svd
    u, s, vh = np.linalg.svd(tmpCovM)
    
    # return vh = Vt
    return vh 
    
# vector transpose for bilinear PCA
def VectorTranspose2Dto2D(inputMatrix, lenOfdim3): # must be long row , short col
    Tmatrix = np.transpose(inputMatrix)
    trow, tcol =  Tmatrix.shape
    tmp3Dmat = Tmatrix.reshape(trow, tcol/lenOfdim3, lenOfdim3)
    tmp3DmatTrans = np.transpose(tmp3Dmat, (1,0,2))
    out2Dmat = tmp3DmatTrans.reshape(tmp3DmatTrans.shape[0],-1)
    return np.transpose(out2Dmat)

# VTK ID list to numpy list
def vtkIdList2PyList (vtkids):
    outList = []    
    for inx in range(vtkids.GetNumberOfIds()):
        outList.append(vtkids.GetId(inx))
    return outList


# def vtkpoints to numpy array (1D)
def vtkPoints2numpyArr(tmpPolydata):
    as_numpy = numpy_support.vtk_to_numpy(tmpPolydata.GetPoints().GetData())
    print ("vtkPoints2numpyArr converted :", as_numpy.shape)
    
    return as_numpy.flatten()

# def vtkpoints to numpy array (2D)
def vtkPoints2numpyArr2D(tmpPolydata):
    as_numpy = numpy_support.vtk_to_numpy(tmpPolydata.GetPoints().GetData())
    print ("vtkPoints2numpyArr converted :", as_numpy.shape)
    
    return as_numpy

#
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm



def getABMat(frame):
        # print ("test", frame.fiducials[0].position[0])
    p1= [frame.fiducials[0].position[0], frame.fiducials[0].position[1], frame.fiducials[0].position[2]]
    p2= [frame.fiducials[1].position[0], frame.fiducials[1].position[1], frame.fiducials[1].position[2]]
    p3= [frame.fiducials[2].position[0], frame.fiducials[2].position[1], frame.fiducials[2].position[2]]
    #p4= [frame.fiducials[3].position[0], frame.fiducials[3].position[1], frame.fiducials[3].position[2]]

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    #print (p1,p2,p3,p4)
    cenPt = np.mean(np.asarray([p1,p2,p3]),axis=0)

    vec1=p2-p1
    vec2=p3-p1
    vecNorm= np.cross(vec2, vec1)
    Zaxis=normalize(vecNorm)
    Yaxis=normalize(vec2)
    Xaxis =normalize(np.cross(Yaxis,Zaxis))
    Amat=np.asarray([Xaxis,Yaxis,Zaxis])
    RAmat =np.transpose(Amat)
    # print("Tran Amat", RAmat)    

    b=-cenPt
    return RAmat, b




##---------------------------------------------------------------------------
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


##---------------------------------------------------------------------------
def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x=q[0]
        y=q[1]
        z=q[2]
        w=q[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians



#---------------------------------------------------------------------
#   generate grid  
#
#   p1-----------p4(unknown)
#   -           -
#   -           -
#   -           -
#   p2----------p3
# 
#  (lengthCount) * (shortCount) = final grid points. 
#  Be aware of above 
#
def generateGrid( threepoints, lengthCount, shortCount): # threepoints is numpy array
    # assume the three points have a certain order !!!
    p1 = threepoints[0]
    p2 = threepoints[1]
    p3 = threepoints[2]
    p4 = p1+p3-p2
    v32 = p3-p2 # vector
    v12 = p1-p2 # vector
    d32 = np.linalg.norm(v32)
    d12 = np.linalg.norm(v12)
    N32 = normalize(v32)
    N21 = normalize(v12)
    GridPoint =[]

    if d32 > d12: # edge D32 is longer
        stepsizeD32 = float(d32/(lengthCount-1.0))
        stepsizeD12 = float(d12/(shortCount-1.0))
        # row corresponds to longer edge! 
        # column corresponds to short edge!
        rowIncrement = stepsizeD32
        colIncrement = stepsizeD12
        rowDir = N32
        colDir = N21
    else:
        stepsizeD32 = float(d32/(shortCount-1.0))
        stepsizeD12 = float(d12/(lengthCount-1.0))
        
        rowIncrement = stepsizeD12
        colIncrement = stepsizeD32
        rowDir = N21
        colDir = N32

    
    # row corresponds to longer edge!  row range = lenghtCount
    # column corresponds to short edge! col range = shortCount
    # start from P2 as ia 
    for irow in range (lengthCount):
        for icol in range(shortCount):
            # new point = p2 + (rowInc*irow)*RowDir
            tNewPointGrid = p2+ (rowIncrement*irow*rowDir) + (colIncrement*icol*colDir)
            GridPoint.append(tNewPointGrid)
    
    return GridPoint


##------------------------------------------------------------------
##
##
## Generate Pivot points 
##
##
##-------------------------------------------------------------------
def generatePivotPoses( pPts, pDir, pRadius=1, pHeight=3, pResolution=5):

    dire = np.array(pDir)

    coneCen = pPts- (normalize(dire)*pHeight/2)
    print(coneCen)

    cone = vtk.vtkConeSource()
    cone.SetHeight( pHeight )
    cone.SetRadius( pRadius )
    cone.SetResolution( pResolution )
    cone.SetCenter(coneCen)
    cone.SetDirection(dire)
    cone.Update()

    poly = vtk.vtkPolyData()
    poly = cone.GetOutput()
    pivotPoints = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    return pivotPoints


## Generate pivot delta angles (Roll Pitch Yall)
##
##------------------------------------------------------------------------
def generatePivotDeltaRPY(pPts, pDir):
    fixpt = pPts[0]
    num_pivotPts = pPts.shape[0]
    eulerList = []
    for i in range(num_pivotPts-1):
        i+=1
        v1 = normalize(fixpt-pPts[i])
        if i ==1:
            v2 = normalize(pDir)
        else:
            v2=normalize(fixpt-pPts[i-1])

        rot21 = rotation_matrix_from_vectors ( v2, v1) # from v2 to v1 's rotation matrix'
        r = R.from_matrix(rot21)
        q = r.as_quat()
        # print ("q", q)

        euler = euler_from_quaternion(q)
        eulerList.append(euler)
        # print("euler", euler)

    return eulerList


# calculate PCA
def calPCA( inputArray ):
    (nsample, mfeatures) = inputArray.shape
    mean_vec = np.mean(inputArray, axis=0)
    meanInputArray= inputArray-mean_vec
    cov_mat = meanInputArray.T.dot(meanInputArray) / (nsample-1)
    U, s, V = np.linalg.svd(cov_mat, full_matrices = True) 
    return V,  mean_vec


def getVTKArrow(startPoint, endPoint):
    colors = vtk.vtkNamedColors()

    # Set the background color.
    colors.SetColor("BkgColor", [26, 51, 77, 255])
    # Create an arrow.
    arrowSource = vtk.vtkArrowSource()
    # Compute a basis
    normalizedX = [0] * 3
    normalizedY = [0] * 3
    normalizedZ = [0] * 3

    rng = vtk.vtkMinimalStandardRandomSequence()
    rng.SetSeed(8775070)  # For testing.

    # The X axis is a vector from start to end
    vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
    length = vtk.vtkMath.Norm(normalizedX)
    vtk.vtkMath.Normalize(normalizedX)
    

    # The Z axis is an arbitrary vector cross X
    arbitrary = [0] * 3
    for i in range(0, 3):
        rng.Next()
        arbitrary[i] = rng.GetRangeValue(-10, 10)
    vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
    vtk.vtkMath.Normalize(normalizedZ)

    # The Y axis is Z cross X
    vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
    matrix = vtk.vtkMatrix4x4()

    # Create the direction cosine matrix
    matrix.Identity()
    for i in range(0, 3):
        matrix.SetElement(i, 0, normalizedX[i])
        matrix.SetElement(i, 1, normalizedY[i])
        matrix.SetElement(i, 2, normalizedZ[i])


    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Translate(startPoint)
    transform.Concatenate(matrix)
    transform.Scale(length, length, length)

    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(transform)
    transformPD.SetInputConnection(arrowSource.GetOutputPort())

    # Create a mapper and actor for the arrow
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()
   
    mapper.SetInputConnection(transformPD.GetOutputPort())
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("Cyan"))

    # Create spheres for start and end point
    sphereStartSource = vtk.vtkSphereSource()
    sphereStartSource.SetCenter(startPoint)
    sphereStartSource.SetRadius(0.8)
    sphereStartMapper = vtk.vtkPolyDataMapper()
    sphereStartMapper.SetInputConnection(sphereStartSource.GetOutputPort())
    sphereStart = vtk.vtkActor()
    sphereStart.SetMapper(sphereStartMapper)
    sphereStart.GetProperty().SetColor(colors.GetColor3d("Yellow"))

    sphereEndSource = vtk.vtkSphereSource()
    sphereEndSource.SetCenter(endPoint)
    sphereEndSource.SetRadius(0.8)
    sphereEndMapper = vtk.vtkPolyDataMapper()
    sphereEndMapper.SetInputConnection(sphereEndSource.GetOutputPort())
    sphereEnd = vtk.vtkActor()
    sphereEnd.SetMapper(sphereEndMapper)
    sphereEnd.GetProperty().SetColor(colors.GetColor3d("Magenta"))

    # return [actor, sphereStart, sphereEnd]
    return  actor 


def createSphereActor(spPos, radius=3, color = [1.0,0.0,0.0], alpha = 1):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(spPos)
        sphere.SetRadius(radius)
        sphereMapper = vtk.vtkPolyDataMapper()
        sphereMapper.SetInputConnection(sphere.GetOutputPort())
        sphereActor = vtk.vtkActor()
        sphereActor.SetMapper(sphereMapper)
        sphereActor.GetProperty().SetColor(color)
        sphereActor.GetProperty().SetOpacity(alpha)
        return sphereActor

##
## get axex actor at a certain location and orientation
##
##
def axexPosition(axes, pos, length_axex=200):
    mat = np.hstack((axes, pos.reshape(-1,1)))
    mat = np.vstack((mat, np.array([0,0,0,1])))


    transform =vtk.vtkTransform();
    transform.SetMatrix(mat.flatten())

    axesActor = vtk.vtkAxesActor()
    axesActor.SetTotalLength(length_axex,length_axex,length_axex)
    axesActor.SetUserTransform(transform)
    return axesActor



##
## Get STL landmarks based on model PCA axes, length, width and height
##
##
def getSTLModel_Landmarker(stlnpPts, boxLen, boxwidth, boxheight):
    v, meanp = calPCA (stlnpPts)
    vX = normalize (v[0]) # X axis is the largest variation direction 
    vY = normalize (v[1]) # y axis is the middle variation direction 
    vZ = normalize (v[2]) # z axis is the small variation direction

    bLen = boxLen/2.0 # half size
    bWid = boxwidth/2.0 # half size
    bheig = boxheight/2.0 # half size

    ##-------------top view----------------
    ##  topleft -----------topRight 
    ##     |                    |
    ##     |                    |
    ##     |                    |
    ##  bottom left--------bottom right
    ##

    topLeft = meanp  + (vX*bLen) -(vY*bheig) + (vZ*bWid)
    topright = meanp + (vX*bLen) -(vY*bheig) - (vZ*bWid)
    bottomleft = meanp  - (vX*bLen) -(vY*bheig) + (vZ*bWid)
    bottomright = meanp - (vX*bLen) -(vY*bheig) - (vZ*bWid)

    return [topLeft, topright, bottomleft, bottomright]



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)
    

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''



    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''



    # get number of dimensions
    mA = A.shape[1]
    mB = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((mA+1,A.shape[0]))
    dst = np.ones((mB+1,B.shape[0]))
    src[:mA,:] = np.copy(A.T)
    dst[:mB,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:mA,:].T, dst[:mB,:].T)
        # print ("distances--------------:" , i, distances)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:mA,:].T, dst[:mB,indices].T)
        # print ("dd--------------:", dd)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # print ("orignal A", A)
    # print ("new A", src[:mA,:].T)

    T,_,_ = best_fit_transform(A, src[:mA,:].T)

    return T, distances, i



def get_distance(A, B):
   
    mA = A.shape[1]
    mB = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((mA+1,A.shape[0]))
    dst = np.ones((mB+1,B.shape[0]))
    src[:mA,:] = np.copy(A.T)
    dst[:mB,:] = np.copy(B.T)
    distances, indices = nearest_neighbor(src[:mA,:].T, dst[:mB,:].T)
    return distances




def get_average_position(new_array, distance, n):
    index = heapq.nlargest(n, range(len(distance)), distance.take)  
    index = np.sort(index)
    # print(index)
    
    x_list = []
    y_list = []
    z_list = []
    for i in range(len(index)):
        order = index[i] 
        for j in range(len(new_array)):
            if order == j:
                x_list.append(new_array[j][0])
                y_list.append(new_array[j][1])
                z_list.append(new_array[j][2])

    x = np.mean(x_list)
    y = np.mean(y_list)
    z = np.mean(z_list)
    avePos = np.array([x,y,z])

    return avePos


def get_farPoints(new_array, distance, n):
    index = heapq.nlargest(n, range(len(distance)), distance.take)  
    index = np.sort(index)
    # print(index)
    outlist = []
    for i in index:
        outlist.append(new_array[i])    

    return np.asarray(outlist)




##
## Create folder for logging data
##
##
def mkdir(usrStr=""):
    dateWithTime = datetime.datetime.now()
    date = datetime.datetime.date(dateWithTime)
    strDate = str(date) 
    path = os.getcwd()
    currentTime =  dateWithTime.strftime("%H_%M_%S")
    print(currentTime)
    # path = path + '/' + strDate + '-' + str(time.localtime().tm_hour)+ '/'    # year - month - date - hour
    path = path + '/' +'LogData/'+ usrStr +'_'+ strDate + '-' + str(currentTime)+ '/'    # year - month - date - hour
    print(path)

    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path) 

    return  path



##
## Save data as .npy
##
##
def txt(path, np_array, array_name,usrStr=""):  
    dateWithTime = datetime.datetime.now() 
    currentTime =  dateWithTime.strftime("%H_%M_%S")    
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)   

    npy_name = path + str(usrStr)+ '_'+ array_name + '_'+ str(currentTime) + '.npy'
    print(npy_name)
 
    np.save(npy_name,np_array)     



def vtkmatrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m



def determineFT_LM_orign_normal(FT_LM):

    ftpts = np.asarray(FT_LM)
    origin = np.mean(ftpts, axis=0)
    v1 = ftpts[0]-ftpts[1]
    v2 = ftpts[1]-ftpts[2]
    vecNorm= np.cross(v1, v2)
    normal=normalize(vecNorm)
    return origin, normal


class vtkPlaneClass():

    def __init__(self, origin=[0,0,0], normal=[1,0,0]):
        self.plane = vtk.vtkPlane()
        self.plane.SetOrigin(origin)
        self.plane.SetNormal(normal)

        
    def projectPoint(self, p1, p2): # intersection with line (p1----p2)
        t = vtk.mutable(0.0)
        x = [0.0, 0.0, 0.0]
        self.plane.IntersectWithLine(p1,p2,t,x)

        return  x ,  not x==[0.0, 0.0, 0.0]


    def createvtkPlanePoly(self, FT_LM, opc = 0.5):
        topright = FT_LM[1]
        topLeft = FT_LM[0]
        bottomright = FT_LM[3]
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetOrigin(topright)
        planeSource.SetPoint1(topLeft )
        planeSource.SetPoint2(bottomright )
        planeSource.Update()
        planePoly = planeSource.GetOutput()
        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(planePoly)
        PlaneActor = vtk.vtkActor() 
        PlaneActor.SetMapper(mapper)
        PlaneActor.GetProperty().SetOpacity(opc)
        return PlaneActor



def checkIf_inbound(boundPts, checkPt):
    vtkBoundPt = numpyArr2vtkPoints(np.asarray(boundPts))

    xMin, xMax, yMin, yMax, zMin, zMax = vtkBoundPt.GetBounds()
    flag = False

    if checkPt[0]>xMin and checkPt[0]<xMax and checkPt[1]>yMin and checkPt[1]<yMax and checkPt[2]>zMin and checkPt[2]<zMax :
        flag = True
    else:
        flag = False

    return flag




##------------------------------------------------------------------------
##
## Generate pivot delta angles (Roll Pitch Yall)
##
##------------------------------------------------------------------------
def generatePivotDeltaRPY_flower(pPts, pDir):
    fixpt = pPts[0]
    num_pivotPts = pPts.shape[0]
    eulerList = []
    for i in range(num_pivotPts-1):
        i+=1
        v1 = normalize(fixpt-pPts[i])
        if i ==1:
            v2 = normalize(pDir)
        else:
            v2=normalize(fixpt-pPts[i-1])

        rot21 = rotation_matrix_from_vectors ( v2, v1) # from v2 to v1 's rotation matrix'
        r = R.from_matrix(rot21)
        q = r.as_quat()
        # print ("q", q)

        euler = euler_from_quaternion(q)
        eulerList.append(euler)
    # print("euler", euler)

    return eulerList