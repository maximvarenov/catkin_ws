import sys
import vtk
from vtk.util import numpy_support
import numpy as np
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import *

# matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.animation import TimedAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import customMath as CM
# OS
import os
from os import listdir
from os.path import isfile, join
 
from scipy.spatial.transform import Rotation as R
import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import heapq
import math
import time, datetime
import os
 
 



#---------------------------------- 
#   
# class for signal/slot
#
#----------------------------------
class clickedSignal(QObject):

    clicked = pyqtSignal(object)
 
    def __init__(self):
        # Initialize the clickedSignal as a QObject
        QObject.__init__(self)
 
    def click(self, point3D):
        ''' emit a signal '''
        self.clicked.emit(point3D)

#---------------------------------- 
#   
# class for mouse action
#
#----------------------------------
class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.AddObserver("KeyPressEvent",self.keypressEvent)
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver('LeftButtonReleaseEvent', self.LeftButtonReleaseEvent)
        
        self.leftbuttonHold = False
        self.HoldL = False
                
        self.accutrans= np.zeros(3)
        self.landMarkList = []
        self.landMarkgtList = []
        self.NumOfSTLpts = 4

        # QT signal -slot 
        # signal object
        self.Asignal = clickedSignal() 
        self.Bsignal = clickedSignal()

    # get Asignal
    def getAsignal(self):
        return self.Asignal

    def getBsignal(self):
        return self.Bsignal
    
    # set polydata
    def setPolydata(self, poly):
        self.poly = poly
        self.buildKDtree()
    
    # build KD tree for quick search
    def buildKDtree(self):
        self.kdtree = vtk.vtkKdTreePointLocator()
        self.kdtree.SetDataSet(self.poly)
        self.kdtree.BuildLocator()
    
    def keypressEvent (self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key =="l":
           self.HoldL = True
        if key =="g":
           self.HoldG = True
        return
        

    def leftButtonPressEvent(self, obj, event):
        # when pressing left button on mouse
        self.leftbuttonHold =True
        
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        
        # get 3D position in the mesh space, but maybe not on vertex
        pickedOnepoint = picker.GetPickPosition() 
        # get flag of if touching with Mesh
        self.ifTouchMesh = picker.GetActor()
       
        # If touch with mesh and pressing "L"
        if self.ifTouchMesh and self.HoldL and len(self.landMarkList)<self.NumOfSTLpts:
            # Get exact point on mesh
            iD = self.kdtree.FindClosestPoint(pickedOnepoint)
            self.pointOnMesh = np.zeros(3)
            self.kdtree.GetDataSet().GetPoint(iD,self.pointOnMesh) 

            # emit signal for slot
            self.Asignal.click(self.pointOnMesh)
            
            # only choose 4 landmark points for pre-registration
            if len(self.landMarkList)<self.NumOfSTLpts:
                self.landMarkList.append(self.pointOnMesh)
                            
            print ("LM point on STL: ", len(self.landMarkList))    


        if self.ifTouchMesh and self.HoldG and len(self.landMarkgtList)<self.NumOfSTLpts:
            # Get exact point on mesh
            iD = self.kdtree.FindClosestPoint(pickedOnepoint)
            self.pointOnMesh = np.zeros(3)
            self.kdtree.GetDataSet().GetPoint(iD,self.pointOnMesh) 

            # emit signal for slot
            self.Bsignal.click(self.pointOnMesh)
            
            # only choose 4 landmark points for pre-registration
            if len(self.landMarkgtList)<self.NumOfSTLpts:
                self.landMarkgtList.append(self.pointOnMesh)
                            
            print ("LM point on gt   STL: ", len(self.landMarkgtList))    

        self.OnLeftButtonDown()
        return
        
    # lef mouse button up
    def LeftButtonReleaseEvent(self, obj, event):
        
        #print "release"
        self.leftbuttonHold =False
        self.HoldL = False                    
        self.OnLeftButtonUp()
        return
        
    def getLandMarkList(self):
        return np.asarray(self.landMarkList)

    def getLandMarkgtList(self):
        return np.asarray(self.landMarkgtList)
    
    def clearLandmarksList(self):
        self.landMarkList = []

    def clearLandmarksgtList(self):
        self.landMarkgtList = []


#---------------------------------- 
#   
# class for geometrical model (normally in STL format)
#
#----------------------------------
class GeoModel(object):  ## ply
    def __init__(self, data = vtk.vtkPolyData()):
        self.geoData = data
        self.if_data_set = False 
    
    def loadFromFile(self, filename):       
        # load STL files 
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()        
        # assign to local variable
        self.geoData = reader.GetOutput()
        print("number of STL points", self.geoData.GetNumberOfPoints())     
        self.if_data_set = True  
 
       
    def getgeoData(self):
        return self.geoData   
    def get_if_data_set(self):
        return self.if_data_set     
 

class GTModel(object):
    def __init__(self, data = vtk.vtkPolyData()):
        self.gtData = data
        self.if_gtData_set = False 

    def loadgtFromFile(self, filename):       
        # load STL files
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()          
        # assign to local variable
        self.gtData = reader.GetOutput()
        print("number of gt points", self.gtData.GetNumberOfPoints())     
        self.if_gtData_set  = True 
       
    def getgtData(self):
        return self.gtData   
    def get_if_gt_data_set(self):
        return self.if_gtData_set     

 
#---------------------------------- 
#   
# vtk widget with renderer
#
#----------------------------------
class vtkViewerFrame(QtWidgets.QFrame):
    def __init__(self, data, data2 ):
        super(vtkViewerFrame,self).__init__()
        
        self.GeoModelData = data
        self.GTModelData = data2
        # FusionTrack/ ultrasound detected points/actors 
        # FusionTrack Landmark points

        # STL points/actors
        self.Stl_LMpts = []      
        self.StlLM_SphereActorList = []
        self.Stl_gt_LMpts=[]
        self.StlLM_gt_SphereActorList =[]
 
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(0, 0, 200)
        self.camera.SetFocalPoint(0, 0, 0) 

        #
        self.init()

    def init(self): 
        self.vtkLayout = QtWidgets.QVBoxLayout()
        self.vtkInterWidget = QVTKRenderWindowInteractor(self)
        self.vtkLayout.addWidget(self.vtkInterWidget)       
        self.setLayout(self.vtkLayout)
        
        # add renderer
        self.ren = vtk.vtkRenderer()
        self.vtkInterWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkInterWidget.GetRenderWindow().GetInteractor()
            
        # add customized interactor
        self.Mousestyle = MouseInteractorHighLightActor()
        self.Mousestyle.SetDefaultRenderer(self.ren)                
        self.iren.SetInteractorStyle(self.Mousestyle)

        # backround color
        self.ren.SetBackground(0.2,0.3,0.4)
             
        self.ren.ResetCamera()
        self.show()
        self.iren.Initialize()
 
    # get MouseStyle access
    def getMouseStyle(self):
        return self.Mousestyle

    # update the SSM data in VTK viewer
    def updateModel (self, GeoModel):
        self.GeoModelData  = GeoModel 
        # connect vtkploydata to Mouse Event
        self.Mousestyle.setPolydata(self.GeoModelData.geoData)

    def updateModelData (self, GetModeldata):
        self.GeoModelData.geoData  = GetModeldata 
        # connect vtkploydata to Mouse Event
        self.Mousestyle.setPolydata(self.GeoModelData.geoData)

    # # update the SSM data in VTK viewer
    def updateGroundtruthModel (self, GTData):
        self.GTModelData  = GTData   
        self.Mousestyle.setPolydata(self.GTModelData.gtData) 

    def updateGroundtruthModelData (self, GTData): 
        self.Mousestyle.setPolydata(self.GTModelData.gtData) 


    def updataVTKdraw(self):        
        # mapper 
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.GeoModelData.geoData)
        self.mapper.Update()
        # actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor([0.8,0,0])
        self.actor.GetProperty().SetOpacity(1)
        self.ren.AddActor(self.actor)
        self.Render()

    def upgtdataVTKdraw(self):        
        self.gtmapper = vtk.vtkPolyDataMapper()
        self.gtmapper.SetInputData(self.GTModelData.gtData)
        self.gtmapper.Update()
        # actor
        self.gtactor = vtk.vtkActor()
        self.gtactor.SetMapper(self.gtmapper)
        self.gtactor.GetProperty().SetColor([1.0,1,1])
        self.gtactor.GetProperty().SetOpacity(0.8)
        self.ren.AddActor(self.gtactor)
        self.Render() 

    ## -----------------------------------------
    ## ICP class:  update the VTk view by new data
    ##
    ##
    def updataVTK_Redraw(self, newpolydata):  
        self.updateModelData(newpolydata)     
        self.mapper.SetInputData(newpolydata)
        self.mapper.Update()        
        self.ren.ResetCamera()
        self.Render()


    def upgtdataVTK_Redraw(self, newpolydata): 
        self.updateGroundtruthModel(newpolydata)     
        self.gtmapper.SetInputData(newpolydata)
        self.gtmapper.Update()        
        self.ren.ResetCamera()
        self.Render()
        self.updateGroundtruthModelData(newpolydata)      

    ## -----------------------------------------
    ## ICP class:      vtk view render (redraw)
    ##
    ##
    def Render(self):
        self.vtkInterWidget.Render()

    ## -----------------------------------------
    ## ICP class:      reset camera
    ##
    ##
    def ResetCamera(self):
        self.ren.ResetCamera()

   
    def clearSphereActorlist_STL(self):
        for arg in self.StlLM_SphereActorList:
            self.ren.RemoveActor(arg)
         # self.ren.ResetCamera()
        self.Render() 
        self.StlLM_SphereActorList = []
        self.Stl_LMpts = []
        for arg in self.StlLM_gt_SphereActorList:
            self.ren.RemoveActor(arg)
         # self.ren.ResetCamera()
        self.Render() 
        self.StlLM_gt_SphereActorList = []
        self.Stl_gt_LMpts = []
 

    ##------------------------------------------------------------------------------------------------------
    ## -----------------------------------------
    ## ICP class: create sphere actor for both
    ##
    ##
    def createSphereActor(self, spPos, radius=2, color = [1.0,0.0,0.0], alpha = 1):
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

    ##----------------------------------------------------------------------------------------------------
    ## ICP class:   FusionTrack: create new Sphere Actor for new FT point
    ##
    ##
    def updateSpheres_Pt(self, newFTPoint, size =2 , color =[1.0,0.0,0.0] ):
        newActor = self.createSphereActor(newFTPoint, size, color = color)
        # save the newFTpoint and new actor
        self.FTpoints.append(newFTPoint)
        self.sphereActorlist.append(newActor)
        # plot 
        for arg in self.sphereActorlist:
            self.ren.RemoveActor(arg)
            self.ren.AddActor(arg)
        self.Render() 
        self.ren.ResetCamera()
        print("Current FTpoints size: ", len(self.FTpoints)) 
        self.calculateDistance( newFTPoint )


 
    #------------------------------------------------------------------------------------------------------
    #-----------------------------------------
    #   ICP class: STL: create new Sphere Actor for new STL point
    #
    def updateSphere_Pt_STL(self, newSTLpoint, size =1):
        newSTLActor = self.createSphereActor(newSTLpoint,size, color=[0.0,0.0,1.0])
        self.Stl_LMpts.append(newSTLpoint)
        self.StlLM_SphereActorList.append(newSTLActor)
        for arg in self.StlLM_SphereActorList:
            self.ren.RemoveActor(arg)
            self.ren.AddActor(arg)
        self.Render() 
        self.ren.ResetCamera() 
        print("Current Stl_LMpts size: ", len(self.Stl_LMpts))


    def updateSphere_gt_STL(self, newSTLpoint, size =1):
        newgtSTLActor = self.createSphereActor(newSTLpoint,size, color=[0.0,1.0,0.0])
        self.Stl_gt_LMpts.append(newSTLpoint)
        self.StlLM_gt_SphereActorList.append(newgtSTLActor)
        for arg in self.StlLM_gt_SphereActorList:
            self.ren.RemoveActor(arg)
            self.ren.AddActor(arg)
        self.Render() 
        self.ren.ResetCamera()
        print("Current Stl_gt_LMpts size: ", len(self.Stl_gt_LMpts))

    #------------------------------------------------------------------------------------------------------
    #-----------------------------------------
    #   ICP class: STL: create new Sphere Actor for new STL point
    #
    def plot_STL_landmark(self, newSTLpoint, size =1):
        newSTLActor = self.createSphereActor(newSTLpoint,size, color=[0.0,0.0,1.0])
        self.StlLM_SphereActorList.append(newSTLActor)
        for arg in self.StlLM_SphereActorList:
            self.ren.RemoveActor(arg)
            self.ren.AddActor(arg)
        self.Render() 
        self.ren.ResetCamera()
        # print("Current Stl_LMpts size: ", len(self.Stl_LMpts))
 
 
    ##------------------------------------------------------------------------------
    ## ICP class:  initial alignment
    ##
    ##
    def initialAlignment(self):
        # retrieval LM from mouse click 
        l_pstl = len(self.Stl_LMpts)
        l_pgtstl = len(self.Stl_gt_LMpts)  

        if l_pstl == l_pgtstl and l_pstl !=0 and l_pgtstl!=0:

            tmpSource = CM.numpyArr2vtkPoints(np.asarray(self.Stl_LMpts))  
            tmpfixed = CM.numpyArr2vtkPoints(np.asarray(self.Stl_gt_LMpts)) 
            
            landmarkTransform = vtk.vtkLandmarkTransform()
            landmarkTransform.SetSourceLandmarks(tmpSource)
            landmarkTransform.SetTargetLandmarks(tmpfixed)
            landmarkTransform.SetModeToRigidBody()
            landmarkTransform.Update()
 
            transformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
            transformPolyDataFilter.SetInputData(self.GeoModelData.geoData)
            transformPolyDataFilter.SetTransform(landmarkTransform)
            transformPolyDataFilter.Update() 
            self.LmRegPoly = transformPolyDataFilter.GetOutput()  
            self.updataVTK_Redraw(self.LmRegPoly)

            STl_lm_vtkpts = CM.numpyArr2vtkPoints(np.asarray(self.Stl_LMpts))
            STL_lm_poly = vtk.vtkPolyData()
            STL_lm_poly.SetPoints(STl_lm_vtkpts) 
            # STL_lm_poly.update()
            # print("initial ",landmarkTransform)

                  
            transformSTL_LM_Filter = vtk.vtkTransformPolyDataFilter()
            transformSTL_LM_Filter.SetInputData(STL_lm_poly)
            transformSTL_LM_Filter.SetTransform(landmarkTransform)
            transformSTL_LM_Filter.Update()
            self.clearSphereActorlist_STL()      
            outputSTL_LM_poly= transformSTL_LM_Filter.GetOutput() 

            as_numpy = numpy_support.vtk_to_numpy(outputSTL_LM_poly.GetPoints().GetData())
            newSTL_LM_pts = as_numpy.tolist()   
            for i in newSTL_LM_pts:
                self.updateSphere_Pt_STL(i)
            self.if_first_Align = True

        else:
            print("Size of Landmarks between STL and FusionTrack is not matching!!!")


        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileName("/media/ruixuan/Volume/ruixuan/Pictures/sawbone/3.stl")
        stlWriter.SetInputData(self.GeoModelData.geoData) 
        stlWriter.Write()




    def ICP_python(self):
        source = CM.vtkPoints2numpyArr2D(self.GeoModelData.geoData ) 
        target = CM.vtkPoints2numpyArr2D( self.GTModelData.gtData)
        # print ("source size: ", source.shape) 
        T, self.lastTime_distances, iterations = CM.icp(source, target, tolerance=0.000001) 
        MeanDisErr = np.mean(self.lastTime_distances)
        # print("distance Sum----: ", MeanDisErr)
      
        InvT =vtk.vtkTransform();
        InvT.SetMatrix(T.flatten()) 
        icpTransformFilter = vtk.vtkTransformPolyDataFilter()
        icpTransformFilter.SetInputData(self.LmRegPoly)
        icpTransformFilter.SetTransform(InvT.GetInverse())
        icpTransformFilter.Update()
        self.LmRegPoly = icpTransformFilter.GetOutput()              
        self.updataVTK_Redraw(self.LmRegPoly)


     
    def distance(self):
        gt = CM.vtkPoints2numpyArr2D(self.GTModelData.gtData)
        geo = CM.vtkPoints2numpyArr2D(self.LmRegPoly)
        print ("source size: ", geo.shape) 
        T, self.lastTime_distances, iterations = CM.icp(gt, geo, tolerance=0.0001) 
        MeanDisErr = np.mean(self.lastTime_distances) 
        print('min',min(self.lastTime_distances))
        print('max',max(self.lastTime_distances)) 
        print("mean ----: ", MeanDisErr)
        print('RMSE',np.sqrt(np.mean(self.lastTime_distances)))
        print('standard ', np. std(self.lastTime_distances) )
        return MeanDisErr

   
    ##------------------------------------------------------------------------------
    ## ICP class:  Create virtual plane for intersection 
    ##
    ##
    def createFT_LM_plane(self):
        if len(self.Stl_gt_LMpts)==4:
            # get orign and norm
            orig, norm = CM.determineFT_LM_orign_normal(self.Stl_gt_LMpts)
            # create vtk plane 
            self.FT_LM_plane = CM.vtkPlaneClass(orig, norm)

            # creat vtk plane actor
            self.planeActor = self.FT_LM_plane.createvtkPlanePoly(self.Stl_gt_LMpts)
            self.ren.AddActor(self.planeActor)
            self.Render() 
        else:
            print(" FT landmarks are not enough!! at least 4 points")

    def checkIf_plane_intersection(self, p1, p2):
        interflag = None
        if self.FT_LM_plane !=None:
            interPt, interflag = self.FT_LM_plane.projectPoint(p1, p2)
            return interPt, interflag
        else:
            print(" vtk plane class is not initialized!!!")
            return [], interflag


 


#-----------------------------------
#
#   Class: control panel
#
#-----------------------------------   
class controlWidget(QtWidgets.QWidget):
    def __init__(self):
        super(controlWidget, self).__init__()       
        self.init()
    
    def init(self):
        fbox = QtWidgets.QFormLayout()      
        self.btn_ICPReg = QtWidgets.QPushButton("calculate distance") 
        self.btn_InitialReg = QtWidgets.QPushButton("Initial registration")
        self.btn_ICP = QtWidgets.QPushButton("ICP registration")
 
 

        # edit box 
        self.lengthBox = QtWidgets.QLineEdit()
        self.lengthBox.setText(str(190.0))
        self.widthBox = QtWidgets.QLineEdit()
        self.widthBox.setText(str(100.0))
        self.heightBox = QtWidgets.QLineEdit()
        self.heightBox.setText(str(80.0))
        self.GridLengthCount = QtWidgets.QLineEdit()
        self.GridLengthCount.setText(str(5))
        self.GridWidthCount = QtWidgets.QLineEdit()
        self.GridWidthCount.setText(str(5))
        self.UserDefineStr = QtWidgets.QLineEdit()
        self.UserDefineStr.setText(str(""))


        # add Qwidgets to fbox
        # fbox.addRow(QtWidgets.QLabel("Fusion Track setting"), QtWidgets.QLabel("  "))      
        # fbox.addRow(QtWidgets.QLabel("Connect to FusionTrack"), self.btn_NewSuggestPoint) 

        fbox.addRow(QtWidgets.QLabel("-----------------"), QtWidgets.QLabel(" Registration ")) 
        fbox.addRow(QtWidgets.QLabel("ICP"), self.btn_ICPReg) 
        fbox.addRow(QtWidgets.QLabel("Initial registration"), self.btn_InitialReg)
        fbox.addRow(QtWidgets.QLabel("ICP registration"), self.btn_ICP)
        self.setLayout(fbox)
 
    # get button for ICP-registration
    def getBtn_ICPReg(self):
        return self.btn_ICPReg
    def getBtn_InitialReg(self):
        return self.btn_InitialReg
    def getBtn_ICP(self):
        return self.btn_ICP




#---------------------------------- 
#   
# class: matplotlib widgets
#
#----------------------------------
class matPlotwidget(QtWidgets.QWidget):
    def __init__(self):
        super(matPlotwidget, self).__init__()

        # a figure instance to plot on
        self.figure = Figure()
        self.canvas = CustomFigCanvas()
        # self.toolbar = NavigationToolbar(self.canvas, self)        
        layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def getCanvas(self):
        return self.canvas
 
    
#---------------------------------- 
#   
#  main application GUI
#
#----------------------------------
class MainWindow(QMainWindow):
    
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
        
        # STL data class instance
        self.GeoModel = GeoModel()
        self.GTModel = GTModel()  
        # initialize GUI 
        self.initUIwindow()
 
    
        
    # GUI function to define all widgets    
    def initUIwindow(self):     
                
        # vtk viewer for 3D model visualization
        self.vtkViewer = vtkViewerFrame(self.GeoModel, self.GTModel)
        self.setCentralWidget(self.vtkViewer)   
        

        # add as a CONTROL panel dock widget
        self.controlWid = controlWidget()       
        self.controlPanel = QtWidgets.QDockWidget("Control", self)
        self.controlPanel.setObjectName("Control")
        self.controlPanel.setWidget( self.controlWid )
        self.controlPanel.setFloating(False)        
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.controlPanel) 
 
   
        self.vtkViewer.getMouseStyle().getAsignal().clicked.connect(self.callback_vtkDrawLSTL)

        self.vtkViewer.getMouseStyle().getBsignal().clicked.connect(self.callback_vtkDrawLgtSTL)
        self.controlWid.getBtn_ICPReg().clicked.connect(self.callback_ICP_Reg)  
        self.controlWid.getBtn_InitialReg().clicked.connect(self.callback_Pre_Reg)  
        self.controlWid.getBtn_ICP().clicked.connect(self.callback_ICP)   
               
        
        # action: exit
        exitAction = QtWidgets.QAction( QtGui.QIcon('icon/exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.closeEventIcon )        
        
        # action: open file (vtk model, SSM )
        openFileActiongt = QtWidgets.QAction(QtGui.QIcon('icon/openfilegt.png'), '&OpenFilegt', self)
        openFileActiongt.setStatusTip('Open file gt ...')
        openFileActiongt.triggered.connect(self.showgtFileDialog)

        openFileAction = QtWidgets.QAction(QtGui.QIcon('icon/openfile.png'), '&OpenFile', self)
        openFileAction.setStatusTip('Open file ...')
        openFileAction.triggered.connect(self.showFileDialog)
                
        # menu bar
        menubar = self.menuBar()
        fileMemu = menubar.addMenu('&File')
        fileMemu.addAction(exitAction)
        fileMemu.addAction(openFileActiongt)
        fileMemu.addAction(openFileAction)

        # tool bar
        self.toolbar = self.addToolBar('&ToolBar1')     
        self.toolbar.addAction(exitAction)
        self.toolbar.addAction(openFileActiongt)
        self.toolbar.addAction(openFileAction)
       
        # status bar
        self.statusBar()      
        self.resize(1200,1000)
        self.center()
        self.setWindowTitle('Automatic ultrasound bone registration v1.0')             
        self.show()



    ##--------------------------------------------------------------------
    ## close icon event
    ##
    ##
    def closeEventIcon(self):
        print("icon close")
        if self.Thread_ReadWrench.alreadyStart() == True:
            self.Thread_ReadWrench.join()
        QtWidgets.qApp.quit()

    ##--------------------------------------------------------------------
    ##
    ## close Event reimplement
    ##
    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'Message', "Are you sure to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
        
        if reply ==QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore() 

    ##--------------------------------------------------------------------
    ##
    ## set the window to center
    ##
    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    ##--------------------------------------------------------------------
    ##
    ## the load SSM Dialog   
    ##
    def showFileDialog(self):
        chooseFilename = QtWidgets.QFileDialog.getOpenFileName(self, 'select statistical model file')
        loadfilename = str(chooseFilename[0])
        # need to convert to string for path
        # 1. load from file
        self.GeoModel.loadFromFile(loadfilename) 
        self.statusBar().showMessage(loadfilename)
        self.vtkViewer.updateModel(self.GeoModel)
        self.vtkViewer.updataVTKdraw() 
        self.vtkViewer.Render()
        # 3. reset camera   
        self.vtkViewer.ResetCamera()


    def showgtFileDialog(self):
        chooseFilename = QtWidgets.QFileDialog.getOpenFileName(self, 'select gt model file')
        loadfilename = str(chooseFilename[0]) 
        self.GTModel.loadgtFromFile(loadfilename) 
        self.statusBar().showMessage(loadfilename)
        self.vtkViewer.updateGroundtruthModel(self.GTModel )
        self.vtkViewer.upgtdataVTKdraw()  
        self.vtkViewer.Render() 
        self.vtkViewer.ResetCamera()
    

    def callback_Pre_Reg(self):
        print("button callback: pre-registration")
        self.vtkViewer.initialAlignment()

    def callback_ICP(self):
        print("button callback: ICP registration")
        self.vtkViewer.ICP_python()


    ##--------------------------------------------------------------------
    ##  
    ## Slot: receive point from Landmarks for STL
    ##
    @pyqtSlot(object)
    def callback_vtkDrawLSTL(self, receive_pt): 
        print ("received point from pyqtslot", receive_pt)
        self.vtkViewer.updateSphere_Pt_STL(receive_pt)
    

    @pyqtSlot(object)
    def callback_vtkDrawLgtSTL(self, receive_pt): 
        print ("received point from pyqtslot", receive_pt)
        self.vtkViewer.updateSphere_gt_STL(receive_pt)

    ##--------------------------------------------------------------------
    ##
    ## callback function for ICP
    ##
    def callback_ICP_Reg(self):
        print("button callback: ICP")
        self.vtkViewer.distance()        
 

 
 



# main function
def main ():    
    app = QApplication(sys.argv)
    window = MainWindow()   
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()






