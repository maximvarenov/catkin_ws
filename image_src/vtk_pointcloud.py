import numpy as np
import heapq
import time
import os
import datetime
import numpy as np
import vtk
import ros_numpy

# import 
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
import random
import os
from os import listdir
from os.path import isfile, join 
import atracsys.stk as tracker_sdk
import customMath as CM
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from random import *
import json 
from scipy.spatial.transform import Rotation as R 
import director.objectmodel as om
import director.vtkNumpy as vnp
 

class TriggerSignal(QObject): 
    cali_trigger = pyqtSignal(object) 
    coll_trigger = pyqtSignal(object) 
    def __init__(self):
        QObject.__init__(self)  
    def trigger_cali_status(self, status):
        self.cali_trigger.emit(status)  


class ROS_Class(object): 
    def __init__(self):
        super(ROS_Class, self).__init__() 
        GUI_ROS_node = rospy.init_node('GUI_ROS', anonymous=True) 
        self.trigger = TriggerSignal() 
        self.pointcloud_numpy
        self.pointcloud_vtk


    def ROS_subscriber(self): 
        subCali_status = rospy.Subscriber("/pcl_recon", PointCloud2, self.callback_pointcloud)
        # rospy.spin()
 

    def callback_pointcloud (self, data): 
        self.pointcloud_numpy = numpy_from_pointcloud2_msg(data)

    def numpy_from_pointcloud2_msg(msg): 
        pc = ros_numpy.numpify(msg)
        num_points = msg.width * msg.height
        points = np.zeros((num_points, 3))
        points[:, 0] = pc['x'].flatten()
        points[:, 1] = pc['y'].flatten()
        points[:, 2] = pc['z'].flatten()
        return points

    def get_pointcloud (self): 
        return self.pointcloud_numpy  


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
        self.clicked.emit(point3D)


#---------------------------------- 
#   
# class for mouse action
#
#----------------------------------
class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None): 
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver('LeftButtonReleaseEvent', self.LeftButtonReleaseEvent)
        
        self.leftbuttonHold = False
        self.HoldL = False
 
        self.Asignal = clickedSignal()

    # get Asignal
    def getAsignal(self):
        return self.Asignal
    
    # set polydata
    def setPolydata(self, poly):
        self.poly = poly
        self.buildKDtree()
    
    # build KD tree for quick search
    def buildKDtree(self):
        self.kdtree = vtk.vtkKdTreePointLocator()
        self.kdtree.SetDataSet(self.poly)
        self.kdtree.BuildLocator()
 
        

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
       
        self.OnLeftButtonDown()
        return
        
    # lef mouse button up
    def LeftButtonReleaseEvent(self, obj, event):
        #print "release"
        self.leftbuttonHold =False
        self.HoldL = False                    
        self.OnLeftButtonUp()
        return
         


#---------------------------------- 
#   
# vtk widget with renderer
#
#----------------------------------
class vtkViewerFrame(QtWidgets.QFrame):
    def __init__(self, data):
        super(vtkViewerFrame,self).__init__()
         
        self.Grid_SphereActor = []
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
  
        # actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.ren.AddActor(self.actor)
        self.Render()

    def updataVTK_Redraw(self, newpolydata):  
        # 1st update modeldata
        self.updateModelData(newpolydata)      
        # mapper 
        self.mapper.SetInputData(newpolydata)
        self.mapper.Update()        
        self.ren.ResetCamera()
        self.Render()

    def Render(self):
        self.vtkInterWidget.Render()
        
    def ResetCamera(self):
        self.ren.ResetCamera()


    def createSphereActor(self, spPos, radius=3, color = [1.0,0.0,0.0], alpha = 1):
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

    def updateSphere(self, newGridPoint, sizeInput = 2, colorinput =[1,1,0]):
        newActor = self.createSphereActor(newGridPoint, radius =sizeInput, color = colorinput)
        self.Grid_LM_pt.append(newGridPoint)
        self.Grid_SphereActor.append(newActor) 
        for arg in self.Grid_SphereActor:
            self.ren.RemoveActor(arg)
            self.ren.AddActor(arg)
        self.Render() 
        self.ren.ResetCamera()

    def clearSphereActorlist(self):
        for arg in self.Grid_SphereActor:
            self.ren.RemoveActor(arg) 
        self.Render() 
        self.Grid_SphereActor = []
        self.Grid_LM_pt = []



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

        # pushbutton           
        self.btn_update_pc = QtWidgets.QPushButton("update pointcloud") 
  
        # edit box 
        self.lengthBox = QtWidgets.QLineEdit()
        self.lengthBox.setText(str(100.0))
        self.widthBox = QtWidgets.QLineEdit()
        self.widthBox.setText(str(50.0))
        self.heightBox = QtWidgets.QLineEdit()
        self.heightBox.setText(str(50.0))
        self.GridLengthCount = QtWidgets.QLineEdit()
        self.GridLengthCount.setText(str(3))
        self.GridWidthCount = QtWidgets.QLineEdit()
        self.GridWidthCount.setText(str(3))
        self.UserDefineStr = QtWidgets.QLineEdit()
        self.UserDefineStr.setText(str(""))


        # add Qwidgets to fbo 
        fbox.addRow(QtWidgets.QLabel("update point cloud"), self.btn_update_pc)  
        self.setLayout(fbox)

  
 

#---------------------------------- 
#   
#  main application GUI
#
#----------------------------------
class MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
         
        # # FusionTrackInstance  
        self.rossub = ROS_Class()
        self.caliMethod = CalibrationFT_KUKA()

        self.initUIwindow()
        self.Init_ROS_subscriber()


        # # list for 3 grid points
        # self.GridPoint3=[]

        # # read json file
        self.caliMethod.setKUKA_LM('cali_poses.json')
        #global variables
        # # empty list for calibration points
        self.caliList_FT = []
        self.Cross_point =[]
        self.new_pose_marker = PoseStamped()
        self.if_cali_check = False
        self.if_cali_start = False



    # GUI function to define all widgets    
    def initUIwindow(self):     
        # # vtk viewer for 3D model visualization
        self.vtkViewer = vtkViewerFrame(self.GeoModel)
        self.setCentralWidget(self.vtkViewer)   

        # add as a CONTROL panel dock widget
        self.controlWid = controlWidget()       
        self.controlPanel = QtWidgets.QDockWidget("Control", self)
        self.controlPanel.setObjectName("Control")
        self.controlPanel.setWidget( self.controlWid )
        self.controlPanel.setFloating(False)        
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.controlPanel) 


        # add matplotlib  
        self.controlWid.getbtn_update_pc().clicked.connect(self.callback_FT_LM_obtain) 
        # # trigger 
        self.rossub.getTriggerSignal().cali_trigger.connect(self.callback_triggerEvent_cali_status)
        self.rossub.getTriggerSignal().coll_trigger.connect(self.callback_triggerEvent_coll_status)
        
        # action: exit
        exitAction = QtWidgets.QAction( QtGui.QIcon('icon/exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.closeEventIcon )        
        
        # action: open file (vtk model, SSM )
        openFileAction = QtWidgets.QAction(QtGui.QIcon('icon/openfile.png'), '&OpenFile', self)
        openFileAction.setStatusTip('Open file ...')
        openFileAction.triggered.connect(self.showFileDialog)
                
        # menu bar
        menubar = self.menuBar()
        fileMemu = menubar.addMenu('&File')
        fileMemu.addAction(exitAction)
        fileMemu.addAction(openFileAction)
        
        # tool bar
        self.toolbar = self.addToolBar('&ToolBar1')     
        self.toolbar.addAction(exitAction)
        self.toolbar.addAction(openFileAction)
       
        # status bar
        self.statusBar()      
        self.resize(1200,1000)
        self.center()
        self.setWindowTitle('Automatic ultrasound calibration v1.0')             
        self.show()

 
    def closeEventIcon(self): 
        QtWidgets.qApp.quit()

  
    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
     
 
    def showFileDialog(self):
        chooseFilename = QtWidgets.QFileDialog.getOpenFileName(self, 'select statistical model file')
        loadfilename = str(chooseFilename[0]) 
        # 1. load from file
        self.GeoModel.loadFromFile(loadfilename) 
        self.statusBar().showMessage(loadfilename) 
        self.vtkViewer.updateModel(self.GeoModel)
        self.vtkViewer.updataVTKdraw() 
        # 2. update viewer
        self.vtkViewer.Render()
        # 3. reset camera   
        self.vtkViewer.ResetCamera() 



     
 
    def Init_ROS_subscriber(self):
        self.rossub.ROS_subscriber()
        self.rossub.ROS_publisher()
        print("ROS subscriber and publisher initialized !!! ")

    def callback_FT_LM_obtain(self):
        pc_numpy = self.rossub.get_pointcloud() 
        pointcloud_vtk = vnp.getVtkPolyDataFromNumpyPoints(pc_numpy)


    @pyqtSlot(object)
    def callback_triggerEvent_cali_status(self, received): 
            if  self.if_cali_check and self.if_cali_start  : # calibration is not done yet!
                curFTpoint = self.FusionTrackInstance.getMarker_6_pose()
                print(" current FT point (marker 6 ) is   ", curFTpoint)
                self.caliList_FT.append(curFTpoint)
                if len(self.caliList_FT) == self.caliMethod.getKUKA_LM_num():
                    print(" now it is ready for calibration calculation")
                    self.caliMethod.setFT_LM(self.caliList_FT)
                    self.caliMethod.initialAlignment_cali()
                    self.caliList_FT=[] 
 



# main function
def main ():    
    app = QApplication(sys.argv)
    window = MainWindow()   
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()
 