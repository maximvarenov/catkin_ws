 
import vtk
import glob
import os 

 
files=glob.glob(r"/media/ruixuan/Volume/ruixuan/Documents/database/us_image/6-1-2021/v_seg/"+r"/*.png")
filePath = vtk.vtkStringArray()
files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
filePath.SetNumberOfValues(len(files))

for i in range(0,len(files),1):
   filePath.SetValue(i,files[i])

reader=vtk.vtkPNGReader()
reader.SetFileNames(filePath)
reader.SetDataSpacing(0.5,0.5,0.5)
reader.Update()
print(reader)

colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(1, 1, 0.0, 0.0) 

# To set different colored pores
colorFunc.AddRGBPoint(2, 0.0, 1, 0.0) # Green
opacity = vtk.vtkPiecewiseFunction()

volumeProperty = vtk.vtkVolumeProperty()
# set the color for volumes
volumeProperty.SetColor(colorFunc)
# To add black as background of Volume
volumeProperty.SetScalarOpacity(opacity)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetIndependentComponents(2)

#Ray cast function know how to render the data
#volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
#volumeMapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
volumeMapper.SetInputConnection(reader.GetOutputPort())
volumeMapper.SetBlendModeToMaximumIntensity()
#volumeMapper.SetBlendModeToAverageIntensity()
#volumeMapper.SetBlendModeToMinimumIntensity()
#volumeMapper.SetBlendModeToComposite()
#volumeMapper.SetBlendModeToAdditive()



volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren = vtk.vtkRenderer()
ren.AddVolume(volume)
#No need to set by default it is black
ren.SetBackground(0, 0, 0)


renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(600, 600)
    
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renWin)

interactor.Initialize()
renWin.Render()
interactor.Start()