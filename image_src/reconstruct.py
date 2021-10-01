import numpy as np
import vtk


# Numpy array to vtk points
def numpyArr2vtkPoints(npArray):
    vtkpoints = vtk.vtkPoints()    
    for i in range(npArray.shape[0]):
        #print npArray[i]
        vtkpoints.InsertNextPoint(npArray[i])       
    return vtkpoints


pts = np.loadtxt("/media/ruixuan/Volume/ruixuan/Documents/us_image/26-06/point8.txt", delimiter=",", unpack=False) #comments="#",
ptsVTK = numpyArr2vtkPoints(pts)
profile = vtk.vtkPolyData()
profile.SetPoints(ptsVTK)


# Delaunay3D is used to triangulate the points. The Tolerance is the
# distance that nearly coincident points are merged
# together. (Delaunay does better if points are well spaced.) The
# alpha value is the radius of circumcircles, circumspheres. Any mesh
# entity whose circumcircle is smaller than this value is output.
delny = vtk.vtkDelaunay3D()
delny.SetInputData(profile)
delny.SetTolerance(5)
delny.SetAlpha(2)
delny.BoundingTriangulationOff()

# # Shrink the result to help see it better.
# shrink = vtk.vtkShrinkFilter()
# shrink.SetInputConnection(delny.GetOutputPort())
# shrink.SetShrinkFactor(0.9)

map = vtk.vtkDataSetMapper()
map.SetInputConnection(delny.GetOutputPort())

triangulation = vtk.vtkActor()
triangulation.SetMapper(map)
triangulation.GetProperty().SetColor(0.3, 0.5, 1)
# triangulation.GetProperty().SetOpacity(0.5)

# Create graphics stuff
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set the background and size
ren.AddActor(triangulation)
ren.SetBackground(1,1,1)
renWin.SetSize(800,600)
renWin.Render()


iren.Initialize()
renWin.Render()
iren.Start()





