import vtk 


reader = vtk.vtkPLYReader()
reader.SetFileName("./bone1.ply")
reader.Update()  


stlWriter = vtk.vtkSTLWriter()
stlWriter.SetFileName("./bone1.STL")
stlWriter.SetInputConnection(reader.GetOutputPort())
stlWriter.Write()
