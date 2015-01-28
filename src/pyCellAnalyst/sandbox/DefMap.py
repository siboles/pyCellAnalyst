import SimpleITK as sitk
import numpy as np
import os, re, fnmatch, string
import vtk
from pyevtk import hl

directory = 'test_stack/'
writer = sitk.ImageFileWriter()
files = fnmatch.filter(sorted(os.listdir(directory)),'*.tif')
counter = [re.search("[0-9]*\.tif",f).group() for f in files]
for i,c in enumerate(counter):
    counter[i] = int(c.replace('.tif',''))
files = np.array(files,dtype=object)
sorter = np.argsort(counter)
files = files[sorter]
imgs = []
for f in files:
    imgs.append(sitk.ReadImage(directory+f,sitk.sitkUInt8))

img1 = sitk.JoinSeries(imgs)
img1.SetSpacing([0.41,0.41,0.34])
img1 = sitk.Cast(img1,sitk.sitkFloat32)
img1 = sitk.Bilateral(img1,domainSigma=1.5)
img1 = sitk.RescaleIntensity(img1,0,255)

transform = sitk.Transform(3,sitk.sitkAffine)
transform.SetParameters([1.1,0,0,0,1.1,0,0,0,0.8,0,0,0.0])

img2 = sitk.Resample(img1,transform)

register = sitk.DiffeomorphicDemonsRegistrationFilter()
register.SetNumberOfIterations(1)
register.SmoothDisplacementFieldOn()
register.SmoothUpdateFieldOff()
register.UseImageSpacingOn()
register.SetUseGradientType(3)
def_map = register.Execute(img2,img1)

not_cells = sitk.GetArrayFromImage(img1)
not_cells = not_cells.swapaxes(0,2)
t = 0.4*float(np.max(not_cells))
not_cells = not_cells < t
print t

def_array = sitk.GetArrayFromImage(def_map)
def_array = def_array.swapaxes(0,2)

origin = img1.GetOrigin()
size = img1.GetSize()
spacing = img1.GetSpacing()
x = np.arange(origin[0],float(size[0])*spacing[0],spacing[0])
y = np.arange(origin[1],float(size[1])*spacing[1],spacing[1])
z = np.arange(origin[2],float(size[2])*spacing[2],spacing[2])

dx = def_array[:,:,:,0]
dx[not_cells] = np.NaN

hl.gridToVTK("./structured",x,y,z,pointData= {"x displacement": dx})
 
## E = []
## for i in xrange(3):
##     row = np.gradient(def_array[:,:,:,i],0.34,0.41,0.41)
##     E.append(row[2])
##     E.append(row[1])
##     E.append(row[0])
## E33 = sitk.GetImageFromArray(E[8])
## E33.SetSpacing([0.41,0.41,0.34])

## dx = sitk.GetImageFromArray(def_array[:,:,:,0])
## dy = sitk.GetImageFromArray(def_array[:,:,:,1])
## dz = sitk.GetImageFromArray(def_array[:,:,:,2])

## sitk.Cast(img1,sitk.sitkUInt8)
## writer.SetFileName('img1.nii')
## writer.Execute(img1)

## sitk.Cast(img2,sitk.sitkUInt8)
## writer.SetFileName('img2.nii')
## writer.Execute(img2)

## writer.SetFileName('E33.nii')
## writer.Execute(E33)



