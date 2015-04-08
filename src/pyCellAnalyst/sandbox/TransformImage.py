import SimpleITK as sitk
import numpy as np
import sys,string

up = 3
img_name = sys.argv[-1]
img1 = sitk.ReadImage(img_name)

size = img1.GetSize()
spacing = img1.GetSpacing()

refine = sitk.ResampleImageFilter()
refine.SetInterpolator(sitk.sitkNearestNeighbor)
refine.SetSize((size[0]*up,size[1]*up,size[2]*up))
refine.SetOutputSpacing((spacing[0]/float(up),spacing[1]/float(up),spacing[2]/float(up)))
img1 = refine.Execute(img1)

transform = sitk.Transform(3,sitk.sitkAffine)
transform.SetParameters([1.1,0,0,0,1.1,0,0,0,0.8,0,0,0])

#img2 = sitk.Resample(img1,transform)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(img1)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetTransform(transform)
img2 = resampler.Execute(img1)

sitk.WriteImage(img2,'/home/scott/test_deform/labels.nii')
sitk.WriteImage(img1,'/home/scott/test_ref/labels.nii')
