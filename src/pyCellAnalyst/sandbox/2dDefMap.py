import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os, re, fnmatch, string

def myshow(img, title=None, margin=0.05, dpi=80 ):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]
    
    elif nda.ndim == 4:
        c = nda.shape[-1]
        
        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")
            
        # take a z-slice
        nda = nda[nda.shape[0]//2,:,:,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]
   
    
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    
    t = ax.imshow(nda,extent=extent,interpolation=None)
    fig.colorbar(t)
    
    '''
    if nda.ndim == 2:
        t.set_cmap("gray")
    '''
    
    if(title):
        plt.title(title)



#im1 = ndimage.imread("material.png",'gray')
#im2 = ndimage.imread("spatial.png",'gray')
im1 = ndimage.imread("img1_cropped.tif")
im2 = ndimage.imread("img2_cropped.tif")
img1 = sitk.GetImageFromArray(im1)
img2 = sitk.GetImageFromArray(im2)

#register = sitk.SymmetricForcesDemonsRegistrationFilter()
register = sitk.DiffeomorphicDemonsRegistrationFilter()
register.SetUseGradientType(2)
register.SetMaximumKernelWidth(100)
register.SetNumberOfIterations(5000)
register.SetSmoothDisplacementField(True)
def_map = register.Execute(img2,img1)
print register

def_array = sitk.GetArrayFromImage(def_map)

E = []
for i in xrange(2):
    row = np.gradient(def_array[:,:,i])
    E.append(row[1])
    E.append(row[0])
plt.colorbar(plt.imshow(def_array[:,:,0]))
#plt.quiver(def_array[:,:,0],def_array[:,:,1])
plt.show()
'''
E11 = sitk.GetImageFromArray(E[0])
myshow(E11)
plt.show()
'''



