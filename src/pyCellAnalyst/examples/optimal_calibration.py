from pyCellAnalyst import Volume
from scipy.optimize import minimize_scalar
import numpy as np
'''
Optimizes the z data spacing to best predict calibration beads of known diameter (6micron)
'''

def obj(x,directory):
    vol = Volume(directory,tratio=0.4,pixel_dim=[0.411,0.411,x],stain='cell',display=False)

    d = vol.dimensions[0]
    r = np.var(d)
    return r

directories = ['dat/cart_calibration/sphere1',
'dat/cart_calibration/sphere2',
'dat/cart_calibration/sphere3']

blresults = []
for directory in directories:
    print 'Optimization for images in directory %s started...' % directory 
    res = minimize_scalar(obj,method='bounded',bounds=(0.1,1),args=(directory,),tol=1e-5)
    print 'The optimal z-spacing for this sphere is %1.10f' % res.x
    blresults.append(res.x)
print 'The mean optimum z-spacing determined for %d spheres is:' % len(directories) 
print np.mean(blresults)
