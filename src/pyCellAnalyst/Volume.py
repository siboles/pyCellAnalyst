from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import os
import re
import string
import warnings
import vtk
import fnmatch
import numpy as np
from vtk.util import vtkImageImportFromArray as vti
from sklearn import (svm, preprocessing)
import SimpleITK as sitk


class Volume(object):

    r"""
    This class will segment objects from 3-D images using user-specified
    routines. The intended purpose is for laser scanning fluorescence
    microscopy of chondrocytes and/or their surrounding matrices.
    Nevertheless, this can be generalized to any 3-D object using any
    imaging modality; however, it is likely the segmentation parameters
    will need to be adjusted. Therefore, in this case, the user should set
    segmentation='User' during Class instantiation, and call the segmentaion
    method with appropriate parameters.

    Parameters
    ----------
    vol_dir : str
        This is required. Currently it is the path to a directory containing
        a stack of TIFF images or a single NifTi (.nii) file. Other formats
        may be supported in the future.
    output_dir : str, optional
        The directory in which to save results. If not specified, a directory
        **vol_dir** + '_results' will be created and used.
    regions : [,[int, int, int, int, int, int], ...], optional
        Cropped regions bounding a single object to segment. In terms of voxel indices
        the order for each region is:
        [top left corner x, y, z, box edge length Lx, Ly, Lz].
        If not specified, the entire image is considered as the region.
    pixel_dim : [float=0.411, float=0.411, float=0.6835], optional
        The physical dimensions of the voxel ordered x, y, and z. If there is a need to correct
        a dimesion such as for the depth distortion in laser scanning microscopy, it should be
        incorporated here. Defaults to [0.411, 0.411, 0.6835]
    stain : str='Foreground', optional
        * 'Foreground' indicates the objects of interest appear bright in the image.
        * 'Background' indicates the objects of interest appear dark.
    segmentation : str='Geodesic', optional
        * 'Threshold' -- indicates to threshold the image at :math:`0.4\times intensity_{max}`.
        * 'Geodesic' -- (default) peform a geodesic active contour segmentation with default settings.
        * 'Edge-Free' -- perform an edge-free active contour segmentation with default dettins.
        * 'User' -- The user will invoke calls to segmentation function with custom settings.
    smoothing_method : str='Curvature Diffusion', optional
        Smoothing method to use on regions of interest.

        * 'None' -- No smoothing will be performed.
        * 'Gaussian' -- Perform Gaussian smoothing.
        * 'Median' -- Apply a median filter.
        * 'Curvature Diffusion' -- Perform curvature-based anisotropic diffusion smoothing.
        * 'Gradient Diffusion' -- Peform classical anisotropic diffusion smoothing.
        * 'Bilateral' -- Apply a bilateral filter.
        * 'Patch-based' -- Perform patch-based denoising.
    smoothing_parameters : dict, optional
        Depends on **smoothing_method**. Field keys are documented in methods **smoothRegion()**.
    
        * 'Gaussian' -- fields are:
            * 'sigma': float=0.5
        * 'Median' -- fields are:
            * 'radius': (int=1, int=1, int=1)
        * 'Curvature Diffusion' -- fields are:
            * 'iterations': int=10
            * 'conductance': float=9.0
        * 'Gradient Diffusion' -- fields are:
            * 'iterations': int=10
            * 'conductance': float=9.0
            * 'time step': float=0.01
        * 'Bilateral' -- fields are:
            * 'domainSigma': float=1.5
            * 'rangeSigma': float=40.0
            * 'samples': int=100
        * 'Patch-based' -- fields are:
            * 'radius': int=3
            * 'iterations': int=10
            * 'patches': int=20
            * 'noise model: str='poisson'
                * options: ('none', 'gaussian', 'poisson', 'rician')
    two_dim : bool=False, optional
        If *True*, will consider each 2-D slice in stack independently in smoothing and segmentation.
        This is not recommended except in special cases.
    bright : bool=False, optional
        If *True*, will perform bright spot removal replacing voxels with intensities
        :math:`\ge 98^{th}` percentile with median filtered (radius=6) value.
    enhance_edge : bool=False, optional
        If *True*, will enhance edges after smoothing using Laplacian sharpening.
    depth_adjust : bool=False, optional
        If *True*, will perform a linear correction for intensity degradation with depth.
    opening : bool=True, optional
        If *True*, will perform a morphological binary opening following thresholding to
        remove spurious connections and islands. If object of interest is thin, this may
        cause problems in which case, setting this to *False* may help.
    fillholes : bool=False, optional
        If *True*, all holes completely internal to the segmented object will be
        considered as part of the object.

    display : bool=True, optional
        If *True*, will spawn a 3-D interactive window rendering of segmented object surfaces.
    handle_overlap : bool=True, optional
        If *True*, overlapping segmented objects will be reclassified using a Support Vector Machine.
    debug : bool=False, optional
        If *True*, will write additional images to disk in NifTi format for debugging purposes.

    Attributes
    ----------
    cells : SimpleITK Image
        An image containing the segmented objects as integer labels. Has the same properties as the
        input image stack.
    thresholds  : [, int, ...]
        The threshold level for each cell
    volumes : [, float, ...]
        List of the physical volumes of the segmented objects.
    centroids : [,[float, float, float], ...]1
        List of centroids of segmented objects in physical space.
    surfaces : [,vtkPolyData, ...]
        List containing VTK STL objects.
    dimensions : [,[float, float, float],...]
        List containing the ellipsoid axis lengths of segmented objects.
        These are determined from the segmented binary images. It is recommended
        to use the values calculated from a 3-D mesh in the **CellMech** class.
    """

    def __init__(self,
                 vol_dir,
                 output_dir=None,
                 regions=None,
                 pixel_dim=[0.411, 0.411, 0.6835],
                 stain='Foreground',
                 segmentation='Geodesic',
                 smoothing_method='Curvature Diffusion',
                 smoothing_parameters={},
                 two_dim=False,
                 bright=False,
                 enhance_edge=False,
                 depth_adjust=False,
                 display=True,
                 handle_overlap=True,
                 debug=False,
                 opening=True,
                 fillholes=True):

        warnings.filterwarnings("ignore")

        self._vol_dir = vol_dir

        if output_dir is None:
            self._output_dir = vol_dir + '_results'
        else:
            self._output_dir = output_dir

        self._pixel_dim = pixel_dim
        self._stain = stain
        self.display = display
        self._img = None

        self.handle_overlap = handle_overlap
        self.smoothing_method = smoothing_method
        self.smoothing_parameters = smoothing_parameters
        self.two_dim = two_dim
        self.bright = bright
        self.enhance_edge = enhance_edge
        self.depth_adjust = depth_adjust
        self.debug = debug
        try:
            if self.debug:
                for p in ['seed*.nii', 'smoothed*.nii', 'edge*.nii']:
                    files = fnmatch.filter(os.listdir(self._output_dir), p)
                    for f in files:
                        filename = str(os.path.normpath(
                            self._output_dir + os.sep + f))
                        os.remove(filename)
        except:
            pass
        self.fillholes = fillholes
        self.opening = opening
        # read in the TIFF stack
        self._parseStack()
        if self.depth_adjust:
            self.adjustForDepth()

        # if regions are not specified, assume there is only one cell
        # and default to whole image
        if regions is None:
            size = np.array(self._img.GetSize(), int) - 1
            if self._img.GetDimension() == 3:
                self._regions = [[1, 1, 1] + list(size)]
                self._img = sitk.MirrorPad(self._img, padLowerBound=(1, 1, 1),
                                           padUpperBound=(1, 1, 1))
                self._img.SetOrigin((0, 0, 0))
            else:
                self._img = sitk.MirrorPad(self._img, padLowerBound=(1, 1),
                                           padUpperBound=(1, 1))
                self._img.SetOrigin((0, 0, 0))
                self._regions = [[1, 1] + list(size)]
        else:
            self._regions = regions
        # define a blank image with the same size and spacing as
        # image stack to add segmented cells to

        self.cells = sitk.Image(self._img.GetSize(), sitk.sitkUInt8)
        self.cells.CopyInformation(self._img)

        # list of smoothed ROIs
        self.smoothed = []
        # list of threshold values
        self.thresholds = []
        # list of levelsets
        self.levelsets = []

        self.surfaces = []

        self.volumes = []
        self.centroids = []
        self.dimensions = []

        #Execute segmentation with default parameters
        #unless specified as 'User'
        if segmentation == 'Threshold':
            self.thresholdSegmentation()
            self.active = False
        elif segmentation == 'Geodesic':
            self.geodesicSegmentation()
            self.active = "Geodesic"
        elif segmentation == 'EdgeFree':
            self.edgeFreeSegmentation()
            self.active = "EdgeFree"
        elif segmentation == 'User':
            self.active = False
            pass
        else:
            raise SystemExit('{:s} is not a supported segmentation method.'
                             .format(segmentation))

        try:
            os.mkdir(self._output_dir)
        except:
            pass

        sitk.WriteImage(self._img,
                        str(os.path.normpath(
                            self._output_dir + os.sep + 'stack.nii')))

    def _parseStack(self):
        for ftype in ['*.nii', '*.tif*']:
            files = fnmatch.filter(sorted(os.listdir(self._vol_dir)), ftype)
            if len(files) > 0:
                break

        if ftype == "*.tif*":
            if len(files) > 1:
                counter = [re.search("[0-9]*\.tif", f).group() for f in files]
                for i, c in enumerate(counter):
                    counter[i] = int(c.replace('.tif', ''))
                files = np.array(files, dtype=object)
                sorter = np.argsort(counter)
                files = files[sorter]
                img = []
                for fname in files:
                    filename = str(
                        os.path.normpath(self._vol_dir + os.sep + fname))
                    img.append(sitk.ReadImage(filename, sitk.sitkFloat32))
                self._img = sitk.RescaleIntensity(sitk.JoinSeries(img), 0.0, 1.0)
                print(("\nImported 3D image stack ranging from {:s} to {:s}"
                      .format(files[0], files[-1])))
            else:
                print(("\nImported 2D image {:s}".format(files[0])))
                filename = str(
                    os.path.normpath(self._vol_dir + os.sep + files[0]))
                self._img = sitk.RescaleIntensity(sitk.ReadImage(filename, sitk.sitkFloat32), 0.0, 1.0)
        elif ftype == "*.nii":
            filename = str(
                os.path.normpath(self._vol_dir + os.sep + files[0]))
            im = sitk.ReadImage(filename, sitk.sitkFloat32)
            self._img = sitk.RescaleIntensity(im, 0.0, 1.0)

        self._img.SetSpacing(self._pixel_dim)

    def smoothRegion(self, img):
        img = sitk.Cast(img, sitk.sitkFloat32)

        if self.smoothing_method == 'None':
            pass
        elif self.smoothing_method == 'Gaussian':
            parameters = {'sigma': 0.5}
            for p in list(self.smoothing_parameters.keys()):
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            img = sitk.DiscreteGaussian(img, variance=parameters['sigma'])

        elif self.smoothing_method == 'Median':
            parameters = {'radius': (1, 1, 1)}
            for p in list(self.smoothing_parameters.keys()):
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            img = sitk.Median(img, radius=parameters['radius'])

        elif self.smoothing_method == 'Curvature Diffusion':
            parameters = {'iterations': 10, 'conductance': 9}
            for p in list(self.smoothing_parameters.keys()):
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            smooth = sitk.CurvatureAnisotropicDiffusionImageFilter()
            smooth.EstimateOptimalTimeStep(img)
            smooth.SetNumberOfIterations(parameters['iterations'])
            smooth.SetConductanceParameter(parameters['conductance'])
            img = smooth.Execute(img)

        elif self.smoothing_method == 'Gradient Diffusion':
            parameters = {'iterations': 10,
                          'conductance': 9,
                          'time step': 0.01}
            for p in list(self.smoothing_parameters.keys()):
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            smooth = sitk.GradientAnisotropicDiffusionImageFilter()
            smooth.SetNumberOfIterations(parameters['iterations'])
            smooth.SetConductanceParameter(parameters['conductance'])
            smooth.SetTimeStep(parameters['time step'])
            img = smooth.Execute(img)

        elif self.smoothing_method == 'Bilateral':
            parameters = {'domainSigma': 1.5,
                          'rangeSigma': 10.0,
                          'samples': 100}
            for p in list(self.smoothing_parameters.keys()):
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            img = sitk.Cast(img, sitk.sitkUInt8)
            img = sitk.Bilateral(
                img,
                domainSigma=parameters['domainSigma'],
                rangeSigma=parameters['rangeSigma'],
                numberOfRangeGaussianSamples=parameters['samples'])

        elif self.smoothing_method == 'Patch-based':
            parameters = {'radius': 4,
                          'iterations': 10,
                          'patches': 20,
                          'noise model': 3}
            noise_models = {'nomodel': 0,
                            'gaussian': 1,
                            'rician': 2,
                            'poisson': 3}
            for p in list(self.smoothing_parameters.keys()):
                try:
                    if p == 'noise model':
                        parameters[p] = noise_models[
                            self.smoothing_parameters[p]]
                    else:
                        parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))

            smooth = sitk.PatchBasedDenoisingImageFilter()
            smooth.KernelBandwidthEstimationOn()
            smooth.SetNoiseModel(parameters['noise model'])
            smooth.SetNoiseModelFidelityWeight(1.0)
            smooth.SetNumberOfSamplePatches(parameters['patches'])
            smooth.SetPatchRadius(parameters['radius'])
            smooth.SetNumberOfIterations(parameters['iterations'])
            img = smooth.Execute(img)
            smooth = sitk.GradientAnisotropicDiffusionImageFilter()
            smooth.SetTimeStep(0.01)
            smooth.SetNumberOfIterations(5)
            smooth.SetConductanceParameter(9.0)
            img = smooth.Execute(img)

        else:
            raise SystemExit(("ERROR: {:s} is not a supported "
                              "smoothing method. "
                              "Options are:\n"
                              "'None'\n"
                              "'Gaussian'\n"
                              "'Median'\n"
                              "'Curvature Diffusion'\n"
                              "'Gradient Diffusion'\n"
                              "'Bilateral'\n"
                              "'Patch-based'".format(self.smoothing_method)))
        #enhance the edges
        if self.enhance_edge:
            img = sitk.LaplacianSharpening(img)
        if self._stain == "Background":
            img = sitk.InvertIntensity(img)
        # replace border pixel values with average of border slices
        img = self._flattenBorder(img)
        img = sitk.AdaptiveHistogramEqualization(img, radius=[int(s / 4) for s in img.GetSize()])

        return sitk.RescaleIntensity(img, 0.0, 1.0)

    def _getMinMax(self, img):
        mm = sitk.MinimumMaximumImageFilter()
        mm.Execute(img)
        return (mm.GetMinimum(), mm.GetMaximum())

    def _getLabelShape(self, img):
        ls = sitk.LabelShapeStatisticsImageFilter()
        ls.Execute(img)
        labels = ls.GetLabels()
        labelshape = {'volume': [],
                      'centroid': [],
                      'ellipsoid diameters': [],
                      'ellipsoid axes': [],
                      'bounding box': [],
                      'border size': []}
        for l in labels:
            labelshape['volume'].append(ls.GetPhysicalSize(l))
            labelshape['centroid'].append(ls.GetCentroid(l))
            labelshape['ellipsoid diameters'].append(
                ls.GetEquivalentEllipsoidDiameter(l))
            labelshape['ellipsoid axes'].append(ls.GetPrincipalAxes(l))
            labelshape['bounding box'].append(ls.GetBoundingBox(l))
            labelshape['border size'].append(ls.GetPerimeterOnBorder(l))
        return labelshape

    def thresholdSegmentation(self, method='Percentage',
                              adaptive=True, ratio=0.4):
        r"""
        Segments object of interest from image using user-specified method.

        Parameters
        ----------
        method : str='Percentage'
            The thresholding method to use. Options are:

            * 'Percentage' --  Threshold at percentage of the maximum voxel intensity.
            * 'Otsu' -- Threshold using Otsu's method
            * 'Huang' -- 
            * 'IsoData'
            * 'Li'
            * 'MaxEntropy' -- Sets the threshold value such that the sum of information entropy (Shannon) in the foreground and background is maximized.
            * 'KittlerIllingworth'
            * 'Moments'
            * 'Yen'
            * 'RenyiEntropy' -- The same as 'MaxEntropy', but uses the Renyi entropy function.
            * 'Shanbhag' -- Extends upon the entropy methods with fuzzy set theory.
        ratio : float=0.4
            Ratio of maximum voxel intensity in the region of interest to threshold at.
            Only used if 'Percentage' method is given.
        adaptive : bool=True
            If *True* will adaptively adjust the determined threshold value until
            the segmented object does not the region boundaries.
        """
        if method not in ['Percentage',
                          'Otsu',
                          'Huang',
                          'IsoData',
                          'Li',
                          'MaxEntropy',
                          'KittlerIllingworth',
                          'Moments',
                          'Yen',
                          'RenyiEntropy',
                          'Shanbhag']:
            raise SystemExit("{:s} is not a supported threshold method."
                             .format(method))
        dimension = self._img.GetDimension()
        for i, region in enumerate(self._regions):
            if dimension == 3:
                roi = sitk.RegionOfInterest(self._img, region[3:], region[0:3])
            else:
                self._pixel_dim = self._pixel_dim[0:2]
                roi = sitk.RegionOfInterest(self._img, region[3:5],
                                            region[0:2])
            #Remove bright spots if bright=True
            if self.bright:
                a = sitk.GetArrayFromImage(roi)
                b = sitk.GetArrayFromImage(sitk.Median(roi, (6, 6, 6)))
                top = np.percentile(a.ravel(), 98)
                #replace only voxels in 98th or higher percentile
                #with median smoothed value
                a[a > top] = b[a > top]
                a = sitk.GetImageFromArray(a)
                a.SetSpacing(roi.GetSpacing())
                a.SetOrigin(roi.GetOrigin())
                a.SetDirection(roi.GetDirection())
                roi = a

            if self.two_dim:
                simg = self.smooth2D(roi)
            else:
                simg = self.smoothRegion(roi)

            print("\n------------------")
            print(("Segmenting Cell {:d}".format(i + 1)))
            print("------------------\n")

            if method == 'Percentage':
                t = self._getMinMax(simg)[1]

                if self.two_dim:
                    seg, thigh, tlow, tlist = self.threshold2D(simg, "Percentage", ratio)
                else:
                    t *= ratio
                    seg = sitk.BinaryThreshold(simg, t, 1e7)

                if self.two_dim:
                    print(("... Threshold using {:s} method ranged: "
                           "{:6.5f}-{:6.5f}".format(method, tlow, thigh)))
                else:
                    print(("... Thresholded using {:s} method at a "
                           "value of: {:6.5f}".format(method, t)))

            elif method == 'Otsu':
                thres = sitk.OtsuThresholdImageFilter()

            elif method == 'Huang':
                thres = sitk.HuangThresholdImageFilter()

            elif method == 'IsoData':
                thres = sitk.IsoDataThresholdImageFilter()

            elif method == 'Li':
                thres = sitk.LiThresholdImageFilter()

            elif method == 'MaxEntropy':
                thres = sitk.MaximumEntropyThresholdImageFilter()

            elif method == 'KittlerIllingworth':
                thres = sitk.KittlerIllingworthThresholdImageFilter()

            elif method == 'Moments':
                thres = sitk.MomentsThresholdImageFilter()

            elif method == 'Yen':
                thres = sitk.YenThresholdImageFilter()

            elif method == 'RenyiEntropy':
                thres = sitk.RenyiEntropyThresholdImageFilter()

            elif method == 'Shanbhag':
                thres = sitk.ShanbhagThresholdImageFilter()

            else:
                raise SystemExit(("Unrecognized value for 'stain' {:s}. "
                                  "Options are 'Foreground' or 'Background'"
                                  .format(self._stain)))

            if not(method == 'Percentage'):
                thres.SetNumberOfHistogramBins(128)
                thres.SetInsideValue(0)
                thres.SetOutsideValue(1)

                if self.two_dim:
                    seg, thigh, tlow, tlist = self.threshold2D(simg, thres, ratio)
                    print(("... Thresholds determined by {:s} method ranged: "
                           "[{:6.5f}-{:6.5f}".format(method, tlow, thigh)))
                else:
                    seg = thres.Execute(simg)
                    t = thres.GetThreshold()
                    print(("... Threshold determined by {:s} method: {:6.5f}"
                          .format(method, t)))

            if adaptive and not(self.two_dim):
                newt = float(np.copy(t))
                if dimension == 3:
                    region_bnds = [(0, region[3]), (0, region[4])]
                else:
                    region_bnds = [(0, region[2]), (0, region[3])]

                cnt = 0
                while True:
                    if self.opening:
                        #Opening (Erosion/Dilation) step to remove islands
                        #smaller than 1 voxels in radius
                        seg = sitk.BinaryMorphologicalOpening(seg, 1)
                    if self.fillholes:
                        seg = sitk.VotingBinaryIterativeHoleFilling(seg)
                    #Get connected regions
                    r = sitk.ConnectedComponent(seg)
                    labelstats = self._getLabelShape(r)
                    d = 1e7
                    region_cent = old_div(np.array(list(seg.GetSize()), float), 2.0)
                    region_cent *= np.array(self._pixel_dim)
                    region_cent += np.array(list(seg.GetOrigin()), float)
                    if cnt == 0:
                        for l, c in enumerate(labelstats['centroid']):
                            dist = np.linalg.norm(np.array(c, float) - region_cent)
                            if dist < d:
                                d = dist
                                label = l + 1
                        mask = r==label
                    else:
                        tmp = sitk.Mask(r, mask)
                        sitk.WriteImage(mask, "mask{:d}.nii".format(i+1))
                        stats = self._getLabelShape(tmp)
                        try:
                            label = np.argmax(stats["volume"]) + 1
                        except:
                            newt -= 0.001
                            seg = sitk.BinaryThreshold(simg, newt, 1e7)
                            break
                    cnt += 1
                    # if exception here, then threshold adjusted too much
                    # and previous increment will be taken
                    try:
                        bb = labelstats['bounding box'][label - 1]
                    except:
                        break

                    if labelstats['border size'][label - 1] > 0:
                        newt += 0.001
                        seg = sitk.BinaryThreshold(simg, newt, 1e7)
                    else:
                        break
                if not(newt == t):
                    print(("... ... Adjusted the threshold to: "
                           "{:6.5f}".format(newt)))
                self.thresholds.append(newt)
            else:
                if self.opening:
                    #Opening (Erosion/Dilation) step to remove islands
                    #smaller than 1 voxels in radius
                    seg = sitk.VotingBinaryIterativeHoleFilling(seg)
                if self.fillholes:
                    seg = sitk.BinaryFillhole(seg != 0)
                #Get connected regions
                r = sitk.ConnectedComponent(seg)
                labelstats = self._getLabelShape(r)
                d = 1e7
                region_cent = old_div(np.array(list(seg.GetSize()), float), 2.0)
                region_cent *= np.array(self._pixel_dim)
                region_cent += np.array(list(seg.GetOrigin()), float)
                for l, c in enumerate(labelstats['centroid']):
                    dist = np.linalg.norm(np.array(c, float) - region_cent)
                    if dist < d:
                        d = dist
                        label = l + 1
                if self.two_dim:
                    self.thresholds.append(np.max(tlist))
                else:
                    self.thresholds.append(t)

            tmp = sitk.Image(self._img.GetSize(), sitk.sitkUInt8)
            tmp.CopyInformation(self._img)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tmp)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            tmp = resampler.Execute((r == label) * (i + 1))
            self.cells = sitk.Add(self.cells, tmp)
            # scale smoothed image if independent slices option flagged
            if self.two_dim:
                simg = self.scale2D(simg, tlist)
            self.smoothed.append(simg)
            if self.debug:
                sitk.WriteImage(sitk.RescaleIntensity(simg, 0, 255),
                                str(os.path.normpath(
                                    self._output_dir + os.sep +
                                    "smoothed_{:03d}.nii".format(i + 1))))
            #Test for overlap
            if self.handle_overlap:
                maxlabel = self._getMinMax(self.cells)[1]
                if maxlabel > (i + 1):
                    self.cells = self._classifyShared(i, self.cells, False)

    def geodesicSegmentation(self,
                             upsampling=2,
                             seed_method='Percentage',
                             adaptive=True,
                             ratio=0.7,
                             canny_variance=(0.05, 0.05, 0.05),
                             cannyUpper=0.0,
                             cannyLower=0.0,
                             propagation=0.15,
                             curvature=0.2,
                             advection=1.0,
                             rms=0.01,
                             active_iterations=200):
        """
        Performs a segmentation using the SimpleITK implementation of the
        Geodesic Active Contour Levelset Segmentation method described in
        (Caselles et al. 1997.) Please also consult SimpleITK's documentation
        of GeodesicActiveContourLevelSetImageFilter. This method will establish
        the initial levelset function by calling the **thresholdSegmentation()**
        method, and calculating a distance map from the resulting binary image.

        Parameters
        ----------
        propagation : float=0.15
            Weight for propagation term in active contour functional.
            Higher values result in faster expansion.
        curvature : float=0.2
            Weight for curvature term in active contour functional.
            Higher values result in smoother segmentation.
        advection : float=1.0
            Weight for advective term in active contour functional.
            Higher values causes the levelset evolution to be drawn
            and stick to edges.
        rms : float=0.01
            The change in root-mean-square difference  at which iterations
            will terminate. This value is divided by the **upsampling** value
            to account the effect of voxel size.
        active_iterations : int=200
            The maximum number of iterations the active contour will conduct.
        upsampling : int=2, optional
            Resample image splitting original voxels this many times.
            Resampling will always be performed to make voxels isotropic,
            because anisotropic voxels can degrade the performance of this algorithm.
        seed_method : str='Percentage'
            Thresholding method used to determine seed image; same as
            **thresholdSegmentation()** **method** parameter. Please consult its
            documentation.
        adaptive : bool=True
            If true will adaptively adjust threshold the threshold value until
            resulting segmentation no longer touches the region of interest bounds.
        ratio : float=0.7
            The ratio to use with 'Percentage' threshold method. This plays no role
            with other seed methods.
        canny_variance : [float=0.05, float=0.05, float=0.05]
            The Gaussian variance for canny edge detection used to generate the edge map for
            this method. Gaussian smoothing is performed during edge detection, but if
            another smoothing method was already performed this can be set low. High values
            results in smoother edges, but risk losing edges when other objects are close.
        cannyUpper : float=0.0
            Ensures voxels in the image gradient with a value higher than this will always be
            considered edges, and never discarded.
        cannyLower : float=0.0
            Ensures voxels in the image gradient with a value lower than this will be discarded.

        """
        self.active = "Geodesic"
        self.thresholdSegmentation(method=seed_method, ratio=ratio,
                                   adaptive=adaptive)
        dimension = self._img.GetDimension()
        newcells = sitk.Image(self.cells.GetSize(), sitk.sitkUInt8)
        newcells.CopyInformation(self.cells)
        for i, region in enumerate(self._regions):
            print("\n-------------------------------------------")
            print(("Evolving Geodesic Active Contour for Cell {:d}"
                  .format(i + 1)))
            print("-------------------------------------------")
            if dimension == 3:
                seed = sitk.RegionOfInterest(self.cells,
                                             region[3:],
                                             region[0:3])
                roi = self.smoothed[i]
                #resample the Region of Interest to improve resolution of
                #derivatives and give closer to isotropic voxels
                zratio = old_div(self._pixel_dim[2], self._pixel_dim[0])
                #adjust size in z to be close to isotropic and double
                #the resolution
                newz = int(zratio * roi.GetSize()[2]) * upsampling
                newzspace = (old_div(float(roi.GetSize()[2]), float(newz))) * self._pixel_dim[2]
                newx = roi.GetSize()[0] * upsampling
                newxspace = old_div(self._pixel_dim[0], float(upsampling))
                newy = roi.GetSize()[1] * upsampling
                newyspace = old_div(self._pixel_dim[1], float(upsampling))
                #Do the resampling
                refine = sitk.ResampleImageFilter()
                refine.SetInterpolator(sitk.sitkBSpline)
                refine.SetSize((newx, newy, newz))
                refine.SetOutputOrigin(roi.GetOrigin())
                refine.SetOutputSpacing((newxspace, newyspace, newzspace))
                refine.SetOutputDirection(roi.GetDirection())
                simg = refine.Execute(roi)
            else:
                seed = sitk.RegionOfInterest(self.cells,
                                             region[3:5],
                                             region[0:2])
                roi = self.smoothed[i]
                #resample the Region of Interest to improve resolution
                #of derivatives
                newx = roi.GetSize()[0] * upsampling
                newxspace = old_div(self._pixel_dim[0], float(upsampling))
                newy = roi.GetSize()[1] * upsampling
                newyspace = old_div(self._pixel_dim[1], float(upsampling))
                #Do the resampling
                refine = sitk.ResampleImageFilter()
                refine.SetInterpolator(sitk.sitkBSpline)
                refine.SetSize((newx, newy))
                refine.SetOutputOrigin(roi.GetOrigin())
                refine.SetOutputSpacing((newxspace, newyspace))
                refine.SetOutputDirection(roi.GetDirection())
                simg = refine.Execute(roi)
            refine.SetInterpolator(sitk.sitkNearestNeighbor)
            seed = refine.Execute(seed)
            if self._getMinMax(seed)[1] < 1:
                seed = self._replaceSeed(seed)
            else:
                #smooth the perimeter of the binary seed
                dm = sitk.SignedMaurerDistanceMap(seed, False, False, False)
                mindist = self._getMinMax(dm)[0]
                #shrink seed by 20%
                seed = dm <= 0.2 * mindist
                labels = sitk.ConnectedComponent(seed)
                labelstats = self._getLabelShape(labels)
                ind = np.argmin(labelstats['border size'])
                seed = labels == ind + 1

            if self.two_dim and dimension == 3:
                seg = self.geodesic2D(seed, simg,
                                      cannyLower, cannyUpper, canny_variance,
                                      upsampling, active_iterations, rms,
                                      propagation, curvature, advection)
            else:
                canny = sitk.CannyEdgeDetection(
                    sitk.Cast(simg, sitk.sitkFloat32),
                    lowerThreshold=cannyLower,
                    upperThreshold=cannyUpper,
                    variance=canny_variance)

                canny = sitk.InvertIntensity(canny, 1)
                canny = sitk.Cast(canny, sitk.sitkFloat32)
                a = sitk.GetArrayFromImage(canny)
                if len(a.shape) == 3:
                    ind = [np.s_[0:2, :, :], np.s_[-2:, :, :],
                           np.s_[:, 0:2, :], np.s_[:, -2:, :],
                           np.s_[:, :, 0:2], np.s_[:, :, -2:]]
                else:
                    ind = [np.s_[0:2, :], np.s_[-2:, :],
                           np.s_[:, 0:2], np.s_[:, -2:]]

                m = np.max(a.ravel())
                for ii in ind:
                    a[ii] = m
                tmp = sitk.GetImageFromArray(a)
                tmp.CopyInformation(canny)
                canny = tmp

                if self.debug:
                    sitk.WriteImage(sitk.RescaleIntensity(simg, 0, 255),
                                    str(os.path.normpath(
                                        self._output_dir +
                                        os.sep + "smoothed_{:03d}.nii"
                                        .format(i + 1))))
                    sitk.WriteImage(sitk.RescaleIntensity(seed, 0, 255),
                                    str(os.path.normpath(
                                        self._output_dir +
                                        os.sep + "seed_{:03d}.nii"
                                        .format(i + 1))))
                    sitk.WriteImage(sitk.RescaleIntensity(canny, 0, 255),
                                    str(os.path.normpath(
                                        self._output_dir +
                                        os.sep + "edge_{:03d}.nii"
                                        .format(i + 1))))
                d = sitk.SignedMaurerDistanceMap(seed, insideIsPositive=False,
                                                 squaredDistance=False,
                                                 useImageSpacing=True)
                gd = sitk.GeodesicActiveContourLevelSetImageFilter()
                gd.SetMaximumRMSError(old_div(rms, float(upsampling)))
                gd.SetNumberOfIterations(active_iterations)
                gd.SetPropagationScaling(propagation)
                gd.SetCurvatureScaling(curvature)
                gd.SetAdvectionScaling(advection)
                seg = gd.Execute(d, canny)
                print("... Geodesic Active Contour Segmentation Completed")
                print(("... ... Elapsed Iterations: {:d}"
                      .format(gd.GetElapsedIterations())))
                print(("... ... Change in RMS Error: {:.3e}"
                      .format(gd.GetRMSChange())))

            self.levelsets.append(seg)
            seg = sitk.BinaryThreshold(seg, -1e7, 0) * (i + 1)
            tmp = sitk.Image(self._img.GetSize(), sitk.sitkUInt8)
            tmp.CopyInformation(self._img)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tmp)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            tmp = resampler.Execute(seg)
            if self.fillholes:
                tmp = sitk.BinaryFillhole(tmp)
            newcells = sitk.Add(newcells, tmp)
            #Handle Overlap
            if self.handle_overlap:
                maxlabel = self._getMinMax(newcells)[1]
                if maxlabel > (i + 1):
                    newcells = self._classifyShared(i, newcells, True)
        self.cells = newcells

    def edgeFreeSegmentation(self,
                             upsampling=2,
                             seed_method='Percentage',
                             adaptive=True,
                             ratio=0.4,
                             lambda1=1.0,
                             lambda2=1.1,
                             curvature=0.0,
                             iterations=20):
        """
        Performs a segmentation using the SimpleITK implementation of the
        Active Contours Without Edges method described in (Chan and Vese. 2001.)
        Please also consult SimpleITK's documentation of ScalarChanAndVeseDenseLevelSetImageFilter.

        Parameters
        ----------
        upsampling : int=2, optional
            Resample image splitting original voxels this many times.
            Resampling will always be performed to make voxels isotropic,
            because anisotropic voxels can degrade the performance of this algorithm.
        seed_method : str='Percentage'
            Thresholding method used to determine seed image; same as
            **thresholdSegmentation()** method parameter. Please consult its
            documentation.
        adaptive : bool=True
            If true will adaptively adjust threshold the threshold value until
            resulting segmentation no longer touches the region of interest bounds.
        ratio : float=0.7
            The ratio to use with 'Percentage' threshold method. This plays no role
            with other seed methods.
        lambda1 : float=1.0
            Weight for internal levelset term contribution to the total energy.
        lambda2 : float=1.1
            Weight for external levelset term contribution to the total energy. 
        curvature : float=0.0
            Weight for curvature. Higher results in smoother levelsets, but less
            ability to capture fine features.
        iterations : int=20
            The number of iterations the active contour method will conduct.
        """
        self.active = "EdgeFree"
        self.thresholdSegmentation(method=seed_method, ratio=ratio,
                                   adaptive=adaptive)
        dimension = self._img.GetDimension()
        newcells = sitk.Image(self.cells.GetSize(), sitk.sitkUInt8)
        newcells.CopyInformation(self.cells)
        for i, region in enumerate(self._regions):
            print("\n-------------------------------------------")
            print(("Evolving Edge-free Active Contour for Cell {:d}"
                  .format(i + 1)))
            print("-------------------------------------------")
            if dimension == 3:
                seed = sitk.RegionOfInterest(self.cells,
                                             region[3:],
                                             region[0:3])
                roi = self.smoothed[i]
                #resample the Region of Interest to improve resolution
                #of derivatives and give closer to isotropic voxels
                zratio = old_div(self._pixel_dim[2], self._pixel_dim[0])
                #adjust size in z to be close to isotropic and
                #double the resolution
                newz = int(zratio * roi.GetSize()[2]) * upsampling
                newzspace = (old_div(float(roi.GetSize()[2]), float(newz))) * self._pixel_dim[2]
                newx = roi.GetSize()[0] * upsampling
                newxspace = old_div(self._pixel_dim[0], float(upsampling))
                newy = roi.GetSize()[1] * upsampling
                newyspace = old_div(self._pixel_dim[1], float(upsampling))
                #Do the resampling
                refine = sitk.ResampleImageFilter()
                refine.SetInterpolator(sitk.sitkBSpline)
                refine.SetSize((newx, newy, newz))
                refine.SetOutputOrigin(roi.GetOrigin())
                refine.SetOutputSpacing((newxspace, newyspace, newzspace))
                refine.SetOutputDirection(roi.GetDirection())
                simg = refine.Execute(roi)
            else:
                seed = sitk.RegionOfInterest(self.cells,
                                             region[3:5],
                                             region[0:2])
                roi = self.smoothed[i]
                #resample the Region of Interest to improve resolution
                #of derivatives
                newx = roi.GetSize()[0] * upsampling
                newxspace = old_div(self._pixel_dim[0], float(upsampling))
                newy = roi.GetSize()[1] * upsampling
                newyspace = old_div(self._pixel_dim[1], float(upsampling))
                #Do the resampling
                refine = sitk.ResampleImageFilter()
                refine.SetInterpolator(sitk.sitkBSpline)
                refine.SetSize((newx, newy))
                refine.SetOutputOrigin(roi.GetOrigin())
                refine.SetOutputSpacing((newxspace, newyspace))
                refine.SetOutputDirection(roi.GetDirection())
                simg = refine.Execute(roi)
            refine.SetInterpolator(sitk.sitkNearestNeighbor)
            seed = refine.Execute(seed)
            if self._getMinMax(seed)[1] < 1:
                seed = self._replaceSeed(seed)
            else:
                #smooth the perimeter of the binary seed
                dm = sitk.SignedMaurerDistanceMap(seed, False, False, False)
                mindist = self._getMinMax(dm)[0]
                #shrink seed by 20%
                seed = dm <= 0.2 * mindist
                labels = sitk.ConnectedComponent(seed)
                labelstats = self._getLabelShape(labels)
                ind = np.argmin(labelstats['border size'])
                seed = labels == ind + 1

            if self.debug:
                sitk.WriteImage(sitk.RescaleIntensity(seed, 0, 255),
                                str(os.path.normpath(
                                    self._output_dir + os.sep +
                                    "seed_{:03d}.nii".format(i + 1))))

            cv = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
            cv.SetNumberOfIterations(iterations)
            cv.UseImageSpacingOn()
            cv.SetHeavisideStepFunction(0)
            cv.SetEpsilon(upsampling)
            cv.SetCurvatureWeight(curvature)
            cv.SetLambda1(lambda1)
            cv.SetLambda2(lambda2)
            if self.two_dim and dimension == 3:
                stack = []
                size = simg.GetSize()
                for sl in range(size[2]):
                    im = sitk.Extract(simg, [size[0], size[1], 0], [0, 0, sl])
                    s = sitk.Extract(seed, [size[0], size[1], 0], [0, 0, sl])
                    phi0 = sitk.SignedMaurerDistanceMap(s,
                                                        insideIsPositive=False,
                                                        squaredDistance=False,
                                                        useImageSpacing=True)
                    stack.append(cv.Execute(phi0, sitk.Cast(im,
                                                            sitk.sitkFloat32)))
                seg = sitk.JoinSeries(stack)
                seg.CopyInformation(simg)
            else:
                phi0 = sitk.SignedMaurerDistanceMap(seed,
                                                    insideIsPositive=False,
                                                    squaredDistance=False,
                                                    useImageSpacing=True)

                seg = cv.Execute(phi0,
                                 sitk.Cast(simg, sitk.sitkFloat32))
                print("... Edge-free Active Contour Segmentation Completed")
                print(("... ... Elapsed Iterations: {:d}"
                      .format(cv.GetElapsedIterations())))
                print(("... ... Change in RMS Error: {:.3e}"
                      .format(cv.GetRMSChange())))

            self.levelsets.append(seg)
            b = sitk.BinaryThreshold(seg, 1e-7, 1e7)
            #Get connected regions
            if self.opening:
                b = sitk.BinaryMorphologicalOpening(b, upsampling)
            r = sitk.ConnectedComponent(b)
            labelstats = self._getLabelShape(r)
            label = np.argmax(labelstats['volume']) + 1
            b = (r == label) * (i + 1)
            tmp = sitk.Image(self._img.GetSize(), sitk.sitkUInt8) 
            tmp.CopyInformation(self._img)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tmp)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            tmp = resampler.Execute(b)
            if self.fillholes:
                tmp = sitk.BinaryFillhole(tmp)
            newcells = sitk.Add(newcells, tmp)
            #Handle Overlap
            if self.handle_overlap:
                maxlabel = self._getMinMax(newcells)[1]
                if maxlabel > (i + 1):
                    newcells = self._classifyShared(i, newcells, True)
        self.cells = newcells

    def _classifyShared(self, i, cells, previous):
        """
        If segmented objects overlap and **handle_overlap** is *True*,
        this will attempt to reclassify the shared voxels using the
        thresholded seed to train a support vector machine. Of course,
        this relies on the seed to not overlap. The user strategy to get
        good results from this would be to use an active contour method,
        with an aggressive thresholding method to produce the seed.

        Returns
        -------
        A modified version of cells attribute with the overlapping objects
        reclassified.
        """
        #cells overlap so use SVM to classify shared voxels
        print("... ... ... WARNING: Segmentation overlapped a previous")
        print("... ... ... Using SVM to classify shared voxels")
        a = sitk.GetArrayFromImage(cells)
        ind2space = np.array(self._pixel_dim, float)[::-1]
        # we can use seeds from a previous segmentation as training
        # for geodesic and edge-free cases
        if previous:
            t = sitk.GetArrayFromImage(self.cells)
            p1 = np.argwhere(t == (i + 1)) * ind2space
            print("\n")
        else:
            print(("... ... ... The training data are often insufficient "
                   "for this segmentation method."))
            print(("... ... ... Please consider using Geodesic or "
                   "EdgeFree options.\n"))
            p1 = np.argwhere(a == (i + 1)) * ind2space
        if p1.size == 0:
            print(("... ... ... The seed from thresholding does not contain any voxels. "
                   "Aborting the SVM to fix overlap."))
            print("... ... ... Please consider using a different thresholding method.\n")

            return cells

        g1 = np.array([i + 1] * p1.shape[0], int)
        labels = np.unique(a)
        b = np.copy(a)
        for l in labels[labels > (i + 1)]:
            if previous:
                p2 = np.argwhere(t == (l - i - 1)) * ind2space
            else:
                p2 = np.argwhere(a == (l - i - 1)) * ind2space
            unknown1 = np.argwhere(a == l) * ind2space
            unknown2 = np.argwhere(a == (i + 1)) * ind2space
            unknown3 = np.argwhere(a == (l - i - 1)) * ind2space
            unknown = np.vstack((unknown1, unknown2, unknown3))
            g2 = np.array([l - i - 1] * p2.shape[0], int)
            X = np.vstack((p1, p2))
            scaler = preprocessing.StandardScaler().fit(X)
            y = np.hstack((g1, g2))
            clf = svm.SVC(kernel='rbf', degree=3, gamma=2)
            clf.fit(scaler.transform(X), y)
            classification = clf.predict(scaler.transform(unknown))
            b[a == l] = classification[0:unknown1.shape[0]]
            b[a == (i + 1)] = classification[unknown1.shape[0]:
                                             unknown1.shape[0] +
                                             unknown2.shape[0]]
            b[a == (l - i - 1)] = classification[unknown1.shape[0] +
                                                 unknown2.shape[0]:]
        cells = sitk.Cast(sitk.GetImageFromArray(b), sitk.sitkUInt8)
        cells.CopyInformation(self._img)
        return cells

    def writeSurfaces(self):
        """"""
        #delete old surfaces
        old_surfaces = fnmatch.filter(os.listdir(self._output_dir), '*.stl')
        for f in old_surfaces:
            os.remove(self._output_dir + os.sep + f)
        #create and write the STLs
        stl = vtk.vtkSTLWriter()
        polywriter = vtk.vtkPolyDataWriter()
        for i, c in enumerate(self._regions):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(self.cells)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            if self._img.GetDimension() == 3:
                roi = sitk.RegionOfInterest(self.cells == (i + 1),
                                            c[3:], c[0:3])
                spacing = self._pixel_dim
                extent = [0, self._img.GetSize()[0],
                          0, self._img.GetSize()[1],
                          0, self._img.GetSize()[2]]
                iso = vtk.vtkImageMarchingCubes()
                iso.ComputeNormalsOff()
            else:
                roi = sitk.RegionOfInterest(self.cells == (i + 1),
                                            c[3:5], c[0:2])
                spacing = self._pixel_dim + [1]
                extent = [0, self._img.GetSize()[0],
                          0, self._img.GetSize()[1],
                          0, 0]
                iso = vtk.vtkMarchingSquares()
            roi = sitk.BinaryDilate(roi, 1)
            if not(self.levelsets):
                smoothed = sitk.Cast(roi,
                                     sitk.sitkFloat32) * self.smoothed[i]
                resampler.SetInterpolator(sitk.sitkBSpline)
                smoothlabel = resampler.Execute(smoothed)
                a = vti.vtkImageImportFromArray()
                a.SetDataSpacing(spacing)
                a.SetDataExtent(extent)
                n = sitk.GetArrayFromImage(smoothlabel)
                a.SetArray(n)
                a.Update()

                voi = vtk.vtkExtractVOI()
                voi.SetInputData(a.GetOutput())
                voi.SetVOI(c[0] - 1, c[0]+c[3] + 1,
                           c[1] - 1, c[1]+c[4] + 1,
                           c[2] - 1, c[2]+c[5] + 1)
                voi.SetSampleRate(1, 1, 1)
                voi.IncludeBoundaryOn()
                voi.Update()

                iso.SetInputData(voi.GetOutput())
                iso.SetValue(0, self.thresholds[i])
                iso.Update()

            else:
                resampler.SetInterpolator(sitk.sitkLinear)
                lvlset = resampler.Execute(self.levelsets[i])
                lvl_roi = sitk.RegionOfInterest(lvlset, c[3:], c[0:3])
                lvlset = sitk.Cast(roi, sitk.sitkFloat32) * lvl_roi
                lvl_label = resampler.Execute(lvlset)
                a = vti.vtkImageImportFromArray()
                a.SetDataSpacing(spacing)
                a.SetDataExtent(extent)
                n = sitk.GetArrayFromImage(lvl_label)
                a.SetArray(n)
                a.Update()
                voi = vtk.vtkExtractVOI()
                voi.SetInputData(a.GetOutput())
                voi.SetVOI(c[0] - 1, c[0]+c[3] + 1,
                           c[1] - 1, c[1]+c[4] + 1,
                           c[2] - 1, c[2]+c[5] + 1)
                voi.SetSampleRate(1, 1, 1)
                voi.IncludeBoundaryOn()
                voi.Update()

                iso.SetInputData(a.GetOutput())
                if self.active == "Geodesic":
                    iso.SetValue(0, -1e-7)
                else:
                    iso.SetValue(0, 1e-7)
                iso.Update()

            triangles = vtk.vtkGeometryFilter()
            triangles.SetInputConnection(iso.GetOutputPort())
            triangles.Update()

            if self._img.GetDimension() == 3:
                smooth = vtk.vtkWindowedSincPolyDataFilter()
                smooth.SetInputConnection(triangles.GetOutputPort())
                smooth.NormalizeCoordinatesOn()
                smooth.SetNumberOfIterations(30)
                smooth.SetPassBand(0.01)
                smooth.SetFeatureAngle(120.0)
                smooth.Update()

                laplaceSmooth = vtk.vtkSmoothPolyDataFilter()
                laplaceSmooth.SetInputConnection(smooth.GetOutputPort())
                laplaceSmooth.SetNumberOfIterations(5)
                laplaceSmooth.Update()

                self.surfaces.append(laplaceSmooth.GetOutput())
                filename = 'cell{:02d}.stl'.format(i + 1)
                stl.SetFileName(
                    str(os.path.normpath(self._output_dir +
                                         os.sep + filename)))
                stl.SetInputData(self.surfaces[-1])
                stl.Write()
            else:
                self.surfaces.append(triangles.GetOutput())
                filename = 'cell{:0d}.vtk'.format(i + 1)
                polywriter.SetFileName(
                    str(os.path.normpath(self._output_dir +
                                         os.sep + filename)))
                polywriter.SetInputData(self.surfaces[-1])
                polywriter.Write()

        if self.display and self._img.GetDimension() == 3:
            N = len(self.surfaces)
            colormap = vtk.vtkLookupTable()
            colormap.SetHueRange(0.9, 0.1)
            colormap.SetTableRange(1, N)
            colormap.SetNumberOfColors(N)
            colormap.Build()
            skins = []
            for i, s in enumerate(self.surfaces):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(s)
                mapper.SetLookupTable(colormap)
                mapper.ScalarVisibilityOff()
                skin = vtk.vtkActor()
                skin.SetMapper(mapper)
                color = [0, 0, 0]
                colormap.GetColor(i + 1, color)
                skin.GetProperty().SetColor(color)
                skins.append(skin)

            #Create a colorbar
            colorbar = vtk.vtkScalarBarActor()
            colorbar.SetLookupTable(colormap)
            colorbar.SetTitle("Cells")
            colorbar.SetNumberOfLabels(N)
            colorbar.SetLabelFormat("%3.0f")

            #Create the renderer, the render window, and the interactor.
            #The renderer draws into the render window, the interactor enables
            #mouse- and keyboard-based interaction with the data
            #within the render window.
            aRenderer = vtk.vtkRenderer()
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(aRenderer)
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)

            #It is convenient to create an initial view of the data.
            #The FocalPoint and Position form a vector direction.
            #Later on in (ResetCamera() method) this vector is used
            #to position the camera to look at the data in this direction.
            aCamera = vtk.vtkCamera()
            aCamera.SetViewUp(0, 0, -1)
            aCamera.SetPosition(0, 1, 1)
            aCamera.SetFocalPoint(0, 0, 0)
            aCamera.ComputeViewPlaneNormal()

            #Actors are added to the renderer.
            #An initial camera view is created.
            # The Dolly() method moves the camera towards the FocalPoint,
            # thereby enlarging the image.
            for skin in skins:
                aRenderer.AddActor(skin)
            aRenderer.AddActor(colorbar)
            aRenderer.SetActiveCamera(aCamera)
            aRenderer.ResetCamera()
            aCamera.Dolly(1.5)

            bounds = a.GetOutput().GetBounds()
            triad = vtk.vtkCubeAxesActor()
            l = 0.5 * (bounds[5] - bounds[4])
            triad.SetBounds([bounds[0], bounds[0] + l,
                             bounds[2], bounds[2] + l,
                             bounds[4], bounds[4] + l])
            triad.SetCamera(aRenderer.GetActiveCamera())
            triad.SetFlyModeToStaticTriad()
            triad.GetXAxesLinesProperty().SetColor(1.0, 0.0, 0.0)
            triad.GetYAxesLinesProperty().SetColor(0.0, 1.0, 0.0)
            triad.GetZAxesLinesProperty().SetColor(0.0, 0.0, 1.0)
            triad.GetXAxesLinesProperty().SetLineWidth(5.0)
            triad.GetYAxesLinesProperty().SetLineWidth(5.0)
            triad.GetZAxesLinesProperty().SetLineWidth(5.0)
            triad.XAxisLabelVisibilityOff()
            triad.YAxisLabelVisibilityOff()
            triad.ZAxisLabelVisibilityOff()
            triad.XAxisTickVisibilityOff()
            triad.YAxisTickVisibilityOff()
            triad.ZAxisTickVisibilityOff()
            triad.XAxisMinorTickVisibilityOff()
            triad.YAxisMinorTickVisibilityOff()
            triad.ZAxisMinorTickVisibilityOff()
            aRenderer.AddActor(triad)
            #Set a background color for the renderer and set the size of the
            #render window (expressed in pixels).
            aRenderer.SetBackground(0.0, 0.0, 0.0)
            renWin.SetSize(800, 600)

            #Note that when camera movement occurs (as it does in the Dolly()
            #method), the clipping planes often need adjusting. Clipping planes
            #consist of two planes: near and far along the view direction. The
            #near plane clips out objects in front of the plane the far plane
            #clips out objects behind the plane. This way only what is drawn
            #between the planes is actually rendered.
            aRenderer.ResetCameraClippingRange()

            im = vtk.vtkWindowToImageFilter()
            im.SetInput(renWin)

            iren.Initialize()
            iren.Start()

            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(renWin)
            windowToImageFilter.SetMagnification(3)
            windowToImageFilter.SetInputBufferTypeToRGBA()
            windowToImageFilter.ReadFrontBufferOff()
            windowToImageFilter.Update()

            pngWriter = vtk.vtkPNGWriter()
            pngWriter.SetFileName(
                str(os.path.normpath(self._output_dir + os.sep + "cells.png")))
            pngWriter.SetInputConnection(windowToImageFilter.GetOutputPort())
            pngWriter.Write()

    def writeLabels(self):
        sitk.WriteImage(self.cells,
                        str(os.path.normpath(
                            self._output_dir + os.sep + 'labels.nii')))

    def getDimensions(self):
        labelstats = self._getLabelShape(self.cells)
        self.volumes = labelstats['volume']
        self.centroids = labelstats['centroid']
        self.dimensions = labelstats['ellipsoid diameters']

    def adjustForDepth(self):
        r"""
        Iterates over slices in 3-D image stack and appends each slice's
        maximum pixel intensity to a list. Then performs a weighted linear
        curve fit with slices below the :math:`2^{nd}` percentile and above
        the :math:`98^{th}` percentile assigned zero weights with all others
        equally weighted. Ratios are then calculated from this fit for all z-depths
        as:

        :math:`\frac{a_0}{a_1 z + a_0}`

        and each slice of the image is multiplied by its corresponding weight
        and reassembled into a 3-D image.

        Returns
        -------
        Replaces image read from disk with and image that is corrected for
        intensity change with depth.
        """
        stack = []
        intensities = []
        size = self._img.GetSize()
        intensities = np.zeros(size[2], np.float32)
        for sl in range(size[2]):
            s = sitk.Extract(self._img, [size[0], size[1], 0], [0, 0, sl])
            intensities[sl] = self._getMinMax(s)[1]
            stack.append(s)
        low = np.percentile(intensities, 2)
        high = np.percentile(intensities, 98)
        w = np.ones(size[2], np.float32)
        w[intensities < low] = 0.0
        w[intensities > high] = 0.0
        x = np.arange(size[2], dtype=np.float32)
        fit = np.polyfit(x, intensities, 1, w=w)
        ratios = old_div(fit[1], (fit[0] * x + fit[1]))
        for i, s in enumerate(stack):
            stack[i] = sitk.Cast(s, sitk.sitkFloat32) * ratios[i]
        nimg = sitk.JoinSeries(stack)
        nimg.CopyInformation(self._img)
        self._img = nimg

    def smooth2D(self, img):
        stack = []
        size = img.GetSize()
        for sl in range(size[2]):
            s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, sl])
            stack.append(self.smoothRegion(s))
        simg = sitk.JoinSeries(stack)
        simg.CopyInformation(img)
        return simg

    def threshold2D(self, img, thres, ratio):
        stack = []
        values = []
        size = img.GetSize()
        if (thres != "Percentage"):
            for sl in range(size[2]):
                s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, sl])
                seg = thres.Execute(s)
                stack.append(sitk.BinaryFillhole(seg != 0))
                values.append(thres.GetThreshold())
        else:
            for sl in range(size[2]):
                s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, sl])
                t = self._getMinMax(s)[1]
                t *= ratio
                seg = sitk.BinaryThreshold(s, t, 1e7)
                stack.append(sitk.BinaryFillhole(seg != 0))
                values.append(t)
 
        seg = sitk.JoinSeries(stack)
        seg.CopyInformation(img)
        return seg, max(values), min(values), values

    def scale2D(self, img, thresh):
        stack = []
        size = img.GetSize()
        maxt = np.max(thresh)
        for i, t in enumerate(thresh):
            s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, i])
            # maximum difference from max of 300%
            if (old_div(maxt, t)) < 3:
                stack.append(sitk.Cast(s, sitk.sitkFloat32) * (old_div(maxt, t)))
            else:
                stack.append(sitk.Cast(s, sitk.sitkFloat32))
        nimg = sitk.JoinSeries(stack)
        nimg.CopyInformation(img)
        return nimg

    def geodesic2D(self, seed, simg,
                   cannyLower, cannyUpper, canny_variance,
                   upsampling, active_iterations, rms,
                   propagation, curvature, advection):
        """
        A 2-D implementation of **geodesicSegmentation()** that operates
        on each slice in the 3-D stack independently.
        """
        gd = sitk.GeodesicActiveContourLevelSetImageFilter()
        gd.SetMaximumRMSError(old_div(rms, float(upsampling)))
        gd.SetNumberOfIterations(active_iterations)
        gd.SetPropagationScaling(propagation)
        gd.SetCurvatureScaling(curvature)
        gd.SetAdvectionScaling(advection)
        stack = []
        size = simg.GetSize()
        for sl in range(size[2]):
            im = sitk.Extract(simg, [size[0], size[1], 0], [0, 0, sl])
            s = sitk.Extract(seed, [size[0], size[1], 0], [0, 0, sl])
            canny = sitk.CannyEdgeDetection(
                sitk.Cast(im, sitk.sitkFloat32),
                lowerThreshold=cannyLower,
                upperThreshold=cannyUpper,
                variance=canny_variance)
            canny = sitk.InvertIntensity(canny, 1)
            canny = sitk.Cast(canny, sitk.sitkFloat32)
            d = sitk.SignedMaurerDistanceMap(s,
                                             insideIsPositive=False,
                                             squaredDistance=False,
                                             useImageSpacing=True)
            stack.append(gd.Execute(d, canny))
        seg = sitk.JoinSeries(stack)
        seg.CopyInformation(simg)
        return seg

    def _replaceSeed(self, seed):
        print(("WARNING: seed for active segmentation was zero; using sphere"
               " with diameter half of minimum region of interest edge."))
        size = np.array(seed.GetSize(), int)
        idx = size * np.array(seed.GetSpacing(), float) / 2.0
        d = old_div(np.min(size), 2)
        seed[seed.TransformPhysicalPointToIndex(idx)] = 1
        seed = sitk.BinaryDilate(seed, d)
        return seed

    def _flattenBorder(self, img):
        """
        To help ensure the segmentation does not touch
        the cropped region border, the voxel intensities of
        the 6 border slices are replaced by the intensity of
        their 1st percentile.
        """
        arr = sitk.GetArrayFromImage(img)
        if len(arr.shape) == 3:
            ind = [np.s_[0, :, :], np.s_[-1, :, :],
                   np.s_[:, 0, :], np.s_[:, -1, :],
                   np.s_[:, : , 0], np.s_[:, :, -1]]
        else:
            ind = [np.s_[0, :], np.s_[-1, :],
                   np.s_[:, 0], np.s_[:, -1]]
        for i in ind:
            arr[i] = np.percentile(arr[i].ravel(), 10)

        nimg = sitk.GetImageFromArray(arr)
        nimg.CopyInformation(img)
        return nimg
