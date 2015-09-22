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

    """
    DESCRIPTION
    This class will segment objects from 3-D images using user-specified
    routines. The intended purpose is for laser scanning fluorescence
    microscopy of chondrocytes and/or their surrounding matrices.
    Nevertheless, this can be generalized to any 3-D object using any
    imaging modality; however, it is likely the segmentation parameters
    will need to be adjusted. Therefore, in this case, the user should set
    segmentation='User' during Class instantiation, and call the segmentaion
    method with appropriate parameters.

    Attributes:
    cells        An image containing the segmented objects as integer labels.
                 Has the same properties as the input image stack.
    thresholds   The threshold level for each cell
    volumes      List of the physical volumes of the segmented objects.
    centroids    List of centroids of segmented objects in physical space.
    surfaces     List containing VTK STL objects.
    dimensions   List containing the ellipsoid axis lengths of segmented
                 objects.
    orientations List containing the basis vectors of ellipsoid axes.
                 Same order as dimensions.
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
        """
        INPUTS
        vol_dir              TYPE: string. This is required. Currently it is
                             the path to a directory containing a stack of
                             TIFF images. Other formats may be supported in
                             the future.
        output_dir           TYPE: string. Directory to write STL surfaces to.
                             If not specifed, will create a directory
                             vol_dir+'_results'.
        regions              TYPE: list of form [[pixel coordinate x, y, z,
                             box edge length x, y, z],[...]].
                             If not specified, assumes whole image region.
        pixel_dim            TYPE: [float, float, float].
                             The physical dimensions of the voxels in the
                             image.
        stain                TYPE: string. Indicates if the object to be
                             segmented is the foreground or the background.
        segmentation         TYPE: string. Execute indicated segmentation
                             using default values.
                             'User'      The user will invoke the segmentation
                             method by calling the function.
                             This allows for parameter specification.
                             'Threshold' Thresholds based on a user-supplied
                                         percentage of the maximum voxel
                                         intensity. See thresholdSegmentation
                                         for other methods available if 'User'
                                         is indicated.
                             'Geodesic'  Uses a geodesic active contour.
                             'EdgeFree'  Uses an edge free active contour.
        smoothing_method     TYPE: string. Smoothing method to use on
                             regions of interest.
        smoothing_parameters TYPE: dictionary. Change smoothing parameters
                             of smoothing method from default by passing a
                             dictionary with key and new value.
                             Dictionary Defaults by Method:
                             'Gaussian':            {'sigma': 0.5}
                             'Median':              {'radius': (1, 1, 1)}
                             'Curvature Diffusion': {'iterations': 10,
                                                     'conductance': 9}
                             'Gradient Diffusion':  {'iterations': 10,
                                                     'conductance': 9,
                                                     'time step': 0.01}
                             'Bilateral':           {'domainSigma': 1.5,
                                                     'rangeSigma': 5.0,
                                                     'samples': 100}
                             'Patch-based':         {'radius': 4,
                                                     'iterations': 10,
                                                     'patches': 20,
                                                     'noise model': 'poisson'}
        two_dim              TYPE: Boolean. If true treat each 2D slice in
                             stack as independent.
        bright               TYPE: Boolean. Whether to replace voxels higher
                             than 98 percentile intensity with median value
                             (radius 6)
        enhance_edge         TYPE: Boolean. Whether to enhance the edges with
                             Laplacian sharpening
        depth_adjust         TYPE: Boolean. Adjust image intensity by a linear
                             function fit to the max intensity vs depth.
        display              TYPE: Boolean. Spawn a window to render the cells
                             or not.
        handle_overlap       TYPE: Boolean. If labelled objects overlap,
                             employs Support Vector Machines to classify
                             the shared voxels.
        debug                TYPE: Boolean. If True, the following images
                             depending on the segmentation method will be
                             output to the output_dir.
                             thresholdSegmentation:
                              smoothed region of interest image as
                              smoothed_[region id].nii e.g. smoothed_001.nii
                             edgeFreeSegmentation:
                              All of the above plus: seed image for each
                              region as seed_[region id].nii e.g. seed_001.nii
                             geodesicSegmentation:
                              All of the above plus: edge map image for each
                              region as edge_[region id].nii e.g. edge_001.nii
        opening              TYPE: Boolean. If True, perform morphological
                             opening. If object to detect is small or thin this
                             setting this False, may be needed.
        fillholes            TYPE: Boolean. If True, holes fully within the
                             segmented object will be filled.
        """
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
        self._imgType = None
        self._imgTypeMax = None

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

        # define a blank image with the same size and spacing as
        # image stack to add segmented cells to
        self.cells = sitk.Image(self._img.GetSize(), self._imgType)
        self.cells.SetSpacing(self._img.GetSpacing())
        self.cells.SetOrigin(self._img.GetOrigin())
        self.cells.SetDirection(self._img.GetDirection())

        # list of smoothed ROIs
        self.smoothed = []
        # list of threshold values
        self.thresholds = []
        # list of levelsets
        self.levelsets = []

        self.surfaces = []
        # if regions are not specified, assume there is only one cell
        # and default to whole image
        if regions is None:
            size = np.array(self._img.GetSize(), int) - 1
            if self._img.GetDimension() == 3:
                self._regions = [[0, 0, 0] + list(size)]
            else:
                self._regions = [[0, 0] + list(size)]
        else:
            self._regions = regions

        self.volumes = []
        self.centroids = []
        self.dimensions = []

        #Execute segmentation with default parameters
        #unless specified as 'User'
        if segmentation == 'Threshold':
            self.thresholdSegmentation()
        elif segmentation == 'Geodesic':
            self.geodesicSegmentation()
        elif segmentation == 'EdgeFree':
            self.edgeFreeSegmentation()
        elif segmentation == 'User':
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
        reader = sitk.ImageFileReader()
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
                    reader.SetFileName(filename)
                    im = reader.Execute()
                    if 'vector' in string.lower(im.GetPixelIDTypeAsString()):
                        img.append(sitk.VectorMagnitude(im))
                    else:
                        img.append(im)
                self._img = sitk.JoinSeries(img)
                print("\nImported 3D image stack ranging from {:s} to {:s}"
                      .format(files[0], files[-1]))
            else:
                print("\nImported 2D image {:s}".format(files[0]))
                filename = str(
                    os.path.normpath(self._vol_dir + os.sep + files[0]))
                reader.SetFileName(filename)
                self._img = reader.Execute()
        elif ftype == "*.nii":
            filename = str(
                os.path.normpath(self._vol_dir + os.sep + files[0]))
            im = sitk.ReadImage(filename)
            if 'vector' in string.lower(im.GetPixelIDTypeAsString()):
                im = sitk.VectorMagnitude(im)
            self._img = im
        #temporarily force convert image to 8bit due to SimpleITK bug
        self._img = sitk.Cast(sitk.RescaleIntensity(self._img, 0, 255),
                              sitk.sitkUInt8)

        self._imgType = self._img.GetPixelIDValue()
        if self._imgType == 1:
            self._imgTypeMax = 255
        elif self._imgType == 3:
            self._imgTypeMax = 65535
        elif self._imgType == 0:
            print("WARNING: Given a 12-bit image.")
            print("This has been converted to 16-bit.")
            self._imgType = 3
            self._imgTypeMax = 65535

        self._img = sitk.Cast(self._img, self._imgType)
        self._img.SetSpacing(self._pixel_dim)

    def smoothRegion(self, img):
        img = sitk.Cast(img, sitk.sitkFloat32)

        if self.smoothing_method == 'None':
            pass
        elif self.smoothing_method == 'Gaussian':
            parameters = {'sigma': 0.5}
            for p in self.smoothing_parameters.keys():
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            img = sitk.DiscreteGaussian(img, variance=parameters['sigma'])

        elif self.smoothing_method == 'Median':
            parameters = {'radius': (1, 1, 1)}
            for p in self.smoothing_parameters.keys():
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            img = sitk.Median(img, radius=parameters['radius'])

        elif self.smoothing_method == 'Curvature Diffusion':
            parameters = {'iterations': 10, 'conductance': 9}
            for p in self.smoothing_parameters.keys():
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
            for p in self.smoothing_parameters.keys():
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
            for p in self.smoothing_parameters.keys():
                try:
                    parameters[p] = self.smoothing_parameters[p]
                except:
                    raise SystemExit("{:s} is not a parameter of {:s}"
                                     .format(p, self.smoothing_method))
            img = sitk.Bilateral(
                img,
                domainSigma=parameters['domainSigma'],
                rangeSigma=parameters['rangeSigma'],
                numberOfRangeGaussianSamples=parameters['samples'])

        elif self.smoothing_method == 'Patch-based':
            parameters = {'radius': 4,
                          'iterations': 10,
                          'patches': 20,
                          'noise model': 'poisson'}
            noise_models = {'nomodel': 0,
                            'gaussian': 1,
                            'rician': 2,
                            'poisson': 3}
            for p in self.smoothing_parameters.keys():
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
        return sitk.Cast(img, sitk.sitkFloat32)

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
                      'bounding box': []}
        for l in labels:
            labelshape['volume'].append(ls.GetPhysicalSize(l))
            labelshape['centroid'].append(ls.GetCentroid(l))
            labelshape['ellipsoid diameters'].append(
                ls.GetEquivalentEllipsoidDiameter(l))
            labelshape['bounding box'].append(ls.GetBoundingBox(l))
        return labelshape

    def thresholdSegmentation(self, method='Percentage',
                              adaptive=True, ratio=0.4):
        """
        DESCRIPTION
        Segments image based on a specified percentage of the maximum voxel
        intensity in the specified region of interest.
        For the case of multiple objects in the region, saves only the object
        with the greatest volume.

        INPUTS
        method     TYPE: string. The thresholding method to use.
                   OPTIONS
                   'Percentage'  Threshold at percentage of the maximum voxel
                                 intensity.
                   'Otsu'
                   For more information on the following consult
                   http://www.insight-journal.org/browse/publication/811
                   and cited original sources.
                   'Huang'
                   'IsoData'
                   'Li'
                   'MaxEntropy'
                   'KittlerIllingworth'
                   'Moments'
                   'Yen'
                   'RenyiEntropy'
                   'Shanbhag'
        ratio      TYPE: float. Percentage to threshold at if using
                   'Percentage' method.
        adaptive   TYPE: Boolean. Whether to adaptively adjust initial
                   threshold until foreground does not touch the region
                   boundaries.
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
            print("Segmenting Cell {:d}".format(i + 1))
            print("------------------\n")

            if method == 'Percentage':
                t = self._getMinMax(simg)[1]
                if self._stain == 'Foreground':
                    if self.two_dim:
                        seg, thigh, tlow, tlist = self.threshold2D(
                            simg, 'PFore', ratio)
                    else:
                        t *= ratio
                        seg = sitk.BinaryThreshold(simg, t, 1e7)
                elif self._stain == 'Background':
                    if self.two_dim:
                        seg, thigh, tlow, tlist = self.threshold2D(
                            simg, 'PBack', ratio)
                    else:
                        t *= (1.0 - ratio)
                        seg = sitk.BinaryThreshold(simg, 0, t)
                else:
                    raise SystemExit(("Unrecognized value for 'stain', {:s}. "
                                      "Options are 'Foreground' or "
                                      "'Background'"
                                      .format(self._stain)))
                if self.two_dim:
                    print(("... Threshold using {:s} method ranged: "
                           "{:d}-{:d}".format(method, int(tlow), int(thigh))))
                else:
                    print(("... Thresholded using {:s} method at a "
                           "value of: {:d}".format(method, int(t))))

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
                thres.SetNumberOfHistogramBins((self._imgTypeMax + 1) / 2)
                if self._stain == 'Foreground':
                    thres.SetInsideValue(0)
                    thres.SetOutsideValue(1)
                elif self._stain == 'Background':
                    thres.SetInsideValue(1)
                    thres.SetOutsideValue(0)
                if self.two_dim:
                    seg, thigh, tlow, tlist = self.threshold2D(
                        simg, thres, ratio)
                    print(("... Thresholds determined by {:s} method ranged: "
                           "[{:d}-{:d}".format(method, int(tlow), int(thigh))))
                else:
                    seg = thres.Execute(simg)
                    t = thres.GetThreshold()
                    print("... Threshold determined by {:s} method: {:d}"
                          .format(method, int(t)))

            if adaptive and not(self.two_dim):
                newt = np.copy(t)
                if dimension == 3:
                    region_bnds = [(0, region[3]), (0, region[4])]
                else:
                    region_bnds = [(0, region[2]), (0, region[3])]
                while True:
                    if self.opening:
                        #Opening (Erosion/Dilation) step to remove islands
                        #smaller than 1 voxels in radius
                        seg = sitk.BinaryMorphologicalOpening(seg, 1)
                    if self.fillholes:
                        seg = sitk.BinaryFillhole(seg != 0)
                    #Get connected regions
                    r = sitk.ConnectedComponent(seg)
                    labelstats = self._getLabelShape(r)
                    d = 1e7
                    region_cent = np.array(list(seg.GetSize()), float) / 2.0
                    region_cent *= np.array(self._pixel_dim)
                    region_cent += np.array(list(seg.GetOrigin()), float)
                    for l, c in enumerate(labelstats['centroid']):
                        dist = np.linalg.norm(np.array(c, float) - region_cent)
                        if dist < d:
                            d = dist
                            label = l + 1
                    # if exception here, then threshold adjusted too much
                    # and previous increment will be taken
                    try:
                        bb = labelstats['bounding box'][label - 1]
                    except:
                        break

                    if dimension == 3:
                        label_bounds = [(bb[0], bb[0] + bb[3]),
                                        (bb[1], bb[1] + bb[4])]
                    else:
                        label_bounds = [(bb[0], bb[0] + bb[2]),
                                        (bb[1], bb[1] + bb[3])]
                    if np.any(np.intersect1d(region_bnds[0],
                                             label_bounds[0])) or \
                       np.any(np.intersect1d(region_bnds[1],
                                             label_bounds[1])):
                        if self._stain == 'Foreground':
                            newt += 0.01 * t
                            seg = sitk.BinaryThreshold(simg, int(newt), 1e7)
                        elif self._stain == 'Background':
                            newt -= 0.01 * t
                            seg = sitk.BinaryThreshold(simg, 0, int(newt))
                    else:
                        break

                if not(newt == t):
                    print(("... ... Adjusted the threshold to: "
                          "{:d}".format(int(newt))))
                self.thresholds.append(newt)
            else:
                if self.opening:
                    #Opening (Erosion/Dilation) step to remove islands
                    #smaller than 1 voxels in radius
                    seg = sitk.BinaryMorphologicalOpening(seg, 1)
                if self.fillholes:
                    seg = sitk.BinaryFillhole(seg != 0)
                #Get connected regions
                r = sitk.ConnectedComponent(seg)
                labelstats = self._getLabelShape(r)
                d = 1e7
                region_cent = np.array(list(seg.GetSize()), float) / 2.0
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

            tmp = sitk.Image(self._img.GetSize(), self._imgType)
            tmp.SetSpacing(self._img.GetSpacing())
            tmp.SetOrigin(self._img.GetOrigin())
            tmp.SetDirection(self._img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tmp)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            tmp = resampler.Execute((r == label) * (i + 1))
            self.cells = sitk.Add(self.cells, sitk.Cast(tmp, self._imgType))
            # scale smoothed image if independent slices option flagged
            if self.two_dim:
                simg = self.scale2D(simg, tlist)
            self.smoothed.append(simg)
            if self.debug:
                sitk.WriteImage(sitk.Cast(simg, self._imgType),
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
                             propagation=0.3,
                             curvature=0.1,
                             advection=1.0,
                             rms=0.005,
                             active_iterations=200):
        """
        DESCRIPTION
        Performs a segmentation using the SimpleITK implementation of the
        Geodesic Active Contour Levelset Segmentation method described in
        (Caselles et al. 1997.)Please also consult SimpleITK's documentation
        of GeodesicActiveContourLevelSetImageFilter.
        This method will establish initial levelsets by calling the
        entropySegmentation() method.

        INPUTS
        upsampling           TYPE: integer. Resample image splitting original
                             voxels into this many. NOTE - Resampling will
                             always be performed to make voxels isotropic.
        seed_method          TYPE: string. Method used to determine seed image.
                             Same as thresholdSegmentation method variable.

                          OPTIONS
                          'Percentage'  Threshold at percentage of the maximum
                                        voxel intensity.
                          'Otsu'
                          For more information on the following consult
                          http://www.insight-journal.org/browse/publication/811
                          and cited original sources.
                          'Huang'
                          'IsoData'
                          'Li'
                          'MaxEntropy'
                          'KittlerIllingworth'
                          'Moments'
                          'Yen'
                          'RenyiEntropy'
                          'Shanbhag'
        adaptive          TYPE: Boolean. If true will adaptively adjust
                          threshold
        ratio             TYPE: float. The ratio to use with 'Percentage' seed
                          method. This plays no role with other seed methods.

        canny_variance    TYPE: [float, float, float].
                          Variance for canny edge detection.

        cannyUpper        TYPE: float. Upper threshold for Canny edge detector.

        cannyLower        TYPE: float. Lower threshold for Canny edge detector.

        propagation       TYPE: float. Weight for propagation term in active
                          contour functional.
                          Higher results in faster expansion.
        curvature         TYPE: float. Weight for curvature term in active
                          contour functional. Higher results in
                          smoother segmentation.
        advection         TYPE: float. Weight for advective term in active
                          contour functional.
                          Higher causes levelset to move toward edges.
        rms               TYPE: float. The change in Root Mean Square at which
                          iterations will terminate.
        active_iterations TYPE: integer. The maximum number of iterations the
                          active contour will conduct.
        """
        self.active = "Geodesic"
        self.thresholdSegmentation(method=seed_method, ratio=ratio,
                                   adaptive=adaptive)
        dimension = self._img.GetDimension()
        newcells = sitk.Image(self.cells.GetSize(), self._imgType)
        newcells.SetSpacing(self.cells.GetSpacing())
        newcells.SetDirection(self.cells.GetDirection())
        newcells.SetOrigin(self.cells.GetOrigin())
        for i, region in enumerate(self._regions):
            print("\n-------------------------------------------")
            print("Evolving Geodesic Active Contour for Cell {:d}"
                  .format(i + 1))
            print("-------------------------------------------")
            if dimension == 3:
                seed = sitk.RegionOfInterest(self.cells,
                                             region[3:],
                                             region[0:3])
                roi = self.smoothed[i]
                #resample the Region of Interest to improve resolution of
                #derivatives and give closer to isotropic voxels
                zratio = self._pixel_dim[2] / self._pixel_dim[0]
                #adjust size in z to be close to isotropic and double
                #the resolution
                newz = int(zratio * roi.GetSize()[2]) * upsampling
                newzspace = (float(roi.GetSize()[2])
                             / float(newz)) * self._pixel_dim[2]
                newx = roi.GetSize()[0] * upsampling
                newxspace = self._pixel_dim[0] / float(upsampling)
                newy = roi.GetSize()[1] * upsampling
                newyspace = self._pixel_dim[1] / float(upsampling)
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
                newxspace = self._pixel_dim[0] / float(upsampling)
                newy = roi.GetSize()[1] * upsampling
                newyspace = self._pixel_dim[1] / float(upsampling)
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
            #smooth the perimeter of the binary seed
            seed = sitk.BinaryMorphologicalClosing(seed == (i + 1), 3)
            seed = sitk.AntiAliasBinary(seed)
            seed = sitk.BinaryThreshold(seed, 0.5, 1e7)
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
                if self.debug:
                    sitk.WriteImage(sitk.Cast(simg, self._imgType),
                                    str(os.path.normpath(
                                        self._output_dir +
                                        os.sep + "smoothed_{:03d}.nii"
                                        .format(i + 1))))
                    sitk.WriteImage(sitk.Cast(seed, self._imgType),
                                    str(os.path.normpath(
                                        self._output_dir +
                                        os.sep + "seed_{:03d}.nii"
                                        .format(i + 1))))
                    sitk.WriteImage(sitk.Cast(canny, self._imgType),
                                    str(os.path.normpath(
                                        self._output_dir +
                                        os.sep + "edge_{:03d}.nii"
                                        .format(i + 1))))
                d = sitk.SignedMaurerDistanceMap(seed, insideIsPositive=False,
                                                 squaredDistance=False,
                                                 useImageSpacing=True)
                gd = sitk.GeodesicActiveContourLevelSetImageFilter()
                gd.SetMaximumRMSError(rms / float(upsampling))
                gd.SetNumberOfIterations(active_iterations)
                gd.SetPropagationScaling(propagation)
                gd.SetCurvatureScaling(curvature)
                gd.SetAdvectionScaling(advection)
                seg = gd.Execute(d, canny)
                print("... Geodesic Active Contour Segmentation Completed")
                print("... ... Elapsed Iterations: {:d}"
                      .format(gd.GetElapsedIterations()))
                print("... ... Change in RMS Error: {:.3e}"
                      .format(gd.GetRMSChange()))
            self.levelsets.append(seg)
            seg = sitk.BinaryThreshold(seg, -1e7, 0) * (i + 1)
            tmp = sitk.Image(self._img.GetSize(), self._imgType)
            tmp.SetSpacing(self._img.GetSpacing())
            tmp.SetOrigin(self._img.GetOrigin())
            tmp.SetDirection(self._img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tmp)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            tmp = resampler.Execute(seg)
            if self.fillholes:
                tmp = sitk.BinaryFillhole(tmp)
            newcells = sitk.Add(newcells, sitk.Cast(tmp, self._imgType))
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
                             curvature=1.0,
                             iterations=20):
        """
        DESCRIPTION
        Performs a segmentation using the SimpleITK implementation of the
        Active Contours Without Edges method described in
        (Chan and Vese. 2001.)
        Please also consult SimpleITK's documentation of
        ScalarChanAndVeseDenseLevelSetImageFilter.
        This method will establish initial levelsets by calling
        the entropySegmentation() method.

        INPUTS
        upsampling  TYPE: integer. Resample image splitting original voxels
                    into this many. NOTE - Resampling will always be performed
                    to make voxels isotropic.
        seed_method TYPE: string. Method used to determine seed image.
                    Same as thresholdSegmentation method variable.

                    OPTIONS
                    'Percentage'  Threshold at percentage of the maximum
                                  voxel intensity.
                    'Otsu'
                    For more information on the following consult
                    http://www.insight-journal.org/browse/publication/811
                    and cited original sources.
                    'Huang'
                    'IsoData'
                    'Li'
                    'MaxEntropy'
                    'KittlerIllingworth'
                    'Moments'
                    'Yen'
                    'RenyiEntropy'
                    'Shanbhag'
        adaptive    TYPE: Boolean. If true will adaptively adjust seed
                    threshold.
        ratio       TYPE: float. The ratio to use with 'Percentage'
                    seed method. This plays no role with other seed methods.

        lambda1     TYPE: float. Weight for internal levelset term.
        lambda2     TYPE: float. Weight for external levelset term.
        curvature   TYPE: float. Weight for curvature. Higher results
                    in smoother levelsets, but less ability to capture
                    fine features.
        iterations  TYPE: integer. The number of iterations the active
                    contour method will conduct.
        """
        self.active = "EdgeFree"
        self.thresholdSegmentation(method=seed_method, ratio=ratio,
                                   adaptive=adaptive)
        dimension = self._img.GetDimension()
        newcells = sitk.Image(self.cells.GetSize(), self._imgType)
        newcells.SetSpacing(self.cells.GetSpacing())
        newcells.SetDirection(self.cells.GetDirection())
        newcells.SetOrigin(self.cells.GetOrigin())
        for i, region in enumerate(self._regions):
            print("\n-------------------------------------------")
            print("Evolving Edge-free Active Contour for Cell {:d}"
                  .format(i + 1))
            print("-------------------------------------------")
            if dimension == 3:
                seed = sitk.RegionOfInterest(self.cells,
                                             region[3:],
                                             region[0:3])
                roi = self.smoothed[i]
                #resample the Region of Interest to improve resolution
                #of derivatives and give closer to isotropic voxels
                zratio = self._pixel_dim[2] / self._pixel_dim[0]
                #adjust size in z to be close to isotropic and
                #double the resolution
                newz = int(zratio * roi.GetSize()[2]) * upsampling
                newzspace = (float(roi.GetSize()[2])
                             / float(newz)) * self._pixel_dim[2]
                newx = roi.GetSize()[0] * upsampling
                newxspace = self._pixel_dim[0] / float(upsampling)
                newy = roi.GetSize()[1] * upsampling
                newyspace = self._pixel_dim[1] / float(upsampling)
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
                newxspace = self._pixel_dim[0] / float(upsampling)
                newy = roi.GetSize()[1] * upsampling
                newyspace = self._pixel_dim[1] / float(upsampling)
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
            #smooth the perimeter of the binary seed
            seed = sitk.BinaryMorphologicalClosing(seed == (i + 1), 3)
            seed = sitk.AntiAliasBinary(seed)
            seed = sitk.BinaryThreshold(seed, 0.5, 1e7)
            if self.debug:
                sitk.WriteImage(sitk.Cast(seed, self._imgType),
                                str(os.path.normpath(
                                    self._output_dir + os.sep +
                                    "seed_{:03d}.nii".format(i + 1))))

            cv = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
            cv.SetNumberOfIterations(iterations)
            cv.UseImageSpacingOn()
            cv.SetHeavisideStepFunction(0)
            cv.SetReinitializationSmoothingWeight(5.0)
            cv.SetEpsilon(upsampling)
            cv.SetCurvatureWeight(curvature)
            cv.SetLambda1(lambda1)
            cv.SetLambda2(lambda2)
            if self.two_dim and dimension == 3:
                stack = []
                size = simg.GetSize()
                for sl in xrange(size[2]):
                    im = sitk.Extract(simg, [size[0], size[1], 0], [0, 0, sl])
                    s = sitk.Extract(seed, [size[0], size[1], 0], [0, 0, sl])
                    phi0 = sitk.SignedMaurerDistanceMap(s,
                                                        insideIsPositive=False,
                                                        squaredDistance=False,
                                                        useImageSpacing=True)
                    stack.append(cv.Execute(phi0, sitk.Cast(im,
                                                            sitk.sitkFloat32)))
                seg = sitk.JoinSeries(stack)
                seg.SetSpacing(simg.GetSpacing())
                seg.SetOrigin(simg.GetOrigin())
                seg.SetDirection(simg.GetDirection())
            else:
                phi0 = sitk.SignedMaurerDistanceMap(seed,
                                                    insideIsPositive=False,
                                                    squaredDistance=False,
                                                    useImageSpacing=True)

                seg = cv.Execute(phi0,
                                 sitk.Cast(simg, sitk.sitkFloat32))
            self.levelsets.append(seg)
            seg = sitk.BinaryThreshold(seg, 1e-7, 1e7)
            #Get connected regions
            if self.opening:
                seg = sitk.BinaryMorphologicalOpening(seg, upsampling)
            r = sitk.ConnectedComponent(seg)
            labelstats = self._getLabelShape(r)
            label = np.argmax(labelstats['volume']) + 1
            seg = (r == label) * (i + 1)
            tmp = sitk.Image(self._img.GetSize(), self._imgType)
            tmp.SetSpacing(self._img.GetSpacing())
            tmp.SetOrigin(self._img.GetOrigin())
            tmp.SetDirection(self._img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tmp)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            tmp = resampler.Execute(seg)
            if self.fillholes:
                tmp = sitk.BinaryFillhole(tmp)
            newcells = sitk.Add(newcells, sitk.Cast(tmp, self._imgType))
            #Handle Overlap
            if self.handle_overlap:
                maxlabel = self._getMinMax(newcells)[1]
                if maxlabel > (i + 1):
                    newcells = self._classifyShared(i, newcells, True)
        self.cells = newcells

    def _classifyShared(self, i, cells, previous):
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
            print(("... ... ... The training data is often insufficient "
                   "for this segmentation method."))
            print(("... ... ... Please consider using Geodesic or "
                   "EdgeFree options.\n"))
            p1 = np.argwhere(a == (i + 1)) * ind2space
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
            clf = svm.SVC(kernel='rbf', degree=3, gamma=2, class_weight='auto')
            clf.fit(scaler.transform(X), y)
            classification = clf.predict(scaler.transform(unknown))
            b[a == l] = classification[0:unknown1.shape[0]]
            b[a == (i + 1)] = classification[unknown1.shape[0]:
                                             unknown1.shape[0] +
                                             unknown2.shape[0]]
            b[a == (l - i - 1)] = classification[unknown1.shape[0] +
                                                 unknown2.shape[0]:]
        cells = sitk.Cast(sitk.GetImageFromArray(b), self._imgType)
        cells.SetSpacing(self._img.GetSpacing())
        cells.SetOrigin(self._img.GetOrigin())
        cells.SetDirection(self._img.GetDirection())
        return cells

    def writeSurfaces(self):
        '''
        if self._img.GetDimension() == 2:
            print(("WARNING: A 2D image was processed, "
                   "so there are no surfaces to write."))
            return
        '''
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
                iso.ComputeNormalsOn()
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

                iso.SetInputData(a.GetOutput())
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
                if self.active == "Nope":
                    smooth.SetNumberOfIterations(200)
                    smooth.Update()
                    self.surfaces.append(smooth.GetOutput())
                else:
                    smooth.SetNumberOfIterations(100)
                    smooth.Update()
                    self.surfaces.append(smooth.GetOutput())
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
        stack = []
        intensities = []
        size = self._img.GetSize()
        intensities = np.zeros(size[2], np.float32)
        for sl in xrange(size[2]):
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
        ratios = fit[1] / (fit[0] * x + fit[1])
        for i, s in enumerate(stack):
            stack[i] = sitk.Cast(s, sitk.sitkFloat32) * ratios[i]
        nimg = sitk.JoinSeries(stack)
        nimg.SetOrigin(self._img.GetOrigin())
        nimg.SetSpacing(self._img.GetSpacing())
        nimg.SetDirection(self._img.GetDirection())
        sitk.Cast(nimg, self._img.GetPixelIDValue())
        self._img = nimg

    def smooth2D(self, img):
        stack = []
        size = img.GetSize()
        for sl in xrange(size[2]):
            s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, sl])
            stack.append(self.smoothRegion(s))
        simg = sitk.JoinSeries(stack)
        simg.SetOrigin(img.GetOrigin())
        simg.SetSpacing(img.GetSpacing())
        simg.SetDirection(img.GetDirection())
        return simg

    def threshold2D(self, img, thres, ratio):
        stack = []
        values = []
        size = img.GetSize()
        if (thres != "PFore" and thres != "PBack"):
            for sl in xrange(size[2]):
                s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, sl])
                seg = thres.Execute(s)
                stack.append(sitk.BinaryFillhole(seg != 0))
                values.append(thres.GetThreshold())
        else:
            for sl in xrange(size[2]):
                s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, sl])
                t = self._getMinMax(s)[1]
                if thres == "PFore":
                    t *= ratio
                    seg = sitk.BinaryThreshold(s, t, 1e7)
                    stack.append(sitk.BinaryFillhole(seg != 0))
                    values.append(t)
                else:
                    t *= (1.0 - ratio)
                    seg = sitk.BinaryThreshold(s, 0, t)
                    if self.fillholes:
                        stack.append(sitk.BinaryFillhole(seg != 0))
                    values.append(t)

        seg = sitk.JoinSeries(stack)
        seg.SetOrigin(img.GetOrigin())
        seg.SetSpacing(img.GetSpacing())
        seg.SetDirection(img.GetDirection())
        return seg, max(values), min(values), values

    def scale2D(self, img, thresh):
        stack = []
        size = img.GetSize()
        maxt = np.max(thresh)
        for i, t in enumerate(thresh):
            s = sitk.Extract(img, [size[0], size[1], 0], [0, 0, i])
            # maximum difference from max of 300%
            if (maxt / t) < 3:
                stack.append(sitk.Cast(s, sitk.sitkFloat32) * (maxt / t))
            else:
                stack.append(sitk.Cast(s, sitk.sitkFloat32))
        nimg = sitk.JoinSeries(stack)
        nimg.SetOrigin(img.GetOrigin())
        nimg.SetSpacing(img.GetSpacing())
        nimg.SetDirection(img.GetDirection())
        return nimg

    def geodesic2D(self, seed, simg,
                   cannyLower, cannyUpper, canny_variance,
                   upsampling, active_iterations, rms,
                   propagation, curvature, advection):
            gd = sitk.GeodesicActiveContourLevelSetImageFilter()
            gd.SetMaximumRMSError(rms / float(upsampling))
            gd.SetNumberOfIterations(active_iterations)
            gd.SetPropagationScaling(propagation)
            gd.SetCurvatureScaling(curvature)
            gd.SetAdvectionScaling(advection)
            stack = []
            size = simg.GetSize()
            for sl in xrange(size[2]):
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
            seg.SetSpacing(simg.GetSpacing())
            seg.SetOrigin(simg.GetOrigin())
            seg.SetDirection(simg.GetDirection())
            return seg
