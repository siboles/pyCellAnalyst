from .Image import Image, EightBitImage, FloatImage
from .RegionsOfInterest import RegionsOfInterest
from .Helpers import FixedDict
import SimpleITK as sitk
import vtk
import numpy as np


def _makeSeed(im, position, radius):
    seedImage = sitk.Image(*im.GetSize(), sitk.sitkUInt8)
    seedImage.CopyInformation(im)
    seedImage[position] = 1
    dm = sitk.SignedMaurerDistanceMap(seedImage, insideIsPositive=False,
                                      squaredDistance=False, useImageSpacing=False)
    seedImage = sitk.BinaryThreshold(dm, -1e7, radius) 
    seedImage = EightBitImage(data=seedImage, spacing=im.GetSpacing())
    return seedImage

class Segmentation(object):
    """
    Description
    -----------
    Base class for image segmentation methods

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
    objectID : int=1, optional
        The label to assign to segmented object.
    """
    def __init__(self, inputImage=None, objectID=1):
        self.inputImage = inputImage
        self.objectID = int(objectID)
        self.parameters = FixedDict({})
        self.isocontour = None

    @property
    def inputImage(self):
        return self._inputImage

    @inputImage.setter
    def inputImage(self, inputImage):
        if not isinstance(inputImage, Image):
            raise AttributeError("inputImage must be a pyCellAnalyst.Image object")
        if inputImage.image is None:
            raise AttributeError("no image data was detected in provided pyCellAnalyst.Image")
        self._inputImage = inputImage

    def getParameters(self):
        """
        Description
        -----------
        Print the current Segmentation parameter values.
        """
        print('Current parameters/values are:')
        for k, v in self.parameters.items():
            print(k, v)

    def _chooseObject(self):
        """
        Chooses object based on distance from the image centroid (weighted by 0.75) and size (weigted by 0.25)
        """
        dim = self.outputImage.image.GetDimension()
        objects = sitk.ConnectedComponent(self.outputImage.image)
        labelstats = sitk.LabelShapeStatisticsImageFilter()
        labelstats.Execute(objects)
        image_centroid = np.array(self.outputImage.image.TransformContinuousIndexToPhysicalPoint(
            [i // 2 for i in self.outputImage.image.GetSize()]))
        max_distance = np.linalg.norm(np.array(self.outputImage.image.GetOrigin()) - image_centroid)
        if labelstats.GetNumberOfLabels() == 0:
            raise RuntimeError("No objects were identified!")
        sizes = np.zeros(labelstats.GetNumberOfLabels(), dtype=float)
        centroids = np.zeros((labelstats.GetNumberOfLabels(), dim), dtype=float)
        for i, l in enumerate(labelstats.GetLabels()):
            centroids[i, :] = labelstats.GetCentroid(l)
            sizes[i] = labelstats.GetNumberOfPixels(l)
        sizes /= sizes.max()
        distances = 1.0 - np.linalg.norm(centroids - image_centroid, axis=1) / max_distance
        probability = 0.75 * distances + 0.25 * sizes
        ind = np.argmax(probability)
        label = labelstats.GetLabels()[ind]
        self.outputImage = EightBitImage((objects == label) * self.objectID, spacing=self.outputImage.spacing)

    def _generateIsoContour(self, baseImage=None):
        tmp = FloatImage(baseImage, spacing=self.inputImage.spacing, normalize=False)
        tmp.convertToVTK()
        iso = vtk.vtkContourFilter()
        iso.SetInputData(tmp.vtkimage)
        iso.ComputeScalarsOff()
        iso.ComputeNormalsOn()
        iso.SetValue(0, self.isovalue)
        iso.Update()

        triangles = vtk.vtkGeometryFilter()
        triangles.SetInputConnection(iso.GetOutputPort())
        triangles.Update()

        deci = vtk.vtkDecimatePro()
        deci.SetInputConnection(triangles.GetOutputPort())
        deci.PreserveTopologyOn()
        deci.SplittingOff()
        deci.SetTargetReduction(0.1)


        if self.outputImage.image.GetDimension() == 3:
            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputConnection(deci.GetOutputPort())
            smooth.NormalizeCoordinatesOn()
            smooth.SetNumberOfIterations(20)
            smooth.SetPassBand(0.01)
            smooth.FeatureEdgeSmoothingOff()
            smooth.Update()

            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.ComputePointNormalsOn()
            normal_generator.SetInputConnection(smooth.GetOutputPort())
            normal_generator.Update()
            self.isocontour = normal_generator.GetOutput()
        else:
            self.isocontour = triangles.GetOutput()

class Threshold(Segmentation):
    """
    Description
    -----------
    Base class for thresholding methods.
    """
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID)
        self.mask = mask
        self._thresholdMethod = None
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None:
            pass
        elif isinstance(mask, EightBitImage):
            pass
        else:
            raise ValueError("Mask must be a pyCellAnalyst.EightBitImage")
        self._mask = mask

    def execute(self):
        if self._thresholdMethod:
            if self._mask:
                self._thresholdMethod.SetMaskValue(1)
                self.outputImage = EightBitImage(self._thresholdMethod.Execute(
                    self._inputImage.image, self._mask.image > 0) * self.objectID,
                                                 spacing=self._inputImage.spacing)
            else:
                self.outputImage = EightBitImage(self._thresholdMethod.Execute(
                    self._inputImage.image) * self.objectID, spacing=self._inputImage.spacing)
            if isinstance(self._thresholdMethod, sitk.BinaryThresholdImageFilter):
                self.isovalue = self._thresholdMethod.GetLowerThreshold()
            else:
                self.isovalue = self._thresholdMethod.GetThreshold()
            self._chooseObject()
            baseImage = sitk.Mask(self._inputImage.image, self.outputImage.image > 0)
            self._generateIsoContour(baseImage=baseImage)
        else:
            raise AttributeError("A pyCellAnalyst.Threshold subclass must be instantiated before calling execute()")

class Otsu(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.OtsuThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class Huang(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.HuangThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class RidlerCalvard(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.IsoDataThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class MaxEntropy(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.MaximumEntropyThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class Moments(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.MomentsThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class Yen(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.YenThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class RenyiEntropy(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.RenyiEntropyThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class Shanbhag(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self._thresholdMethod = sitk.ShanbhagThresholdImageFilter()
        self._thresholdMethod.SetInsideValue(0)
        self._thresholdMethod.SetOutsideValue(1)

class Percentage(Threshold):
    def __init__(self, inputImage=None, objectID=1, mask=None, percentage=40.0):
        super().__init__(inputImage=inputImage, objectID=objectID, mask=mask)
        self.percentage = percentage
        self._thresholdMethod = sitk.BinaryThresholdImageFilter()
        self._thresholdMethod.SetLowerThreshold(self.percentage / 100.0)
        self._thresholdMethod.SetUpperThreshold(1.1)

class GeodesicActiveContour(Segmentation):
    """
    Description
    -----------
    Performs a segmentation using the SimpleITK implementation of the
    Geodesic Active Contour Levelset Segmentation method described in
    (Caselles et al. 1997.) Please also consult SimpleITK's documentation
    of GeodesicActiveContourLevelSetImageFilter. The initial levelset function
    is determined from the seed parameter.

    Parameters
    ----------
    inputImage : pyCellAnalyst.FloatImage
        The image to segment. It is highly recommended that this be filtered since the
        only smoothing aspects of this method is through **curvatureScaling**.
    objectID : int=1, optional
        The integer label to assign to segmented object.
    seed : default=None, pyCellAnalyst.EightBitImage, [int=x, int=y, int=z, int=radius], optional
        If None seed will be a sphere in center of image.
        If list spherical seed will be constructed from this. Length must be 3 for 2D image; 4 for 3D.
        If pyCellAnalyst.EightBitImage (most likely from a previous segmentation method) this will be the seed.
        This will typically be the ideal seed.
    propagationScaling : float=1.0, optional
        Weight for propagation term in active contour functional.
        Higher values result in faster expansion.
    curvatureScaling : float=0.7, optional
        Weight for curvature term in active contour functional.
        Higher values result in smoother segmentation.
    advectionScaling : float=1.0, optional
        Weight for advective term in active contour functional.
        Higher values causes the levelset evolution to be drawn
        and stick to edges.
    maximumRMSError : float=0.01, optional
        The change in root-mean-square difference  at which iterations
        will terminate.
    numberOfIterations : int=1000, optional
        The maximum number of iterations the active contour will conduct.
    """
    def __init__(self, inputImage=None, objectID=1, seed=None,
                 propagationScaling=1.0, curvatureScaling=0.7, advectionScaling=1.0,
                 maximumRMSError=0.01, numberOfIterations=1000):
        super().__init__(inputImage=inputImage, objectID=objectID)
        self.seed = seed
        self.parameters = FixedDict({'propagationScaling': float(propagationScaling),
                                     'curvatureScaling': float(curvatureScaling),
                                     'advectionScaling': float(advectionScaling),
                                     'maximumRMSError': float(maximumRMSError),
                                     'numberOfIterations': int(numberOfIterations)})

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        if seed is None:
            print('::WARNING:: No seed image was provided for {}. Assuming sphere at image center with a radius 1/4 of the smallest image edge length.'.format(self.__class__.__name__))
            seed = _makeSeed(self._inputImage.image,
                             [i // 2 for i in self._inputImage.image.GetSize()],
                             min([i / 4.0 for i in self._inputImage.image.GetSize()]))
        elif isinstance(seed, list):
            if self.__inputImage.image.GetDimension() != len(seed):
                raise ValueError('If seed is provided as a list, it must include a pixel coordinate for each image dimension and a radius.')
            seed = _makeSeed(self._inputImage.image,
                             seed[0:self._inputImage.image.GetDimension()],
                             seed[self._inputImage.image.GetDimension()])
        elif isinstance(seed, EightBitImage):
            pass
        else:
            raise ValueError('Unsupported {} provided for seed parameter.'.format(type(seed)))

        self.__seed = seed

    def execute(self):
        d = sitk.SignedMaurerDistanceMap(self.__seed.image, insideIsPositive=False, squaredDistance=False,
                                         useImageSpacing=True)
        #grad = sitk.GradientMagnitude(self._inputImage.image)
        #grad = sitk.RescaleIntensity(sitk.Cast(sitk.BoundedReciprocal(grad), sitk.sitkFloat32), 0.0, 1.0)
        canny = sitk.CannyEdgeDetection(self._inputImage.image, 0.01, 0.1)
        speed = sitk.SignedMaurerDistanceMap(sitk.Cast(canny, sitk.sitkUInt8), squaredDistance=True, useImageSpacing=True)
        gd = sitk.GeodesicActiveContourLevelSetImageFilter()
        gd.SetPropagationScaling(self.parameters['propagationScaling'])
        gd.SetCurvatureScaling(self.parameters['curvatureScaling'])
        gd.SetAdvectionScaling(self.parameters['advectionScaling'])
        gd.SetMaximumRMSError(self.parameters['maximumRMSError'])
        gd.SetNumberOfIterations(self.parameters['numberOfIterations'])
        ls = gd.Execute(d, speed)
        print("... Geodesic Active Contour Segmentation Completed")
        print(("... ... Elapsed Iterations: {:d}"
                .format(gd.GetElapsedIterations())))
        print(("... ... Change in RMS Error: {:.3e}"
                .format(gd.GetRMSChange())))
        b = sitk.BinaryThreshold(ls, -1e7, min(self._inputImage.spacing) / 2.0)
        self.outputImage = EightBitImage(b*self.objectID, spacing=self._inputImage.image.GetSpacing())
        self._chooseObject()
        self.edgePotential = Image(speed, spacing=self._inputImage.spacing)
        self.isovalue = 1e-7
        self._generateIsoContour(baseImage=ls)

