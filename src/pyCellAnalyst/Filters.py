from copy import deepcopy

import SimpleITK as sitk
from .Image import FloatImage
from .Helpers import FixedDict

class Filter(object):
    """
    Description
    -----------
    Base class for image filters

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
    """
    def __init__(self, inputImage=None):
        self._inputImage = inputImage
        self.parameters = FixedDict({})

    @property
    def inputImage(self):
        return self._inputImage

    @inputImage.setter
    def inputImage(self, inputImage):
        if not isinstance(inputImage, FloatImage):
            raise AttributeError("inputImage must be a pyCellAnalyst.FloatImage object")
        if inputImage.image is None:
            raise AttributeError("no image data was detected in provided pyCellAnalyst.Float32Image")
        self._inputImage = inputImage

    def getParameters(self):
        """
        Description
        -----------
        Print the current filter parameter values.
        """
        print('Current filter parameters/values are:')
        for k, v in self.parameters.items():
            print(k, v)


class Gaussian(Filter):
    """
    Description
    -----------
    Gaussian smoothing via discrete convolution.

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method
    variance : float=0.5, optional

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image. Generated when **execute()** is called.
    """
    def __init__(self, inputImage=None, variance=0.5):
        super().__init__(inputImage)
        self.parameters = FixedDict({'variance': 0.5})

    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        self.outputImage = FloatImage(sitk.DiscreteGaussian(self._inputImage.image, **self.parameters),
                                      spacing=self._inputImage.spacing)

class Median(Filter):
    """
    Description
    -----------
    Median image filter.

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method.
    radius : (int=1, int=1, int=1), optional
        The radius of the median kernel.

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image.
    """
    def __init__(self, inputImage=None, radius=[1,1,1]):
        super().__init__(inputImage)
        self.parameters = FixedDict({'radius': [1, 1, 1]})

    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        self.outputImage = FloatImage(sitk.Median(self._inputImage.image, **self.parameters),
                                      spacing=self._inputImage.spacing)

class CurvatureAnisotropicDiffusion(Filter):
    """
    Description
    -----------
    Curvature based anisotropic diffusion image filter. This filter is edge-preserving
    and can also enhance edges. Performs better than gradient anisotropic diffusion filter
    when gradient between edge and background is weak.

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method.
    iterations : int=20, optional
    conductance : float=9.0, optional

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image. Generated when **execute()** is called.
    """
    def __init__(self, inputImage=None, iterations=20, conductance=9):
        super().__init__(inputImage)
        self.parameters = FixedDict({'iterations': iterations,
                                     'conductance': conductance})

    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        f = sitk.CurvatureAnisotropicDiffusionImageFilter()
        f.EstimateOptimalTimeStep(self._inputImage.image)
        f.SetNumberOfIterations(self.parameters['iterations'])
        f.SetConductanceParameter(self.parameters['conductance'])
        self.outputImage = FloatImage(f.Execute(self._inputImage.image),
                                      self._inputImage.spacing)

class GradientAnisotropicDiffusion(Filter):
    """
    Description
    -----------
    Gradient based anisotropic diffusion image filter. This filter is edge-preserving
    and can also enhance edges.

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method.
    iterations : int=20, optional
    conductance : float=9.0, optional
    time_step : float=0.01, optional

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image. Generated when **execute()** is called.
    """
    def __init__(self, inputImage=None, iterations=20, conductance=9, time_step=0.01):
        super().__init__(inputImage)
        self.parameters = FixedDict({'iterations': iterations,
                                     'conductance': conductance,
                                     'time_step': time_step})

    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        f = sitk.GradientAnisotropicDiffusionImageFilter()
        f.SetNumberOfIterations(self.parameters['iterations'])
        f.SetConductanceParameter(self.parameters['conductance'])
        f.SetTimeStep(self.parameters['time_step'])
        self.outputImage = FloatImage(f.Execute(self._inputImage.image),
                                      spacing=self._inputImage.spacing)

class Bilateral(Filter):
    """
    Description
    -----------

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method.
    domainSigma : float=0.5, optional
    rangeSigma : float=5.0, optional
    numberOfRangeGaussianSamples : int=100, optional

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image. Generated when **execute()** is called.
    """
    def __init__(self, inputImage=None, domainSigma=0.5, rangeSigma=5.0, numberOfRangeGaussianSamples=100):
        super().__init__(inputImage)
        self.parameters = FixedDict({'domainSigma': domainSigma,
                                     'rangeSigma': rangeSigma,
                                     'numberOfRangeGaussianSamples': numberOfRangeGaussianSamples})


    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        self.outputImage = FloatImage(sitk.Bilateral(self._inputImage.image, **self.parameters),
                                      self._inputImage.spacing)

class PatchBasedDenoising(Filter):
    """
    Description
    -----------

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method.
    radius : int=4, optional
    iterations : int=10, optional
    patches : int=10, optional
    noise_model : 'poisson', optional

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image. Generated when **execute()** is called.
    """
    def __init__(self, inputImage=None, radius=4, iterations=10, patches=10, noise_model='poisson'):
        super().__init__(inputImage)
        self.parameters = FixedDict({'radius': radius,
                                     'iterations': iterations,
                                     'patches': patches,
                                     'noise_model': noise_model})
        self.__noise_models = {'nomodel': 0,
                               'gaussian': 1,
                               'rician': 2,
                               'poisson': 3}
        try:
            self.__noise_models[noise_model]
        except:
            msg = 'Acceptable noise models are:'
            msg += '\n'.join('{:s}'.format(m) for m in self.__noise_models)
            raise ValueError(msg)

    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        oldspacing = deepcopy(self._inputImage.spacing)
        self._inputImage.spacing = [1.0]*len(oldspacing)
        f = sitk.PatchBasedDenoisingImageFilter()
        f.SetNoiseModel(self.__noise_models[self.parameters['noise_model']])
        f.SetNoiseModelFidelityWeight(1.0)
        f.SetNumberOfSamplePatches(self.parameters['patches'])
        f.SetPatchRadius(self.parameters['radius'])
        f.SetNumberOfIterations(self.parameters['iterations'])
        self.outputImage = FloatImage(f.Execute(self._inputImage.image), spacing=oldspacing)
        self._inputImage.spacing = oldspacing

class EnhanceEdges(Filter):
    """
    Description
    -----------
    Sharpens the image where Laplacian magnitude is high indicating a rapid intensity change.
    This will enhance the edges, but can also undesirably highlight other features. This is
    best used following a less feature preserving filter e.g. **Bilateral**.

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        Must be set prior to calling **execute()** method.

    Attributes
    ----------
    outputImage : pyCellAnalyst.Image
        The filtered image. Generated when **execute()** is called.
    """
    def __init__(self, inputImage=None):
        super().__init__(inputImage)

    def execute(self):
        if self._inputImage is None:
            raise AttributeError("Before executing the filter, you must define the inputImage.")
        self.outputImage = FloatImage(sitk.LaplacianSharpening(self._inputImage.image),
                                      spacing=self._inputImage.spacing)
