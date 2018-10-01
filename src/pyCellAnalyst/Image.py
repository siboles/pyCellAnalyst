import os
import re
import fnmatch

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk

class Image(object):
    def __init__(self, data=None, spacing=None):
        self.__supportedImageTypes = (".tif",
                                      ".tiff",
                                      ".png",
                                      ".jpg",
                                      ".jpeg",
                                      ".nii")
        if data is None:
            raise ValueError('data must be a file, directory, SimpleITK image, or ndarray')
        elif isinstance(data, sitk.Image):
            self.image = data
        elif isinstance(data, np.ndarray):
            self.image = sitk.GetImageFromArray(data)
        elif os.path.isfile(data):
            self._parseImageFile(data)
        elif os.path.isdir(data):
            self._parseImageSequence(data)

        self.spacing = spacing
        self.vtkimage = None

    @property
    def spacing(self):
        return self.__spacing

    @spacing.setter
    def spacing(self, spacing):
        if spacing is None:
            print(':::WARNING::: Image spacing was not specified.')
            self.__spacing = self.image.GetSpacing()
            self.image.SetSpacing(self.__spacing)
        else:
            self.__spacing = spacing
            self.image.SetSpacing(self.__spacing)

    def _parseImageFile(self, p):
        filename, file_extension = os.path.splitext(p)
        if file_extension.lower() in self.__supportedImageTypes:
            self.image = sitk.ReadImage(p, sitk.sitkFloat32)
            print('Imported image {:s}'.format(p))
        else:
            raise ValueError('Unsupported file type with extension, {:s}, detected.'.format(file_extension)+
                             '\n'.join('{:s}'.format(t) for t in self.__supportedImageTypes))

    def _parseImageSequence(self, p):
        files = sorted(os.listdir(p))
        file_extensions = [os.path.splitext(f)[1].lower() for f in files]
        ftypes = []
        for t in self.__supportedImageTypes:
            if t in file_extensions:
                ftypes.append(t)
        if len(ftypes) > 1:
            raise RuntimeError('The following file types were detected in {:s}:'.format(p)+
                               '\n'.join('{:s}'.format(t) for t in ftypes)+
                               '\nPlease only include files of one image type.')
        elif len(ftypes) == 0:
            raise RuntimeError('No supported files were found in {:s}'.format(p))

        files = fnmatch.filter(files, '*{:s}'.format(ftypes[0]))

        if len(files) > 1:
            counter = [re.search('[0-9]*\{:s}'.format(ftypes[0]), f).group() for f in files]
            for i, c in enumerate(counter):
                counter[i] = int(c.replace('{:s}'.format(ftypes[0]), ''))
            files = np.array(files, dtype=object)
            sorter = np.argsort(counter)
            files = files[sorter]
            img = [sitk.ReadImage(os.path.join(p, f), sitk.sitkFloat32) for f in files]
            img = sitk.JoinSeries(img)
            print('\nImported 3D image stack ranging from {:s} to {:s}'.format(files[0], files[-1]))
        else:
            img = sitk.ReadImage(os.path.join(p, files[0]), sitk.sitkFloat32)
            print('\nImported 2D image {:s}'.format(files[0]))
        self.image = img

    def getPixelRange(self):
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(self.image)
        return minmax.GetMinimum(), minmax.GetMaximum()

    def invert(self):
        self.image = sitk.InvertIntensity(self.image, self.getPixelRange()[1])

    def resample(self, spacing=None):
        try:
            spacing = [float(spacing)] * len(self.spacing)
        except:
            if spacing is None:
                print('::WARNING:: No spacing for resampling was indicated. Making isotropic to smallest edge length.')
                spacing = [min(self.spacing)] * len(self.spacing)
            elif isinstance(spacing, list):
                if len(spacing) != len(self.spacing):
                    raise ValueError('spacing must have length equal to image dimension.')
            else:
                raise ValueError('Unsupported {} type provided for spacing.'.format(type(spacing)))
        newsize = (np.array(self.image.GetSize()) * np.array(self.image.GetSpacing()) / np.array(spacing)).astype(int)
        newsize = newsize.tolist()
        rs = sitk.ResampleImageFilter()
        if self.image.GetPixelID() in [1, 3]:
            rs.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            rs.SetInterpolator(sitk.sitkBSpline)
        rs.SetSize(newsize)
        rs.SetOutputOrigin(self.image.GetOrigin())
        print('Resampling image to: ', spacing)
        rs.SetOutputSpacing(spacing)
        self.image = rs.Execute(self.image)
        self.spacing = spacing

    def convertToNumpy(self):
        self.numpyimage = sitk.GetArrayFromImage(self.image)

    def convertToVTK(self):
        origin = list(self.image.GetOrigin())
        spacing = list(self.image.GetSpacing())
        dimensions= list(self.image.GetSize())
        if self.image.GetDimension() == 2:
            origin += [0.0]
            spacing += [1.0]
            dimensions += [1]

        vtkimage = vtk.vtkImageData()
        vtkimage.SetOrigin(origin)
        vtkimage.SetSpacing(spacing)
        vtkimage.SetDimensions(dimensions)

        pixel_type = {1: vtk.VTK_UNSIGNED_CHAR,
                      3: vtk.VTK_UNSIGNED_INT,
                      8: vtk.VTK_FLOAT,
                      9: vtk.VTK_DOUBLE}

        intensities = numpy_to_vtk(sitk.GetArrayFromImage(self.image).ravel(), deep=True,
                                   array_type=pixel_type[self.image.GetPixelID()])
        intensities.SetName("Intensity")
        intensities.SetNumberOfComponents(1)

        vtkimage.GetPointData().SetScalars(intensities)
        self.vtkimage = vtkimage

    def writeAsNifti(self, name=None):
        if name is None:
            name = 'output'
        sitk.WriteImage(self.image, '{:s}.nii'.format(name))

    def writeAsVTK(self, name=None):
        if name is None:
            name = 'output'

        if self.vtkimage is None:
            self.convertToVTK()
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName('{:s}.vti'.format(name))
        writer.SetInputData(self.vtkimage)
        writer.Update()
        writer.Write()
        print('... Wrote image to {:s}.vti'.format(name))

class FloatImage(Image):
    def __init__(self, data=None, spacing=None, normalize=True):
        super().__init__(data=data, spacing=spacing)
        self.image = sitk.Cast(self.image, sitk.sitkFloat32)
        if normalize:
            self.image = sitk.RescaleIntensity(self.image, 0.0, 1.0)

class DoubleImage(Image):
    def __init__(self, data=None, spacing=None, normalize=True):
        super().__init__(data=data, spacing=spacing)
        self.image = sitk.Cast(self.image, sitk.sitkFloat64)
        if normalize:
            self.image = sitk.RescaleIntensity(self.image, 0.0, 1.0)

class EightBitImage(Image):
    def __init__(self, data=None, spacing=None):
        super().__init__(data=data, spacing=spacing)
        self.image = sitk.Cast(self.image, sitk.sitkUInt8)

class SixteenBitImage(Image):
    def __init__(self, data=None, spacing=None):
        super().__init__(data=data, spacing=spacing)
        self.image = sitk.Cast(self.image, sitk.sitkUInt16)


