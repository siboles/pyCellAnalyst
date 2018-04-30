import os

from .Image import Image
import numpy as np
import SimpleITK as sitk
from openpyxl import load_workbook

class RegionsOfInterest(object):
    """
    Description
    -----------
    Define images of regions of interest within a larger image. Each new image
    retains its location in the larger image.

    Parameters
    ----------
    inputImage : pyCellAnalyst.Image
        The image to consider regions of interest within
    regions_of_interest : .xlsx file, [[int,..]], ndarray
        The origin and size in pixels of the regions of interest
    order : 'ImageJ', 'OriginSize', optional
        How regions are defined in the supplied .xlsx file
    start_row : int=1, optional
        Row in file where region definitions begin
    start_col : int=1, optional
        Column in file where region defintions begin

    Attributes
    -----------
    images : [pyCellAnalyst.Image,...]
       List of new images for each region of interest
    """
    def __init__(self, inputImage=None, regions_of_interest=None, order='ImageJ', start_row=1, start_col=1):
        self.order = order
        self.start_row = start_row
        self.start_col = start_col
        self.inputImage = inputImage
        self.regions_of_interest = regions_of_interest

    @property
    def inputImage(self):
        return self.__inputImage

    @inputImage.setter
    def inputImage(self, inputImage):
        if not isinstance(inputImage, Image):
            raise TypeError('inputImage must be a pyCellAnalyst.Image object.')
        self.__inputImage = inputImage

    @property
    def regions_of_interest(self):
        return self.__regions_of_interest

    @regions_of_interest.setter
    def regions_of_interest(self, regions_of_interest):
        if regions_of_interest is None:
            print("::WARNING:: regions_of_interest unspecified. Assuming whole image.")
            regions_of_interest = np.array([[0,0,0]+[i for i in self.inputImage.image.GetSize()]]).astype(int)
        elif isinstance(regions_of_interest, list):
            regions_of_interest = np.array(regions_of_interest)
        elif isinstance(regions_of_interest, np.ndarray):
            pass
        elif os.path.isfile(regions_of_interest):
            regions_of_interest = self.__parseExcelFile(regions_of_interest)
        else:
            raise TypeError('regions_of_interest must be a file, list, or numpy array.')

        if len(regions_of_interest.shape) != 2:
            raise ValueError('regions_of_interest should be an array of order 2.')

        if len(regions_of_interest[0,:]) != 2*self.__inputImage.image.GetDimension():
            raise ValueError('Each row of regions_of_interest should have a length of 2 times the image dimension.')

        self.__regions_of_interest = regions_of_interest
        self.__getImages()

    def __parseExcelFile(self, filename):
        if self.order not in ('ImageJ', 'OriginSize'):
            raise AttributeError('order must be either ImageJ or OriginSize')

        wb = load_workbook(filename)
        sheets = wb.get_sheet_names()
        if len(sheets) > 1:
            print('::WARNING:: Multiple sheets in region_of_interest file. Assuming first sheet contains information.')

        ws = wb.get_sheet_by_name(sheets[0])
        values = np.array(list(ws.values))[self.start_row:,self.start_col:].astype(int)
        if self.order == 'ImageJ':
            if values.shape[1] == 5:
                if values.shape[0] % 2 != 0:
                    raise ValueError('An odd number of rows were read from regions_of_interest file and ImageJ was specified for order. Something is wrong!')
                regions = np.zeros((values.shape[0] // 2, 6))
                cnt = 0
                for i in range(values.shape[0]):
                    if i % 2 == 1:
                        regions[cnt,[0,1,3,4]] = values[i,[0,1,2,3]]
                        regions[cnt,2] = values[i-1,4]
                        regions[cnt,5] = values[i,4] - values[i-1,4]
                        cnt += 1
        else:
            regions = values
        return regions.astype(int)

    def __getImages(self):
        dim = self.inputImage.image.GetDimension()
        self.images = []
        for i in range(self.regions_of_interest.shape[0]):
            size = [int(ind) for ind in self.regions_of_interest[i,dim:]]
            origin = [int(ind) for ind in self.regions_of_interest[i,0:dim]]
            self.images.append(self.__inputImage.__class__(sitk.RegionOfInterest(self.inputImage.image,
                                                           size,
                                                           origin),
                                                           spacing=self.inputImage.image.GetSpacing()))


