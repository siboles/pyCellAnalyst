import math

from pyCellAnalyst.Image import Image
import SimpleITK as sitk
import vtk

class SliceInteractorStyle(vtk.vtkInteractorStyleImage):
    """
    Description
    -----------
    Custom vtkInteractorStyleImage class allows for cycling slices using up and down
    arrows. All attributes are private.
    """
    def __init__(self, imageViewer=None, statusMapper=None):
        if imageViewer is None or not isinstance(imageViewer, vtk.vtkImageViewer2):
            raise ValueError('imageViewer must be a vtkImageViewer2 object.')
        self.__imageViewer = imageViewer
        self.__statusMapper = statusMapper
        self.__minslice = self.__imageViewer.GetSliceMin()
        self.__maxslice = self.__imageViewer.GetSliceMax()
        self.__slice = self.__imageViewer.GetSlice()
        self.__activeRenderer = None

    def keyPressEvent(self, obj, event):
        if self.__activeRenderer == self.__imageViewer.GetRenderer():
            self.__key = obj.GetKeySym()
            if self.__key == "Up":
                self.moveSliceForward()
            elif self.__key == "Down":
                self.moveSliceBackward()
            elif self.__key == "BackSpace":
                self.__imageViewer.GetRenderer().GetActiveCamera().SetViewUp(0, 1, 0)
                self.__imageViewer.GetRenderer().ResetCamera()
                self.__imageViewer.Render()

    def getRenderer(self, obj, event):
        curr_pos = obj.GetEventPosition()
        self.__activeRenderer = obj.FindPokedRenderer(*curr_pos)

    def moveSliceForward(self):
        if self.__slice < self.__maxslice:
            self.__slice += 1
            self.__imageViewer.SetSlice(self.__slice)
            self.__statusMapper.SetInput(u'Slice {}/{}'.format(
                self.__imageViewer.GetSlice(), self.__imageViewer.GetSliceMax()))
            self.__imageViewer.Render()

    def moveSliceBackward(self):
        if self.__slice > self.__minslice:
            self.__slice -= 1
            self.__imageViewer.SetSlice(self.__slice)
            self.__statusMapper.SetInput('Slice {}/{}'.format(
                self.__imageViewer.GetSlice(), self.__imageViewer.GetSliceMax()))
            self.__imageViewer.Render()

class SliceViewer(object):
    """
    Description
    -----------
    Visualize the image stack using the vtkImageViewer2 class. Cycle slices with
    up and down arrows. If a list of images is provided the window will be divided
    into multiple viewports. If a layout is specified, viewports will be defined
    based on this. Otherwise, the division will be horizontally.

    Parameters
    ----------
    inputImages : pyCellAnalyst.Image, list(pyCellAnalyst.Image)
        Single or list of pyCellAnalyst.Image objects to render.
    titles : str, list(str), optional
        Titles to assign to each image panel
    """
    def __init__(self, inputImages=None, titles=None, layout=None):
        self.inputImages = inputImages
        for im in self.__inputImages:
            im.convertToVTK()
        self.__vtkImages = [im.vtkimage for im in self.__inputImages]

        self.titles = titles
        self.layout = layout

    @property
    def inputImages(self):
        return self.__inputImages

    @inputImages.setter
    def inputImages(self, inputImages):
        if inputImages is None:
            raise ValueError('inputImages must be provided')
        elif not isinstance(inputImages, list):
            inputImages = [inputImages]
        for d in inputImages:
            if not isinstance(d, Image):
                d = Image(d)
        self.__inputImages = inputImages

    @property
    def titles(self):
        return self.__titles

    @titles.setter
    def titles(self, titles):
        if titles is None:
            titles = ["" for i in self.__inputImages]
        elif not isinstance(titles, list):
            titles = [titles]

        if len(titles) != len(self.__inputImages):
            print(("::WARNING:: provided list of titles is of length {:d}"
                   " while provided inputImages list is of length {:d}.").format(
                       len(titles), len(self.__inputImages)))
            tmp = ["" for i in self.__inputImages]
            titles.extend(tmp[len(titles):])
        self.__titles = [str(t) for t in titles]

    @property
    def layout(self):
        return self.__layout

    @layout.setter
    def layout(self, layout):
        if layout is None:
            layout = (1, len(self.__inputImages))
        elif layout[0]*layout[1] < len(self.__inputImages):
            adjusted = [layout[0], int(math.ceil(len(self.__inputImages) / layout[0]))]
            print(("::WARNING:: provided layout ({},{}) cannot fit all images."
                   " Adjusted to ({},{})").format(layout[0], layout[1], adjusted[0], adjusted[1]))
            layout = adjusted
        self.__layout = (int(layout[0]), int(layout[1]))


    def view(self):
        """
        Description
        -----------
        Renders the SlicerViewer scenes.
        """
        renderWindow = vtk.vtkRenderWindow()
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        xstep = 1.0 / self.__layout[1]
        ystep = 1.0 / self.__layout[0]
        xmin = [i*xstep for i in range(self.layout[1])] * self.__layout[0]
        xmax = [(i+1)*xstep for i in range(self.layout[1])] * self.__layout[0]
        ymin = [i*ystep for i in range(self.__layout[0]) for j in range(self.__layout[1])][::-1]
        ymax = [(i+1)*ystep for i in range(self.__layout[0]) for j in range(self.__layout[1])][::-1]
        for i, im in enumerate(self.__vtkImages):
            intensity_range = (im.GetPointData().GetArray("Intensity").GetValueRange())
            mean_intensity = (intensity_range[0] + intensity_range[1]) / 2.0
            imageViewer = vtk.vtkImageViewer2()
            imageViewer.SetInputData(im)
            imageViewer.SetColorLevel(mean_intensity)
            imageViewer.SetColorWindow(mean_intensity)
            imageViewer.SetSlice((imageViewer.GetSliceMax() - imageViewer.GetSliceMin()) // 2)
            imageViewer.SetRenderWindow(renderWindow)

            sliceTextProp = vtk.vtkTextProperty()
            sliceTextProp.SetFontFamilyToCourier()
            sliceTextProp.SetFontSize(14)
            sliceTextProp.SetVerticalJustificationToBottom()
            sliceTextProp.SetJustificationToLeft()

            sliceTextMapper = vtk.vtkTextMapper()
            sliceTextMapper.SetInput('Slice {:d}/{:d}'.format(imageViewer.GetSlice(), imageViewer.GetSliceMax()))
            sliceTextMapper.SetTextProperty(sliceTextProp)

            sliceTextActor = vtk.vtkActor2D()
            sliceTextActor.SetMapper(sliceTextMapper)
            sliceTextActor.SetPosition(15,10)

            titleTextProp = vtk.vtkTextProperty()
            titleTextProp.SetFontFamilyToCourier()
            titleTextProp.SetFontSize(16)
            titleTextProp.SetVerticalJustificationToTop()
            titleTextProp.SetJustificationToCentered()

            titleTextMapper = vtk.vtkTextMapper()
            titleTextMapper.SetInput(self.__titles[i])
            titleTextMapper.SetTextProperty(titleTextProp)

            titleTextActor = vtk.vtkActor2D()
            titleTextActor.SetMapper(titleTextMapper)
            titleTextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            titleTextActor.SetPosition(0.5,0.95)

            imageViewer.SetupInteractor(renderWindowInteractor)

            myInteractorStyle = SliceInteractorStyle(imageViewer=imageViewer, statusMapper=sliceTextMapper)
            myInteractorStyle.SetInteractionModeToImageSlicing()

            renderWindowInteractor.AddObserver('KeyPressEvent', myInteractorStyle.keyPressEvent)
            renderWindowInteractor.AddObserver('MouseMoveEvent', myInteractorStyle.getRenderer)
            renderWindowInteractor.SetInteractorStyle(myInteractorStyle)

            imageViewer.GetRenderer().AddActor2D(sliceTextActor)
            imageViewer.GetRenderer().AddActor2D(titleTextActor)
            imageViewer.GetRenderer().SetViewport(xmin[i], ymin[i], xmax[i], ymax[i])
            imageViewer.Render()
            imageViewer.GetRenderer().ResetCamera()
            imageViewer.Render()

        renderWindowInteractor.Start()

class viewer3d():
    def __init__(self):
        self.image = None
        self.polydata = None

    def setImage(self):
        pass

    def addPolyData(self, polydata):
        pass
