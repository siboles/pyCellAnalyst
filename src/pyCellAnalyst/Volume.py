import os, re, string, warnings, platform, vtk, fnmatch, shutil
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import numpy as np
from scipy import stats


class Volume(object):
    def __init__(self,vol_dir,pixel_dim=[0.411,0.411,0.6835],tratio=0.4,stain='cell',display=True,left=[0,1e64,0],right=[1e64,0,1e64],counter=0):
        #first check what OS we are running on
        op_sys = platform.platform()
        if 'Windows' in op_sys:
            self._path_dlm = '\\'
        elif 'Linux' in op_sys:
            self._path_dlm = '/'
        else:
            print('WARNING: This module is untested on MAC.')
            self._path_dlm = '/'
        warnings.filterwarnings("ignore")

        self._vol_dir = vol_dir
        self._tratio = tratio
        self._pixel_dim = pixel_dim
        self._stain = stain
        self._display = display
        self._left = map(int,left)
        self._right = map(int,right)
        self._counter = int(counter)
        self.cells = []
        if self._stain == 'cell':
            self._t = 0
        else:
            self._t = 255

        self.volumes = [] #The volumes for each cell in the field of view
        self.points = [] #The vertices for each cell in the field of view
        self.dimensions = []
        # read in the TIFF stack
        self._parseStack()
        self._getThresh()
        self._writeThresh()
        self._makeSTL()
        self._getDimensions()


    def _parseStack(self):
        self._stack = []
        cnt = 0
        for fname in sorted(os.listdir(self._vol_dir)):
            if '.tif' in fname.lower():
                img = ImageOps.grayscale(Image.open(self._vol_dir+self._path_dlm+fname)).filter(ImageFilter.MedianFilter(5))
                if self._left[1] == 1e64:
                    self._left[1] = img.size[0]
                if self._right[0] == 1e64:
                    self._right[0] = img.size[1]

                if cnt >= self._left[2] and cnt <= self._right[2]:
                    imsize = img.size
                    p = np.array(list(img.getdata()),int).reshape(imsize)
                    if self._left[0] > 0:
                        p[:,0:self._left[0]] = self._t
                    if self._right[0] < imsize[1]:
                        p[:,self._right[0]:] = self._t
                    if self._left[1] < imsize[0]:
                        p[self._left[1]:,:] = self._t
                    if self._right[1] > 0:
                        p[0:self._right[1],:] = self._t
                    p = list(p.flatten())
                    img.putdata(p)
                else:
                    table = [self._t]*256
                    img = img.point(table)
                self._stack.append(img)
                cnt += 1

    def _getThresh(self):
        self._thresh = []
        for i in self._stack:
            self._thresh.append(int(round(self._tratio*i.getextrema()[1]+0.5)))
        self._thresh = np.array(self._thresh,int)


    def _writeThresh(self):
        self._gray_dir = self._vol_dir+'_gray'
        try:
            os.mkdir(self._gray_dir)
        except:
            pass

        cnt = 0
        for i in xrange(len(self._thresh)):
            img = self._stack[i]
            img.save(self._gray_dir+self._path_dlm+'cell%04d.tif' % cnt,'TIFF')
            cnt += 1
            

    def _makeSTL(self):
        local_dir = self._gray_dir
        surface_dir = self._vol_dir+'_surfaces'+self._path_dlm
        try:
            os.mkdir(surface_dir)
        except:
            pass
        files = fnmatch.filter(sorted(os.listdir(local_dir)),'*.tif')
        counter = re.search("[0-9]*\.tif", files[0]).group()
        prefix = self._path_dlm+string.replace(files[0],counter,'')
        counter = str(len(counter)-4)
        prefixImageName = local_dir + prefix

        ### Create the renderer, the render window, and the interactor. The renderer
        # The following reader is used to read a series of 2D slices (images)
        # that compose the volume. The slice dimensions are set, and the
        # pixel spacing. The data Endianness must also be specified. The reader
        v16=vtk.vtkTIFFReader()

        v16.SetFilePrefix(prefixImageName)
        v16.SetDataExtent(0,100,0,100,1,len(files))
        v16.SetFilePattern("%s%0"+counter+"d.tif")
        v16.Update()

        im = v16.GetOutput()
        im.SetSpacing(self._pixel_dim[0],self._pixel_dim[1],self._pixel_dim[2])

        smooth=vtk.vtkImageGaussianSmooth()
        smooth.SetDimensionality(3)
        smooth.SetStandardDeviation(1.2,1.2,1.2)
        smooth.SetRadiusFactors(2,2,2)
        smooth.SetInputData(im)

        cutoff = stats.mstats.mquantiles(self._thresh[self._thresh>1],[0.95])[0] #take the 95th quantile as the cutoff
        thres=vtk.vtkImageThreshold()
        thres.SetInputConnection(smooth.GetOutputPort())
        thres.ThresholdByLower(0)
        thres.ThresholdByUpper(cutoff)

        iso=vtk.vtkImageMarchingCubes()
        iso.SetInputConnection(thres.GetOutputPort())
        iso.SetValue(0,cutoff*1.05)

        regions = vtk.vtkConnectivityFilter()
        regions.SetInputConnection(iso.GetOutputPort())
        regions.SetExtractionModeToAllRegions()
        regions.ColorRegionsOn()
        regions.Update()

        N = regions.GetNumberOfExtractedRegions()
        for i in xrange(N):
            r = vtk.vtkConnectivityFilter()
            r.SetInputConnection(iso.GetOutputPort())
            r.SetExtractionModeToSpecifiedRegions()
            r.AddSpecifiedRegion(i)
            g = vtk.vtkExtractUnstructuredGrid()
            g.SetInputConnection(r.GetOutputPort())
            geo = vtk.vtkGeometryFilter()
            geo.SetInputConnection(g.GetOutputPort())
            geo.Update()
            t = vtk.vtkTriangleFilter()
            t.SetInputConnection(geo.GetOutputPort())
            t.Update()
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(t.GetOutputPort())
            s = vtk.vtkSmoothPolyDataFilter()
            s.SetInputConnection(cleaner.GetOutputPort())
            s.SetNumberOfIterations(50)
            dl = vtk.vtkDelaunay3D()
            dl.SetInputConnection(s.GetOutputPort())
            dl.Update()

            self.cells.append(dl)

        for i in xrange(N):
            g = vtk.vtkGeometryFilter()
            g.SetInputConnection(self.cells[i].GetOutputPort())
            t = vtk.vtkTriangleFilter()
            t.SetInputConnection(g.GetOutputPort())

            #get the surface points of the cells and save to points attribute
            v = t.GetOutput()
            points = []
            for j in xrange(v.GetNumberOfPoints()):
                p = [0,0,0]
                v.GetPoint(j,p)
                points.append(p)
            self.points.append(points)

            #get the volume of the cell
            vo = vtk.vtkMassProperties()
            vo.SetInputConnection(t.GetOutputPort())
            self.volumes.append(vo.GetVolume())

            stl = vtk.vtkSTLWriter()
            stl.SetInputConnection(t.GetOutputPort())
            stl.SetFileName(surface_dir+'cell%02d.stl' % (i+self._counter))
            stl.Write()

        if self._display:
            skinMapper = vtk.vtkDataSetMapper()
            skinMapper.SetInputConnection(regions.GetOutputPort())
            skinMapper.SetScalarRange(regions.GetOutput().GetPointData().GetArray("RegionId").GetRange())
            skinMapper.SetColorModeToMapScalars()
            #skinMapper.ScalarVisibilityOff()
            skinMapper.Update()

            skin = vtk.vtkActor()
            skin.SetMapper(skinMapper)
            #skin.GetProperty().SetColor(0,0,255)

            # An outline provides context around the data.
            #
            outlineData = vtk.vtkOutlineFilter()
            outlineData.SetInputConnection(v16.GetOutputPort())

            mapOutline = vtk.vtkPolyDataMapper()
            mapOutline.SetInputConnection(outlineData.GetOutputPort())

            outline = vtk.vtkActor()
            #outline.SetMapper(mapOutline)
            #outline.GetProperty().SetColor(0,0,0)

            colorbar = vtk.vtkScalarBarActor()
            colorbar.SetLookupTable(skinMapper.GetLookupTable())
            colorbar.SetTitle("Cells")
            colorbar.SetNumberOfLabels(N)


            # Create the renderer, the render window, and the interactor. The renderer
            # draws into the render window, the interactor enables mouse- and 
            # keyboard-based interaction with the data within the render window.
            #
            aRenderer = vtk.vtkRenderer()
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(aRenderer)
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)

            # It is convenient to create an initial view of the data. The FocalPoint
            # and Position form a vector direction. Later on (ResetCamera() method)
            # this vector is used to position the camera to look at the data in
            # this direction.
            aCamera = vtk.vtkCamera()
            aCamera.SetViewUp (0, 0, -1)
            aCamera.SetPosition (0, 1, 0)
            aCamera.SetFocalPoint (0, 0, 0)
            aCamera.ComputeViewPlaneNormal()

            # Actors are added to the renderer. An initial camera view is created.
            # The Dolly() method moves the camera towards the FocalPoint,
            # thereby enlarging the image.
            aRenderer.AddActor(outline)
            aRenderer.AddActor(skin)
            aRenderer.AddActor(colorbar)
            aRenderer.SetActiveCamera(aCamera)
            aRenderer.ResetCamera ()
            aCamera.Dolly(1.5)

            # Set a background color for the renderer and set the size of the
            # render window (expressed in pixels).
            aRenderer.SetBackground(0.0,0.0,0.0)
            renWin.SetSize(800, 600)

            # Note that when camera movement occurs (as it does in the Dolly()
            # method), the clipping planes often need adjusting. Clipping planes
            # consist of two planes: near and far along the view direction. The 
            # near plane clips out objects in front of the plane the far plane
            # clips out objects behind the plane. This way only what is drawn
            # between the planes is actually rendered.
            aRenderer.ResetCameraClippingRange()

            im=vtk.vtkWindowToImageFilter()
            im.SetInput(renWin)

            iren.Initialize();
            iren.Start();

        #remove gray directory
        shutil.rmtree(local_dir)

    def _getDimensions(self):
        '''
        #Using Principal Component Analysis
        for p in self.points:
            x = np.array(p)

            [lam,vec] = np.linalg.eig(np.cov(x.T))
            max_loc = np.zeros((3,1),int)
            for i in xrange(3):
                max_loc[i,0] = np.argmax(abs(vec[:,i]))
            dimensions = np.zeros((3,1),float)
            for i in xrange(3):
                dimensions[max_loc[i,0],0] = lam[i]

            self.dimensions.append(dimensions)
        '''
        #Using Mass Moment of Inertia tensor
        for c in self.cells:
            dl = c
            tvol = []
            tcent = []
            for i in xrange(dl.GetOutput().GetNumberOfCells()):
                tetra = dl.GetOutput().GetCell(i)

                points = tetra.GetPoints().GetData()
                center = [0.,0.,0.]
                tetra.TetraCenter(points.GetTuple(0),
                        points.GetTuple(1),
                        points.GetTuple(2),
                        points.GetTuple(3),
                        center)
                tcent.append(center)
                tvol.append(tetra.ComputeVolume(points.GetTuple(0),
                        points.GetTuple(1),
                        points.GetTuple(2),
                        points.GetTuple(3)))

            vol = np.sum(tvol)
            #get the cell centroid
            c = np.zeros((3,),float)
            for i in xrange(len(tvol)):
                c[0] += tvol[i]*tcent[i][0]
                c[1] += tvol[i]*tcent[i][1]
                c[2] += tvol[i]*tcent[i][2]
            c /= vol

            I = np.zeros((3,3),float)
            for i in xrange(len(tvol)):
                tcent[i][0] -= c[0]
                tcent[i][1] -= c[1]
                tcent[i][2] -= c[2]
                I[0,0] += tvol[i]*(tcent[i][1]**2+tcent[i][2]**2)
                I[0,1] += -tvol[i]*(tcent[i][0]*tcent[i][1])
                I[0,2] += -tvol[i]*(tcent[i][0]*tcent[i][2])
                I[1,1] += tvol[i]*(tcent[i][0]**2+tcent[i][2]**2)
                I[1,2] += -tvol[i]*(tcent[i][1]*tcent[i][2])
                I[2,2] += tvol[i]*(tcent[i][0]**2+tcent[i][1]**2)
            I[1,0] = I[0,1]
            I[2,0] = I[0,2]
            I[2,1] = I[1,2]


            [lam,vec] = np.linalg.eig(I)
            order = np.argsort(lam)
            v = vec[order,:]
            global_order = []
            for i in xrange(3):
                global_order.append(np.argmax(abs(v[:,i])))

            global_order = global_order[::-1]

            lam.sort()

            a = []
            a.append(np.sqrt((5./2.)*(lam[1]+lam[0]-lam[2])/vol))
            a.append(np.sqrt((5./2.)*(lam[2]+lam[0]-lam[1])/vol))
            a.append(np.sqrt((5./2.)*(lam[2]+lam[1]-lam[0])/vol))

            d = [a[global_order[0]],a[global_order[1]],a[global_order[2]]]

            self.dimensions.append(d)
