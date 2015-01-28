import os, re, string, warnings, platform, vtk, fnmatch, shutil
from PIL import Image
from PIL import ImageOps
from PIL import ImageMath
import numpy as np
from scipy import stats
import scipy as sp
from vtk.util import vtkImageExportToArray as vte
from vtk.util import vtkImageImportFromArray as vti
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,whiten,vq

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
    
    if nda.ndim == 2:
        t.set_cmap("gray")
    
    if(title):
        plt.title(title)

def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s,:,:] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[:,:,s] for s in zslices]
    
    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))
    
        
    img_null = sitk.Image([0,0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())
    
    img_slices = []
    d = 0
    
    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1
        
    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1
     
    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1
    
    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen,d])
        #TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0,img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [d,maxlen]))
            img = sitk.Compose(img_comps)
            
    
    myshow(img, title, margin, dpi)

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
        #self._getDimensions()

    def _parseStack(self):
        self._stack = []
        cnt = 0
        files = fnmatch.filter(sorted(os.listdir(self._vol_dir)),'*.tif')
        counter = [re.search("[0-9]*\.tif",f).group() for f in files]
        for i,c in enumerate(counter):
            counter[i] = int(c.replace('.tif',''))
        files = np.array(files,dtype=object)
        sorter = np.argsort(counter)
        files = files[sorter]
        for fname in files:
            if '.tif' in fname.lower():
                #enhancer = ImageEnhance.Contrast(img)
                img = ImageOps.grayscale(Image.open(self._vol_dir+self._path_dlm+fname))
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

        v = vte.vtkImageExportToArray()
        v.SetInputData(im)

        n = np.float32(v.GetArray())
        idx = np.argwhere(n)
        (ystart,xstart,zstart), (ystop,xstop,zstop) = idx.min(0),idx.max(0)+1
        I,J,K = n.shape
        if ystart > 5:
            ystart -= 5
        else:
            ystart = 0
        if ystop < I-5:
            ystop += 5
        else:
            ystop = I
        if xstart > 5:
            xstart -= 5
        else:
            xstart = 0
        if xstop < J-5:
            xstop += 5
        else:
            xstop = J
        if zstart > 5:
            zstart -= 5
        else:
            zstart = 0
        if zstop < K-5:
            zstop += 5
        else:
            zstop = K

        a = n[ystart:ystop,xstart:xstop,zstart:zstop]
        itk_img = sitk.GetImageFromArray(a)
        itk_img.SetSpacing([self._pixel_dim[0],self._pixel_dim[1],self._pixel_dim[2]])
        
        print "\n"
        print "-------------------------------------------------------"
        print "-- Applying Patch Based Denoising - this can be slow --"
        print "-------------------------------------------------------"
        print "\n"
        pb = sitk.PatchBasedDenoisingImageFilter()
        pb.KernelBandwidthEstimationOn()
        pb.SetNoiseModel(3) #use a Poisson noise model since this is confocal
        pb.SetNoiseModelFidelityWeight(1)
        pb.SetNumberOfSamplePatches(20)
        pb.SetPatchRadius(4)
        pb.SetNumberOfIterations(10)

        fimg = pb.Execute(itk_img)
        b = sitk.GetArrayFromImage(fimg)
        intensity = b.max()

        #grad = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
        #grad.SetSigma(0.05)
        gf = sitk.GradientMagnitudeImageFilter()
        gf.UseImageSpacingOn()
        grad = gf.Execute(fimg)
        edge = sitk.Cast(sitk.BoundedReciprocal( grad ),sitk.sitkFloat32)


        print "\n"
        print "-------------------------------------------------------"
        print "---- Thresholding to deterimine initial level sets ----"
        print "-------------------------------------------------------"
        print "\n"
        t = 0.5
        seed = sitk.BinaryThreshold(fimg,t*intensity)
        #Opening (Erosion/Dilation) step to remove islands smaller than 2 voxels in radius)
        seed = sitk.BinaryMorphologicalOpening(seed,2)
        seed = sitk.BinaryFillhole(seed!=0)
        #Get connected regions
        r = sitk.ConnectedComponent(seed)
        labels = sitk.GetArrayFromImage(r)
        ids = sorted(np.unique(labels))
        N = len(ids)
        if N > 2:
            i = np.copy(N)
            while i == N and (t-self._tratio)>-1e-7:
                t -= 0.01
                seed = sitk.BinaryThreshold(fimg,t*intensity)
                #Opening (Erosion/Dilation) step to remove islands smaller than 2 voxels in radius)
                seed = sitk.BinaryMorphologicalOpening(seed,2)
                seed = sitk.BinaryFillhole(seed!=0)
                #Get connected regions
                r = sitk.ConnectedComponent(seed)
                labels = sitk.GetArrayFromImage(r)
                i = len(np.unique(labels))
                if i > N:
                    N = np.copy(i)
            t+=0.01
        else:
            t = np.copy(self._tratio)
        seed = sitk.BinaryThreshold(fimg,t*intensity)
        #Opening (Erosion/Dilation) step to remove islands smaller than 2 voxels in radius)
        seed = sitk.BinaryMorphologicalOpening(seed,2)
        seed = sitk.BinaryFillhole(seed!=0)
        #Get connected regions
        r = sitk.ConnectedComponent(seed)
        labels = sitk.GetArrayFromImage(r)
        labels = np.unique(labels)[1:]

        '''
        labels[labels==0] = -1
        labels = sitk.GetImageFromArray(labels)
        labels.SetSpacing([self._pixel_dim[0],self._pixel_dim[1],self._pixel_dim[2]])
        #myshow3d(labels,zslices=range(20))
        #plt.show()
        ls = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
        ls.UseImageSpacingOn()
        ls.SetLambda2(1.5)
        #ls.SetCurvatureWeight(1.0)
        ls.SetAreaWeight(1.0)
        #ls.SetReinitializationSmoothingWeight(1.0)
        ls.SetNumberOfIterations(100)
        seg = ls.Execute(sitk.Cast(labels,sitk.sitkFloat32),sitk.Cast(fimg,sitk.sitkFloat32))
        seg = sitk.Cast(seg,sitk.sitkUInt8)
        seg = sitk.BinaryMorphologicalOpening(seg,1)
        seg = sitk.BinaryFillhole(seg!=0)
        #Get connected regions
        #r = sitk.ConnectedComponent(seg)
        contours = sitk.BinaryContour(seg)
        myshow3d(sitk.LabelOverlay(sitk.Cast(fimg,sitk.sitkUInt8),contours),zslices=range(fimg.GetSize()[2]))
        plt.show()
        '''

        segmentation = sitk.Image(r.GetSize(),sitk.sitkUInt8)
        segmentation.SetSpacing([self._pixel_dim[0],self._pixel_dim[1],self._pixel_dim[2]])
        for l in labels:
            d = sitk.SignedMaurerDistanceMap(r==l,insideIsPositive=False,squaredDistance=True,useImageSpacing=True)
            #d = sitk.BinaryThreshold(d,-1000,0)
            #d = sitk.Cast(d,edge.GetPixelIDValue() )*-1+0.5
            #d = sitk.Cast(d,edge.GetPixelIDValue() )
            seg = sitk.GeodesicActiveContourLevelSetImageFilter()
            seg.SetPropagationScaling(1.0)
            seg.SetAdvectionScaling(1.0)
            seg.SetCurvatureScaling(0.5)
            seg.SetMaximumRMSError(0.01)
            levelset = seg.Execute(d,edge)
            levelset = sitk.BinaryThreshold(levelset,-1000,0)
            segmentation = sitk.Add(segmentation,levelset)
            print ("RMS Change for Cell %d: "% l,seg.GetRMSChange())
            print ("Elapsed Iterations for Cell %d: "% l, seg.GetElapsedIterations())
        '''
        contours = sitk.BinaryContour(segmentation)
        myshow3d(sitk.LabelOverlay(sitk.Cast(fimg,sitk.sitkUInt8),contours),zslices=range(fimg.GetSize()[2]))
        plt.show()
        '''

        n[ystart:ystop,xstart:xstop,zstart:zstop] = sitk.GetArrayFromImage(segmentation)*100

        i = vti.vtkImageImportFromArray()
        i.SetDataSpacing([self._pixel_dim[0],self._pixel_dim[1],self._pixel_dim[2]])
        i.SetDataExtent([0,100,0,100,1,len(files)])
        i.SetArray(n)
        i.Update()

        thres=vtk.vtkImageThreshold()
        thres.SetInputData(i.GetOutput())
        thres.ThresholdByLower(0)
        thres.ThresholdByUpper(101)

        iso=vtk.vtkImageMarchingCubes()
        iso.SetInputConnection(thres.GetOutputPort())
        iso.SetValue(0,1)

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
