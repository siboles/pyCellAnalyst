import vtk,os,itertools
import SimpleITK as sitk
import numpy as np
from vtk.util import numpy_support

class CellMech(object):
    '''
    USAGE: Will read STL files from two directories and calculate the complete strain tensor for each volume.
           The STL files must be named the same in each directory, so they are matched appropriately.
    INPUT:
           ref_dir - the directory containing the STL files corresponding to the reference (undeformed) state
           def_dir - the directory containing the STL files corresponding to the deformed state
           labels    TYPE: Boolean. If True deformation analysis will be performed on the labels.nii images.
                           This will call deformableRegistration(), which will calculate a displacement map between the two label images.
           surfaces  TYPE: Boolean. If True, deformation analysis will be performed between STL surfaces.
                           This will assume a single affine transformation of the cell, and calculate this iteratively with
                           vtkIterativeClosestPointTransform().
           NOTE: In the case that both labels and surfaces are set to True, both analyses will be performed.
    MEMBER ATTRIBUTES:
           self.rmeshes - list of delaunay tessalations of the reference STL files
           self.dmeshes - list of delaunay tessalations of the deformed STL files
           self.defmeshes - list of the reference delaunay tesselations deformed to the shape of the deformed STLs
           self.cell_strains - list of numpy arrays containing the homogeneous strain tensor for each cell
           self.rvols - volumes of reference state STLs
           self.dvols - volumes of deformed state STLs
           self.vstrains - volumetric strains of cells
           self.effstrains - effective strain of cells
           self.ecm_strain - ecm strain assuming homogeneity across the region containing cells
           self.rcentroids - centroids of reference state cells
           self.dcentroids - centroids of deformed state cells
    '''
    def __init__(self,ref_dir=None,def_dir=None,labels=True,surfaces=False):
        if ref_dir is None:
            raise SystemExit("You must indicate a directory containing reference state STLs. Terminating...")
        if def_dir is None:
            raise SystemExit("You must indicate a directory containing deformed state STLs. Terminating...")
        self._ref_dir = ref_dir
        self._def_dir = def_dir
        self.rlabels = []
        self.dlabels = []
        self.rsurfs = []
        self.dsurfs = []
        self.rcentroids = []
        self.dcentroids = []
        self.cell_strains = []
        self.ecm_strain = None
        self.rvols = []
        self.dvols = []
        self.raxes = []
        self.daxes = []
        self.vstrains = []
        self.effstrains = []

        if self.surfaces:
            self._readstls()
            self._getmech()
        self._deform()

    def _readstls(self):
        for fname in sorted(os.listdir(self._ref_dir)):
            if '.stl' in fname.lower():
                reader = vtk.vtkSTLReader()
                reader.SetFileName(self._ref_dir+'/'+fname)
                reader.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputConnection(reader.GetOutputPort())
                triangles.Update()
                self.rsurfs.append(triangles.GetOutput())

        for fname in sorted(os.listdir(self._def_dir)):
            if '.stl' in fname.lower():
                reader = vtk.vtkSTLReader()
                reader.SetFileName(self._def_dir+'/'+fname)
                reader.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputConnection(reader.GetOutputPort())
                triangles.Update()
                self.dsurfs.append(triangles.GetOutput())

    def _deform(self):
        for i in xrange(len(self.rcentroids)):
            # volumetric strains
            self.vstrains.append(self.dvols[i]/self.rvols[i]-1)
            ICP = vtk.vtkIterativeClosestPointTransform()
            ICP.SetSource(self.rsurfs[i])
            ICP.SetTarget(self.dsurfs[i])
            ICP.GetLandmarkTransform().SetModeToAffine()
            ICP.SetMaximumMeanDistance(0.001)
            ICP.SetCheckMeanDistance(1)
            ICP.SetMaximumNumberOfIterations(5000)
            ICP.StartByMatchingCentroidsOn()
            ICP.Update()
            F = np.zeros((3,3),float)
            for j in xrange(3):
                for k in xrange(3):
                    F[j,k] = ICP.GetMatrix().GetElement(j,k)
            E = 0.5*(np.dot(F.T,F)-np.eye(3))
            v_err = np.linalg.det(F) - (self.vstrains[-1]+1)
            print "Error in Cell %d volume" % i, v_err
            self.cell_strains.append(E)

    def deformableRegistration(self):
        register = sitk.DiffeomorphicDemonsRegistrationFilter()
        register.SetNumberOfIterations(100)
        register.SmoothDisplacementFieldOn()
        register.SmoothUpdateFieldOff()
        register.UseImageSpacingOn()
        register.SetUseGradientType(3)
        for m,s in itertools.izip(self.material,self.spatial):
            self.displacements.append(register.Execute(s,m))
            a = sitk.GetArrayFromImage(self.displacements[-1])
            a = a.swapaxes(0,2)
            origin = m.GetOrigin()
            size = m.GetSize()
            dX = m.GetSpacing()
            nodes = np.meshgrid(np.linspace(origin[0],size[0],dX[0]),np.linspace(origin[1],size[1],dX[1]),np.linspace(origin[2],size[2],dX[2]))
            
    def _getmech(self):
        #get the ECM strain
        rc = np.array(self.rcentroids)
        dc = np.array(self.dcentroids)
        if rc.shape[0] < 4:
            print("WARNING: There are less than 4 objects in the space; therefore, tissue strain was not calculated.")
            return
        da = numpy_support.numpy_to_vtk(rc)
        p = vtk.vtkPoints()
        p.SetData(da)
        pd = vtk.vtkPolyData()
        pd.SetPoints(p)

        tet = vtk.vtkDelaunay3D()
        tet.SetInputData(pd)
        tet.Update()
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(tet.GetOutput())
        quality.Update()
        mq = quality.GetOutput().GetCellData().GetArray("Quality")
        mq = numpy_support.vtk_to_numpy(mq)
        try:
            btet = np.argmin(abs(mq-1.0)) # tet with edge ratio closest to 1
        except:
            return
        idlist = tet.GetOutput().GetCell(btet).GetPointIds()
        P = np.zeros((4,3),float)
        p = np.zeros((4,3),float)
        for i in xrange(idlist.GetNumberOfIds()):
            P[i,:] = rc[idlist.GetId(i),:]
            p[i,:] = dc[idlist.GetId(i),:]
        
        X = np.array([P[1,:]-P[0,:],
        P[2,:]-P[0,:],
        P[3,:]-P[0,:],
        P[3,:]-P[1,:],
        P[3,:]-P[2,:],
        P[2,:]-P[1,:]],float)
        
        x = np.array([p[1,:]-p[0,:],
        p[2,:]-p[0,:],
        p[3,:]-p[0,:],
        p[3,:]-p[1,:],
        p[3,:]-p[2,:],
        p[2,:]-p[1,:]],float)
        
        #assemble the system
        dX = np.zeros((6,6),float)
        ds = np.zeros((6,1),float)
        for i in xrange(6):
            dX[i,0] = 2*X[i,0]**2
            dX[i,1] = 2*X[i,1]**2
            dX[i,2] = 2*X[i,2]**2
            dX[i,3] = 4*X[i,0]*X[i,1]
            dX[i,4] = 4*X[i,0]*X[i,2]
            dX[i,5] = 4*X[i,1]*X[i,2]

            ds[i,0] = np.linalg.norm(x[i,:])**2-np.linalg.norm(X[i,:])**2

        E = np.linalg.solve(dX,ds)
        E = np.array([[E[0,0],E[3,0],E[4,0]],[E[3,0],E[1,0],E[5,0]],[E[4,0],E[5,0],E[2,0]]],float)
        self.ecm_strain = E

    def getDimensions(self):
        labelstats = self._getLabelShape(self.rlabels)
        self.rvols = labelstats['volume']
        self.rcentroids = labelstats['centroid']
        self.raxes = labelstats['ellipsoid diameters']
        labelstats = self._getLabelShape(self.dlabels)
        self.dvols = labelstats['volume']
        self.dcentroids = labelstats['centroid']
        self.daxes = labelstats['ellipsoid diameters']
        
    def _getLabelShape(self,img):
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
            labelshape['ellipsoid diameters'].append(ls.GetEquivalentEllipsoidDiameter(l))
            labelshape['bounding box'].append(ls.GetBoundingBox(l))
        return labelshape

