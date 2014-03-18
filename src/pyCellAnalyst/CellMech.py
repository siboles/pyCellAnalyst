import vtk,os
import numpy as np

class CellMech(object):
    '''
    USAGE: Will read STL files from two directories and calculate the complete strain tensor for each volume.
           The STL files must be named the same in each directory, so they are matched appropriately.
    INPUT:
           ref_dir - the directory containing the STL files corresponding to the reference (undeformed) state
           def_dir - the directory containing the STL files corresponding to the deformed state
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
    def __init__(self,ref_dir=None,def_dir=None):
        if ref_dir is None:
            raise SystemExit("You must indicate a directory containing reference state STLs. Terminating...")
        if def_dir is None:
            raise SystemExit("You must indicate a directory containing deformed state STLs. Terminating...")
        self._ref_dir = ref_dir
        self._def_dir = def_dir
        self.rmeshes = []
        self.dmeshes = []
        self.rsurfs = []
        self.dsurfs = []
        self.rcentroids = []
        self.dcentroids = []
        self.cell_strains = []
        self.ecm_strain = None
        self.rvols = []
        self.dvols = []
        self.vstrains = []
        self.effstrains = []

        self._readstls()
        self._deform()
        self._getmech()

    def _readstls(self):
        for fname in sorted(os.listdir(self._ref_dir)):
            if '.stl' in fname.lower():
                reader = vtk.vtkSTLReader()
                reader.SetFileName(self._ref_dir+'/'+fname)
                reader.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputConnection(reader.GetOutputPort())
                triangles.Update()
                self.rsurfs.append(triangles)
                dl = vtk.vtkDelaunay3D()
                dl.SetInputConnection(triangles.GetOutputPort())
                dl.Update()
                self.rmeshes.append(dl)
                vol, cent = self._getMassProps(self.rmeshes[-1])
                self.rvols.append(vol)
                self.rcentroids.append(cent)

        for fname in sorted(os.listdir(self._def_dir)):
            if '.stl' in fname.lower():
                reader = vtk.vtkSTLReader()
                reader.SetFileName(self._def_dir+'/'+fname)
                reader.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputConnection(reader.GetOutputPort())
                triangles.Update()
                self.dsurfs.append(triangles)
                dl = vtk.vtkDelaunay3D()
                dl.SetInputConnection(triangles.GetOutputPort())
                dl.Update()
                self.dmeshes.append(dl)
                vol, cent = self._getMassProps(self.dmeshes[-1])
                self.dvols.append(vol)
                self.dcentroids.append(cent)


    def _deform(self):
        #align centroids
        for i in xrange(len(self.rcentroids)):
            #create a polydata set of the cell in deformed state
            pd = vtk.vtkGeometryFilter()
            pd.SetInputConnection(self.dsurfs[i].GetOutputPort())
            pd.Update()
            #Now create a modified BSP tree for fast searching
            #http://www.vtk.org/Wiki/VTK/Examples/Cxx/DataStructures/ModifiedBSPTreeIntersectWithLine
            tree = vtk.vtkModifiedBSPTree()
            tree.SetDataSet(pd.GetOutput())
            tree.BuildLocator()
            p2 = self.dcentroids[i]
            #find the translation vector to align centroids
            tv = self.dcentroids[i]-self.rcentroids[i]
            #loop over the tetrahedrons and update their vertices to the new position
            m = vtk.vtkGeometryFilter()
            m.SetInputConnection(self.rsurfs[i].GetOutputPort())
            m.Update()
            N = m.GetOutput().GetNumberOfPoints()
            X = np.zeros((N,3),float)
            x = np.zeros((N,3),float)
            for j in xrange(N):
                p = np.array(m.GetOutput().GetPoint(j),float)
                p1 = p+tv
                X[j,:] = p1-p2 
                u = p1-p2
                u/= np.linalg.norm(u)
                l = p2+1000*u
                tol = 1e-8
                points = vtk.vtkPoints()
                cells = vtk.vtkIdList()
                tree.IntersectWithLine(p2,l,tol,points,cells)
                intersection = [0.,0.,0.]
                points.GetPoint(0,intersection)
                x[j,:] = np.array(intersection)-p2
            self._getCellStrain(X,x)
            # volumetric strains
            self.vstrains.append(self.dvols[i]/self.rvols[i]-1)

    def _getCellStrain(self,X,x):
        N = X.shape[0]
        #assemble the system
        dX = np.zeros((N,6),float)
        ds = np.zeros((N,1),float)
        for i in xrange(N):
            dX[i,0] = 2*X[i,0]**2
            dX[i,1] = 2*X[i,1]**2
            dX[i,2] = 2*X[i,2]**2
            dX[i,3] = 4*X[i,0]*X[i,1]
            dX[i,4] = 4*X[i,0]*X[i,2]
            dX[i,5] = 4*X[i,1]*X[i,2]

            ds[i,0] = np.linalg.norm(x[i,:])**2-np.linalg.norm(X[i,:])**2

        E = np.linalg.lstsq(dX,ds)[0]
        E = np.array([[E[0,0],E[3,0],E[4,0]],[E[3,0],E[1,0],E[5,0]],[E[4,0],E[5,0],E[3,0]]],float)
        self.cell_strains.append(E)
        

    def _getmech(self):
        #get the ECM strain
        rc = np.array(self.rcentroids)
        dc = np.array(self.dcentroids)
        #make all line segments
        n = rc.shape[0]
        X = []
        x = []
        ds = []
        for i in xrange(n-1):
            for j in xrange(n-i-1):
                X.append(rc[i,:] - rc[j+i+1,:])
                x.append(dc[i,:] - dc[j+i+1,:])
        X = np.array(X,float)
        x = np.array(x,float)
        #assemble the system
        dX = np.zeros((X.shape[0],6),float)
        ds = np.zeros((X.shape[0],1),float)
        for i in xrange(X.shape[0]):
            dX[i,0] = 2*X[i,0]**2
            dX[i,1] = 2*X[i,1]**2
            dX[i,2] = 2*X[i,2]**2
            dX[i,3] = 4*X[i,0]*X[i,1]
            dX[i,4] = 4*X[i,0]*X[i,2]
            dX[i,5] = 4*X[i,1]*X[i,2]

            ds[i,0] = np.linalg.norm(x[i,:])**2-np.linalg.norm(X[i,:])**2

        E = np.linalg.lstsq(dX,ds)[0]
        E = np.array([[E[0,0],E[3,0],E[4,0]],[E[3,0],E[1,0],E[5,0]],[E[4,0],E[5,0],E[3,0]]],float)
        self.ecm_strain = E

    def _getMassProps(self,mesh):
        tvol = []
        tcent = []
        for i in xrange(mesh.GetOutput().GetNumberOfCells()):
            tetra = mesh.GetOutput().GetCell(i)
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
        tvol = np.array(tvol)
        tcent = np.array(tcent)
        volume = np.sum(tvol)
        cx = np.sum(tvol*tcent[:,0])/volume
        cy = np.sum(tvol*tcent[:,1])/volume
        cz = np.sum(tvol*tcent[:,2])/volume
        centroid = np.hstack((cx,cy,cz))
        return volume,centroid

