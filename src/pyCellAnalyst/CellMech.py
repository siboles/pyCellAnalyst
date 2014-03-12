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
           self.strains - list of numpy arrays containing the strain tensor for each element in the meshes
           self.rvols - volumes of reference state STLs
           self.dvols - volumes of deformed state STLs
           self.vstrains - volumetric strains of full cells
           self.effstrains - volume averaged effective strain of full cells
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
        self.rcentroids = []
        self.dcentroids = []
        self.strains = []
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
            pd.SetInputConnection(self.dmeshes[i].GetOutputPort())
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
            m = self.rmeshes[i]
            N = m.GetOutput().GetNumberOfCells()
            local_strains = np.zeros((N,6),float)
            for j in xrange(N):
                tetra = m.GetOutput().GetCell(j)
                p0 = [tetra.GetPoints().GetData().GetTuple(0),
                        tetra.GetPoints().GetData().GetTuple(1),
                        tetra.GetPoints().GetData().GetTuple(2),
                        tetra.GetPoints().GetData().GetTuple(3)]
                for k in xrange(4):
                    tetra.GetPoints().SetPoint(k,p0[k]+tv)
                p_def = np.zeros((4,3),float)
                p_ref = np.zeros((4,3),float)
                for k in xrange(4):
                    p1 = np.array(list(tetra.GetPoints().GetData().GetTuple(k)),float)
                    uv = p1-np.array(p2)
                    uv /= np.linalg.norm(uv)
                    pp1 = p1+uv*100
                    tol = 1e-8
                    points = vtk.vtkPoints()
                    cells = vtk.vtkIdList()
                    tree.IntersectWithLine(pp1,p2,tol,points,cells)
                    intersection = [0.,0.,0.]
                    points.GetPoint(0,intersection)
                    u = np.array(intersection)-p1
                    p_ref[k,:] = p0[k]+tv
                    p_def[k,:] = p0[k]+tv+u
                local_strains[j,:] = self._getTetStrain(p_ref,p_def)
        print local_strains

    def _getTetStrain(self,r,d):
        X = []
        x = []
        for i in xrange(3):
            for j in xrange(3-i):
                X.append(r[i,:]-r[j+i+1,:])
                x.append(d[i,:]-d[j+i+1,:])
        X = np.array(X,float)
        x = np.array(x,float)
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
        return E.T
        
        
        

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
        print E
            

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
