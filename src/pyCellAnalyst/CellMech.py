import vtk,os,random
import numpy as np
from scipy.optimize import minimize

def positiveVolRatio(x,a):
    F = np.matrix([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],x[8]]],float)
    return np.linalg.det(F)

def matchVolStrain(x,a):
    vs = a[3]
    F = np.matrix([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],x[8]]],float)
    return np.linalg.det(F)-(vs+1)

def obj(x,a):
    r =a[0]
    d = a[1]
    p2 = a[2]
    vs = a[3]

    F = np.matrix([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],x[8]]],float)
    r = np.dot(F,r)
    r = r.T
    N = r.shape[0]
    ssr = 0
    for i in xrange(N):
        p1 = r[i,:]
        p1 = np.array(p1).flatten()
        u = (p1-p2)/np.linalg.norm(p1-p2)
        l = p2+1000*u  # a 1000 um line segment from centroid passing through p1
        tol = 1e-8
        points = vtk.vtkPoints()
        cells = vtk.vtkIdList()
        d.IntersectWithLine(p2,l,tol,points,cells)
        intersection = [0.,0.,0.]
        points.GetPoint(0,intersection)
        ssr += np.linalg.norm(p1-intersection)**2
    return ssr/N

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
        self.raxes = []
        self.daxes = []
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

                vol, cent, axes = self._getMassProps(self.rmeshes[-1])
                self.rvols.append(vol)
                self.rcentroids.append(cent)
                self.raxes.append(axes)

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

                vol, cent, axes = self._getMassProps(self.dmeshes[-1])
                self.dvols.append(vol)
                self.dcentroids.append(cent)
                self.daxes.append(axes)


    def _deform(self):
        #align centroids
        for i in xrange(len(self.rcentroids)):
            # volumetric strains
            self.vstrains.append(self.dvols[i]/self.rvols[i]-1)
            #create a polydata set of the cell in deformed state
            pd = vtk.vtkGeometryFilter()
            pd.SetInputConnection(self.dsurfs[i].GetOutputPort())
            pd.Update()

            #Build the Modified BSPTree
            tree = vtk.vtkModifiedBSPTree()
            tree.SetDataSet(pd.GetOutput())
            tree.BuildLocator()

            #find the translation vector to align centroids
            tv = self.dcentroids[i]-self.rcentroids[i]
            #loop over the tetrahedrons and update their vertices to the new position
            m = vtk.vtkGeometryFilter()
            m.SetInputConnection(self.rsurfs[i].GetOutputPort())
            m.Update()
            N = m.GetOutput().GetNumberOfPoints()
            rpoints = np.zeros((3,N),float)
            for j in xrange(N):
                p = np.array(m.GetOutput().GetPoint(j),float)+tv
                rpoints[:,j] = p.T
            e = self.raxes[i]
            ep = self.daxes[i]
            A = np.zeros((9,9),float)
            b = np.zeros((9,1),float)
            A[0,0:3] = e[:,0].T
            A[1,3:6] = e[:,0].T
            A[2,6:9] = e[:,0].T
            A[3,0:3] = e[:,1].T
            A[4,3:6] = e[:,1].T
            A[5,6:9] = e[:,1].T
            A[6,0:3] = e[:,2].T
            A[7,3:6] = e[:,2].T
            A[8,6:9] = e[:,2].T
            b[0:3,0] = ep[:,0]
            b[3:6,0] = ep[:,1]
            b[6:9,0] = ep[:,2]

            F = np.linalg.solve(A,b)
            x = F.flatten()
            F = np.matrix([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],x[8]]],float)
            E = 0.5*(np.dot(F.T,F)-np.eye(3))
            print E

            bounds = [(0.2,1.8),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(0.2,1.8),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(0.2,1.8)]
            a = [rpoints,tree,self.dcentroids[i],self.vstrains[i]]
            options = {"eps":1e-6}
            constraints = [{"type":"eq","fun":matchVolStrain,"args":(a,)},{"type":"ineq","fun":positiveVolRatio,"args":(a,)}]
            res = minimize(obj,x,args=(a,),method="SLSQP",bounds=bounds,options=options)
            x = res.x
            F = np.matrix([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],x[8]]],float)
            E = 0.5*(np.dot(F.T,F)-np.eye(3))
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
        E = np.array([[E[0,0],E[3,0],E[4,0]],[E[3,0],E[1,0],E[5,0]],[E[4,0],E[5,0],E[2,0]]],float)
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

        I = np.zeros((3,3),float)
        for i in xrange(len(tvol)):
            tcent[i,:] -= centroid
            I[0,0] += tvol[i]*(tcent[i,1]**2+tcent[i,2]**2)
            I[0,1] += -tvol[i]*(tcent[i,0]*tcent[i,1])
            I[0,2] += -tvol[i]*(tcent[i,0]*tcent[i,2])
            I[1,1] += tvol[i]*(tcent[i,0]**2+tcent[i,2]**2)
            I[1,2] += -tvol[i]*(tcent[i,1]*tcent[i,2])
            I[2,2] += tvol[i]*(tcent[i,0]**2+tcent[i,1]**2)
        I[1,0] = I[0,1]
        I[2,0] = I[0,2]
        I[2,1] = I[1,2]

        [lam,vec] = np.linalg.eig(I)
        order = np.argsort(lam)[::-1]
        l = lam[order]
        v = vec[:,order]

        a = []
        c = 5./2./volume
        a.append(np.sqrt(c*(lam[1]+lam[2]-lam[0])))
        a.append(np.sqrt(c*(lam[0]+lam[2]-lam[1])))
        a.append(np.sqrt(c*(lam[0]+lam[1]-lam[2])))

        axes = np.zeros((3,3),float)
        for i in xrange(3):
            axes[:,i] = a[i]*vec[:,i]

        return volume,centroid,axes
