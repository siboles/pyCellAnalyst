import vtk,os,random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.linalg import sqrtm
def matchVol(x,sF,devF,R,sv,mv,tv,material,spatial,rc,sc):
    nF = x[0]*sF+x[1]*devF
    return abs(sv/mv-np.linalg.det(nF))

def obj(x,sF,devF,R,sv,mv,tv,material,spatial,rc,sc):
    nF = np.dot(R,x[0]*sF)+np.dot(R,x[1]*devF)
    #Make a copy of material configuration and deform this with nF
    nm = vtk.vtkPolyData()
    nm.DeepCopy(material)
    pcoords = vtk.vtkFloatArray()
    pcoords.SetNumberOfComponents(3)
    pcoords.SetNumberOfTuples(nm.GetNumberOfPoints())
    for i in xrange(nm.GetNumberOfPoints()):
        p = [0.,0.,0.]
        nm.GetPoint(i,p)
        p = np.dot(nF,p-rc)
        p.flatten()
        pcoords.SetTuple3(i,p[0]+sc[0],p[1]+sc[1],p[2]+sc[2])

    points = vtk.vtkPoints()
    points.SetData(pcoords)
    nm.SetPoints(points)
    nm.GetPoints().Modified()

    #calculate both the intersection and the union of the two polydata
    intersect = vtk.vtkBooleanOperationPolyDataFilter()
    intersect.SetOperation(1)
    intersect.SetInputData(0,nm)
    intersect.SetInputData(1,spatial)
    intersect.Update()

    union = vtk.vtkBooleanOperationPolyDataFilter()
    union.SetOperation(0)
    union.SetInputData(0,nm)
    union.SetInputData(1,spatial)
    union.Update()

    unionmass = vtk.vtkMassProperties()
    unionmass.SetInputConnection(union.GetOutputPort())
    vol_union = unionmass.GetVolume()

    if intersect.GetOutput().GetNumberOfPoints() > 0:
        intmass = vtk.vtkMassProperties()
        intmass.SetInputConnection(intersect.GetOutputPort())
        vol_int = intmass.GetVolume()
    else:
        #penalize with distance between centroids 
        w = sc.T-np.dot(nF,rc.T)
        c = np.linalg.norm(w)
        vol_int = c*vol_union 

    diff = max([abs(sv-vol_int),abs(sv-vol_union)])
    return diff

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
        self.localFs = []

        self._readstls()
        self._pointwiseF()
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
                self.dsurfs.append(triangles.GetOutput())

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
            tv = self.dcentroids[i]-self.rcentroids[i]
            bounds = [(0.2,1.5),(0.01,1.5)]
            x = [1.,1.]
            # get U
            U = sqrtm(np.dot(self.localFs[i].T,self.localFs[i]))
            # get rigid rotation R
            R = np.dot(self.localFs[i],np.linalg.inv(U))
            # split F into deviatoric and spherical tensors
            a = 1./3.*np.trace(U)
            sU = a*np.eye(3)
            devU = U-sU
            print "Starting optimization for Cell %d" % i
            args=(sU,devU,R,self.dvols[i],self.rvols[i],tv,self.rsurfs[i],self.dsurfs[i],self.rcentroids[i],self.dcentroids[i])
            constraints = {"type":"eq","fun":matchVol,"args":args}
            res = minimize(obj,x,args=args,method="SLSQP",bounds=bounds,jac=False,constraints=constraints,options={"disp":True})
            x = res.x
            F = np.dot(R,x[0]*sU)+np.dot(R,x[1]*devU)
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

    def _pointwiseF(self):
        d = np.array(self.rcentroids)
        tree = KDTree(d)
        # get the 4 nearest neighbors (obviously the first one is the point itself)
        nn = tree.query(d,k=4)[1]
        N = len(self.rcentroids)
        X = []
        x = []
        for i in xrange(N):
            W = np.zeros((3,3),float)
            w = np.zeros((3,3),float)
            for j in xrange(3):
                W[j,:] = self.rcentroids[nn[i,j+1]]-self.rcentroids[i]
                w[j,:] = self.dcentroids[nn[i,j+1]]-self.dcentroids[i]
            X.append(W)
            x.append(w)
        #Solve the linear system for each pointwise F
        for i in xrange(N):
            W = X[i]
            w = x[i]
            A = np.zeros((9,9),float)
            A[0,0:3] = W[0,:]
            A[1,3:6] = W[0,:]
            A[2,6:9] = W[0,:]
            A[3,0:3] = W[1,:]
            A[4,3:6] = W[1,:]
            A[5,6:9] = W[1,:]
            A[6,0:3] = W[2,:]
            A[7,3:6] = W[2,:]
            A[8,6:9] = W[2,:]
            b = w.flatten().T
            F = np.linalg.solve(A,b)
            self.localFs.append(F.reshape(3,3))

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
