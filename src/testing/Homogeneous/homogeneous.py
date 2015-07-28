import vtk
import numpy as np
import os
import sys
from pyCellAnalyst import CellMech
import pickle


def writePoly(name, tri):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(name)
    writer.SetInputData(tri.GetOutput())
    writer.Write()


def writeSTL(name, tri):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(name)
    writer.SetInputData(tri.GetOutput())
    writer.Write()

try:
    os.mkdir("MaterialCase")
except:
    pass
try:
    os.mkdir("SpatialCase")
except:
    pass

if len(sys.argv) == 1:
    seed = np.random.randint(0, sys.maxint)
    print("New seed generated for RNG: {:d}".format(seed))
elif len(sys.argv) == 2:
    seed = int(sys.argv[-1])
    print("RNG seed specified by user: {:d}".format(seed))
else:
    raise SystemExit(
        ("Too many arguments: expected 0 (generate new random seed) ",
         "or 1 (user-specified seed). Exitting..."))
np.random.seed(seed)
N = 1000
truth = np.zeros((N, 3, 3), np.float64)
for i in xrange(N):
    a = np.random.uniform(3.0, 5.0)
    b = np.random.uniform(2.0, 3.0)
    c = np.random.uniform(2.0, 3.0)
    N1 = np.random.uniform(0.8, 1.2)
    N2 = np.random.uniform(0.8, 1.2)

    #principal stretches
    lam1 = np.random.uniform(0.8, 1.2)
    lam2 = np.random.uniform(0.8, 1.2)
    lam3 = np.random.uniform(0.8, 1.2)

    #Euler angles for rotation from Cartesian basis
    #the aim is to randomly define an eigenbasis for
    #the principal stretches that transforms to identity {e_i}
    #by this rotation generating a U containing dilatation
    #and shear with no worry of not being positive definite
    #Euler angle definition (extrinsic):
    #   alpha - rotation about reference z
    #   beta - rotation about reference x
    #   gamma - rotation about z
    alpha = np.random.uniform(0, np.pi / 4.0)
    beta = np.random.uniform(0, np.pi / 4.0)
    gamma = np.random.uniform(0, np.pi / 4.0)

    Q = np.zeros((3, 3), np.float64)
    Q[0, 0] = np.cos(alpha) * np.cos(gamma) - np.cos(beta) * np.sin(alpha) * np.sin(gamma)
    Q[0, 1] = -np.cos(alpha) * np.sin(gamma) - np.cos(beta) * np.cos(gamma) * np.sin(alpha)
    Q[0, 2] = np.sin(alpha) * np.sin(beta)
    Q[1, 0] = np.cos(gamma) * np.sin(alpha) + np.cos(alpha) * np.cos(beta) * np.sin(gamma)
    Q[1, 1] = np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.sin(alpha) * np.sin(gamma)
    Q[1, 2] = -np.cos(alpha) * np.sin(beta)
    Q[2, 0] = np.sin(beta) * np.sin(gamma)
    Q[2, 1] = np.cos(gamma) * np.sin(beta)
    Q[2, 2] = np.cos(beta)

    #$U = \sum_1^3 \lambda_i \mathbf{r}_i \outer \mathbf{r}_i$
    U = np.zeros((3, 3), np.float64)
    l = [lam1, lam2, lam3]
    for j in xrange(3):
        r = np.dot(Q, np.eye(3)[:, j])
        U += np.outer(l[j] * r, r)

    ellipFunc = vtk.vtkParametricSuperEllipsoid()
    ellipFunc.SetXRadius(a)
    ellipFunc.SetYRadius(b)
    ellipFunc.SetZRadius(c)
    ellipFunc.SetN1(N1)
    ellipFunc.SetN2(N2)

    ellip = vtk.vtkParametricFunctionSource()
    ellip.SetParametricFunction(ellipFunc)
    ellip.SetUResolution(50)
    ellip.SetVResolution(30)
    ellip.SetWResolution(30)
    ellip.SetScalarModeToNone()
    ellip.Update()

    d = np.pi / 15 * lam1 * a
    resample = vtk.vtkPolyDataPointSampler()
    resample.SetInputData(ellip.GetOutput())
    resample.SetDistance(d)
    resample.Update()

    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(resample.GetOutput())
    delaunay.Update()

    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(delaunay.GetOutput())
    geo.Update()

    decim = vtk.vtkDecimatePro()
    decim.SetInputData(geo.GetOutput())
    decim.SetTargetReduction(.2)
    decim.Update()

    writeSTL("MaterialCase/cell{:04d}.stl".format(i + 1), decim)

    transform = vtk.vtkTransform()
    tmp = np.eye(4)
    tmp[0:3, 0:3] = U
    transform.SetMatrix(tmp.ravel())
    transform.Update()

    spatial = vtk.vtkTransformPolyDataFilter()
    spatial.SetTransform(transform)
    spatial.SetInputData(decim.GetOutput())
    spatial.Update()

    writeSTL("SpatialCase/cell{:04d}.stl".format(i + 1), spatial)
    truth[i, :, :] = 0.5 * (np.dot(U.T, U) - np.eye(3))

mech = CellMech(
    ref_dir="MaterialCase",
    def_dir="SpatialCase",
    rigidInitial=False,
    deformable=False,
    saveFEA=False,
    display=False)
residual = np.array(mech.cell_strains) - truth
results = {"residual": residual,
           "truth": truth}

fid = open("results.pkl", "wb")
pickle.dump(results, fid, 2)
fid.close()
