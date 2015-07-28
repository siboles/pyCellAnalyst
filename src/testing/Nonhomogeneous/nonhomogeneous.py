import vtk
import numpy as np
import os
import sys
from pyCellAnalyst import CellMech
from vtk.util import numpy_support
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
try:
    os.mkdir("DistanceErrors")
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
cases = ("MaterialCase", "SpatialCase")
N = 1000
voxel_dims = np.zeros((N, 3), np.float64)
for i in xrange(N):
    dims = np.zeros((2, 3), np.float64)
    for j, case in enumerate(cases):
        a = 5.0
        b = 2.0
        c = 2.0
        lam1 = np.random.uniform(0.9, 1.1)
        lam2 = np.random.uniform(0.9, 1.1)
        lam3 = np.random.uniform(0.9, 1.1)
        N1 = np.random.uniform(0.8, 1.2)
        N2 = np.random.uniform(0.8, 1.2)

        ellipFunc = vtk.vtkParametricSuperEllipsoid()
        ellipFunc.SetXRadius(lam1 * a)
        ellipFunc.SetYRadius(lam2 * b)
        ellipFunc.SetZRadius(lam3 * c)
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

        writeSTL("{:s}/cell{:04d}.stl".format(case, i + 1), decim)
        dims[j, 0] = lam1 * a / 50
        dims[j, 1] = lam2 * b / 50
        dims[j, 2] = lam3 * c / 50

    voxel_dims[i, :] = np.max(dims, axis=0)

mech = CellMech(
    ref_dir="MaterialCase",
    def_dir="SpatialCase",
    rigidInitial=False,
    deformable=True,
    saveFEA=False,
    deformableSettings={'Iterations': 200,
                        'Maximum RMS': 0.01,
                        'Displacement Smoothing': 1.5,
                        'Precision': 0.02},
    display=False)

results = {"seed": seed,
           "rms": np.zeros(N, np.float32),
           "voxel_dims": voxel_dims}
for i in xrange(N):
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(mech.rsurfs[i])
    probe.SetSourceData(mech.cell_fields[i])
    probe.Update()

    poly = probe.GetOutput()
    poly.GetPointData().SetActiveVectors("Displacement")

    warp = vtk.vtkWarpVector()
    warp.SetInputData(poly)
    warp.Update()

    dist = vtk.vtkDistancePolyDataFilter()
    dist.SetInputData(0, mech.dsurfs[i])
    dist.SetInputData(1, warp.GetPolyDataOutput())
    dist.Update()
    residual = numpy_support.vtk_to_numpy(
        dist.GetOutput().GetPointData().GetArray("Distance"))
    rms = np.linalg.norm(residual) / np.sqrt(residual.size)
    results["rms"][i] = rms
    writePoly("DistanceErrors/dist_error{:04d}.vtp".format(i + 1), dist)

fid = open("results.pkl", "wb")
pickle.dump(results, fid, 2)
fid.close()
