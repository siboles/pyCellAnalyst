import os
import argparse

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np



def generateSuperEllipsoid(a, b, c, n1, n2):
    superEllipsoid = vtk.vtkParametricSuperEllipsoid()
    superEllipsoid.SetXRadius(a)
    superEllipsoid.SetYRadius(b)
    superEllipsoid.SetZRadius(c)
    superEllipsoid.SetN1(n1)
    superEllipsoid.SetN2(n2)

    ratios = np.array([a, b, c])
    ratios = ratios / np.max(ratios)
    superEllipsoidSource = vtk.vtkParametricFunctionSource()
    superEllipsoidSource.SetParametricFunction(superEllipsoid)
    superEllipsoidSource.SetUResolution(np.ceil(100*ratios[0]).astype(int))
    superEllipsoidSource.SetVResolution(np.ceil(100*ratios[1]).astype(int))
    superEllipsoidSource.SetWResolution(np.ceil(100*ratios[2]).astype(int))

    superEllipsoidSource.Update()
    polydata = superEllipsoidSource.GetOutput()
    return polydata

def packObjects(objects, spacing):
    bb = np.array([p.GetBounds() for p in objects])
    bb[:,0::2] -= np.array(spacing)
    bb[:,1::2] += np.array(spacing)
    dimensions = np.zeros((bb.shape[0], 3), dtype=float)
    dimensions[:,0] = bb[:,1] - bb[:,0]
    dimensions[:,1] = bb[:,3] - bb[:,2]
    dimensions[:,2] = bb[:,5] - bb[:,4]
    #sort first by height
    order = np.argsort(dimensions[:,2])
    #sort sub-groups of 4 minimizing total surface area
    for i in np.arange(0,order.size,4):
        remainder = order.size - i
        if remainder > 4:
            sortx = np.argsort(dimensions[order[i:i+4], 0])
            order[i:i+4] = order[i:i+4][sortx]
            # swap middle two if y lengths sum to less
            sum1 = dimensions[order[i],1] + dimensions[order[i+1],1]
            sum2 = dimensions[order[i+2],1] + dimensions[order[i+3],1]
            sum3 = dimensions[order[i],1] + dimensions[order[i+2],1]
            sum4 = dimensions[order[i+1],1] + dimensions[order[i+3],1]
            if max([sum1, sum2]) > max([sum3,sum4]):
                order[i:i+4] = order[i:i+4][[0,2,1,3]]
        else:
            sortx = np.argsort(dimensions[order[i::], 0])
            order[i::] = order[i::][sortx]

    scene = vtk.vtkAppendPolyData()
    h = 0
    cnt = 0
    for i in np.arange(0,order.size,4):
        remainder = order.size - i
        if remainder > 4:
            ind = order[i:i+4]
        else:
            ind = order[i::]
        h += np.max(dimensions[ind,2])
        for j in range(ind.size):
            cnt += 1
            tx = vtk.vtkTransform()
            mz = h - bb[ind[j], 5]
            if j == 0:
                mx = bb[ind[j], 0]
                my = bb[ind[j], 3]
            elif j == 1:
                mx = bb[ind[j], 0]
                my = bb[ind[j], 2]
            elif j == 2:
                mx = bb[ind[j], 1]
                my = bb[ind[j], 2]
            elif j == 3:
                mx = bb[ind[j], 1]
                my = bb[ind[j], 3]
            tx.Translate([mx, my, mz])
            txFilter = vtk.vtkTransformPolyDataFilter()
            txFilter.SetInputData(objects[ind[j]])
            txFilter.SetTransform(tx)
            scene.AddInputConnection(txFilter.GetOutputPort())
            scene.Update()
    return scene.GetOutput()

def deformPolyData(p, spacing, scale):
    # create thin-plate spline control points
    bounds = p.GetBounds()
    step_size = np.array(spacing) * 10
    div = [np.ceil(np.abs(bounds[2*i+1] - bounds[2*i])/ step_size[i]).astype(int) for i in range(3)]
    x, xstep = np.linspace(bounds[0], bounds[1], div[0], retstep=True)
    y, ystep = np.linspace(bounds[2], bounds[3], div[1], retstep=True)
    z, zstep = np.linspace(bounds[4], bounds[5], div[2], retstep=True)
    sourcepoints = np.meshgrid(x, y, z)
    x_perturb = np.random.normal(loc=0.0, scale=scale * xstep, size=sourcepoints[0].size)
    y_perturb = np.random.normal(loc=0.0, scale=scale * ystep, size=sourcepoints[1].size)
    z_perturb = np.random.normal(loc=0.0, scale=scale * zstep, size=sourcepoints[2].size)

    allsourcepoints = np.zeros(sourcepoints[0].size * 3)
    allsourcepoints[0::3] = sourcepoints[0].ravel()
    allsourcepoints[1::3] = sourcepoints[1].ravel()
    allsourcepoints[2::3] = sourcepoints[2].ravel()

    alltargetpoints = np.copy(allsourcepoints)
    alltargetpoints[0::3] += x_perturb
    alltargetpoints[1::3] += y_perturb
    alltargetpoints[2::3] += z_perturb

    sourcePoints = vtk.vtkPoints()
    targetPoints = vtk.vtkPoints()
    arr1 = numpy_support.numpy_to_vtk(allsourcepoints, deep=True, array_type=vtk.VTK_DOUBLE)
    arr1.SetNumberOfComponents(3)
    arr2 = numpy_support.numpy_to_vtk(alltargetpoints, deep=True, array_type=vtk.VTK_DOUBLE)
    arr2.SetNumberOfComponents(3)
    sourcePoints.SetData(arr1)
    targetPoints.SetData(arr2)

    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(sourcePoints)
    transform.SetTargetLandmarks(targetPoints)
    transform.SetBasisToR()

    polytransform = vtk.vtkTransformPolyDataFilter()
    polytransform.SetInputData(p)
    polytransform.SetTransform(transform)
    polytransform.Update()

    polydata = polytransform.GetOutput()
    return polydata

def poly2img(p, spacing, noiseLevel):
    bb = np.array(p.GetBounds())
    extent = [np.ceil(np.abs(bb[2*i+1] - bb[2*i]) / spacing[i]).astype(int) + 10 for i in range(3)]

    arr = numpy_support.numpy_to_vtk(np.ones(extent, np.float32).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    arr.SetName('Intensity')
    arr.SetNumberOfComponents(1)

    img = vtk.vtkImageData()
    img.SetSpacing(spacing)
    img.SetExtent((0, extent[0]-1, 0, extent[1]-1, 0, extent[2]-1))
    img.SetOrigin([bb[0] - 5*spacing[0],
                   bb[2] - 5*spacing[1],
                   bb[4] - 5*spacing[2]])
    img.GetPointData().SetScalars(arr)
    p2im = vtk.vtkPolyDataToImageStencil()
    p2im.SetInputData(p)
    p2im.SetOutputOrigin(img.GetOrigin())
    p2im.SetOutputSpacing(img.GetSpacing())
    p2im.SetOutputWholeExtent(img.GetExtent())
    p2im.SetTolerance(min(spacing) / 10.0)
    p2im.Update()

    imstenc = vtk.vtkImageStencil()
    imstenc.SetInputData(img)
    imstenc.SetStencilConnection(p2im.GetOutputPort())
    imstenc.ReverseStencilOff()
    imstenc.SetBackgroundValue(0.0)
    imstenc.Update()
    img = imstenc.GetOutput()

    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetArray('Intensity')).reshape(extent[2], extent[1], extent[0])
    itk_img = sitk.GetImageFromArray(arr)
    itk_img.SetSpacing(img.GetSpacing())
    itk_img.SetOrigin(img.GetOrigin())

    itk_img = sitk.AdditiveGaussianNoise(itk_img, standardDeviation=noiseLevel)
    itk_img = sitk.RescaleIntensity(itk_img, 0.0, 1.0)

    mask = sitk.BinaryThreshold(itk_img, 0.5, 1e3)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask)
    regions = []
    for l in ls.GetLabels():
        bb = ls.GetBoundingBox(l)
        origin = [i-2 for i in bb[0:3]]
        size = [i+4 for i in bb[3:]]
        regions.append(origin + size)
    return itk_img, regions

def generateTestImages(a=2.0, b=1.0, c=1.0, n1=0.9, n2=0.9, spacing=[0.1, 0.1, 0.1], output=None, deformed=0, noiseLevel=0.3, number=1):
    if output is None:
        root = os.getcwd()
    elif os.path.isabs(output):
        root = output
    else:
        root = os.path.join(os.getcwd(), output)

    if not os.path.exists(root):
        os.mkdir(root)

    if number > 1:
        a = np.random.uniform(low=0.85*a, high=1.15*a, size=number)
        b = np.random.uniform(low=0.85*b, high=1.15*b, size=number)
        c = np.random.uniform(low=0.85*c, high=1.15*c, size=number)
    else:
        a = [a]
        b = [b]
        c = [c]

    objects = []
    for i in range(number):
        objects.append(generateSuperEllipsoid(a[i], b[i], c[i], n1, n2))

    polydata = packObjects(objects, spacing)
    refpolydata = deformPolyData(polydata, spacing, 0.1)
    refimg, regions = poly2img(refpolydata, spacing, noiseLevel)

    allregions = {"refernce": [regions], "deformed": []}
    sitk.WriteImage(refimg, os.path.join(root, "ref.nii"))
    for i in range(deformed):
        defpolydata = deformPolyData(refpolydata, spacing, 0.1)
        defimg, regions = poly2img(refpolydata, spacing, noiseLevel)
        allregions["deformed"].append(regions)
        sitk.WriteImage(defimg, os.path.join(root, "def_{:03d}.nii".format(i+1)))
    return root, allregions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates 3D image(s) of cell-like geometry. Optionally, generate reference and deformed pairs. If number > 1 then radii are varied +/- 15%.')
    parser.add_argument('-a', type=float, default=2.0, help='float : x-radius of base super-ellipsoid')
    parser.add_argument('-b', type=float, default=1.0, help='float : y-radius of base super-ellipsoid')
    parser.add_argument('-c', type=float, default=1.0, help='float : z-radius of base super-ellipsoid')
    parser.add_argument('-n1', type=float, default=0.9,
                        help='float : shape parameter in v; (0.0,1.0) square to rounded corners 1.0 is ellipsoid, > 1.0 concave with sharp edges')
    parser.add_argument('-n2', type=float, default=0.9,
                        help='float : shape parameter in u; (0.0,1.0) square to rounded corners 1.0 is ellipsoid, > 1.0 concave with sharp edges')
    parser.add_argument('-output', type=str, default=None, help='str : output directory to write images to')
    parser.add_argument('-number', type=int, default=1, help='int : how many clustered cells to generate')
    parser.add_argument('-deformed', type=int, default=0, help='int : how many deformed images to generate.')
    parser.add_argument('-noise_level', type=float, default=0.3, help='float : standard deviation of Gaussian noise to add to normalized image.')
    args = parser.parse_args()
    generateTestImages(a=args.a, b=args.b, c=args.c, n1=args.n1, n2=args.n2, output=args.output, deformed=args.deformed, number=args.number)
