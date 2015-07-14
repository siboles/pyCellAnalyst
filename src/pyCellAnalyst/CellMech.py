import vtk
import os
import pickle
import SimpleITK as sitk
import numpy as np
from vtk.util import vtkImageImportFromArray as vti
from vtk.util.numpy_support import vtk_to_numpy
from meshpy.tet import (MeshInfo, build, Options)


class CellMech(object):

    '''
    USAGE: Will read STL files from two directories and calculate the
           complete strain tensor for each volume.
           The STL files must be named the same in each directory,
           so they are matched appropriately.
    INPUT:
           ref_dir    TYPE: String. The directory containing the STL files
                      corresponding to the reference (undeformed) state
           def_dir    TYPE: String. The directory containing the STL files
                      corresponding to the deformed state
           deformable TYPE: Boolean. If True deformable image registration
                      will be performed on the labels.nii images.
                      This will call deformableRegistration(), which will
                      calculate a displacement map between the
                      two label images.
           saveFEA    TYPE: Boolean. If True will save nodes, elements,
                      surface nodes, and displacement boundary conditions
                      in a dictionary to cell{:02d}.pkl. This information
                      can then be used to run Finite element analysis in
                      whatever software the user desires.
    MEMBER ATTRIBUTES:
           self.rmeshes      TYPE: List. Delaunay tessalations of the
                             reference STL files
           self.dmeshes      TYPE: List. Delaunay tessalations of the
                             deformed STL files
           self.defmeshes    TYPE: List. Reference delaunay tesselations
                             deformed to the shape of the deformed STLs
           self.cell_strains TYPE: List. Numpy arrays containing the
                             homogeneous strain tensor for each cell
           self.rvols        TYPE: List. Volumes of reference state STLs
           self.dvols        TYPE: List. Volumes of deformed state STLs
           self.vstrains     TYPE: List. Volumetric strains of cells
           self.effstrains   TYPE: List. Effective strain of cells
           self.ecm_strain   TYPE: List. ECM strain assuming homogeneity
                             across the region containing cells
           self.rcentroids   TYPE: List. Centroids of reference state cells
           self.dcentroids   TYPE: List. Centroids of deformed state cells
    '''
    def __init__(self,
                 ref_dir=None,
                 def_dir=None,
                 rigidInitial=True,
                 deformable=False,
                 saveFEA=False,
                 deformableSettings={'Iterations': 200,
                                     'Maximum RMS': 0.01,
                                     'Displacement Smoothing': 3.0,
                                     'Precision': 0.01}):
        if ref_dir is None:
            raise SystemExit(("You must indicate a directory containing "
                              "reference state STLs. Terminating..."))
        if def_dir is None:
            raise SystemExit(("You must indicate a directory containing "
                              "deformed state STLs. Terminating..."))
        self._ref_dir = ref_dir
        self._def_dir = def_dir
        self.rigidInitial = rigidInitial
        self.rimg = sitk.ReadImage(
            str(os.path.normpath(self._ref_dir + os.sep + "stack.nii")))
        self.dimg = sitk.ReadImage(
            str(os.path.normpath(self._def_dir + os.sep + "stack.nii")))
        self.rlabels = sitk.ReadImage(
            str(os.path.normpath(self._ref_dir + os.sep + "labels.nii")))
        self.dlabels = sitk.ReadImage(
            str(os.path.normpath(self._def_dir + os.sep + "labels.nii")))
        self.deformable = deformable
        self.saveFEA = saveFEA
        self.deformableSettings = deformableSettings
        self.rsurfs = []
        self.dsurfs = []
        self.rmeshes = []
        self.rcentroids = []
        self.dcentroids = []
        self.cell_strains = []
        self.ecm_strain = None
        self.rvols = []
        self.dvols = []
        self.rbbox = []
        self.dbbox = []
        self.raxes = []
        self.daxes = []
        self.vstrains = []
        self.effstrains = []
        self.cell_fields = []

        self.rigidTransforms = []

        self._elements = []
        self._nodes = []
        self._snodes = []
        self._bcs = []

        self.getDimensions()
        self._readstls()
        if not(self.rsurfs):
            raise Exception(("No 3D surfaces detected. Currently 2D analysis "
                             "is not supported, so nothing was done."))
        self._getECMstrain()
        self._deform()

        if self.deformable:
            self.deformableRegistration()
        # no support for 2D FEA yet
        if self.saveFEA:
            for i, bc in enumerate(self._bcs):
                fea = {'nodes': self._nodes[i],
                       'elements': self._elements[i],
                       'surfaces': self._snodes[i],
                       'boundary conditions': bc}
                fid = open(str(os.path.normpath(
                    self._def_dir + os.sep + 'cellFEA{:02d}.pkl'
                    .format(i))), 'wb')
                pickle.dump(fea, fid)
                fid.close()
        print("Analysis of {:s} completed...".format(self._def_dir))

    def _readstls(self):
        "Read in STL files if self.surfaces is True"
        for fname in sorted(os.listdir(self._ref_dir)):
            if '.stl' in fname.lower():
                reader = vtk.vtkSTLReader()
                reader.SetFileName(
                    str(os.path.normpath(self._ref_dir + os.sep + fname)))
                reader.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputConnection(reader.GetOutputPort())
                triangles.Update()
                self.rsurfs.append(triangles.GetOutput())
                self._make3Dmesh(
                    str(os.path.normpath(self._ref_dir + os.sep + fname)))

        for fname in sorted(os.listdir(self._def_dir)):
            if '.stl' in fname.lower():
                reader = vtk.vtkSTLReader()
                reader.SetFileName(
                    str(os.path.normpath(self._def_dir + os.sep + fname)))
                reader.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputConnection(reader.GetOutputPort())
                triangles.Update()
                self.dsurfs.append(triangles.GetOutput())

    def _deform(self):
        for i in xrange(len(self.rcentroids)):
            # volumetric strains
            self.vstrains.append(self.dvols[i] / self.rvols[i] - 1)

            ICP = vtk.vtkIterativeClosestPointTransform()
            rcopy = vtk.vtkPolyData()
            dcopy = vtk.vtkPolyData()
            rcopy.DeepCopy(self.rsurfs[i])
            dcopy.DeepCopy(self.dsurfs[i])
            ICP.SetSource(rcopy)
            ICP.SetTarget(dcopy)
            if self.rigidInitial:
                ICP.GetLandmarkTransform().SetModeToRigidBody()
                ICP.SetMaximumMeanDistance(0.001)
                ICP.SetCheckMeanDistance(1)
                ICP.SetMaximumNumberOfIterations(5000)
                ICP.StartByMatchingCentroidsOn()
                ICP.Update()
                trans = vtk.vtkTransform()
                trans.SetMatrix(ICP.GetMatrix())
                trans.Update()
                self.rigidTransforms.append(trans)
                rot = vtk.vtkTransformPolyDataFilter()
                rot.SetInputData(rcopy)
                rot.SetTransform(trans)
                rot.Update()
                ICP.GetLandmarkTransform().SetModeToAffine()
                ICP.SetSource(rot.GetOutput())
                ICP.Update()
            else:
                ICP.GetLandmarkTransform().SetModeToAffine()
                ICP.SetMaximumMeanDistance(0.001)
                ICP.SetCheckMeanDistance(1)
                ICP.SetMaximumNumberOfIterations(5000)
                ICP.StartByMatchingCentroidsOn()
                ICP.Update()

            F = np.zeros((3, 3), float)
            for j in xrange(3):
                for k in xrange(3):
                    F[j, k] = ICP.GetMatrix().GetElement(j, k)
            E = 0.5 * (np.dot(F.T, F) - np.eye(3))
            self.cell_strains.append(E)

    def deformableRegistration(self):
        for r, region in enumerate(self.rbbox):
            print("Performing deformable image registration for object {:d}"
                  .format(r + 1))
            rimg, dimg = self._poly2img(self.deformableSettings['Precision'],
                                        r)
            dimg.SetOrigin(rimg.GetOrigin())

            rimg = sitk.AntiAliasBinary(rimg)
            dimg = sitk.AntiAliasBinary(dimg)

            sitk.WriteImage(rimg, "reference{:d}.nii".format(r))
            sitk.WriteImage(dimg, "deformed{:d}.nii".format(r))

            #peform the deformable registration
            register = sitk.FastSymmetricForcesDemonsRegistrationFilter()
            register.SetNumberOfIterations(
                self.deformableSettings['Iterations'])
            register.SetMaximumRMSError(self.deformableSettings['Maximum RMS'])
            register.SmoothDisplacementFieldOn()
            register.SetStandardDeviations(
                self.deformableSettings['Displacement Smoothing'])
            register.SmoothUpdateFieldOff()
            register.UseImageSpacingOn()
            register.SetMaximumUpdateStepLength(2.0)
            register.SetUseGradientType(0)
            disp_field = register.Execute(rimg, dimg)
            print("...Elapsed iterations: {:d}"
                  .format(register.GetElapsedIterations()))
            print("...Change in RMS error: {:6.3f}"
                  .format(register.GetRMSChange()))

            #translate displacement field to VTK regular grid
            a = sitk.GetArrayFromImage(disp_field)
            disp = vtk.vtkImageData()
            disp.SetOrigin(rimg.GetOrigin())
            disp.SetSpacing(rimg.GetSpacing())
            disp.SetDimensions(rimg.GetSize())
            arr = vtk.vtkDoubleArray()
            arr.SetNumberOfComponents(3)
            arr.SetNumberOfTuples(disp.GetNumberOfPoints())
            #flatten array for translation to VTK
            data1 = np.ravel(a[:, :, :, 0])
            data2 = np.ravel(a[:, :, :, 1])
            data3 = np.ravel(a[:, :, :, 2])
            for i in xrange(disp.GetNumberOfPoints()):
                arr.SetTuple3(i, data1[i], data2[i], data3[i])
            disp.GetPointData().SetVectors(arr)

            #calculate the strain from displacement field
            getStrain = vtk.vtkCellDerivatives()
            getStrain.SetInputData(disp)
            getStrain.SetTensorModeToComputeStrain()
            getStrain.Update()
            #add the strain tensor to the displacement field structured grid
            strains = getStrain.GetOutput()
            c2p = vtk.vtkCellDataToPointData()
            c2p.PassCellDataOff()
            c2p.SetInputData(strains)
            c2p.Update()
            disp = c2p.GetOutput()

            #use VTK probe filter to interpolate displacements and strains
            #to 3D meshes of cells and save as UnstructuredGrid (.vtu)
            # to visualize in ParaView; this is a linear interpolation
            print("...Interpolating displacements to 3D mesh.")
            if self.rigidInitial:
                #transform 3D Mesh
                tf = vtk.vtkTransformFilter()
                tf.SetInputData(self.rmeshes[r])
                tf.SetTransform(self.rigidTransforms[r])
                tf.Update()
                rmesh = tf.GetOutput()
            else:
                rmesh = self.rmeshes[r]
            probe = vtk.vtkProbeFilter()
            probe.SetInputData(rmesh)
            probe.SetSourceData(disp)
            probe.Update()
            field = probe.GetOutput()

            self.cell_fields.append(field)
            if self.saveFEA:
                idisp = field.GetPointData().GetVectors()
                bcs = np.zeros((len(self._snodes[r]), 3), float)
                for j, node in enumerate(self._snodes[r]):
                    d = idisp.GetTuple3(node - 1)
                    bcs[j, 0] = d[0]
                    bcs[j, 1] = d[1]
                    bcs[j, 2] = d[2]
                self._bcs.append(bcs)
            idWriter = vtk.vtkXMLUnstructuredGridWriter()
            idWriter.SetFileName(
                str(os.path.normpath(self._def_dir + '/cell{:02d}.vtu'
                                     .format(r + 1))))
            idWriter.SetInputData(self.cell_fields[r])
            idWriter.Write()
        print("Registration completed.")

    def _getECMstrain(self):
        #get the ECM strain
        rc = np.array(self.rcentroids)
        dc = np.array(self.dcentroids)
        if rc.shape[0] < 4:
            print(("WARNING: There are less than 4 objects in the space; "
                   "therefore, tissue strain was not calculated."))
            return
        da = vti.numpy_to_vtk(rc)
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
        mq = vti.vtk_to_numpy(mq)
        try:
            #tet with edge ratio closest to 1
            btet = np.argmin(abs(mq - 1.0))
        except:
            return
        idlist = tet.GetOutput().GetCell(btet).GetPointIds()
        P = np.zeros((4, 3), float)
        p = np.zeros((4, 3), float)
        for i in xrange(idlist.GetNumberOfIds()):
            P[i, :] = rc[idlist.GetId(i), :]
            p[i, :] = dc[idlist.GetId(i), :]
        X = np.array([P[1, :] - P[0, :],
                      P[2, :] - P[0, :],
                      P[3, :] - P[0, :],
                      P[3, :] - P[1, :],
                      P[3, :] - P[2, :],
                      P[2, :] - P[1, :]], float)

        x = np.array([p[1, :] - p[0, :],
                      p[2, :] - p[0, :],
                      p[3, :] - p[0, :],
                      p[3, :] - p[1, :],
                      p[3, :] - p[2, :],
                      p[2, :] - p[1, :]], float)

        #assemble the system
        dX = np.zeros((6, 6), float)
        ds = np.zeros((6, 1), float)
        for i in xrange(6):
            dX[i, 0] = 2 * X[i, 0] ** 2
            dX[i, 1] = 2 * X[i, 1] ** 2
            dX[i, 2] = 2 * X[i, 2] ** 2
            dX[i, 3] = 4 * X[i, 0] * X[i, 1]
            dX[i, 4] = 4 * X[i, 0] * X[i, 2]
            dX[i, 5] = 4 * X[i, 1] * X[i, 2]

            ds[i, 0] = np.linalg.norm(
                x[i, :]) ** 2 - np.linalg.norm(X[i, :]) ** 2

        E = np.linalg.solve(dX, ds)
        E = np.array([[E[0, 0], E[3, 0], E[4, 0]],
                      [E[3, 0], E[1, 0], E[5, 0]],
                      [E[4, 0], E[5, 0], E[2, 0]]], float)
        self.ecm_strain = E

    def getDimensions(self):
        labelstats = self._getLabelShape(self.rlabels)
        self.rvols = labelstats['volume']
        self.rcentroids = labelstats['centroid']
        self.raxes = labelstats['ellipsoid diameters']
        self.rbbox = labelstats['bounding box']
        labelstats = self._getLabelShape(self.dlabels)
        self.dvols = labelstats['volume']
        self.dcentroids = labelstats['centroid']
        self.daxes = labelstats['ellipsoid diameters']
        self.dbbox = labelstats['bounding box']

    def _getLabelShape(self, img):
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
            labelshape['ellipsoid diameters'].append(
                ls.GetEquivalentEllipsoidDiameter(l))
            labelshape['bounding box'].append(ls.GetBoundingBox(l))
        return labelshape

    def _make3Dmesh(self, filename):
        s = MeshInfo()
        s.load_stl(filename)
        #use TETGEN to generate mesh
        #switches:
        # p -
        # q - refine mesh to improve quality
        #     1.2 minimum edge ratio
        #     minangle=15
        # Y - do not edit surface mesh
        # O - perform mesh optimization
        #     optlevel=9
        mesh = build(s, options=Options("pq1.2YO",
                                        optlevel=9,
                                        minangle=15,
                                        optpasses=10))
        elements = list(mesh.elements)
        nodes = list(mesh.points)
        faces = np.array(mesh.faces)
        s_nodes = list(np.unique(np.ravel(faces)))
        tetraPoints = vtk.vtkPoints()
        tetraPoints.SetNumberOfPoints(len(nodes))
        for i, p in enumerate(nodes):
            tetraPoints.InsertPoint(i, p[0], p[1], p[2])
        tetraElements = []
        for i, e in enumerate(elements):
            tetraElements.append(vtk.vtkTetra())
            for j in xrange(4):
                tetraElements[i].GetPointIds().SetId(j, e[j] - 1)
        vtkMesh = vtk.vtkUnstructuredGrid()
        vtkMesh.Allocate(i + 1, i + 1)
        for i, e in enumerate(tetraElements):
            vtkMesh.InsertNextCell(e.GetCellType(), e.GetPointIds())
        vtkMesh.SetPoints(tetraPoints)
        self.rmeshes.append(vtkMesh)
        self._snodes.append(s_nodes)
        self._elements.append(elements)
        self._nodes.append(nodes)

    def _poly2img(self, res, ind):
        dim = int(np.ceil(1.0 / res)) + 6
        rpoly = vtk.vtkPolyData()
        rpoly.DeepCopy(self.rsurfs[ind])
        dpoly = self.dsurfs[ind]
        if self.rigidInitial:
            rot = vtk.vtkTransformPolyDataFilter()
            rot.SetInputData(rpoly)
            rot.SetTransform(self.rigidTransforms[ind])
            rot.Update()
            rpoly = rot.GetOutput()

        rbounds = np.zeros(6, np.float32)
        dbounds = np.copy(rbounds)

        rpoly.GetBounds(rbounds)
        dpoly.GetBounds(dbounds)

        spacing = np.zeros(3, np.float32)
        for i in xrange(3):
            rspan = rbounds[2 * i + 1] - rbounds[2 * i]
            dspan = dbounds[2 * i + 1] - dbounds[2 * i]
            spacing[i] = np.max([rspan, dspan]) * res

        imgs = []
        for i in xrange(2):
            arr = np.ones((dim, dim, dim), np.uint8) * 255
            arr2img = vti.vtkImageImportFromArray()
            arr2img.SetDataSpacing(spacing)
            arr2img.SetDataExtent((0, dim - 1, 0, dim - 1, 0, dim - 1))
            arr2img.SetArray(arr)
            arr2img.Update()
            if i == 0:
                rimg = arr2img.GetOutput()
                rimg.SetOrigin((rbounds[0] - spacing[0] * 3,
                                rbounds[2] - spacing[1] * 3,
                                rbounds[4] - spacing[2] * 3))
            else:
                dimg = arr2img.GetOutput()
                dimg.SetOrigin((dbounds[0] - spacing[0] * 3,
                                dbounds[2] - spacing[1] * 3,
                                dbounds[4] - spacing[2] * 3))
        imgs = []
        for (pd, img) in [(rpoly, rimg), (dpoly, dimg)]:
            pol2stenc = vtk.vtkPolyDataToImageStencil()
            pol2stenc.SetInputData(pd)
            pol2stenc.SetOutputOrigin(img.GetOrigin())
            pol2stenc.SetOutputSpacing(img.GetSpacing())
            pol2stenc.SetOutputWholeExtent(img.GetExtent())
            pol2stenc.Update()

            imgstenc = vtk.vtkImageStencil()
            imgstenc.SetInputData(img)
            imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
            imgstenc.ReverseStencilOff()
            imgstenc.SetBackgroundValue(0)
            imgstenc.Update()

            arr = vtk_to_numpy(imgstenc.GetOutput().GetPointData().GetArray(0))
            arr = arr.reshape(dim, dim, dim)
            itk_img = sitk.GetImageFromArray(arr)
            itk_img.SetSpacing(img.GetSpacing())
            itk_img.SetOrigin(img.GetOrigin())
            imgs.append(itk_img)

        return (imgs[0], imgs[1])
