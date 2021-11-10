import fnmatch

from .Helpers import FixedDict
import vtk
import SimpleITK as sitk
import numpy as np

class ObjectPairs(object):
    def __init__(self, reference=None, deformed=None, identifier=None):
        self.reference = reference
        self.deformed = deformed
        self.results = FixedDict({"Ellipsoidal": FixedDict({"Major Axis Length": [],
                                                            "Major Axis Direction": [],
                                                            "Major Axis Strain": [],
                                                            "Middle Axis Length": [],
                                                            "Middle Axis Direction": [],
                                                            "Middle Axis Strain": [],
                                                            "Minor Axis Length": [],
                                                            "Minor Axis Direction": [],
                                                            "Minor Axis Strain": [],
                                                            "Volumetric Strain": [],
                                                            "Volumetric Strain Error": []}),
                                  "Affine": FixedDict({"Transform": [],
                                                       "Deformation Gradient": [],
                                                       "Strain": [],
                                                       "Principal Strain 1": [],
                                                       "Principal Strain 1 Direction": [],
                                                       "Principal Strain 2": [],
                                                       "Principal Strain 2 Direction": [],
                                                       "Principal Strain 3": [],
                                                       "Principal Strain 3 Direction": [],
                                                       "Maximum Shear Strain": [],
                                                       "Volumetric Strain": [],
                                                       "Volumetric Strain Error": []}),
                                  "DiffeomorphicDemons": FixedDict({"Transform": [],
                                                                    "Displacement": [],
                                                                    "Deformation Gradient": [],
                                                                    "Strain": [],
                                                                    "Principal Strain 1": [],
                                                                    "Principal Strain 1 Direction": [],
                                                                    "Principal Strain 2": [],
                                                                    "Principal Strain 2 Direction": [],
                                                                    "Principal Strain 3": [],
                                                                    "Principal Strain 3 Direction": [],
                                                                    "Maximum Shear Strain": [],
                                                                    "Volumetric Strain": [],
                                                                    "Volumetric Strain Error": []}),
                                  "Volumetric Strain": []})

    @property
    def reference(self):
        return self.__reference

    @reference.setter
    def reference(self, reference):
        if reference is None:
            raise AttributeError('A reference surface(s) needs to be defined.')
        elif os.path.isfile(reference):
            reference = self.__parseSurfaceFile(reference)
        elif os.path.isdir(reference):
            reference = self.__parseSurfaceDirectory(reference)
        elif isinstance(reference, list):
            for r in reference:
                if os.path.isfile(r):
                    r = self.__parseSurfaceFile(r)
                elif isinstance(r, vtk.vtkPolyData):
                    pass
                else:
                    raise ValueError('Unsupported type {} detected in reference list.'.format(type(r)))
        elif isinstance(reference, vtk.vtkPolyData):
            pass
        self.__reference = reference

    @property
    def deformed(self, deformed):
        return self.__deformed

    @deformed.setter
    def deformed(self, deformed):
        if deformed is None:
            raise AttributeError('A deformed surface(s) needs to be defined.')
        elif os.path.isfile(deformed):
            deformed = [self.__parseSurfaceFile(deformed)]
        elif os.path.isdir(deformed):
            deformed = self.__parseSurfaceDirectory(deformed)
        elif isinstance(deformed, list):
            for d in deformed:
                if os.path.isfile(d):
                    d = self.__parseSurfaceFile(d)
                elif isinstance(r, vtk.vtkPolyData):
                    pass
                else:
                    raise ValueError('Unsupported type {} detected in deformed list.'.format(type(d)))
        elif isinstance(deformed, vtk.vtkPolyData):
            pass
        self.__deformed = deformed

    def __parseSurfaceFile(self, f):
        if f.lower().endswith('vtk'):
            reader = vtk.vtkPolyDataReader()
        elif f.lower().endswith('vtp'):
            reader = vtk.vtkXMLPolyDataReader()
        elif f.lower().endswith('stl'):
            reader = vtk.vtkSTLReader()
        else:
            print("::WARNING:: File extension not recognized. Attempting to read as a VTP.")
            reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f)
        return [reader.GetOutput()]

    def __parseSurfaceDirectory(self, d):
        files = sorted(os.listdir(d))
        stlfiles = fnmatch.filter(files, '*.stl')
        vtkfiles = fnmatch.filter(files, '*.vtk')
        vtpfiles = fnmatch.filter(files, '*.vtp')
        filenumber = [len(i) for i in (vtpfiles, vtkfiles, stlfiles)]
        if sum(i > 0 for i in filenumber) > 1:
            greatest = np.argmax(filenumber)
            if greatest == 0:
                print(("::WARNING::  Multiple supported filetypes detected in this directory."
                       " Since there are more or an equal number of VTP files, these will be read."))
                reader = vtk.vtkXMLPolyDataReader()
                files = vtpfiles
            elif greatest == 1:
                print(("::WARNING::  Multiple supported filetypes detected in this directory."
                       " Since there are more or an equal number of VTK files, these will be read."))
                reader = vtk.vtkPolyDataReader()
                files = vtkfiles
            elif greatest == 2:
                print(("::WARNING::  Multiple supported filetypes detected in this directory."
                       " Since there are more or an equal number of STL files, these will be read."))
                reader = vtk.vtkSTLReader()
                files = vtkfiles
        surfaces = []
        for f in files:
            reader.SetFileName(f)
            surfaces.append(reader.GetOutput())
        return surfaces

    def getDeformationByIterativeClosestPoint():
        icp = vtk.vtkIterativeClosestPointTransform()
        for (r, d) in zip(self.__reference, self.__deformed):
            icp.SetSource(r)
            icp.SetTarget(d)
            icp.GetLandmarkTransform().SetModeToAffine()
            icp.SetMaximumMeanDistance(0.001)
            icp.SetCheckMeanDistance(1)
            icp.SetMaximumNumberOfIterations(10000)
            icp.StartByMatchingCentroidsOn()
            icp.Update()

            F = np.zeros((3,3), float)
            for i in range(3):
                for j in range(3):
                    F[i, j] = icp.GetMatrix().GetElement(i, j)
            E = 0.5 * (np.dot(F.T), F - np.eye(3))
            self.results["Strain"].append(np.array([E]))
            self.results["Transform"].append(icp.GetTransform())
            self.__getPrincipalStrains()

    def getDeformationByEllipsoidalMethod():
        pass

    def getDeformationByDiffeomorpicDemons():
        pass

    def __getPrincipalStrains():
        l, e = np.linalg.eigh(self.results["Strain"][-1])

    def saveAsNumpy():
        pass

    def saveAsUnstructuredGrid():
        pass

    def saveAsPolyData():
        pass
