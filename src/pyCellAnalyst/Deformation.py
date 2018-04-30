import fnmatch

from .Helpers import FixedDict
import vtk
import SimpleITK as sitk

class ObjectPairs(object):
    def __init__(self, reference=None, deformed=None, identifier=None):
        self.reference = reference
        self.deformed = deformed

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
        elif f.lower().endswith('stl'):
            reader = vtk.vtkSTLReader()
        else:
            print("::WARNING:: File extension not recognized. Attempting to read as an STL.")
            reader = vtk.vtkSTLReader()
        reader.SetFileName(f)
        return reader.GetOutput()

    def __parseSurfaceDirectory(self, d):
        files = sorted(os.listdir(d))
        stlfiles = fnmatch.filter(files, '*.stl')
        vtkfiles = fnmatch.filter(files, '*.vtk')
        if len(stlfiles) > 0 and len(vtkfiles) > 0:
            if len(stlfiles) >= len(vtkfiles):
                print(("::WARNING:: Both STL and VTK files detected in this directory."
                       " Since there are more or an equal number of STL files, these will be read."))
                reader = vtk.vtkSTLReader()
                files = stlfiles
            elif len(vtkfiles) > len(stlfiles):
                print(("::WARNING:: Both STL and VTK files detected in this directory."
                       " Since there are more VTK files, these will be read."))
                reader = vtk.vtkPolyDataReader()
                files = vtkfiles
        surfaces = []
        for f in files:
            reader.SetFileName(f)
            surfaces.append(reader.GetOutput())

class AffineDeformation(ObjectPairs):
    pass

class PrincipalMomentsOfInertia(ObjectPairs):
    pass

class DiffeomorphicDemons(ObjectPairs):
    pass
