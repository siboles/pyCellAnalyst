from setuptools import setup
from codecs import open
import os
import shutil
import urllib
import platform
import string
import site
import subprocess
import shlex
import fnmatch
import zipfile

print("Thank you for choosing to install pyCellAnalyst!\n")
print("For safety and stability, it is recommmended that pyCellAnalyst is installed\nin a Python virtual environment.\n")
print("Please verify that you are in the virtual environment you wish to install to.\nThe name will appear in parentheses at the beginning of your command prompt\ne.g. (pyCellAnalyst).\n")

print("Please consult the Installation section of our documentation\nfor a guide on how to create virtual environments.\n")

print("To switch to a virtual environment type: workon ENVIRONMENT_NAME\ne.g. workon pyCellAnalyst\n")

proceed = raw_input("Do you wish to proceed (y/n)?") or "n"

if proceed.lower() == "n":
    print("\nGoodbye!\n")
    raise SystemExit
print("\nWonderful! Here we go...\n")

# handle mess of dependencies
cached_win_wheels = (("https://github.com/siboles/pyCellAnalyst/raw/master/cached_binaries/numpy-1.9.3+mkl-cp27-none-win32.whl", "numpy-1.9.3+mkl-cp27-none-win32.whl"),
                     ("https://github.com/siboles/pyCellAnalyst/raw/master/cached_binaries/scipy-0.16.1-cp27-none-win32.whl","scipy-0.16.1-cp27-none-win32.whl"),
                     ("https://github.com/siboles/pyCellAnalyst/raw/master/cached_binaries/PLATLIB.zip","PLATLIB.zip"),
                     ("https://github.com/siboles/pyCellAnalyst/raw/master/cached_binaries/MeshPy-2014.1-cp27-none-win32.whl","MeshPy-2014.1-cp27-none-win32.whl"))

if "windows" in platform.system().lower():
    path = os.getenv('PATH')
    if not "python27" in path.lower():
        raise SystemExit("***Exiting***\npython.exe is not in system path.\nPlease add the location of python.exe to your PATH environment variable and rerun.")

    venv = os.getenv('VIRTUAL_ENV')
    if venv is not None:
        tcllib = os.getenv('TCL_LIBRARY')
        if tcllib is None:
            path = path.lower().split(";")
            pythonstr = min(fnmatch.filter(path, "*python*"), key=len)
            tcldir = os.path.join(pythonstr, "tcl")
            dirs = [name for name in os.listdir(tcldir) if os.path.isdir(os.path.join(tcldir, name))]
            tcldir = fnmatch.filter(dirs, "tcl8.*")[0]
            activatepath = os.path.join(venv, "Scripts")
            subprocess.call("echo set TCL_LIBRARY={:s} >> {:s}{:s}activate.bat".format(
                tcldir, activatepath, os.sep))

if not "windows" in platform.system().lower():
    f = open("requirements.txt", "r")
    for l in f.readlines():
        if "SimpleITK" in l:
            subprocess.call("easy_install {:s}".format(l), shell=True)
        else:
            subprocess.call("pip install {:s}".format(l), shell=True)
else:
    #Install cached wheels for things that require compilation (numpy, scipy, matplotlib, MeshPy)
    modules = ("numpy", "scipy", "matplotlib", "meshpy.tet")
    for i, (l, f) in enumerate(cached_win_wheels):
            try:
                eval("import {:s}".format(modules[i]))
            except:
                print("Downloading {:s}...".format(f))
                urllib.urlretrieve(l, f)
            if modules[i] == "matplotlib":
                zfile = zipfile.ZipFile(f)
                if venv is None:
                    sitedir = os.path.join(pythonstr, "Lib", "site-packages")
                    zfile.extractall(sitedir, zfile.namelist()[1:])
                else:
                    sitedir = os.path.join(venv, "Lib", "site-packages")
                    zfile.extractall(sitedir, zfile.namelist()[1:])
            else:
                subprocess.call("pip install {:s}".format(f), shell=True)
                os.remove(f)

    f = open("requirements.txt", "r")
    for l in f.readlines():
        if "SimpleITK" in l:
            subprocess.call("easy_install {:s}".format(l), shell=True)
        elif any([s in l.lower() for s in ("numpy", "scipy", "matplotlib", "meshpy")]):
            continue
        else:
            subprocess.call("pip install {:s}".format(l), shell=True)

here = os.path.abspath(os.path.dirname(__file__))
final_messages = []

with open(os.path.join(os.path.join(here, os.path.pardir), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

try:
    import vtk
    vers = vtk.VTK_VERSION
    if float(vers[0:3]) < 6.1:
        print "Outdated version of VTK detected. Downloading version 6.3.0..."
        if "windows" in platform.system().lower():
            urllib.urlretrieve("https://github.com/siboles/pyCellAnalyst/blob/master/cached_binaries/VTK-6.2.0-cp27-none-win32.whl","VTK-6.2.0-cp27-none-win32.whl")
            subprocess.call("pip install VTK-6.2.0-cp27-none-win32.whl", shell=True)
            os.remove("VTK-6.2.0-cp27-none-win32.whl")
        elif "linux" in platform.system().lower():
            urllib.urlretrieve("http://www.vtk.org/files/release/6.3/vtkpython-6.3.0-Linux-64bit.tar.gz",
                               "vtk_python.tar.gz")
            subprocess.call(shlex.split("tar -zxf vtk_python.tar.gz"))
            subprocess.call("mv VTK-6.3.0-Linux-64bit/lib/python2.7/site-packages/vtk $VIRTUAL_ENV/lib/python2.7/site-packages", shell=True)
            subprocess.call("mv VTK-6.3.0-Linux-64bit/lib/* $VIRTUAL_ENV/lib/", shell=True)
            os.remove("vtk_python.tar.gz")
            subprocess.call(shlex.split("rm -r VTK-6.3.0-Linux-64bit"))
            final_messages.append(("Downloaded and extracted VTK 6.3.0 to $VIRTUAL_ENV/lib/python2.7/site-packages/vtk\n"
                                   "... You'll need to link the libraries appropriately for your flavor."))

        elif "darwin" in platform.system().lower():
            urllib.urlretrieve("http://www.vtk.org/files/release/6.3/vtkpython-6.3.0-Darwin-64bit.dmg",
                               "vtk_python.dmg")
            final_messages.append("Downloaded a disk image for VTK. Install following instructions for .dmg on Mac...")
except ImportError:
    print("No version of VTK was detected. Downloading version 6.3.0...")
    if "windows" in platform.system().lower():
        urllib.urlretrieve("https://github.com/siboles/pyCellAnalyst/blob/master/cached_binaries/VTK-6.2.0-cp27-none-win32.whl","VTK-6.2.0-cp27-none-win32.whl")
        subprocess.call("pip install VTK-6.2.0-cp27-none-win32.whl", shell=True)
        os.remove("VTK-6.2.0-cp27-none-win32.whl")
    elif "linux" in platform.system().lower():
        urllib.urlretrieve("http://www.vtk.org/files/release/6.3/vtkpython-6.3.0-Linux-64bit.tar.gz",
        "vtk_python.tar.gz")

        subprocess.call(shlex.split("tar -zxf vtk_python.tar.gz"))
        subprocess.call("mv VTK-6.3.0-Linux-64bit/lib/python2.7/site-packages/vtk $VIRTUAL_ENV/lib/python2.7/site-packages", shell=True)
        subprocess.call("mv VTK-6.3.0-Linux-64bit/lib/* $VIRTUAL_ENV/lib/", shell=True)
        os.remove("vtk_python.tar.gz")
        subprocess.call(shlex.split("rm -r VTK-6.3.0-Linux-64bit"))
        final_messages.append(("Downloaded and extracted VTK 6.3.0 to $VIRTUAL_ENV/lib/python2.7/site-packages/vtk\n"
                               "... You'll need to link the libraries appropriately for your flavor."))

    elif "darwin" in platform.system().lower():
        urllib.urlretrieve("http://www.vtk.org/files/release/6.3/vtkpython-6.3.0-Darwin-64bit.dmg",
                           "vtk_python.dmg")
        final_messages.append("Downloaded a disk image for VTK. Install following instructions for .dmg on Mac...")

setup(
    name = 'pyCellAnalyst',
    version = '1.0.2',
    description = 'An extensive module for image processing, segmentation, and deformation analysis. Initially aimed at processing 3-D microscopy of cells, this may have applications for other data types as well.',
    packages = ['pyCellAnalyst'],
    long_description = long_description,
    url = "https://github.com/siboles/pyCellAnalyst",
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'MIT',

    py_modules = ['pyCellAnalyst.__init__','pyCellAnalyst.Volume','pyCellAnalyst.CellMech', 'pyCellAnalyst.GUI', 'pyCellAnalyst.FEA_GUI'],

    download_url = "https://github.com/siboles/pyCellAnalyst/tarball/1.0.2",
)

for message in final_messages:
    print(message+"\n")
