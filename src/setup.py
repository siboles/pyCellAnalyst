import subprocess
from setuptools import setup
from setuptools.dist import Distribution
from codecs import open
import os
import platform
import string
import fnmatch

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(os.path.join(here, os.path.pardir), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False
if "windows" in platform.system().lower():
    subprocess.call(["pip", "install", "numpy", "--index-url", "https://webdisk.ucalgary.ca/~ssibole/public_html/"])
    subprocess.call(["pip", "install", "-r", os.path.join(here,"requirements_win.txt")])
    if os.getenv("VIRTUAL_ENV") is not None:
        if os.getenv("TCL_LIBRARY") is None:
            path = os.getenv("PATH").lower().split(";")
            pythonpath = min(fnmatch.filter(path, "*python2*"), key=len)
            tcldir = os.path.join(pythonpath, "tcl")
            dirs = [name for name in os.listdir(tcldir) if os.path.isdir(os.path.join(tcldir, name))]
            tcldir = os.path.join(tcldir, fnmatch.filter(dirs, "tcl8.*")[0])
            subprocess.call("echo set TCL_LIBRARY={:s} >> {:s}".format(tcldir,
                                                                       os.path.join(os.getenv("VIRTUAL_ENV"),
                                                                                    "Scripts",
                                                                                    "activate.bat")), shell=True)
else:
    subprocess.call(["pip", "install", "-r", os.path.join(here,"requirements.txt")])
setup(
    name = 'pyCellAnalyst',
    version = 'v1.0.3b',
    description = 'An extensive module for image processing, segmentation, and deformation analysis. Initially aimed at processing 3-D microscopy of cells, this may have applications for other data types as well.',
    packages = ['pyCellAnalyst'],
    long_description = long_description,
    url = "https://github.com/siboles/pyCellAnalyst",
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'MIT',
    py_modules = ['pyCellAnalyst.__init__','pyCellAnalyst.Volume','pyCellAnalyst.CellMech', 'pyCellAnalyst.GUI', 'pyCellAnalyst.FEA_GUI'],
    distclass=BinaryDistribution,
    #download_url = "https://github.com/siboles/pyCellAnalyst/tarball/1.0.2",
)
