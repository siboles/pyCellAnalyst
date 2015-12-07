import subprocess
from setuptools import setup
from setuptools.dist import Distribution
from codecs import open
import os
import platform
import string

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(os.path.join(here, os.path.pardir), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False
if "windows" in platform.system().lower():
    subprocess.call(["pip", "install", "-r", os.path.join(here,"requirements_win.txt")])
else:
    subprocess.call(["pip", "install", "-r", os.path.join(here,"requirements.txt")])
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
    distclass=BinaryDistribution,
    #download_url = "https://github.com/siboles/pyCellAnalyst/tarball/1.0.2",
)
