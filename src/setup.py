from distutils.core import setup

setup(
    name = 'pyCellAnalyst',
    version = '1.0.0',
    package = ['pyCellAnalyst'],
    py_modules = ['pyCellAnalyst.__init__','pyCellAnalyst.Volume','pyCellAnalyst.CellMech'],
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'MIT',
    description = 'Reconstructs STLs for cells from 3D TIFF image data (may later be extended) and calculates geometric and mechanical metrics',
) 
