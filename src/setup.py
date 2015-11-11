from distutils.core import setup

setup(
    name = 'pyCellAnalyst',
    version = '1.0.1',
    package = ['pyCellAnalyst'],
    py_modules = ['pyCellAnalyst.__init__','pyCellAnalyst.Volume','pyCellAnalyst.CellMech', 'pyCellAnalyst.GUI', 'pyCellAnalyst.FEA_GUI'],
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'MIT',
    description = 'An extensive module for image processing, segmentation, and deformation analysis. Initially aimed at processing 3-D microscopy of cells, this may have applications for other data types as well.',
    url = "https://github.com/siboles/pyCellAnalyst",
    download_url = "https://github.com/siboles/pyCellAnalyst/tarball/1.0.1",
    install_requires = [
        "MeshPy==2014.1",
        "SimpleITK==0.9.1",
        "VTK==5.8.0",
        "febio==0.1.2",
        "numpy==1.8.2",
        "scikit-learn==0.15.0",
        "scipy==0.13.3",
        "wquantiles==0.3",
        "xlrd==0.9.3",
        "xlwt==0.7.5",
    ],
) 
