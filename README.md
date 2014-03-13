pyCellAnalyst
=============

Reconstructs STLs for cells from 3D TIFF image data (may later be extended) and calculates geometric and mechanical metrics

Installation:
=============
On Linux systems:

    in directory containing setup.py type: sudo python setup.py install
    
    This will install under your default python2.7 dist-packages folder e.g. /usr/lib/python2.7/dist-pakages/

    To see other options type

    python setup.py --help

On Windows systems:

    in directory containing setup.py type: python setup.py install

Third Party Software:
=====================
To use this module you will need Visual Toolkit 6.1.0 (previously releases untested) and its Python wrappers.  On Linux, the VTK site-packages directory should be appended to your PYTHONPATH.  Likewise you'll need to add the VTK lib directory to LD_LIBRARY_PATH.  How to do this varies by Linux distribution.

On Windows, you'll need to add VTK lib to your PATH system environment variable and VTK site-packages to your PYTHONPATH system environment variable.

NumPy, SciPy, and the Python Imaging Library (PIL) are also required. 
