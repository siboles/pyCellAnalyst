## Welcome ##

pyCellAnalyst is free, open-source software aimed at reconstructing biological objects from 3D image data under different mechanical conditions, and then calculating the deformation between those conditions. Its initial application was for data obtained from multi-photon laser scanning microscopy of chondrocytes within articular cartilage. That said, the software may be applied to many other data types. Currently, the supported input formats of 3D image are either a sequence of TIFF or [NifTi][1]; however, the extension to other formats should be straightforward and can be implemented upon request.

## Installation ##

**Source Code**

pyCellAnalyst is hosted and maintained at Github (which is linked to this project). The necessary source code for the module can be either cloned using git or can be downloaded as a .zip file. A working installation of Python is required for these software. At this point, it is preferable that the user installs the default [Python][2] as opposed to a packaged distribution such as Enthought or Anaconda.

**Windows Users:** It is recommended to add the directory containing python.exe to your system path. The install Wizard can do this, but by default the option is not active. Either activate this during the installation or add the directory containing python.exe to your System Environment Variable *Path*.

**Dependencies**

A number of dependencies are also required. After Python is installed, the user is encouraged to make use of the automated scripts for installing on

Windows

    ../pyCellAnalyst/windows_installer.py

Linux

    ../pyCellAnalyst/linux_installer.py
    
by simply navigating to this directory and typing

Windows

    python windows_installer.py

Linux

    sudo python linux_installer.py

**NOTE:** Since pyCellAnalyst currently requires the beta version of SimpleITK 0.9.0 there is no support for 64 bit Windows. The windows_installer.py script will create a functional build if run from a [32-bit Python Interpreter][3]. If 64 bit Python in Windows is absolutely necessay, the user will currently have to compile SimpleITK themself.

**LINUX USERS PLEASE READ:** For system safety this installer does not link the libraries for VTK. To do this:

**CAUTION:** This can cause irrevokable damage to your system if done incorrectly. If you are not on Ubuntu 14 or later please consult ldconfig for whatever flavour of OS you are running.

On Ubuntu 14 or later (other flavours may differ) in a command terminal:

    cd /etc/ld.so.conf.d
    sudo echo "/usr/lib/VTK-6.2.0-Linux-64bit/lib" > vtk.conf
    sudo ldconfig

Alternatively, packages can be individually retrieved and installed. The list of dependencies is:

 - [Visualization Toolkit 6.1.0+ (VTK)][4] 
 - [Simple Insight Toolkit 0.9.0+ (SimpleITK)][5]
 - [SciPy][6]
 - [Numpy][7]
 - [scikit-learn][8]
 - [pip][9]
 - [setuptools][10]
 - [matplotlib][11]
 - [MeshPy][12]

Windows users: Unofficial builds of all of these except SimpleITK are provided here: [32-bit][13] and [64-bit][14]. Thank you to [Christoph Gohlke][15] for providing these builds.

## Setup ##

pyCellAnalyst uses distutils for easy setup. Simply type in directory ../pyCellAnalyst/src/ on

Windows

    python setup.py install
    
Linux

    sudo python setup.py install

## Utilities ##

To ease usage, some Graphical User Interface (GUI) utilities are provided in: ../pyCellAnalyst/src/utilities/

 - CellSegmentGUI.py: the main GUI for segmentation and deformation analysis
 - FEBioAnalysis.py:  if deformable registration is performed and data are saved for finite element analysis, this GUI can be used to automatically generate, solve, and post-process models in the open-source software, FEBio. The additional module, pyFEBio, is needed for this.


  [1]: http://nifti.nimh.nih.gov/nifti-1
  [2]: http://www.python.org
  [3]: https://www.python.org/ftp/python/2.7.9/python-2.7.9.msi
  [4]: http://www.vtk.org/download/
  [5]: http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/
  [6]: http://www.scipy.org/
  [7]: http://www.numpy.org/
  [8]: http://scikit-learn.org/stable/
  [9]: https://pypi.python.org/pypi/pip
  [10]: https://pypi.python.org/pypi/setuptools
  [11]: http://matplotlib.org/
  [12]: https://pypi.python.org/pypi/MeshPy
  [13]: https://osf.io/6ihzk/
  [14]: https://osf.io/h3tcu/
  [15]: http://www.lfd.uci.edu/~gohlke/pythonlibs/
