Installation
============

.. toctree::
   :maxdepth: 1
   :glob:

Anaconda
--------

pyCellAnalyst now requires `Anaconda <https://www.continuum.io/downloads>`_ Python from Continuum Analytics. You can install either the full Anaconda package or Miniconda. 

.. note::

   You must install the 64 bit 2.x version not the 3.x version.

Installing from Anaconda Cloud
------------------------------

1. Add additional channels to fetch packages from.

   .. code-block:: guess

      conda config --add channels conda-forge
      conda config --add channels salilab
      conda config --add channels SimpleITK
      conda config --add channels siboles

2. Install

   To install to your root conda environment:

   .. code-block:: guess

      conda install -c siboles pycellanalyst

   To install to a conda virtual environment for just pyCellAnalyst:

   .. code-block:: guess

      conda create --name pyCell pycellanalyst

Optional
--------

It is still possible to install pyCellAnalyst to a standard Python build. Although the dependencies must be resolved by the user. pyCellAnalyst depends on the following:

Available in PyPi

- numpy
- scipy
- scikit-learn
- matplotlib
- wquantiles
- xlrd
- xlwt
- trimesh 
- febio

Others:

- `VTK <http://www.vtk.org/download/>`_
- `SimpleITK <http://www.simpleitk.org/SimpleITK/resources/software.html>`_
- `tetmesh <https://github.com/siboles/tetmesh>`_

Troubleshooting
---------------

- If the Anaconda version of Python does not start when you type python in a command terminal you likely need to unset your PYTHONPATH and PYTHONHOME variables.
- Other problems with Anaconda can be researched `here <http://conda.pydata.org/docs/troubleshooting.html>`_
- If the pycellanalyst does not work with conda install on Linux, the user can build the package themself.
  - Download the `source code <https://github.com/siboles/pyCellAnalyst/archive/master.zip>`_
  - Unzip and open a command terminal in the *src* directory containing *build.sh*.
  - Type the following:

    .. code-block:: guess

       conda build .
       conda install --use-local pycellanalyst
