
Introduction
============

.. toctree::
   :maxdepth: 1
   :glob:

Object Segmentation
-------------------

An example usage is as follows:

.. code-block:: python

                from pyCellAnalyst import Volume

                # Instantiate a Volume object with default parameters
                # This will execute the segmentation and populate all class attributes
                # Note: PATH_TO_IMAGE_DIRECTORY must be changed to the location of the either
                # a directory containing a single sequence of 2-D TIFF or a single NiFTi format ".nii"
                # 3-D image. Likewise, each [int, int, int, int, int, int] must be replaced by the
                # position and size of regions of interest (as indices)
                # Please consult the pyCellAnalyst.Voluume class reference below to explore parameter options. 
                vol = Volume.Volume("PATH_TO_IMAGE_DIRECTORY", regions=[,[int, int, int, int, int, int],...])
                # print the volumes of each segmented object
                print vol.volumes

This code block will create a new directory "PATH_TO_IMAGE_DIRECTORY"+"_results" where the segmented object surfaces are saved in stereolithographic format (.stl). Also, a 3-D label image of the reconstructed objects "labels.nii" is also saved here.

Deformation Analysis
--------------------

1. The moment of inertia tensor, **I** is calculated discretely for each object in the reference and deformed states. An ellipsoid with the same principal moments of inertia, the eigenvalues of **I**, is then determined for each reference and deformed object. Deformation can then be characterized as the stretch along the ellipsoid axes between the reference and deformed states. This is analagous to the principal stretches (which of course can also be converted to strains); however, it is impossible to separate rigid body rotation from shear with this approach.
2. The optimal affine transformation mapping the each reference surface to its deformed pair can be calculated using an interative closest point minimization. The objective function to minimize is the sum of the Euclidean distances from the vertices of the reference surface after affine transformation to their nearest neighbor vertices on the deformed surface. By default a rigid body transformation will first be optimized by the same method, before attempting to dfind the best affine transformation. Assuming uniform deformation on the object, the linear transformation matrix from this optimal affine transformation is the deformation gradient, :math:`\mathbf{F}`. The Green-Lagrange strain tensor, :math:`\mathbf{E} = \frac{1}{2}(\mathbf{F}^T.\mathbf{F} - \mathbf{I})` is then calculated for each object pair.
3. Deformable image registration can be performed to determine the optimal diffeomorphism between the reference and deformed objects. This will yield a displacement vector field in the reference state that is then interpolated to the object vertices. Images are reconstructed from the object surfaces at a user-specified precision expressed as a ratio of the bounding box edge lengths e.g. 0.01 will result in 100 voxels in the :math:`i,\,j,` and :math:`k` directions with length of :math:`\frac{L_x}{100},\,\frac{L_y}{100},` and :math:`\frac{L_z}{100}`. 


To perform deformation analysis on these segmented objects:

.. code-block:: python

                from pyCellAnalyst import CellMech

                # Instantiate a CellMech object with default parameters
                # This will calculate object volumes, ellipsoids of equivalent principal moments of inertia,
                # and the optimal affine transformation between reference and deformed object pairs.
                # Please consult the pyCellAnalyst.CellMech class reference below to explore parameter options.
                mech = CellMech.CellMech("PATH_TO_REFERENCE_DIRECTORY", "PATH_TO_DEFORMED_DIRECTORY")
