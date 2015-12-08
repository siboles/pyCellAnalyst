Finite Element Analysis Graphical User Interface
================================================

.. tcotree::
   :maxdepth: 1
   :glob:

Pre-requisites
--------------

This graphical user intereface (GUI) uses the FEBio finite element solver, which can be downloaded `here <http://www.febio.org>`_. FEBio is aimed at solving problems in biomechanics, which often have both geometric and material non-linearity as well as anisotropy and multiple physical phases. This and the fact that FEBio is open-source make it an excellent solver for finite element analysis of the cells.

After running the install wizard, it may be advantage to add the FEBio binary (.exe in Windows, .lnx64 (arbitrary) in Linux) to your system path. This is not necessary for this utility though.

Peforming and Analysis
----------------------

Start the GUI, by opening a command terminal and typing:

.. code-block:: guess

   python -m pyCellAnalyst.FEA_GUI

Running for the First Time
^^^^^^^^^^^^^^^^^^^^^^^^^^

The GUI looks for a file in your HOME directory called .pyCellAnalystFEA.pth containing the path to your FEBio binary. If this is not found or the path contained within is incorrect, a file browser will spawn instructing you to navigate to the FEBio executable and select it. Do this and the file will be generated or modified.

Importing Model Definition Pickles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the option to save for FEA is active and a deformable image registration analysis is performed, files with the naming convention **cellFEA{:02d}.pkl** will be written to the corresponding results directory. These files contain all the necessary information to generate a finite element model of the cell and its deformation except for material properties.

Simply click the **Add** button on the Analysis tab and select pickle files for the cell(s) you wish to model. One or more files can be selected at a time. To remove files that were added simply select them from the listbox and click **Remove**

Setting Outputs
^^^^^^^^^^^^^^^

Checkboxes are provided to indicate what variables to output as results. Select or deselect these as you wish.

Setting Analysis Options
^^^^^^^^^^^^^^^^^^^^^^^^

The GUI will automatically generate plots of the results for each analysis and write them to a directory named FEA_analysis_{TIME_STAMP} one directory level above the pickle folders. The options are as follows:

- Generate Histograms - For each cell, a histogram of each selected output variable will be generated from the values calculated for each element. The histograms are volumetrically weighted, such that the integral area is always 1. This will be done for all mechanical treatments of the cell included in the analysis.

- Tukey Boxplots - For each cell, a volumetrically weighted Tukey boxplot will be generated for each selected output variable. This will be done for all mechanical treatments of the cell included in the analysis.

- Calculate Differences - Since the finite element mesh used for each cell in the analysis is the same for all mechanical treatments (the reference state mesh), paired differences for each output variable can be calculated for all elements. The root-mean-square differences for all combinations of treatments within a cell are written to disk as heatmaps. If *Convert to VTK* is also selected the differences are also saved on each tetrahedron.

- Convert to VTK - An unstructured grid representation of each cell will be written in VTK format (.vtu). All selected output variables will be saved on the tetrahedrons. These can be further visualized and analyzed in software such as Paraview.

Assigning Material Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second tab in the GUI, *Material Model*, provides options to define the material properties of the cells. The simplest model would be an isotropic ground substance such as a neo-Hookean or Mooney-Rivlin material with no tensile network.

To attempt to model the cytoskeleton, a transversely-isotropic ground substance can be selected to represent the microtubules. The symmetry axes for this material are oriented perpendicular to the local iso-distance contours measured from the cell surface.

To model the actin filaments, *Tensile Fibres* can be added. The contribution of these is modelled as a probability density function in spherical space. **ksi1** is the axis of an ellipsoid oriented tangent to the local iso-distance contour, and can be thought of as the fibre stiffness in that direction. **ksi2** and **ksi3** are the other ellipsoidal axes and are forced to be equal. Likewise, the parameters **beta1**, **beta2**, and **beta3**, also represent the ellipsoidal axes of a probability density function, but these govern the non-linearity of the fibre stiffnesses. Since a derivative is taken, values of 2 for **beta** represent linear stiffness behaviour, and also the lower bound allowed for the value.



