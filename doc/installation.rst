Installation
============

.. toctree::
   :maxdepth: 1
   :glob:
   :numbered:

Python
------

A Windows video installation guide is provided `here <https://youtu.be/0Zp8ijjXQ6s>`_. 

pyCellAnalyst requires a `Python <https://www.python.org/downloads/>`_ environment to be installed.

.. note::

   You must install the 2.* version not the 3.* version. On Windows, it is also highly recommended to install the 32-bit (x86) version. It is also recommended to install the standard Python not a pre-packaged distribution such as Enthought or Anaconda. These distributions come with older versions of VTK and SimpleITK and will likely causes errors.

If you are on Linux, you likely already have a Python installation.

Installation Script
-------------------
Scripts are provided to automatically download and install all the dependencies needed by pyCellAnalyst.

On Windows open a command terminal:

1. Type *cmd* in the start menu search bar
2. Right-click the command terminal application icon that appears and select *Run as administrator*
3. Navigate to the directory in the downloaded pyCellAnalyst package containing *windows_installer.py* and run it by typing:

   .. code-block:: guess

      cd PATH_TO_INSTALLER
      python windows_installer.py

   where *PATH_TO_INSTALLER* is the directory containing *windows_installer.py*.

.. note::
   It may be easiest to simply navigate to the correct folder using Windows explorer and copy the path from the address bar. Keybinding for copy and paste do not work in a command terminal, so one must right-click and choose paste.

On Linux open a command terminal and type:

   .. code-block:: guess

      cd PATH_TO_INSTALLER
      sudo python linux_installer.py

If you do not have superuser priviliges on your machine, you will need to ask your system administrator to change you Python dist-packages install location. Then you can execute the above commands without *sudo*.

Installing the Module
---------------------

In a command terminal, navigate to the folder containing *setup.py* (*PATH_TO_SETUP*) and run the script:

    .. code-block:: guess

       cd PATH_TO_SETUP
       python setup.py install

On Linux, you will need to prepend the second line with *sudo* or change the dist-packages install location.

       
