Installation
============

.. toctree::
   :maxdepth: 1
   :glob:

Python
------

pyCellAnalyst requires a `Python <https://www.python.org/downloads/>`_ environment to be installed.

.. note::

   You must install the 2.x version not the 3.x version. It is recommended to install a 64-bit version, since the memory limitation of 32-bit has been an issue for some datasets. It is also recommended to install the standard Python not a pre-packaged distribution such as Enthought or Anaconda. These distributions come with older versions of VTK and SimpleITK and will likely causes errors.

.. note::

   If you are on Linux, you likely already have a Python installation. Most distributions ship with Python but some, such as Arch Linux, don't. Please consult the documentation for your particular distibution to see how to install. Remember a Python version 2.x **not** version 3.x is required.

It is highly recommended to install pyCellAnalyst into a virtual environment. This ensures that all the dependencies pyCellAnalyst requires are satisfied with the exact version used in development. Also, there is no danger of harming the system level Python. This is especially important in Linux, since most distributions depend on Python heavily. Finally, this gives the freedom to use pre-built packages such as Anaconda or Enthought for normal usage, while easily switching to the pyCellAnalyst specific environment when it is needed.

However, if you wish to install to your default Python directory you can skip to **Step 4** of `Installing the Module and Dependencies`_.

Creating a Python Virtual Environment
-------------------------------------

.. note::

   A script to automate the following steps is provided for Windows (*make_win_venv.bat*) and Linux/Mac (*make_linux_mac_venv.sh*). Please feel free to use those. The virtual environment will be named *pyCellAnalyst*. 

To create a virtual environment on Windows do the following:

1. Type *cmd* in the start menu search bar, press enter to launch a command prompt.
2. Install *virtualenv* and *virtualenvwrapper-win* using *pip*.

   .. code-block:: guess

      pip install virtualenv virtualenvwrapper-win
3. Create a virtual environment with whatever name you wish, here we use *pyCellAnalyst*

   .. code-block:: guess

      mkvirtualenv pyCellAnalyst
4. When a virtual environment is first created you will automatically be switched to it. The virtual environment you are currently in is indicated by the name in the parentheses before the command prompt *e.g. (pyCellAnalyst)*
   * To switch to a virtual environment type:

     .. code-block:: guess

        workon ENVIRONMENT_NAME
   * To exit the virtual environment type:

     .. code-block:: guess

        deactivate
   * For more commands please consult the `documentation <http://virtualenvwrapper.readthedocs.org/en/latest/command_ref.html>`_ and the `screencast <http://mathematism.com/2009/07/30/presentation-pip-and-virtualenv/>`_.

In Linux or Mac do the following:

1. Open a command terminal.
2. Install *virtualenv* and *virtualenvwrapper-win* using *pip*.

   .. code-block:: guess

      sudo pip install virtualenv virtualenvwrapper

.. note::

   The *sudo* in the above command temporarily elevates you to root (administrator). It will prompt you for a password, which will be the same as your login password. If you do not have the ability to use *sudo*, you will need to talk to your IT administrator(s) to do the above step.
3. Type the following in the command terminal and press enter (or .bash_profile, .zprofile, etc.):

   .. code-block:: guess

      echo "export WORKON_HOME=$HOME/Envs" >> ~/.profile
4. And type this as well and press enter:

   .. code-block:: guess

      echo "source /usr/local/bin/virtualenvwrapper.sh >> ~/.profile
5. Now enter:

   .. code-block:: guess

      source ~/.profile

.. note::

   The command in Step 5 executes the commands that are recorded in the file ~/.profile. This execution is also performed when you logon to your computer, so you will never have to do this again.

6. Create the virtual environment:

   .. code-block:: guess

      mkvirtualenv pyCellAnalyst

.. note::

   See the virtual environment commands to switch to and from environments given in Step 4 of the Windows instructions. For more information consult the `documentation <http://virtualenvwrapper.readthedocs.org/en/latest/command_ref.html>`_ and the `screencast <http://mathematism.com/2009/07/30/presentation-pip-and-virtualenv/>`_.



Installing the Module and Dependencies
--------------------------------------

Again it is recommended that you are in a Python virtual environment when you do this installation (although it is not required).

* Install the latest version from github:

  1. Either download and extract the .zip `file <https://github.com/siboles/pyCellAnalyst/archive/master.zip>`_ or clone the `repository <https://github.com/siboles/pyCellAnalyst>`_  using git.
  2. Navigate to the *src* folder in the project where *setup.py* is located.
  3. Switch to the virtual environment you wish to install into e.g in a command terminal.

     .. code-block:: guess

        workon pyCellAnalyst
  4. Perform the installation by entering:

     .. code-block:: guess

        python setup.py install

