import subprocess, platform

if '64' in platform.architecture()[0]:
    arch = '64'
else:
    arch = '32'

#figure out the flavour of linux
try:
    subprocess.call("apt-get",shell=True)
    flavour = "debian"
    repo = "apt-get install"
    print("A Debian-based OS was detected. Using APT for installer.")
except OSError:
    try:
        subprocess.call("yum",shell=True)
        flavour = "redhat"
        repo = "yum install"
        print("A Red Hat based OS was detected. Using YUM for installer.")
    except OSError:
        try:
            subprocess.call("pacman",shell=True)
            flavour = "arch"
            repo = "pacman -S"
            print("An A")
        except OSError:
            try:
                subprocess.call("zypper",shell=True)
                flavour = "opensuse"
                repo = "zypper install"
            except OSError:
                print("Sorry but your Linux flavour is not handled. If you wish please modify this script appropriately to call your software manager.")

#OS software manager installs
try:
    import numpy
except ImportError:
    print("Numpy not detected. Installing...")
    subprocess.call(repo+" python-numpy",shell=True)

try:
    import scipy
except ImportError:
    print("Scipy not detected. Installing...")
    subprocess.call(repo+" python-scipy",shell=True)

try:
    import setuptools
except ImportError:
    print("Python setuptools not detected. Installing...")
    subprocess.call(repo+" python-setuptools",shell=True)

try:
    import matplotlib.pyplot
except ImportError:
    print("Matplotlib not detected. Installing...")
    subprocess.call(repo+" python-matplotlib",shell=True)

try:
    import pip
except ImportError:
    print("PIP not detected. Installing...")
    subprocess.call(repo+" python-pip",shell=True)


# Installations from PyPI    
try:
    import weighted
except ImportError:
    print("Weighted quantiles not detected. Installing...")
    subprocess.call("pip install wquantiles",shell=True)

try:
    import sklearn
except ImportError:
    print("Scikit-learn not detected. Installing...")
    subprocess.call("pip install scikit-learn",shell=True)

try:
    import xlrd
except ImportError:
    print("Xlrd not detected. Installing...")
    subprocess.call("pip install xlrd",shell=True)

try:
    import meshpy
except ImportError:
    print("MeshPy was not detected. Installing...")
    subprocess.call("pip install meshpy",shell=True)

# Installations fetching from the web with wget
try:
    import SimpleITK as sitk
    vers = sitk.Version_VersionString()
    if float(vers[0:3]) < 0.9:
        print("SimpleITK version {:s} detected. Downloading newer version and installing...".format(vers))
        subprocess.call("pip uninstall SimpleITK",shell=True)
        if arch == '64':
            subprocess.call("wget http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-py2.7-linux-x86_64.egg/download -O simple_itk.egg",shell=True)
            subprocess.call("easy_install simple_itk.egg",shell=True)
        else:
            subprocess.call("wget http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-py2.7-linux-i686.egg/download -O simple_itk.egg", shell=True)
            subprocess.call("easy_install simple_itk.egg",shell=True)
        os.remove("simple_itk.egg")
    
except ImportError:
    print("SimpleITK was not detected. Downloading and installing...")
    if arch == '64':
        subprocess.call("wget http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-py2.7-linux-x86_64.egg/download -O simple_itk.egg",shell=True)
        subprocess.call("easy_install simple_itk.egg",shell=True)
    else:
        subprocess.call("wget http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-py2.7-linux-i686.egg/download -O simple_itk.egg", shell=True)
        subprocess.call("easy_install simple_itk.egg",shell=True)
    os.remove("simple_itk.egg")

try:
    import vtk
    vers = vtk.VTK_VERSION
    if float(vers[0:3]) < 6.1:
        print("VTK version {:s} detected. Downloading newer version...")
        if arch =='64':
            print("Downloading...")
            subprocess.call("wget http://www.vtk.org/files/release/6.2/vtkpython-6.2.0-Linux-64bit.tar.gz -O vtk_python.tar.gz",shell=True)
            subprocess.call("tar -zxvf vtk_python.tar.gz -C /usr/lib",shell=True)
            os.remove("vtk_python.tar.gz")
            print("Downloaded and extracted to /usr/lib. Please consult installation instructions to link libraries.")
        else:
            print("WARNING: A 32 bit binary of VTK is not available for Linux. Sorry, but you will have to build it from source.")        
except ImportError:
    print("VTK not detected.")
    if arch =='64':
        print("Downloading...")
        subprocess.call("wget http://www.vtk.org/files/release/6.2/vtkpython-6.2.0-Linux-64bit.tar.gz -O vtk_python.tar.gz",shell=True)
        subprocess.call("tar -zxvf vtk_python.tar.gz -C /usr/lib",shell=True)
        subprocess.call("echo export PYTHONPATH=$PYTHONPATH:/usr/lib/VTK-6.2.0-Linux-64bit/lib/python2.7/site-packages",shell=True)
        subprocess.call("source ~/.profile")
        os.remove("vtk_python.tar.gz")
        print("Downloaded and extracted to /usr/lib. Please consult installation instructions to link libraries.")
    else:
        print("WARNING: A 32 bit binary of VTK is not available for Linux. Sorry, but you will have to build it from source.")
