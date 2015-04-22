import subprocess, os, struct, urllib

def remove_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
        

if struct.calcsize("P") == 8: #size of void is 8 bytes
    arch = '64'
else:
    arch = '32'

# Windows builds of popular scientific python modules
# This is an unofficial source and thus comes as is with absolutely no warranty
try:
    import pip
    pip_run = "C:\\Python27\\Scripts\\pip2.7.exe"
except ImportError:
    print("pip was not detected. Downloading and installing...")
    subprocess.call("python get-pip.py",shell=True)
    subprocess.call("setx Path \"%Path%;C:\\Python27\\Scripts\" /M",shell=True)
    pip_run = "C:\\Python27\\Scripts\\pip2.7.exe"
    
try:
    import numpy
except ImportError:
    print("NumPy was not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/chpxs/?action=download&version=1","numpy.whl")
        subprocess.call(pip_run+" install numpy.whl",shell=True)
        remove_file("numpy.whl")
    else:
        urllib.urlretrieve("https://osf.io/rag2d/?action=download&version=1","numpy.whl")
        subprocess.call(pip_run+" install numpy.whl",shell=True)
        remove_file("numpy.whl")

try:
    import scipy
except ImportError:
    print("SciPy was not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/hj69m/?action=download&version=1","scipy.whl")
        subprocess.call(pip_run+" install scipy.whl",shell=True)
        remove_file("scipy.whl")
    else:
        urllib.urlretrieve("https://osf.io/dkptx/?action=download&version=1","scipy.whl")
        subprocess.call(pip_run+" install scipy.whl",shell=True)
        remove_file("scipy.whl")

try:
    import matplotlib
except ImportError:
    print("Matplotlib was not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/3wjna/?action=download&version=1","matplotlib.whl")
        subprocess.call(pip_run+" install matplotlib.whl",shell=True)
        remove_file("matplotlib.whl")
    else:
        urllib.urlretrieve("https://osf.io/5gpvy/?action=download&version=1","matplotlib-1.4.3-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install matplotlib.whl",shell=True)
        remove_file("matplotlib.whl")

try:
    import sklearn
except ImportError:
    print("Scikit-learn not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/4cpez/?action=download&version=1","scikit-learn.whl")
        subprocess.call(pip_run+" install scikit-learn.whl",shell=True)
        remove_file("scikit-learn.whl")
    else:
        urllib.urlretrieve("https://osf.io/ji8tc/?action=download&version=1","scikit-learn.whl")
        subprocess.call(pip_run+" install scikit-learn.whl",shell=True)
        remove_file("scikit-learn.whl")

try:
    import meshpy
except ImportError:
    print("Meshpy not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/jd32t/?action=download&version=1","meshpy.whl")
        subprocess.call(pip_run+" install meshpy.whl",shell=True)
        remove_file("meshpy.whl")
    else:
        urllib.urlretrieve("https://osf.io/wy3qd/?action=download&version=1","meshpy.whl")
        subprocess.call(pip_run+" install meshpy.whl",shell=True)
        remove_file("meshpy.whl")

try:
    import xlrd
except ImportError:
    print("xlrd not detected. Downloading and installing...")
    urllib.urlretrieve("https://osf.io/ixyu5/?action=download&version=1","xlrd.whl")
    subprocess.call(pip_run+" install xlrd.whl",shell=True)
    remove_file("xlrd.whl")

try:
    import vtk
    vers = vtk.VTK_VERSION
    if float(vers[0:3]) < 6.1:
        if arch == '64':
            urllib.urlretrieve("https://osf.io/b2tfg/?action=download&version=1","vtk.whl")
            subprocess.call(pip_run+" install vtk.whl",shell=True)
            remove_file("vtk.whl")
        else:
            urllib.urlretrieve("https://osf.io/396zp/?action=download&version=1","vtk.whl")
            subprocess.call(pip_run+" install vtk.whl",shell=True)
            remove_file("vtk.whl")
        
except ImportError:
    print("VTK not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/b2tfg/?action=download&version=1","vtk.whl")
        subprocess.call(pip_run+" install vtk.whl",shell=True)
        remove_file("vtk.whl")
    else:
        urllib.urlretrieve("https://osf.io/396zp/?action=download&version=1","vtk.whl")
        subprocess.call(pip_run+" install vtk.whl",shell=True)
        remove_file("vtk.whl")

try:
    import SimpleITK as sitk
    vers = sitk.Version_VersionString()
    if float(vers[0:3]) < 0.9:
        print("SimpleITK version {:s} detected. Downloading newer version and installing...".format(vers))
        subprocess.call(pip_run+" uninstall SimpleITK",shell=True)
        urllib.urlretrieve("http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-cp27-none-win32.whl/download","SimpleITK-0.9.0b01-cp27-none-win32.whl")
        subprocess.call(pip_run+" install simpleitk.whl",shell=True)
        remove_file("simpleitk.whl")
except ImportError:
    print("SimpleITK not detected. Downloading and installing...")
    urllib.urlretrieve("http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-cp27-none-win32.whl/download","SimpleITK-0.9.0b01-cp27-none-win32.whl")
    subprocess.call(pip_run+" install simpleitk.whl",shell=True)
    remove_file("simpleitk.whl")
