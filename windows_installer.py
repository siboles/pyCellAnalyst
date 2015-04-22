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
        urllib.urlretrieve("https://osf.io/chpxs/?action=download&version=1","numpy-1.9.2+mkl-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install numpy-1.9.2+mkl-cp27-none-win_amd64.whl",shell=True)
        remove_file("numpy-1.9.2+mkl-cp27-none-win_amd64.whl")
    else:
        urllib.urlretrieve("https://osf.io/rag2d/?action=download&version=1","numpy-1.9.2+mkl-cp27-none-win32.whl")
        subprocess.call(pip_run+" install numpy-1.9.2+mkl-cp27-none-win32.whl",shell=True)
        remove_file("numpy-1.9.2+mkl-cp27-none-win32.whl")

try:
    import scipy
except ImportError:
    print("SciPy was not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/hj69m/?action=download&version=1","scipy-0.15.1-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install scipy-0.15.1-cp27-none-win_amd64.whl",shell=True)
        remove_file("scipy-0.15.1-cp27-none-win_amd64.whl")
    else:
        urllib.urlretrieve("https://osf.io/dkptx/?action=download&version=1","scipy-0.15.1-cp27-none-win32.whl")
        subprocess.call(pip_run+" install scipy-0.15.1-cp27-none-win32.whl",shell=True)
        remove_file("scipy-0.15.1-cp27-none-win32.whl")
    
try:
    import matplotlib
except ImportError:
    print("Matplotlib was not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/3wjna/?action=download&version=1","matplotlib-1.4.3-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install matplotlib-1.4.3-cp27-none-win_amd64.whl",shell=True)
        remove_file("matplotlib-1.4.3-cp27-none-win_amd64.whl")
    else:
        urllib.urlretrieve("https://osf.io/5gpvy/?action=download&version=1","matplotlib-1.4.3-cp27-none-win32.whl")
        subprocess.call(pip_run+" install matplotlib-1.4.3-cp27-none-win32.whl",shell=True)
        remove_file("matplotlib-1.4.3-cp27-none-win32.whl")

try:
    import sklearn
except ImportError:
    print("Scikit-learn not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/4cpez/?action=download&version=1","scikit_learn-0.16.0-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install scikit_learn-0.16.0-cp27-none-win_amd64.whl",shell=True)
        remove_file("scikit_learn-0.16.0-cp27-none-win_amd64.whl")
    else:
        urllib.urlretrieve("https://osf.io/ji8tc/?action=download&version=1","scikit_learn-0.16.0-cp27-none-win32.whl")
        subprocess.call(pip_run+" install scikit_learn-0.16.0-cp27-none-win32.whl",shell=True)
        remove_file("scikit_learn-0.16.0-cp27-none-win32.whl")

try:
    import meshpy
except ImportError:
    print("Meshpy not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/jd32t/?action=download&version=1","MeshPy-2014.1-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install MeshPy-2014.1-cp27-none-win_amd64.whl",shell=True)
        remove_file("MeshPy-2014.1-cp27-none-win_amd64.whl")
    else:
        urllib.urlretrieve("https://osf.io/wy3qd/?action=download&version=1","MeshPy-2014.1-cp27-none-win32.whl")
        subprocess.call(pip_run+" install MeshPy-2014.1-cp27-none-win32.whl",shell=True)
        remove_file("MeshPy-2014.1-cp27-none-win32.whl")

try:
    import xlrd
except ImportError:
    print("xlrd not detected. Downloading and installing...")
    urllib.urlretrieve("https://osf.io/ixyu5/?action=download&version=1","xlrd-0.9.3-py2.py3-none-any.whl")
    subprocess.call(pip_run+" install xlrd-0.9.3-py2.py3-none-any.whl",shell=True)
    remove_file("xlrd-0.9.3-py2.py3-none-any.whl")

try:
    import vtk
    vers = vtk.VTK_VERSION
    if float(vers[0:3]) < 6.1:
        if arch == '64':
            urllib.urlretrieve("https://osf.io/b2tfg/?action=download&version=1","VTK-6.1.0-cp27-none-win_amd64.whl")
            subprocess.call(pip_run+" install VTK-6.1.0-cp27-none-win_amd64.whl",shell=True)
            remove_file("vtk.whl")
        else:
            urllib.urlretrieve("https://osf.io/396zp/?action=download&version=1","VTK-6.1.0-cp27-none-win32.whl")
            subprocess.call(pip_run+" install VTK-6.1.0-cp27-none-win32.whl",shell=True)
            remove_file("VTK-6.1.0-cp27-none-win32.whl")
        
except ImportError:
    print("VTK not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("https://osf.io/b2tfg/?action=download&version=1","VTK-6.1.0-cp27-none-win_amd64.whl")
        subprocess.call(pip_run+" install VTK-6.1.0-cp27-none-win_amd64.whl",shell=True)
        remove_file("VTK-6.1.0-cp27-none-win_amd64.whl")
    else:
        urllib.urlretrieve("https://osf.io/396zp/?action=download&version=1","VTK-6.1.0-cp27-none-win32.whl")
        subprocess.call(pip_run+" install VTK-6.1.0-cp27-none-win32.whl",shell=True)
        remove_file("VTK-6.1.0-cp27-none-win32.whl")

try:
    import SimpleITK as sitk
    vers = sitk.Version_VersionString()
    if float(vers[0:3]) < 0.9:
        print("SimpleITK version {:s} detected. Downloading newer version and installing...".format(vers))
        subprocess.call(pip_run+" uninstall SimpleITK",shell=True)
        urllib.urlretrieve("http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-cp27-none-win32.whl/download","SimpleITK-0.9.0b01-cp27-none-win32.whl")
        subprocess.call(pip_run+" install SimpleITK-0.9.0b01-cp27-none-win32.whl",shell=True)
        remove_file("SimpleITK-0.9.0b01-cp27-none-win32.whl")
except ImportError:
    print("SimpleITK not detected. Downloading and installing...")
    urllib.urlretrieve("http://sourceforge.net/projects/simpleitk/files/SimpleITK/0.9b01/Python/SimpleITK-0.9.0b01-cp27-none-win32.whl/download","SimpleITK-0.9.0b01-cp27-none-win32.whl")
    subprocess.call(pip_run+" install SimpleITK-0.9.0b01-cp27-none-win32.whl",shell=True)
    remove_file("SimpleITK-0.9.0b01-cp27-none-win32.whl")
