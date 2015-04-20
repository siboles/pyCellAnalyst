import subprocess, os, struct, urllib

if struct.calcsize("P") == 8: #size of void is 8 bytes
    arch = '64'
else:
    arch = '32'

# Windows builds of popular scientific python modules
# This is an unofficial source and thus comes as is with absolutely no warranty

try:
    import numpy
except ImportError:
    print("Numpy was not detected. Downloading and installing...")
    if arch == '64':
        urllib.urlretrieve("")
    
