try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
    name='pyCellAnalyst',
    version='2.0',
    author='Scott Sibole',
    packages=['pyCellAnalyst', 'pyCellAnalyst.tests'],
    py_modules=['pyCellAnalyst.Image',
                'pyCellAnalyst.Filters',
                'pyCellAnalyst.FilteringPipeline',
                'pyCellAnalyst.Visualization'])
