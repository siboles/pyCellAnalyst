try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
    name='pyCellAnalyst',
    version='2.0',
    author='Scott Sibole',
    packages=['pyCellAnalyst',
              'pyCellAnalyst.tests',
              'pyCellAnalyst.util'])
