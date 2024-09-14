#to set helper.py as a local package
from setuptools import find_packages, setup
setup(packages=find_packages()) 
#find_packages(): will look for the constructor file in every folder. If it finds it, the folder will be considered as a local package