#!/usr/bin/env python3

"""
setup.py file for feedforward_closedloop_learning
"""

from setuptools import setup
from setuptools import Extension
import os
from sys import platform
import numpy

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if platform == "linux" or platform == "linux2" or platform == "darwin":
    fcl_module = Extension('_feedforward_closedloop_learning',
		       sources=['fcl.i','fcl.cpp','fcl_util.cpp','fcl/bandpass.cpp','fcl/layer.cpp','fcl/neuron.cpp'],
		       extra_compile_args=['-std=c++11','-O3'],
                       include_dirs=[numpy.get_include()],
                       swig_opts=['-c++','-py3']
                       )
elif platform == "win32":
    fcl_module = Extension('_feedforward_closedloop_learning',
		       sources=['fcl.i','fcl.cpp','fcl_util.cpp','fcl/bandpass.cpp','fcl/layer.cpp','fcl/neuron.cpp'],
		       extra_compile_args=['-D_CRT_SECURE_NO_WARNINGS'],
                       include_dirs=[numpy.get_include()],
                       swig_opts=['-c++','-py3']
                       )



setup (name = 'feedforward_closedloop_learning',
       version = '2.0.0',
       author      = "Bernd Porr, Paul Miller",
       author_email = "bernd@glasgowneuro.tech",
       url = "https://github.com/glasgowneuro/feedforward_closedloop_learning",
       description = 'Feedforward Closedloop Learning (FCL)',
       long_description=read('README_py.rst'),
       ext_modules = [fcl_module],
       py_modules = ["feedforward_closedloop_learning"],
       license='GPL 3.0',
       install_requires=[
          'numpy',
       ],
       classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python'
          ]
      )
