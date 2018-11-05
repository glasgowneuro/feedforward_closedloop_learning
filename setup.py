#!/usr/bin/env python3

"""
setup.py file for feedback_closedloop_learning
"""

from setuptools import setup
from setuptools import Extension
import os
from sys import platform
import numpy

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

fcl_module = Extension('_feedback_closedloop_learning',
		       sources=['fcl.i','fcl.cpp','fcl/bandpass.cpp','fcl/layer.cpp','fcl/neuron.cpp'],
		       extra_compile_args=['-std=c++11','-O3'],
                       include_dirs=[numpy.get_include()],
                       swig_opts=['-c++','-py3']
)

						   
setup (name = 'feedback_closedloop_learning',
       version = '1.0.0',
       author      = "Bernd Porr, Paul Miller",
       author_email = "bernd@glasgowneuro.tech",
       url = "https://github.com/glasgowneuro/feedback_closedloop_learning",
       description = 'Feedback closed loop learning',
       long_description=read('README'),
       ext_modules = [fcl_module],
       py_modules = ["feedback_closedloop_learning"],
       license='GPL 3.0',
       install_requires=[
          'numpy',
       ],
       classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: POSIX',
          'Programming Language :: Python'
          ]
      )
