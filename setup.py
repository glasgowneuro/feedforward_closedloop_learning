#!/usr/bin/env python3

"""
setup.py file for deep_feedback_learning
"""

from setuptools import setup
from setuptools import Extension
import os
from sys import platform

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

dfl_module = Extension('_deep_feedback_learning',
		       sources=['deep_feedback_learning.i'],
		       extra_compile_args=['-std=c++11'],
                       extra_link_args=['libdeep_feedback_learning_static.a'],
                       swig_opts=['-c++','-py3']
)

						   
setup (name = 'deep_feedback_learning',
       version = '1.0.1',
       author      = "Bernd Porr",
       author_email = "bernd@glasgowneuro.tech",
       url = "https://github.com/glasgowneuro/deep_feedback_learning",
       description = 'Deep feedback learning',
       long_description=read('README'),
       ext_modules = [dfl_module],
       py_modules = ["deep_feedback_learning"],
       license='GPL 3.0',
       classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: POSIX',
          'Programming Language :: Python'
          ]
      )
