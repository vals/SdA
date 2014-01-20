#!/usr/bin/env python
""" Setup file for stacked denoising autoencoders script
"""
from setuptools import setup, find_packages

setup(name='SdA',
	  version='0.0',
	  author='Valentine Svensson',
	  scripts=['scripts/sda.py'],
	  packages=find_packages())
