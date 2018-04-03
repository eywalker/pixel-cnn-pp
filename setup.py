#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='pixel_cnn',
    version='0.0.0',
    description='PixelCNN',
    author='',
    author_email='',
    url='https://github.com/eywalker/pixelcnn-pp',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'pytorch'],
)
