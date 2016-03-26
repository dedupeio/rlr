#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError :
    raise ImportError("setuptools module required, please go to https://pypi.python.org/pypi/setuptools and follow the instructions for installing setuptools")


setup(
    name='rlr',
    url='https://github.com/datamade/rlr',
    version='2.3',
    description='Case weighted L2 regularized logistic regression',
    packages=['rlr'],
    install_requires=['numpy', 'pylbfgs', 'future'],
    license='Apache 2.0 License: https://www.apache.org/licenses/LICENSE-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis']
    )

