"""
Generation and Analysis of Speckle Patterns.

Documentation and examples are at <https://pyspeckle2.readthedocs.io>

Specific help is available for each of the one and two dimensional
speckle functions below.

One dimensional functions::

    pyspeckle.create_exp_1D(M, mean, stdev, cl)
    pyspeckle.create_gaussian_1D(M, mean, stdev, cl)
    pyspeckle.autocorrelation(x)

Two dimensional functions::

    pyspeckle.local_contrast_2D(x, kernel)
    pyspeckle.local_contrast_2D_plot(x, kernel)
    pyspeckle.create_Exponential(M, pix_per_speckle)
    pyspeckle.create_Rayleigh(M, pix_per_speckle)
    pyspeckle.statistics_plot(x)

Three dimensional functions::

    pyspeckle.create_Exponential_3D(M, pix_per_speckle)
    pyspeckle.create_Rayleigh_3D(M, pix_per_speckle)
"""
__version__ = '0.5.0'
__author__ = 'Scott Prahl'
__email__ = 'scott.prahl@oit.edu'
__copyright__ = 'Copyright 2020-23, Scott Prahl'
__license__ = 'MIT'
__url__ = 'https://github.com/scottprahl/pyspeckle.git'

from .pyspeckle import *
