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
    pyspeckle.slice_plot(data, x, y, z)

The implementation is split across `core`, `speckle_1D`, `speckle_2D`, and
`speckle_3D`, but every public function is re-exported here, so
`pyspeckle.create_Exponential(...)` is the intended way to reach them.
"""

__version__ = "0.6.1"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2018-26, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/pyspeckle"

from .core import *
from .speckle_1D import *
from .speckle_2D import *
from .speckle_3D import *
