"""
Generation and Analysis of Speckle Patterns.

Documentation and examples are at <https://pyspeckle2.readthedocs.io>

Specific help is available for each of the one and two dimensional
speckle functions below.

One dimensional functions::

    pyspeckle.create_exponential_1D(M, pix_per_speckle)
    pyspeckle.create_unpolarized_1D(M, pix_per_speckle)
    pyspeckle.create_exp_corr_1D(M, mean, stdev, cl)
    pyspeckle.create_gaussian_corr_1D(M, mean, stdev, cl)
    pyspeckle.create_phase_screen_1D(M, sigma, cl)
    pyspeckle.local_contrast_1D(x, kernel)
    pyspeckle.local_contrast_1D_plot(x, kernel)
    pyspeckle.autocorrelation(x)

Correlated random numbers, the Gaussian copula of Duncan & Kirkpatrick::

    pyspeckle.box_muller(mu, sigma, N)
    pyspeckle.zvalues(r, N)
    pyspeckle.tvalues(r, N)

`create_exponential_1D` is speckle, named for its irradiance distribution.
`create_exp_corr_1D` and `create_gaussian_corr_1D` return normally distributed
values and are named for their autocorrelation.

Two dimensional functions::

    pyspeckle.local_contrast_2D(x, kernel)
    pyspeckle.local_contrast_2D_plot(x, kernel)
    pyspeckle.create_exponential_2D(M, pix_per_speckle)
    pyspeckle.create_unpolarized_2D(M, pix_per_speckle)
    pyspeckle.create_phase_screen_2D(M, sigma, cl)
    pyspeckle.statistics_plot(x)

Three dimensional functions::

    pyspeckle.create_exponential_3D(M, pix_per_speckle)
    pyspeckle.create_unpolarized_3D(M, pix_per_speckle)
    pyspeckle.local_contrast_3D(x, kernel)
    pyspeckle.slice_plot(data, x, y, z)

The implementation is split across `core`, `speckle_1D`, `speckle_2D`, and
`speckle_3D`, but every public function is re-exported here, so
`pyspeckle.create_exponential_2D(...)` is the intended way to reach them.
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
