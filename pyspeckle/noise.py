"""
Correlated Gaussian random fields.

These are **not** speckle.  `create_correlated` returns normally distributed
values whose *autocorrelation* has a chosen shape; speckle irradiance, by
contrast, is exponentially distributed and never negative.

The correlation length `cl` is defined the same way for both shapes: it is the
lag at which the autocorrelation falls to 1/e.

    'exponential'   exp(-x/cl)
    'gaussian'      exp(-(x/cl)**2)

Fields like these model rough surface profiles, correlated detector signals,
and other inputs to a scattering calculation.  `create_phase_screen` is the
same object measured in radians, which is what partially developed speckle
needs.

Both are built by spectral synthesis: white noise is filtered by the square
root of the power spectral density, which by the Wiener-Khinchin theorem is
the transform of the wanted autocorrelation.  The result is isotropic in two
and three dimensions, which matters because a separable product of
one-dimensional exponentials is diamond shaped rather than radially
symmetric.  The construction assumes the field is periodic, so `cl` should be
a small fraction of the array; a factor of 50 or more is comfortable.
"""

import numpy as np

from .core import _normalize_shape

__all__ = (
    "create_correlated",
    "create_phase_screen",
)


def create_correlated(shape, mean, stdev, cl, correlation="gaussian"):
    """
    Generate a correlated Gaussian random field.

    `shape` follows the numpy convention used by `np.ones`, exactly as in
    `create_exponential`: an integer gives a one-dimensional sequence and a
    tuple gives two or three dimensions::

        create_correlated(100000, 8000, 800, 300)      # a long trace
        create_correlated((256, 256), 0, 1, 16)        # a rough surface
        create_correlated((64, 64, 64), 0, 1, 8)       # a correlated volume

    The values are normally distributed with the requested mean and standard
    deviation.  `cl` is the lag at which the autocorrelation falls to 1/e, so
    'exponential' gives exp(-x/cl) and 'gaussian' gives exp(-(x/cl)**2).

    Note that this is unrelated to `create_exponential`, which returns speckle
    irradiance.  Here the word names the autocorrelation, not the distribution
    of the values.

    Args:
        shape:       integer or tuple giving the shape of the field
        mean:        average value
        stdev:       standard deviation
        cl:          correlation length [pixels]
        correlation: 'gaussian' or 'exponential' autocorrelation

    Returns:
        array with the requested shape
    """
    dims = _normalize_shape(shape)

    if stdev < 0:
        raise ValueError("Standard deviation stdev must be non-negative.")

    if cl <= 0:
        raise ValueError("Correlation length cl must be positive.")

    # spectral synthesis wraps around, so the field must be much longer than
    # the correlation length; this is the same guard the 1D routines had
    if min(dims) <= 2 * cl:
        raise ValueError("Every dimension must be at least twice the correlation length cl.")

    lcorrelation = correlation.lower()

    # signed lags, wrapped, so the autocorrelation is periodic on the grid
    axes = [np.fft.fftfreq(n, d=1 / n) for n in dims]
    grid = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(sum(g**2 for g in grid))

    if lcorrelation == "gaussian":
        acf = np.exp(-((r / cl) ** 2))
    elif lcorrelation == "exponential":
        acf = np.exp(-r / cl)
    else:
        raise ValueError("correlation must be 'gaussian' or 'exponential'")

    # power spectral density; tiny negative lobes are numerical noise
    psd = np.maximum(np.fft.fftn(acf).real, 0)

    white = np.fft.fftn(np.random.normal(size=dims))
    field = np.fft.ifftn(white * np.sqrt(psd)).real

    field -= field.mean()
    deviation = field.std() or 1
    return mean + stdev * field / deviation


def create_phase_screen(shape, sigma, cl, correlation="gaussian"):
    """
    Generate a correlated Gaussian phase screen.

    This is `create_correlated` with zero mean, measured in **radians**.  It
    models the phase imposed by a rough surface, and is the input needed for
    partially developed speckle: multiply `exp(1j*screen)` by an aperture and
    transform, exactly as `create_exponential` does with uniform random phase.

    The fraction of the field left unscattered is exp(-sigma**2).  Large
    `sigma` scrambles the phase completely and recovers the fully developed
    limit that `create_exponential` produces directly; small `sigma` leaves a
    strong coherent component and the speckle is only partially developed.
    How that coherent component appears, and therefore what contrast is
    measured, depends on the observing geometry.

    Args:
        shape:       integer or tuple giving the shape of the screen
        sigma:       standard deviation of the phase [radians]
        cl:          correlation length [pixels]
        correlation: 'gaussian' or 'exponential' autocorrelation

    Returns:
        zero-mean array of phases in radians with the requested shape
    """
    return create_correlated(shape, 0, sigma, cl, correlation=correlation)
