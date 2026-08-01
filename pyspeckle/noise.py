"""
Correlated Gaussian sequences.

These are **not** speckle.  Both routines return normally distributed values
and are named for the shape of their autocorrelation, not for the
distribution of the values: `create_exp_corr_1D` gives exp(-x/cl) and
`create_gaussian_corr_1D` gives exp(-x**2/2cl**2).  Speckle irradiance, by
contrast, is exponentially distributed and never negative.

They model rough surface profiles, correlated detector signals, and other
inputs to a scattering calculation.  For the phase imposed by a rough surface
use `create_phase_screen`, which is the same kind of object measured in
radians.

The exponential form uses an AR(1) recursion, which is inherently sequential
and has no higher-dimensional analogue, so both routines are one dimensional.
"""

import numpy as np

__all__ = (
    "create_exp_corr_1D",
    "create_gaussian_corr_1D",
)


def create_exp_corr_1D(M, mean, stdev, cl):
    """
    Generate an array of length M of values with exponential autocorrelation.

    The returned array will have the autocorrelation function exp(-x/cl).

    The speckle pattern will also have a normal probability density function
    with the specified mean and standard deviation.

    see https://www.cmu.edu/biolphys/deserno/pdf/corr_gaussian_random.pdf

    Args:
        M:     dimension of desired array     [-]
        mean:  average value of signal        [gray levels]
        stdev:   standard deviation of signal [gray levels]
        cl:    correlation length             [# of pixels]

    Returns:
        array of length M
    """
    if cl <= 0:
        raise ValueError("Correlation length cl must be positive.")

    if M <= 2 * cl:
        raise ValueError("Array size M must be at least twice the correlation length cl.")

    if stdev < 0:
        raise ValueError("Standard deviation std must be non-negative.")

    f = np.exp(-1 / cl)
    fsqrt = np.sqrt(1 - f * f)

    # gaussian deviates with mean=0 and variance=1
    g = np.random.normal(size=M)
    r = np.zeros(M)

    r[0] = g[0]
    for i in range(1, M):
        r[i] = f * r[i - 1] + fsqrt * g[i]

    return mean + stdev * r


def create_gaussian_corr_1D(M, mean, stdev, cl):
    """
    Generate an array of length M of values with Gaussian autocorrelation.

    The generated speckle pattern will be characterized by the autocorrelation
    function exp(-0.5*(x/cl)**2)

    The speckle pattern will also have a normal probability density function
    with the specified mean and standard deviation.

    The Nyquist sampling theorem sets a lower limit of the sampling
    frequency; M/cl must be greater than 2.  However, to achieve
    adequate Gaussian statistics M/cl should be much larger
    larger than this (say more than 50).

    see: <http://www.mysimlabs.com/matlab/surfgen/rsgeng1D.m>

    Args:
        M:     dimension of desired array     [-]
        mean:  average value of signal        [gray levels]
        stdev:   standard deviation of signal [gray levels]
        cl:    correlation length             [# of pixels]

    Returns:
        array of length M
    """
    if cl <= 0:
        raise ValueError("Correlation length cl must be positive.")

    if M <= 2 * cl:
        raise ValueError("Array size M must be at least twice the correlation length cl.")

    if stdev < 0:
        raise ValueError("Standard deviation std must be non-negative.")

    Z = np.random.normal(0, stdev, M)  # zero mean

    # Gaussian filter
    x = np.linspace(-M / 2, M / 2, M) / cl
    F = np.exp(-2 * x**2)

    # Fourier transform the signal and filter
    fZ = np.fft.fft(Z)
    fF = np.fft.fft(F)

    # correlation is the scaled inverse Fourier transform of the product
    f = np.sqrt(2 / cl / np.sqrt(np.pi)) * np.fft.ifft(fZ * fF)

    # shift the correlation
    return mean + f.real
