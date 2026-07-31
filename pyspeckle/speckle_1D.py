"""
Generation of one-dimensional speckle sequences.

This module holds two different kinds of generator, and the names say which.

`create_exponential_1D` and `create_unpolarized_1D` are true speckle: a
random-phase aperture is Fourier transformed and squared, exactly as in 2D.
The word names the irradiance distribution.

`create_exp_corr_1D` and `create_gaussian_corr_1D` instead shape Gaussian deviates
to a target autocorrelation length -- the first with an AR(1) recursion, the
second by FFT convolution with a Gaussian kernel.  Both return normally
distributed values, and the word names the autocorrelation, not the amplitude
distribution.
"""

import numpy as np

from .core import _local_contrast

__all__ = (
    "create_exponential_1D",
    "create_unpolarized_1D",
    "create_exp_corr_1D",
    "create_gaussian_corr_1D",
    "local_contrast_1D",
)


def local_contrast_1D(x, kernel):
    """
    Calculate local (1D) speckle contrast along a line.

    The kernel is a 1D array describing the window over which contrast is
    calculated.  For example, `np.ones(5)` averages over five samples.

    Only valid positions of the correlation are returned, so an M-long
    pattern with an N-long kernel yields M-N+1 values, none of which are
    contaminated by zero padding at the ends.

    Args:
        x: 1D speckle pattern
        kernel: 1D window over which contrast is to be calculated

    Returns:
        1D_contrast_array, total_contrast
    """
    return _local_contrast(x, kernel)


def create_exponential_1D(M, pix_per_speckle, polarization=1):
    """
    Generate a length M polarized, fully-developed speckle irradiance pattern.

    This is the one-dimensional form of `create_exponential_2D`: uniformly
    distributed phases fill a segment aperture, the result is Fourier
    transformed, and the magnitude is squared.  The irradiance is therefore
    exponentially distributed with unit speckle contrast.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    `polarization=0` sums two independent patterns to give unpolarized
    speckle, whose irradiance is gamma distributed with shape 2 and whose
    contrast is 1/sqrt(2).

    Note that this is unrelated to `create_exp_corr_1D`, which returns
    normally distributed values with an exponential autocorrelation.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008)

    Args:
        M:               length of desired speckle array
        pix_per_speckle: number of pixels per smallest speckle
        polarization:    degree of polarization

    Returns:
        array of length M
    """
    if polarization < 0 or polarization > 1:
        raise ValueError("bad polarization. It must be 0 <= polarization <= 1.")

    if pix_per_speckle < 1:
        raise ValueError("pix_per_speckle must be at least 1.")

    if polarization < 1:
        y1 = create_exponential_1D(M, pix_per_speckle, polarization=1)
        y2 = create_exponential_1D(M, pix_per_speckle, polarization=1)
        return 0.5 * (1 + polarization) * y1 + 0.5 * (1 - polarization) * y2

    x_radius = int(M / 2)

    if x_radius < 1:
        raise ValueError("M must be at least 2 so that the aperture is one pixel or more.")

    L = int(pix_per_speckle * 2 * x_radius)

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L)

    # in one dimension every aperture is just a segment
    mask = np.zeros(L, dtype=bool)
    mask[: 2 * x_radius] = True

    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fft(x))
    x = abs(x) ** 2

    # extract the first M values and normalize
    y = x[:M]
    ymax = np.max(y) or 1
    return y / ymax


def create_unpolarized_1D(M, pix_per_speckle):
    """
    Generate a length M unpolarized speckle irradiance pattern.

    The pattern is the incoherent sum of two independent fully-developed
    speckle patterns, one per polarization state.  Its irradiance therefore
    follows a gamma distribution with shape 2, and the speckle contrast is
    1/sqrt(2) rather than the unity contrast of the polarized case.

    This is the zero-polarization limit that Duncan & Kirkpatrick call
    Rayleigh, and it is exactly
    `create_exponential_1D(..., polarization=0)`.

    Args:
        M:               length of desired speckle array
        pix_per_speckle: number of pixels per smallest speckle

    Returns:
        array of length M
    """
    return create_exponential_1D(M, pix_per_speckle, polarization=0)


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
