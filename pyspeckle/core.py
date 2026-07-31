"""
Shared helpers for pyspeckle.

This module holds the pieces that are not specific to one dimensionality:
the autocorrelation used throughout, and the Gaussian copula utilities from
Duncan & Kirkpatrick used to generate correlated random sequences.
"""

import numpy as np
import scipy.signal
import scipy.stats

__all__ = (
    "autocorrelation",
    "box_muller",
    "zvalues",
    "tvalues",
)


def _phase_screen(dims, sigma, cl, shape="gaussian"):
    """
    Generate a correlated Gaussian phase screen of any dimensionality.

    This backs `create_phase_screen_1D` and `create_phase_screen_2D`.  White
    noise is filtered by the square root of the power spectral density, which
    by the Wiener-Khinchin theorem is the transform of the wanted
    autocorrelation.  The result is isotropic, which matters in two dimensions
    because `exp(-|x|/cl) * exp(-|y|/cl)` is diamond shaped rather than
    radially symmetric.

    Args:
        dims:  tuple giving the shape of the screen, e.g. (M,) or (M, M)
        sigma: standard deviation of the phase [radians]
        cl:    correlation length [pixels]
        shape: 'gaussian' or 'exponential' autocorrelation

    Returns:
        zero-mean array of phases with the requested shape
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if cl <= 0:
        raise ValueError("Correlation length cl must be positive.")

    if min(dims) < 2:
        raise ValueError("Screen must be at least 2 pixels across.")

    lshape = shape.lower()

    # signed lags, wrapped, so the autocorrelation is periodic on the grid
    axes = [np.fft.fftfreq(n, d=1 / n) for n in dims]
    grid = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(sum(g**2 for g in grid))

    if lshape == "gaussian":
        acf = np.exp(-0.5 * (r / cl) ** 2)
    elif lshape == "exponential":
        acf = np.exp(-r / cl)
    else:
        raise ValueError("shape must be 'gaussian' or 'exponential'")

    # power spectral density; tiny negative lobes are numerical noise
    psd = np.maximum(np.fft.fftn(acf).real, 0)

    white = np.fft.fftn(np.random.normal(size=dims))
    screen = np.fft.ifftn(white * np.sqrt(psd)).real

    screen -= screen.mean()
    deviation = screen.std() or 1
    return sigma * screen / deviation


def _sqrt_matrix(x):
    """
    Generate the square root of x but scaled as integers from 0-255.

    Args:
        x: numpy array to be scaled
    Returns:
        scaled array of integers
    """
    mx = np.max(x) or 1
    y = 255 * np.sqrt(x / mx)
    return y.astype(int)


def _local_contrast(x, kernel):
    """
    Calculate local speckle contrast over a sliding window.

    This backs `local_contrast_1D`, `local_contrast_2D`, and
    `local_contrast_3D`; the kernel must have the same number of dimensions
    as the speckle pattern.

    Only valid positions of the correlation are returned, so no value is
    contaminated by zero padding at the edges.  An M-long pattern with an
    N-long kernel yields M-N+1 values, and likewise per axis in 2D and 3D.

    Args:
        x: speckle pattern
        kernel: region over which contrast is to be calculated

    Returns:
        local_contrast_image, total_contrast
    """
    if np.ndim(x) != np.ndim(kernel):
        raise ValueError("kernel must have the same number of dimensions as the speckle pattern.")

    # normalization total for kernel
    Nk = np.sum(kernel)
    # contrast of raw image
    K = np.std(x) / np.mean(x)

    # local mean and local mean square over the kernel
    mu_x = scipy.signal.correlate(x, kernel, mode="valid") / Nk
    mu_x2 = scipy.signal.correlate(x**2, kernel, mode="valid") / Nk

    # local variance, clipped because rounding can push it slightly below zero
    var_x = np.maximum(mu_x2 - mu_x**2, 0)
    C = np.sqrt(var_x) / mu_x
    return C, K


def autocorrelation(x):
    """
    Find the autocorrelation of a 1D array.

    This is a little different from the standard autocorrelation because
    (1) the mean is subtracted before correlation
    (2) the autocorrelation is normalized to maximum value
    (3) only the right hand side of the symmetric function is returned

    Args:
        x: 1D array

    Returns:
        autocorrelation array of same length
    """
    xx = x.astype(float)
    mean = np.mean(x)
    xx -= mean
    result = np.correlate(xx, xx, mode="full")
    # could also use the faster(?)
    #   result = signal.fftconvolve(sig, sig[::-1], mode='full')

    mx = np.max(result) or 1
    middle = len(result) // 2
    return result[middle:] / mx


def box_muller(mu, sigma, N=1):
    """
    Generate random pairs of normally distributed numbers.

    Box and Muller generates pairs of independent, standard,
    normally distributed (zero expectation, unit variance) random numbers,
    given a source of uniformly distributed random numbers.

    This is `Y1` and `Y2` in the Gaussian copula construction.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008), eq. (5a)

    Args:
        mu: average value
        sigma: standard deviation of normal distribution
        N: number of pairs to generate

    Returns:
        pairs of random numbers
    """
    x1 = np.random.rand(N)
    x2 = np.random.rand(N)
    tmp = sigma * np.sqrt(-2 * np.log(x1))
    y1 = mu + tmp * np.cos(2 * np.pi * x2)
    y2 = mu + tmp * np.sin(2 * np.pi * x2)
    return y1, y2


def zvalues(r, N=1):
    """
    Generate correlated pairs of standard normal deviates.

    Each returned array is normally distributed with zero mean and unit
    variance, and the two arrays have correlation coefficient `r`.

    This is `Z1` and `Z2` in the Gaussian copula construction, obtained by
    scaling and rotating the independent pair from `box_muller`.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008), eq. (7)

    Args:
        r: correlation coefficient of the pair, -1 <= r <= 1
        N: number of pairs to generate

    Returns:
        pairs of correlated standard normal deviates
    """
    y1, y2 = box_muller(0, 1, N)
    z1 = (np.sqrt(1 + r) * y1 - np.sqrt(1 - r) * y2) / np.sqrt(2)
    z2 = (np.sqrt(1 + r) * y1 + np.sqrt(1 - r) * y2) / np.sqrt(2)
    return z1, z2


def tvalues(r, N=1):
    """
    Generate correlated pairs of uniformly distributed numbers.

    Correlated standard normal deviates are mapped through the normal CDF,
    so each returned array is uniform on [0, 1] while the pair remains
    dependent.  This is `T1` and `T2`, the percentile transformation that
    completes the Gaussian copula construction.

    The `T` is the paper's variable name and has nothing to do with the
    Student t-distribution.  Note also that `r` sets the correlation of the
    underlying normal pair; the correlation of the returned uniforms is
    (6/pi)*arcsin(r/2), which is 0.483 when r=0.5.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008), eq. (8)

    Args:
        r: correlation coefficient of the underlying normal pair
        N: number of pairs to generate

    Returns:
        pairs of correlated numbers uniform on [0, 1]
    """
    z1, z2 = zvalues(r, N=N)
    t1 = scipy.stats.norm.cdf(z1)
    t2 = scipy.stats.norm.cdf(z2)
    return t1, t2
