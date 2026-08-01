"""
One-dimensional speckle analysis and correlated random sequences.

Speckle traces themselves come from `create_exponential` in `core`.  What
lives here is specific to one dimension.

`create_exp_corr_1D` and `create_gaussian_corr_1D` shape Gaussian deviates to
a target autocorrelation length -- the first with an AR(1) recursion, the
second by FFT convolution with a Gaussian kernel.  Both return normally
distributed values, and the word names the autocorrelation, not the amplitude
distribution.  They are not speckle.
"""

import numpy as np
import matplotlib.pyplot as plt

from .core import local_contrast

__all__ = (
    "create_exp_corr_1D",
    "create_gaussian_corr_1D",
    "local_contrast_1D_plot",
    "statistics_plot_1D",
)


def statistics_plot_1D(x, initialize=True):
    """
    Plot the first and second-order statistics of a 1D speckle pattern.

    This is the one-dimensional form of `statistics_plot_2D`.  Four panels are
    drawn: the trace itself, the probability density function of the
    irradiance, the power spectral density, and the same density on a log
    scale.

    The PDF conforms to the formal definition that it integrate to unity.
    Also displayed is the contrast defined as the quotient of the standard
    deviation and the mean, which is one for fully developed polarized
    speckle.

    The power spectral density shows the bandwidth limit imposed by the
    aperture.  When it reaches the edge of the plot the pattern is sampled at
    Nyquist, two pixels per speckle; when it fills half the plot the smallest
    speckle is four pixels across.

    Args:
        x:          1D speckle pattern to be analyzed
        initialize: boolean to initialize the plot

    Returns:
        nothing
    """
    try:
        y = x.compressed()  # if masked array
    except AttributeError:
        y = x  # not a masked array

    ave = np.mean(y)
    std = np.std(y)

    if initialize:
        plt.subplots(2, 2, figsize=(14, 12))

    # Speckle Realization
    plt.subplot(2, 2, 1)
    plt.plot(x, ".", markersize=2)
    plt.title("Speckle Irradiance, Contrast K=%.2f" % (std / ave))
    plt.xlabel("Position (pixels)")
    plt.ylabel("Irradiance")

    # Histogram of Probability Distribution Function
    plt.subplot(2, 2, 2)
    num_bins = 30
    # density=True scales the bins so that the PDF integrates to unity
    pdf, bins = np.histogram(y, bins=num_bins, density=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, pdf, align="center", width=width, color="gray")
    plt.xlabel("Irradiance (gray level/pixel)")
    plt.ylabel(r"Probability Distribution Function, $p_I(i)$")
    plt.title("Average = %.2f, Standard Deviation = %.2f" % (ave, std))

    # Power Spectral Density
    plt.subplot(2, 2, 3)
    psd = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(len(x)))
    plt.semilogy(freq, psd, lw=0.5)
    plt.title("Power Spectral Density")
    plt.xlabel("Spatial Frequency (1/pixels)")
    plt.ylabel("PSD")
    plt.xlim(-0.5, 0.5)

    # Probability Distribution Function on Log Scale
    plt.subplot(2, 2, 4)
    plt.semilogy(center, pdf, "r.")
    plt.title("Speckle Contrast, K=%.3f" % (std / ave))
    plt.xlabel("Irradiance")
    plt.ylabel(r"Probability Distribution Function, $p_I(i)$")


def local_contrast_1D_plot(x, kernel):
    """
    Create a graph showing local and global speckle contrast.

    This is the one-dimensional form of `local_contrast_2D_plot`.  The two
    panels on the left show individual samples as small points rather than
    images: the speckle trace above, the local contrast below.  Points rather
    than a line because adjacent samples are discrete and a connecting line
    turns solid once the trace is more than a few thousand points long.  The
    two panels on the right are the matching histograms.

    The kernel is a 1D array describing the window over which contrast is
    calculated.  For example, `np.ones(5)` averages over five samples.

    Because only valid positions of the correlation are returned, the contrast
    trace is shorter than the speckle trace.  It is plotted against the centre
    of its window so that the two left panels share a position axis.

    Args:
        x:       1D speckle pattern for which contrast is to be calculated
        kernel:  1D window over which contrast is to be calculated

    Returns:
        nothing
    """
    C, K = local_contrast(x, kernel)

    plt.subplots(2, 2, figsize=(14, 12))
    plt.subplot(221)

    plt.plot(x, ".", markersize=2)
    plt.xlabel("Position (pixels)")
    plt.ylabel("Irradiance")
    plt.title("Speckle Realization, Overall Contrast=%0.2f" % K)

    plt.subplot(222)
    # density=True scales the bins so that the PDF integrates to unity
    pdf, bins = np.histogram(x, bins=30, density=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, pdf, align="center", width=width)
    plt.title("PDF of Speckle Realization")
    plt.xlabel("Gray level, g")
    plt.ylabel("PDF")

    plt.subplot(223)
    # shift by half a kernel so each contrast lands at the centre of its window
    plt.plot(np.arange(len(C)) + (len(kernel) - 1) / 2, C, ".", markersize=2)
    plt.xlabel("Position (pixels)")
    plt.ylabel("Local contrast, C")
    plt.title("Local speckle contrast")

    plt.subplot(224)
    pdf, bins = np.histogram(C, bins=20, density=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, pdf, align="center", width=width)
    plt.title("PDF of Local Speckle Contrast")
    plt.xlabel("Local contrast, C")
    plt.ylabel("PDF")


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
