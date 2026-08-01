"""
Plots of speckle patterns and their statistics.

Everything here is dimension specific, because a trace, an image, and a volume
need different presentation: `statistics_plot_1D` plots points where
`statistics_plot_2D` shows an image, and a volume is shown as orthogonal
slices by `slice_plot`.

The generators themselves are dimension agnostic and live in `core`.
"""

import numpy as np
import matplotlib.pyplot as plt

from .core import _sqrt_matrix, local_contrast

__all__ = (
    "statistics_plot_1D",
    "statistics_plot_2D",
    "local_contrast_1D_plot",
    "local_contrast_2D_plot",
    "slice_plot",
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


def statistics_plot_2D(x, initialize=True):
    """
    Plot the first and second-order statistics of a speckle pattern.

    This routine calculates and plots the probability density function,
    PDF and the power spectral density, PSD.

    The PDF conforms to the formal definition that it integrate to unity.
    Also displayed is the contrast defined as the quotient of the standard
    deviation and the mean.

    Note that the PSD can be used to establish the dimensions of the
    minimum speckle size. When the display reaches the edge of the image,
    the speckle pattern (in that dimension) is at Nyquist, i.e., two
    pixels per (minimum) speckle. When the display occupies half of the
    image, the minimum speckle size is four pixels, etc. Of course this
    criterion applies separately in each dimension (horizontal and
    vertical); the speckle pattern need not be isotropic.

    Finally note that the display of the speckle pattern is the square
    root of its intensity. The square root operation has the effect of
    compressing the dynamic range of the pattern. A fully developed
    speckle pattern is of such high contrast (theoretically unity) that a
    display of the intensity itself does not reveal the nuance of the
    pattern.

    Args:
        x:       speckle pattern to be analyzed
        initialize: boolean to initialize the plot

    Returns:
        nothing
    """
    # with_extremes returns a new colormap, so the registered "gray" is untouched
    mymap = plt.get_cmap("gray").with_extremes(bad="blue")

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
    plt.imshow(_sqrt_matrix(x), cmap=mymap)
    plt.title("Sqrt() of Speckle Irradiance")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Position (pixels)")

    # Histogram of Probability Distribution Function
    plt.subplot(2, 2, 2)
    num_bins = 30
    #    plt.gca().set_aspect('equal')
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
    plt.gca().set_aspect("equal")
    psd = np.fft.fftshift(np.fft.fft2(x))
    psd = 2 * np.log(abs(psd))
    plt.imshow(psd, cmap=mymap, extent=[-0.5, 0.5, -0.5, 0.5])
    plt.title("Log() of Power Spectral Density")
    plt.xlabel("Spatial Frequency (1/pixels)")
    plt.ylabel("Spatial Frequency (1/pixels)")

    # Probability Distribution Function on Log Scale
    plt.subplot(2, 2, 4)
    #    plt.gca().set_aspect('equal')
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


def local_contrast_2D_plot(x, kernel):
    """
    Create a graph showing local and global spatial contrast.

    The kernel is an N x N array that describes the region over which
    contrast should be calculated.  For example, `np.ones((5,5))` would
    represent a 5x5 square.

    Args:
        x:       speckle pattern for which contrast is to be calculated
        kernel:  small region over which contrast is to be calculated

    Returns:
        nothing
    """
    C, K = local_contrast(x, kernel)

    plt.subplots(2, 2, figsize=(14, 12))
    plt.subplot(221)

    plt.imshow(_sqrt_matrix(x), cmap="gray")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Position (pixels)")
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
    plt.imshow(_sqrt_matrix(C), cmap="gray")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Position (pixels)")
    plt.title("Local speckle contrast")

    plt.subplot(224)
    pdf, bins = np.histogram(C, bins=20, density=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, pdf, align="center", width=width)
    plt.title("PDF of Local Speckle Contrast")
    plt.xlabel("Local contrast, C")
    plt.ylabel("PDF")


def slice_plot(data, x, y, z, initialize=True, show_sqrt=True):
    """
    Plot the x, y, and z slices of 3D data cube.

    Args:
        data:       3D speckle pattern to be plotted
        x: constant x slice
        y: constant y slice
        z: constant z slice
        initialize: boolean to initialize plot
        show_sqrt: take sqrt() of image for better visualization

    Returns:
        nothing
    """
    # with_extremes returns a new colormap, so the registered "gray" is untouched
    mymap = plt.get_cmap("gray").with_extremes(bad="blue")

    if initialize:
        plt.subplots(2, 2, figsize=(9, 9))

    plt.subplot(2, 2, 1)
    plt.gca().set_aspect("equal")
    zz = data[:, :, z]
    if show_sqrt:
        zz = _sqrt_matrix(zz)
    plt.imshow(zz, cmap=mymap)
    plt.title("Constant Z=%d values" % z)
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")

    plt.subplot(2, 2, 2)
    plt.gca().set_aspect("equal")
    yy = data[:, y, :]
    if show_sqrt:
        yy = _sqrt_matrix(yy)
    plt.imshow(yy, cmap=mymap)
    plt.title("Constant Y=%d values" % y)
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Z Position (pixels)")

    plt.subplot(2, 2, 3)
    plt.gca().set_aspect("equal")
    xx = data[x, :, :]
    if show_sqrt:
        xx = _sqrt_matrix(xx)
    plt.imshow(xx, cmap=mymap)
    plt.title("Constant X=%d values" % x)
    plt.xlabel("Y Position (pixels)")
    plt.ylabel("Z Position (pixels)")

    plt.subplot(2, 2, 4)
    plt.gca().axis("off")
