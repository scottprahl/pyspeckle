"""
Plots of speckle patterns and their statistics.

`statistics_plot` and `local_contrast_plot` take the dimensionality from the
array they are given: a trace is drawn as points and an image as an image.
A volume needs different treatment entirely and is shown as orthogonal slices
by `slice_plot`.

The generators themselves are dimension agnostic and live in `core`.
"""

import numpy as np
import matplotlib.pyplot as plt

from .core import _sqrt_matrix, local_contrast

__all__ = (
    "statistics_plot",
    "local_contrast_plot",
    "slice_plot",
)


def _pdf_bar(values, bins, title, color=None):
    """
    Draw a probability density histogram into the current subplot.

    Args:
        values: data to bin
        bins:   number of bins
        title:  title for the panel
        color:  bar color, or None for the matplotlib default

    Returns:
        bin centers and the density in each bin
    """
    # density=True scales the bins so that the PDF integrates to unity
    pdf, edges = np.histogram(values, bins=bins, density=True)
    width = 0.7 * (edges[1] - edges[0])
    center = (edges[:-1] + edges[1:]) / 2
    plt.bar(center, pdf, align="center", width=width, color=color)
    plt.title(title)
    return center, pdf


def statistics_plot(x, initialize=True):
    """
    Plot the first and second-order statistics of a speckle pattern.

    The dimensionality is taken from `x`, so the same call works for a trace
    and for an image.  A one-dimensional pattern is drawn as points and its
    power spectral density as a curve; a two-dimensional one is drawn as an
    image and its power spectral density as a log image.  The two histogram
    panels are the same either way.

    The PDF conforms to the formal definition that it integrate to unity.
    Also displayed is the contrast defined as the quotient of the standard
    deviation and the mean, which is one for fully developed polarized
    speckle.

    The power spectral density shows the bandwidth limit imposed by the
    aperture.  When it reaches the edge of the plot the pattern is sampled at
    Nyquist, two pixels per speckle; when it fills half the plot the smallest
    speckle is four pixels across.  In two dimensions the criterion applies
    separately along each axis, so the pattern need not be isotropic.

    In two dimensions the pattern is displayed as the square root of its
    irradiance, which compresses the dynamic range; fully developed speckle
    has such high contrast that displaying the irradiance itself hides the
    detail.

    For a volume use `slice_plot`, or pass a single slice here.

    Args:
        x:          1D or 2D speckle pattern to be analyzed
        initialize: boolean to initialize the plot

    Returns:
        nothing
    """
    ndim = np.ndim(x)
    if ndim not in (1, 2):
        raise ValueError("statistics_plot needs a 1D or 2D pattern; use slice_plot for a volume.")

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
    if ndim == 1:
        plt.plot(x, ".", markersize=2)
        plt.title("Speckle Irradiance, Contrast K=%.2f" % (std / ave))
        plt.ylabel("Irradiance")
    else:
        # with_extremes returns a new colormap, so the registered "gray" is untouched
        mymap = plt.get_cmap("gray").with_extremes(bad="blue")
        plt.imshow(_sqrt_matrix(x), cmap=mymap)
        plt.title("Sqrt() of Speckle Irradiance")
        plt.ylabel("Position (pixels)")
    plt.xlabel("Position (pixels)")

    # Histogram of Probability Distribution Function
    plt.subplot(2, 2, 2)
    center, pdf = _pdf_bar(y, 30, "Average = %.2f, Standard Deviation = %.2f" % (ave, std), color="gray")
    plt.xlabel("Irradiance (gray level/pixel)")
    plt.ylabel(r"Probability Distribution Function, $p_I(i)$")

    # Power Spectral Density
    plt.subplot(2, 2, 3)
    if ndim == 1:
        psd = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
        freq = np.fft.fftshift(np.fft.fftfreq(len(x)))
        plt.semilogy(freq, psd, lw=0.5)
        plt.title("Power Spectral Density")
        plt.ylabel("PSD")
        plt.xlim(-0.5, 0.5)
    else:
        plt.gca().set_aspect("equal")
        psd = 2 * np.log(abs(np.fft.fftshift(np.fft.fft2(x))))
        plt.imshow(psd, cmap=plt.get_cmap("gray").with_extremes(bad="blue"), extent=[-0.5, 0.5, -0.5, 0.5])
        plt.title("Log() of Power Spectral Density")
        plt.ylabel("Spatial Frequency (1/pixels)")
    plt.xlabel("Spatial Frequency (1/pixels)")

    # Probability Distribution Function on Log Scale
    plt.subplot(2, 2, 4)
    plt.semilogy(center, pdf, "r.")
    plt.title("Speckle Contrast, K=%.3f" % (std / ave))
    plt.xlabel("Irradiance")
    plt.ylabel(r"Probability Distribution Function, $p_I(i)$")


def local_contrast_plot(x, kernel):
    """
    Create a graph showing local and global speckle contrast.

    The dimensionality is taken from `x`.  A one-dimensional pattern and its
    contrast are drawn as points, because adjacent samples are discrete and a
    connecting line turns solid once the trace is more than a few thousand
    points long; a two-dimensional one is drawn as an image.  The two
    histogram panels are the same either way.

    The kernel must have the same number of dimensions as the pattern.  For
    example, `np.ones(5)` for a trace and `np.ones((5, 5))` for an image.

    Because only valid positions of the correlation are returned, the contrast
    is smaller than the pattern.  In one dimension it is plotted against the
    centre of its window so that the two left panels share a position axis.

    Args:
        x:       1D or 2D speckle pattern for which contrast is calculated
        kernel:  region over which contrast is to be calculated

    Returns:
        nothing
    """
    ndim = np.ndim(x)
    if ndim not in (1, 2):
        raise ValueError("local_contrast_plot needs a 1D or 2D pattern.")

    C, K = local_contrast(x, kernel)

    plt.subplots(2, 2, figsize=(14, 12))

    plt.subplot(221)
    if ndim == 1:
        plt.plot(x, ".", markersize=2)
        plt.ylabel("Irradiance")
    else:
        plt.imshow(_sqrt_matrix(x), cmap="gray")
        plt.ylabel("Position (pixels)")
    plt.xlabel("Position (pixels)")
    plt.title("Speckle Realization, Overall Contrast=%0.2f" % K)

    plt.subplot(222)
    _pdf_bar(x, 30, "PDF of Speckle Realization")
    plt.xlabel("Gray level, g")
    plt.ylabel("PDF")

    plt.subplot(223)
    if ndim == 1:
        # shift by half a kernel so each contrast lands at the centre of its window
        plt.plot(np.arange(len(C)) + (len(kernel) - 1) / 2, C, ".", markersize=2)
        plt.ylabel("Local contrast, C")
    else:
        plt.imshow(_sqrt_matrix(C), cmap="gray")
        plt.ylabel("Position (pixels)")
    plt.xlabel("Position (pixels)")
    plt.title("Local speckle contrast")

    plt.subplot(224)
    _pdf_bar(C, 20, "PDF of Local Speckle Contrast")
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
