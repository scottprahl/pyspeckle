"""
Generation and analysis of two-dimensional speckle patterns.

A uniform random-phase field is masked by an aperture, Fourier transformed,
and squared to give a fully developed speckle irradiance pattern.

see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
Vol. 6855 (2008)
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from .core import _sqrt_matrix

__all__ = (
    "create_exponential_2D",
    "create_rayleigh_2D",
    "local_contrast_2D",
    "local_contrast_2D_plot",
    "statistics_plot",
)


def _create_mask(M, x_radius, y_radius, shape="ellipse"):
    """
    Create a MxM boolean mask for a particular beam shape.

    The resulting shape is in the top left corner of the the returned array.

    The points inside the mask will be set to True.  Three shapes
    are supported: 'ellipse', 'rectangle', or 'annulus'.

    For example `._create_mask(10,3,4,'ellipse').astype(int)` yields

    [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    When shape is 'annulus' then the outer circle radius is the max(x_radius, y_radius)
    and then inner radius is the other.

    Args:
        M:        dimension of desired image
        x_radius: half the horizontal width of the ellipse (in pixels)
        y_radius: half the vertical width of the ellipse (in pixels)
        shape:    'ellipse', 'rectangle', or 'annulus' describing the laser shape

    Returns:
        M x M boolean array
    """
    if M < 2 * max(x_radius, y_radius):
        raise ValueError("Array size M must be at least twice the radius.")

    Y, X = np.ogrid[:M, :M]

    lshape = shape.lower()

    if lshape in ("rectangle", "square"):
        mask1 = X < 2 * x_radius
        mask2 = Y < 2 * y_radius
        mask = np.logical_and(mask2, mask1)

    elif lshape == "annulus":
        rmax = max(x_radius, y_radius)
        rmin = min(x_radius, y_radius)
        dist1 = np.sqrt((X - rmax) ** 2 + (Y - rmax) ** 2) / rmax
        mask1 = dist1 <= 1
        dist2 = np.sqrt((X - rmax) ** 2 + (Y - rmax) ** 2) / rmin
        mask2 = dist2 > 1
        mask = np.logical_and(mask2, mask1)

    elif lshape == "ellipse":
        dist = np.sqrt((X - x_radius) ** 2 / x_radius**2 + (Y - y_radius) ** 2 / y_radius**2)
        mask = dist <= 1

    else:
        raise ValueError("shape must be 'ellipse', 'rectangle', or 'annulus'")

    return mask


def create_exponential_2D(M, pix_per_speckle, alpha=1, shape="ellipse", polarization=1):
    """
    Generate an M x M polarized, fully-developed speckle irradiance pattern.

    The speckle pattern will have an exponential probability distribution
    function that is spatially bandwidth-limited by the specified pixels per
    speckle.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    Non-circular speckle is supported using `alpha`.  This is defined as the
    ratio of horizontal speckle size to vertical speckle size.  `alpha=1`
    is circular and `alpha=2` will have speckles that are twice as wide as
    they are tall.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008)

    Args:
        M:               dimension of desired square speckle image
        pix_per_speckle: number of pixels per smallest speckle.
        alpha:           ratio of horizontal to vertical speckle size
        shape:           'ellipse', 'rectangle', or 'annulus'
        polarization:    degree of polarization

    Returns:
        M x M speckle image
    """
    if polarization < 0 or polarization > 1:
        raise ValueError("bad polarization. It must be 0 <= polarization <= 1.")

    if pix_per_speckle < 1:
        raise ValueError("pix_per_speckle must be at least 1.")

    if polarization < 1:
        y1 = create_exponential_2D(M, pix_per_speckle, alpha=alpha, shape=shape, polarization=1)
        y2 = create_exponential_2D(M, pix_per_speckle, alpha=alpha, shape=shape, polarization=1)
        return 0.5 * (1 + polarization) * y1 + 0.5 * (1 - polarization) * y2

    x_radius = int(M / 2)
    y_radius = int(alpha * M / 2)

    if x_radius < 1 or y_radius < 1:
        raise ValueError("M and alpha must be big enough that both radii are at least one pixel.")

    L = int(pix_per_speckle * 2 * max(x_radius, y_radius))

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L)

    mask = _create_mask(L, x_radius, y_radius, shape=shape)

    # generate circular fill pattern
    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fft2(x))
    x = abs(x) ** 2

    # extract the M x M matrix and normalize
    y = x[:M, :M]
    ymax = np.max(y) or 1
    return y / ymax


def create_rayleigh_2D(N, pix_per_speckle, alpha=1, shape="ellipse"):
    """
    Generate an N x N unpolarized speckle irradiance pattern.

    The speckle pattern will have a Rayleigh distribution and results from
    the incoherent sum of two speckle patterns.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    Non-circular speckle is supported using `alpha`.  This is defined as the
    ratio of horizontal speckle size to vertical speckle size.  `alpha=1`
    is circular and `alpha=2` will have speckles that are twice as wide as
    they are tall.

    Args:
        N:                dimension of desired square speckle image
        pix_per_speckle:  number of pixels per smallest speckle.
        alpha:            ratio of horizontal to vertical speckle size
        shape:            'ellipse' or 'rectangle' describing the laser shape

    Returns:
        N x N speckle image
    """
    y1 = create_exponential_2D(N, pix_per_speckle, shape=shape, alpha=alpha)
    y2 = create_exponential_2D(N, pix_per_speckle, shape=shape, alpha=alpha)
    return (y1 + y2) / 2


def local_contrast_2D(x, kernel):
    """
    Calculate local (2D) spatial contrast and determine first-order statistics.

    The kernel is an N x N array that describes the region over which
    contrast should be calculated.  For example, `np.ones((5,5))` would
    represent a 5x5 square.

    Note that the dimensions of the `2D_contrast_image` will not be the same as for
    the speckle pattern as only valid pixels resulting from the convolution are
    returned.  An M x M pattern with an N x N kernel yields (M-N+1) x (M-N+1)
    values, none of which are contaminated by zero padding at the edges.

    Args:
        x: 2D speckle pattern
        kernel: 2D region over which contrast is to be calculated

    Returns:
        2D_contrast_image, total_contrast
    """
    # normalization total for kernel
    Nk = np.sum(kernel)
    # contrast of raw image
    K = np.std(x) / np.mean(x)

    # local mean and local mean square over the kernel
    mu_x = scipy.signal.correlate2d(x, kernel, mode="valid") / Nk
    mu_x2 = scipy.signal.correlate2d(x**2, kernel, mode="valid") / Nk

    # local variance, clipped because rounding can push it slightly below zero
    var_x = np.maximum(mu_x2 - mu_x**2, 0)
    C = np.sqrt(var_x) / mu_x
    return C, K


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
    C, K = local_contrast_2D(x, kernel)

    plt.subplots(2, 2, figsize=(14, 12))
    plt.subplot(221)

    plt.imshow(_sqrt_matrix(x), cmap="gray")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Position (pixels)")
    plt.title("Speckle Realization, Overall Contrast=%0.2f" % K)

    plt.subplot(222)
    hist, bins = np.histogram(x, bins=30)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align="center", width=width)
    plt.title("PDF of Speckle Realization")
    plt.xlabel("Gray level, g")
    plt.ylabel("PDF")

    plt.subplot(223)
    plt.imshow(_sqrt_matrix(C), cmap="gray")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Position (pixels)")
    plt.title("Local speckle contrast")

    plt.subplot(224)
    hist, bins = np.histogram(C, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align="center", width=width)
    plt.title("PDF of Local Speckle Contrast")
    plt.xlabel("Local contrast, C")
    plt.ylabel("PDF")


def statistics_plot(x, initialize=True):
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
    mymap = plt.get_cmap("gray").copy()
    mymap.set_bad("blue")

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
