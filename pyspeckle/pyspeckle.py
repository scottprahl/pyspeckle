# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

"""
Generate and analyze speckle patterns.

A port of the SimSpeckle collection of routines (by Duncan and Kirkpatrick)
to track and analyze laser speckle.

Documentation and examples are available at <https://pyspeckle2.readthedocs.io>
"""

import copy
import scipy.signal
import scipy.stats
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

__all__ = ('create_exp_1D',
           'create_gaussian_1D',
           'autocorrelation',
           'local_contrast_2D',
           'local_contrast_2D_plot',
           'create_Exponential',
           'create_Rayleigh',
           'statistics_plot',
           'slice_plot',
           'create_Exponential_3D',
           'create_Rayleigh_3D')


def _sqrt_matrix(x):
    """
    Generate the square root of x but scaled as integers from 0-255.

    Args:
        x: numpy array to be scaled
    nscaled array of bytes
    """
    mx = np.max(x)
    if mx == 0:
        mx = 1
    y = 255 * np.sqrt(x / mx)
    return y.astype(int)


def local_contrast_2D(x, kernel):
    """
    Calculate local (2D) spatial contrast and determine first-order statistics.

    The kernel is an N x N array that describes the region over which
    contrast should be calculated.  For example, `np.ones((5,5))` would
    represent a 5x5 square.

    Note that the dimensions of the `2D_contrast_image` will not be the same as for
    the speckle pattern as only valid pixels resulting from the convolution are
    returned.

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

    # local speckle contrast
    mu_x = scipy.signal.correlate2d(x, kernel, mode='same') / Nk
    var_x = scipy.signal.correlate2d((x - mu_x)**2, kernel, mode='same') / Nk / Nk
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

    plt.imshow(_sqrt_matrix(x), cmap='gray')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Position (pixels)')
    plt.title('Speckle Realization, Overall Contrast=%0.2f' % K)

    plt.subplot(222)
    hist, bins = np.histogram(x, bins=30)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('PDF of Speckle Realization')
    plt.xlabel('Gray level, g')
    plt.ylabel('PDF')

    plt.subplot(223)
    plt.imshow(_sqrt_matrix(C), cmap='gray')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Position (pixels)')
    plt.title('Local speckle contrast')

    plt.subplot(224)
    hist, bins = np.histogram(C, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('PDF of Local Speckle Contrast')
    plt.xlabel('Local contrast, C')
    plt.ylabel('PDF')


def _create_mask(M, x_radius, y_radius, shape='ellipse'):
    """
    Create a boolean mask for shapesize.

    The points inside the mask will be set to True.  Three shapes
    are supported: 'ellipse', 'square', or 'annulus'.

    Args:
        M:        dimension of desired image
        x_radius: half the horizontal width of the ellipse
        y_radius: half the vertical width of the ellipse
        shape:    'ellipse', 'square', or 'annulus' describing the laser shape

    Returns:
        M x M boolean array
    """
    Y, X = np.ogrid[:M, :M]

    if shape == 'square':
        dist = np.floor(X / x_radius / 2) + np.floor(Y / y_radius / 2)
        mask = dist < 1
    elif shape == 'annulus':
        rmax = max(x_radius, y_radius)
        rmin = min(x_radius, y_radius)
        dist1 = np.sqrt((X - rmax)**2 + (Y - rmax)**2) / rmax
        mask1 = dist1 < 1
        dist2 = np.sqrt((X - rmax)**2 + (Y - rmax)**2) / rmin
        mask2 = dist2 > 1
        mask = np.logical_and(mask2, mask1)
    else:
        dist = np.sqrt((X - x_radius)**2 / x_radius**2
                       + (Y - y_radius)**2 / y_radius**2)
        mask = dist < 1
    return mask


def create_exp_1D(M, mean, stdev, cl):
    """
    Generate an array of length M of values with exponential autocorrelation.

    The returned array will have the autocorrelation function exp(-x/cl).

    The speckle pattern will also have a normal probability density function
    with the specified mean and standard deviation.

    see https://www.cmu.edu/biolphys/deserno/pdf/corr_gaussian_random.pdf

    Args:
        M:     dimension of desired array     [-]
        mean:  average value of signal        [gray levels]
        std:   standard deviation of signal   [gray levels]
        cl:    correlation length             [# of pixels]

    Returns:
        array of length M
    """
    f = np.exp(-1 / cl)
    fsqrt = np.sqrt(1 - f * f)

    # gaussian deviates with mean=0 and variance=1
    g = np.random.normal(size=M)
    r = np.zeros(M)

    r[0] = g[0]
    for i in range(1, M):
        r[i] = f * r[i - 1] + fsqrt * g[i]

    return mean + stdev * r


def create_gaussian_1D(M, mean, stdev, cl):
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
        std:   standard deviation of signal   [gray levels]
        cl:    correlation length             [# of pixels]

    Returns:
        array of length M
    """
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
    mean = np.mean(x)
    x -= mean
    result = np.correlate(x, x, mode='full')
# could also use the faster(?)
#   result = signal.fftconvolve(sig, sig[::-1], mode='full')

    mx = np.max(result)
    middle = len(result) // 2
    return result[middle:] / mx


def create_Exponential(M, pix_per_speckle, alpha=1, shape='ellipse', polarization=1):
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
    is circular and `alpha=2` will have speckles that are twice as tall as
    they are wide.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008)

    Args:
        M:               dimension of desired square speckle image
        pix_per_speckle: number of pixels per smallest speckle.
        alpha:           ratio of horizontal to vertical speckle size
        shape:           'ellipse', 'square', or 'annulus'
        polarization:    degree of polarization

    Returns:
        M x M speckle image
    """
    if polarization < 1:
        y1 = create_Exponential(M, pix_per_speckle, alpha=alpha, shape=shape, polarization=1)
        y2 = create_Exponential(M, pix_per_speckle, alpha=alpha, shape=shape, polarization=1)
        return 0.5 * (1 + polarization) * y1 + 0.5 * (1 - polarization) * y2

    x_radius = int(M / 2)
    y_radius = int(alpha * M / 2)

    L = pix_per_speckle * 2 * max(x_radius, y_radius)

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L)

    mask = _create_mask(L, x_radius, y_radius, shape=shape)

    # generate circular fill pattern
    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fft2(x))
    x = abs(x)**2

    # extract the M x M matrix and normalize
    y = x[:M, :M]
    ymax = np.max(y)
    return y / ymax


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

    Returns:
        nothing
    """
    mymap = copy.copy(matplotlib.cm.get_cmap("gray"))
    mymap.set_bad('blue')

    try:
        y = x.compressed()  # if masked array
    except AttributeError:
        y = x               # not a masked array

    ave = np.mean(y)
    std = np.std(y)

    if initialize:
        plt.subplots(2, 2, figsize=(14, 12))

    # Speckle Realization
    plt.subplot(2, 2, 1)
    plt.imshow(_sqrt_matrix(x), cmap=mymap)
    plt.title('Sqrt() of Speckle Irradiance')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Position (pixels)')

    # Histogram of Probability Distribution Function
    plt.subplot(2, 2, 2)
    num_bins = 30
#    plt.gca().set_aspect('equal')
    hist, bins = np.histogram(y, bins=num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='gray')
    plt.xlabel('Irradiance (gray level/pixel)')
    plt.ylabel(r'Probability Distribution Function, $p_I(i)$')
    plt.title('Average = %.2f, Standard Deviation = %.2f' % (ave, std))

    # Power Spectral Density
    plt.subplot(2, 2, 3)
    plt.gca().set_aspect('equal')
    psd = np.fft.fftshift(np.fft.fft2(x))
    psd = 2 * np.log(abs(psd))
    plt.imshow(psd, cmap=mymap, extent=[-0.5, 0.5, -0.5, 0.5])
    plt.title('Log() of Power Spectral Density')
    plt.xlabel('Spatial Frequency (1/pixels)')
    plt.ylabel('Spatial Frequency (1/pixels)')

    # Probability Distribution Function on Log Scale
    plt.subplot(2, 2, 4)
#    plt.gca().set_aspect('equal')
#    pdf = hist / (np.sum(hist) * (bins[1] - bins[0]))

    plt.semilogy(center, hist, 'r.')
    plt.title('Speckle Contrast, K=%.3f' % (std / ave))
    plt.xlabel('Irradiance')
    plt.ylabel(r'Probability Distribution Function, $p_I(i)$')


def create_Rayleigh(N, pix_per_speckle, alpha=1, shape='ellipse'):
    """
    Generate an M x M unpolarized speckle irradiance pattern.

    The speckle pattern will have a Rayleigh distribution and results from
    the incoherent sum of two speckle patterns.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    Non-circular speckle is supported using `alpha`.  This is defined as the
    ratio of horizontal speckle size to vertical speckle size.  `alpha=1`
    is circular and `alpha=2` will have speckles that are twice as tall as
    they are wide.

    Args:
        M:                dimension of desired square speckle image
        pix_per_speckle:  number of pixels per smallest speckle.
        alpha:            ratio of horizontal width to vertical width
        shape:            'ellipse' or 'square' describing the laser shape

    Returns:
        M x M speckle image
    """
    y1 = create_Exponential(N, pix_per_speckle, shape=shape, alpha=alpha)
    y2 = create_Exponential(N, pix_per_speckle, shape=shape, alpha=alpha)
    return (y1 + y2) / 2


def _create_mask_3D(M, x_radius, y_radius, z_radius, shape='ellipsoid'):
    """
    Create 3D boolean mask for designated shape.

    The points inside the mask will be set to True.  Three shapes
    are supported: 'cube', 'shell', or 'ellipsoid'.

    Args:
        M:        dimension of desired image
        x_radius: half the horizontal width of the ellipse
        y_radius: half the vertical width of the ellipse
        z_radius: half the vertical width of the ellipse
        shape:    'ellipse', 'square', or 'annulus' describing the laser shape

    Returns:
        M x M boolean array
    """
    X, Y, Z = np.ogrid[:M, :M, :M]

    if shape == 'cube':
        dist = np.floor(X / x_radius / 2) + np.floor(Y / y_radius / 2) + np.floor(Z / z_radius / 2)
        mask = dist < 1
    elif shape == 'shell':
        rmax = max(x_radius, y_radius, z_radius)
        rmin = min(x_radius, y_radius, z_radius)
        dist1 = np.sqrt((X - rmax)**2 + (Y - rmax)**2 + (Z - rmax)**2) / rmax
        mask1 = dist1 < 1
        dist2 = np.sqrt((X - rmax)**2 + (Y - rmax)**2 + (Z - rmax)**2) / rmin
        mask2 = dist2 > 1
        mask = np.logical_and(mask2, mask1)
    else:
        dist = np.sqrt((X - x_radius)**2 / x_radius**2
                       + (Y - y_radius)**2 / y_radius**2
                       + (Z - z_radius)**2 / z_radius**2)
        mask = dist <= 1
    return mask


def create_Exponential_3D(M, pix_per_speckle, alpha=1, beta=1, shape='ellipsoid', polarization=1):
    """
    Generate an M x M x M polarized, fully-developed speckle irradiance pattern.

    The speckle pattern will have an exponential probability distribution
    function that is spatially bandwidth-limited by the specified pixels per
    speckle.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    Non-circular speckle is supported using `alpha` and `beta`.  This is defined
    as the ratio of x-speckle size to y-speckle size (or x to z).  `alpha=1`
    is circular and `alpha=2` will have speckles that with y-dimensions that
    are twice the x-dimension.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008)

    Args:
        M:               dimension of desired square speckle image
        pix_per_speckle: number of pixels per smallest speckle.
        alpha:           ratio of x to y speckle size
        beta:            ratio of x to z speckle size
        shape:           'cube', 'shell', or 'ellipsoid'
        polarization:    degree of polarization (0-1)

    Returns:
        M x M X M speckle image
    """
    if polarization < 1:
        y1 = create_Exponential_3D(M, pix_per_speckle, alpha=alpha, shape=shape, polarization=1)
        y2 = create_Exponential_3D(M, pix_per_speckle, alpha=alpha, shape=shape, polarization=1)
        return 0.5 * (1 + polarization) * y1 + 0.5 * (1 - polarization) * y2

    x_radius = int(M / 2)
    y_radius = int(alpha * M / 2)
    z_radius = int(beta * M / 2)

    L = pix_per_speckle * 2 * max(x_radius, y_radius, z_radius)

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L, L)

    mask = _create_mask_3D(L, x_radius, y_radius, z_radius, shape=shape)

    # generate circular fill pattern
    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fftn(x))
    x = abs(x)**2

    # extract the M x M matrix and normalize
    y = x[:M, :M, :M]
    ymax = np.max(y)
    return y / ymax


def create_Rayleigh_3D(M, pix_per_speckle, alpha=1, beta=1, shape='ellipsoid'):
    """
    Generate an M x M x M unpolarized speckle irradiance pattern.

    The speckle pattern will have a Rayleigh distribution and results from
    the incoherent sum of two speckle patterns.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    Non-circular speckle is supported using `alpha`.  This is defined as the
    ratio of horizontal speckle size to vertical speckle size.  `alpha=1`
    is circular and `alpha=2` will have speckles that are twice as tall as
    they are wide.

    Args:
        M:                dimension of desired square speckle image
        pix_per_speckle:  number of pixels per smallest speckle.
        alpha:            ratio of x to y speckle size
        beta:             ratio of x to z speckle size
        shape:           'cube', 'shell', or 'ellipsoid'

    Returns:
        M x M X M speckle image
    """
    return create_Exponential_3D(M, pix_per_speckle, alpha, beta, shape, 0)


def slice_plot(data, x, y, z, initialize=True, show_sqrt=True):
    """
    Plot the x, y, and z slices of 3D data cube.

    Args:
        data:       3D speckle pattern to be plotted
        x: constant x slice
        y: constant y slice
        z: constant z slice

    Returns:
        nothing
    """
    mymap = copy.copy(matplotlib.cm.get_cmap("gray"))
    mymap.set_bad('blue')

    if initialize:
        plt.subplots(2, 2, figsize=(14, 12))

    plt.subplot(2, 2, 1)
    plt.gca().set_aspect('equal')
    zz = data[:, :, z]
    if show_sqrt:
        zz = _sqrt_matrix(zz)
    plt.imshow(zz, cmap=mymap)
    plt.title('Constant Z=%d values' % z)
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')

    plt.subplot(2, 2, 2)
    plt.gca().set_aspect('equal')
    yy = data[:, y, :]
    if show_sqrt:
        yy = _sqrt_matrix(yy)
    plt.imshow(yy, cmap=mymap)
    plt.title('Constant Y=%d values' % y)
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Z Position (pixels)')

    plt.subplot(2, 2, 3)
    plt.gca().set_aspect('equal')
    xx = data[x, :, :]
    if show_sqrt:
        xx = _sqrt_matrix(xx)
    plt.imshow(xx, cmap=mymap)
    plt.title('Constant X=%d values' % x)
    plt.xlabel('Y Position (pixels)')
    plt.ylabel('Z Position (pixels)')

    plt.subplot(2, 2, 4)
    plt.gca().axis('off')


def box_muller(mu, sigma, N=1):
    """
    Generate random pairs of normally distributed numbers.

    Box and Muller generates pairs of independent, standard,
    normally distributed (zero expectation, unit variance) random numbers,
    given a source of uniformly distributed random numbers.

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
    Generate random pairs for the CDF a normal distribution.

    The z-values are from the cumulative distribution function of the
    normal distribution.

    Args:
        r: radius of the CDF
        N: number of pairs to generate

    Returns:
        pairs of random numbers
    """
    y1, y2 = box_muller(0, 1, N)
    z1 = (np.sqrt(1 + r) * y1 - np.sqrt(1 - r) * y2) / np.sqrt(2)
    z2 = (np.sqrt(1 + r) * y1 + np.sqrt(1 - r) * y2) / np.sqrt(2)
    return z1, z2


def tvalues(r, N=1):
    """
    Generate random pairs for the student t-distribution.

    Args:
        r: radius of the CDF
        N: number of pairs to generate

    Returns:
        pairs of random numbers
    """
    z1, z2 = zvalues(r, N=N)
    t1 = scipy.stats.norm.cdf(z1)
    t2 = scipy.stats.norm.cdf(z2)
    return t1, t2
