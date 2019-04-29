# pylint: disable=invalid-name
"""
Useful basic routines for analyzing speckle

To do:
    * Add more routines
    * Properly document

Scott Prahl
May 2018
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm

__all__ = ['create_exp_1D',
           'local_contrast_2D',
           'local_contrast_2D_plot',
           'create_Exponential',
           'create_Rayleigh',
           'statistics_plot']


def _sqrt_matrix(x):
    """
    Generate the square root of x but scaled as integers from 0-255

    Args:
        x numpy array to be scaled
    Returns:
        same size array
    """
    mx = np.max(x)
    if mx == 0:
        mx = 1
    y = 255 * np.sqrt(x / mx)
    return y.astype(int)


def local_contrast_2D(x, kernel):
    """
    Calculate local (2D) spatial contrast and determine first-order statistics.

    Args:
    x       speckle pattern for which contrast is to be calculated
    kernel  small region over which contrast is to be calculated
            e.g., np.ones((5,5))

    Returns:
            contrast_image, total_contrast

    Note that the dimensions of the contrast_image will not be the same as for
    the speckle pattern as only valid pixels resulting from the convolution are
    returned.
    """

    # normalization total for kernel
    Nk = np.sum(kernel)
    # contrast of raw image
    K = np.std(x) / np.mean(x)

    # local speckle contrast
    mu_x = signal.correlate2d(x, kernel, mode='same') / Nk
    var_x = signal.correlate2d((x - mu_x)**2, kernel, mode='same') / Nk / Nk
    C = np.sqrt(var_x) / mu_x
    return C, K


def local_contrast_2D_plot(x, kernel):
    """
    Create a graph showing local and global spatial contrast
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
    return plt


def _create_mask(M, x_radius, y_radius, spot='ellipse'):
    """
    Create an elliptical boolean mask with points inside the ellipse True

    Args:
    M                dimension of desired image
    x_radius         half the horizontal width of the ellipse
    y_radius         half the vertical width of the ellipse
    spot             'ellipse', 'square', or 'annulus' describing the laser spot

    Returns:
                     M x M boolean array
    """

    Y, X = np.ogrid[:M, :M]

    if spot == 'square':
        dist = np.floor(X / x_radius / 2) + np.floor(Y / y_radius / 2)
        mask = dist < 1
    elif spot == 'annulus':
        rmax = max(x_radius, y_radius)
        rmin = min(x_radius, y_radius)
        dist1 = np.sqrt((X - rmax)**2 + (Y - rmax)**2) / rmax
        mask1 = dist1 < 1
        dist2 = np.sqrt((X - rmax)**2 + (Y - rmax)**2) / rmin
        mask2 = dist2 > 1
        mask = np.logical_and(mask2, mask1)
    else:
        dist = np.sqrt((X - x_radius)**2 / x_radius**2 +
                       (Y - y_radius)**2 / y_radius**2)
        mask = dist < 1
    return mask


def create_exp_1D(M, mean, stdev, tau):
    """
    Generate an array of length M of values with exponential autocorrelation
    
    The returned array will the specified mean and standard deviation but
    will also have an autocorrelation function given by exp(-1/tau)

    see https://www.cmu.edu/biolphys/deserno/pdf/corr_gaussian_random.pdf

    Args:
    M     dimension of desired array
    mean  average value of signal
    std   standard deviation of signal
    tau   exp(-1/tau) is the autocorrelation

    Returns:
            array of length M
    """
    f = np.exp(-1/tau)
    fsqrt = np.sqrt(1-f*f)
    
    g = np.random.normal(size=M)  # gaussian deviates with mean=0 and variance=1
    r = np.zeros(M)
    
    r[0] = g[0]
    for i in range(1,M):
        r[i] = f * r[i-1] + fsqrt * g[i]
        
    return mean + stdev * r


def create_Exponential(M, pix_per_speckle, alpha=1, spot='ellipse', polarization=1):
    """
    Generate an M x M polarized, fully-developed speckle irradiance pattern

    The speckle pattern will have an exponential probability distribution
    function that is spatially bandwidth-limited by the specified pixels per
    speckle.

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008)

    Args:
    M                dimension of desired square speckle image
    pix_per_speckle  number of pixels per smallest speckle.
                     pix_per_speckle=2 means sampling is at Nyquist limit
                     pix_per_speckle=4 has four pixels per smallest speckle
    alpha            ratio of horizontal width to vertical width
                     1 => equal
                     2 => vertical is twice horizontal
    spot             'ellipse' or 'square' describing the laser spot

    Returns:
                     M x M speckle image
    """

    if polarization < 1:
        y1 = create_Exponential(M, pix_per_speckle, alpha=alpha, spot=spot, polarization=1)
        y2 = create_Exponential(M, pix_per_speckle, alpha=alpha, spot=spot, polarization=1)
        return 0.5*(1+polarization)*y1 + 0.5*(1-polarization)*y2

    x_radius = int(M / 2)
    y_radius = int(alpha * M / 2)

    L = pix_per_speckle * 2 * max(x_radius, y_radius)

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L)

    mask = _create_mask(L, x_radius, y_radius, spot=spot)

    # generate circular fill pattern
    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fft2(x))
    x = abs(x)**2

    # extract the M x M matrix and normalize
    y = x[:M, :M]
    ymax = np.max(y)
    return y / ymax


def statistics_plot(x):
    """
    Create a plot of the first and second-order statistics of a speckle pattern

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
    """

    mymap = matplotlib.cm.gray
    mymap.set_bad('blue')

    try:
        y = x.compressed()  # if masked array
    except AttributeError:
        y = x               # not a masked array

    ave = np.mean(y)
    std = np.std(y)

    fig, ax = plt.subplots(2, 2, figsize=(14, 12))

    # Speckle Realization
    plt.subplot(2, 2, 1)
    plt.imshow(_sqrt_matrix(x), cmap=mymap)
    plt.title('Sqrt() of Speckle Irradiance')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Position (pixels)')

    # Histogram of Probability Distribution Function
    plt.subplot(2, 2, 2)
    num_bins = 30
    ax[0, 1].set_aspect('equal')
    hist, bins = np.histogram(y, bins=num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='gray')
    plt.xlabel('Irradiance (gray level/pixel)')
    plt.ylabel(r'Probability Distribution Function, $p_I(i)$')
    plt.title('Average = %.2f, Standard Deviation = %.2f' % (ave, std))

    # Power Spectral Density
    plt.subplot(2, 2, 3)
    ax[1, 0].set_aspect('equal')
    psd = np.fft.fftshift(np.fft.fft2(x))
    psd = 2 * np.log(abs(psd))
    plt.imshow(psd, cmap=mymap, extent=[-0.5, 0.5, -0.5, 0.5])
    plt.title('Log() of Power Spectral Density')
    plt.xlabel('Spatial Frequency (1/pixels)')
    plt.ylabel('Spatial Frequency (1/pixels)')

    # Probability Distribution Function on Log Scale
    plt.subplot(2, 2, 4)
    ax[1, 1].set_aspect('equal')
 #    pdf = hist / (np.sum(hist) * (bins[1] - bins[0]))

    plt.semilogy(center, hist, 'r.')
    plt.title('Speckle Contrast, K=%.3f' % (std / ave))
    plt.xlabel('Irradiance')
    plt.ylabel(r'Probability Distribution Function, $p_I(i)$')

    return plt


def create_Rayleigh(N, pix_per_speckle, alpha=1, spot='ellipse'):
    """
    Generate an M x M unpolarized speckle irradiance pattern.

    The speckle pattern will have a Rayleigh distribution and results from
    the incoherent sum of two speckle patterns.

    Args:
    M                dimension of desired square speckle image
    pix_per_speckle  number of pixels per smallest speckle.
                     pix_per_speckle=2 means sampling is at Nyquist limit
                     pix_per_speckle=4 has four pixels per smallest speckle
    alpha            ratio of horizontal width to vertical width
                     1 => equal
                     2 => vertical is twice horizontal
    spot             'ellipse' or 'square' describing the laser spot

    Returns:
                     M x M speckle image
    """
    y1 = create_Exponential(N, pix_per_speckle, spot=spot, alpha=alpha)
    y2 = create_Exponential(N, pix_per_speckle, spot=spot, alpha=alpha)
    return (y1 + y2) / 2


def box_muller(mu, sigma, N=1):
    x1 = np.random.rand(N)
    x2 = np.random.rand(N)
    tmp = sigma * np.sqrt(-2 * np.log(x1))
    y1 = mu + tmp * np.cos(2 * np.pi * x2)
    y2 = mu + tmp * np.sin(2 * np.pi * x2)
    return y1, y2


def zvalues(r, N=1):
    y1, y2 = box_muller(0, 1, N)
    z1 = (np.sqrt(1 + r) * y1 - np.sqrt(1 - r) * y2) / np.sqrt(2)
    z2 = (np.sqrt(1 + r) * y1 + np.sqrt(1 - r) * y2) / np.sqrt(2)
    return z1, z2


def tvalues(r, N=1):
    z1, z2 = zvalues(r, N=N)
    t1 = norm.cdf(z1)
    t2 = norm.cdf(z2)
    return t1, t2
