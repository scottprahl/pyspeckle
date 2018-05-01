"""
Useful basic routines for analyzing speckle

To do:
    * Add more routines
    * Properly document

Scott Prahl
Apr 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


__all__ = ['local_contrast_2D',
           'local_contrast_2D_plt',
           'create_objective_speckle']


def local_contrast_2D(x, kernel):
    """
    Calculate local (2D) spatial contrast and determine first order statistics.
    
    Args:
    x      speckle pattern for which contrast is to be calculated
    kernel local region over which contrast is to be calculated, e.g., np.ones(7,5)

    Returns:
    C = resulting contrast image
    
    Note that the dimensions of C are not the same as for the speckle
    pattern, x as only valid pixels resulting from the convolution are
    returned.
    """
    
    plt.subplots(2,2,figsize=(14,12))
    Nk = np.sum(kernel)                                  # normalization total for kernel
    K = np.std(x)/np.mean(x)                             # contrast of raw image

    # local speckle contrast
    mu_x = signal.correlate2d(im, kernel, mode='same')/Nk
    var_x = signal.correlate2d((im-mu_x)**2, kernel, mode='same')/Nk/Nk
    C = np.sqrt(var_x) / mu_x
    return C
    
    
def local_contrast_2D_plt(x, kernel):
    """
    Create a graph showing local and global spatial contrast
    """
    
    C = local_contrast_2D(x, kernel)
    
    plt.subplot(221)
    plt.imshow(np.sqrt(x).astype(int),cmap='gray')       # sqrt is a better display
    plt.xlabel('Horizontal Position (pixels)')
    plt.ylabel('Vertical Position (pixels)')
    plt.title('Speckle Image, Overall Contrast=%0.2f'%K)

    plt.subplot(222)
    hist, bins = np.histogram(x, bins=30, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('PDF of speckle realization')
    plt.xlabel('Gray level, g')
    plt.ylabel('PDF')

    plt.subplot(223)
    scale = 255/np.sqrt(C.max())
    plt.imshow((scale*np.sqrt(C)).astype(int),cmap='gray')
    plt.xlabel('Horizontal Position (pixels)')
    plt.ylabel('Vertical Position (pixels)')
    plt.title('Local speckle contrast')

    plt.subplot(224)
    hist, bins = np.histogram(C, bins=20, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2    
    plt.bar(center, hist, align='center', width=width)
    plt.title('PDF of Local Speckle Contrast')
    plt.xlabel('Local contrast, C')
    plt.ylabel('PDF')
    return plt

def create_objective_speckle(M, pix_per_speckle, seed=None):
    """
    Generate 2-D a fully-developed objective speckle with exponential statistics

    Args:
    M                dimension of desired square speckle image 
    pix_per_speckle  number of pixels per speckle (on average).  This is the
                     factor by which pattern is low-pass filtered.
                     2 means speckle pattern is at Nyquist - two pixels per speckle
                     4 means there will be four pixels per speckle (on average)
    seed             when non-zero, will be used to seed the random number generator to a 
                     particular initial state (allowing reproducible speckle patterns).

    Returns:
                     M x M speckle image

    Note the circular mask produces speckles with isotropic second-order statistics.
    """

    if seed != None :
        np.random.seed(seed)

    L = pix_per_speckle * M

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L)

    # create circular mask with diameter M in the upper-left corner
    R = int(M/2)
    mask = _create_circular_mask(L, L, center=[R,R], radius=R)

    # generate circular fill pattern
    x = np.exp(1j * phase) * mask
    
    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fft2(x))
    x = abs(x)**2
    
    # extract M x M matrix and normalize before returning
    y = x[:M, :M]
    ymax = np.max(y)
    return y/ymax

def create_objective_speckle_square(M, pix_per_speckle, seed=None):
    """
    Generate 2-D a fully-developed objective speckle with exponential statistics

    Args:
    M                dimension of desired square speckle image 
    pix_per_speckle  number of pixels per speckle (on average).  This is the
                     factor by which pattern is low-pass filtered.
                     2 means speckle pattern is at Nyquist - two pixels per speckle
                     4 means there will be four pixels per speckle (on average)
    seed             when non-zero, will be used to seed the random number generator to a 
                     particular initial state (allowing reproducible speckle patterns).

    Returns:
                     M x M speckle image

    Note the circular mask produces speckles with isotropic second-order statistics.
    See SimSpeckle routine speckle_gen1b()
    """

    if seed != None :
        np.random.seed(seed)

    L = pix_per_speckle * M

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L)

    # generate square fill pattern
    x = np.exp(1j * phase)
    
    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fft2(x))
    x = abs(x)**2
    
    # extract M x M matrix and normalize before returning
    y = x[:M, :M]
    ymax = np.max(y)
    return y/ymax

