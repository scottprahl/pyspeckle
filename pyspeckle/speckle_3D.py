"""
Generation and analysis of three-dimensional speckle patterns.

The 3D generators follow the same masked random-phase recipe as the 2D ones,
with an ellipsoidal aperture and a three-dimensional transform.

see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
Vol. 6855 (2008)
"""

import numpy as np
import matplotlib.pyplot as plt

from .core import _sqrt_matrix

__all__ = (
    "create_exponential_3D",
    "create_unpolarized_3D",
    "slice_plot",
)


def _create_mask_3D(M, x_radius, y_radius, z_radius, shape="ellipsoid"):
    """
    Create 3D boolean mask for designated shape.

    The points inside the mask will be set to True.  Three shapes
    are supported: 'cube', 'shell', or 'ellipsoid'.

    Args:
        M:        dimension of desired image
        x_radius: half the width of the ellipsoid along x (in pixels)
        y_radius: half the width of the ellipsoid along y (in pixels)
        z_radius: half the width of the ellipsoid along z (in pixels)
        shape:    'cube', 'shell', or 'ellipsoid' describing the laser shape

    Returns:
        M x M x M boolean array
    """
    if M < 2 * max(x_radius, y_radius, z_radius):
        raise ValueError("Array size M must be at least twice the radius.")

    X, Y, Z = np.ogrid[:M, :M, :M]

    lshape = shape.lower()

    if lshape == "cube":
        dist = np.floor(X / x_radius / 2) + np.floor(Y / y_radius / 2) + np.floor(Z / z_radius / 2)
        mask = dist < 1
    elif lshape == "shell":
        rmax = max(x_radius, y_radius, z_radius)
        rmin = min(x_radius, y_radius, z_radius)
        dist1 = np.sqrt((X - rmax) ** 2 + (Y - rmax) ** 2 + (Z - rmax) ** 2) / rmax
        mask1 = dist1 < 1
        dist2 = np.sqrt((X - rmax) ** 2 + (Y - rmax) ** 2 + (Z - rmax) ** 2) / rmin
        mask2 = dist2 > 1
        mask = np.logical_and(mask2, mask1)
    elif lshape == "ellipsoid":
        dist = np.sqrt(
            (X - x_radius) ** 2 / x_radius**2 + (Y - y_radius) ** 2 / y_radius**2 + (Z - z_radius) ** 2 / z_radius**2
        )
        mask = dist <= 1
    else:
        raise ValueError("shape must be 'cube', 'shell', or 'ellipsoid'")

    return mask


def create_exponential_3D(M, pix_per_speckle, alpha=1, beta=1, shape="ellipsoid", polarization=1):
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
    and `beta=1` is spherical; `alpha=2` will have speckles whose x-dimension
    is twice the y-dimension, and `beta=2` twice the z-dimension.

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
    if polarization < 0 or polarization > 1:
        raise ValueError("bad polarization. It must be 0 <= polarization <= 1.")

    if pix_per_speckle < 1:
        raise ValueError("pix_per_speckle must be at least 1.")

    if polarization < 1:
        y1 = create_exponential_3D(M, pix_per_speckle, alpha=alpha, beta=beta, shape=shape, polarization=1)
        y2 = create_exponential_3D(M, pix_per_speckle, alpha=alpha, beta=beta, shape=shape, polarization=1)
        return 0.5 * (1 + polarization) * y1 + 0.5 * (1 - polarization) * y2

    x_radius = int(M / 2)
    y_radius = int(alpha * M / 2)
    z_radius = int(beta * M / 2)

    if x_radius < 1 or y_radius < 1 or z_radius < 1:
        raise ValueError("M, alpha, and beta must be big enough that all radii are at least one pixel.")

    L = int(pix_per_speckle * 2 * max(x_radius, y_radius, z_radius))

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(L, L, L)

    mask = _create_mask_3D(L, x_radius, y_radius, z_radius, shape=shape)

    # generate circular fill pattern
    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fftn(x))
    x = abs(x) ** 2

    # extract the M x M matrix and normalize
    y = x[:M, :M, :M]
    ymax = np.max(y) or 1
    return y / ymax


def create_unpolarized_3D(M, pix_per_speckle, alpha=1, beta=1, shape="ellipsoid"):
    """
    Generate an M x M x M unpolarized speckle irradiance pattern.

    The pattern is the incoherent sum of two independent fully-developed
    speckle patterns, one per polarization state.  Its irradiance therefore
    follows a gamma distribution with shape 2, and the speckle contrast is
    1/sqrt(2) rather than the unity contrast of the polarized case.

    This is the zero-polarization limit that Duncan & Kirkpatrick call
    Rayleigh, and it is exactly
    `create_exponential_3D(..., polarization=0)`.

    The resolution is specified by the parameter `pix_per_speckle` and refers
    to the smallest speckle size.  Thus `pix_per_speckle=2` means sampling is
    at the Nyquist limit and `pix_per_speckle=4` will have four pixels across
    the smallest speckle.

    Non-spherical speckle is supported using `alpha` and `beta`.  This is
    defined as the ratio of x-speckle size to y-speckle size (or x to z).
    `alpha=1` and `beta=1` is spherical; `alpha=2` will have speckles whose
    x-dimension is twice the y-dimension, and `beta=2` twice the z-dimension.

    Args:
        M:                dimension of desired square speckle image
        pix_per_speckle:  number of pixels per smallest speckle.
        alpha:            ratio of x to y speckle size
        beta:             ratio of x to z speckle size
        shape:           'cube', 'shell', or 'ellipsoid'

    Returns:
        M x M X M speckle image
    """
    return create_exponential_3D(M, pix_per_speckle, alpha, beta, shape, 0)


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
    mymap = plt.get_cmap("gray").copy()
    mymap.set_bad("blue")

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
