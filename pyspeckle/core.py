"""
Speckle generation and the machinery shared across dimensions.

`create_exponential` and `create_unpolarized` build speckle in one, two, or
three dimensions from a numpy-style shape, so they live here rather than in a
dimension-specific module.  This module also holds the autocorrelation used
throughout and the Gaussian copula utilities from Duncan & Kirkpatrick.
"""

import numpy as np
import scipy.signal
import scipy.stats

__all__ = (
    "create_exponential",
    "local_contrast",
    "create_unpolarized",
    "create_from_aperture",
    "create_boiling_speckle",
    "create_oct_speckle",
    "autocorrelation",
    "box_muller",
    "zvalues",
    "tvalues",
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


def _validate_speckle(pix_per_speckle, polarization):
    """
    Check the two arguments every speckle generator shares.

    Args:
        pix_per_speckle: number of pixels per smallest speckle
        polarization:    degree of polarization

    Returns:
        nothing
    """
    if polarization < 0 or polarization > 1:
        raise ValueError("bad polarization. It must be 0 <= polarization <= 1.")

    if pix_per_speckle < 1:
        raise ValueError("pix_per_speckle must be at least 1.")


def _speckle_from_mask(mask, dims, polarization=1):
    """
    Build a speckle pattern from an aperture mask of any dimensionality.

    This backs `create_exponential`.  The aperture is filled with uniformly
    distributed phase, transformed, and the magnitude squared; only the mask
    differs between dimensions.

    `polarization < 1` mixes two independent patterns built from the same
    aperture, which at zero gives unpolarized speckle.

    Args:
        mask:         boolean aperture, the same length along every axis
        dims:         tuple giving the shape of the pattern to extract
        polarization: degree of polarization

    Returns:
        array with the requested shape
    """
    if polarization < 1:
        y1 = _speckle_from_mask(mask, dims, polarization=1)
        y2 = _speckle_from_mask(mask, dims, polarization=1)
        return 0.5 * (1 + polarization) * y1 + 0.5 * (1 - polarization) * y2

    # phases uniformly distributed from 0 to 2*pi
    phase = 2 * np.pi * np.random.rand(*mask.shape)
    x = np.exp(1j * phase) * mask

    # take the FFT and square it
    x = np.fft.fftshift(np.fft.fftn(x))
    x = abs(x) ** 2

    # extract the requested corner and normalize
    y = x[tuple(slice(None, n) for n in dims)]
    ymax = np.max(y) or 1
    return y / ymax


def _normalize_shape(shape):
    """
    Normalize a numpy-style shape into a tuple and check it.

    Args:
        shape: integer, or a tuple of 1, 2, or 3 integers

    Returns:
        tuple of dimensions
    """
    dims = (shape,) if np.isscalar(shape) else tuple(shape)

    if len(dims) not in (1, 2, 3):
        raise ValueError("shape must describe 1, 2, or 3 dimensions.")

    if min(dims) < 2:
        raise ValueError("every dimension must be at least 2 pixels.")

    return dims


def create_exponential(shape, pix_per_speckle, alpha=None, beta=None, aperture=None, polarization=1):
    """
    Generate a fully-developed speckle irradiance pattern.

    `shape` follows the numpy convention used by `np.ones`: an integer gives a
    one-dimensional pattern, and a tuple gives two or three dimensions::

        create_exponential(1024, 8)             # 1024 samples
        create_exponential((201, 201), 2)       # 201 x 201 image
        create_exponential((32, 32, 32), 2)     # 32 x 32 x 32 volume

    The irradiance is exponentially distributed with unit speckle contrast.
    The resolution is set by `pix_per_speckle`, which refers to the smallest
    speckle: 2 is the Nyquist limit and 4 puts four pixels across it.

    Arguments that do not apply to the requested dimensionality are rejected
    rather than quietly ignored:

    * one dimension takes none of `alpha`, `beta`, or `aperture`, because a
      one-dimensional aperture is always a segment
    * two dimensions take `alpha` and `aperture`, one of 'ellipse',
      'rectangle', or 'annulus', but not `beta`
    * three dimensions take `alpha`, `beta`, and `aperture`, one of
      'ellipsoid', 'cube', or 'shell'

    `alpha` is the ratio of x speckle size to y, and `beta` the ratio of x to
    z, so values above one stretch the speckle along x.  `polarization=0` sums
    two independent patterns to give unpolarized speckle with contrast
    1/sqrt(2).

    see Duncan & Kirkpatrick, "Algorithms for simulation of speckle," in SPIE
    Vol. 6855 (2008)

    Args:
        shape:           integer or tuple giving the shape of the pattern
        pix_per_speckle: number of pixels per smallest speckle
        alpha:           ratio of x to y speckle size (2D and 3D)
        beta:            ratio of x to z speckle size (3D only)
        aperture:        shape of the illuminated aperture (2D and 3D)
        polarization:    degree of polarization

    Returns:
        array with the requested shape
    """
    dims = _normalize_shape(shape)
    ndim = len(dims)
    _validate_speckle(pix_per_speckle, polarization)

    if ndim == 1 and (alpha is not None or beta is not None or aperture is not None):
        raise ValueError("alpha, beta, and aperture do not apply in one dimension.")

    if ndim == 2 and beta is not None:
        raise ValueError("beta does not apply in two dimensions; use alpha.")

    if alpha is None:
        alpha = 1
    if beta is None:
        beta = 1

    # the aperture is square; the requested shape is cropped out of the result
    M = max(dims)
    radii = [int(M / 2), int(alpha * M / 2), int(beta * M / 2)][:ndim]

    if min(radii) < 1:
        raise ValueError("shape, alpha, and beta must give radii of at least one pixel.")

    L = int(pix_per_speckle * 2 * max(radii))

    if ndim == 1:
        # in one dimension every aperture is just a segment
        mask = np.zeros(L, dtype=bool)
        mask[: 2 * radii[0]] = True
    elif ndim == 2:
        mask = _create_mask(L, *radii, shape=aperture or "ellipse")
    else:
        mask = _create_mask_3D(L, *radii, shape=aperture or "ellipsoid")

    return _speckle_from_mask(mask, dims, polarization)


def create_from_aperture(aperture, shape, polarization=1):
    """
    Generate speckle from an arbitrary illuminated aperture.

    `create_exponential` builds the aperture itself from a handful of named
    shapes.  This takes the aperture directly, so the pupil can be anything:
    several separated openings, an obstruction, a measured pupil function.
    The array is filled with uniform random phase, transformed, and the
    magnitude squared, exactly as `create_exponential` does.

    Speckle size is set by how large the illuminated region is relative to the
    array it sits in, not by where it sits, since position only adds a phase
    ramp that the magnitude squared discards.  A smaller opening in the same
    array gives coarser speckle.

    A pupil with two separated openings gives speckle crossed by interference
    fringes, the speckle counterpart of Young's two-slit experiment.  The
    fringe period is the array width divided by the separation, so pushing the
    openings apart packs the fringes closer together.

    Args:
        aperture:     array describing the illuminated pupil, square, 1 to 3D
        shape:        integer or tuple giving the shape of the pattern wanted
        polarization: degree of polarization

    Returns:
        array with the requested shape
    """
    aperture = np.asarray(aperture)
    dims = _normalize_shape(shape)

    if polarization < 0 or polarization > 1:
        raise ValueError("bad polarization. It must be 0 <= polarization <= 1.")

    if aperture.ndim != len(dims):
        raise ValueError("aperture must have the same number of dimensions as the requested shape.")

    if len(set(aperture.shape)) != 1:
        raise ValueError("aperture must be the same length along every axis.")

    if any(n > a for n, a in zip(dims, aperture.shape)):
        raise ValueError("the requested shape must fit inside the aperture array.")

    if not np.any(aperture):
        raise ValueError("aperture is empty; nothing is illuminated.")

    return _speckle_from_mask(aperture, dims, polarization)


def create_boiling_speckle(shape, pix_per_speckle, frames=None):
    """
    Generate a sequence of speckle patterns that decorrelate as they boil.

    A single fixed random-phase screen is illuminated through a circular
    pupil, and the pupil is translated one column per frame.  The speckle
    pattern changes continuously, and once the pupil has moved its own
    diameter the pattern is completely decorrelated from where it started.
    This is the back focal plane geometry of Duncan & Kirkpatrick, and it is
    how object-plane motion appears when observed behind a lens.

    The correlation between the first frame and each later one is returned
    along with the closed-form prediction, which for focal-plane speckle is
    the square of the autocorrelation of a circular disc::

        ACF(a) = (2/pi) * (arccos(a) - a * sqrt(1 - a**2))
        theory = ACF ** 2

    where `a` is the pupil displacement as a fraction of its diameter.

    The pupil edge is deliberately softened with a small Gaussian kernel.  A
    hard edge makes the sequence alias in time, because the pupil then gains
    and loses whole scatterers in single steps.

    `shape` must be square and two dimensional.  The returned cube has the
    frame as its last axis, so `cube[:, :, k]` is one pattern and `slice_plot`
    can display the volume directly.

    Args:
        shape:           tuple giving the shape of each frame, e.g. (128, 128)
        pix_per_speckle: number of pixels per smallest speckle, at least 2
        frames:          number of frames, defaulting to a full decorrelation

    Returns:
        cube, correlation with the first frame, and the theoretical correlation
    """
    dims = _normalize_shape(shape)

    if len(dims) != 2 or dims[0] != dims[1]:
        raise ValueError("shape must be square and two dimensional, for example (128, 128).")

    # the pupil slides a full diameter, so the phase screen must be twice as
    # wide as the pupil
    if pix_per_speckle < 2:
        raise ValueError("pix_per_speckle must be at least 2 so the pupil can slide a full diameter.")

    M = dims[0]
    if frames is None:
        frames = M + 1
    if frames < 1 or frames > M + 1:
        raise ValueError("frames must be between 1 and one more than the width of the pattern.")

    L = int(pix_per_speckle * M)

    # circular pupil, then soften the edge to avoid temporal aliasing
    u = np.linspace(-1, 1, M)
    X, Y = np.meshgrid(u, u, indexing="ij")
    pupil = (np.hypot(X, Y) <= 1).astype(float)
    smooth = np.array([[1, 4, 1], [4, 12, 4], [1, 4, 1]]) / 32
    pupil = scipy.signal.convolve2d(pupil, smooth, mode="same")

    # one fixed screen; the pupil moves across it
    phase = np.exp(2j * np.pi * np.random.rand(L, L))

    cube = np.empty((M, M, frames))
    for n in range(frames):
        window = np.zeros((L, L), dtype=complex)
        window[:M, n : n + M] = phase[:M, n : n + M] * pupil
        cube[:, :, n] = (abs(np.fft.fft2(window)) ** 2)[:M, :M]

    cube /= np.max(cube) or 1

    first = cube[:, :, 0].ravel()
    correlation = np.array([np.corrcoef(first, cube[:, :, n].ravel())[0, 1] for n in range(frames)])

    lag = np.arange(frames) / M
    acf = (2 / np.pi) * (np.arccos(lag) - lag * np.sqrt(1 - lag**2))
    return cube, correlation, acf**2


def create_oct_speckle(shape, pix_per_speckle, mua=2, musp=30, dz=1 / 300):
    """
    Generate an OCT B-scan whose speckle statistics change with depth.

    Rows are depth and columns are lateral position.  Two things happen as the
    beam goes deeper.  The speckle contrast falls, because light reaching depth
    has been scattered many times and arrives as more than one independent
    contribution, and the mean irradiance falls, because the tissue absorbs and
    scatters the light away.

    The contrast change is a sweep of the degree of polarization.  At the
    surface the light is a single fully developed pattern, `polarization = 1`
    and `K = 1`.  At the bottom two independent patterns add in irradiance,
    `polarization = 0` and `K = 1/sqrt(2)`, which is what `create_unpolarized`
    produces.  In between, the contrast of the returned image is::

        K = sqrt((1 + polarization**2) / 2)

    which is the third value returned.  The mean irradiance decays as
    `exp(-mu_eff * z)`, with the diffusion approximation for the effective
    attenuation coefficient::

        mu_eff = sqrt(3 * mua * (mua + musp))

    The defaults describe aorta and put one scattering event in each row.

    This is a translation of SimSpeckle's `OCT_speckle3.m`, with one change.
    That script blends the surface pattern against an already-unpolarized
    pattern, so three independent patterns contribute at intermediate depth
    with weights `1-a`, `a/2`, and `a/2`.  Its contrast therefore falls past
    `1/sqrt(2)` to a minimum of `1/sqrt(3)` at two thirds depth and then climbs
    back, which is not what its own comments describe.  Blending two patterns
    instead falls monotonically to `1/sqrt(2)` and reaches exactly the
    unpolarized pattern at the bottom.

    Args:
        shape:           tuple giving the depth and width of the image
        pix_per_speckle: number of pixels per smallest speckle
        mua:             absorption coefficient [1/cm]
        musp:            reduced scattering coefficient [1/cm]
        dz:              depth of one row [cm]

    Returns:
        image, depth of each row, and the expected contrast of each row
    """
    dims = _normalize_shape(shape)
    _validate_speckle(pix_per_speckle, 1)

    if len(dims) != 2:
        raise ValueError("an OCT image is depth by width, so shape must have two dimensions.")

    if mua < 0 or musp < 0:
        raise ValueError("mua and musp must not be negative.")

    if dz <= 0:
        raise ValueError("dz must be positive.")

    rows = dims[0]
    surface = create_exponential(dims, pix_per_speckle)
    deep = create_exponential(dims, pix_per_speckle)
    deep = deep * (np.mean(surface) / np.mean(deep))  # equal means, as the original does

    # depth sweeps the degree of polarization from one at the surface to zero
    polarization = 1 - np.arange(rows) / (rows - 1)
    p = polarization[:, np.newaxis]
    image = 0.5 * (1 + p) * surface + 0.5 * (1 - p) * deep

    depth = np.arange(rows) * dz
    mu_eff = np.sqrt(3 * mua * (mua + musp))
    image *= np.exp(-mu_eff * depth)[:, np.newaxis]

    contrast = np.sqrt((1 + polarization**2) / 2)
    return image, depth, contrast


def create_unpolarized(shape, pix_per_speckle, alpha=None, beta=None, aperture=None):
    """
    Generate an unpolarized speckle irradiance pattern.

    The pattern is the incoherent sum of two independent fully-developed
    speckle patterns, one per polarization state.  Its irradiance therefore
    follows a gamma distribution with shape 2, and the speckle contrast is
    1/sqrt(2) rather than the unity contrast of the polarized case.

    This is the zero-polarization limit that Duncan & Kirkpatrick call
    Rayleigh, and it is exactly `create_exponential(..., polarization=0)`.
    `shape` and the remaining arguments behave as they do there.

    Args:
        shape:           integer or tuple giving the shape of the pattern
        pix_per_speckle: number of pixels per smallest speckle
        alpha:           ratio of x to y speckle size (2D and 3D)
        beta:            ratio of x to z speckle size (3D only)
        aperture:        shape of the illuminated aperture (2D and 3D)

    Returns:
        array with the requested shape
    """
    return create_exponential(shape, pix_per_speckle, alpha=alpha, beta=beta, aperture=aperture, polarization=0)


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


def local_contrast(x, kernel):
    """
    Calculate local speckle contrast over a sliding window.

    The kernel must have the same number of dimensions as the speckle
    pattern, so the same function serves one, two, and three dimensions.

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

    if Nk < 2:
        raise ValueError("kernel must cover at least two samples.")

    # float, because squaring an integer image silently wraps: a uint8 255**2
    # comes back as 1 and the variance then clips to zero everywhere
    x = np.asarray(x, dtype=float)

    # contrast of raw image
    K = np.std(x) / np.mean(x)

    # local mean and the summed square over the kernel
    mu_x = scipy.signal.correlate(x, kernel, mode="valid") / Nk
    sum_sq = scipy.signal.correlate(x**2, kernel, mode="valid")

    # the unbiased sample variance, as in the SimSpeckle original; dividing by
    # Nk instead would bias the contrast low by sqrt(Nk/(Nk-1)), which is 12%
    # for a five-sample window.  Clipped because rounding can push it below zero
    var_x = np.maximum((sum_sq - Nk * mu_x**2) / (Nk - 1), 0)
    C = np.sqrt(var_x) / mu_x
    return C, K


def autocorrelation(x):
    """
    Find the autocorrelation of an array of any dimension.

    This is a little different from the standard autocorrelation because
    (1) the mean is subtracted before correlation
    (2) the autocorrelation is normalized to maximum value
    (3) only non-negative lags are returned

    The last point is what keeps the result the same shape as the input.  The
    full correlation is symmetric about zero lag, so in one dimension the
    right hand side carries everything; in two or three the same is true of
    the quadrant or octant reaching out from the origin, and `result[0]`,
    `result[0, 0]`, or `result[0, 0, 0]` is the zero-lag peak.

    Args:
        x: array of any dimension

    Returns:
        autocorrelation array of the same shape
    """
    xx = np.asarray(x, dtype=float)
    xx = xx - np.mean(xx)
    result = scipy.signal.correlate(xx, xx, mode="full")

    mx = np.max(result) or 1
    # zero lag sits at the middle of each axis, exactly where correlate puts it
    middle = tuple(slice(n - 1, None) for n in xx.shape)
    return result[middle] / mx


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
