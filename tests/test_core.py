"""Tests for pyspeckle.core: speckle generation, contrast, and the copula chain."""

import numpy as np
import pytest
import scipy.stats
import pyspeckle


def test_autocorrelation_length():
    """Test length of autocorrelation."""
    arr = np.array([1, 2, 3, 4, 5])
    assert len(pyspeckle.autocorrelation(arr)) == len(arr)


def test_autocorrelation_value():
    """Test max value of autocorrelation."""
    arr = np.array([1, 2, 3, 4, 5])
    autocorr = pyspeckle.autocorrelation(arr)
    assert autocorr[0] == 1  # It's normalized to have a max of 1


def test_autocorrelation_value2():
    """Test autocorrelation with zeros."""
    arr = np.array([0, 0, 0])
    autocorr = pyspeckle.autocorrelation(arr)
    assert autocorr[0] == 0  # It's normalized to have a max of 1


def test_create_exponential_shape_and_range():
    """Output is length M, normalized to a maximum of one."""
    result = pyspeckle.create_exponential(64, 2)
    assert result.shape == (64,)
    assert 0 <= np.min(result)
    assert np.max(result) <= 1.0


def test_exponential_2D_shape_of_output():
    """Test shape of create_exponential."""
    result = pyspeckle.create_exponential((10, 10), 2)
    assert result.shape == (10, 10)


def test_create_exponential_shape():
    """Test shape of create_exponential with params."""
    speckle = pyspeckle.create_exponential((50, 50), 2, alpha=1, aperture="ellipse", polarization=1)
    assert speckle.shape == (50, 50)


def test_exponential_2D_maximum_value():
    """Test max value of create_exponential."""
    result = pyspeckle.create_exponential((10, 10), 2)
    assert np.max(result) <= 1.0


def test_create_exponential_fractional_pix_per_speckle():
    """Non-integer pixels per speckle used to raise TypeError from np.random.rand."""
    result = pyspeckle.create_exponential((64, 64), 2.5)
    assert result.shape == (64, 64)
    assert np.max(result) <= 1.0


def test_create_exponential_speckle_size():
    """Speckle size grows in proportion to pix_per_speckle."""

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line) < 0.5)

    narrow = half_width(pyspeckle.create_exponential(4096, 2))
    wide = half_width(pyspeckle.create_exponential(4096, 8))
    assert wide > 2 * narrow


def test_exponential_2D_non_circular_shapes():
    """Verify that other shapes work with create_exponential."""
    apertures = ["ellipse", "rectangle", "annulus", "ELLIPSE", "Rectangle", "ANNULus"]
    for aperture in apertures:
        result = pyspeckle.create_exponential((10, 10), 2, aperture=aperture)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


@pytest.mark.parametrize(
    "generator,expected",
    [
        (pyspeckle.create_exponential, 1.0),  # exponential irradiance
        (pyspeckle.create_unpolarized, 1 / np.sqrt(2)),  # gamma-2 irradiance
    ],
)
def test_speckle_contrast_1D(generator, expected):
    """Speckle contrast is unity when polarized and 1/sqrt(2) when not."""
    speckle = generator(16384, 2)
    contrast = np.std(speckle) / np.mean(speckle)
    assert abs(contrast - expected) < 0.05


@pytest.mark.parametrize(
    "generator,expected",
    [
        (pyspeckle.create_exponential, 1.0),  # exponential irradiance
        (pyspeckle.create_unpolarized, 1 / np.sqrt(2)),  # gamma-2 irradiance
    ],
)
def test_speckle_contrast_2D(generator, expected):
    """Speckle contrast is unity when polarized and 1/sqrt(2) when not."""
    speckle = generator(256, 2)
    contrast = np.std(speckle) / np.mean(speckle)
    assert abs(contrast - expected) < 0.05


@pytest.mark.parametrize(
    "generator,expected",
    [
        (pyspeckle.create_exponential, 1.0),  # exponential irradiance
        (pyspeckle.create_unpolarized, 1 / np.sqrt(2)),  # gamma-2 irradiance
    ],
)
def test_speckle_contrast_3D(generator, expected):
    """Speckle contrast is unity when polarized and 1/sqrt(2) when not."""
    speckle = generator((32, 32, 32), 2)
    contrast = np.std(speckle) / np.mean(speckle)
    assert abs(contrast - expected) < 0.05


def test_speckle_contrast_2D_independent_of_resolution():
    """Contrast is a first-order statistic, so pix_per_speckle must not change it."""
    for pix_per_speckle in (2, 4, 8):
        speckle = pyspeckle.create_exponential((256, 256), pix_per_speckle)
        assert abs(np.std(speckle) / np.mean(speckle) - 1) < 0.1


def test_create_exponential_polarization_sweep():
    """Contrast falls monotonically from 1 to 1/sqrt(2) as polarization goes to zero."""
    contrasts = []
    for polarization in (1.0, 0.75, 0.5, 0.25, 0.0):
        speckle = pyspeckle.create_exponential(16384, 2, polarization=polarization)
        contrasts.append(np.std(speckle) / np.mean(speckle))
    assert abs(contrasts[0] - 1) < 0.05
    assert abs(contrasts[-1] - 1 / np.sqrt(2)) < 0.05
    assert np.all(np.diff(contrasts) < 0.01)  # non-increasing


def test_create_exponential_invalid_pol1():
    """Test invalid polarization."""
    with pytest.raises(ValueError):
        pyspeckle.create_exponential((10, 10), 2, polarization=-1)


def test_create_exponential_invalid_pol2():
    """Test2 invalid polarization."""
    with pytest.raises(ValueError):
        pyspeckle.create_exponential((10, 10), 2, polarization=2)


def test_exponential_2D_polarization_values():
    """Test valid polarizations."""
    for polarization in [0, 0.5, 1]:
        result = pyspeckle.create_exponential((10, 10), 2, polarization=polarization)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


def test_create_unpolarized_matches_zero_polarization():
    """The wrapper is exactly create_exponential(..., polarization=0)."""
    np.random.seed(3)
    wrapper = pyspeckle.create_unpolarized(256, 2)
    np.random.seed(3)
    explicit = pyspeckle.create_exponential(256, 2, polarization=0)
    assert np.array_equal(wrapper, explicit)


def test_create_unpolarized_irradiance_is_gamma2():
    """Unpolarized irradiance follows a gamma distribution with shape 2."""
    speckle = pyspeckle.create_unpolarized(16384, 2)
    speckle = speckle / np.mean(speckle)
    # decimate to near-independent samples; neighbours within a speckle correlate
    assert scipy.stats.kstest(speckle[::16], "gamma", args=(2, 0, 0.5)).pvalue > 0.01


def test_create_unpolarized_uses_beta():
    """Verify beta reaches the mask; the unpolarized recursion used to drop it."""
    M = 24
    speckle = pyspeckle.create_unpolarized((M, M, M), 2, beta=3)
    assert speckle.shape == (M, M, M)

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line.astype(float)) < 0.5)

    x_width = np.mean([half_width(speckle[:, j, k]) for j in range(0, M, 4) for k in range(0, M, 4)])
    z_width = np.mean([half_width(speckle[i, j, :]) for i in range(0, M, 4) for j in range(0, M, 4)])

    # beta>1 stretches the speckle along x; dropping beta leaves it isotropic
    assert x_width / z_width > 1.4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarization": -1},
        {"polarization": 2},
        {"pix_per_speckle": 0.5},
        {"shape": 1},
        {"alpha": 2},  # anisotropy is meaningless along a line
        {"beta": 2},
        {"aperture": "ellipse"},  # a 1D aperture is always a segment
    ],
)
def test_create_exponential_1D_invalid_args(kwargs):
    """Bad arguments, and arguments that do not apply in 1D, raise ValueError."""
    args = {"shape": 64, "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_exponential(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pix_per_speckle": 0.5},  # undersampled
        {"shape": (1, 1)},  # radius rounds down to zero
        {"alpha": 0.001},  # y radius rounds down to zero
        {"beta": 2},  # there is no third axis in 2D
        {"aperture": "banana"},
    ],
)
def test_create_exponential_2D_invalid_args(kwargs):
    """Bad arguments, and arguments that do not apply in 2D, raise ValueError."""
    args = {"shape": (64, 64), "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_exponential(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarization": -5},
        {"polarization": 2},
        {"aperture": "banana"},
        {"pix_per_speckle": 0.5},
        {"shape": (1, 1, 1)},
        {"shape": (8, 8, 8, 8)},  # four dimensions are not supported
    ],
)
def test_create_exponential_3D_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"shape": (8, 8, 8), "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_exponential(**args)


def test_local_contrast_1D_matches_global():
    """Local contrast over a long window should approach the global contrast."""
    speckle = pyspeckle.create_exponential(8192, 2)
    n = 51
    C, K = pyspeckle.local_contrast(speckle, np.ones(n))

    # only valid positions of the correlation are returned
    assert C.shape == (speckle.size - n + 1,)

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_2D_matches_global():
    """Local contrast over a large kernel should approach the global contrast."""
    speckle = pyspeckle.create_exponential((256, 256), 2)
    n = 15
    C, K = pyspeckle.local_contrast(speckle, np.ones((n, n)))

    # only valid pixels of the convolution are returned
    assert C.shape == (speckle.shape[0] - n + 1, speckle.shape[1] - n + 1)

    # fully developed speckle has unity contrast, and a 15x15 window recovers
    # most of it; a missing kernel normalization drops this by a factor of n
    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_3D_matches_global():
    """Local contrast over a large volume should approach the global contrast."""
    M, n = 32, 9
    speckle = pyspeckle.create_exponential((M, M, M), 2)
    C, K = pyspeckle.local_contrast(speckle, np.ones((n, n, n)))

    # only valid positions of the correlation are returned
    assert C.shape == (M - n + 1,) * 3

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_2D_integer_image():
    """An 8-bit image must not overflow when squared; it used to give all zeros."""
    speckle = pyspeckle.create_exponential((128, 128), 4)
    as_uint8 = (255 * speckle).astype(np.uint8)

    C, K = pyspeckle.local_contrast(as_uint8, np.ones((10, 10)))
    assert np.all(np.isfinite(C))
    assert C.max() > 0.1  # was exactly 0 everywhere while uint8 wrapped
    assert abs(K - 1) < 0.2

    # the same data as float must give the same answer
    C_float, _ = pyspeckle.local_contrast(as_uint8.astype(float), np.ones((10, 10)))
    assert np.allclose(C, C_float)


def test_local_contrast_1D_rejects_wrong_kernel_rank():
    """A 2D kernel cannot be used on a 1D pattern."""
    speckle = pyspeckle.create_exponential(256, 2)
    with pytest.raises(ValueError):
        pyspeckle.local_contrast(speckle, np.ones((3, 3)))


def test_local_contrast_3D_rejects_wrong_kernel_rank():
    """A 2D kernel cannot be used on a 3D pattern."""
    speckle = pyspeckle.create_exponential((16, 16, 16), 2)
    with pytest.raises(ValueError):
        pyspeckle.local_contrast(speckle, np.ones((3, 3)))


def test_ellipse_mask():
    """Basic functionality for ellipse mask."""
    mask = pyspeckle.core._create_mask(10, 3, 4)  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[5, 5]
    assert mask[0, 6] == 0
    assert mask[9, 9] == 0


def test_rectangle_mask():
    """Basic functionality for rect mask."""
    mask = pyspeckle.core._create_mask(10, 3, 4, shape="rectangle")  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[0, 0]
    assert mask[7, 5]
    assert mask[6, 8] == 0
    assert mask[7, 6] == 0


def test_annulus_mask():
    """Basic functionality for annular mask."""
    mask = pyspeckle.core._create_mask(10, 3, 4, shape="annulus")  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[0, 0] == 0
    assert mask[4, 4] == 0
    assert mask[0, 4]
    assert mask[4, 0]
    assert mask[4, 8]
    assert mask[8, 4]


def test_mask_2D_too_small():
    """The mask array must be at least twice the largest radius."""
    with pytest.raises(ValueError):
        pyspeckle.core._create_mask(6, 5, 4)  # pylint: disable=protected-access


def test_mask_2D_unknown_shape():
    """An unrecognized aperture shape is rejected."""
    with pytest.raises(ValueError):
        pyspeckle.core._create_mask(10, 3, 4, shape="banana")  # pylint: disable=protected-access


def test_mask_3D_cube():
    """The cube aperture fills the corner block of side 2*radius."""
    mask = pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="cube")  # pylint: disable=protected-access
    assert mask.shape == (16, 16, 16)
    assert mask.sum() == 8**3
    assert mask[0, 0, 0]
    assert mask[7, 7, 7]
    assert not mask[8, 0, 0]


def test_mask_3D_shell():
    """The shell aperture is hollow, keeping points between the two radii."""
    mask = pyspeckle.core._create_mask_3D(16, 2, 4, 4, shape="shell")  # pylint: disable=protected-access
    assert mask.shape == (16, 16, 16)
    assert not mask[4, 4, 4]  # centre is hollow
    assert mask[4, 4, 7]  # inside the outer radius
    assert not mask[0, 0, 0]  # outside the outer radius


def test_mask_3D_case_insensitive():
    """The 3D mask should fold case like the 2D one."""
    upper = pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="Ellipsoid")  # pylint: disable=protected-access
    lower = pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="ellipsoid")  # pylint: disable=protected-access
    assert np.array_equal(upper, lower)


def test_mask_3D_unknown_shape():
    """Unknown 3D shapes used to fall through silently to an ellipsoid."""
    with pytest.raises(ValueError):
        pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="banana")  # pylint: disable=protected-access


def test_mask_3D_too_small():
    """The 3D mask array must be at least twice the largest radius."""
    with pytest.raises(ValueError):
        pyspeckle.core._create_mask_3D(8, 6, 4, 4)  # pylint: disable=protected-access


def test_sqrt_matrix_scales_to_255():
    """The largest value maps to 255 and the square root compresses the range."""
    scaled = pyspeckle.core._sqrt_matrix(np.array([0.0, 0.25, 1.0]))  # pylint: disable=protected-access
    assert scaled.dtype.kind == "i"
    assert list(scaled) == [0, 127, 255]  # 255*sqrt(0.25) = 127.5 truncated


def test_sqrt_matrix_is_scale_invariant():
    """Only the ratio to the maximum matters, not the absolute values."""
    small = pyspeckle.core._sqrt_matrix(np.array([0.0, 0.25, 1.0]))  # pylint: disable=protected-access
    large = pyspeckle.core._sqrt_matrix(np.array([0.0, 25.0, 100.0]))  # pylint: disable=protected-access
    assert np.array_equal(small, large)


def test_sqrt_matrix_all_zeros():
    """An all-zero pattern must not divide by zero."""
    scaled = pyspeckle.core._sqrt_matrix(np.zeros(4))  # pylint: disable=protected-access
    assert np.all(scaled == 0)


def test_sqrt_matrix_preserves_shape():
    """Two-dimensional input keeps its shape."""
    scaled = pyspeckle.core._sqrt_matrix(np.array([[0.0, 1.0], [4.0, 9.0]]))  # pylint: disable=protected-access
    assert scaled.shape == (2, 2)
    assert scaled[1, 1] == 255


def test_box_muller_moments():
    """Both returned arrays are normal with the requested mean and stdev."""
    y1, y2 = pyspeckle.box_muller(3, 2, N=100000)
    assert len(y1) == len(y2) == 100000
    for y in (y1, y2):
        assert abs(np.mean(y) - 3) < 0.05
        assert abs(np.std(y) - 2) < 0.05


def test_box_muller_pair_is_independent():
    """The Box-Muller pair is uncorrelated."""
    y1, y2 = pyspeckle.box_muller(0, 1, N=100000)
    assert abs(np.corrcoef(y1, y2)[0, 1]) < 0.02


def test_zvalues_correlation_is_r():
    """Verify that r is the correlation coefficient of a standard normal pair."""
    for r in (0.0, 0.5, 0.9):
        z1, z2 = pyspeckle.zvalues(r, N=200000)
        assert abs(np.corrcoef(z1, z2)[0, 1] - r) < 0.02
        assert abs(np.std(z1) - 1) < 0.02
        assert abs(np.mean(z1)) < 0.02


def test_tvalues_are_uniform():
    """The percentile transform gives uniform marginals, not Student t."""
    t1, t2 = pyspeckle.tvalues(0.5, N=100000)
    assert t1.min() >= 0 and t1.max() <= 1
    assert scipy.stats.kstest(t1, "uniform").pvalue > 0.01
    # correlation of the uniforms is (6/pi)*arcsin(r/2), not r itself
    expected = (6 / np.pi) * np.arcsin(0.5 / 2)
    assert abs(np.corrcoef(t1, t2)[0, 1] - expected) < 0.02
