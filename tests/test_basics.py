"""Tests of basic functionality of pyspeckle."""

import numpy as np
import pytest
import pyspeckle


@pytest.fixture(autouse=True)
def seed_rng():
    """
    Seed numpy before every test.

    Several tests assert on sample statistics whose tolerances are only a few
    standard deviations wide.  Unseeded, create_gaussian_1D(1000, 10, 2, 5)
    lands outside its 0.5 tolerance in about one run per hundred, which is
    frequent enough to fail CI at random.
    """
    np.random.seed(0)


def test_create_exp_1D_output_length():
    """Test length of create_exp_1D."""
    arr = pyspeckle.create_exp_1D(100, 10, 2, 5)
    assert len(arr) == 100


def test_create_exp_1D_mean_and_std():
    """Test mean and stdev of create_exp_1D."""
    arr = pyspeckle.create_exp_1D(1000, 10, 2, 5)
    assert abs(np.mean(arr) - 10) < 0.8  # A small tolerance might be needed due to randomness
    assert abs(np.std(arr) - 2) < 0.8


@pytest.mark.parametrize("M,mean,stdev,cl", [(0, 10, 2, 5), (100, 10, -2, 5), (100, 10, 2, -5), (100, 10, 2, 51)])
def test_create_exp_1D_invalid_args(M, mean, stdev, cl):
    """Test bad inputs to create_exp_1D."""
    with pytest.raises(ValueError):
        pyspeckle.create_exp_1D(M, mean, stdev, cl)


def test_create_gaussian_1D_output_length():
    """Test length of create_gaussian_1D."""
    arr = pyspeckle.create_gaussian_1D(100, 10, 2, 5)
    assert len(arr) == 100


def test_create_gaussian_1D_mean_and_std():
    """Test mean and stdev of create_gaussian_1D output."""
    arr = pyspeckle.create_gaussian_1D(1000, 10, 2, 5)
    assert abs(np.mean(arr) - 10) < 0.5
    assert abs(np.std(arr) - 2) < 0.5


@pytest.mark.parametrize(
    "M,mean,stdev,cl", [(0, 10, 2, 5), (100, 10, -2, 5), (100, 10, 2, -5), (100, 10, 2, 51)]
)  # M/cl < 2
def test_create_gaussian_1D_invalid_args(M, mean, stdev, cl):
    """Test bad inputs to create_gaussian_1D."""
    with pytest.raises(ValueError):  # or another appropriate exception based on behavior
        pyspeckle.create_gaussian_1D(M, mean, stdev, cl)


# Tests for autocorrelation
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


# Test for create_Exponential
def test_Exponential_shape_of_output():
    """Test shape of create_Exponential."""
    result = pyspeckle.create_Exponential(10, 2)
    assert result.shape == (10, 10)


def test_create_Exponential_shape():
    """Test shape of create_Exponential with params."""
    speckle = pyspeckle.create_Exponential(50, 2, alpha=1, shape="ellipse", polarization=1)
    assert speckle.shape == (50, 50)


def test_Exponential_maximum_value():
    """Test max value of create_Exponential."""
    result = pyspeckle.create_Exponential(10, 2)
    assert np.max(result) <= 1.0


def test_Exponential_non_circular_shapes():
    """Verify that other shapes work with create_Exponential."""
    shapes = ["ellipse", "rectangle", "annulus", "ELLIPSE", "Rectangle", "ANNULus"]
    for shape in shapes:
        result = pyspeckle.create_Exponential(10, 2, shape=shape)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


def test_create_Exponential_invalid_pol1():
    """Test invalid polarization."""
    with pytest.raises(ValueError):
        pyspeckle.create_Exponential(10, 2, polarization=-1)


def test_create_Exponential_invalid_pol2():
    """Test2 invalid polarization."""
    with pytest.raises(ValueError):
        pyspeckle.create_Exponential(10, 2, polarization=2)


def test_Exponential_polarization_values():
    """Test valid polarizations."""
    for polarization in [0, 0.5, 1]:
        result = pyspeckle.create_Exponential(10, 2, polarization=polarization)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"M": 64, "pix_per_speckle": 0.5},  # undersampled
        {"M": 1, "pix_per_speckle": 2},  # x radius rounds down to zero
        {"M": 64, "pix_per_speckle": 2, "alpha": 0.001},  # y radius rounds down to zero
    ],
)
def test_create_Exponential_invalid_geometry(kwargs):
    """Bad geometry should raise ValueError instead of failing inside numpy."""
    with pytest.raises(ValueError):
        pyspeckle.create_Exponential(**kwargs)


def test_create_Exponential_fractional_pix_per_speckle():
    """Non-integer pixels per speckle used to raise TypeError from np.random.rand."""
    result = pyspeckle.create_Exponential(64, 2.5)
    assert result.shape == (64, 64)
    assert np.max(result) <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarization": -5},
        {"polarization": 2},
        {"shape": "banana"},
        {"pix_per_speckle": 0.5},
        {"M": 1},
    ],
)
def test_create_Exponential_3D_invalid_args(kwargs):
    """The 3D generator should validate its arguments like the 2D one."""
    args = {"M": 8, "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_Exponential_3D(**args)


# Tests for local_contrast_2D
def test_local_contrast_2D_matches_global():
    """Local contrast over a large kernel should approach the global contrast."""
    speckle = pyspeckle.create_Exponential(256, 2)
    n = 15
    C, K = pyspeckle.local_contrast_2D(speckle, np.ones((n, n)))

    # only valid pixels of the convolution are returned
    assert C.shape == (speckle.shape[0] - n + 1, speckle.shape[1] - n + 1)

    # fully developed speckle has unity contrast, and a 15x15 window recovers
    # most of it; a missing kernel normalization drops this by a factor of n
    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


# Tests for create_Rayleigh_3D
def test_create_Rayleigh_3D_uses_beta():
    """Verify beta reaches the mask; the unpolarized recursion used to drop it."""
    M = 24
    speckle = pyspeckle.create_Rayleigh_3D(M, 2, beta=3)
    assert speckle.shape == (M, M, M)

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line.astype(float)) < 0.5)

    x_width = np.mean([half_width(speckle[:, j, k]) for j in range(0, M, 4) for k in range(0, M, 4)])
    z_width = np.mean([half_width(speckle[i, j, :]) for i in range(0, M, 4) for j in range(0, M, 4)])

    # beta>1 stretches the speckle along x; dropping beta leaves it isotropic
    assert x_width / z_width > 1.4


def test_ellipse_mask():
    """Basic functionality for ellipse mask."""
    mask = pyspeckle.speckle_2D._create_mask(10, 3, 4)  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[5, 5]
    assert mask[0, 6] == 0
    assert mask[9, 9] == 0


def test_rectangle_mask():
    """Basic functionality for rect mask."""
    mask = pyspeckle.speckle_2D._create_mask(10, 3, 4, shape="rectangle")  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[0, 0]
    assert mask[7, 5]
    assert mask[6, 8] == 0
    assert mask[7, 6] == 0


def test_annulus_mask():
    """Basic functionality for annular mask."""
    mask = pyspeckle.speckle_2D._create_mask(10, 3, 4, shape="annulus")  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[0, 0] == 0
    assert mask[4, 4] == 0
    assert mask[0, 4]
    assert mask[4, 0]
    assert mask[4, 8]
    assert mask[8, 4]


def test_mask_3D_case_insensitive():
    """The 3D mask should fold case like the 2D one."""
    upper = pyspeckle.speckle_3D._create_mask_3D(16, 4, 4, 4, shape="Ellipsoid")  # pylint: disable=protected-access
    lower = pyspeckle.speckle_3D._create_mask_3D(16, 4, 4, 4, shape="ellipsoid")  # pylint: disable=protected-access
    assert np.array_equal(upper, lower)


def test_mask_3D_unknown_shape():
    """Unknown 3D shapes used to fall through silently to an ellipsoid."""
    with pytest.raises(ValueError):
        pyspeckle.speckle_3D._create_mask_3D(16, 4, 4, 4, shape="banana")  # pylint: disable=protected-access


def test_mask_3D_too_small():
    """The 3D mask array must be at least twice the largest radius."""
    with pytest.raises(ValueError):
        pyspeckle.speckle_3D._create_mask_3D(8, 6, 4, 4)  # pylint: disable=protected-access
