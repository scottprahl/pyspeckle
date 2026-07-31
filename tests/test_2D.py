"""Tests for pyspeckle.speckle_2D."""

import numpy as np
import pytest
import pyspeckle


def test_exponential_2D_shape_of_output():
    """Test shape of create_exponential_2D."""
    result = pyspeckle.create_exponential_2D(10, 2)
    assert result.shape == (10, 10)


def test_create_exponential_2D_shape():
    """Test shape of create_exponential_2D with params."""
    speckle = pyspeckle.create_exponential_2D(50, 2, alpha=1, shape="ellipse", polarization=1)
    assert speckle.shape == (50, 50)


def test_exponential_2D_maximum_value():
    """Test max value of create_exponential_2D."""
    result = pyspeckle.create_exponential_2D(10, 2)
    assert np.max(result) <= 1.0


def test_exponential_2D_non_circular_shapes():
    """Verify that other shapes work with create_exponential_2D."""
    shapes = ["ellipse", "rectangle", "annulus", "ELLIPSE", "Rectangle", "ANNULus"]
    for shape in shapes:
        result = pyspeckle.create_exponential_2D(10, 2, shape=shape)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


def test_create_exponential_2D_invalid_pol1():
    """Test invalid polarization."""
    with pytest.raises(ValueError):
        pyspeckle.create_exponential_2D(10, 2, polarization=-1)


def test_create_exponential_2D_invalid_pol2():
    """Test2 invalid polarization."""
    with pytest.raises(ValueError):
        pyspeckle.create_exponential_2D(10, 2, polarization=2)


def test_exponential_2D_polarization_values():
    """Test valid polarizations."""
    for polarization in [0, 0.5, 1]:
        result = pyspeckle.create_exponential_2D(10, 2, polarization=polarization)
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
def test_create_exponential_2D_invalid_geometry(kwargs):
    """Bad geometry should raise ValueError instead of failing inside numpy."""
    with pytest.raises(ValueError):
        pyspeckle.create_exponential_2D(**kwargs)


def test_create_exponential_2D_fractional_pix_per_speckle():
    """Non-integer pixels per speckle used to raise TypeError from np.random.rand."""
    result = pyspeckle.create_exponential_2D(64, 2.5)
    assert result.shape == (64, 64)
    assert np.max(result) <= 1.0


def test_local_contrast_2D_matches_global():
    """Local contrast over a large kernel should approach the global contrast."""
    speckle = pyspeckle.create_exponential_2D(256, 2)
    n = 15
    C, K = pyspeckle.local_contrast_2D(speckle, np.ones((n, n)))

    # only valid pixels of the convolution are returned
    assert C.shape == (speckle.shape[0] - n + 1, speckle.shape[1] - n + 1)

    # fully developed speckle has unity contrast, and a 15x15 window recovers
    # most of it; a missing kernel normalization drops this by a factor of n
    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


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
