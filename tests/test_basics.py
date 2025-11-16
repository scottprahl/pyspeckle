"""Tests of basic functionality of pyspeckle."""

import numpy as np
import pytest
import pyspeckle


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


def test_ellipse_mask():
    """Basic functionality for ellipse mask."""
    mask = pyspeckle.pyspeckle._create_mask(10, 3, 4)  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[5, 5]
    assert mask[0, 6] == 0
    assert mask[9, 9] == 0


def test_rectangle_mask():
    """Basic functionality for rect mask."""
    mask = pyspeckle.pyspeckle._create_mask(10, 3, 4, shape="rectangle")  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[0, 0]
    assert mask[7, 5]
    assert mask[6, 8] == 0
    assert mask[7, 6] == 0


def test_annulus_mask():
    """Basic functionality for annular mask."""
    mask = pyspeckle.pyspeckle._create_mask(10, 3, 4, shape="annulus")  # pylint: disable=protected-access
    assert mask.shape == (10, 10)
    assert mask[0, 0] == 0
    assert mask[4, 4] == 0
    assert mask[0, 4]
    assert mask[4, 0]
    assert mask[4, 8]
    assert mask[8, 4]
