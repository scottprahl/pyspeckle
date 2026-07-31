"""Tests for pyspeckle.speckle_1D."""

import numpy as np
import pytest
import pyspeckle


def test_create_exp_corr_1D_output_length():
    """Test length of create_exp_corr_1D."""
    arr = pyspeckle.create_exp_corr_1D(100, 10, 2, 5)
    assert len(arr) == 100


def test_create_exp_corr_1D_mean_and_std():
    """Test mean and stdev of create_exp_corr_1D."""
    arr = pyspeckle.create_exp_corr_1D(1000, 10, 2, 5)
    assert abs(np.mean(arr) - 10) < 0.8  # A small tolerance might be needed due to randomness
    assert abs(np.std(arr) - 2) < 0.8


@pytest.mark.parametrize("M,mean,stdev,cl", [(0, 10, 2, 5), (100, 10, -2, 5), (100, 10, 2, -5), (100, 10, 2, 51)])
def test_create_exp_corr_1D_invalid_args(M, mean, stdev, cl):
    """Test bad inputs to create_exp_corr_1D."""
    with pytest.raises(ValueError):
        pyspeckle.create_exp_corr_1D(M, mean, stdev, cl)


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
