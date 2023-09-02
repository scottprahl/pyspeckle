import numpy as np
import pytest
import pyspeckle


# Tests for create_exp_1D
def test_create_exp_1D_output_length():
    arr = pyspeckle.create_exp_1D(100, 10, 2, 5)
    assert len(arr) == 100


def test_create_exp_1D_mean_and_std():
    arr = pyspeckle.create_exp_1D(1000, 10, 2, 5)
    assert abs(np.mean(arr) - 10) < 0.8  # A small tolerance might be needed due to randomness
    assert abs(np.std(arr) - 2) < 0.8


@pytest.mark.parametrize("M,mean,stdev,cl", [(0, 10, 2, 5), (100, 10, -2, 5), (100, 10, 2, -5), (100, 10, 2, 51)])
def test_create_exp_1D_invalid_args(M, mean, stdev, cl):
    with pytest.raises(ValueError):  # or another appropriate exception based on behavior
        pyspeckle.create_exp_1D(M, mean, stdev, cl)


# Tests for create_gaussian_1D
def test_create_gaussian_1D_output_length():
    arr = pyspeckle.create_gaussian_1D(100, 10, 2, 5)
    assert len(arr) == 100


def test_create_gaussian_1D_mean_and_std():
    arr = pyspeckle.create_gaussian_1D(1000, 10, 2, 5)
    assert abs(np.mean(arr) - 10) < 0.5
    assert abs(np.std(arr) - 2) < 0.5


@pytest.mark.parametrize("M,mean,stdev,cl", [(0, 10, 2, 5), (100, 10, -2, 5), (100, 10, 2, -5), (100, 10, 2, 51)])  # M/cl < 2
def test_create_gaussian_1D_invalid_args(M, mean, stdev, cl):
    with pytest.raises(ValueError):  # or another appropriate exception based on behavior
        pyspeckle.create_gaussian_1D(M, mean, stdev, cl)


# Tests for autocorrelation
def test_autocorrelation_length():
    arr = np.array([1, 2, 3, 4, 5])
    assert len(pyspeckle.autocorrelation(arr)) == len(arr)


def test_autocorrelation_value():
    arr = np.array([1, 2, 3, 4, 5])
    autocorr = pyspeckle.autocorrelation(arr)
    assert autocorr[0] == 1  # It's normalized to have a max of 1


def test_autocorrelation_value2():
    arr = np.array([0, 0, 0])
    autocorr = pyspeckle.autocorrelation(arr)
    assert autocorr[0] == 0  # It's normalized to have a max of 1


# Test for create_Exponential
def test_Exponential_shape_of_output():
    result = pyspeckle.create_Exponential(10, 2)
    assert result.shape == (10, 10)


def test_create_Exponential_shape():
    speckle = pyspeckle.create_Exponential(50, 2, alpha=1, shape='ellipse', polarization=1)
    assert speckle.shape == (50, 50)


def test_Exponential_maximum_value():
    result = pyspeckle.create_Exponential(10, 2)
    assert np.max(result) <= 1.0


def test_Exponential_non_circular_shapes():
    shapes = ['ellipse', 'rectangle', 'annulus', 'ELLIPSE', 'Rectangle', 'ANNULus']
    for shape in shapes:
        result = pyspeckle.create_Exponential(10, 2, shape=shape)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


def test_create_Exponential_invalid_pol1():
    with pytest.raises(ValueError):  # or another appropriate exception based on behavior
        pyspeckle.create_Exponential(10, 2, polarization=-1)


def test_create_Exponential_invalid_pol2():
    with pytest.raises(ValueError):  # or another appropriate exception based on behavior
        pyspeckle.create_Exponential(10, 2, polarization=2)


def test_Exponential_polarization_values():
    for polarization in [0, 0.5, 1]:
        result = pyspeckle.create_Exponential(10, 2, polarization=polarization)
        assert result.shape == (10, 10)
        assert np.max(result) <= 1.0


# Test for _create_mask
def test_ellipse_mask():
    mask = pyspeckle.pyspeckle._create_mask(10, 3, 4)
    assert mask.shape == (10, 10)
    assert mask[5, 5]
    assert mask[0, 6] == 0
    assert mask[9, 9] == 0


def test_rectangle_mask():
    mask = pyspeckle.pyspeckle._create_mask(10, 3, 4, shape='rectangle')
    assert mask.shape == (10, 10)
    assert mask[0, 0]
    assert mask[7, 5]
    assert mask[6, 8] == 0
    assert mask[7, 6] == 0


def test_annulus_mask():
    mask = pyspeckle.pyspeckle._create_mask(10, 3, 4, shape='annulus')
    assert mask.shape == (10, 10)
    assert mask[0, 0] == 0
    assert mask[4, 4] == 0
    assert mask[0, 4]
    assert mask[4, 0]
    assert mask[4, 8]
    assert mask[8, 4]
