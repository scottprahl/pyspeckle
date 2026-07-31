"""Tests for pyspeckle.speckle_1D."""

import numpy as np
import pytest
import pyspeckle


def test_create_exponential_1D_shape_and_range():
    """Output is length M, normalized to a maximum of one."""
    result = pyspeckle.create_exponential_1D(64, 2)
    assert result.shape == (64,)
    assert 0 <= np.min(result)
    assert np.max(result) <= 1.0


def test_create_exponential_1D_is_fully_developed():
    """Polarized speckle irradiance is exponential, so the contrast is unity."""
    v = np.concatenate([pyspeckle.create_exponential_1D(4096, 2) for _ in range(20)])
    v = v / np.mean(v)
    assert abs(np.std(v) - 1) < 0.15  # exponential irradiance -> K = 1


def test_create_exponential_1D_unpolarized_contrast():
    """Summing two independent patterns drops the contrast to 1/sqrt(2)."""
    v = np.concatenate([pyspeckle.create_exponential_1D(4096, 2, polarization=0) for _ in range(20)])
    v = v / np.mean(v)
    assert abs(np.std(v) - 1 / np.sqrt(2)) < 0.15


def test_create_unpolarized_1D_matches_zero_polarization():
    """The wrapper is exactly create_exponential_1D(..., polarization=0)."""
    np.random.seed(3)
    wrapper = pyspeckle.create_unpolarized_1D(256, 2)
    np.random.seed(3)
    explicit = pyspeckle.create_exponential_1D(256, 2, polarization=0)
    assert np.array_equal(wrapper, explicit)


def test_create_unpolarized_1D_contrast():
    """Unpolarized speckle has gamma-2 irradiance, so the contrast is 1/sqrt(2)."""
    v = np.concatenate([pyspeckle.create_unpolarized_1D(4096, 2) for _ in range(20)])
    v = v / np.mean(v)
    assert abs(np.std(v) - 1 / np.sqrt(2)) < 0.15


def test_create_exponential_1D_speckle_size():
    """Speckle size grows in proportion to pix_per_speckle."""

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line) < 0.5)

    narrow = half_width(pyspeckle.create_exponential_1D(4096, 2))
    wide = half_width(pyspeckle.create_exponential_1D(4096, 8))
    assert wide > 2 * narrow


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarization": -1},
        {"polarization": 2},
        {"pix_per_speckle": 0.5},
        {"M": 1},
    ],
)
def test_create_exponential_1D_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"M": 64, "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_exponential_1D(**args)


def test_local_contrast_1D_matches_global():
    """Local contrast over a long window should approach the global contrast."""
    speckle = pyspeckle.create_exponential_1D(8192, 2)
    n = 51
    C, K = pyspeckle.local_contrast_1D(speckle, np.ones(n))

    # only valid positions of the correlation are returned
    assert C.shape == (speckle.size - n + 1,)

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_1D_rejects_wrong_kernel_rank():
    """A 2D kernel cannot be used on a 1D pattern."""
    speckle = pyspeckle.create_exponential_1D(256, 2)
    with pytest.raises(ValueError):
        pyspeckle.local_contrast_1D(speckle, np.ones((3, 3)))


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


def test_create_gaussian_corr_1D_output_length():
    """Test length of create_gaussian_corr_1D."""
    arr = pyspeckle.create_gaussian_corr_1D(100, 10, 2, 5)
    assert len(arr) == 100


def test_create_gaussian_corr_1D_mean_and_std():
    """Test mean and stdev of create_gaussian_corr_1D output."""
    arr = pyspeckle.create_gaussian_corr_1D(1000, 10, 2, 5)
    assert abs(np.mean(arr) - 10) < 0.5
    assert abs(np.std(arr) - 2) < 0.5


@pytest.mark.parametrize(
    "M,mean,stdev,cl", [(0, 10, 2, 5), (100, 10, -2, 5), (100, 10, 2, -5), (100, 10, 2, 51)]
)  # M/cl < 2
def test_create_gaussian_corr_1D_invalid_args(M, mean, stdev, cl):
    """Test bad inputs to create_gaussian_corr_1D."""
    with pytest.raises(ValueError):  # or another appropriate exception based on behavior
        pyspeckle.create_gaussian_corr_1D(M, mean, stdev, cl)
