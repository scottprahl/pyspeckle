"""Tests for pyspeckle.core."""

import numpy as np
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


# Tests for the private display helper used by the plotting routines
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


# Tests for the Gaussian copula chain, Duncan & Kirkpatrick eqs. (5a), (7), (8)
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
