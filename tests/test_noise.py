"""Tests for pyspeckle.noise: correlated Gaussian fields and phase screens."""

import numpy as np
import pytest
import scipy.stats
import pyspeckle


@pytest.mark.parametrize("correlation", ["exponential", "gaussian"])
def test_create_correlated_statistics(correlation):
    """Length, mean, and standard deviation come out as requested."""
    arr = pyspeckle.create_correlated(100, 10, 2, 5, correlation=correlation)
    assert len(arr) == 100

    arr = pyspeckle.create_correlated(10000, 10, 2, 5, correlation=correlation)
    assert abs(np.mean(arr) - 10) < 0.5
    assert abs(np.std(arr) - 2) < 0.5


@pytest.mark.parametrize("correlation", ["exponential", "gaussian"])
def test_create_correlated_length_is_the_1_over_e_lag(correlation):
    """The cl argument means the same for both shapes: the lag where the ACF hits 1/e."""
    cl = 32
    field = pyspeckle.create_correlated(131072, 0, 1, cl, correlation=correlation)
    acf = pyspeckle.autocorrelation(field)
    assert abs(acf[cl] - np.exp(-1)) < 0.05


@pytest.mark.parametrize("correlation", ["exponential", "gaussian"])
def test_create_correlated_is_gaussian(correlation):
    """The values are normally distributed whatever the correlation shape."""
    field = pyspeckle.create_correlated(65536, 0, 1, 16, correlation=correlation)
    assert abs(scipy.stats.skew(field)) < 0.1
    assert abs(scipy.stats.kurtosis(field)) < 0.3


def test_create_correlated_higher_dimensions():
    """The same call works in two and three dimensions."""
    surface = pyspeckle.create_correlated((256, 256), 8000, 800, 16)
    assert surface.shape == (256, 256)
    assert abs(np.mean(surface) - 8000) < 20
    assert abs(np.std(surface) - 800) < 20

    volume = pyspeckle.create_correlated((32, 32, 32), 0, 1, 4)
    assert volume.shape == (32, 32, 32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": 0},
        {"stdev": -2},
        {"cl": -5},
        {"cl": 51},  # correlation longer than half the array
        {"correlation": "banana"},
        {"shape": (8, 8, 8, 8)},
    ],
)
def test_create_correlated_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"shape": 100, "mean": 10, "stdev": 2, "cl": 5}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_correlated(**args)


@pytest.mark.parametrize("sigma,cl", [(1.0, 8), (0.3, 20), (2.5, 4)])
def test_create_phase_screen_statistics(sigma, cl):
    """The screen is zero mean with the requested standard deviation."""
    screen = pyspeckle.create_phase_screen(65536, sigma, cl)
    assert screen.shape == (65536,)
    assert abs(np.mean(screen)) < 0.01 * max(sigma, 0.1)
    assert abs(np.std(screen) - sigma) < 0.01 * sigma


def test_create_phase_screen_2D_statistics():
    """The screen is square, zero mean, with the requested standard deviation."""
    screen = pyspeckle.create_phase_screen((256, 256), 1.5, 8)
    assert screen.shape == (256, 256)
    assert abs(np.mean(screen)) < 0.02
    assert abs(np.std(screen) - 1.5) < 0.02


def test_create_phase_screen_is_gaussian():
    """The phase is normally distributed, unlike the uniform phase of speckle."""
    screen = pyspeckle.create_phase_screen(65536, 1.0, 8)
    assert abs(scipy.stats.skew(screen)) < 0.1
    assert abs(scipy.stats.kurtosis(screen)) < 0.2


@pytest.mark.parametrize("correlation", ["gaussian", "exponential", "GAUSSIAN"])
def test_create_phase_screen_autocorrelation(correlation):
    """The screen reproduces the requested autocorrelation."""
    cl = 8
    screen = pyspeckle.create_phase_screen(65536, 1.0, cl, correlation=correlation)
    acf = pyspeckle.autocorrelation(screen)
    lags = np.array([4, 8, 16])
    if correlation.lower() == "gaussian":
        want = np.exp(-((lags / cl) ** 2))
    else:
        want = np.exp(-lags / cl)
    assert np.max(np.abs(acf[lags] - want)) < 0.05


def test_create_phase_screen_is_isotropic():
    """Correlation must be radial, not the diamond a separable product gives."""
    screen = pyspeckle.create_phase_screen((512, 512), 1.0, 8)
    centred = screen - screen.mean()
    power = np.abs(np.fft.fft2(centred)) ** 2
    acf = np.fft.ifft2(power).real
    acf /= acf[0, 0]

    # the 6-8-10 triple puts all four lags at exactly the same radius, so an
    # isotropic screen gives the same correlation at every one of them
    same_radius = [acf[0, 10], acf[10, 0], acf[6, 8], acf[8, 6]]
    assert max(same_radius) - min(same_radius) < 0.08


def test_create_phase_screen_coherent_fraction():
    """The unscattered fraction of the field is exp(-sigma**2)."""
    for sigma in (0.5, 1.0, 2.0):
        screen = pyspeckle.create_phase_screen(65536, sigma, 8)
        coherent = abs(np.mean(np.exp(1j * screen))) ** 2
        assert abs(coherent - np.exp(-(sigma**2))) < 0.02


def test_create_phase_screen_2D_coherent_fraction():
    """The unscattered fraction of the field is exp(-sigma**2)."""
    for sigma in (0.5, 1.0, 2.0):
        screen = pyspeckle.create_phase_screen((256, 256), sigma, 4)
        coherent = abs(np.mean(np.exp(1j * screen))) ** 2
        assert abs(coherent - np.exp(-(sigma**2))) < 0.03


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": -1},
        {"cl": 0},
        {"cl": -5},
        {"correlation": "banana"},
        {"shape": 1},
        {"shape": (8, 8, 8, 8)},  # four dimensions are not supported
    ],
)
def test_create_phase_screen_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"shape": 64, "sigma": 1.0, "cl": 8}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_phase_screen(**args)


@pytest.mark.parametrize("kwargs", [{"sigma": -1}, {"cl": 0}, {"correlation": "banana"}, {"shape": (1, 1)}])
def test_create_phase_screen_2D_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"shape": (64, 64), "sigma": 1.0, "cl": 8}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_phase_screen(**args)
