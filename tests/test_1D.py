"""Tests for pyspeckle.speckle_1D."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
import pyspeckle


def test_create_exponential_1D_shape_and_range():
    """Output is length M, normalized to a maximum of one."""
    result = pyspeckle.create_exponential_1D(64, 2)
    assert result.shape == (64,)
    assert 0 <= np.min(result)
    assert np.max(result) <= 1.0


@pytest.mark.parametrize(
    "generator,expected",
    [
        (pyspeckle.create_exponential_1D, 1.0),  # exponential irradiance
        (pyspeckle.create_unpolarized_1D, 1 / np.sqrt(2)),  # gamma-2 irradiance
    ],
)
def test_speckle_contrast_1D(generator, expected):
    """Speckle contrast is unity when polarized and 1/sqrt(2) when not."""
    speckle = generator(16384, 2)
    contrast = np.std(speckle) / np.mean(speckle)
    assert abs(contrast - expected) < 0.05


def test_create_exponential_1D_polarization_sweep():
    """Contrast falls monotonically from 1 to 1/sqrt(2) as polarization goes to zero."""
    contrasts = []
    for polarization in (1.0, 0.75, 0.5, 0.25, 0.0):
        speckle = pyspeckle.create_exponential_1D(16384, 2, polarization=polarization)
        contrasts.append(np.std(speckle) / np.mean(speckle))
    assert abs(contrasts[0] - 1) < 0.05
    assert abs(contrasts[-1] - 1 / np.sqrt(2)) < 0.05
    assert np.all(np.diff(contrasts) < 0.01)  # non-increasing


def test_create_unpolarized_1D_matches_zero_polarization():
    """The wrapper is exactly create_exponential_1D(..., polarization=0)."""
    np.random.seed(3)
    wrapper = pyspeckle.create_unpolarized_1D(256, 2)
    np.random.seed(3)
    explicit = pyspeckle.create_exponential_1D(256, 2, polarization=0)
    assert np.array_equal(wrapper, explicit)


def test_create_unpolarized_1D_irradiance_is_gamma2():
    """Unpolarized irradiance follows a gamma distribution with shape 2."""
    speckle = pyspeckle.create_unpolarized_1D(16384, 2)
    speckle = speckle / np.mean(speckle)
    # decimate to near-independent samples; neighbours within a speckle correlate
    assert scipy.stats.kstest(speckle[::16], "gamma", args=(2, 0, 0.5)).pvalue > 0.01


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


@pytest.mark.parametrize("sigma,cl", [(1.0, 8), (0.3, 20), (2.5, 4)])
def test_create_phase_screen_1D_statistics(sigma, cl):
    """The screen is zero mean with the requested standard deviation."""
    screen = pyspeckle.create_phase_screen_1D(65536, sigma, cl)
    assert screen.shape == (65536,)
    assert abs(np.mean(screen)) < 0.01 * max(sigma, 0.1)
    assert abs(np.std(screen) - sigma) < 0.01 * sigma


def test_create_phase_screen_1D_is_gaussian():
    """The phase is normally distributed, unlike the uniform phase of speckle."""
    screen = pyspeckle.create_phase_screen_1D(65536, 1.0, 8)
    assert abs(scipy.stats.skew(screen)) < 0.1
    assert abs(scipy.stats.kurtosis(screen)) < 0.2


@pytest.mark.parametrize("shape", ["gaussian", "exponential", "GAUSSIAN"])
def test_create_phase_screen_1D_autocorrelation(shape):
    """The screen reproduces the requested autocorrelation."""
    cl = 8
    screen = pyspeckle.create_phase_screen_1D(65536, 1.0, cl, shape=shape)
    acf = pyspeckle.autocorrelation(screen)
    lags = np.array([4, 8, 16])
    if shape.lower() == "gaussian":
        want = np.exp(-0.5 * (lags / cl) ** 2)
    else:
        want = np.exp(-lags / cl)
    assert np.max(np.abs(acf[lags] - want)) < 0.05


def test_create_phase_screen_1D_coherent_fraction():
    """The unscattered fraction of the field is exp(-sigma**2)."""
    for sigma in (0.5, 1.0, 2.0):
        screen = pyspeckle.create_phase_screen_1D(65536, sigma, 8)
        coherent = abs(np.mean(np.exp(1j * screen))) ** 2
        assert abs(coherent - np.exp(-(sigma**2))) < 0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": -1},
        {"cl": 0},
        {"cl": -5},
        {"shape": "banana"},
        {"M": 1},
    ],
)
def test_create_phase_screen_1D_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"M": 64, "sigma": 1.0, "cl": 8}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_phase_screen_1D(**args)


def test_local_contrast_1D_matches_global():
    """Local contrast over a long window should approach the global contrast."""
    speckle = pyspeckle.create_exponential_1D(8192, 2)
    n = 51
    C, K = pyspeckle.local_contrast_1D(speckle, np.ones(n))

    # only valid positions of the correlation are returned
    assert C.shape == (speckle.size - n + 1,)

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_1D_plot_draws_four_panels():
    """The left column plots points, the right column histograms."""
    speckle = pyspeckle.create_exponential_1D(512, 8)
    pyspeckle.local_contrast_1D_plot(speckle, np.ones(25))
    axes = plt.gcf().get_axes()
    assert len(axes) == 4

    # left column: unconnected points, no images and no bars
    for panel in (axes[0], axes[2]):
        assert len(panel.get_lines()) == 1
        trace = panel.get_lines()[0]
        assert trace.get_marker() == "."
        assert trace.get_linestyle() == "None"
        assert not panel.get_images()
        assert not panel.patches

    # right column: histograms
    assert len(axes[1].patches) == 30
    assert len(axes[3].patches) == 20

    assert "Overall Contrast" in axes[0].get_title()
    assert axes[2].get_title() == "Local speckle contrast"


def test_local_contrast_1D_plot_histograms_are_densities():
    """Both panels labelled PDF must integrate to unity, not show raw counts."""
    speckle = pyspeckle.create_exponential_1D(1024, 8)
    pyspeckle.local_contrast_1D_plot(speckle, np.ones(25))
    for panel in (plt.gcf().get_axes()[1], plt.gcf().get_axes()[3]):
        # bars are drawn at 70% of the bin width, so widen them back out
        total = sum(p.get_height() * (p.get_width() / 0.7) for p in panel.patches)
        assert abs(total - 1) < 0.01


def test_local_contrast_1D_plot_aligns_the_two_traces():
    """The contrast trace is centred on its window, so positions line up."""
    speckle = pyspeckle.create_exponential_1D(512, 8)
    n = 25
    pyspeckle.local_contrast_1D_plot(speckle, np.ones(n))
    axes = plt.gcf().get_axes()
    contrast_x = axes[2].get_lines()[0].get_xdata()
    assert contrast_x[0] == (n - 1) / 2
    assert contrast_x[-1] == speckle.size - 1 - (n - 1) / 2


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
