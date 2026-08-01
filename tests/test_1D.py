"""Tests for pyspeckle.speckle_1D."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
import pyspeckle


def test_create_exponential_shape_and_range():
    """Output is length M, normalized to a maximum of one."""
    result = pyspeckle.create_exponential(64, 2)
    assert result.shape == (64,)
    assert 0 <= np.min(result)
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


def test_create_exponential_polarization_sweep():
    """Contrast falls monotonically from 1 to 1/sqrt(2) as polarization goes to zero."""
    contrasts = []
    for polarization in (1.0, 0.75, 0.5, 0.25, 0.0):
        speckle = pyspeckle.create_exponential(16384, 2, polarization=polarization)
        contrasts.append(np.std(speckle) / np.mean(speckle))
    assert abs(contrasts[0] - 1) < 0.05
    assert abs(contrasts[-1] - 1 / np.sqrt(2)) < 0.05
    assert np.all(np.diff(contrasts) < 0.01)  # non-increasing


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


def test_create_exponential_speckle_size():
    """Speckle size grows in proportion to pix_per_speckle."""

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line) < 0.5)

    narrow = half_width(pyspeckle.create_exponential(4096, 2))
    wide = half_width(pyspeckle.create_exponential(4096, 8))
    assert wide > 2 * narrow


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


@pytest.mark.parametrize("sigma,cl", [(1.0, 8), (0.3, 20), (2.5, 4)])
def test_create_phase_screen_statistics(sigma, cl):
    """The screen is zero mean with the requested standard deviation."""
    screen = pyspeckle.create_phase_screen(65536, sigma, cl)
    assert screen.shape == (65536,)
    assert abs(np.mean(screen)) < 0.01 * max(sigma, 0.1)
    assert abs(np.std(screen) - sigma) < 0.01 * sigma


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


def test_create_phase_screen_coherent_fraction():
    """The unscattered fraction of the field is exp(-sigma**2)."""
    for sigma in (0.5, 1.0, 2.0):
        screen = pyspeckle.create_phase_screen(65536, sigma, 8)
        coherent = abs(np.mean(np.exp(1j * screen))) ** 2
        assert abs(coherent - np.exp(-(sigma**2))) < 0.02


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


def test_local_contrast_1D_matches_global():
    """Local contrast over a long window should approach the global contrast."""
    speckle = pyspeckle.create_exponential(8192, 2)
    n = 51
    C, K = pyspeckle.local_contrast(speckle, np.ones(n))

    # only valid positions of the correlation are returned
    assert C.shape == (speckle.size - n + 1,)

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_1D_plot_draws_four_panels():
    """The left column plots points, the right column histograms."""
    speckle = pyspeckle.create_exponential(512, 8)
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
    speckle = pyspeckle.create_exponential(1024, 8)
    pyspeckle.local_contrast_1D_plot(speckle, np.ones(25))
    for panel in (plt.gcf().get_axes()[1], plt.gcf().get_axes()[3]):
        # bars are drawn at 70% of the bin width, so widen them back out
        total = sum(p.get_height() * (p.get_width() / 0.7) for p in panel.patches)
        assert abs(total - 1) < 0.01


def test_local_contrast_1D_plot_aligns_the_two_traces():
    """The contrast trace is centred on its window, so positions line up."""
    speckle = pyspeckle.create_exponential(512, 8)
    n = 25
    pyspeckle.local_contrast_1D_plot(speckle, np.ones(n))
    axes = plt.gcf().get_axes()
    contrast_x = axes[2].get_lines()[0].get_xdata()
    assert contrast_x[0] == (n - 1) / 2
    assert contrast_x[-1] == speckle.size - 1 - (n - 1) / 2


def test_statistics_plot_1D_draws_four_panels():
    """Four panels: the trace, its PDF, the PSD, and the PDF on a log scale."""
    speckle = pyspeckle.create_exponential(4096, 8)
    pyspeckle.statistics_plot_1D(speckle)
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    assert "Contrast" in axes[0].get_title()
    assert "Standard Deviation" in axes[1].get_title()
    assert axes[2].get_title() == "Power Spectral Density"
    assert "Speckle Contrast" in axes[3].get_title()


def test_statistics_plot_1D_normalized_pdf():
    """The plotted histogram is a density, so its bars integrate to unity."""
    speckle = pyspeckle.create_exponential(4096, 8)
    pyspeckle.statistics_plot_1D(speckle)
    bars = plt.gcf().get_axes()[1].patches
    # bars are drawn at 70% of the bin width, so widen them back out
    total = sum(p.get_height() * (p.get_width() / 0.7) for p in bars)
    assert abs(total - 1) < 0.01


def test_statistics_plot_1D_psd_shows_band_limit():
    """The PSD cuts off at 1/pix_per_speckle, which is what makes it useful."""
    pix_per_speckle = 8
    speckle = pyspeckle.create_exponential(8192, pix_per_speckle)
    pyspeckle.statistics_plot_1D(speckle)

    freq, psd = plt.gcf().get_axes()[2].get_lines()[0].get_data()
    significant = np.abs(freq[psd > psd.max() * 1e-6])
    assert abs(significant.max() - 1 / pix_per_speckle) < 0.01


def test_statistics_plot_1D_masked_and_no_initialize():
    """A masked array uses compressed(); initialize=False reuses the figure."""
    speckle = pyspeckle.create_exponential(2048, 8)
    masked = np.ma.masked_where(speckle > 0.8, speckle)
    pyspeckle.statistics_plot_1D(masked)
    assert len(plt.gcf().get_axes()) == 4

    plt.subplots(2, 2)
    pyspeckle.statistics_plot_1D(speckle, initialize=False)
    assert len(plt.gcf().get_axes()) == 4


def test_local_contrast_1D_rejects_wrong_kernel_rank():
    """A 2D kernel cannot be used on a 1D pattern."""
    speckle = pyspeckle.create_exponential(256, 2)
    with pytest.raises(ValueError):
        pyspeckle.local_contrast(speckle, np.ones((3, 3)))


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
