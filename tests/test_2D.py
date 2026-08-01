"""Tests for pyspeckle.speckle_2D."""

import matplotlib.pyplot as plt
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


@pytest.mark.parametrize(
    "generator,expected",
    [
        (pyspeckle.create_exponential_2D, 1.0),  # exponential irradiance
        (pyspeckle.create_unpolarized_2D, 1 / np.sqrt(2)),  # gamma-2 irradiance
    ],
)
def test_speckle_contrast_2D(generator, expected):
    """Speckle contrast is unity when polarized and 1/sqrt(2) when not."""
    speckle = generator(256, 2)
    contrast = np.std(speckle) / np.mean(speckle)
    assert abs(contrast - expected) < 0.05


def test_speckle_contrast_2D_independent_of_resolution():
    """Contrast is a first-order statistic, so pix_per_speckle must not change it."""
    for pix_per_speckle in (2, 4, 8):
        speckle = pyspeckle.create_exponential_2D(256, pix_per_speckle)
        assert abs(np.std(speckle) / np.mean(speckle) - 1) < 0.1


def test_create_phase_screen_2D_statistics():
    """The screen is square, zero mean, with the requested standard deviation."""
    screen = pyspeckle.create_phase_screen_2D(256, 1.5, 8)
    assert screen.shape == (256, 256)
    assert abs(np.mean(screen)) < 0.02
    assert abs(np.std(screen) - 1.5) < 0.02


def test_create_phase_screen_2D_is_isotropic():
    """Correlation must be radial, not the diamond a separable product gives."""
    screen = pyspeckle.create_phase_screen_2D(512, 1.0, 8)
    centred = screen - screen.mean()
    power = np.abs(np.fft.fft2(centred)) ** 2
    acf = np.fft.ifft2(power).real
    acf /= acf[0, 0]

    # the 6-8-10 triple puts all four lags at exactly the same radius, so an
    # isotropic screen gives the same correlation at every one of them
    same_radius = [acf[0, 10], acf[10, 0], acf[6, 8], acf[8, 6]]
    assert max(same_radius) - min(same_radius) < 0.08


def test_create_phase_screen_2D_coherent_fraction():
    """The unscattered fraction of the field is exp(-sigma**2)."""
    for sigma in (0.5, 1.0, 2.0):
        screen = pyspeckle.create_phase_screen_2D(256, sigma, 4)
        coherent = abs(np.mean(np.exp(1j * screen))) ** 2
        assert abs(coherent - np.exp(-(sigma**2))) < 0.03


@pytest.mark.parametrize("kwargs", [{"sigma": -1}, {"cl": 0}, {"shape": "banana"}, {"M": 1}])
def test_create_phase_screen_2D_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"M": 64, "sigma": 1.0, "cl": 8}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_phase_screen_2D(**args)


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


def test_local_contrast_2D_integer_image():
    """An 8-bit image must not overflow when squared; it used to give all zeros."""
    speckle = pyspeckle.create_exponential_2D(128, 4)
    as_uint8 = (255 * speckle).astype(np.uint8)

    C, K = pyspeckle.local_contrast_2D(as_uint8, np.ones((10, 10)))
    assert np.all(np.isfinite(C))
    assert C.max() > 0.1  # was exactly 0 everywhere while uint8 wrapped
    assert abs(K - 1) < 0.2

    # the same data as float must give the same answer
    C_float, _ = pyspeckle.local_contrast_2D(as_uint8.astype(float), np.ones((10, 10)))
    assert np.allclose(C, C_float)


def test_mask_2D_too_small():
    """The mask array must be at least twice the largest radius."""
    with pytest.raises(ValueError):
        pyspeckle.speckle_2D._create_mask(6, 5, 4)  # pylint: disable=protected-access


def test_mask_2D_unknown_shape():
    """An unrecognized aperture shape is rejected."""
    with pytest.raises(ValueError):
        pyspeckle.speckle_2D._create_mask(10, 3, 4, shape="banana")  # pylint: disable=protected-access


def test_local_contrast_2D_plot_draws_four_panels():
    """The plot routine fills a 2x2 figure and labels the overall contrast."""
    speckle = pyspeckle.create_exponential_2D(64, 2)
    pyspeckle.local_contrast_2D_plot(speckle, np.ones((5, 5)))
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    assert "Overall Contrast" in axes[0].get_title()
    assert axes[2].get_title() == "Local speckle contrast"


def test_local_contrast_2D_plot_histograms_are_densities():
    """Both panels labelled PDF must integrate to unity, not show raw counts."""
    speckle = pyspeckle.create_exponential_2D(64, 2)
    pyspeckle.local_contrast_2D_plot(speckle, np.ones((5, 5)))
    for panel in (plt.gcf().get_axes()[1], plt.gcf().get_axes()[3]):
        # bars are drawn at 70% of the bin width, so widen them back out
        total = sum(p.get_height() * (p.get_width() / 0.7) for p in panel.patches)
        assert abs(total - 1) < 0.01


def test_statistics_plot_2D_draws_four_panels():
    """The statistics routine fills a 2x2 figure and reports mean, stdev, contrast."""
    speckle = pyspeckle.create_exponential_2D(64, 2)
    pyspeckle.statistics_plot_2D(speckle)
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    assert "Standard Deviation" in axes[1].get_title()
    assert "Speckle Contrast" in axes[3].get_title()


def test_statistics_plot_2D_normalized_pdf():
    """The plotted histogram is a density, so its bars integrate to unity."""
    speckle = pyspeckle.create_exponential_2D(64, 2)
    pyspeckle.statistics_plot_2D(speckle)
    bars = plt.gcf().get_axes()[1].patches
    # bars are drawn at 70% of the bin width, so widen them back out
    total = sum(patch.get_height() * (patch.get_width() / 0.7) for patch in bars)
    assert abs(total - 1) < 0.01


def test_statistics_plot_2D_accepts_masked_array():
    """A masked array uses compressed() for the statistics and blue for bad pixels."""
    speckle = pyspeckle.create_exponential_2D(64, 2)
    masked = np.ma.masked_where(speckle > 0.8, speckle)
    pyspeckle.statistics_plot_2D(masked)
    assert len(plt.gcf().get_axes()) == 4


def test_statistics_plot_2D_without_initialize():
    """initialize=False draws onto an existing figure instead of making one."""
    speckle = pyspeckle.create_exponential_2D(64, 2)
    plt.subplots(2, 2)
    pyspeckle.statistics_plot_2D(speckle, initialize=False)
    assert len(plt.gcf().get_axes()) == 4


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
