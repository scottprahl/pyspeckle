"""Tests for pyspeckle.plots."""

import matplotlib.pyplot as plt
import numpy as np
import pyspeckle


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


def test_statistics_plot_2D_draws_four_panels():
    """The statistics routine fills a 2x2 figure and reports mean, stdev, contrast."""
    speckle = pyspeckle.create_exponential((64, 64), 2)
    pyspeckle.statistics_plot_2D(speckle)
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    assert "Standard Deviation" in axes[1].get_title()
    assert "Speckle Contrast" in axes[3].get_title()


def test_statistics_plot_2D_normalized_pdf():
    """The plotted histogram is a density, so its bars integrate to unity."""
    speckle = pyspeckle.create_exponential((64, 64), 2)
    pyspeckle.statistics_plot_2D(speckle)
    bars = plt.gcf().get_axes()[1].patches
    # bars are drawn at 70% of the bin width, so widen them back out
    total = sum(patch.get_height() * (patch.get_width() / 0.7) for patch in bars)
    assert abs(total - 1) < 0.01


def test_statistics_plot_2D_accepts_masked_array():
    """A masked array uses compressed() for the statistics and blue for bad pixels."""
    speckle = pyspeckle.create_exponential((64, 64), 2)
    masked = np.ma.masked_where(speckle > 0.8, speckle)
    pyspeckle.statistics_plot_2D(masked)
    assert len(plt.gcf().get_axes()) == 4


def test_statistics_plot_2D_without_initialize():
    """initialize=False draws onto an existing figure instead of making one."""
    speckle = pyspeckle.create_exponential((64, 64), 2)
    plt.subplots(2, 2)
    pyspeckle.statistics_plot_2D(speckle, initialize=False)
    assert len(plt.gcf().get_axes()) == 4


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


def test_local_contrast_2D_plot_draws_four_panels():
    """The plot routine fills a 2x2 figure and labels the overall contrast."""
    speckle = pyspeckle.create_exponential((64, 64), 2)
    pyspeckle.local_contrast_2D_plot(speckle, np.ones((5, 5)))
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    assert "Overall Contrast" in axes[0].get_title()
    assert axes[2].get_title() == "Local speckle contrast"


def test_local_contrast_2D_plot_histograms_are_densities():
    """Both panels labelled PDF must integrate to unity, not show raw counts."""
    speckle = pyspeckle.create_exponential((64, 64), 2)
    pyspeckle.local_contrast_2D_plot(speckle, np.ones((5, 5)))
    for panel in (plt.gcf().get_axes()[1], plt.gcf().get_axes()[3]):
        # bars are drawn at 70% of the bin width, so widen them back out
        total = sum(p.get_height() * (p.get_width() / 0.7) for p in panel.patches)
        assert abs(total - 1) < 0.01


def test_slice_plot_draws_three_slices():
    """slice_plot fills a 2x2 figure with one panel per axis and a blank fourth."""
    data = pyspeckle.create_exponential((16, 16, 16), 2)
    pyspeckle.slice_plot(data, 8, 8, 8)
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    assert axes[0].get_title() == "Constant Z=8 values"
    assert axes[1].get_title() == "Constant Y=8 values"
    assert axes[2].get_title() == "Constant X=8 values"
    assert not axes[3].axison  # fourth panel is switched off


def test_slice_plot_without_sqrt_or_initialize():
    """show_sqrt=False plots raw irradiance; initialize=False reuses the figure."""
    data = pyspeckle.create_exponential((16, 16, 16), 2)
    plt.subplots(2, 2)
    pyspeckle.slice_plot(data, 8, 8, 8, initialize=False, show_sqrt=False)
    axes = plt.gcf().get_axes()
    assert len(axes) == 4
    # raw irradiance is normalized to a max of one, not scaled to 0-255
    assert axes[0].get_images()[0].get_array().max() <= 1.0
