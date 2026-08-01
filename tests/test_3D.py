"""Tests for pyspeckle.speckle_3D."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyspeckle


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarization": -5},
        {"polarization": 2},
        {"aperture": "banana"},
        {"pix_per_speckle": 0.5},
        {"shape": (1, 1, 1)},
        {"shape": (8, 8, 8, 8)},  # four dimensions are not supported
    ],
)
def test_create_exponential_3D_invalid_args(kwargs):
    """Bad arguments raise ValueError rather than failing inside numpy."""
    args = {"shape": (8, 8, 8), "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_exponential(**args)


def test_create_unpolarized_uses_beta():
    """Verify beta reaches the mask; the unpolarized recursion used to drop it."""
    M = 24
    speckle = pyspeckle.create_unpolarized((M, M, M), 2, beta=3)
    assert speckle.shape == (M, M, M)

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line.astype(float)) < 0.5)

    x_width = np.mean([half_width(speckle[:, j, k]) for j in range(0, M, 4) for k in range(0, M, 4)])
    z_width = np.mean([half_width(speckle[i, j, :]) for i in range(0, M, 4) for j in range(0, M, 4)])

    # beta>1 stretches the speckle along x; dropping beta leaves it isotropic
    assert x_width / z_width > 1.4


@pytest.mark.parametrize(
    "generator,expected",
    [
        (pyspeckle.create_exponential, 1.0),  # exponential irradiance
        (pyspeckle.create_unpolarized, 1 / np.sqrt(2)),  # gamma-2 irradiance
    ],
)
def test_speckle_contrast_3D(generator, expected):
    """Speckle contrast is unity when polarized and 1/sqrt(2) when not."""
    speckle = generator((32, 32, 32), 2)
    contrast = np.std(speckle) / np.mean(speckle)
    assert abs(contrast - expected) < 0.05


def test_local_contrast_3D_matches_global():
    """Local contrast over a large volume should approach the global contrast."""
    M, n = 32, 9
    speckle = pyspeckle.create_exponential((M, M, M), 2)
    C, K = pyspeckle.local_contrast(speckle, np.ones((n, n, n)))

    # only valid positions of the correlation are returned
    assert C.shape == (M - n + 1,) * 3

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_3D_rejects_wrong_kernel_rank():
    """A 2D kernel cannot be used on a 3D pattern."""
    speckle = pyspeckle.create_exponential((16, 16, 16), 2)
    with pytest.raises(ValueError):
        pyspeckle.local_contrast(speckle, np.ones((3, 3)))


def test_mask_3D_cube():
    """The cube aperture fills the corner block of side 2*radius."""
    mask = pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="cube")  # pylint: disable=protected-access
    assert mask.shape == (16, 16, 16)
    assert mask.sum() == 8**3
    assert mask[0, 0, 0]
    assert mask[7, 7, 7]
    assert not mask[8, 0, 0]


def test_mask_3D_shell():
    """The shell aperture is hollow, keeping points between the two radii."""
    mask = pyspeckle.core._create_mask_3D(16, 2, 4, 4, shape="shell")  # pylint: disable=protected-access
    assert mask.shape == (16, 16, 16)
    assert not mask[4, 4, 4]  # centre is hollow
    assert mask[4, 4, 7]  # inside the outer radius
    assert not mask[0, 0, 0]  # outside the outer radius


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


def test_mask_3D_case_insensitive():
    """The 3D mask should fold case like the 2D one."""
    upper = pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="Ellipsoid")  # pylint: disable=protected-access
    lower = pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="ellipsoid")  # pylint: disable=protected-access
    assert np.array_equal(upper, lower)


def test_mask_3D_unknown_shape():
    """Unknown 3D shapes used to fall through silently to an ellipsoid."""
    with pytest.raises(ValueError):
        pyspeckle.core._create_mask_3D(16, 4, 4, 4, shape="banana")  # pylint: disable=protected-access


def test_mask_3D_too_small():
    """The 3D mask array must be at least twice the largest radius."""
    with pytest.raises(ValueError):
        pyspeckle.core._create_mask_3D(8, 6, 4, 4)  # pylint: disable=protected-access
