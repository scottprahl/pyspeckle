"""Tests for pyspeckle.speckle_3D."""

import numpy as np
import pytest
import pyspeckle


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarization": -5},
        {"polarization": 2},
        {"shape": "banana"},
        {"pix_per_speckle": 0.5},
        {"M": 1},
    ],
)
def test_create_exponential_3D_invalid_args(kwargs):
    """The 3D generator should validate its arguments like the 2D one."""
    args = {"M": 8, "pix_per_speckle": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        pyspeckle.create_exponential_3D(**args)


def test_create_unpolarized_3D_uses_beta():
    """Verify beta reaches the mask; the unpolarized recursion used to drop it."""
    M = 24
    speckle = pyspeckle.create_unpolarized_3D(M, 2, beta=3)
    assert speckle.shape == (M, M, M)

    def half_width(line):
        """Lag at which the autocorrelation first falls below one half."""
        return np.argmax(pyspeckle.autocorrelation(line.astype(float)) < 0.5)

    x_width = np.mean([half_width(speckle[:, j, k]) for j in range(0, M, 4) for k in range(0, M, 4)])
    z_width = np.mean([half_width(speckle[i, j, :]) for i in range(0, M, 4) for j in range(0, M, 4)])

    # beta>1 stretches the speckle along x; dropping beta leaves it isotropic
    assert x_width / z_width > 1.4


def test_local_contrast_3D_matches_global():
    """Local contrast over a large volume should approach the global contrast."""
    M, n = 32, 9
    speckle = pyspeckle.create_exponential_3D(M, 2)
    C, K = pyspeckle.local_contrast_3D(speckle, np.ones((n, n, n)))

    # only valid positions of the correlation are returned
    assert C.shape == (M - n + 1,) * 3

    assert abs(K - 1) < 0.2
    assert abs(np.mean(C) - 1) < 0.2


def test_local_contrast_3D_rejects_wrong_kernel_rank():
    """A 2D kernel cannot be used on a 3D pattern."""
    speckle = pyspeckle.create_exponential_3D(16, 2)
    with pytest.raises(ValueError):
        pyspeckle.local_contrast_3D(speckle, np.ones((3, 3)))


def test_mask_3D_case_insensitive():
    """The 3D mask should fold case like the 2D one."""
    upper = pyspeckle.speckle_3D._create_mask_3D(16, 4, 4, 4, shape="Ellipsoid")  # pylint: disable=protected-access
    lower = pyspeckle.speckle_3D._create_mask_3D(16, 4, 4, 4, shape="ellipsoid")  # pylint: disable=protected-access
    assert np.array_equal(upper, lower)


def test_mask_3D_unknown_shape():
    """Unknown 3D shapes used to fall through silently to an ellipsoid."""
    with pytest.raises(ValueError):
        pyspeckle.speckle_3D._create_mask_3D(16, 4, 4, 4, shape="banana")  # pylint: disable=protected-access


def test_mask_3D_too_small():
    """The 3D mask array must be at least twice the largest radius."""
    with pytest.raises(ValueError):
        pyspeckle.speckle_3D._create_mask_3D(8, 6, 4, 4)  # pylint: disable=protected-access
