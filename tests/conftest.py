"""Shared fixtures for the pyspeckle test suite."""

import matplotlib

# must precede any import of pyplot, which pyspeckle does at import time
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import pytest  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def seed_rng():
    """
    Seed numpy before every test.

    Several tests assert on sample statistics whose tolerances are only a few
    standard deviations wide.  Unseeded, create_gaussian_corr_1D(1000, 10, 2, 5)
    lands outside its 0.5 tolerance in about one run per hundred, which is
    frequent enough to fail CI at random.
    """
    np.random.seed(0)


@pytest.fixture(autouse=True)
def close_figures():
    """
    Close any figures a test opened.

    The plotting routines draw onto the current figure and never close it, so
    without this matplotlib warns once more than 20 accumulate.
    """
    yield
    plt.close("all")
