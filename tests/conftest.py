"""Shared fixtures for the pyspeckle test suite."""

import numpy as np
import pytest


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
