"""Unit tests for shelf-life estimation direction and rough magnitude."""

import numpy as np
import pytest

from src.regression import estimate_shelf_life, fit_ols


def test_decreasing_assay_shelf_life_positive():
    # Nearly deterministic decreasing assay; LSL = 90
    t = np.array([0.0, 3, 6, 9, 12, 18, 24], dtype=float)
    y = 100.0 - 0.4 * t  # hits 90 at t=25 on the mean
    reg = fit_ols(t, y)
    sl = estimate_shelf_life(reg, spec_limit=90.0, direction="decreasing", alpha=0.05, t_max=60)
    assert sl.intersect_ok
    assert sl.shelf_life is not None
    assert sl.shelf_life > 0
    # One-sided lower bound intersects earlier than mean-cross (25)
    assert sl.shelf_life < 25.0
    # Should still be reasonably near (noise-free → close to 25, slightly less)
    assert sl.shelf_life > 15.0


def test_increasing_impurity_shelf_life():
    t = np.array([0.0, 3, 6, 9, 12, 18, 24], dtype=float)
    y = 0.05 + 0.02 * t  # mean hits 0.5 at t=22.5
    reg = fit_ols(t, y)
    sl = estimate_shelf_life(reg, spec_limit=0.5, direction="increasing", alpha=0.05, t_max=60)
    assert sl.intersect_ok
    assert sl.shelf_life is not None
    assert 5.0 < sl.shelf_life < 22.5


def test_already_out_of_spec_at_zero():
    t = np.array([0.0, 3, 6, 9, 12], dtype=float)
    y = 85.0 - 0.2 * t  # already below 90
    reg = fit_ols(t, y)
    sl = estimate_shelf_life(reg, spec_limit=90.0, direction="decreasing", alpha=0.05)
    assert sl.shelf_life == 0.0


def test_flat_line_may_not_intersect():
    t = np.array([0.0, 6, 12, 18, 24], dtype=float)
    y = np.array([99.0, 99.1, 98.9, 99.0, 99.05])
    reg = fit_ols(t, y)
    sl = estimate_shelf_life(reg, spec_limit=90.0, direction="decreasing", alpha=0.05, t_max=36)
    # With near-zero slope and high intercept, may not intersect in range
    # Either no intersect or a very large shelf life
    if sl.shelf_life is not None:
        assert sl.shelf_life > 20
