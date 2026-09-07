"""Unit tests for OLS regression helpers."""

import numpy as np
import pytest

from src.regression import apply_transform, fit_ols, mean_se_at_t, confidence_band


def test_perfect_line_fit():
    t = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
    y = 100.0 - 0.5 * t  # exact line
    reg = fit_ols(t, y, transform="none")
    assert reg.n == 5
    assert abs(reg.beta0 - 100.0) < 1e-8
    assert abs(reg.beta1 - (-0.5)) < 1e-8
    assert reg.r_squared > 0.9999
    assert reg.df_resid == 3


def test_transform_log():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    yt = apply_transform(y, "log")
    np.testing.assert_allclose(yt, np.log(y))


def test_confidence_band_widens_away_from_mean():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 24, 13)
    y = 100 - 0.3 * t + rng.normal(0, 0.2, size=len(t))
    reg = fit_ols(t, y)
    mean, lo, up = confidence_band(reg, np.array([t.mean(), t.max()]), alpha=0.05, two_sided=True)
    width_center = up[0] - lo[0]
    width_edge = up[1] - lo[1]
    assert width_edge >= width_center - 1e-9


def test_mean_se_positive():
    t = np.array([0.0, 6.0, 12.0, 18.0, 24.0])
    y = np.array([100.0, 98.5, 97.0, 95.8, 94.2])
    reg = fit_ols(t, y)
    se = mean_se_at_t(reg, np.array([12.0]))
    assert se[0] > 0
