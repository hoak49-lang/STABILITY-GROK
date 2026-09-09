"""Unit tests for ANCOVA poolability."""

import numpy as np
import pandas as pd
import pytest

from src.pooling import analyze_multibatch, assess_poolability


def _make_pooled_batches():
    """Three batches from nearly the same line → should pool."""
    rows = []
    for b, off in [("A", 0.0), ("B", 0.05), ("C", -0.05)]:
        for t in [0, 3, 6, 9, 12, 18, 24]:
            rows.append(
                {
                    "batch": b,
                    "time": t,
                    "response": 100.0 + off - 0.35 * t,
                    "condition": "25C/60%RH",
                }
            )
    return pd.DataFrame(rows)


def _make_different_slope_batches():
    rows = []
    for b, slope in [("A", -0.2), ("B", -0.6)]:
        for t in [0, 3, 6, 9, 12, 18, 24]:
            rows.append(
                {
                    "batch": b,
                    "time": t,
                    "response": 100.0 + slope * t,
                    "condition": "25C/60%RH",
                }
            )
    return pd.DataFrame(rows)


def test_poolable_batches():
    df = _make_pooled_batches()
    p = assess_poolability(df, alpha_pool=0.25)
    assert p.n_batches == 3
    assert p.pool_all or p.pool_slopes  # nearly identical → poolable
    assert p.recommendation in ("pooled", "common_slope")


def test_different_slopes_not_pooled():
    df = _make_different_slope_batches()
    p = assess_poolability(df, alpha_pool=0.25)
    assert p.slope_equal is False
    assert p.recommendation == "separate"


def test_multibatch_reports_minimum_when_separate():
    df = _make_different_slope_batches()
    res = analyze_multibatch(
        df, spec_limit=90.0, direction="decreasing", alpha=0.05, alpha_pool=0.25
    )
    assert res.reported_mode == "minimum_of_batches"
    assert res.reported_shelf_life is not None
    # Steeper batch B hits 90 sooner on the mean (t=10/0.6≈16.7); shelf < that
    assert res.reported_shelf_life < 30


def test_single_batch_poolability():
    df = _make_pooled_batches()
    df = df[df["batch"] == "A"]
    p = assess_poolability(df, alpha_pool=0.25)
    assert p.n_batches == 1
    assert p.recommendation == "pooled"
