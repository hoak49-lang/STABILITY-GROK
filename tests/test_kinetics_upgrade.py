"""Tests for upgraded kinetics: orders, RH parse, CI, T+RH, shelf-life."""

import numpy as np
import pandas as pd
import pytest

from src.arrhenius import (
    R_KCAL,
    estimate_rates_by_condition,
    fit_arrhenius,
    normalize_order,
    parse_rh_percent,
    parse_temperature_c,
    predict_k,
    project_shelf_life,
)


def test_normalize_order_aliases():
    assert normalize_order("zero") == "zero"
    assert normalize_order("first-order") == "first"
    assert normalize_order("2") == "second"


def test_parse_rh_from_condition():
    assert parse_rh_percent("25C/60%RH") == 60.0
    assert parse_rh_percent("40°C/75% RH") == 75.0
    assert parse_rh_percent("30C") is None
    assert parse_temperature_c("25C/60%RH") == 25.0


def test_first_order_rate_estimation():
    rows = []
    for t in [0, 3, 6, 9, 12]:
        rows.append(
            {
                "batch": "A",
                "time": t,
                "response": 100.0 * np.exp(-0.01 * t),
                "condition": "40C/75%RH",
            }
        )
    df = pd.DataFrame(rows)
    rates = estimate_rates_by_condition(df, kinetics_order="first")
    assert len(rates) == 1
    assert rates.iloc[0]["rate_abs_slope"] == pytest.approx(0.01, rel=1e-4)
    assert rates.iloc[0]["rh_percent"] == 75.0


def test_second_order_rate_estimation():
    k2 = 0.001
    rows = []
    for t in [0, 5, 10, 15, 20]:
        y = 1.0 / (0.01 + k2 * t)
        rows.append({"batch": "A", "time": t, "response": y, "condition": "50C"})
    rates = estimate_rates_by_condition(pd.DataFrame(rows), kinetics_order="second")
    assert rates.iloc[0]["rate_abs_slope"] == pytest.approx(k2, rel=1e-3)


def test_fit_arrhenius_has_ea_ci():
    ea_true, ln_A = 18.0, 25.0
    rows = []
    for tc in [25, 30, 40, 50]:
        k = float(np.exp(ln_A - ea_true / (R_KCAL * (tc + 273.15))))
        rows.append({"temp_c": tc, "k": k, "rh_percent": 60.0})
    res = fit_arrhenius(pd.DataFrame(rows), r_unit="kcal", predict_temp_c=25.0)
    assert res.valid
    assert res.ea_ci_low is not None and res.ea_ci_high is not None
    assert res.ea_ci_low < res.ea < res.ea_ci_high
    assert res.predicted_k_ci_low is not None
    assert res.predicted_k_ci_low <= res.predicted_k <= res.predicted_k_ci_high


def test_t_rh_model_recovers_positive_B():
    rows = []
    ln_A, slope, B = 20.0, -8000.0, 1.5
    for tc, rh in [(25, 60), (30, 65), (40, 75), (50, 75), (60, 40)]:
        inv = 1.0 / (tc + 273.15)
        lnk = ln_A + slope * inv + B * (rh / 100.0)
        rows.append({"temp_c": tc, "k": float(np.exp(lnk)), "rh_percent": rh})
    res = fit_arrhenius(
        pd.DataFrame(rows),
        r_unit="kcal",
        include_rh=True,
        predict_temp_c=25.0,
        predict_rh=60.0,
    )
    assert res.valid
    assert res.model_kind == "t_rh"
    assert res.rh_coef_B == pytest.approx(1.5, rel=1e-5, abs=1e-4)


def test_project_shelf_life_first_order():
    proj = project_shelf_life(
        rate_k=0.01,
        intercept=100.0,
        spec_limit=90.0,
        direction="decreasing",
        kinetics_order="first",
        rate_k_low=0.008,
        rate_k_high=0.012,
    )
    assert proj.valid
    expected = float(np.log(100 / 90) / 0.01)
    assert proj.shelf_life == pytest.approx(expected, rel=1e-6)
    assert proj.shelf_life_ci_low is not None
    assert proj.shelf_life_ci_low < proj.shelf_life < proj.shelf_life_ci_high


def test_predict_k_with_rh_term():
    k = predict_k(10.0, -5000.0, 25.0, rh_coef_B=1.0, rh_percent=60.0)
    inv = 1.0 / (25 + 273.15)
    expected2 = float(np.exp(10.0 + (-5000.0) * inv + 1.0 * 0.6))
    assert k == pytest.approx(expected2)
