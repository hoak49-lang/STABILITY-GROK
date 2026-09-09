"""Unit tests for Arrhenius fit and k prediction at known toy values."""

import numpy as np
import pandas as pd
import pytest

from src.arrhenius import (
    R_KCAL,
    R_KJ,
    estimate_rates_by_condition,
    fit_arrhenius,
    normalize_rates_table,
    parse_temperature_c,
    predict_k,
    project_shelf_life_zero_order,
    resolve_r,
)


def _toy_rates(ea: float, ln_A: float, temps_c, r: float = R_KCAL) -> pd.DataFrame:
    rows = []
    for tc in temps_c:
        k = float(np.exp(ln_A - ea / (r * (tc + 273.15))))
        rows.append({"temp_c": tc, "k": k})
    return pd.DataFrame(rows)


def test_parse_temperature_variants():
    assert parse_temperature_c("25C/60%RH") == 25.0
    assert parse_temperature_c("40°C") == 40.0
    assert parse_temperature_c("30C") == 30.0
    assert parse_temperature_c("50") == 50.0
    assert parse_temperature_c("ambient") is None


def test_resolve_r_units():
    r, label, ea_u = resolve_r("kcal")
    assert r == pytest.approx(R_KCAL)
    assert "kcal" in ea_u
    r2, _, ea2 = resolve_r("kJ")
    assert r2 == pytest.approx(R_KJ)
    assert ea2.startswith("kJ")


def test_fit_arrhenius_recovers_known_ea_kcal():
    ea_true, ln_A_true = 18.0, 25.0
    df = _toy_rates(ea_true, ln_A_true, [25, 30, 40, 50], r=R_KCAL)
    res = fit_arrhenius(df, r_unit="kcal", predict_temp_c=25.0)
    assert res.valid
    assert res.ea == pytest.approx(ea_true, rel=1e-6, abs=1e-4)
    assert res.ln_A == pytest.approx(ln_A_true, rel=1e-6, abs=1e-4)
    assert res.r_squared == pytest.approx(1.0, abs=1e-10)
    assert res.ea_unit == "kcal/mol"
    # predicted k at 25 matches constructed
    k25 = float(np.exp(ln_A_true - ea_true / (R_KCAL * (25 + 273.15))))
    assert res.predicted_k == pytest.approx(k25, rel=1e-6)
    assert res.predicted_rate_at_25 == pytest.approx(k25, rel=1e-6)


def test_fit_arrhenius_kj_unit():
    ea_kcal, ln_A = 18.0, 25.0
    ea_kj = ea_kcal * 4.184
    df = _toy_rates(ea_kj, ln_A, [25, 40, 60], r=R_KJ)
    res = fit_arrhenius(df, r_unit="kJ", predict_temp_c=25.0)
    assert res.valid
    assert res.ea == pytest.approx(ea_kj, rel=1e-5, abs=1e-3)
    assert res.ea_unit == "kJ/mol"
    assert res.ea_kcal_mol == pytest.approx(ea_kcal, rel=1e-5, abs=1e-3)


def test_predict_k_at_custom_temperature():
    ea_true, ln_A_true = 15.0, 20.0
    df = _toy_rates(ea_true, ln_A_true, [30, 40, 50], r=R_KCAL)
    res = fit_arrhenius(df, r_unit="kcal", predict_temp_c=25.0)
    assert res.valid
    k30 = predict_k(res.ln_A, res.fit_slope, 30.0)
    expected = float(np.exp(ln_A_true - ea_true / (R_KCAL * (30 + 273.15))))
    assert k30 == pytest.approx(expected, rel=1e-5)


def test_normalize_accepts_k_alias():
    raw = pd.DataFrame({"temperature": [25, 40], "rate": [0.01, 0.05]})
    out = normalize_rates_table(raw)
    assert list(out.columns)[:2] == ["temp_c", "rate_abs_slope"]
    assert len(out) == 2


def test_insufficient_points_invalid():
    df = pd.DataFrame({"temp_c": [25], "k": [0.01]})
    res = fit_arrhenius(df)
    assert not res.valid
    assert res.ea is None


def test_zero_or_negative_rates_dropped():
    df = pd.DataFrame({"temp_c": [25, 40, 50], "k": [0.01, 0.0, -1.0]})
    res = fit_arrhenius(df)
    assert not res.valid  # only one positive rate left


def test_project_shelf_life_decreasing():
    # intercept 100, spec 90, k=0.4 → t* = 25
    proj = project_shelf_life_zero_order(
        rate_k=0.4, intercept=100.0, spec_limit=90.0, direction="decreasing", temp_c=25.0
    )
    assert proj.valid
    assert proj.shelf_life == pytest.approx(25.0)


def test_project_shelf_life_increasing():
    proj = project_shelf_life_zero_order(
        rate_k=0.02, intercept=0.05, spec_limit=0.5, direction="increasing", temp_c=25.0
    )
    assert proj.valid
    assert proj.shelf_life == pytest.approx((0.5 - 0.05) / 0.02)


def test_project_already_oos():
    proj = project_shelf_life_zero_order(
        rate_k=0.1, intercept=85.0, spec_limit=90.0, direction="decreasing"
    )
    assert proj.valid
    assert proj.shelf_life == 0.0


def test_estimate_rates_by_condition_smoke():
    # linear response at two temps
    rows = []
    for tc, slope in [(25, -0.25), (40, -1.0)]:
        for t in [0, 3, 6, 9, 12]:
            rows.append(
                {
                    "batch": "A",
                    "time": t,
                    "response": 100 + slope * t,
                    "condition": f"{tc}C",
                }
            )
    df = pd.DataFrame(rows)
    rates = estimate_rates_by_condition(df)
    assert len(rates) == 2
    assert set(rates["temp_c"]) == {25.0, 40.0}
    r25 = rates.loc[rates["temp_c"] == 25.0, "rate_abs_slope"].iloc[0]
    assert r25 == pytest.approx(0.25, abs=1e-6)
    arr = fit_arrhenius(rates, r_unit="kcal", predict_temp_c=25.0)
    assert arr.valid
