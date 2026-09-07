"""Optional Arrhenius exploratory analysis (SUPPORTIVE ONLY).

NOT a substitute for real-time ICH Q1A / Q1E long-term stability evaluation.
Clearly labeled as exploratory / supportive in the UI and README.

Model
-----
    ln(k) = ln(A) − Ea / (R · T)

where:
  - k  = degradation rate (user-entered, or |slope| of response vs time
         as a zero-order rate proxy on the modeling scale)
  - T  = absolute temperature (Kelvin)
  - R  = gas constant (kcal/(mol·K) or kJ/(mol·K))
  - Ea = activation energy (same energy unit as R·mol)
  - A  = pre-exponential factor

Assumptions (documented for the user)
-------------------------------------
  1. Rate proxy: when deriving k from stability data, k ≈ |OLS slope| of
     response vs time (zero-order kinetics on the chosen scale). This is a
     common exploratory simplification — not a full kinetic model selection.
  2. First-order option is NOT fitted here; users may enter ln-transformed
     rates externally if they prefer a first-order proxy.
  3. Humidity / packaging / other factors are ignored.
  4. Predicted shelf life from extrapolated k uses a simple zero-order mean
     crossing (intercept → spec) — NOT the ICH Q1E one-sided CI of the mean.
  5. Results are SUPPORTIVE / exploratory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .regression import fit_ols

# Gas constants
R_KCAL = 0.0019872042586067  # kcal/(mol·K)
R_KJ = 0.008314462618  # kJ/(mol·K)

R_OPTIONS = {
    "kcal": (R_KCAL, "kcal/(mol·K)", "kcal/mol"),
    "kJ": (R_KJ, "kJ/(mol·K)", "kJ/mol"),
}


@dataclass
class ArrheniusResult:
    temps_c: np.ndarray
    rates: np.ndarray  # k per temperature
    inv_T: np.ndarray  # 1/T in Kelvin
    ln_k: np.ndarray
    ea: Optional[float]  # activation energy in ea_unit
    ea_unit: str
    ea_kcal_mol: Optional[float]
    ea_kj_mol: Optional[float]
    ln_A: Optional[float]
    fit_slope: Optional[float]  # d(ln k)/d(1/T) = −Ea/R
    r_gas: float
    r_unit_key: str
    r_label: str
    r_squared: Optional[float]
    predict_temp_c: Optional[float]
    predicted_k: Optional[float]
    predicted_rate_at_25: Optional[float]  # convenience alias when T=25
    message: str
    valid: bool

    @property
    def ea_j_mol(self) -> Optional[float]:
        """Legacy accessor: Ea in J/mol."""
        if self.ea_kj_mol is None:
            return None
        return self.ea_kj_mol * 1000.0


@dataclass
class ShelfLifeProjection:
    """Exploratory zero-order mean-crossing shelf life from predicted k."""

    shelf_life: Optional[float]
    intercept: float
    spec_limit: float
    direction: str
    rate_k: float
    temp_c: float
    message: str
    valid: bool


def parse_temperature_c(condition: str) -> Optional[float]:
    """Extract Celsius from strings like '25C/60%RH', '40°C', '30C'."""
    import re

    s = str(condition).replace("°", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[Cc]", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^(-?\d+(?:\.\d+)?)$", s.strip())
    if m:
        return float(m.group(1))
    return None


def resolve_r(r_unit: str) -> Tuple[float, str, str]:
    """Return (R_value, R_label, Ea_unit) for 'kcal' or 'kJ'."""
    key = r_unit.strip()
    if key in ("kj", "KJ", "kJ/mol", "kJ/(mol·K)"):
        key = "kJ"
    elif key in ("kcal", "Kcal", "kcal/mol", "kcal/(mol·K)"):
        key = "kcal"
    if key not in R_OPTIONS:
        raise ValueError(f"Unsupported R unit '{r_unit}'. Use 'kcal' or 'kJ'.")
    return R_OPTIONS[key][0], R_OPTIONS[key][1], R_OPTIONS[key][2]


def normalize_rates_table(rates_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to columns temp_c, rate_abs_slope (k).

    Accepts aliases:
      temperature / temp / temp_c / T_C
      k / rate / rate_abs_slope / rate_k
    """
    if rates_df is None or len(rates_df) == 0:
        return pd.DataFrame(columns=["temp_c", "rate_abs_slope"])

    d = rates_df.copy()
    colmap = {c.lower().strip(): c for c in d.columns}

    temp_aliases = ("temp_c", "temp", "temperature", "t_c", "t", "celsius")
    rate_aliases = ("rate_abs_slope", "k", "rate", "rate_k", "abs_slope", "slope_abs")

    temp_col = next((colmap[a] for a in temp_aliases if a in colmap), None)
    rate_col = next((colmap[a] for a in rate_aliases if a in colmap), None)

    if temp_col is None or rate_col is None:
        raise ValueError(
            "Bảng tốc độ cần cột nhiệt độ (temp_c) và k (rate). "
            "Rates table needs temperature (temp_c) and k (rate) columns."
        )

    out = pd.DataFrame(
        {
            "temp_c": pd.to_numeric(d[temp_col], errors="coerce"),
            "rate_abs_slope": pd.to_numeric(d[rate_col], errors="coerce"),
        }
    )
    if "condition" in d.columns:
        out["condition"] = d["condition"].values
    elif "label" in d.columns:
        out["condition"] = d["label"].values
    return out


def estimate_rates_by_condition(
    df: pd.DataFrame,
    transform: str = "none",
) -> pd.DataFrame:
    """Fit |slope| per storage condition; attach parsed temperature.

    Assumption: k ≈ |OLS slope| of response vs time (zero-order rate proxy).
    """
    rows = []
    for cond, sub in df.groupby("condition"):
        temp = parse_temperature_c(cond)
        if len(sub) < 3:
            continue
        try:
            reg = fit_ols(sub["time"].values, sub["response"].values, transform=transform)
            rate = abs(reg.beta1)
            rows.append(
                {
                    "condition": cond,
                    "temp_c": temp,
                    "slope": reg.beta1,
                    "rate_abs_slope": rate,
                    "intercept": reg.beta0,
                    "n": reg.n,
                    "r_squared": reg.r_squared,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def predict_k(
    ln_A: float,
    fit_slope: float,
    temp_c: float,
) -> float:
    """Predict k at temp_c (°C) from fitted ln(k) = ln_A + fit_slope / T(K)."""
    inv_T = 1.0 / (float(temp_c) + 273.15)
    return float(np.exp(ln_A + fit_slope * inv_T))


def project_shelf_life_zero_order(
    rate_k: float,
    intercept: float,
    spec_limit: float,
    direction: str = "decreasing",
    temp_c: float = 25.0,
) -> ShelfLifeProjection:
    """Exploratory shelf life via zero-order mean crossing (NOT ICH Q1E CI).

    decreasing: y = intercept − k·t  →  t* = (intercept − spec) / k
    increasing: y = intercept + k·t  →  t* = (spec − intercept) / k

    rate_k must be positive (|slope|).
    """
    disclaimer = (
        "SUPPORTIVE ONLY — zero-order mean crossing from extrapolated k; "
        "NOT ICH Q1E one-sided CI shelf life. "
        "CHỈ HỖ TRỢ — giao trung bình bậc không từ k ngoại suy; "
        "không phải shelf life CI một phía ICH Q1E."
    )
    if rate_k is None or not np.isfinite(rate_k) or rate_k <= 0:
        return ShelfLifeProjection(
            shelf_life=None,
            intercept=intercept,
            spec_limit=spec_limit,
            direction=direction,
            rate_k=float(rate_k) if rate_k is not None else float("nan"),
            temp_c=temp_c,
            message="k phải > 0 / rate k must be positive. " + disclaimer,
            valid=False,
        )

    direction = (direction or "decreasing").lower()
    if direction.startswith("dec"):
        delta = intercept - spec_limit
        direction = "decreasing"
    else:
        delta = spec_limit - intercept
        direction = "increasing"

    if delta <= 0:
        return ShelfLifeProjection(
            shelf_life=0.0,
            intercept=intercept,
            spec_limit=spec_limit,
            direction=direction,
            rate_k=float(rate_k),
            temp_c=temp_c,
            message=(
                "Đã ngoài / tại spec tại t=0 (theo trung bình). "
                "Already at/beyond spec at t=0 (mean). "
                + disclaimer
            ),
            valid=True,
        )

    t_star = float(delta / rate_k)
    return ShelfLifeProjection(
        shelf_life=t_star,
        intercept=intercept,
        spec_limit=spec_limit,
        direction=direction,
        rate_k=float(rate_k),
        temp_c=temp_c,
        message=disclaimer,
        valid=True,
    )


def fit_arrhenius(
    rates_df: pd.DataFrame,
    *,
    r_unit: str = "kcal",
    predict_temp_c: float = 25.0,
) -> ArrheniusResult:
    """Linear regression of ln(k) vs 1/T(K).

    Requires at least 2 temperatures with positive rates.
    r_unit: 'kcal' or 'kJ' selects gas constant and Ea reporting unit.
    """
    r_val, r_label, ea_unit = resolve_r(r_unit)
    r_key = "kJ" if ea_unit.startswith("kJ") else "kcal"

    empty = ArrheniusResult(
        temps_c=np.array([]),
        rates=np.array([]),
        inv_T=np.array([]),
        ln_k=np.array([]),
        ea=None,
        ea_unit=ea_unit,
        ea_kcal_mol=None,
        ea_kj_mol=None,
        ln_A=None,
        fit_slope=None,
        r_gas=r_val,
        r_unit_key=r_key,
        r_label=r_label,
        r_squared=None,
        predict_temp_c=float(predict_temp_c),
        predicted_k=None,
        predicted_rate_at_25=None,
        message="",
        valid=False,
    )

    try:
        d = normalize_rates_table(rates_df)
    except ValueError as exc:
        empty.message = str(exc)
        return empty

    d = d.dropna(subset=["temp_c", "rate_abs_slope"]).copy()
    d = d[d["rate_abs_slope"] > 0]
    if len(d) < 2:
        empty.message = (
            "Cần ≥2 điều kiện nhiệt độ với k>0. "
            "Need ≥2 temperature conditions with positive k."
        )
        return empty

    T_k = d["temp_c"].values.astype(float) + 273.15
    inv_T = 1.0 / T_k
    ln_k = np.log(d["rate_abs_slope"].values.astype(float))
    X = sm.add_constant(inv_T)
    model = sm.OLS(ln_k, X).fit()
    ln_A = float(model.params[0])
    fit_slope = float(model.params[1])  # = −Ea/R
    ea = -fit_slope * r_val

    if r_key == "kcal":
        ea_kcal = ea
        ea_kj = ea * 4.184
    else:
        ea_kj = ea
        ea_kcal = ea / 4.184

    pred_T = float(predict_temp_c)
    k_pred = predict_k(ln_A, fit_slope, pred_T)
    k25 = predict_k(ln_A, fit_slope, 25.0)

    return ArrheniusResult(
        temps_c=d["temp_c"].values.astype(float),
        rates=d["rate_abs_slope"].values.astype(float),
        inv_T=inv_T,
        ln_k=ln_k,
        ea=float(ea),
        ea_unit=ea_unit,
        ea_kcal_mol=float(ea_kcal),
        ea_kj_mol=float(ea_kj),
        ln_A=ln_A,
        fit_slope=fit_slope,
        r_gas=r_val,
        r_unit_key=r_key,
        r_label=r_label,
        r_squared=float(model.rsquared),
        predict_temp_c=pred_T,
        predicted_k=k_pred,
        predicted_rate_at_25=k25,
        message=(
            "Arrhenius exploratory fit OK. "
            "CHỈ MANG TÍNH HỖ TRỢ / thăm dò — không thay thế đánh giá dài hạn ICH Q1A/Q1E. "
            "SUPPORTIVE / exploratory ONLY — not a substitute for long-term ICH evaluation."
        ),
        valid=True,
    )


def default_rates_editor_frame() -> pd.DataFrame:
    """Empty-ish template for Streamlit data_editor."""
    return pd.DataFrame(
        {
            "temp_c": [25.0, 40.0, 50.0],
            "k": [0.00460543, 0.01973660, 0.04830710],
            "label": ["long-term", "accelerated", "stress"],
        }
    )
