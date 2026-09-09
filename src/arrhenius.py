"""Optional Arrhenius / kinetics exploratory analysis (SUPPORTIVE ONLY).

NOT a substitute for real-time ICH Q1A / Q1E long-term stability evaluation.
Clearly labeled as exploratory / supportive in the UI and README.

Theory reference (Chow, Statistical Design and Analysis of Stability Studies,
Ch. 2; standard pharmaceutical kinetics):
  - Zero-order:  Y(t) = Y0 − k0·t
  - First-order: ln Y(t) = ln Y0 − k1·t
  - Second-order: 1/Y(t) = 1/Y0 + k2·t   (rarely used for drug assay)
  - Arrhenius:   ln(k) = ln(A) − Ea/(R·T)
  - Extended T+RH MVP: ln(k) = ln(A) − Ea/(R·T) + B·(RH/100)
  - Optional light term: + C·light_exposure (user units, e.g. lux·h / 1e6)

Assumptions (documented for the user)
-------------------------------------
  1. Rate proxy from stability data uses OLS slope on the order-appropriate
     scale (response / ln(response) / 1/response).
  2. Humidity / light terms are exploratory MVP — not a full Q1B / moisture
     sorption model.
  3. Predicted shelf life from extrapolated k uses mean crossing — NOT the
     ICH Q1E one-sided CI of the mean.
  4. Results are SUPPORTIVE / exploratory only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .regression import fit_ols

# Gas constants
R_KCAL = 0.0019872042586067  # kcal/(mol·K)
R_KJ = 0.008314462618  # kJ/(mol·K)

R_OPTIONS = {
    "kcal": (R_KCAL, "kcal/(mol·K)", "kcal/mol"),
    "kJ": (R_KJ, "kJ/(mol·K)", "kJ/mol"),
}

ORDER_ALIASES = {
    "zero": "zero",
    "0": "zero",
    "zero-order": "zero",
    "bậc không": "zero",
    "first": "first",
    "1": "first",
    "first-order": "first",
    "bậc một": "first",
    "second": "second",
    "2": "second",
    "second-order": "second",
    "bậc hai": "second",
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
    fit_slope: Optional[float]  # d(ln k)/d(1/T) = −Ea/R  (classic model)
    r_gas: float
    r_unit_key: str
    r_label: str
    r_squared: Optional[float]
    predict_temp_c: Optional[float]
    predicted_k: Optional[float]
    predicted_rate_at_25: Optional[float]
    message: str
    valid: bool
    # --- upgraded fields (optional / CI) ---
    kinetics_order: str = "zero"
    model_kind: str = "classic"  # classic | t_rh | t_rh_light
    n: int = 0
    df_resid: int = 0
    ea_se: Optional[float] = None
    ea_ci_low: Optional[float] = None
    ea_ci_high: Optional[float] = None
    ln_A_se: Optional[float] = None
    predicted_k_ci_low: Optional[float] = None
    predicted_k_ci_high: Optional[float] = None
    rh_coef_B: Optional[float] = None  # coefficient on RH/100
    light_coef_C: Optional[float] = None
    predict_rh: Optional[float] = None
    predict_light: Optional[float] = None
    rhs: Optional[np.ndarray] = None
    lights: Optional[np.ndarray] = None
    alpha_ci: float = 0.05
    assumptions: str = ""

    @property
    def ea_j_mol(self) -> Optional[float]:
        """Legacy accessor: Ea in J/mol."""
        if self.ea_kj_mol is None:
            return None
        return self.ea_kj_mol * 1000.0


@dataclass
class ShelfLifeProjection:
    """Exploratory mean-crossing shelf life from predicted k."""

    shelf_life: Optional[float]
    intercept: float
    spec_limit: float
    direction: str
    rate_k: float
    temp_c: float
    message: str
    valid: bool
    kinetics_order: str = "zero"
    shelf_life_ci_low: Optional[float] = None
    shelf_life_ci_high: Optional[float] = None


def normalize_order(order: str) -> str:
    key = (order or "zero").strip().lower()
    if key not in ORDER_ALIASES:
        raise ValueError(
            f"Unsupported kinetics order '{order}'. Use zero / first / second."
        )
    return ORDER_ALIASES[key]


def parse_temperature_c(condition: str) -> Optional[float]:
    """Extract Celsius from strings like '25C/60%RH', '40°C', '30C'."""
    s = str(condition).replace("°", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[Cc]", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^(-?\d+(?:\.\d+)?)$", s.strip())
    if m:
        return float(m.group(1))
    return None


def parse_rh_percent(condition: str) -> Optional[float]:
    """Extract %RH from strings like '25C/60%RH', '40°C/75% RH'."""
    s = str(condition)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*RH", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"/\s*(\d+(?:\.\d+)?)\s*%", s)
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
    """Normalize to columns temp_c, rate_abs_slope (k), optional rh_percent, light.

    Accepts aliases:
      temperature / temp / temp_c / T_C
      k / rate / rate_abs_slope / rate_k
      rh / rh_percent / humidity / %RH
      light / light_exposure / lux_hours
    """
    if rates_df is None or len(rates_df) == 0:
        return pd.DataFrame(columns=["temp_c", "rate_abs_slope"])

    d = rates_df.copy()
    colmap = {c.lower().strip(): c for c in d.columns}

    temp_aliases = ("temp_c", "temp", "temperature", "t_c", "t", "celsius")
    rate_aliases = ("rate_abs_slope", "k", "rate", "rate_k", "abs_slope", "slope_abs")
    rh_aliases = ("rh_percent", "rh", "humidity", "%rh", "rh_%", "relative_humidity")
    light_aliases = ("light", "light_exposure", "lux_hours", "lux_h", "light_dose")

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
    rh_col = next((colmap[a] for a in rh_aliases if a in colmap), None)
    if rh_col is not None:
        out["rh_percent"] = pd.to_numeric(d[rh_col], errors="coerce")
    light_col = next((colmap[a] for a in light_aliases if a in colmap), None)
    if light_col is not None:
        out["light"] = pd.to_numeric(d[light_col], errors="coerce")

    if "condition" in d.columns:
        out["condition"] = d["condition"].values
        if "rh_percent" not in out.columns:
            out["rh_percent"] = [parse_rh_percent(c) for c in d["condition"]]
    elif "label" in d.columns:
        out["condition"] = d["label"].values
        if "rh_percent" not in out.columns:
            out["rh_percent"] = [parse_rh_percent(c) for c in d["label"]]
    return out


def _transform_for_order(y: np.ndarray, order: str) -> np.ndarray:
    order = normalize_order(order)
    y = np.asarray(y, dtype=float)
    if order == "zero":
        return y
    if order == "first":
        if np.any(y <= 0):
            raise ValueError("First-order requires response > 0 (ln scale).")
        return np.log(y)
    # second
    if np.any(y == 0):
        raise ValueError("Second-order requires response ≠ 0 (1/Y scale).")
    return 1.0 / y


def estimate_rates_by_condition(
    df: pd.DataFrame,
    transform: str = "none",
    kinetics_order: str = "zero",
) -> pd.DataFrame:
    """Fit |slope| per storage condition on order-appropriate scale.

    kinetics_order:
      zero  — |OLS slope| of response vs time
      first — |OLS slope| of ln(response) vs time
      second — |OLS slope| of 1/response vs time

    Legacy `transform` still applied via fit_ols when order=zero and transform≠none.
    """
    order = normalize_order(kinetics_order)
    rows = []
    for cond, sub in df.groupby("condition"):
        temp = parse_temperature_c(cond)
        rh = parse_rh_percent(cond)
        if len(sub) < 3:
            continue
        try:
            y_raw = sub["response"].values.astype(float)
            t = sub["time"].values.astype(float)
            if order == "zero" and transform and transform != "none":
                reg = fit_ols(t, y_raw, transform=transform)
                slope = reg.beta1
                intercept = reg.beta0
                n = reg.n
                r2 = reg.r_squared
            else:
                y = _transform_for_order(y_raw, order)
                X = sm.add_constant(t)
                model = sm.OLS(y, X).fit()
                intercept = float(model.params[0])
                slope = float(model.params[1])
                n = int(model.nobs)
                r2 = float(model.rsquared)
            rate = abs(slope)
            rows.append(
                {
                    "condition": cond,
                    "temp_c": temp,
                    "rh_percent": rh,
                    "slope": slope,
                    "rate_abs_slope": rate,
                    "intercept": intercept,
                    "n": n,
                    "r_squared": r2,
                    "kinetics_order": order,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def predict_k(
    ln_A: float,
    fit_slope: float,
    temp_c: float,
    *,
    rh_coef_B: float = 0.0,
    rh_percent: float = 0.0,
    light_coef_C: float = 0.0,
    light: float = 0.0,
) -> float:
    """Predict k at temp_c (°C) from fitted ln(k) model."""
    inv_T = 1.0 / (float(temp_c) + 273.15)
    ln_k = ln_A + fit_slope * inv_T + rh_coef_B * (float(rh_percent) / 100.0) + light_coef_C * float(light)
    return float(np.exp(ln_k))


def project_shelf_life_zero_order(
    rate_k: float,
    intercept: float,
    spec_limit: float,
    direction: str = "decreasing",
    temp_c: float = 25.0,
) -> ShelfLifeProjection:
    """Exploratory shelf life via zero-order mean crossing (NOT ICH Q1E CI)."""
    return project_shelf_life(
        rate_k=rate_k,
        intercept=intercept,
        spec_limit=spec_limit,
        direction=direction,
        temp_c=temp_c,
        kinetics_order="zero",
    )


def project_shelf_life(
    rate_k: float,
    intercept: float,
    spec_limit: float,
    direction: str = "decreasing",
    temp_c: float = 25.0,
    kinetics_order: str = "zero",
    rate_k_low: Optional[float] = None,
    rate_k_high: Optional[float] = None,
) -> ShelfLifeProjection:
    """Exploratory mean-crossing shelf life for zero / first / second order.

    intercept is on the *modeling* scale:
      zero: Y0
      first: ln(Y0)   — pass ln of initial assay if known, or use Y0 with order=first
                       via intercept_is_raw= handled below by accepting raw Y0 always
                       and transforming internally for clarity.
    We accept raw Y0 (positive) for all orders and convert internally.
    """
    order = normalize_order(kinetics_order)
    disclaimer = (
        f"SUPPORTIVE ONLY — {order}-order mean crossing from extrapolated k; "
        "NOT ICH Q1E one-sided CI shelf life. "
        f"CHỈ HỖ TRỢ — giao trung bình bậc {order} từ k ngoại suy; "
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
            kinetics_order=order,
        )

    direction = (direction or "decreasing").lower()
    if direction.startswith("dec"):
        direction = "decreasing"
    else:
        direction = "increasing"

    y0 = float(intercept)
    spec = float(spec_limit)
    k = float(rate_k)

    def _t_star(k_use: float) -> Optional[float]:
        if k_use <= 0 or not np.isfinite(k_use):
            return None
        if order == "zero":
            if direction == "decreasing":
                delta = y0 - spec
            else:
                delta = spec - y0
            if delta <= 0:
                return 0.0
            return float(delta / k_use)
        if order == "first":
            if y0 <= 0 or spec <= 0:
                return None
            if direction == "decreasing":
                # ln Y = ln Y0 − k t  →  t = ln(Y0/spec)/k
                if y0 <= spec:
                    return 0.0
                return float(np.log(y0 / spec) / k_use)
            # increasing impurity rarely first-order on impurity itself; keep formula
            if spec <= y0:
                return 0.0
            return float(np.log(spec / y0) / k_use)
        # second-order: 1/Y = 1/Y0 + k t  (decreasing potency)
        if y0 == 0 or spec == 0:
            return None
        if direction == "decreasing":
            # Y decreases → 1/Y increases
            if y0 <= spec:
                return 0.0
            return float((1.0 / spec - 1.0 / y0) / k_use)
        if spec <= y0:
            return 0.0
        return float((1.0 / y0 - 1.0 / spec) / k_use)

    t_star = _t_star(k)
    if t_star is None:
        return ShelfLifeProjection(
            shelf_life=None,
            intercept=y0,
            spec_limit=spec,
            direction=direction,
            rate_k=k,
            temp_c=temp_c,
            message="Không tính được shelf life (giá trị không hợp lệ). " + disclaimer,
            valid=False,
            kinetics_order=order,
        )

    t_lo = t_hi = None
    # Conservative band: higher k → shorter shelf life for decreasing
    if rate_k_low is not None and rate_k_high is not None:
        t_from_high = _t_star(float(rate_k_high))
        t_from_low = _t_star(float(rate_k_low))
        vals = [v for v in (t_from_high, t_from_low) if v is not None]
        if vals:
            t_lo = float(min(vals))
            t_hi = float(max(vals))

    return ShelfLifeProjection(
        shelf_life=float(t_star),
        intercept=y0,
        spec_limit=spec,
        direction=direction,
        rate_k=k,
        temp_c=temp_c,
        message=disclaimer,
        valid=True,
        kinetics_order=order,
        shelf_life_ci_low=t_lo,
        shelf_life_ci_high=t_hi,
    )


def fit_arrhenius(
    rates_df: pd.DataFrame,
    *,
    r_unit: str = "kcal",
    predict_temp_c: float = 25.0,
    include_rh: bool = False,
    include_light: bool = False,
    predict_rh: float = 60.0,
    predict_light: float = 0.0,
    alpha_ci: float = 0.05,
    kinetics_order: str = "zero",
) -> ArrheniusResult:
    """Linear regression of ln(k) vs 1/T(K) [, RH/100] [, light].

    Requires at least 2 temperatures with positive rates (classic), or more
    points than parameters for extended models.
    """
    order = normalize_order(kinetics_order)
    r_val, r_label, ea_unit = resolve_r(r_unit)
    r_key = "kJ" if ea_unit.startswith("kJ") else "kcal"

    assumptions = (
        "SUPPORTIVE / exploratory. Classic Arrhenius ln(k)=ln(A)−Ea/(R·T). "
        "Extended MVP may add B·(RH/100) and C·light. "
        f"Kinetics order for rate derivation context: {order}. "
        "Not a substitute for ICH Q1E long-term evaluation."
    )

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
        kinetics_order=order,
        alpha_ci=float(alpha_ci),
        assumptions=assumptions,
        predict_rh=float(predict_rh),
        predict_light=float(predict_light),
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

    use_rh = bool(include_rh)
    use_light = bool(include_light)
    rh_vals = None
    light_vals = None

    if use_rh:
        if "rh_percent" not in d.columns or d["rh_percent"].isna().all():
            empty.message = (
                "Đã chọn mô hình T+RH nhưng thiếu cột RH / parse được từ condition. "
                "RH required for T+RH model but missing."
            )
            return empty
        rh_vals = d["rh_percent"].fillna(0.0).values.astype(float)
    if use_light:
        if "light" not in d.columns or d["light"].isna().all():
            empty.message = (
                "Đã chọn hạng mục ánh sáng nhưng thiếu cột light. "
                "Light term selected but light column missing."
            )
            return empty
        light_vals = d["light"].fillna(0.0).values.astype(float)

    # Build design matrix
    cols = [np.ones(len(d)), inv_T]
    names = ["const", "inv_T"]
    if use_rh:
        cols.append(rh_vals / 100.0)
        names.append("rh_frac")
    if use_light:
        cols.append(light_vals)
        names.append("light")
    X = np.column_stack(cols)
    n_params = X.shape[1]
    if len(d) < n_params:
        empty.message = (
            f"Không đủ điểm ({len(d)}) cho mô hình {n_params} tham số. "
            f"Need at least {n_params} points."
        )
        return empty
    if len(d) == n_params and n_params > 2:
        empty.message = (
            f"Mô hình mở rộng cần >{n_params} điểm để ước lượng lỗi. "
            f"Extended model needs more than {n_params} points for residual df."
        )
        return empty

    model = sm.OLS(ln_k, X).fit()
    ln_A = float(model.params[0])
    fit_slope = float(model.params[1])  # = −Ea/R
    ea = -fit_slope * r_val
    B = float(model.params[2]) if use_rh else 0.0
    C = float(model.params[2 + int(use_rh)]) if use_light else 0.0

    if r_key == "kcal":
        ea_kcal = ea
        ea_kj = ea * 4.184
    else:
        ea_kj = ea
        ea_kcal = ea / 4.184

    # Confidence intervals via t critical
    df_resid = int(model.df_resid) if model.df_resid is not None else max(len(d) - n_params, 0)
    tcrit = float(stats.t.ppf(1.0 - alpha_ci / 2.0, df_resid)) if df_resid > 0 else 1.96
    se_params = np.asarray(model.bse, dtype=float)
    if df_resid <= 0:
        ea_se = None
        ea_ci_low = ea_ci_high = None
        ln_A_se = None
    else:
        ea_se = float(se_params[1] * r_val) if len(se_params) > 1 else None
        ea_ci_low = ea - tcrit * ea_se if ea_se is not None else None
        ea_ci_high = ea + tcrit * ea_se if ea_se is not None else None
        ln_A_se = float(se_params[0]) if len(se_params) else None

    def _pred_ln_k_ci(temp_c: float, rh: float, light: float) -> Tuple[float, float, float]:
        x = [1.0, 1.0 / (temp_c + 273.15)]
        if use_rh:
            x.append(rh / 100.0)
        if use_light:
            x.append(light)
        x = np.asarray(x, dtype=float)
        ln_mean = float(np.dot(model.params, x))
        if df_resid <= 0:
            return ln_mean, ln_mean, ln_mean
        try:
            cov = np.asarray(model.cov_params(), dtype=float)
            var = float(x @ cov @ x)
            se = np.sqrt(max(var, 0.0))
        except Exception:
            se = 0.0
        lo = ln_mean - tcrit * se
        hi = ln_mean + tcrit * se
        return ln_mean, lo, hi

    pred_T = float(predict_temp_c)
    rh_p = float(predict_rh) if use_rh else 0.0
    light_p = float(predict_light) if use_light else 0.0
    ln_mean, ln_lo, ln_hi = _pred_ln_k_ci(pred_T, rh_p, light_p)
    k_pred = float(np.exp(ln_mean))
    if df_resid <= 0:
        k_lo = k_hi = None
    else:
        k_lo = float(np.exp(ln_lo))
        k_hi = float(np.exp(ln_hi))
    ln25, _, _ = _pred_ln_k_ci(25.0, rh_p if use_rh else 0.0, light_p if use_light else 0.0)
    k25 = float(np.exp(ln25))

    if use_rh and use_light:
        model_kind = "t_rh_light"
    elif use_rh:
        model_kind = "t_rh"
    elif use_light:
        model_kind = "t_light"
    else:
        model_kind = "classic"

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
            f"Arrhenius exploratory fit OK ({model_kind}). "
            "CHỈ MANG TÍNH HỖ TRỢ / thăm dò — không thay thế đánh giá dài hạn ICH Q1A/Q1E. "
            "SUPPORTIVE / exploratory ONLY — not a substitute for long-term ICH evaluation."
        ),
        valid=True,
        kinetics_order=order,
        model_kind=model_kind,
        n=len(d),
        df_resid=df_resid,
        ea_se=ea_se,
        ea_ci_low=ea_ci_low,
        ea_ci_high=ea_ci_high,
        ln_A_se=ln_A_se,
        predicted_k_ci_low=k_lo,
        predicted_k_ci_high=k_hi,
        rh_coef_B=B if use_rh else None,
        light_coef_C=C if use_light else None,
        predict_rh=rh_p if use_rh else None,
        predict_light=light_p if use_light else None,
        rhs=rh_vals,
        lights=light_vals,
        alpha_ci=float(alpha_ci),
        assumptions=assumptions,
    )


def default_rates_editor_frame() -> pd.DataFrame:
    """Template for Streamlit data_editor (T, k, optional RH/light)."""
    return pd.DataFrame(
        {
            "temp_c": [25.0, 40.0, 50.0],
            "k": [0.00460543, 0.01973660, 0.04830710],
            "rh_percent": [60.0, 75.0, 75.0],
            "light": [0.0, 0.0, 0.0],
            "label": ["long-term", "accelerated", "stress"],
        }
    )
