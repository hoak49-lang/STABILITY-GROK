"""OLS regression, confidence bands, and shelf-life estimation (ICH Q1E).

Shelf life is the time at which the one-sided (1-alpha) confidence bound
for the mean response intersects the specification limit:
  - decreasing assay  -> lower confidence bound meets lower spec
  - increasing impurity -> upper confidence bound meets upper spec

References:
  ICH Q1E Evaluation of Stability Data (2003)
  ICH Q1A(R2) Stability Testing of New Drug Substances and Products
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import statsmodels.api as sm


@dataclass
class RegressionResult:
    """Single linear (or transformed) regression fit."""

    beta0: float
    beta1: float
    se_beta0: float
    se_beta1: float
    t_beta0: float
    t_beta1: float
    p_beta0: float
    p_beta1: float
    r_squared: float
    adj_r_squared: float
    n: int
    df_resid: int
    sigma: float  # residual SE
    sst: float
    sse: float
    ssr: float
    f_stat: float
    f_pvalue: float
    anova: pd.DataFrame
    residuals: np.ndarray
    fitted: np.ndarray
    time: np.ndarray
    response: np.ndarray
    response_raw: np.ndarray
    transform: str
    model: object = field(repr=False, default=None)
    x_mean: float = 0.0
    sxx: float = 0.0

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Coefficient": ["Intercept (β0)", "Slope (β1)"],
                "Estimate": [self.beta0, self.beta1],
                "SE": [self.se_beta0, self.se_beta1],
                "t": [self.t_beta0, self.t_beta1],
                "p-value": [self.p_beta0, self.p_beta1],
            }
        )


@dataclass
class ShelfLifeResult:
    shelf_life: Optional[float]
    spec_limit: float
    direction: str  # "decreasing" | "increasing"
    alpha: float
    conf_level: float
    intersect_ok: bool
    message: str
    times_grid: np.ndarray
    mean_pred: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    one_sided_bound: np.ndarray


def apply_transform(y: np.ndarray, transform: str) -> np.ndarray:
    transform = (transform or "none").lower()
    if transform in ("none", "identity", ""):
        return y.astype(float)
    if transform == "log":
        if np.any(y <= 0):
            raise ValueError("Log transform requires response > 0.")
        return np.log(y.astype(float))
    if transform == "sqrt":
        if np.any(y < 0):
            raise ValueError("Sqrt transform requires response >= 0.")
        return np.sqrt(y.astype(float))
    raise ValueError(f"Unknown transform: {transform}")


def invert_transform(y_t: np.ndarray, transform: str) -> np.ndarray:
    transform = (transform or "none").lower()
    if transform in ("none", "identity", ""):
        return y_t
    if transform == "log":
        return np.exp(y_t)
    if transform == "sqrt":
        return np.square(y_t)
    raise ValueError(f"Unknown transform: {transform}")


def fit_ols(
    time: np.ndarray,
    response: np.ndarray,
    transform: str = "none",
) -> RegressionResult:
    """Fit y = β0 + β1 * t via statsmodels OLS (on optionally transformed y)."""
    time = np.asarray(time, dtype=float)
    response_raw = np.asarray(response, dtype=float)
    y = apply_transform(response_raw, transform)
    X = sm.add_constant(time)
    model = sm.OLS(y, X).fit()

    beta0, beta1 = float(model.params[0]), float(model.params[1])
    se0, se1 = float(model.bse[0]), float(model.bse[1])
    t0, t1 = float(model.tvalues[0]), float(model.tvalues[1])
    p0, p1 = float(model.pvalues[0]), float(model.pvalues[1])

    fitted = model.fittedvalues
    resid = model.resid
    n = int(model.nobs)
    df_resid = int(model.df_resid)
    sigma = float(np.sqrt(model.scale))
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum(resid ** 2))
    ssr = sst - sse
    f_stat = float(model.fvalue) if model.fvalue is not None else np.nan
    f_pvalue = float(model.f_pvalue) if model.f_pvalue is not None else np.nan

    anova = pd.DataFrame(
        {
            "Source": ["Regression", "Residual", "Total"],
            "DF": [1, df_resid, n - 1],
            "SS": [ssr, sse, sst],
            "MS": [ssr / 1 if 1 else np.nan, sse / df_resid if df_resid else np.nan, np.nan],
            "F": [f_stat, np.nan, np.nan],
            "p-value": [f_pvalue, np.nan, np.nan],
        }
    )

    x_mean = float(time.mean())
    sxx = float(np.sum((time - x_mean) ** 2))

    return RegressionResult(
        beta0=beta0,
        beta1=beta1,
        se_beta0=se0,
        se_beta1=se1,
        t_beta0=t0,
        t_beta1=t1,
        p_beta0=p0,
        p_beta1=p1,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n=n,
        df_resid=df_resid,
        sigma=sigma,
        sst=sst,
        sse=sse,
        ssr=ssr,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        anova=anova,
        residuals=np.asarray(resid),
        fitted=np.asarray(fitted),
        time=time,
        response=y,
        response_raw=response_raw,
        transform=transform or "none",
        model=model,
        x_mean=x_mean,
        sxx=sxx,
    )


def mean_se_at_t(reg: RegressionResult, t: np.ndarray) -> np.ndarray:
    """Standard error of the fitted mean at time t."""
    t = np.asarray(t, dtype=float)
    if reg.sxx <= 0:
        return np.full_like(t, np.nan, dtype=float)
    return reg.sigma * np.sqrt(1.0 / reg.n + (t - reg.x_mean) ** 2 / reg.sxx)


def confidence_band(
    reg: RegressionResult,
    t: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mean, lower, upper) confidence band for the mean response.

    For one-sided shelf-life, use two_sided=False so the critical value is
    t_(1-alpha, df) rather than t_(1-alpha/2, df). Both bounds are still
    returned; the relevant one-sided bound is selected by direction.
    """
    t = np.asarray(t, dtype=float)
    mean = reg.beta0 + reg.beta1 * t
    se = mean_se_at_t(reg, t)
    df = max(reg.df_resid, 1)
    if two_sided:
        crit = float(stats.t.ppf(1 - alpha / 2, df))
    else:
        crit = float(stats.t.ppf(1 - alpha, df))
    lower = mean - crit * se
    upper = mean + crit * se
    return mean, lower, upper


def infer_direction(slope: float, direction: Optional[str] = None) -> str:
    if direction in ("decreasing", "increasing"):
        return direction
    return "decreasing" if slope <= 0 else "increasing"


def estimate_shelf_life(
    reg: RegressionResult,
    spec_limit: float,
    direction: Optional[str] = None,
    alpha: float = 0.05,
    t_max: float = 120.0,
    n_grid: int = 2000,
) -> ShelfLifeResult:
    """
    Solve for t* where the one-sided (1-alpha) confidence bound of the mean
    meets the specification limit (on the modeling scale).

    Spec limit is assumed to be on the same scale as the raw response;
    if a transform was used, the limit is transformed accordingly.
    """
    direction = infer_direction(reg.beta1, direction)
    conf_level = 1.0 - alpha

    # Transform spec to modeling scale
    try:
        spec_t = float(apply_transform(np.array([spec_limit]), reg.transform)[0])
    except Exception as exc:
        return ShelfLifeResult(
            shelf_life=None,
            spec_limit=spec_limit,
            direction=direction,
            alpha=alpha,
            conf_level=conf_level,
            intersect_ok=False,
            message=f"Không thể biến đổi giới hạn / Cannot transform spec: {exc}",
            times_grid=np.array([]),
            mean_pred=np.array([]),
            lower_bound=np.array([]),
            upper_bound=np.array([]),
            one_sided_bound=np.array([]),
        )

    t_obs_max = float(np.max(reg.time)) if len(reg.time) else 36.0
    t_hi = max(t_max, t_obs_max * 3, 36.0)
    times = np.linspace(0.0, t_hi, n_grid)
    mean, lower, upper = confidence_band(reg, times, alpha=alpha, two_sided=False)

    if direction == "decreasing":
        bound = lower
        # f(t) = bound(t) - spec; shelf life when bound drops to spec
        def f(tt: float) -> float:
            m, lo, _ = confidence_band(reg, np.array([tt]), alpha=alpha, two_sided=False)
            return float(lo[0] - spec_t)
    else:
        bound = upper

        def f(tt: float) -> float:
            m, _, up = confidence_band(reg, np.array([tt]), alpha=alpha, two_sided=False)
            return float(spec_t - up[0])

    # At t=0, for a valid study, f should be > 0 (still within spec with margin)
    f0 = f(0.0)
    f_end = f(t_hi)

    shelf = None
    msg = ""
    ok = False

    # Check if already out of spec at t=0
    if f0 <= 0:
        shelf = 0.0
        ok = True
        msg = (
            "Giới hạn tin cậy đã chạm/ vượt spec tại t=0 / "
            "Confidence bound already at/beyond spec at t=0."
        )
    elif f0 > 0 and f_end < 0:
        # Root in (0, t_hi)
        try:
            shelf = float(brentq(f, 0.0, t_hi, xtol=1e-6, maxiter=200))
            ok = True
            msg = (
                f"Thời hạn bảo quản ước tính / Estimated shelf life = {shelf:.2f} tháng/months "
                f"(one-sided {conf_level:.0%} CI, α={alpha})."
            )
        except ValueError as exc:
            msg = f"Không tìm được nghiệm / Root finding failed: {exc}"
    elif f0 > 0 and f_end >= 0:
        # Bound never reaches spec within search range
        # Could still try extending, but report as > t_hi
        shelf = None
        ok = False
        msg = (
            f"Không giao spec trong khoảng 0–{t_hi:.0f} tháng / "
            f"No intersection with spec within 0–{t_hi:.0f} months. "
            "Shelf life may exceed search range (or slope near zero)."
        )
    else:
        msg = "Không xác định được shelf life / Unable to determine shelf life."

    return ShelfLifeResult(
        shelf_life=shelf,
        spec_limit=spec_limit,
        direction=direction,
        alpha=alpha,
        conf_level=conf_level,
        intersect_ok=ok,
        message=msg,
        times_grid=times,
        mean_pred=mean,
        lower_bound=lower,
        upper_bound=upper,
        one_sided_bound=bound,
    )


def fit_and_shelflife(
    df: pd.DataFrame,
    spec_limit: float,
    direction: Optional[str] = None,
    transform: str = "none",
    alpha: float = 0.05,
    t_max: float = 120.0,
) -> Tuple[RegressionResult, ShelfLifeResult]:
    """Convenience: fit OLS on dataframe columns time/response and estimate shelf life."""
    reg = fit_ols(df["time"].values, df["response"].values, transform=transform)
    sl = estimate_shelf_life(
        reg, spec_limit=spec_limit, direction=direction, alpha=alpha, t_max=t_max
    )
    return reg, sl
