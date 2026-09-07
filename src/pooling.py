"""Multi-batch poolability (ANCOVA) per ICH Q1E spirit.

ICH Q1E suggests testing equality of slopes and intercepts across batches
before pooling. A relatively large significance level (commonly 0.25) is
used for poolability tests to avoid pooling when batches differ.

Default alpha_pool = 0.25 (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .regression import (
    RegressionResult,
    ShelfLifeResult,
    estimate_shelf_life,
    fit_ols,
    infer_direction,
)


@dataclass
class PoolabilityResult:
    n_batches: int
    batches: List[str]
    alpha_pool: float
    slope_equal: bool
    intercept_equal: bool
    pool_slopes: bool
    pool_all: bool  # common slope AND intercept
    recommendation: str  # "pooled" | "common_slope" | "separate"
    slope_test: Dict
    intercept_test: Dict
    message: str
    anova_full: Optional[pd.DataFrame] = None


@dataclass
class MultiBatchResult:
    poolability: PoolabilityResult
    pooled_reg: Optional[RegressionResult]
    pooled_shelf: Optional[ShelfLifeResult]
    per_batch: Dict[str, Tuple[RegressionResult, ShelfLifeResult]]
    reported_shelf_life: Optional[float]
    reported_mode: str  # "pooled" | "minimum_of_batches"
    message: str


def _batch_dummies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    batches = sorted(df["batch"].astype(str).unique().tolist())
    dummies = pd.get_dummies(df["batch"].astype(str), prefix="batch", drop_first=True)
    # Ensure numeric float
    dummies = dummies.astype(float)
    return dummies, batches


def assess_poolability(
    df: pd.DataFrame,
    transform: str = "none",
    alpha_pool: float = 0.25,
) -> PoolabilityResult:
    """
    ANCOVA-style tests:

    1. Equality of slopes: compare model with batch*time interaction vs
       model with common slope (batch intercepts only).
    2. Equality of intercepts (given common slope): compare common-slope
       model vs fully pooled (no batch terms).

    If slopes differ (p < alpha_pool) -> separate analyses.
    If slopes equal but intercepts differ -> common slope / separate intercepts
      (for shelf life ICH often still uses separate or most conservative).
    If both equal -> fully pooled.
    """
    from .regression import apply_transform

    data = df.copy()
    data["batch"] = data["batch"].astype(str)
    data["y"] = apply_transform(data["response"].values, transform)
    data["t"] = data["time"].astype(float)
    batches = sorted(data["batch"].unique().tolist())
    n_batches = len(batches)

    if n_batches < 2:
        return PoolabilityResult(
            n_batches=n_batches,
            batches=batches,
            alpha_pool=alpha_pool,
            slope_equal=True,
            intercept_equal=True,
            pool_slopes=True,
            pool_all=True,
            recommendation="pooled",
            slope_test={"p": np.nan, "F": np.nan, "df1": 0, "df2": 0},
            intercept_test={"p": np.nan, "F": np.nan, "df1": 0, "df2": 0},
            message="Chỉ có 1 lô — dùng mô hình đơn lô / Single batch — use single-batch model.",
        )

    # Full interaction model: y ~ t * C(batch)
    try:
        m_full = smf.ols("y ~ t * C(batch)", data=data).fit()
        m_common_slope = smf.ols("y ~ t + C(batch)", data=data).fit()
        m_pooled = smf.ols("y ~ t", data=data).fit()
    except Exception as exc:
        return PoolabilityResult(
            n_batches=n_batches,
            batches=batches,
            alpha_pool=alpha_pool,
            slope_equal=False,
            intercept_equal=False,
            pool_slopes=False,
            pool_all=False,
            recommendation="separate",
            slope_test={},
            intercept_test={},
            message=f"ANCOVA fit failed: {exc}",
        )

    # Partial F: slopes — full vs common slope
    anova_slope = sm.stats.anova_lm(m_common_slope, m_full)
    # anova_lm comparison rows: index 0 = restricted, 1 = full
    try:
        f_slope = float(anova_slope.loc[1, "F"])
        p_slope = float(anova_slope.loc[1, "Pr(>F)"])
        df1_s = float(anova_slope.loc[1, "df_diff"]) if "df_diff" in anova_slope.columns else float(
            m_full.df_model - m_common_slope.df_model
        )
        df2_s = float(m_full.df_resid)
    except Exception:
        # Manual SSR comparison
        sse_r = float(np.sum(m_common_slope.resid ** 2))
        sse_f = float(np.sum(m_full.resid ** 2))
        df1_s = float(m_full.df_model - m_common_slope.df_model)
        df2_s = float(m_full.df_resid)
        if df1_s > 0 and df2_s > 0 and sse_f > 0:
            f_slope = ((sse_r - sse_f) / df1_s) / (sse_f / df2_s)
            p_slope = float(1 - stats.f.cdf(f_slope, df1_s, df2_s))
        else:
            f_slope, p_slope = np.nan, np.nan

    slope_equal = bool(p_slope >= alpha_pool) if np.isfinite(p_slope) else False

    # Intercepts given common slope: common slope vs pooled
    anova_int = sm.stats.anova_lm(m_pooled, m_common_slope)
    try:
        f_int = float(anova_int.loc[1, "F"])
        p_int = float(anova_int.loc[1, "Pr(>F)"])
        df1_i = float(anova_int.loc[1, "df_diff"]) if "df_diff" in anova_int.columns else float(
            m_common_slope.df_model - m_pooled.df_model
        )
        df2_i = float(m_common_slope.df_resid)
    except Exception:
        sse_r = float(np.sum(m_pooled.resid ** 2))
        sse_f = float(np.sum(m_common_slope.resid ** 2))
        df1_i = float(m_common_slope.df_model - m_pooled.df_model)
        df2_i = float(m_common_slope.df_resid)
        if df1_i > 0 and df2_i > 0 and sse_f > 0:
            f_int = ((sse_r - sse_f) / df1_i) / (sse_f / df2_i)
            p_int = float(1 - stats.f.cdf(f_int, df1_i, df2_i))
        else:
            f_int, p_int = np.nan, np.nan

    intercept_equal = bool(p_int >= alpha_pool) if np.isfinite(p_int) else False

    pool_slopes = slope_equal
    pool_all = slope_equal and intercept_equal

    if pool_all:
        recommendation = "pooled"
        msg = (
            f"Các lô có thể gộp (slope p={p_slope:.4f}, intercept p={p_int:.4f} ≥ α={alpha_pool}). "
            f"Batches poolable (slopes & intercepts)."
        )
    elif pool_slopes:
        recommendation = "common_slope"
        msg = (
            f"Đồng nhất slope (p={p_slope:.4f}) nhưng khác intercept (p={p_int:.4f}). "
            f"Common slope, different intercepts — dùng phân tích theo lô / per-batch "
            f"(conservative min shelf life)."
        )
    else:
        recommendation = "separate"
        msg = (
            f"Slope khác nhau giữa các lô (p={p_slope:.4f} < α={alpha_pool}). "
            f"Separate batch models; báo cáo shelf life nhỏ nhất / report minimum."
        )

    anova_tbl = pd.DataFrame(
        {
            "Test": ["Equality of slopes", "Equality of intercepts | common slope"],
            "F": [f_slope, f_int],
            "df1": [df1_s, df1_i],
            "df2": [df2_s, df2_i],
            "p-value": [p_slope, p_int],
            f"Poolable (α={alpha_pool})": [slope_equal, intercept_equal],
        }
    )

    return PoolabilityResult(
        n_batches=n_batches,
        batches=batches,
        alpha_pool=alpha_pool,
        slope_equal=slope_equal,
        intercept_equal=intercept_equal,
        pool_slopes=pool_slopes,
        pool_all=pool_all,
        recommendation=recommendation,
        slope_test={"p": p_slope, "F": f_slope, "df1": df1_s, "df2": df2_s},
        intercept_test={"p": p_int, "F": f_int, "df1": df1_i, "df2": df2_i},
        message=msg,
        anova_full=anova_tbl,
    )


def analyze_multibatch(
    df: pd.DataFrame,
    spec_limit: float,
    direction: Optional[str] = None,
    transform: str = "none",
    alpha: float = 0.05,
    alpha_pool: float = 0.25,
    t_max: float = 120.0,
    force_pooled: Optional[bool] = None,
) -> MultiBatchResult:
    """
    Full multi-batch workflow:
      - test poolability
      - if poolable (or force_pooled): fit pooled model + shelf life
      - always fit per-batch
      - reported shelf life = pooled if pool_all else min of batch shelf lives
    """
    pool = assess_poolability(df, transform=transform, alpha_pool=alpha_pool)
    per_batch: Dict[str, Tuple[RegressionResult, ShelfLifeResult]] = {}

    for b in pool.batches:
        sub = df[df["batch"].astype(str) == b]
        if len(sub) < 3:
            continue
        reg = fit_ols(sub["time"].values, sub["response"].values, transform=transform)
        dir_b = infer_direction(reg.beta1, direction)
        sl = estimate_shelf_life(
            reg, spec_limit=spec_limit, direction=dir_b, alpha=alpha, t_max=t_max
        )
        per_batch[b] = (reg, sl)

    use_pooled = pool.pool_all if force_pooled is None else force_pooled

    pooled_reg = None
    pooled_shelf = None
    if use_pooled or pool.pool_slopes:
        # For fully pooled, use all data as one series
        pooled_reg = fit_ols(df["time"].values, df["response"].values, transform=transform)
        dir_p = infer_direction(pooled_reg.beta1, direction)
        pooled_shelf = estimate_shelf_life(
            pooled_reg, spec_limit=spec_limit, direction=dir_p, alpha=alpha, t_max=t_max
        )

    if use_pooled and pooled_shelf is not None and pooled_shelf.shelf_life is not None:
        reported = pooled_shelf.shelf_life
        mode = "pooled"
        msg = pool.message + f" Shelf life (pooled) = {reported:.2f} months."
    else:
        lives = [
            sl.shelf_life
            for _, sl in per_batch.values()
            if sl.shelf_life is not None
        ]
        if lives:
            reported = float(min(lives))
            mode = "minimum_of_batches"
            msg = (
                pool.message
                + f" Shelf life báo cáo (min các lô) / reported (minimum) = {reported:.2f} months."
            )
        else:
            reported = None
            mode = "minimum_of_batches"
            msg = pool.message + " Không ước tính được shelf life / Unable to estimate shelf life."

    # If force separate even when poolable
    if force_pooled is False:
        lives = [
            sl.shelf_life
            for _, sl in per_batch.values()
            if sl.shelf_life is not None
        ]
        reported = float(min(lives)) if lives else None
        mode = "minimum_of_batches"
        msg = "Người dùng chọn phân tích riêng lô / User forced separate batches. " + (
            f"Min shelf life = {reported:.2f} months." if reported is not None else ""
        )

    return MultiBatchResult(
        poolability=pool,
        pooled_reg=pooled_reg if use_pooled else (pooled_reg if pool.pool_all else pooled_reg),
        pooled_shelf=pooled_shelf if use_pooled else pooled_shelf,
        per_batch=per_batch,
        reported_shelf_life=reported,
        reported_mode=mode,
        message=msg,
    )
