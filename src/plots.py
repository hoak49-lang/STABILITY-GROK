"""Matplotlib plots for stability / shelf-life analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .regression import RegressionResult, ShelfLifeResult, confidence_band, invert_transform


def _to_display(y_model: np.ndarray, transform: str) -> np.ndarray:
    return invert_transform(np.asarray(y_model, dtype=float), transform)


def plot_stability(
    df: pd.DataFrame,
    reg: RegressionResult,
    shelf: ShelfLifeResult,
    title: str = "Stability / Shelf-Life Plot",
    batch_col: bool = True,
    show_two_sided_band: bool = True,
    alpha_band: float = 0.05,
) -> plt.Figure:
    """Scatter + fitted line + one-sided CI bound + spec + shelf-life marker."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if batch_col and "batch" in df.columns and df["batch"].nunique() > 1:
        for b, sub in df.groupby("batch"):
            ax.scatter(sub["time"], sub["response"], label=f"Batch {b}", s=40, zorder=3)
    else:
        ax.scatter(df["time"], df["response"], label="Data / Dữ liệu", s=40, c="C0", zorder=3)

    # Grid for display (invert transform for plotting on raw scale)
    t_grid = shelf.times_grid if len(shelf.times_grid) else np.linspace(0, max(df["time"].max() * 1.5, 24), 200)
    mean_m, lo_m, up_m = confidence_band(reg, t_grid, alpha=alpha_band, two_sided=False)
    # Also two-sided for visual band if requested
    mean2, lo2, up2 = confidence_band(reg, t_grid, alpha=alpha_band, two_sided=True)

    mean_d = _to_display(mean_m, reg.transform)
    lo_d = _to_display(lo_m, reg.transform)
    up_d = _to_display(up_m, reg.transform)
    lo2_d = _to_display(lo2, reg.transform)
    up2_d = _to_display(up2, reg.transform)

    ax.plot(t_grid, mean_d, "k-", lw=1.8, label="Fitted mean / Trung bình khớp")
    if show_two_sided_band:
        ax.fill_between(
            t_grid, lo2_d, up2_d, color="gray", alpha=0.2, label=f"Two-sided {(1-alpha_band):.0%} CI (mean)"
        )

    # One-sided bound used for shelf life
    if shelf.direction == "decreasing":
        ax.plot(t_grid, lo_d, "b--", lw=1.5, label=f"One-sided lower {(1-shelf.alpha):.0%} bound")
    else:
        ax.plot(t_grid, up_d, "b--", lw=1.5, label=f"One-sided upper {(1-shelf.alpha):.0%} bound")

    ax.axhline(shelf.spec_limit, color="r", ls=":", lw=1.8, label=f"Spec limit = {shelf.spec_limit}")

    if shelf.shelf_life is not None:
        ax.axvline(
            shelf.shelf_life,
            color="green",
            ls="-.",
            lw=1.8,
            label=f"Shelf life = {shelf.shelf_life:.2f} mo",
        )
        ax.plot(shelf.shelf_life, shelf.spec_limit, "go", ms=8, zorder=4)

    ax.set_xlabel("Time (months) / Thời gian (tháng)")
    ax.set_ylabel("Response / Đáp ứng")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(reg: RegressionResult, title: str = "Residual diagnostics") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].scatter(reg.fitted, reg.residuals, s=35, c="C0")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xlabel("Fitted / Khớp")
    axes[0].set_ylabel("Residual / Phần dư")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].grid(True, alpha=0.3)

    # Simple QQ-like via sorted residuals vs normal scores
    from scipy import stats as sp_stats
    n = len(reg.residuals)
    osm = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sorted_res = np.sort(reg.residuals)
    axes[1].scatter(osm, sorted_res, s=35, c="C1")
    # reference line
    if n >= 2:
        slope, intercept = np.polyfit(osm, sorted_res, 1)
        xline = np.array([osm.min(), osm.max()])
        axes[1].plot(xline, intercept + slope * xline, "k-", lw=1)
    axes[1].set_xlabel("Theoretical quantile")
    axes[1].set_ylabel("Sample residual")
    axes[1].set_title("Normal QQ (residuals)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_multibatch_overlay(
    df: pd.DataFrame,
    per_batch: Dict,
    pooled_shelf: Optional[ShelfLifeResult] = None,
    spec_limit: Optional[float] = None,
    title: str = "Multi-batch overlay",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10.colors
    for i, (b, (reg, sl)) in enumerate(per_batch.items()):
        sub = df[df["batch"].astype(str) == b]
        c = colors[i % len(colors)]
        ax.scatter(sub["time"], sub["response"], label=f"Batch {b}", s=35, color=c)
        t_max = max(sub["time"].max() * 1.2, 24)
        t_grid = np.linspace(0, t_max, 100)
        mean = reg.beta0 + reg.beta1 * t_grid
        mean_d = _to_display(mean, reg.transform)
        ax.plot(t_grid, mean_d, color=c, lw=1.5)
        if sl.shelf_life is not None:
            ax.axvline(sl.shelf_life, color=c, ls="--", alpha=0.6)

    if spec_limit is not None:
        ax.axhline(spec_limit, color="r", ls=":", lw=1.8, label=f"Spec = {spec_limit}")

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Response")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_arrhenius(arr, rates_df: pd.DataFrame = None) -> plt.Figure:
    """Arrhenius plot: ln(k) vs 1000/T with OLS fit line (exploratory)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not arr.valid:
        ax.text(0.5, 0.5, arr.message, ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Arrhenius (exploratory) — insufficient data")
        return fig
    ax.scatter(arr.inv_T * 1000, arr.ln_k, s=60, c="C0", zorder=3, label="Quan sát / Observed")
    x = np.linspace(arr.inv_T.min(), arr.inv_T.max(), 80)
    slope = arr.fit_slope if arr.fit_slope is not None else (-arr.ea_j_mol / 8.314)
    y = arr.ln_A + slope * x
    ax.plot(x * 1000, y, "k-", lw=1.8, label="Fit ln(k)=ln(A)−Ea/(R·T)")
    # Mark predicted temperature if available
    if arr.predict_temp_c is not None and arr.predicted_k is not None:
        inv_p = 1000.0 / (arr.predict_temp_c + 273.15)
        ax.scatter(
            [inv_p],
            [np.log(arr.predicted_k)],
            s=80,
            c="C3",
            marker="D",
            zorder=4,
            label=f"Dự đoán @ {arr.predict_temp_c:g}°C",
        )
    ea_txt = f"{arr.ea:.2f} {arr.ea_unit}" if arr.ea is not None else "n/a"
    ax.set_xlabel("1000 / T (1/K)")
    ax.set_ylabel("ln(k)")
    ax.set_title(
        f"Arrhenius exploratory — Ea ≈ {ea_txt} "
        f"(R²={arr.r_squared:.3f})\nSUPPORTIVE / thăm dò ONLY"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
