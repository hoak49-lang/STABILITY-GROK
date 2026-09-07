"""Multi-attribute shelf-life analysis (assay, impurity, …).

Runs ICH Q1E regression (+ optional multi-batch pooling) independently per
attribute, then aggregates:

    overall product shelf life = minimum of attribute shelf lives
    (conservative ICH-style)

Single-attribute data remains fully supported via a one-element attribute list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .pooling import MultiBatchResult, analyze_multibatch
from .regression import (
    RegressionResult,
    ShelfLifeResult,
    estimate_shelf_life,
    fit_ols,
    infer_direction,
)


@dataclass
class AttributeConfig:
    """Per-attribute analysis settings."""

    name: str
    direction: str = "decreasing"  # decreasing | increasing | auto
    spec_limit: float = 90.0
    transform: str = "none"

    def direction_arg(self) -> Optional[str]:
        if self.direction in ("decreasing", "increasing"):
            return self.direction
        return None  # auto


@dataclass
class AttributeResult:
    """Result of analyzing one attribute (single- or multi-batch)."""

    attribute: str
    config: AttributeConfig
    kind: str  # "single" | "multi"
    reported_shelf_life: Optional[float]
    direction_used: str
    message: str
    # single-batch fields
    reg: Optional[RegressionResult] = None
    shelf: Optional[ShelfLifeResult] = None
    batch_name: Optional[str] = None
    # multi-batch fields
    mb: Optional[MultiBatchResult] = None
    # data used
    df: Optional[pd.DataFrame] = None

    def to_summary_row(self) -> Dict[str, Any]:
        return {
            "Attribute / Chỉ tiêu": self.attribute,
            "Direction / Hướng": self.direction_used,
            "Spec limit": self.config.spec_limit,
            "Transform": self.config.transform,
            "Kind": self.kind,
            "Shelf life (months)": self.reported_shelf_life,
            "Mode": (
                self.mb.reported_mode
                if self.mb is not None
                else ("single_batch" if self.kind == "single" else "")
            ),
        }


@dataclass
class MultiAttributeResult:
    """Aggregated multi-attribute session result."""

    attributes: List[AttributeResult]
    overall_shelf_life: Optional[float]
    limiting_attribute: Optional[str]
    summary: pd.DataFrame
    message: str

    @property
    def n_attributes(self) -> int:
        return len(self.attributes)


def aggregate_overall_shelf_life(
    shelf_lives: List[Optional[float]],
) -> Optional[float]:
    """Conservative overall = minimum of finite attribute shelf lives.

    None / missing values are ignored. If none are finite, returns None.
    """
    vals = [float(v) for v in shelf_lives if v is not None]
    if not vals:
        return None
    return float(min(vals))


def analyze_one_attribute(
    df: pd.DataFrame,
    config: AttributeConfig,
    *,
    alpha: float = 0.05,
    alpha_pool: float = 0.25,
    t_max: float = 120.0,
    force_pooled: Optional[bool] = None,
) -> AttributeResult:
    """Run single- or multi-batch ICH analysis for one attribute.

    Expects ``df`` with columns batch, time, response, condition
    (already filtered to one attribute / condition as needed).
    """
    work = df.copy()
    if "response" not in work.columns:
        raise ValueError("DataFrame must contain a 'response' column.")

    n_batches = int(work["batch"].nunique())
    dir_arg = config.direction_arg()

    if n_batches <= 1:
        batch_name = str(work["batch"].iloc[0]) if len(work) else "?"
        reg = fit_ols(work["time"].values, work["response"].values, transform=config.transform)
        direction_used = infer_direction(reg.beta1, dir_arg)
        shelf = estimate_shelf_life(
            reg,
            spec_limit=config.spec_limit,
            direction=direction_used,
            alpha=alpha,
            t_max=t_max,
        )
        return AttributeResult(
            attribute=config.name,
            config=config,
            kind="single",
            reported_shelf_life=shelf.shelf_life,
            direction_used=direction_used,
            message=shelf.message,
            reg=reg,
            shelf=shelf,
            batch_name=batch_name,
            df=work,
        )

    mb = analyze_multibatch(
        work,
        spec_limit=config.spec_limit,
        direction=dir_arg,
        transform=config.transform,
        alpha=alpha,
        alpha_pool=alpha_pool,
        t_max=t_max,
        force_pooled=force_pooled,
    )
    # Prefer pooled direction if available, else first per-batch
    if mb.pooled_shelf is not None:
        direction_used = mb.pooled_shelf.direction
    elif mb.per_batch:
        direction_used = next(iter(mb.per_batch.values()))[1].direction
    else:
        direction_used = config.direction if config.direction != "auto" else "decreasing"

    return AttributeResult(
        attribute=config.name,
        config=config,
        kind="multi",
        reported_shelf_life=mb.reported_shelf_life,
        direction_used=direction_used,
        message=mb.message,
        mb=mb,
        df=work,
    )


def analyze_attributes(
    data_by_attribute: Dict[str, pd.DataFrame],
    configs: List[AttributeConfig],
    *,
    alpha: float = 0.05,
    alpha_pool: float = 0.25,
    t_max: float = 120.0,
    force_pooled: Optional[bool] = None,
) -> MultiAttributeResult:
    """Analyze each configured attribute and aggregate overall shelf life."""
    results: List[AttributeResult] = []
    for cfg in configs:
        if cfg.name not in data_by_attribute:
            raise KeyError(f"No data for attribute '{cfg.name}'.")
        df_attr = data_by_attribute[cfg.name]
        if df_attr is None or len(df_attr) == 0:
            raise ValueError(f"Empty data for attribute '{cfg.name}'.")
        results.append(
            analyze_one_attribute(
                df_attr,
                cfg,
                alpha=alpha,
                alpha_pool=alpha_pool,
                t_max=t_max,
                force_pooled=force_pooled,
            )
        )

    lives = [r.reported_shelf_life for r in results]
    overall = aggregate_overall_shelf_life(lives)

    limiting = None
    if overall is not None:
        for r in results:
            if r.reported_shelf_life is not None and abs(r.reported_shelf_life - overall) < 1e-9:
                limiting = r.attribute
                break

    summary = pd.DataFrame([r.to_summary_row() for r in results])
    if overall is not None:
        msg = (
            f"Overall product shelf life (min of {len(results)} attribute(s)) = "
            f"{overall:.2f} months"
            + (f" — limited by '{limiting}'." if limiting else ".")
            + f" / Thời hạn bảo quản tổng thể (min các chỉ tiêu) = {overall:.2f} tháng."
        )
    else:
        msg = (
            "Unable to determine overall shelf life (no attribute intersected spec). "
            "/ Không xác định được shelf life tổng thể."
        )

    return MultiAttributeResult(
        attributes=results,
        overall_shelf_life=overall,
        limiting_attribute=limiting,
        summary=summary,
        message=msg,
    )


def default_config_for_attribute(name: str) -> AttributeConfig:
    """Heuristic defaults from attribute name (assay↓ / impurity↑)."""
    key = (name or "").lower().strip()
    impurity_keys = ("impurity", "imp", "tap_chat", "tạp", "degrad", "related")
    assay_keys = ("assay", "content", "ham_luong", "hàm_lượng", "potency", "api")
    if any(k in key for k in impurity_keys):
        return AttributeConfig(name=name, direction="increasing", spec_limit=0.5, transform="none")
    if any(k in key for k in assay_keys):
        return AttributeConfig(name=name, direction="decreasing", spec_limit=90.0, transform="none")
    return AttributeConfig(name=name, direction="decreasing", spec_limit=90.0, transform="none")
