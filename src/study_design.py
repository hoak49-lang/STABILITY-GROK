"""ICH Q1D-oriented study design: Full / Bracketing / Matrixing sampling matrices.

Supportive planning helper — not a regulatory certification system.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_TIMEPOINTS = [0, 3, 6, 9, 12, 18, 24, 36]

CLIMATE_ZONES = [
    "I (temperate)",
    "II (subtropical / Mediterranean)",
    "III (hot / dry)",
    "IVa (hot / humid)",
    "IVb (hot / very humid)",
]

DESIGN_TYPES = ["Full", "Bracketing", "Matrixing"]

CONDITION_PRESETS = [
    "25C/60%RH (long-term)",
    "30C/65%RH (intermediate)",
    "30C/75%RH (long-term zone IV)",
    "40C/75%RH (accelerated)",
]

# Roles used in UI / reports
CONDITION_ROLES = {
    "long-term": "LT-Q1E",
    "intermediate": "Intermediate",
    "accelerated": "LHCT",
}


@dataclass
class ProductInfo:
    name: str = ""
    dosage_form: str = ""
    strengths: List[str] = field(default_factory=list)
    packs: List[str] = field(default_factory=list)
    proposed_storage: str = "Store below 30°C"
    climate_zone: str = "IVb (hot / very humid)"


@dataclass
class StudyDesign:
    product: ProductInfo = field(default_factory=ProductInfo)
    design_type: str = "Full"
    timepoints: List[float] = field(default_factory=lambda: list(DEFAULT_TIMEPOINTS))
    conditions: List[str] = field(default_factory=lambda: list(CONDITION_PRESETS[:2] + [CONDITION_PRESETS[-1]]))
    batches: List[str] = field(default_factory=lambda: ["B1", "B2", "B3"])
    include_intermediate: bool = True
    notes: str = ""
    # Bracketing: only extremes of strength/pack are fully tested
    bracket_strength_extremes_only: bool = True
    bracket_pack_extremes_only: bool = True
    # Matrixing: fraction of (time × factor) combinations per batch
    matrix_fraction: float = 0.5
    matrix_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product": asdict(self.product),
            "design_type": self.design_type,
            "timepoints": list(self.timepoints),
            "conditions": list(self.conditions),
            "batches": list(self.batches),
            "include_intermediate": self.include_intermediate,
            "notes": self.notes,
            "bracket_strength_extremes_only": self.bracket_strength_extremes_only,
            "bracket_pack_extremes_only": self.bracket_pack_extremes_only,
            "matrix_fraction": self.matrix_fraction,
            "matrix_seed": self.matrix_seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyDesign":
        prod = data.get("product") or {}
        product = ProductInfo(
            name=str(prod.get("name", "")),
            dosage_form=str(prod.get("dosage_form", "")),
            strengths=list(prod.get("strengths") or []),
            packs=list(prod.get("packs") or []),
            proposed_storage=str(prod.get("proposed_storage", "Store below 30°C")),
            climate_zone=str(prod.get("climate_zone", CLIMATE_ZONES[-1])),
        )
        return cls(
            product=product,
            design_type=str(data.get("design_type", "Full")),
            timepoints=[float(t) for t in (data.get("timepoints") or DEFAULT_TIMEPOINTS)],
            conditions=list(data.get("conditions") or CONDITION_PRESETS[:2] + [CONDITION_PRESETS[-1]]),
            batches=list(data.get("batches") or ["B1", "B2", "B3"]),
            include_intermediate=bool(data.get("include_intermediate", True)),
            notes=str(data.get("notes", "")),
            bracket_strength_extremes_only=bool(data.get("bracket_strength_extremes_only", True)),
            bracket_pack_extremes_only=bool(data.get("bracket_pack_extremes_only", True)),
            matrix_fraction=float(data.get("matrix_fraction", 0.5)),
            matrix_seed=int(data.get("matrix_seed", 42)),
        )


def design_wizard_notes(design_type: str) -> Dict[str, str]:
    """Prerequisites / risks for Full / Bracketing / Matrixing (ICH Q1D spirit)."""
    dt = (design_type or "Full").strip().title()
    if dt == "Bracketing":
        return {
            "title": "Bracketing (ICH Q1D)",
            "prerequisites": (
                "Use when intermediate strengths/packs are reasonably represented by extremes "
                "(same formulation platform, proportional composition or same container/closure "
                "family with comparable protective properties)."
            ),
            "risks": (
                "If degradation or moisture uptake is not monotonic with strength/pack barrier, "
                "extremes may not bracket intermediates — full or matrixing may be safer."
            ),
            "when_appropriate": (
                "Multiple strengths with same excipient ratios, or pack sizes of the same "
                "material/closure where surface-area/volume extremes bound the risk."
            ),
        }
    if dt == "Matrixing":
        return {
            "title": "Matrixing (ICH Q1D)",
            "prerequisites": (
                "Factor effects (batch, strength, pack) should be limited; schedule must still "
                "cover all combinations at key timepoints (e.g. 0 and end of study) for integrity."
            ),
            "risks": (
                "Reduced power to detect factor×time interactions; incomplete coverage can "
                "complicate poolability and shelf-life justification."
            ),
            "when_appropriate": (
                "Many similar configurations where full testing is resource-heavy and prior "
                "knowledge supports limited interaction risk."
            ),
        }
    return {
        "title": "Full design",
        "prerequisites": "Test all strength × pack × condition × timepoint combinations (per batch plan).",
        "risks": "Highest resource cost; still requires scientific justification of conditions/zone.",
        "when_appropriate": "New product / limited prior knowledge / complex packaging or formulation differences.",
    }


def classify_condition_role(condition: str) -> str:
    """Map a condition label to long-term / intermediate / accelerated / other."""
    s = str(condition).lower()
    if "accelerat" in s or "40c" in s or "40°c" in s or "lhct" in s:
        return "accelerated"
    if "intermed" in s or ("30c" in s and "75" not in s and "zone iv" not in s):
        # 30C/65%RH typically intermediate; 30C/75% often long-term zone IV
        if "65" in s or "intermed" in s:
            return "intermediate"
    if "30c/75" in s or "zone iv" in s or "long-term" in s or "25c" in s or "30c" in s:
        return "long-term"
    if "long" in s:
        return "long-term"
    return "other"


def method_tag_for_role(role: str) -> str:
    role = (role or "other").lower()
    if role == "long-term":
        return "LT-Q1E"
    if role == "accelerated":
        return "LHCT"
    if role == "intermediate":
        return "Intermediate"
    if role in ("arrhenius", "arrhenius-supportive"):
        return "Arrhenius-supportive"
    return "Other"


def _extremes(items: Sequence[str]) -> List[str]:
    vals = [str(x).strip() for x in items if str(x).strip()]
    if not vals:
        return ["(n/a)"]
    if len(vals) == 1:
        return vals
    return [vals[0], vals[-1]]


def _selected_factors(
    strengths: Sequence[str],
    packs: Sequence[str],
    design_type: str,
    bracket_strength: bool,
    bracket_pack: bool,
) -> Tuple[List[str], List[str]]:
    s = [str(x).strip() for x in strengths if str(x).strip()] or ["(n/a)"]
    p = [str(x).strip() for x in packs if str(x).strip()] or ["(n/a)"]
    dt = (design_type or "Full").strip().title()
    if dt == "Bracketing":
        if bracket_strength and len(s) > 2:
            s = _extremes(s)
        if bracket_pack and len(p) > 2:
            p = _extremes(p)
    return s, p


def _matrix_keep(key: str, fraction: float, seed: int) -> bool:
    """Deterministic pseudo-random keep decision for matrixing cells."""
    frac = min(max(float(fraction), 0.1), 1.0)
    # Always keep time 0 and terminal handled by caller; this is for middle cells
    h = (hash((seed, key)) & 0xFFFFFFFF)
    return (h / 0xFFFFFFFF) < frac


def generate_sampling_matrix(design: StudyDesign) -> pd.DataFrame:
    """
    Generate sampling plan rows: time × condition × strength × pack × batch.

    Full: all combinations.
    Bracketing: only extreme strengths/packs (configurable).
    Matrixing: reduced middle timepoints (keeps t=0 and max t for all factor combos).
    """
    strengths, packs = _selected_factors(
        design.product.strengths,
        design.product.packs,
        design.design_type,
        design.bracket_strength_extremes_only,
        design.bracket_pack_extremes_only,
    )
    conditions = list(design.conditions) or ["25C/60%RH (long-term)"]
    if not design.include_intermediate:
        conditions = [c for c in conditions if classify_condition_role(c) != "intermediate"]
        if not conditions:
            conditions = ["25C/60%RH (long-term)"]

    batches = list(design.batches) or ["B1"]
    times = sorted({float(t) for t in design.timepoints})
    if not times:
        times = list(DEFAULT_TIMEPOINTS)

    dt = (design.design_type or "Full").strip().title()
    t_min, t_max = times[0], times[-1]
    rows: List[Dict[str, Any]] = []

    for batch in batches:
        for cond in conditions:
            role = classify_condition_role(cond)
            tag = method_tag_for_role(role)
            for strength in strengths:
                for pack in packs:
                    for t in times:
                        include = True
                        reason = "full"
                        if dt == "Matrixing":
                            if t in (t_min, t_max):
                                include = True
                                reason = "matrix-anchor"
                            else:
                                key = f"{batch}|{cond}|{strength}|{pack}|{t}"
                                include = _matrix_keep(key, design.matrix_fraction, design.matrix_seed)
                                reason = "matrix-sample" if include else "matrix-skip"
                        elif dt == "Bracketing":
                            reason = "bracket"
                        if include:
                            rows.append(
                                {
                                    "batch": batch,
                                    "condition": cond,
                                    "condition_role": role,
                                    "method_tag": tag,
                                    "strength": strength,
                                    "pack": pack,
                                    "time_months": t,
                                    "design_type": dt,
                                    "include_reason": reason,
                                    "product": design.product.name or "",
                                }
                            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "batch",
                "condition",
                "condition_role",
                "method_tag",
                "strength",
                "pack",
                "time_months",
                "design_type",
                "include_reason",
                "product",
            ]
        )
    return df.sort_values(
        ["batch", "condition", "strength", "pack", "time_months"]
    ).reset_index(drop=True)


def sampling_matrix_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def protocol_to_json(design: StudyDesign, matrix: Optional[pd.DataFrame] = None) -> str:
    payload: Dict[str, Any] = {"design": design.to_dict()}
    if matrix is not None and len(matrix) > 0:
        payload["sampling_matrix"] = matrix.to_dict(orient="records")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def protocol_from_json(text: str) -> Tuple[StudyDesign, Optional[pd.DataFrame]]:
    data = json.loads(text)
    if "design" in data:
        design = StudyDesign.from_dict(data["design"])
        matrix = None
        if data.get("sampling_matrix"):
            matrix = pd.DataFrame(data["sampling_matrix"])
        return design, matrix
    # bare design dict
    return StudyDesign.from_dict(data), None


def sample_design() -> StudyDesign:
    """Demo protocol for MVP / tests."""
    return StudyDesign(
        product=ProductInfo(
            name="Demo Tablet XYZ",
            dosage_form="Film-coated tablet",
            strengths=["5 mg", "10 mg", "20 mg"],
            packs=["Alu-Alu blister", "HDPE bottle 30s", "HDPE bottle 100s"],
            proposed_storage="Store below 30°C · Protect from moisture",
            climate_zone="IVb (hot / very humid)",
        ),
        design_type="Bracketing",
        timepoints=list(DEFAULT_TIMEPOINTS),
        conditions=[
            "30C/75%RH (long-term zone IV)",
            "30C/65%RH (intermediate)",
            "40C/75%RH (accelerated)",
        ],
        batches=["B1", "B2", "B3"],
        include_intermediate=True,
        notes="Sample Q1D bracketing on strength/pack extremes.",
        bracket_strength_extremes_only=True,
        bracket_pack_extremes_only=True,
    )
