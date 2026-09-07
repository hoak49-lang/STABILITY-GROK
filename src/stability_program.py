"""Multi-product stability program: catalog, schedule, due reminders, results.

MVP persistence: session_state + JSON/CSV download/upload (no mandatory DB).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PULL_STATUSES = ["Upcoming", "Due", "Overdue", "Completed"]


@dataclass
class ProgramProduct:
    product_id: str
    name: str
    dosage_form: str = ""
    strengths: List[str] = field(default_factory=list)
    packs: List[str] = field(default_factory=list)


@dataclass
class ProgramStudy:
    study_id: str
    product_id: str
    title: str
    design_type: str = "Full"
    climate_zone: str = ""
    proposed_storage: str = ""
    start_date: str = ""  # ISO YYYY-MM-DD
    conditions: List[str] = field(default_factory=list)
    batches: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    packs: List[str] = field(default_factory=list)
    timepoints: List[float] = field(default_factory=list)
    status: str = "Running"  # Running / Closed
    notes: str = ""


@dataclass
class SamplingPull:
    pull_id: str
    study_id: str
    batch: str
    condition: str
    strength: str
    pack: str
    time_months: float
    due_date: str  # ISO
    status: str = "Upcoming"  # Upcoming/Due/Overdue/Completed
    result_response: Optional[float] = None
    result_attribute: str = "Assay"
    completed_date: str = ""
    notes: str = ""


@dataclass
class StabilityProgram:
    products: List[ProgramProduct] = field(default_factory=list)
    studies: List[ProgramStudy] = field(default_factory=list)
    pulls: List[SamplingPull] = field(default_factory=list)
    lead_days: int = 14  # 7 / 14 / 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "products": [asdict(p) for p in self.products],
            "studies": [asdict(s) for s in self.studies],
            "pulls": [asdict(p) for p in self.pulls],
            "lead_days": self.lead_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StabilityProgram":
        products = [ProgramProduct(**p) for p in (data.get("products") or [])]
        studies = []
        for s in data.get("studies") or []:
            studies.append(
                ProgramStudy(
                    study_id=str(s["study_id"]),
                    product_id=str(s["product_id"]),
                    title=str(s.get("title", "")),
                    design_type=str(s.get("design_type", "Full")),
                    climate_zone=str(s.get("climate_zone", "")),
                    proposed_storage=str(s.get("proposed_storage", "")),
                    start_date=str(s.get("start_date", "")),
                    conditions=list(s.get("conditions") or []),
                    batches=list(s.get("batches") or []),
                    strengths=list(s.get("strengths") or []),
                    packs=list(s.get("packs") or []),
                    timepoints=[float(t) for t in (s.get("timepoints") or [])],
                    status=str(s.get("status", "Running")),
                    notes=str(s.get("notes", "")),
                )
            )
        pulls = []
        for p in data.get("pulls") or []:
            pulls.append(
                SamplingPull(
                    pull_id=str(p["pull_id"]),
                    study_id=str(p["study_id"]),
                    batch=str(p.get("batch", "")),
                    condition=str(p.get("condition", "")),
                    strength=str(p.get("strength", "")),
                    pack=str(p.get("pack", "")),
                    time_months=float(p.get("time_months", 0)),
                    due_date=str(p.get("due_date", "")),
                    status=str(p.get("status", "Upcoming")),
                    result_response=(
                        None
                        if p.get("result_response") in (None, "")
                        else float(p["result_response"])
                    ),
                    result_attribute=str(p.get("result_attribute", "Assay")),
                    completed_date=str(p.get("completed_date", "")),
                    notes=str(p.get("notes", "")),
                )
            )
        return cls(
            products=products,
            studies=studies,
            pulls=pulls,
            lead_days=int(data.get("lead_days", 14)),
        )


def _parse_iso(d: str) -> Optional[date]:
    if not d:
        return None
    try:
        return date.fromisoformat(str(d)[:10])
    except ValueError:
        return None


def add_months(start: date, months: float) -> date:
    """Approximate calendar advance by months (30.44-day mean month)."""
    days = int(round(float(months) * 30.4375))
    return start + timedelta(days=days)


def compute_pull_status(
    due_date: str,
    *,
    today: Optional[date] = None,
    lead_days: int = 14,
    completed: bool = False,
) -> str:
    """Upcoming / Due / Overdue / Completed based on due date and lead window."""
    if completed:
        return "Completed"
    due = _parse_iso(due_date)
    if due is None:
        return "Upcoming"
    today = today or date.today()
    lead = max(int(lead_days), 0)
    if today > due:
        return "Overdue"
    if today >= due - timedelta(days=lead):
        return "Due"
    return "Upcoming"


def refresh_pull_statuses(
    program: StabilityProgram,
    *,
    today: Optional[date] = None,
) -> StabilityProgram:
    today = today or date.today()
    for pull in program.pulls:
        completed = pull.status == "Completed" or pull.result_response is not None
        if pull.status == "Completed" or (
            pull.result_response is not None and pull.completed_date
        ):
            pull.status = "Completed"
            continue
        if pull.result_response is not None and not pull.completed_date:
            pull.completed_date = today.isoformat()
            pull.status = "Completed"
            continue
        pull.status = compute_pull_status(
            pull.due_date, today=today, lead_days=program.lead_days, completed=False
        )
    return program


def build_pulls_from_matrix(
    study: ProgramStudy,
    matrix: pd.DataFrame,
    *,
    start_date: Optional[str] = None,
) -> List[SamplingPull]:
    """Create SamplingPull list from a Q1D sampling matrix DataFrame."""
    start = _parse_iso(start_date or study.start_date) or date.today()
    pulls: List[SamplingPull] = []
    if matrix is None or len(matrix) == 0:
        return pulls
    for i, row in matrix.iterrows():
        t = float(row.get("time_months", 0))
        due = add_months(start, t)
        pull_id = f"{study.study_id}-P{i+1:04d}"
        pulls.append(
            SamplingPull(
                pull_id=pull_id,
                study_id=study.study_id,
                batch=str(row.get("batch", "")),
                condition=str(row.get("condition", "")),
                strength=str(row.get("strength", "")),
                pack=str(row.get("pack", "")),
                time_months=t,
                due_date=due.isoformat(),
                status="Upcoming",
                result_attribute="Assay",
            )
        )
    return pulls


def build_pulls_from_study_schedule(study: ProgramStudy) -> List[SamplingPull]:
    """Manual schedule: cartesian of study factors × timepoints."""
    start = _parse_iso(study.start_date) or date.today()
    batches = study.batches or ["B1"]
    conditions = study.conditions or ["25C/60%RH (long-term)"]
    strengths = study.strengths or ["(n/a)"]
    packs = study.packs or ["(n/a)"]
    times = study.timepoints or [0, 3, 6, 12]
    pulls: List[SamplingPull] = []
    n = 0
    for batch in batches:
        for cond in conditions:
            for strength in strengths:
                for pack in packs:
                    for t in times:
                        n += 1
                        due = add_months(start, float(t))
                        pulls.append(
                            SamplingPull(
                                pull_id=f"{study.study_id}-P{n:04d}",
                                study_id=study.study_id,
                                batch=batch,
                                condition=cond,
                                strength=strength,
                                pack=pack,
                                time_months=float(t),
                                due_date=due.isoformat(),
                                status="Upcoming",
                            )
                        )
    return pulls


def dashboard_counts(
    program: StabilityProgram,
    *,
    today: Optional[date] = None,
) -> Dict[str, int]:
    refresh_pull_statuses(program, today=today)
    running = sum(1 for s in program.studies if s.status == "Running")
    upcoming = sum(1 for p in program.pulls if p.status == "Upcoming")
    due = sum(1 for p in program.pulls if p.status == "Due")
    overdue = sum(1 for p in program.pulls if p.status == "Overdue")
    completed = sum(1 for p in program.pulls if p.status == "Completed")
    return {
        "running_studies": running,
        "upcoming": upcoming,
        "due": due,
        "overdue": overdue,
        "completed": completed,
        "total_pulls": len(program.pulls),
    }


def pulls_to_dataframe(program: StabilityProgram) -> pd.DataFrame:
    if not program.pulls:
        return pd.DataFrame(
            columns=[
                "pull_id",
                "study_id",
                "batch",
                "condition",
                "strength",
                "pack",
                "time_months",
                "due_date",
                "status",
                "result_attribute",
                "result_response",
                "completed_date",
                "notes",
            ]
        )
    return pd.DataFrame([asdict(p) for p in program.pulls])


def results_to_analysis_frame(
    program: StabilityProgram,
    study_id: str,
    *,
    attribute: Optional[str] = None,
    condition: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert completed pulls with numeric results into analysis-ready long format:
    batch, time, response, condition [, attribute].
    """
    rows = []
    for p in program.pulls:
        if p.study_id != study_id:
            continue
        if p.result_response is None:
            continue
        if attribute and p.result_attribute != attribute:
            continue
        if condition and p.condition != condition:
            continue
        rows.append(
            {
                "batch": p.batch,
                "time": p.time_months,
                "response": float(p.result_response),
                "condition": p.condition,
                "attribute": p.result_attribute or "Assay",
                "strength": p.strength,
                "pack": p.pack,
            }
        )
    return pd.DataFrame(rows)


def program_to_json(program: StabilityProgram) -> str:
    return json.dumps(program.to_dict(), ensure_ascii=False, indent=2)


def program_from_json(text: str) -> StabilityProgram:
    return StabilityProgram.from_dict(json.loads(text))


def mark_pull_completed(
    program: StabilityProgram,
    pull_id: str,
    response: float,
    *,
    attribute: str = "Assay",
    completed_on: Optional[str] = None,
    notes: str = "",
) -> bool:
    for p in program.pulls:
        if p.pull_id == pull_id:
            p.result_response = float(response)
            p.result_attribute = attribute
            p.completed_date = completed_on or date.today().isoformat()
            p.status = "Completed"
            if notes:
                p.notes = notes
            return True
    return False


def sample_program() -> StabilityProgram:
    """Demo multi-product program with mixed pull statuses."""
    today = date.today()
    products = [
        ProgramProduct(
            product_id="P-XYZ",
            name="Demo Tablet XYZ",
            dosage_form="Film-coated tablet",
            strengths=["5 mg", "20 mg"],
            packs=["Alu-Alu blister", "HDPE bottle 100s"],
        ),
        ProgramProduct(
            product_id="P-ABC",
            name="Demo Capsule ABC",
            dosage_form="Hard capsule",
            strengths=["50 mg"],
            packs=["Blister"],
        ),
    ]
    studies = [
        ProgramStudy(
            study_id="ST-XYZ-001",
            product_id="P-XYZ",
            title="XYZ formal stability (Zone IVb)",
            design_type="Bracketing",
            climate_zone="IVb (hot / very humid)",
            proposed_storage="Store below 30°C",
            start_date=(today - timedelta(days=200)).isoformat(),
            conditions=[
                "30C/75%RH (long-term zone IV)",
                "40C/75%RH (accelerated)",
            ],
            batches=["B1", "B2"],
            strengths=["5 mg", "20 mg"],
            packs=["Alu-Alu blister", "HDPE bottle 100s"],
            timepoints=[0, 3, 6, 9, 12],
            status="Running",
        ),
        ProgramStudy(
            study_id="ST-ABC-001",
            product_id="P-ABC",
            title="ABC supportive accelerated",
            design_type="Full",
            climate_zone="II (subtropical / Mediterranean)",
            start_date=(today - timedelta(days=90)).isoformat(),
            conditions=["40C/75%RH (accelerated)"],
            batches=["C1"],
            strengths=["50 mg"],
            packs=["Blister"],
            timepoints=[0, 1, 3, 6],
            status="Running",
        ),
    ]
    # Build a compact pull set manually for predictable statuses
    pulls: List[SamplingPull] = []
    # Completed past pulls for XYZ B1 long-term
    start_xyz = _parse_iso(studies[0].start_date) or today
    for t, resp in [(0, 100.1), (3, 99.4), (6, 98.7)]:
        due = add_months(start_xyz, t)
        pulls.append(
            SamplingPull(
                pull_id=f"ST-XYZ-001-T{int(t)}-B1-LT",
                study_id="ST-XYZ-001",
                batch="B1",
                condition="30C/75%RH (long-term zone IV)",
                strength="5 mg",
                pack="Alu-Alu blister",
                time_months=float(t),
                due_date=due.isoformat(),
                status="Completed",
                result_response=resp,
                result_attribute="Assay",
                completed_date=due.isoformat(),
            )
        )
    # Due / Overdue / Upcoming
    pulls.append(
        SamplingPull(
            pull_id="ST-XYZ-001-T9-B1-LT",
            study_id="ST-XYZ-001",
            batch="B1",
            condition="30C/75%RH (long-term zone IV)",
            strength="5 mg",
            pack="Alu-Alu blister",
            time_months=9.0,
            due_date=(today + timedelta(days=3)).isoformat(),
            status="Due",
            result_attribute="Assay",
        )
    )
    pulls.append(
        SamplingPull(
            pull_id="ST-XYZ-001-T6-B2-LT",
            study_id="ST-XYZ-001",
            batch="B2",
            condition="30C/75%RH (long-term zone IV)",
            strength="20 mg",
            pack="HDPE bottle 100s",
            time_months=6.0,
            due_date=(today - timedelta(days=10)).isoformat(),
            status="Overdue",
            result_attribute="Assay",
        )
    )
    pulls.append(
        SamplingPull(
            pull_id="ST-XYZ-001-T12-B1-LT",
            study_id="ST-XYZ-001",
            batch="B1",
            condition="30C/75%RH (long-term zone IV)",
            strength="5 mg",
            pack="Alu-Alu blister",
            time_months=12.0,
            due_date=(today + timedelta(days=60)).isoformat(),
            status="Upcoming",
            result_attribute="Assay",
        )
    )
    pulls.append(
        SamplingPull(
            pull_id="ST-ABC-001-T3-C1-ACC",
            study_id="ST-ABC-001",
            batch="C1",
            condition="40C/75%RH (accelerated)",
            strength="50 mg",
            pack="Blister",
            time_months=3.0,
            due_date=(today + timedelta(days=5)).isoformat(),
            status="Due",
            result_attribute="Assay",
        )
    )
    prog = StabilityProgram(products=products, studies=studies, pulls=pulls, lead_days=14)
    return refresh_pull_statuses(prog, today=today)
