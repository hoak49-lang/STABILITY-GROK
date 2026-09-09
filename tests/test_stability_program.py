"""Tests for stability program due-date logic and results feed."""

from datetime import date, timedelta

from src.stability_program import (
    StabilityProgram,
    ProgramStudy,
    SamplingPull,
    add_months,
    compute_pull_status,
    dashboard_counts,
    mark_pull_completed,
    program_from_json,
    program_to_json,
    refresh_pull_statuses,
    results_to_analysis_frame,
    sample_program,
    build_pulls_from_study_schedule,
)


def test_compute_pull_status_windows():
    today = date(2026, 9, 8)
    assert compute_pull_status("2026-09-20", today=today, lead_days=14) == "Due"
    assert compute_pull_status("2026-12-01", today=today, lead_days=14) == "Upcoming"
    assert compute_pull_status("2026-09-01", today=today, lead_days=14) == "Overdue"
    assert compute_pull_status("2026-09-01", today=today, completed=True) == "Completed"


def test_add_months_and_refresh():
    start = date(2026, 1, 1)
    assert add_months(start, 0) == start
    due6 = add_months(start, 6)
    assert due6 > start
    prog = StabilityProgram(
        pulls=[
            SamplingPull(
                pull_id="P1",
                study_id="S1",
                batch="A",
                condition="25C/60%RH",
                strength="5",
                pack="Blister",
                time_months=6,
                due_date=(date.today() - timedelta(days=2)).isoformat(),
                status="Upcoming",
            )
        ],
        lead_days=7,
    )
    refresh_pull_statuses(prog)
    assert prog.pulls[0].status == "Overdue"


def test_dashboard_and_sample_program():
    prog = sample_program()
    counts = dashboard_counts(prog)
    assert counts["running_studies"] >= 1
    assert counts["total_pulls"] >= 1
    assert counts["overdue"] + counts["due"] + counts["upcoming"] + counts["completed"] == counts[
        "total_pulls"
    ]


def test_mark_completed_and_analysis_frame():
    prog = sample_program()
    # find an incomplete pull
    target = next(p for p in prog.pulls if p.status != "Completed")
    ok = mark_pull_completed(prog, target.pull_id, 98.5, attribute="Assay")
    assert ok
    df = results_to_analysis_frame(prog, target.study_id)
    assert len(df) >= 1
    assert {"batch", "time", "response", "condition", "attribute"}.issubset(df.columns)


def test_build_schedule_and_json_roundtrip():
    study = ProgramStudy(
        study_id="S1",
        product_id="P1",
        title="T",
        start_date="2026-01-01",
        batches=["B1"],
        conditions=["25C/60%RH (long-term)"],
        strengths=["5 mg"],
        packs=["Blister"],
        timepoints=[0, 3, 6],
    )
    pulls = build_pulls_from_study_schedule(study)
    assert len(pulls) == 3
    prog = StabilityProgram(studies=[study], pulls=pulls, lead_days=30)
    text = program_to_json(prog)
    prog2 = program_from_json(text)
    assert len(prog2.pulls) == 3
    assert prog2.lead_days == 30
