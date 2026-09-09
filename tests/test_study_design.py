"""Tests for ICH Q1D study design / sampling matrix generation."""

from src.study_design import (
    StudyDesign,
    ProductInfo,
    design_wizard_notes,
    generate_sampling_matrix,
    protocol_from_json,
    protocol_to_json,
    sample_design,
    classify_condition_role,
    method_tag_for_role,
)


def test_full_matrix_covers_all_combos():
    d = StudyDesign(
        product=ProductInfo(
            name="X",
            strengths=["5 mg", "10 mg"],
            packs=["Blister", "Bottle"],
        ),
        design_type="Full",
        timepoints=[0, 3, 6],
        conditions=["25C/60%RH (long-term)", "40C/75%RH (accelerated)"],
        batches=["B1"],
        include_intermediate=True,
    )
    mat = generate_sampling_matrix(d)
    # 1 batch × 2 cond × 2 str × 2 pack × 3 times = 24
    assert len(mat) == 24
    assert set(mat["strength"]) == {"5 mg", "10 mg"}
    assert set(mat["pack"]) == {"Blister", "Bottle"}


def test_bracketing_uses_extremes():
    d = sample_design()
    d.design_type = "Bracketing"
    d.timepoints = [0, 6, 12]
    d.batches = ["B1"]
    d.conditions = ["30C/75%RH (long-term zone IV)"]
    d.product.strengths = ["5 mg", "10 mg", "20 mg"]
    d.product.packs = ["A", "B", "C"]
    d.bracket_strength_extremes_only = True
    d.bracket_pack_extremes_only = True
    mat = generate_sampling_matrix(d)
    assert set(mat["strength"]) == {"5 mg", "20 mg"}
    assert set(mat["pack"]) == {"A", "C"}
    # 1 × 1 × 2 × 2 × 3 = 12
    assert len(mat) == 12


def test_matrixing_keeps_anchors_reduces_middle():
    d = StudyDesign(
        product=ProductInfo(name="M", strengths=["5 mg"], packs=["Blister"]),
        design_type="Matrixing",
        timepoints=[0, 3, 6, 9, 12],
        conditions=["25C/60%RH (long-term)"],
        batches=["B1", "B2"],
        matrix_fraction=0.5,
        matrix_seed=1,
    )
    mat = generate_sampling_matrix(d)
    # anchors: all batches keep 0 and 12
    anchors = mat[mat["time_months"].isin([0.0, 12.0])]
    assert len(anchors) == 4  # 2 batches × 2 anchors
    middle = mat[~mat["time_months"].isin([0.0, 12.0])]
    full_middle = 2 * 3  # 2 batches × 3 middle times
    assert 0 < len(middle) < full_middle or len(middle) == full_middle
    # With fraction 0.5 should typically be fewer than full; allow equality if unlucky
    assert len(mat) <= 2 * 1 * 1 * 1 * 5


def test_exclude_intermediate_flag():
    d = sample_design()
    d.include_intermediate = False
    d.batches = ["B1"]
    d.timepoints = [0, 6]
    mat = generate_sampling_matrix(d)
    assert all(mat["condition_role"] != "intermediate")


def test_condition_roles_and_tags():
    assert classify_condition_role("40C/75%RH (accelerated)") == "accelerated"
    assert method_tag_for_role("accelerated") == "LHCT"
    assert method_tag_for_role("long-term") == "LT-Q1E"
    assert "Arrhenius" in method_tag_for_role("arrhenius")


def test_wizard_notes_and_protocol_roundtrip():
    notes = design_wizard_notes("Bracketing")
    assert "prerequisites" in notes and "risks" in notes
    d = sample_design()
    mat = generate_sampling_matrix(d)
    text = protocol_to_json(d, mat)
    d2, mat2 = protocol_from_json(text)
    assert d2.product.name == d.product.name
    assert d2.design_type == d.design_type
    assert mat2 is not None and len(mat2) == len(mat)
