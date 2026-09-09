"""Tests for session JSON export/import."""

import json

import pandas as pd

from src.session_io import export_session, import_session


def test_roundtrip_df_and_model():
    state = {
        "df": pd.DataFrame(
            {
                "batch": ["A", "A"],
                "time": [0, 3],
                "response": [100.0, 99.0],
                "condition": ["25C", "25C"],
            }
        ),
        "data_label": "test",
        "study_design": {"product_name": "Demo"},
        "stability_program": {"products": []},
        "sampling_matrix": None,
        "condition": "25C",
        "spec_limit": 90.0,
        "direction": "decreasing",
        "transform": "none",
        "alpha": 0.05,
        "alpha_pool": 0.25,
        "t_max": 36.0,
        "pool_mode": "Tự động (ANCOVA)",
        "attr_configs": {"Assay": {"direction": "decreasing", "spec_limit": 90.0, "transform": "none"}},
        "selected_attributes": ["Assay"],
        "analysis": {
            "method_tag": "LT-Q1E",
            "condition": "25C",
            "alpha": 0.05,
            "ma": None,
        },
        "kinetics_order": "first",
        "p8_fields": {"header": {"ten_thuoc": "X"}},
    }
    raw = export_session(state)
    data = json.loads(raw)
    assert data["schema_version"] == 1
    assert data["df"] and len(data["df"]) == 2
    assert data["analysis_meta"]["method_tag"] == "LT-Q1E"

    updates, warnings = import_session(raw)
    assert updates["df"] is not None
    assert len(updates["df"]) == 2
    assert updates["spec_limit"] == 90.0
    assert updates["kinetics_order"] == "first"
    assert updates["analysis"] is None
    assert any("re-run" in w.lower() or "chạy lại" in w.lower() for w in warnings)
