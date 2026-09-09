"""Tests for ACTD ASEAN P.8 report export."""

import pandas as pd

from src.p8_report import (
    P8Fields,
    P8Header,
    build_p8_from_app,
    build_p8_html,
    demo_p8_fields,
    export_p8_pdf_bytes,
)


def test_demo_html_contains_sections():
    fields = demo_p8_fields()
    html = build_p8_html(fields)
    assert "P.8.1" in html
    assert "P.8.2" in html
    assert "P.8.3" in html
    assert "P.8.4" in html
    assert "ACTD ASEAN" in html
    assert "hỗ trợ" in html.lower() or "HSĐK" in html


def test_demo_pdf_nonzero():
    pdf = export_p8_pdf_bytes(demo_p8_fields())
    assert isinstance(pdf, (bytes, bytearray))
    assert len(pdf) > 500
    assert pdf[:4] == b"%PDF"


def test_build_from_dataframe_fills_tables():
    df = pd.DataFrame(
        {
            "batch": ["A", "A", "A", "A"],
            "time": [0, 3, 6, 0],
            "response": [100.0, 99.0, 98.0, 0.05],
            "condition": ["25C/60%RH", "25C/60%RH", "25C/60%RH", "40C/75%RH"],
            "attribute": ["Assay", "Assay", "Assay", "Impurity"],
        }
    )
    fields = build_p8_from_app(df=df, data_label="unit-test")
    assert fields.batch_design_rows
    assert fields.study_condition_rows
    assert fields.results_tables
    html = build_p8_html(fields)
    assert "Assay" in html or "25C" in html


def test_placeholder_when_empty():
    fields = P8Fields(header=P8Header())
    html = build_p8_html(fields)
    assert "[…]" in html
