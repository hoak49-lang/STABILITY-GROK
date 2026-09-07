"""PDF export produces non-empty bytes for a toy analysis result."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.attributes import AttributeConfig, analyze_attributes
from src.report import build_html_report, build_pdf_report, dataframe_to_html, export_pdf_bytes


def _toy_multi_attribute_result():
    t = np.array([0.0, 3, 6, 9, 12, 18, 24], dtype=float)
    assay = 100.0 - 0.35 * t
    impurity = 0.05 + 0.015 * t
    df_assay = pd.DataFrame(
        {
            "batch": ["B1"] * len(t),
            "time": t,
            "response": assay,
            "condition": ["25C/60RH"] * len(t),
        }
    )
    df_imp = pd.DataFrame(
        {
            "batch": ["B1"] * len(t),
            "time": t,
            "response": impurity,
            "condition": ["25C/60RH"] * len(t),
        }
    )
    configs = [
        AttributeConfig(name="Assay", direction="decreasing", spec_limit=90.0),
        AttributeConfig(name="Impurity", direction="increasing", spec_limit=0.5),
    ]
    return analyze_attributes(
        {"Assay": df_assay, "Impurity": df_imp},
        configs,
        alpha=0.05,
        alpha_pool=0.25,
        t_max=60.0,
    )


def test_build_pdf_report_non_empty_toy():
    ma = _toy_multi_attribute_result()
    assert ma.overall_shelf_life is not None

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [100, 98, 96])
    ax.set_title("Toy plot")

    meta = {
        "Condition": "25C/60RH",
        "N attributes": ma.n_attributes,
        "Overall shelf life (months)": ma.overall_shelf_life,
        "Limiting attribute": ma.limiting_attribute,
        "α (one-sided CI)": 0.05,
    }
    sections = []
    for ar in ma.attributes:
        coef_df = ar.reg.summary_table() if ar.reg is not None else None
        anova_df = ar.reg.anova if ar.reg is not None else None
        sections.append(
            {
                "name": ar.attribute,
                "meta_text": f"Direction: {ar.direction_used}; Spec: {ar.config.spec_limit}",
                "coef_df": coef_df,
                "anova_df": anova_df,
                "shelf_text": ar.message,
                "figures": [fig] if ar.attribute == "Assay" else [],
            }
        )

    pdf = build_pdf_report(
        title="ICH Stability / Shelf-Life Report (Multi-Attribute)",
        meta=meta,
        summary_df=ma.summary,
        coef_df=ma.summary,
        anova_df=None,
        shelf_text=ma.message,
        attribute_sections=sections,
        figures=[],
        notes="Toy unit-test report. Supportive analysis only.",
    )
    plt.close(fig)

    assert isinstance(pdf, (bytes, bytearray))
    assert len(pdf) > 500
    assert pdf[:4] == b"%PDF"


def test_export_pdf_bytes_wrapper():
    ma = _toy_multi_attribute_result()
    meta = {
        "N attributes": ma.n_attributes,
        "Overall shelf life (months)": ma.overall_shelf_life,
    }
    pdf = export_pdf_bytes(
        title="Toy PDF",
        meta=meta,
        summary_df=ma.summary,
        coef_df=ma.summary,
        shelf_text=ma.message,
        notes="test",
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 200


def test_html_report_still_works():
    ma = _toy_multi_attribute_result()
    html = build_html_report(
        title="HTML toy",
        meta={"N attributes": ma.n_attributes},
        coef_html=dataframe_to_html(ma.summary),
        anova_html="<p>n/a</p>",
        shelf_html=f"<p>{ma.message}</p>",
        notes="html path",
        summary_html=dataframe_to_html(ma.summary),
        attribute_sections=[
            {
                "name": ar.attribute,
                "meta_html": f"<p>{ar.direction_used}</p>",
                "coef_html": dataframe_to_html(ar.reg.summary_table()) if ar.reg else "",
                "anova_html": "",
                "shelf_html": f"<p>{ar.message}</p>",
                "figures": [],
            }
            for ar in ma.attributes
        ],
    )
    assert "ICH Q1A(R2)" in html
    assert "Disclaimer" in html
    assert "Assay" in html
