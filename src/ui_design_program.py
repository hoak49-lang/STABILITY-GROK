"""Streamlit UI blocks for Study Design (Q1D) and Stability Program."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.stability_program import (
    ProgramProduct,
    ProgramStudy,
    StabilityProgram,
    build_pulls_from_matrix,
    build_pulls_from_study_schedule,
    dashboard_counts,
    mark_pull_completed,
    program_from_json,
    program_to_json,
    pulls_to_dataframe,
    refresh_pull_statuses,
    results_to_analysis_frame,
    sample_program,
)
from src.study_design import (
    CLIMATE_ZONES,
    CONDITION_PRESETS,
    DEFAULT_TIMEPOINTS,
    DESIGN_TYPES,
    ProductInfo,
    StudyDesign,
    design_wizard_notes,
    generate_sampling_matrix,
    protocol_from_json,
    protocol_to_json,
    sample_design,
    sampling_matrix_to_csv,
    classify_condition_role,
    method_tag_for_role,
)
from src.ui_styles import callout, empty_state, metric_cards, section_header


def _parse_list(text: str) -> List[str]:
    parts = []
    for chunk in str(text).replace(";", ",").split(","):
        v = chunk.strip()
        if v:
            parts.append(v)
    return parts


def _ensure_design_state() -> None:
    if "study_design" not in st.session_state:
        st.session_state["study_design"] = sample_design().to_dict()
    if "sampling_matrix" not in st.session_state:
        st.session_state["sampling_matrix"] = None


def _ensure_program_state() -> None:
    if "stability_program" not in st.session_state:
        st.session_state["stability_program"] = sample_program().to_dict()


def _design_from_state() -> StudyDesign:
    return StudyDesign.from_dict(st.session_state.get("study_design") or sample_design().to_dict())


def _program_from_state() -> StabilityProgram:
    return StabilityProgram.from_dict(
        st.session_state.get("stability_program") or sample_program().to_dict()
    )


def _save_design(design: StudyDesign, matrix: Optional[pd.DataFrame] = None) -> None:
    st.session_state["study_design"] = design.to_dict()
    if matrix is not None:
        st.session_state["sampling_matrix"] = matrix


def _save_program(program: StabilityProgram) -> None:
    refresh_pull_statuses(program)
    st.session_state["stability_program"] = program.to_dict()


def render_study_design_page() -> None:
    """Step: Thiết kế nghiên cứu / Study Design (ICH Q1D)."""
    _ensure_design_state()
    section_header("1", "Thiết kế nghiên cứu", "Study Design · ICH Q1D")
    callout(
        "Hỗ trợ lập kế hoạch lấy mẫu theo tinh thần <b>ICH Q1D</b> (Full / Bracketing / Matrixing). "
        "Công cụ hỗ trợ — <b>không</b> thay thế đánh giá khoa học / đăng ký.",
        kind="warn",
    )

    design = _design_from_state()
    prod = design.product

    c_load1, c_load2, c_load3 = st.columns(3)
    with c_load1:
        if st.button("Load sample design", use_container_width=True):
            d = sample_design()
            mat = generate_sampling_matrix(d)
            _save_design(d, mat)
            st.rerun()
    with c_load2:
        up = st.file_uploader("Upload protocol JSON", type=["json"], key="design_json_up")
        if up is not None:
            try:
                d, mat = protocol_from_json(up.read().decode("utf-8"))
                if mat is None:
                    mat = generate_sampling_matrix(d)
                _save_design(d, mat)
                st.success("Protocol loaded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with c_load3:
        st.caption("Persist via session + JSON download below.")

    st.markdown("#### Thông tin sản phẩm / Product info")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Tên sản phẩm / Product name", value=prod.name)
        dosage = st.text_input("Dạng bào chế / Dosage form", value=prod.dosage_form)
        strengths_txt = st.text_input(
            "Hàm lượng / Strengths (comma-separated)",
            value=", ".join(prod.strengths),
        )
        packs_txt = st.text_input(
            "Quy cách bao bì / Packs (comma-separated)",
            value=", ".join(prod.packs),
        )
    with c2:
        storage = st.text_input("Bảo quản đề xuất / Proposed storage", value=prod.proposed_storage)
        zone = st.selectbox(
            "Vùng khí hậu / Climate zone",
            CLIMATE_ZONES,
            index=CLIMATE_ZONES.index(prod.climate_zone)
            if prod.climate_zone in CLIMATE_ZONES
            else len(CLIMATE_ZONES) - 1,
        )
        batches_txt = st.text_input(
            "Lô / Batches (comma-separated)",
            value=", ".join(design.batches),
        )
        include_inter = st.checkbox(
            "Gồm điều kiện trung gian / Include intermediate",
            value=design.include_intermediate,
        )

    st.markdown("#### Kiểu thiết kế / Design type (ICH Q1D)")
    design_type = st.radio("Design type", DESIGN_TYPES, index=DESIGN_TYPES.index(design.design_type) if design.design_type in DESIGN_TYPES else 0, horizontal=True)
    notes = design_wizard_notes(design_type)
    with st.expander(f"Wizard notes — {notes['title']}", expanded=True):
        st.markdown(f"**When appropriate:** {notes['when_appropriate']}")
        st.markdown(f"**Prerequisites:** {notes['prerequisites']}")
        st.markdown(f"**Risks:** {notes['risks']}")

    c3, c4, c5 = st.columns(3)
    with c3:
        bracket_s = st.checkbox(
            "Bracket strengths (extremes only)",
            value=design.bracket_strength_extremes_only,
            disabled=design_type != "Bracketing",
        )
        bracket_p = st.checkbox(
            "Bracket packs (extremes only)",
            value=design.bracket_pack_extremes_only,
            disabled=design_type != "Bracketing",
        )
    with c4:
        matrix_frac = st.slider(
            "Matrixing fraction (middle timepoints)",
            0.1,
            1.0,
            float(design.matrix_fraction),
            0.05,
            disabled=design_type != "Matrixing",
        )
        matrix_seed = st.number_input(
            "Matrix seed",
            value=int(design.matrix_seed),
            disabled=design_type != "Matrixing",
        )
    with c5:
        tp_txt = st.text_input(
            "Timepoints (months)",
            value=", ".join(str(int(t) if float(t).is_integer() else t) for t in design.timepoints),
        )
        cond_default = design.conditions or CONDITION_PRESETS
        conditions = st.multiselect(
            "Điều kiện / Conditions",
            options=list(dict.fromkeys(CONDITION_PRESETS + list(design.conditions))),
            default=[c for c in cond_default if c in (CONDITION_PRESETS + list(design.conditions))],
        )

    free_notes = st.text_area("Ghi chú protocol / Notes", value=design.notes, height=80)

    # Build design object
    try:
        timepoints = [float(x.strip()) for x in tp_txt.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid timepoints")
        timepoints = list(DEFAULT_TIMEPOINTS)

    new_design = StudyDesign(
        product=ProductInfo(
            name=name,
            dosage_form=dosage,
            strengths=_parse_list(strengths_txt),
            packs=_parse_list(packs_txt),
            proposed_storage=storage,
            climate_zone=zone,
        ),
        design_type=design_type,
        timepoints=timepoints or list(DEFAULT_TIMEPOINTS),
        conditions=conditions or list(CONDITION_PRESETS[:1]),
        batches=_parse_list(batches_txt) or ["B1"],
        include_intermediate=include_inter,
        notes=free_notes,
        bracket_strength_extremes_only=bracket_s,
        bracket_pack_extremes_only=bracket_p,
        matrix_fraction=float(matrix_frac),
        matrix_seed=int(matrix_seed),
    )

    if st.button("⚙ Generate sampling matrix", type="primary"):
        mat = generate_sampling_matrix(new_design)
        _save_design(new_design, mat)
        st.success(f"Generated {len(mat)} sampling cells.")
    else:
        # keep form edits in session even before generate
        st.session_state["study_design"] = new_design.to_dict()

    mat = st.session_state.get("sampling_matrix")
    if isinstance(mat, pd.DataFrame) and len(mat) > 0:
        section_header(None, "Ma trận lấy mẫu", "Sampling matrix")
        # Role summary
        role_counts = mat.groupby("condition_role").size().to_dict()
        cards = [
            {"label": "Cells", "value": str(len(mat)), "tone": "info"},
            {"label": "Long-term", "value": str(role_counts.get("long-term", 0)), "tone": "ok"},
            {"label": "Accelerated (LHCT)", "value": str(role_counts.get("accelerated", 0)), "tone": "warn"},
            {"label": "Intermediate", "value": str(role_counts.get("intermediate", 0)), "tone": "neutral"},
        ]
        metric_cards(cards)
        st.dataframe(mat, use_container_width=True, hide_index=True)

        csv_bytes = sampling_matrix_to_csv(mat).encode("utf-8")
        json_bytes = protocol_to_json(new_design, mat).encode("utf-8")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "⬇ Sampling plan CSV",
                csv_bytes,
                file_name="sampling_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇ Protocol JSON",
                json_bytes,
                file_name="study_design_protocol.json",
                mime="application/json",
                use_container_width=True,
            )
        with d3:
            if st.button("→ Send matrix to Stability Program", use_container_width=True):
                _ensure_program_state()
                prog = _program_from_state()
                # Upsert product + study from design
                pid = "P-" + (new_design.product.name or "PROD").replace(" ", "")[:12].upper()
                sid = "ST-" + pid + "-001"
                # replace existing sample study link if same id
                prog.products = [p for p in prog.products if p.product_id != pid]
                prog.products.append(
                    ProgramProduct(
                        product_id=pid,
                        name=new_design.product.name,
                        dosage_form=new_design.product.dosage_form,
                        strengths=list(new_design.product.strengths),
                        packs=list(new_design.product.packs),
                    )
                )
                study = ProgramStudy(
                    study_id=sid,
                    product_id=pid,
                    title=f"{new_design.product.name} · {new_design.design_type}",
                    design_type=new_design.design_type,
                    climate_zone=new_design.product.climate_zone,
                    proposed_storage=new_design.product.proposed_storage,
                    start_date=date.today().isoformat(),
                    conditions=list(new_design.conditions),
                    batches=list(new_design.batches),
                    strengths=list(new_design.product.strengths),
                    packs=list(new_design.product.packs),
                    timepoints=list(new_design.timepoints),
                    status="Running",
                    notes=new_design.notes,
                )
                prog.studies = [s for s in prog.studies if s.study_id != sid]
                prog.studies.append(study)
                # Replace pulls for this study
                prog.pulls = [p for p in prog.pulls if p.study_id != sid]
                prog.pulls.extend(build_pulls_from_matrix(study, mat))
                _save_program(prog)
                st.success(f"Program updated: study {sid} with {len(mat)} pulls.")
    else:
        empty_state(
            "Chưa có ma trận lấy mẫu",
            "Fill product info and click Generate sampling matrix.",
            "🗂",
        )

    callout(
        "Tiếp theo: <b>2 · Chương trình / Stability Program</b> để theo dõi lịch lấy mẫu, "
        "hoặc <b>3 · Dữ liệu</b> để phân tích Q1E.",
        kind="info",
    )


def render_stability_program_page() -> None:
    """Step: Stability Program (multi-product catalog + due reminders)."""
    _ensure_program_state()
    section_header("2", "Chương trình ổn định", "Stability Program")
    callout(
        "Danh mục sản phẩm / nghiên cứu / lô / điều kiện · lịch lấy mẫu · nhắc hạn "
        "(lead 7/14/30 ngày). MVP: <b>session_state + JSON/CSV</b> (không bắt buộc DB).",
        kind="info",
    )

    prog = _program_from_state()

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        if st.button("Load sample program", use_container_width=True):
            _save_program(sample_program())
            st.rerun()
    with top2:
        lead = st.selectbox("Lead days (Due window)", [7, 14, 30], index=[7, 14, 30].index(prog.lead_days) if prog.lead_days in (7, 14, 30) else 1)
        if lead != prog.lead_days:
            prog.lead_days = int(lead)
            _save_program(prog)
            st.rerun()
    with top3:
        up = st.file_uploader("Upload program JSON", type=["json"], key="prog_json_up")
        if up is not None:
            try:
                _save_program(program_from_json(up.read().decode("utf-8")))
                st.success("Program loaded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with top4:
        st.download_button(
            "⬇ Program JSON",
            program_to_json(prog).encode("utf-8"),
            file_name="stability_program.json",
            mime="application/json",
            use_container_width=True,
        )

    counts = dashboard_counts(prog)
    metric_cards(
        [
            {"label": "Running studies", "value": str(counts["running_studies"]), "tone": "ok"},
            {"label": "Upcoming pulls", "value": str(counts["upcoming"]), "tone": "info"},
            {"label": "Due", "value": str(counts["due"]), "tone": "warn"},
            {"label": "Overdue", "value": str(counts["overdue"]), "tone": "danger" if counts["overdue"] else "neutral"},
        ]
    )

    tab_cat, tab_sched, tab_res, tab_feed = st.tabs(
        ["Catalog", "Schedule / Reminders", "Enter results", "Feed → Analysis"]
    )

    with tab_cat:
        st.markdown("##### Products")
        if prog.products:
            st.dataframe(pd.DataFrame([p.__dict__ for p in prog.products]), use_container_width=True, hide_index=True)
        st.markdown("##### Studies")
        if prog.studies:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "study_id": s.study_id,
                            "product_id": s.product_id,
                            "title": s.title,
                            "design_type": s.design_type,
                            "start_date": s.start_date,
                            "status": s.status,
                            "n_batches": len(s.batches),
                            "n_conditions": len(s.conditions),
                        }
                        for s in prog.studies
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("Add / replace study (manual schedule)", expanded=False):
            sid = st.text_input("study_id", value="ST-NEW-001")
            pid = st.text_input("product_id", value="P-NEW")
            title = st.text_input("title", value="New stability study")
            start = st.date_input("start_date", value=date.today())
            batches = st.text_input("batches", value="B1, B2")
            strengths = st.text_input("strengths", value="10 mg")
            packs = st.text_input("packs", value="Blister")
            conds = st.multiselect("conditions", CONDITION_PRESETS, default=[CONDITION_PRESETS[0], CONDITION_PRESETS[-1]])
            tps = st.text_input("timepoints", value="0, 3, 6, 9, 12")
            dtype = st.selectbox("design_type", DESIGN_TYPES)
            if st.button("Create study + schedule pulls"):
                try:
                    tplist = [float(x.strip()) for x in tps.split(",") if x.strip()]
                except ValueError:
                    st.error("Bad timepoints")
                    tplist = [0, 3, 6, 12]
                # ensure product
                if not any(p.product_id == pid for p in prog.products):
                    prog.products.append(
                        ProgramProduct(
                            product_id=pid,
                            name=title,
                            strengths=_parse_list(strengths),
                            packs=_parse_list(packs),
                        )
                    )
                study = ProgramStudy(
                    study_id=sid,
                    product_id=pid,
                    title=title,
                    design_type=dtype,
                    start_date=start.isoformat(),
                    conditions=conds,
                    batches=_parse_list(batches),
                    strengths=_parse_list(strengths),
                    packs=_parse_list(packs),
                    timepoints=tplist,
                    status="Running",
                )
                prog.studies = [s for s in prog.studies if s.study_id != sid]
                prog.studies.append(study)
                prog.pulls = [p for p in prog.pulls if p.study_id != sid]
                # Prefer design matrix if same session product design exists
                mat = st.session_state.get("sampling_matrix")
                if isinstance(mat, pd.DataFrame) and len(mat) > 0 and dtype != "Full":
                    prog.pulls.extend(build_pulls_from_matrix(study, mat))
                else:
                    prog.pulls.extend(build_pulls_from_study_schedule(study))
                _save_program(prog)
                st.success(f"Created {sid}")
                st.rerun()

    with tab_sched:
        refresh_pull_statuses(prog)
        _save_program(prog)
        dfp = pulls_to_dataframe(prog)
        if len(dfp) == 0:
            empty_state("No pulls", "Generate from Design or create a study schedule.", "📅")
        else:
            filt = st.multiselect(
                "Filter status",
                ["Upcoming", "Due", "Overdue", "Completed"],
                default=["Due", "Overdue", "Upcoming"],
            )
            view = dfp[dfp["status"].isin(filt)] if filt else dfp
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Pulls CSV",
                view.to_csv(index=False).encode("utf-8"),
                file_name="program_pulls.csv",
                mime="text/csv",
            )

    with tab_res:
        open_pulls = [p for p in prog.pulls if p.status != "Completed"]
        if not open_pulls:
            st.info("All pulls completed (or none loaded).")
        else:
            labels = {
                p.pull_id: f"{p.pull_id} · {p.batch} · t={p.time_months} · {p.condition} · {p.status}"
                for p in open_pulls
            }
            choice = st.selectbox("Select pull", list(labels.keys()), format_func=lambda k: labels[k])
            attr = st.text_input("Attribute / Chỉ tiêu", value="Assay")
            val = st.number_input("Result / Response", value=99.0, step=0.01)
            note = st.text_input("Notes", value="")
            if st.button("Save result", type="primary"):
                mark_pull_completed(prog, choice, float(val), attribute=attr, notes=note)
                _save_program(prog)
                st.success("Saved.")
                st.rerun()

    with tab_feed:
        study_ids = [s.study_id for s in prog.studies]
        if not study_ids:
            empty_state("No studies", "Create a study first.", "📦")
        else:
            sid = st.selectbox("Study for analysis feed", study_ids)
            conds = sorted({p.condition for p in prog.pulls if p.study_id == sid})
            cond = st.selectbox("Condition filter (optional)", ["(all)"] + conds)
            df_an = results_to_analysis_frame(
                prog,
                sid,
                condition=None if cond == "(all)" else cond,
            )
            if len(df_an) == 0:
                st.warning("No completed results with numeric response for this study.")
            else:
                # annotate condition_type
                df_an = df_an.copy()
                df_an["condition_type"] = df_an["condition"].map(classify_condition_role)
                st.dataframe(df_an, use_container_width=True, hide_index=True)
                if st.button("Load into Data step (session df)", type="primary"):
                    # Store analysis-ready frame
                    keep_cols = ["batch", "time", "response", "condition", "attribute", "condition_type"]
                    out = df_an[[c for c in keep_cols if c in df_an.columns]].copy()
                    st.session_state["df"] = out
                    st.session_state["data_label"] = f"Program feed: {sid}"
                    st.session_state["analysis"] = None
                    # Prefer long-term condition if present
                    lt = out[out["condition_type"] == "long-term"]["condition"].unique().tolist()
                    if lt:
                        st.session_state["condition"] = lt[0]
                    st.success("Loaded into session — go to 3 · Dữ liệu / 4 · Mô hình.")
                roles = df_an["condition_type"].value_counts().to_dict()
                st.caption(
                    "Method tags: "
                    + ", ".join(f"{method_tag_for_role(r)}={n}" for r, n in roles.items())
                )

    callout(
        "Kết quả đã hoàn thành có thể nạp vào bước <b>Dữ liệu</b> để chạy Q1E (LT-Q1E). "
        "Accelerated = LHCT (supportive).",
        kind="info",
    )
