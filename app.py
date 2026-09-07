"""
ICH-STABILITY-GROK
Pharmaceutical shelf-life (stability) prediction — Streamlit UI
Vietnamese labels + English technical terms
Aligned with ICH Q1A(R2) / Q1D / Q1E (supportive analysis only)

Supports Study Design (Q1D), Stability Program, and single-/multi-attribute
Q1E shelf-life analysis. Overall product shelf life = minimum of attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.arrhenius import (
    default_rates_editor_frame,
    estimate_rates_by_condition,
    fit_arrhenius,
    normalize_rates_table,
    project_shelf_life_zero_order,
)
from src.attributes import (
    AttributeConfig,
    analyze_attributes,
    default_config_for_attribute,
)
from src.data_io import (
    detect_data_shape,
    ensure_attribute_column,
    ensure_condition_type,
    filter_condition,
    infer_condition_type,
    list_attributes,
    load_table,
    make_empty_entry_frame,
    method_tag_for_condition_type,
    prepare_dataframe,
    split_by_attribute,
    suggest_wide_response_columns,
    validate_for_analysis,
    wide_to_long,
)
from src.study_design import classify_condition_role, method_tag_for_role, sampling_matrix_to_csv
from src.ui_design_program import render_stability_program_page, render_study_design_page
from src.plots import plot_arrhenius, plot_multibatch_overlay, plot_residuals, plot_stability
from src.report import build_html_report, dataframe_to_html, export_pdf_bytes
from src.ui_styles import (
    brand_header,
    callout,
    empty_state,
    inject_styles,
    metric_cards,
    pill,
    section_header,
)

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="ICH-STABILITY-GROK",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "df": None,
    "data_label": None,
    "default_spec": 90.0,
    "default_dir": "decreasing",
    "condition": "(Tất cả / All)",
    "spec_limit": 90.0,
    "direction": "decreasing",
    "transform": "none",
    "alpha": 0.05,
    "alpha_pool": 0.25,
    "t_max": 120.0,
    "pool_mode": "Tự động (ANCOVA)",
    "attr_configs": {},  # name -> dict(direction, spec_limit, transform)
    "selected_attributes": [],
    "analysis": None,
    "study_design": None,
    "sampling_matrix": None,
    "stability_program": None,
}


def _ensure_state() -> None:
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ensure_state()


def _load_sample(name: str) -> pd.DataFrame:
    return prepare_dataframe(pd.read_csv(DATA_DIR / name))


def _fmt_shelf(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _show_figure(fig, width: str = "stretch") -> None:
    """Display matplotlib figure with consistent sizing."""
    try:
        st.pyplot(fig, width=width, clear_figure=False)
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def _is_multi_attr(df: Optional[pd.DataFrame]) -> bool:
    if df is None or len(df) == 0:
        return False
    attrs = list_attributes(df)
    return len(attrs) >= 2


def _attribute_names(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or len(df) == 0:
        return []
    attrs = list_attributes(df)
    if attrs:
        return attrs
    return ["Response"]


def _init_attr_configs(names: List[str], force: bool = False) -> None:
    """Ensure session attr_configs has an entry for each attribute name."""
    cfgs: Dict[str, dict] = dict(st.session_state.get("attr_configs") or {})
    for name in names:
        if force or name not in cfgs:
            d = default_config_for_attribute(name)
            # For single legacy "Response", keep session defaults
            if name in ("Response", "(single)"):
                d = AttributeConfig(
                    name=name,
                    direction=st.session_state.get("default_dir", "decreasing"),
                    spec_limit=float(st.session_state.get("default_spec", 90.0)),
                    transform="none",
                )
            cfgs[name] = {
                "direction": d.direction,
                "spec_limit": float(d.spec_limit),
                "transform": d.transform,
            }
    # Drop configs for attributes no longer present
    st.session_state["attr_configs"] = {k: v for k, v in cfgs.items() if k in names}
    sel = st.session_state.get("selected_attributes") or []
    if not sel or force:
        st.session_state["selected_attributes"] = list(names)
    else:
        st.session_state["selected_attributes"] = [a for a in sel if a in names] or list(names)


def _configs_from_session(names: List[str]) -> List[AttributeConfig]:
    cfgs = st.session_state.get("attr_configs") or {}
    out: List[AttributeConfig] = []
    for name in names:
        c = cfgs.get(name) or default_config_for_attribute(name).__dict__
        out.append(
            AttributeConfig(
                name=name,
                direction=c.get("direction", "decreasing"),
                spec_limit=float(c.get("spec_limit", 90.0)),
                transform=c.get("transform", "none"),
            )
        )
    return out


def _build_attr_figures(attr_res, alpha: float) -> List:
    """Create matplotlib figures for one AttributeResult."""
    figures = []
    df_an = attr_res.df
    if df_an is None or len(df_an) == 0:
        return figures
    label = attr_res.attribute

    if attr_res.kind == "single" and attr_res.reg is not None and attr_res.shelf is not None:
        fig = plot_stability(
            df_an,
            attr_res.reg,
            attr_res.shelf,
            title=f"{label} · Batch {attr_res.batch_name} · Shelf life",
            alpha_band=alpha,
        )
        fig_r = plot_residuals(attr_res.reg, title=f"Residuals — {label}")
        figures.extend([fig, fig_r])
    elif attr_res.kind == "multi" and attr_res.mb is not None:
        mb = attr_res.mb
        if mb.pooled_reg is not None and mb.reported_mode == "pooled" and mb.pooled_shelf is not None:
            fig = plot_stability(
                df_an,
                mb.pooled_reg,
                mb.pooled_shelf,
                title=f"{label} · Pooled",
                alpha_band=alpha,
            )
            figures.append(fig)
            figures.append(plot_residuals(mb.pooled_reg, title=f"Residuals — {label} pooled"))
        for b, (reg, sl) in mb.per_batch.items():
            fig_b = plot_stability(
                df_an[df_an["batch"].astype(str) == b],
                reg,
                sl,
                title=f"{label} · Batch {b}",
                batch_col=False,
                alpha_band=alpha,
            )
            figures.append(fig_b)
        fig_o = plot_multibatch_overlay(
            df_an,
            mb.per_batch,
            mb.pooled_shelf,
            spec_limit=attr_res.config.spec_limit,
            title=f"{label} · Multi-batch overlay",
        )
        figures.append(fig_o)
    return figures


def _render_single_attr_detail(attr_res, alpha: float, figures: List) -> None:
    """Render detail block for one attribute (used in Results expanders)."""
    st.write(attr_res.message)
    if attr_res.kind == "single" and attr_res.reg is not None:
        left, right = st.columns(2)
        with left:
            st.markdown("**Hệ số hồi quy / Coefficients**")
            st.dataframe(attr_res.reg.summary_table(), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**ANOVA**")
            st.dataframe(attr_res.reg.anova, use_container_width=True, hide_index=True)
        st.caption(
            f"σ = **{attr_res.reg.sigma:.6g}** · n = {attr_res.reg.n} · df = {attr_res.reg.df_resid}"
        )
    elif attr_res.kind == "multi" and attr_res.mb is not None:
        mb = attr_res.mb
        pool = mb.poolability
        st.markdown(
            f"Poolability: **{pool.recommendation}** · mode = **{mb.reported_mode}** · "
            f"α_pool context from session"
        )
        if pool.anova_full is not None:
            st.dataframe(pool.anova_full, use_container_width=True, hide_index=True)
        rows = []
        for b, (reg, sl) in mb.per_batch.items():
            rows.append(
                {
                    "Batch": b,
                    "β0": reg.beta0,
                    "β1": reg.beta1,
                    "R²": reg.r_squared,
                    "Shelf life (mo)": sl.shelf_life,
                    "Direction": sl.direction,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if mb.pooled_reg is not None and mb.reported_mode == "pooled":
            st.markdown("**Mô hình gộp / Pooled**")
            st.dataframe(mb.pooled_reg.summary_table(), use_container_width=True, hide_index=True)

    for fig in figures:
        _show_figure(fig)


# ---------------------------------------------------------------------------
# Sidebar — step navigator + status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ICH-STABILITY-GROK")
    st.caption("Shelf-life / Stability · ICH Q1A / Q1D / Q1E")
    st.markdown("---")

    STEPS = [
        "1 · Thiết kế / Study Design",
        "2 · Chương trình / Stability Program",
        "3 · Dữ liệu / Data",
        "4 · Mô hình / Model",
        "5 · Gộp lô / Pooling",
        "6 · Kết quả / Results",
        "7 · Báo cáo / Report",
        "Arrhenius (exploratory)",
        "Giới thiệu / About",
    ]
    step = st.radio("Quy trình / Workflow", STEPS, label_visibility="collapsed")

    st.markdown("---")
    df_ok = st.session_state["df"] is not None and len(st.session_state["df"]) > 0
    has_run = st.session_state["analysis"] is not None
    status_bits = []
    status_bits.append(
        pill("Có dữ liệu" if df_ok else "Chưa có dữ liệu", "ok" if df_ok else "warn")
    )
    status_bits.append(
        pill("Đã chạy" if has_run else "Chưa phân tích", "ok" if has_run else "info")
    )
    st.markdown(" ".join(status_bits), unsafe_allow_html=True)
    if df_ok:
        dtmp = st.session_state["df"]
        attrs = list_attributes(dtmp)
        n_attr = len(attrs) if attrs else 1
        st.caption(
            f"{len(dtmp)} điểm · {dtmp['batch'].nunique()} lô · "
            f"{dtmp['condition'].nunique()} điều kiện · {n_attr} chỉ tiêu"
        )
        if attrs:
            st.caption("Chỉ tiêu: " + ", ".join(attrs))
        if st.session_state.get("data_label"):
            st.caption(st.session_state["data_label"])

    st.markdown("---")
    callout(
        "⚠️ Hỗ trợ phân tích thống kê — <b>không</b> phải hệ thống chứng nhận "
        "cho hồ sơ đăng ký.<br/>"
        "Supportive tool — <b>NOT</b> a certified regulatory system.",
        kind="warn",
    )


# ===========================================================================
# ABOUT
# ===========================================================================
if step.startswith("Giới thiệu"):
    brand_header("Giới thiệu / About & Methods", "ICH Q1A(R2) · Q1D · Q1E · supportive analysis")
    section_header(None, "Mục đích", "Purpose")
    st.markdown(
        """
Ứng dụng ước tính **thời hạn bảo quản (shelf life)** từ dữ liệu ổn định
theo tinh thần **ICH Q1A(R2)** và **ICH Q1E**, tương tự phân tích
Minitab *Stability / Shelf Life*. Hỗ trợ **nhiều chỉ tiêu** (assay, impurity, …)
trong một phiên — shelf life sản phẩm = **min** các chỉ tiêu.
"""
    )
    section_header(None, "Phương pháp", "Methods")
    st.markdown(
        r"""
1. **Hồi quy tuyến tính (OLS):** \(y = \beta_0 + \beta_1 t\)
   (tuỳ chọn biến đổi log / sqrt).
2. **Shelf life:** thời điểm biên tin cậy **một phía (1−α)** của *mean response*
   giao với giới hạn chất lượng (spec):
   - Assay giảm → **lower bound** gặp lower spec
   - Impurity tăng → **upper bound** gặp upper spec
3. **Nhiều lô (multi-batch):** kiểm định ANCOVA đồng nhất slope / intercept
   với α poolability mặc định **0.25** (có thể chỉnh). Nếu không gộp được,
   báo cáo **shelf life nhỏ nhất** giữa các lô (conservative).
4. **Nhiều chỉ tiêu (multi-attribute):** phân tích độc lập từng chỉ tiêu;
   **overall product shelf life = min** các shelf life chỉ tiêu.
5. **Study Design (Q1D):** Full / Bracketing / Matrixing sampling plan (supportive planning).
6. **Stability Program:** multi-product catalog, due reminders, results feed into Q1E.
7. **Condition roles:** Long-term → **LT-Q1E** (primary); Accelerated → **LHCT** (supportive);
   Arrhenius → **Arrhenius-supportive**.
8. **Arrhenius (SUPPORTIVE):** fit ln(k)=ln(A)−Ea/(R·T) từ bảng T+k hoặc
   |slope| proxy; chọn R (kcal/kJ); dự đoán k @ T dài hạn; shelf life ngoại suy
   (bậc không) chỉ mang tính thăm dò — **không** thay thế ICH Q1A/Q1E dài hạn.
"""
    )
    section_header(None, "Tài liệu tham chiếu", "References")
    st.markdown(
        """
- ICH Q1A(R2) *Stability Testing of New Drug Substances and Products*
- ICH Q1D *Bracketing and Matrixing Designs*
- ICH Q1E *Evaluation of Stability Data*
"""
    )
    section_header(None, "Tuyên bố", "Disclaimer")
    callout(
        "Phần mềm này <b>không</b> tuyên bố chứng nhận quy định (regulatory certification), "
        "<b>không</b> thay thế đánh giá chuyên gia QA/RA, và chỉ mang tính hỗ trợ phân tích.",
        kind="danger",
    )
    st.stop()


# ===========================================================================
# ARRHENIUS
# ===========================================================================
if step.startswith("Arrhenius"):
    brand_header(
        "Arrhenius — Exploratory / Thăm dò",
        "SUPPORTIVE only · không thay thế đánh giá dài hạn ICH Q1A/Q1E",
    )
    callout(
        "<b>Method tag: Arrhenius-supportive</b> — SUPPORTIVE / HỖ TRỢ ONLY — "
        "Không thay thế nghiên cứu ổn định dài hạn ICH Q1A(R2) / Q1E. "
        "Mô hình: ln(k) = ln(A) − Ea/(R·T). "
        "k có thể nhập trực tiếp hoặc ước lượng ≈ |độ dốc OLS| (proxy bậc không). "
        "Kết quả mang tính thăm dò / exploratory.",
        kind="warn",
    )
    with st.expander("Giả định / Assumptions (đọc trước)", expanded=False):
        st.markdown(
            """
- **Rate proxy:** khi suy từ dữ liệu ổn định, k ≈ |OLS slope| của response theo thời gian
  (động học **bậc không** trên thang mô hình). Không phải lựa chọn mô hình động học đầy đủ.
- **Bậc một:** module này không fit first-order; có thể nhập k đã biến đổi bên ngoài nếu cần.
- Bỏ qua độ ẩm / bao bì / yếu tố khác ngoài nhiệt độ.
- Shelf life dự phóng từ k ngoại suy dùng **giao trung bình bậc không** (intercept → spec) —
  **không** phải biên tin cậy một phía ICH Q1E.
- Kết quả **SUPPORTIVE / exploratory only**.
            """
        )

    section_header("A", "Hằng số khí & nhiệt độ dự đoán", "Gas constant & prediction T")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        r_choice = st.selectbox(
            "Hằng số khí R / Gas constant R",
            ["kcal/(mol·K)", "kJ/(mol·K)"],
            index=0,
            help="Chọn đơn vị R → Ea báo cáo cùng hệ (kcal/mol hoặc kJ/mol).",
        )
    r_unit = "kcal" if r_choice.startswith("kcal") else "kJ"
    with c2:
        predict_temp = st.number_input(
            "Nhiệt độ dài hạn dự đoán / Long-term T (°C)",
            min_value=-20.0,
            max_value=80.0,
            value=25.0,
            step=1.0,
            help="Ví dụ 25°C (long-term) hoặc 30°C theo vùng khí hậu.",
        )
    with c3:
        st.caption(f"R unit key: **{r_unit}** · predict @ **{predict_temp:g}°C**")

    section_header("B", "Nguồn tốc độ k", "Rate source")
    input_mode = st.radio(
        "Cách nhập k / How to provide rates",
        [
            "Nhập / tải bảng T + k (user rates)",
            "Ước lượng |slope| từ dữ liệu đa điều kiện (derived)",
        ],
        horizontal=True,
        key="arr_input_mode",
    )

    rates = None
    intercept_for_proj = None

    if input_mode.startswith("Nhập"):
        src_k = st.radio(
            "Nguồn bảng k / Rates table source",
            [
                "Sample Arrhenius rates",
                "Nhập tay / Manual table",
                "Upload CSV (temp_c, k)",
            ],
            horizontal=True,
            key="arr_k_src",
        )
        rates_raw = None
        if src_k.startswith("Sample"):
            rates_raw = pd.read_csv(DATA_DIR / "sample_arrhenius_rates.csv")
            st.caption("Sample: `data/sample_arrhenius_rates.csv` (toy Ea ≈ 18 kcal/mol).")
        elif src_k.startswith("Upload"):
            upk = st.file_uploader(
                "Upload CSV tốc độ / rates (cột temp_c + k)",
                type=["csv", "xlsx", "xls"],
                key="arr_rates_up",
            )
            if upk is not None:
                try:
                    rates_raw = load_table(upk)
                except Exception as exc:
                    st.error(str(exc))
            else:
                empty_state("Chưa tải file k", "Upload a rates CSV with temp_c and k.", "📂")
        else:
            if "arr_editor_df" not in st.session_state:
                st.session_state["arr_editor_df"] = default_rates_editor_frame()
            st.caption("Chỉnh bảng nhiệt độ (°C) và k — có thể thêm/xóa dòng.")
            edited = st.data_editor(
                st.session_state["arr_editor_df"],
                num_rows="dynamic",
                use_container_width=True,
                key="arr_rates_editor",
            )
            rates_raw = edited
            st.session_state["arr_editor_df"] = edited

        if rates_raw is not None and len(rates_raw) > 0:
            try:
                rates = normalize_rates_table(rates_raw)
                with st.expander("Bảng k đã chuẩn hóa / Normalized rates", expanded=True):
                    st.dataframe(rates, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(str(exc))

    else:
        # Derived from multi-condition stability data
        callout(
            "Giả định: k ≈ |slope| OLS của response vs time (proxy bậc không / zero-order). "
            "Cần ≥2 điều kiện nhiệt độ parse được từ cột condition (vd. 25C, 40C/75%RH).",
            kind="info",
        )
        src = st.radio(
            "Nguồn dữ liệu ổn định / Stability data source",
            [
                "Sample Arrhenius conditions",
                "Sample multi-batch",
                "Upload CSV/Excel",
                "Dùng dữ liệu đang tải / Use loaded data",
            ],
            horizontal=True,
            key="arr_stab_src",
        )
        df_arr = None
        if src.startswith("Sample Arrhenius"):
            df_arr = _load_sample("sample_arrhenius_conditions.csv")
        elif src.startswith("Sample multi"):
            df_arr = _load_sample("sample_multibatch.csv")
        elif src.startswith("Upload"):
            up = st.file_uploader(
                "Upload CSV / Excel ổn định",
                type=["csv", "xlsx", "xls"],
                key="arr_up",
            )
            if up is not None:
                try:
                    raw = load_table(up)
                    shape = detect_data_shape(raw)
                    if shape == "wide":
                        cands = suggest_wide_response_columns(raw)
                        pick = st.multiselect(
                            "Chọn cột response / Response columns",
                            cands,
                            default=cands[:1],
                            key="arr_wide_pick",
                        )
                        if pick:
                            df_arr = wide_to_long(raw, pick)
                    else:
                        df_arr = prepare_dataframe(raw)
                except Exception as exc:
                    st.error(str(exc))
            else:
                empty_state("Chưa tải file", "Upload a CSV or Excel file to begin.", "📂")
        else:
            if st.session_state["df"] is not None and len(st.session_state["df"]) > 0:
                df_arr = st.session_state["df"]
            else:
                empty_state(
                    "Chưa có dữ liệu trong phiên",
                    "Load data in step 1 · Dữ liệu, or choose Sample / Upload here.",
                    "📋",
                )

        if df_arr is not None and len(df_arr) > 0:
            attrs = list_attributes(df_arr)
            if attrs:
                atr = st.selectbox("Chỉ tiêu / Attribute (Arrhenius)", attrs, key="arr_attr")
                df_arr = df_arr[df_arr["attribute"].astype(str) == atr].drop(
                    columns=["attribute"]
                )
            with st.expander("Xem dữ liệu / Preview", expanded=False):
                st.dataframe(df_arr, use_container_width=True, hide_index=True)
            section_header("B2", "Tốc độ theo điều kiện", "Rates per condition (|slope|)")
            rates = estimate_rates_by_condition(df_arr)
            st.dataframe(rates, use_container_width=True, hide_index=True)
            if rates is not None and len(rates) and "intercept" in rates.columns:
                # Prefer long-term-ish intercept (closest to predict_temp) for projection default
                valid_r = rates.dropna(subset=["temp_c"])
                if len(valid_r):
                    idx = (valid_r["temp_c"] - float(predict_temp)).abs().idxmin()
                    intercept_for_proj = float(valid_r.loc[idx, "intercept"])

    section_header("C", "Kết quả Arrhenius", "Arrhenius results")
    if rates is None or len(rates) == 0:
        empty_state(
            "Chưa có bảng tốc độ",
            "Enter or derive k at ≥2 temperatures to fit Arrhenius.",
            "🌡️",
        )
    else:
        arr = fit_arrhenius(rates, r_unit=r_unit, predict_temp_c=float(predict_temp))
        st.info(arr.message)
        if arr.valid:
            metric_cards(
                [
                    {
                        "label": f"Ea ({arr.ea_unit})",
                        "value": f"{arr.ea:.2f}",
                        "tone": "ok",
                    },
                    {
                        "label": "ln(A)",
                        "value": f"{arr.ln_A:.4f}",
                        "tone": "neutral",
                    },
                    {
                        "label": "R²",
                        "value": f"{arr.r_squared:.4f}",
                        "tone": "ok",
                    },
                    {
                        "label": f"k dự đoán @ {arr.predict_temp_c:g}°C",
                        "value": f"{arr.predicted_k:.6g}",
                        "hint": "Predicted rate",
                        "tone": "info",
                    },
                ]
            )
            st.caption(
                f"R = {arr.r_gas:.6g} {arr.r_label} · "
                f"Ea cũng ≈ {arr.ea_kcal_mol:.2f} kcal/mol / {arr.ea_kj_mol:.2f} kJ/mol · "
                f"k@25°C = {arr.predicted_rate_at_25:.6g}"
            )
            fig = plot_arrhenius(arr, rates)
            _show_figure(fig)
            plt.close(fig)

            section_header(
                "D",
                "Dự phóng shelf life (tuỳ chọn)",
                "Optional shelf-life projection",
            )
            callout(
                "CẢNH BÁO MẠNH / STRONG DISCLAIMER: đây là giao trung bình bậc không từ k ngoại suy — "
                "KHÔNG thay thế shelf life theo biên tin cậy một phía ICH Q1E từ dữ liệu dài hạn.",
                kind="warn",
            )
            do_proj = st.checkbox(
                "Ước lượng shelf life thăm dò tại T dự đoán / Exploratory shelf life at predicted T",
                value=False,
                key="arr_do_proj",
            )
            if do_proj:
                # Prefer session attribute/spec settings when available
                def_spec = float(st.session_state.get("spec_limit", 90.0))
                def_dir = st.session_state.get("direction", "decreasing")
                cfgs = st.session_state.get("attr_configs") or {}
                if cfgs:
                    # use first selected attribute config if present
                    sel = st.session_state.get("selected_attributes") or list(cfgs.keys())
                    if sel and sel[0] in cfgs:
                        def_spec = float(cfgs[sel[0]].get("spec_limit", def_spec))
                        def_dir = cfgs[sel[0]].get("direction", def_dir)

                p1, p2, p3 = st.columns(3)
                with p1:
                    proj_intercept = st.number_input(
                        "Hệ số chặn / Intercept (t=0)",
                        value=float(
                            intercept_for_proj
                            if intercept_for_proj is not None
                            else (100.0 if str(def_dir).startswith("dec") else 0.05)
                        ),
                        format="%.4f",
                        key="arr_proj_intercept",
                    )
                with p2:
                    proj_spec = st.number_input(
                        "Giới hạn spec / Spec limit",
                        value=float(def_spec),
                        format="%.4f",
                        key="arr_proj_spec",
                    )
                with p3:
                    dir_opts = ["decreasing (assay ↓)", "increasing (impurity ↑)"]
                    dir_idx = 0 if str(def_dir).startswith("dec") else 1
                    proj_dir_label = st.selectbox(
                        "Hướng / Direction",
                        dir_opts,
                        index=dir_idx,
                        key="arr_proj_dir",
                    )
                proj_dir = "decreasing" if proj_dir_label.startswith("dec") else "increasing"
                proj = project_shelf_life_zero_order(
                    rate_k=float(arr.predicted_k),
                    intercept=float(proj_intercept),
                    spec_limit=float(proj_spec),
                    direction=proj_dir,
                    temp_c=float(arr.predict_temp_c),
                )
                st.warning(proj.message)
                if proj.valid and proj.shelf_life is not None:
                    metric_cards(
                        [
                            {
                                "label": f"Shelf life thăm dò @ {proj.temp_c:g}°C (tháng)",
                                "value": f"{proj.shelf_life:.2f}",
                                "hint": "Zero-order mean crossing",
                                "tone": "warn",
                            },
                            {
                                "label": "k dùng để dự phóng",
                                "value": f"{proj.rate_k:.6g}",
                                "tone": "neutral",
                            },
                        ]
                    )
        else:
            st.warning(arr.message)

    st.stop()


# ===========================================================================
# Shared brand for analysis steps
# ===========================================================================
brand_header(
    "ICH Stability Platform / Nền tảng ổn định",
    "Q1D Design · Program · Q1E shelf-life · method tags LT-Q1E / LHCT / Arrhenius-supportive",
)


# ===========================================================================
# STEP 1 — STUDY DESIGN (Q1D)
# ===========================================================================
if step.startswith("1 ·"):
    render_study_design_page()
    st.stop()


# ===========================================================================
# STEP 2 — STABILITY PROGRAM
# ===========================================================================
elif step.startswith("2 ·"):
    render_stability_program_page()
    st.stop()


# ===========================================================================
# STEP 3 — DATA
# ===========================================================================
elif step.startswith("3 ·"):
    section_header("3", "Dữ liệu", "Data")
    data_mode = st.radio(
        "Nguồn / Source",
        [
            "Sample: assay giảm (decreasing)",
            "Sample: impurity tăng (increasing)",
            "Sample: multi-batch",
            "Sample: đa chỉ tiêu (multi-attribute)",
            "Upload CSV / Excel",
            "Nhập tay / Manual entry",
        ],
        horizontal=True,
        key="data_mode_radio",
    )

    df = None
    label = data_mode
    default_spec = 90.0
    default_dir = "decreasing"

    if data_mode.startswith("Sample: assay"):
        df = _load_sample("sample_assay_decreasing.csv")
        default_spec, default_dir = 90.0, "decreasing"
    elif data_mode.startswith("Sample: impurity"):
        df = _load_sample("sample_impurity_increasing.csv")
        default_spec, default_dir = 0.5, "increasing"
    elif data_mode.startswith("Sample: multi-batch"):
        df = _load_sample("sample_multibatch.csv")
        default_spec, default_dir = 90.0, "decreasing"
    elif data_mode.startswith("Sample: đa") or "multi-attribute" in data_mode:
        df = _load_sample("sample_multi_attribute.csv")
        default_spec, default_dir = 90.0, "decreasing"
    elif data_mode.startswith("Upload"):
        up = st.file_uploader(
            "Chọn file CSV hoặc Excel / Choose CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="main_upload",
        )
        st.caption(
            "Định dạng: long (`batch,time,response,condition` ± `attribute`/`chỉ tiêu`) "
            "hoặc wide (nhiều cột response — sẽ được ánh xạ)."
        )
        if up is not None:
            try:
                raw = load_table(up)
                shape = detect_data_shape(raw)
                st.caption(f"Phát hiện dạng dữ liệu / Detected shape: **{shape}**")
                if shape == "wide":
                    cands = suggest_wide_response_columns(raw)
                    st.markdown("**Ánh xạ cột response → chỉ tiêu / Map response columns → attributes**")
                    selected = st.multiselect(
                        "Cột đo lường / Measurement columns",
                        cands,
                        default=cands,
                        key="wide_cols",
                    )
                    col_map = {}
                    if selected:
                        for c in selected:
                            col_map[c] = st.text_input(
                                f"Tên chỉ tiêu cho `{c}` / Attribute name",
                                value=str(c).title(),
                                key=f"wide_name_{c}",
                            )
                    if selected and st.button("Áp dụng ánh xạ / Apply mapping", key="wide_apply"):
                        df = wide_to_long(raw, col_map)
                        label = f"Upload (wide): {up.name}"
                    elif selected:
                        # Auto-apply for convenience when names filled
                        df = wide_to_long(raw, col_map)
                        label = f"Upload (wide): {up.name}"
                else:
                    df = prepare_dataframe(raw)
                    label = f"Upload: {up.name}"
            except Exception as exc:
                st.error(str(exc))
        else:
            empty_state(
                "Chưa chọn file",
                "Upload CSV/Excel with columns: batch, time, response, condition "
                "(+ optional attribute) — or wide measurement columns.",
                "📂",
            )
    else:
        entry_mode = st.radio(
            "Kiểu nhập / Entry type",
            ["Một chỉ tiêu / Single", "Nhiều chỉ tiêu / Multi-attribute"],
            horizontal=True,
            key="manual_entry_mode",
        )
        multi = entry_mode.startswith("Nhiều")
        st.caption(
            "Cột: **batch**, **time**, **response**, **condition**"
            + (" + **attribute**" if multi else "")
        )
        edited = st.data_editor(
            make_empty_entry_frame(8 if not multi else 12, multi_attribute=multi),
            num_rows="dynamic",
            use_container_width=True,
            key="manual_editor",
        )
        try:
            df = prepare_dataframe(edited)
            label = "Manual entry" + (" (multi-attribute)" if multi else "")
        except Exception as exc:
            st.warning(str(exc))
            df = None

    if df is not None and len(df) > 0:
        prev_label = st.session_state.get("data_label")
        df = ensure_condition_type(df)
        st.session_state["df"] = df
        st.session_state["data_label"] = label
        names = _attribute_names(df)
        if prev_label != label:
            st.session_state["default_spec"] = default_spec
            st.session_state["default_dir"] = default_dir
            st.session_state["spec_limit"] = float(default_spec)
            st.session_state["direction"] = default_dir
            st.session_state["analysis"] = None
            _init_attr_configs(names, force=True)
        else:
            _init_attr_configs(names, force=False)

        attrs = list_attributes(df)
        ok_msgs = []
        if attrs:
            parts = split_by_attribute(df)
            all_ok = True
            for a, sub in parts.items():
                ok, msg = validate_for_analysis(sub)
                if not ok:
                    all_ok = False
                    ok_msgs.append(f"{a}: {msg}")
            ok = all_ok
            msg = "; ".join(ok_msgs) if ok_msgs else "OK"
        else:
            ok, msg = validate_for_analysis(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Số điểm / Points", len(df))
        c2.metric("Số lô / Batches", int(df["batch"].nunique()))
        c3.metric("Điều kiện / Conditions", int(df["condition"].nunique()))
        c4.metric("Time max (mo)", f"{df['time'].max():.1f}")
        c5.metric("Chỉ tiêu / Attributes", len(attrs) if attrs else 1)
        if not ok:
            st.error(msg)
        else:
            st.success("Dữ liệu hợp lệ / Data ready for analysis.")
            if attrs:
                st.info("Chỉ tiêu phát hiện / Attributes: **" + "**, **".join(attrs) + "**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        callout(
            "Tiếp theo: chuyển sang <b>4 · Mô hình / Model</b> để thiết lập spec, direction, transform "
            "(theo từng chỉ tiêu nếu đa chỉ tiêu).",
            kind="info",
        )
    elif not data_mode.startswith("Upload"):
        empty_state("Chưa có dữ liệu", "No data loaded yet. Choose a sample or enter values.", "📋")
        st.session_state["df"] = None


# ===========================================================================
# STEP 4 — MODEL
# ===========================================================================
elif step.startswith("4 ·"):
    section_header("4", "Mô hình", "Model settings")
    df = st.session_state["df"]
    if df is None or len(df) == 0:
        empty_state(
            "Chưa có dữ liệu",
            "Go to step 3 · Dữ liệu and load a sample, upload, or enter data.",
            "📋",
        )
        st.stop()

    names = _attribute_names(df)
    _init_attr_configs(names)
    multi = len(names) >= 2 or ("attribute" in df.columns and len(list_attributes(df)) >= 1)

    conditions = ["(Tất cả / All)"] + sorted(df["condition"].unique().tolist())
    cur_cond = st.session_state.get("condition", conditions[0])
    if cur_cond not in conditions:
        cur_cond = conditions[0]

    st.markdown("**Điều kiện chung / Shared condition & CI**")
    c1, c2, c3 = st.columns(3)
    with c1:
        condition = st.selectbox(
            "Điều kiện bảo quản / Storage condition",
            conditions,
            index=conditions.index(cur_cond),
        )
    with c2:
        alpha = st.number_input(
            "α (shelf-life CI)",
            min_value=0.001,
            max_value=0.2,
            value=float(st.session_state.get("alpha", 0.05)),
            step=0.01,
            help="One-sided confidence level = 1−α for the mean bound.",
        )
    with c3:
        t_max = st.number_input(
            "t_max tìm nghiệm (months)",
            min_value=12.0,
            max_value=240.0,
            value=float(st.session_state.get("t_max", 120.0)),
        )

    st.session_state["condition"] = condition
    st.session_state["alpha"] = float(alpha)
    st.session_state["t_max"] = float(t_max)
    _role = classify_condition_role(str(condition))
    _tag = method_tag_for_role(_role)
    if condition == "(Tất cả / All)":
        callout(
            "Đang chọn <b>Tất cả</b> điều kiện — khuyến nghị lọc <b>long-term</b> "
            "cho quyết định shelf-life LT-Q1E. Accelerated = LHCT (supportive).",
            kind="warn",
        )
    elif _tag == "LT-Q1E":
        callout(
            f"<b>Method tag: {_tag}</b> — Long-term primary Q1E path · role <code>{_role}</code>.",
            kind="ok",
        )
    elif _tag == "LHCT":
        callout(
            f"<b>Method tag: {_tag}</b> — Accelerated supportive only · role <code>{_role}</code>.",
            kind="warn",
        )
    else:
        callout(
            f"<b>Method tag: {_tag}</b> · role <code>{_role}</code>.",
            kind="info",
        )

    # Attribute selection + per-attribute settings
    if multi and len(names) >= 1:
        section_header(None, "Thiết lập theo chỉ tiêu", "Per-attribute settings")
        selected = st.multiselect(
            "Chọn chỉ tiêu phân tích / Attributes to analyse",
            names,
            default=st.session_state.get("selected_attributes") or names,
            key="attr_select_widget",
        )
        if not selected:
            st.warning("Chọn ít nhất một chỉ tiêu / Select at least one attribute.")
            selected = list(names)
        st.session_state["selected_attributes"] = selected

        cfgs = dict(st.session_state.get("attr_configs") or {})
        dir_opts = ["decreasing", "increasing", "auto"]
        tf_opts = ["none", "log", "sqrt"]

        for name in selected:
            cur = cfgs.get(name) or default_config_for_attribute(name).__dict__
            with st.expander(f"⚙️ {name}", expanded=len(selected) <= 3):
                a1, a2, a3 = st.columns(3)
                with a1:
                    d_cur = cur.get("direction", "decreasing")
                    if d_cur not in dir_opts:
                        d_cur = "decreasing"
                    direction = st.selectbox(
                        "Hướng / Direction",
                        dir_opts,
                        index=dir_opts.index(d_cur),
                        key=f"dir_{name}",
                        help="decreasing = assay↓; increasing = impurity↑",
                    )
                with a2:
                    spec_limit = st.number_input(
                        "Giới hạn spec / Spec limit",
                        value=float(cur.get("spec_limit", 90.0)),
                        format="%.6f",
                        key=f"spec_{name}",
                    )
                with a3:
                    tf_cur = cur.get("transform", "none")
                    transform = st.selectbox(
                        "Biến đổi / Transform",
                        tf_opts,
                        index=tf_opts.index(tf_cur) if tf_cur in tf_opts else 0,
                        key=f"tf_{name}",
                    )
                cfgs[name] = {
                    "direction": direction,
                    "spec_limit": float(spec_limit),
                    "transform": transform,
                }
        st.session_state["attr_configs"] = cfgs

        # Keep legacy single keys in sync with first selected (backward compat)
        first = selected[0]
        st.session_state["spec_limit"] = float(cfgs[first]["spec_limit"])
        st.session_state["direction"] = cfgs[first]["direction"]
        st.session_state["transform"] = cfgs[first]["transform"]
    else:
        # Classic single-attribute controls
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Giới hạn / Specification**")
            spec_limit = st.number_input(
                "Giới hạn spec / Spec limit",
                value=float(st.session_state.get("spec_limit", st.session_state["default_spec"])),
                format="%.6f",
                help="Lower limit for decreasing assay; upper limit for increasing impurity.",
            )
            dir_opts = ["decreasing", "increasing", "auto"]
            dir_cur = st.session_state.get("direction", st.session_state["default_dir"])
            if dir_cur not in dir_opts:
                dir_cur = "decreasing"
            direction = st.selectbox(
                "Hướng / Direction",
                dir_opts,
                index=dir_opts.index(dir_cur),
                help="decreasing = assay↓ (lower CI bound); increasing = impurity↑ (upper CI bound).",
            )
        with c2:
            st.markdown("**Hồi quy / Regression**")
            tf_opts = ["none", "log", "sqrt"]
            tf_cur = st.session_state.get("transform", "none")
            transform = st.selectbox(
                "Biến đổi / Transform",
                tf_opts,
                index=tf_opts.index(tf_cur) if tf_cur in tf_opts else 0,
            )

        st.session_state["spec_limit"] = float(spec_limit)
        st.session_state["direction"] = direction
        st.session_state["transform"] = transform
        # Mirror into attr_configs for unified run path
        aname = names[0] if names else "Response"
        st.session_state["selected_attributes"] = [aname]
        st.session_state["attr_configs"] = {
            aname: {
                "direction": direction,
                "spec_limit": float(spec_limit),
                "transform": transform,
            }
        }

    df_an = filter_condition(df, condition)
    n_sel = len(st.session_state.get("selected_attributes") or names)
    metric_cards(
        [
            {"label": "Points (filtered)", "value": str(len(df_an)), "tone": "neutral"},
            {"label": "Batches", "value": str(df_an["batch"].nunique()), "tone": "neutral"},
            {"label": "Attributes", "value": str(n_sel), "tone": "info"},
            {"label": "1−α (one-sided)", "value": f"{1 - alpha:.0%}", "tone": "ok"},
        ]
    )
    callout(
        "Tiếp theo: <b>5 · Gộp lô / Pooling</b> (ANCOVA) nếu có nhiều lô, "
        "rồi chạy phân tích ở bước <b>4 · Kết quả</b>.",
        kind="info",
    )


# ===========================================================================
# STEP 5 — POOLING
# ===========================================================================
elif step.startswith("5 ·"):
    section_header("5", "Gộp lô", "Pooling / ANCOVA")
    df = st.session_state["df"]
    if df is None or len(df) == 0:
        empty_state("Chưa có dữ liệu", "Go to step 3 · Dữ liệu first.", "📋")
        st.stop()

    df_an = filter_condition(df, st.session_state["condition"])
    # Max batches across attributes (pooling applied per attribute)
    parts = split_by_attribute(ensure_attribute_column(df_an))
    n_batches = max((int(sub["batch"].nunique()) for sub in parts.values()), default=0)

    if n_batches <= 1:
        callout(
            f"Chỉ có <b>{n_batches}</b> lô sau khi lọc điều kiện — "
            "poolability ANCOVA không áp dụng. Phân tích đơn lô sẽ được dùng ở bước Kết quả "
            "(áp dụng cho từng chỉ tiêu).",
            kind="info",
        )
        st.session_state["pool_mode"] = "Tự động (ANCOVA)"
    else:
        st.markdown(
            f"Phát hiện tối đa **{n_batches} lô** · ICH Q1E khuyến nghị kiểm định đồng nhất "
            "slope/intercept trước khi gộp (α poolability thường **0.25**). "
            "Gộp lô được áp dụng **riêng từng chỉ tiêu**."
        )

    pool_opts = [
        "Tự động (ANCOVA)",
        "Bắt buộc gộp (force pooled)",
        "Tách riêng (force separate)",
    ]
    pm_cur = st.session_state.get("pool_mode", pool_opts[0])
    if pm_cur not in pool_opts:
        pm_cur = pool_opts[0]

    c1, c2 = st.columns(2)
    with c1:
        pool_mode = st.selectbox(
            "Chế độ gộp lô / Pooling mode",
            pool_opts,
            index=pool_opts.index(pm_cur),
            disabled=(n_batches <= 1),
        )
    with c2:
        alpha_pool = st.number_input(
            "α poolability (ANCOVA)",
            min_value=0.01,
            max_value=0.5,
            value=float(st.session_state.get("alpha_pool", 0.25)),
            step=0.01,
            help="ICH Q1E commonly uses a large α (e.g. 0.25) for poolability tests.",
            disabled=(n_batches <= 1),
        )

    st.session_state["pool_mode"] = pool_mode
    st.session_state["alpha_pool"] = float(alpha_pool)

    with st.expander("Giải thích chế độ / Mode notes", expanded=False):
        st.markdown(
            """
- **Tự động (ANCOVA):** kiểm định đồng nhất; nếu không gộp được → shelf life = **min** các lô.
- **Bắt buộc gộp:** buộc dùng mô hình pooled (cần thận trọng).
- **Tách riêng:** luôn báo cáo theo từng lô; shelf life báo cáo = min.
- Với **đa chỉ tiêu**, cùng chế độ gộp áp dụng cho mỗi chỉ tiêu; overall = **min** các chỉ tiêu.
"""
        )

    metric_cards(
        [
            {"label": "N batches (max)", "value": str(n_batches), "tone": "neutral"},
            {"label": "Pooling mode", "value": pool_mode.split("(")[0].strip(), "tone": "info"},
            {
                "label": "α poolability",
                "value": f"{alpha_pool:.2f}",
                "tone": "ok" if n_batches > 1 else "neutral",
            },
        ]
    )
    callout("Chuyển sang <b>6 · Kết quả / Results</b> và nhấn <b>Chạy phân tích</b>.", kind="info")


# ===========================================================================
# STEP 6 — RESULTS
# ===========================================================================
elif step.startswith("6 ·"):
    section_header("6", "Kết quả", "Results")
    df = st.session_state["df"]
    if df is None or len(df) == 0:
        empty_state("Chưa có dữ liệu", "Complete step 3 · Dữ liệu before running analysis.", "📋")
        st.stop()

    condition = st.session_state["condition"]
    alpha = float(st.session_state["alpha"])
    alpha_pool = float(st.session_state["alpha_pool"])
    t_max = float(st.session_state["t_max"])
    pool_mode = st.session_state["pool_mode"]

    df_an = filter_condition(df, condition)
    if len(df_an) < 3:
        st.error("Không đủ điểm sau khi lọc điều kiện / Not enough points after filter.")
        st.stop()

    names = _attribute_names(df_an)
    _init_attr_configs(names)
    selected = st.session_state.get("selected_attributes") or names
    selected = [a for a in selected if a in names] or names

    with st.expander("Thiết lập hiện tại / Current settings", expanded=False):
        st.write(
            {
                "Condition": condition,
                "Attributes": selected,
                "Per-attribute": {
                    a: (st.session_state.get("attr_configs") or {}).get(a) for a in selected
                },
                "α (CI)": alpha,
                "α poolability": alpha_pool,
                "t_max": t_max,
                "Pooling": pool_mode,
                "N points": len(df_an),
            }
        )

    run = st.button("▶ Chạy phân tích / Run analysis", type="primary", use_container_width=False)

    force_pooled = None
    if pool_mode.startswith("Bắt buộc"):
        force_pooled = True
    elif pool_mode.startswith("Tách"):
        force_pooled = False

    if run:
        try:
            # Build data_by_attribute
            work = ensure_attribute_column(
                df_an,
                default_name=(names[0] if names else "Response"),
            )
            # If user renamed single to Response but data has no attr col originally
            parts = split_by_attribute(work)
            # Map "(single)" → configured name
            if list(parts.keys()) == ["(single)"]:
                parts = {selected[0]: parts["(single)"]}

            data_by_attr = {}
            for a in selected:
                if a not in parts:
                    st.error(f"Không có dữ liệu cho chỉ tiêu / No data for attribute: {a}")
                    st.stop()
                ok, msg = validate_for_analysis(parts[a])
                if not ok:
                    st.error(f"{a}: {msg}")
                    st.stop()
                data_by_attr[a] = parts[a]

            configs = _configs_from_session(selected)
            ma = analyze_attributes(
                data_by_attr,
                configs,
                alpha=alpha,
                alpha_pool=alpha_pool,
                t_max=t_max,
                force_pooled=force_pooled,
            )

            figures_by_attr: Dict[str, list] = {}
            all_figures: list = []
            for ar in ma.attributes:
                figs = _build_attr_figures(ar, alpha)
                figures_by_attr[ar.attribute] = figs
                all_figures.extend(figs)

            _role = classify_condition_role(str(condition))
            _tag = method_tag_for_role(_role)
            st.session_state["analysis"] = {
                "kind": "multi_attribute" if len(ma.attributes) > 1 else (
                    ma.attributes[0].kind if ma.attributes else "single"
                ),
                "multi_attr": True,
                "condition": condition,
                "condition_role": _role,
                "method_tag": _tag,
                "alpha": alpha,
                "alpha_pool": alpha_pool,
                "t_max": t_max,
                "pool_mode": pool_mode,
                "ma": ma,
                "figures_by_attr": figures_by_attr,
                "figures": all_figures,
                "message": ma.message,
                # legacy single fields for report fallback
                "spec_limit": configs[0].spec_limit if configs else None,
                "transform": configs[0].transform if configs else "none",
            }
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    analysis = st.session_state.get("analysis")
    if analysis is None:
        empty_state(
            "Chưa chạy phân tích",
            "Adjust Model / Pooling settings, then click Run analysis.",
            "▶",
        )
        st.stop()

    # Prefer new multi-attribute payload; fall back gracefully
    ma = analysis.get("ma")
    if ma is None:
        empty_state("Kết quả cũ không tương thích", "Please re-run analysis.", "⚠")
        st.stop()

    # ---- Overall summary ----
    tone = "ok" if ma.overall_shelf_life is not None else "warn"
    cards = [
        {
            "label": "Overall shelf life (mo)",
            "value": _fmt_shelf(ma.overall_shelf_life),
            "hint": "min of attributes" + (
                f" · limited by {ma.limiting_attribute}" if ma.limiting_attribute else ""
            ),
            "tone": tone,
        },
        {"label": "N attributes", "value": str(ma.n_attributes), "tone": "info"},
        {
            "label": "Limiting attribute",
            "value": str(ma.limiting_attribute or "—"),
            "tone": "warn" if ma.limiting_attribute else "neutral",
        },
        {
            "label": "Condition",
            "value": str(analysis.get("condition", ""))[:28],
            "tone": "neutral",
        },
    ]
    metric_cards(cards)
    st.info(ma.message)

    # Method tag from selected condition role
    cond_label = str(analysis.get("condition") or "")
    role = analysis.get("condition_role") or classify_condition_role(cond_label)
    tag = analysis.get("method_tag") or method_tag_for_role(role)
    if tag == "LT-Q1E":
        callout(
            f"<b>Method tag: {tag}</b> — Long-term / điều kiện thường · "
            "primary ICH Q1E shelf-life decision path (one-sided CI of the mean vs spec).",
            kind="ok",
        )
    elif tag == "LHCT":
        callout(
            f"<b>Method tag: {tag}</b> — Accelerated / LHCT · <b>supportive</b> only; "
            "not the primary long-term Q1E decision path. Bridging notes recommended.",
            kind="warn",
        )
    elif "Arrhenius" in str(tag):
        callout(
            f"<b>Method tag: {tag}</b> — exploratory / supportive only.",
            kind="danger",
        )
    else:
        callout(
            f"<b>Method tag: {tag}</b> · condition role: <code>{role}</code> · "
            f"condition: <code>{cond_label}</code>",
            kind="info",
        )

    section_header(None, "Tóm tắt theo chỉ tiêu", "Per-attribute shelf life")
    st.dataframe(ma.summary, use_container_width=True, hide_index=True)
    callout(
        "Overall product shelf life = <b>minimum</b> of attribute shelf lives "
        "(conservative ICH-style).",
        kind="info",
    )

    figures_by_attr = analysis.get("figures_by_attr") or {}
    for ar in ma.attributes:
        shelf_txt = _fmt_shelf(ar.reported_shelf_life)
        with st.expander(
            f"📊 {ar.attribute} — shelf life {shelf_txt} mo · {ar.direction_used} · {ar.kind}",
            expanded=(ma.n_attributes == 1),
        ):
            _render_single_attr_detail(ar, analysis["alpha"], figures_by_attr.get(ar.attribute, []))

    callout("Xuất báo cáo HTML/PDF ở bước <b>7 · Báo cáo / Report</b>.", kind="info")


# ===========================================================================
# STEP 7 — REPORT
# ===========================================================================
elif step.startswith("7 ·"):
    section_header("7", "Báo cáo", "Report export")
    analysis = st.session_state.get("analysis")
    if analysis is None:
        empty_state(
            "Chưa có kết quả để xuất",
            "Run analysis in step 6 · Kết quả first.",
            "📄",
        )
        st.stop()

    ma = analysis.get("ma")
    if ma is None:
        empty_state("Kết quả cũ không tương thích", "Please re-run analysis.", "⚠")
        st.stop()

    figures_by_attr = analysis.get("figures_by_attr") or {}
    all_figures = list(analysis.get("figures") or [])

    _role = analysis.get("condition_role") or classify_condition_role(str(analysis.get("condition") or ""))
    _tag = analysis.get("method_tag") or method_tag_for_role(_role)
    meta = {
        "Condition": analysis.get("condition"),
        "Condition role": _role,
        "Method tag": _tag,
        "N attributes": ma.n_attributes,
        "Overall shelf life (months)": (
            ma.overall_shelf_life if ma.overall_shelf_life is not None else "N/A"
        ),
        "Limiting attribute": ma.limiting_attribute or "—",
        "α (one-sided CI)": analysis.get("alpha"),
        "α poolability": analysis.get("alpha_pool"),
        "Pooling mode": analysis.get("pool_mode"),
        "Aggregation": "min of attribute shelf lives (conservative)",
    }

    metric_cards(
        [
            {
                "label": "Overall shelf life (mo)",
                "value": _fmt_shelf(ma.overall_shelf_life),
                "tone": "ok",
            },
            {"label": "N attributes", "value": str(ma.n_attributes), "tone": "info"},
            {
                "label": "Limiting",
                "value": str(ma.limiting_attribute or "—")[:24],
                "tone": "warn",
            },
        ]
    )

    attr_sections = []
    for ar in ma.attributes:
        sec_meta = (
            f"<p><b>Direction:</b> {ar.direction_used} · "
            f"<b>Spec:</b> {ar.config.spec_limit} · "
            f"<b>Transform:</b> {ar.config.transform} · "
            f"<b>Shelf life:</b> {_fmt_shelf(ar.reported_shelf_life)} months</p>"
        )
        if ar.kind == "single" and ar.reg is not None:
            coef_h = dataframe_to_html(ar.reg.summary_table())
            anova_h = dataframe_to_html(ar.reg.anova)
        elif ar.kind == "multi" and ar.mb is not None:
            rows = []
            for b, (reg, sl) in ar.mb.per_batch.items():
                rows.append(
                    {
                        "Batch": b,
                        "β0": reg.beta0,
                        "β1": reg.beta1,
                        "R²": reg.r_squared,
                        "Shelf life": sl.shelf_life,
                    }
                )
            coef_h = dataframe_to_html(pd.DataFrame(rows)) if rows else "<p>(none)</p>"
            anova_h = (
                dataframe_to_html(ar.mb.poolability.anova_full)
                if ar.mb.poolability.anova_full is not None
                else "<p>(none)</p>"
            )
        else:
            coef_h, anova_h = "<p>(none)</p>", "<p>(none)</p>"

        attr_sections.append(
            {
                "name": ar.attribute,
                "meta_html": sec_meta,
                "coef_html": coef_h,
                "anova_html": anova_h,
                "shelf_html": f"<p>{ar.message}</p>",
                "figures": figures_by_attr.get(ar.attribute, []),
            }
        )

    # Enrich attribute sections with DataFrames for native PDF export
    for sec, ar in zip(attr_sections, ma.attributes):
        if ar.kind == "single" and ar.reg is not None:
            sec["coef_df"] = ar.reg.summary_table()
            sec["anova_df"] = ar.reg.anova
        elif ar.kind == "multi" and ar.mb is not None:
            rows = []
            for b, (reg, sl) in ar.mb.per_batch.items():
                rows.append(
                    {
                        "Batch": b,
                        "β0": reg.beta0,
                        "β1": reg.beta1,
                        "R²": reg.r_squared,
                        "Shelf life": sl.shelf_life,
                    }
                )
            sec["coef_df"] = pd.DataFrame(rows) if rows else None
            sec["anova_df"] = (
                ar.mb.poolability.anova_full
                if ar.mb.poolability.anova_full is not None
                else None
            )
        sec["meta_text"] = (
            f"Direction: {ar.direction_used} · Spec: {ar.config.spec_limit} · "
            f"Transform: {ar.config.transform} · "
            f"Shelf life: {_fmt_shelf(ar.reported_shelf_life)} months"
        )
        sec["shelf_text"] = ar.message

    report_notes = (
        f"Method tag: {_tag} (condition role: {_role}). "
        "Overall product shelf life = minimum of attribute shelf lives "
        "(conservative ICH-style). Per-attribute regression uses one-sided "
        "confidence bound of the mean (ICH Q1E) for long-term (LT-Q1E). "
        "Accelerated (LHCT) and Arrhenius are supportive only. "
        "Supportive tool — NOT regulatory certification. "
        "Thời hạn bảo quản tổng thể = min các chỉ tiêu."
    )
    report_title = (
        "ICH Stability / Shelf-Life Report (Multi-Attribute)"
        if ma.n_attributes > 1
        else "ICH Stability / Shelf-Life Report"
    )

    anova_top = None
    if ma.n_attributes == 1 and ma.attributes and ma.attributes[0].reg is not None:
        anova_top = ma.attributes[0].reg.anova

    # PDF first (uses live matplotlib figures); HTML may close figures afterward
    pdf_bytes = export_pdf_bytes(
        title=report_title,
        meta=meta,
        summary_df=ma.summary if ma.n_attributes > 1 else None,
        coef_df=ma.summary,
        anova_df=anova_top,
        shelf_text=ma.message,
        attribute_sections=attr_sections if ma.n_attributes > 1 else None,
        figures=all_figures if ma.n_attributes == 1 else [],
        notes=report_notes,
        html=None,
    )

    # Top-level coef/anova = summary table
    extra_html = ""
    mat = st.session_state.get("sampling_matrix")
    if isinstance(mat, pd.DataFrame) and len(mat) > 0:
        preview = mat.head(40)
        extra_html = (
            "<h2>Study Design sampling plan (ICH Q1D) — excerpt</h2>"
            "<p>Supportive planning matrix from Design step "
            f"({len(mat)} cells; showing first {len(preview)}).</p>"
            + dataframe_to_html(preview)
        )

    callout(
        f"<b>Method tag on this report: {_tag}</b> · role <code>{_role}</code>. "
        "LT-Q1E = primary long-term decision; LHCT / Arrhenius-supportive = supportive only.",
        kind="ok" if _tag == "LT-Q1E" else "warn",
    )

    html = build_html_report(
        title=report_title,
        meta=meta,
        coef_html=dataframe_to_html(ma.summary),
        anova_html="<p>See per-attribute sections below.</p>"
        if ma.n_attributes > 1
        else (
            dataframe_to_html(ma.attributes[0].reg.anova)
            if ma.attributes and ma.attributes[0].reg is not None
            else ""
        ),
        shelf_html=f"<p><b>[{_tag}] {ma.message}</b></p>",
        figures=all_figures if ma.n_attributes == 1 else [],
        notes=report_notes,
        attribute_sections=attr_sections if ma.n_attributes > 1 else None,
        summary_html=dataframe_to_html(ma.summary) if ma.n_attributes > 1 else "",
        extra_sections_html=extra_html,
    )

    st.markdown("**Tải xuống / Download**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇ Download HTML report",
            html,
            file_name="shelf_life_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "⬇ Download PDF report",
            pdf_bytes,
            file_name="shelf_life_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    st.caption(
        "PDF dùng reportlab (pip-only). WeasyPrint vẫn là fallback tuỳ chọn. "
        "/ PDF via reportlab after pip install; WeasyPrint optional fallback."
    )

    with st.expander("Xem trước metadata / Preview meta", expanded=True):
        st.json(meta)
        st.dataframe(ma.summary, use_container_width=True, hide_index=True)
