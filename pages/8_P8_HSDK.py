"""Streamlit multipage: P.8 ACTD ASEAN export (standalone page)."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="P.8 HSĐK", page_icon="📄", layout="wide")

st.title("Báo cáo P.8 (ACTD ASEAN)")
st.caption("Hỗ trợ soạn thảo HSĐK — không phải hồ sơ đã phê duyệt")

try:
    from src.p8_report import (
        P8Fields,
        build_p8_from_app,
        build_p8_html,
        demo_p8_fields,
        export_p8_pdf_bytes,
    )
except Exception as exc:
    st.error(f"Không import được src.p8_report: {exc}")
    st.stop()

analysis = st.session_state.get("analysis")
df = st.session_state.get("df")
design = st.session_state.get("study_design")
program = st.session_state.get("stability_program")

st.info(
    "Nếu chưa chạy phân tích ở app chính, vẫn tải được **Demo P.8**. "
    "Có kết quả Q1E thì báo cáo sẽ điền shelf-life tự động."
)

if st.session_state.get("p8_fields") is None:
    st.session_state["p8_fields"] = P8Fields().to_dict()
p8_dict = dict(st.session_state["p8_fields"] or {})
hdr = dict(p8_dict.get("header") or {})

with st.expander("P.8 header (chỉnh tay)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        hdr["ten_thuoc"] = st.text_input("Tên thuốc", value=str(hdr.get("ten_thuoc") or ""), key="pg_p8_ten")
        hdr["duoc_chat_ham_luong"] = st.text_input(
            "Dược chất - hàm lượng", value=str(hdr.get("duoc_chat_ham_luong") or ""), key="pg_p8_dc"
        )
        hdr["dang_bao_che"] = st.text_input(
            "Dạng bào chế", value=str(hdr.get("dang_bao_che") or ""), key="pg_p8_dbc"
        )
        hdr["nha_san_xuat"] = st.text_input(
            "Nhà sản xuất", value=str(hdr.get("nha_san_xuat") or ""), key="pg_p8_nsx"
        )
    with c2:
        hdr["quy_cach_dong_goi"] = st.text_input(
            "Quy cách đóng gói", value=str(hdr.get("quy_cach_dong_goi") or ""), key="pg_p8_qc"
        )
        hdr["ma_bao_cao"] = st.text_input(
            "Mã báo cáo/phiên bản", value=str(hdr.get("ma_bao_cao") or ""), key="pg_p8_ma"
        )
        hdr["nguoi_lap_kiem_phe_duyet"] = st.text_input(
            "Người lập / kiểm / phê duyệt",
            value=str(hdr.get("nguoi_lap_kiem_phe_duyet") or ""),
            key="pg_p8_nguoi",
        )
        p8_dict["dieu_kien_bao_quan"] = st.text_input(
            "Điều kiện bảo quản ghi nhãn",
            value=str(p8_dict.get("dieu_kien_bao_quan") or ""),
            key="pg_p8_dkbq",
        )
    p8_dict["header"] = hdr
    st.session_state["p8_fields"] = p8_dict

p8_html = ""
p8_pdf = b""
err = None
try:
    p8_obj = build_p8_from_app(
        p8_fields=P8Fields.from_dict(st.session_state.get("p8_fields")),
        df=df,
        analysis=analysis,
        study_design=design if isinstance(design, dict) else None,
        program=program if isinstance(program, dict) else None,
        data_label=st.session_state.get("data_label"),
    )
    st.session_state["p8_fields"] = p8_obj.to_dict()
    p8_html = build_p8_html(p8_obj)
    p8_pdf = export_p8_pdf_bytes(p8_obj)
except Exception as exc:
    err = str(exc)
    st.error(f"Lỗi tạo P.8 từ dữ liệu app: {exc}")
    st.caption("Vẫn có thể tải Demo P.8 bên dưới.")

a, b, c = st.columns(3)
with a:
    if p8_html:
        st.download_button(
            "⬇ P.8 HTML",
            p8_html,
            file_name="P8_ACTD_ASEAN_draft.html",
            mime="text/html",
            use_container_width=True,
            key="pg_dl_p8_html",
        )
with b:
    if p8_pdf:
        st.download_button(
            "⬇ P.8 PDF",
            p8_pdf,
            file_name="P8_ACTD_ASEAN_draft.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pg_dl_p8_pdf",
        )
with c:
    if st.button("Tạo Demo P.8", use_container_width=True, key="pg_p8_demo"):
        demo = demo_p8_fields()
        st.session_state["pg_p8_demo_html"] = build_p8_html(demo)
        st.session_state["pg_p8_demo_pdf"] = export_p8_pdf_bytes(demo)

if st.session_state.get("pg_p8_demo_html"):
    st.download_button(
        "⬇ Demo P.8 HTML",
        st.session_state["pg_p8_demo_html"],
        file_name="P8_demo.html",
        mime="text/html",
        key="pg_dl_demo_html",
    )
    st.download_button(
        "⬇ Demo P.8 PDF",
        st.session_state["pg_p8_demo_pdf"],
        file_name="P8_demo.pdf",
        mime="application/pdf",
        key="pg_dl_demo_pdf",
    )

if p8_html and not err:
    with st.expander("Xem trước HTML (đầu file)", expanded=False):
        st.code(p8_html[:2000] + ("\n…" if len(p8_html) > 2000 else ""), language="html")
