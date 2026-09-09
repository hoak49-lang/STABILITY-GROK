"""ACTD ASEAN P.8 (Độ ổn định thành phẩm) drafting aid.

Fills a Vietnamese P.8-style report from app session data.
Missing fields become “[…]” placeholders.
SUPPORTIVE drafting aid for HSĐK — NOT an approved dossier replacement.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

VN_TZ = timezone(timedelta(hours=7))
PLACEHOLDER = "[…]"

DISCLAIMER_VI = (
    "Đây là công cụ hỗ trợ soạn thảo nội dung P.8 theo khung ACTD ASEAN — "
    "KHÔNG phải hồ sơ đăng ký (HSĐK) đã phê duyệt và không thay thế đánh giá "
    "chuyên gia QA/RA / cơ quan quản lý."
)
DISCLAIMER_EN = (
    "Supportive drafting aid for ACTD ASEAN P.8 content — NOT an approved "
    "regulatory dossier (HSĐK) replacement and not regulatory advice."
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FONT = _PACKAGE_ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
_BUNDLED_FONT_BOLD = _PACKAGE_ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf"


def _nz(val: Any, default: str = PLACEHOLDER) -> str:
    if val is None:
        return default
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return default
    return s


def _fmt_num(val: Any, digits: int = 2) -> str:
    if val is None:
        return PLACEHOLDER
    try:
        if isinstance(val, float) and (val != val):  # NaN
            return PLACEHOLDER
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return _nz(val)


@dataclass
class P8Header:
    ten_thuoc: str = PLACEHOLDER
    duoc_chat_ham_luong: str = PLACEHOLDER
    dang_bao_che: str = PLACEHOLDER
    nha_san_xuat: str = PLACEHOLDER
    quy_cach_dong_goi: str = PLACEHOLDER
    ma_bao_cao: str = PLACEHOLDER
    ngay_chot_du_lieu: str = PLACEHOLDER
    nguoi_lap_kiem_phe_duyet: str = PLACEHOLDER


@dataclass
class P8Fields:
    header: P8Header = field(default_factory=P8Header)
    # P.8.1 A
    muc_tieu: str = PLACEHOLDER
    de_cuong: str = PLACEHOLDER
    thiet_ke_rut_gon: str = PLACEHOLDER
    # P.8.1 B
    chi_tieu_da_kiem: str = PLACEHOLDER
    xu_huong: str = PLACEHOLDER
    khac_biet_lo: str = PLACEHOLDER
    ngoai_tieu_chuan: str = PLACEHOLDER
    phan_tich_thong_ke: str = PLACEHOLDER
    gioi_han_du_lieu: str = PLACEHOLDER
    # P.8.1 C
    dl_dai_han_thang: str = PLACEHOLDER
    dl_cap_toc_thang: str = PLACEHOLDER
    han_dung_thang: str = PLACEHOLDER
    dieu_kien_bao_quan: str = PLACEHOLDER
    bao_bi_ap_dung: str = PLACEHOLDER
    han_dung_sau_mo: str = PLACEHOLDER
    co_so_ho_tro: str = PLACEHOLDER
    # P.8.2
    lo_tiep_tuc: str = PLACEHOLDER
    lo_bo_sung: str = PLACEHOLDER
    bao_bi_ham_luong: str = PLACEHOLDER
    dk_bao_quan_mau: str = PLACEHOLDER
    moc_kiem_tra_con: str = PLACEHOLDER
    chi_tieu_gioi_han: str = PLACEHOLDER
    phuong_phap_phan_tich: str = PLACEHOLDER
    don_vi_chiu_trach_nhiem: str = PLACEHOLDER
    cam_ket_text: str = PLACEHOLDER
    dai_dien: str = PLACEHOLDER
    chuc_danh_ky: str = PLACEHOLDER
    # P.8.3
    tieu_chuan_phuong_phap_note: str = PLACEHOLDER
    # P.8.4
    cold_chain: str = "Không áp dụng / Not applicable — [nêu lý do]."
    # tables as records
    batch_design_rows: List[Dict[str, str]] = field(default_factory=list)
    study_condition_rows: List[Dict[str, str]] = field(default_factory=list)
    spec_method_rows: List[Dict[str, str]] = field(default_factory=list)
    results_tables: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "P8Fields":
        if not data:
            return cls()
        header_raw = data.get("header") or {}
        header = P8Header(**{k: header_raw.get(k, PLACEHOLDER) for k in P8Header.__dataclass_fields__})
        kwargs = {k: data.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__ if k != "header"}
        return cls(header=header, **kwargs)


def _infer_max_time(df: Optional[pd.DataFrame], condition_substr: Sequence[str]) -> Optional[float]:
    if df is None or len(df) == 0 or "condition" not in df.columns:
        return None
    mask = df["condition"].astype(str).str.lower().apply(
        lambda s: any(x in s for x in condition_substr)
    )
    sub = df.loc[mask] if mask.any() else df
    if "time" not in sub.columns or len(sub) == 0:
        return None
    try:
        return float(pd.to_numeric(sub["time"], errors="coerce").max())
    except Exception:
        return None


def build_p8_from_app(
    *,
    p8_fields: Optional[P8Fields] = None,
    df: Optional[pd.DataFrame] = None,
    analysis: Optional[dict] = None,
    study_design: Optional[dict] = None,
    program: Optional[dict] = None,
    data_label: Optional[str] = None,
) -> P8Fields:
    """Merge explicit P8 fields with inferred values from app state."""
    fields = p8_fields or P8Fields()
    h = fields.header

    # Design product info
    if study_design:
        h.ten_thuoc = _nz(h.ten_thuoc if h.ten_thuoc != PLACEHOLDER else study_design.get("product_name"))
        if h.ten_thuoc == PLACEHOLDER:
            h.ten_thuoc = _nz(study_design.get("product_name"))
        strengths = study_design.get("strengths") or study_design.get("strength_list")
        if strengths and h.duoc_chat_ham_luong == PLACEHOLDER:
            h.duoc_chat_ham_luong = ", ".join(str(x) for x in strengths) if isinstance(strengths, list) else _nz(strengths)
        if h.dang_bao_che == PLACEHOLDER:
            h.dang_bao_che = _nz(study_design.get("dosage_form"))
        packs = study_design.get("packs") or study_design.get("pack_list")
        if packs and h.quy_cach_dong_goi == PLACEHOLDER:
            h.quy_cach_dong_goi = ", ".join(str(x) for x in packs) if isinstance(packs, list) else _nz(packs)
        if h.dieu_kien_bao_quan == PLACEHOLDER or fields.dieu_kien_bao_quan == PLACEHOLDER:
            storage = study_design.get("proposed_storage") or study_design.get("storage")
            if storage:
                fields.dieu_kien_bao_quan = _nz(storage)
                if h.dieu_kien_bao_quan == PLACEHOLDER:
                    pass
        design_type = study_design.get("design_type") or study_design.get("design")
        if design_type and fields.thiet_ke_rut_gon == PLACEHOLDER:
            fields.thiet_ke_rut_gon = _nz(design_type)

    if program and isinstance(program, dict):
        products = program.get("products") or []
        if products and h.ten_thuoc == PLACEHOLDER:
            p0 = products[0] if isinstance(products[0], dict) else {}
            h.ten_thuoc = _nz(p0.get("name") or p0.get("product_name"))

    # Analysis-driven conclusions
    ma = None
    if analysis and isinstance(analysis, dict):
        ma = analysis.get("ma")
        tag = analysis.get("method_tag") or PLACEHOLDER
        cond = analysis.get("condition") or PLACEHOLDER
        if fields.phan_tich_thong_ke == PLACEHOLDER:
            fields.phan_tich_thong_ke = (
                f"Phân tích hỗ trợ ICH Q1E (OLS, CI một phía của trung bình). "
                f"Method tag: {tag}. Điều kiện phân tích: {cond}."
            )
        if ma is not None:
            overall = getattr(ma, "overall_shelf_life", None)
            limiting = getattr(ma, "limiting_attribute", None)
            if fields.han_dung_thang == PLACEHOLDER and overall is not None:
                fields.han_dung_thang = _fmt_num(overall, 1)
            if fields.co_so_ho_tro == PLACEHOLDER:
                fields.co_so_ho_tro = (
                    f"Overall shelf life (min chỉ tiêu) = {_fmt_num(overall, 2)} tháng; "
                    f"chỉ tiêu hạn chế: {_nz(limiting)}. "
                    f"Nguồn dữ liệu: {_nz(data_label)}. "
                    "Kết quả mang tính hỗ trợ soạn thảo — không thay thế HSĐK."
                )
            if fields.chi_tieu_da_kiem == PLACEHOLDER and hasattr(ma, "summary"):
                try:
                    names = list(ma.summary.get("Attribute", ma.summary.get("attribute", [])))
                    if not names and hasattr(ma, "attributes"):
                        names = [getattr(a, "attribute", "?") for a in ma.attributes]
                    fields.chi_tieu_da_kiem = ", ".join(str(n) for n in names) if names else PLACEHOLDER
                except Exception:
                    pass
            if fields.xu_huong == PLACEHOLDER and hasattr(ma, "attributes"):
                bits = []
                for ar in ma.attributes:
                    bits.append(
                        f"{ar.attribute}: {ar.direction_used}, "
                        f"shelf life {_fmt_num(ar.reported_shelf_life, 2)} mo"
                    )
                fields.xu_huong = "; ".join(bits) if bits else PLACEHOLDER

    # Time coverage from df
    if df is not None and len(df) > 0:
        lt = _infer_max_time(df, ("25", "30", "long", "dài hạn", "lt"))
        acc = _infer_max_time(df, ("40", "acceler", "cấp tốc", "lhct"))
        if fields.dl_dai_han_thang == PLACEHOLDER and lt is not None:
            fields.dl_dai_han_thang = _fmt_num(lt, 0)
        if fields.dl_cap_toc_thang == PLACEHOLDER and acc is not None:
            fields.dl_cap_toc_thang = _fmt_num(acc, 0)

        # Batch design rows
        if not fields.batch_design_rows and "batch" in df.columns:
            for b in sorted(df["batch"].astype(str).unique()):
                fields.batch_design_rows.append(
                    {
                        "so_lo": b,
                        "ngay_sx": PLACEHOLDER,
                        "co_lo": PLACEHOLDER,
                        "ham_luong": PLACEHOLDER,
                        "bao_bi": PLACEHOLDER,
                        "ngay_bat_dau": PLACEHOLDER,
                    }
                )

        # Study condition rows
        if not fields.study_condition_rows and "condition" in df.columns:
            for cond in sorted(df["condition"].astype(str).unique()):
                tmax = float(pd.to_numeric(df.loc[df["condition"].astype(str) == cond, "time"], errors="coerce").max())
                role = "Dài hạn"
                cl = cond.lower()
                if "40" in cl or "acceler" in cl:
                    role = "Lão hóa cấp tốc"
                elif "30" in cl and ("65" in cl or "75" in cl or "inter" in cl):
                    role = "Trung gian / bổ sung"
                fields.study_condition_rows.append(
                    {
                        "nghien_cuu": role,
                        "nhiet_do_do_am": cond,
                        "lich_kiem_tra": PLACEHOLDER,
                        "thoi_gian_du_lieu": _fmt_num(tmax, 0),
                    }
                )

        # Spec / method rows from attributes
        if not fields.spec_method_rows:
            attrs = []
            if "attribute" in df.columns:
                attrs = sorted(df["attribute"].astype(str).unique())
            elif analysis and analysis.get("ma") is not None:
                ma2 = analysis["ma"]
                attrs = [getattr(a, "attribute", "Response") for a in getattr(ma2, "attributes", [])]
            if not attrs:
                attrs = ["Response"]
            cfgs = {}
            if analysis and analysis.get("ma") is not None:
                for a in getattr(analysis["ma"], "attributes", []):
                    cfg = getattr(a, "config", None)
                    if cfg is not None:
                        cfgs[a.attribute] = cfg
            for name in attrs:
                cfg = cfgs.get(name)
                lim = _fmt_num(getattr(cfg, "spec_limit", None)) if cfg else PLACEHOLDER
                fields.spec_method_rows.append(
                    {
                        "chi_tieu": name,
                        "gioi_han": lim,
                        "don_vi": PLACEHOLDER,
                        "phuong_phap": PLACEHOLDER,
                        "tham_chieu": PLACEHOLDER,
                    }
                )

        # Results tables: one per batch × condition (compact)
        if not fields.results_tables:
            work = df.copy()
            if "attribute" not in work.columns:
                work["attribute"] = "Response"
            for (batch, cond), sub in work.groupby(["batch", "condition"]):
                times = sorted(pd.to_numeric(sub["time"], errors="coerce").dropna().unique().tolist())
                # pivot-ish rows
                rows = []
                for attr, asub in sub.groupby("attribute"):
                    row = {"chi_tieu": str(attr), "gioi_han": PLACEHOLDER}
                    for t in times:
                        val = asub.loc[pd.to_numeric(asub["time"], errors="coerce") == t, "response"]
                        key = f"t_{t:g}"
                        row[key] = _fmt_num(val.iloc[0], 4) if len(val) else PLACEHOLDER
                    rows.append(row)
                fields.results_tables.append(
                    {
                        "so_lo": str(batch),
                        "bao_bi": PLACEHOLDER,
                        "dieu_kien": str(cond),
                        "timepoints": [f"{t:g}" for t in times],
                        "rows": rows,
                    }
                )

    if fields.muc_tieu == PLACEHOLDER:
        fields.muc_tieu = (
            f"Đánh giá độ ổn định của sản phẩm {_nz(h.ten_thuoc)} trong bao bì "
            f"{_nz(h.quy_cach_dong_goi)} nhằm hỗ trợ hạn dùng và điều kiện bảo quản đề xuất."
        )
    if fields.header.ngay_chot_du_lieu == PLACEHOLDER:
        fields.header.ngay_chot_du_lieu = datetime.now(VN_TZ).strftime("%Y-%m-%d")

    fields.header = h
    return fields


def build_p8_html(fields: P8Fields) -> str:
    """HTML report matching P.8 section spirit."""
    h = fields.header
    now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M ICT")

    def table(headers: List[str], rows: List[List[str]]) -> str:
        th = "".join(f"<th>{x}</th>" for x in headers)
        body = ""
        for r in rows:
            body += "<tr>" + "".join(f"<td>{_nz(c)}</td>" for c in r) + "</tr>"
        if not rows:
            body = f"<tr><td colspan='{len(headers)}'>{PLACEHOLDER}</td></tr>"
        return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"

    batch_rows = [
        [r.get("so_lo"), r.get("ngay_sx"), r.get("co_lo"), r.get("ham_luong"), r.get("bao_bi"), r.get("ngay_bat_dau")]
        for r in fields.batch_design_rows
    ]
    study_rows = [
        [r.get("nghien_cuu"), r.get("nhiet_do_do_am"), r.get("lich_kiem_tra"), r.get("thoi_gian_du_lieu")]
        for r in fields.study_condition_rows
    ]
    spec_rows = [
        [r.get("chi_tieu"), r.get("gioi_han"), r.get("don_vi"), r.get("phuong_phap"), r.get("tham_chieu")]
        for r in fields.spec_method_rows
    ]

    results_html = ""
    for block in fields.results_tables:
        tps = block.get("timepoints") or []
        headers = ["Chỉ tiêu", "Giới hạn"] + [f"Mốc {t}" for t in tps]
        rows = []
        for r in block.get("rows") or []:
            rows.append(
                [r.get("chi_tieu"), r.get("gioi_han")]
                + [r.get(f"t_{t}", PLACEHOLDER) for t in tps]
            )
        results_html += (
            f"<h4>Số lô: {_nz(block.get('so_lo'))} · Bao bì: {_nz(block.get('bao_bi'))} · "
            f"Điều kiện: {_nz(block.get('dieu_kien'))}</h4>"
            + table(headers, rows)
        )
    if not results_html:
        results_html = f"<p>{PLACEHOLDER}</p>"

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<title>P.8 Độ ổn định thành phẩm — ACTD ASEAN (draft)</title>
<style>
body {{ font-family: DejaVu Sans, Arial, sans-serif; margin: 2rem; color: #222; line-height: 1.45; }}
h1 {{ font-size: 1.35rem; border-bottom: 2px solid #1a4a7a; padding-bottom: .3rem; }}
h2 {{ font-size: 1.15rem; color: #1a4a7a; margin-top: 1.6rem; }}
h3 {{ font-size: 1.05rem; margin-top: 1.1rem; }}
h4 {{ font-size: .95rem; margin-top: .8rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .6rem 0 1rem; font-size: .9rem; }}
th, td {{ border: 1px solid #bbb; padding: .35rem .5rem; vertical-align: top; }}
th {{ background: #eef3f8; }}
.meta td:first-child {{ font-weight: 600; width: 32%; background: #f7f9fb; }}
.disclaimer {{ background: #fff3cd; border: 1px solid #e0a800; padding: .75rem 1rem; margin: 1rem 0; }}
.footer {{ color: #666; font-size: .85rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="disclaimer"><b>Tuyên bố / Disclaimer:</b> {DISCLAIMER_VI}<br/>{DISCLAIMER_EN}</div>
<h1>P.8. ĐỘ ỔN ĐỊNH THÀNH PHẨM<br/><small>Mẫu báo cáo hồ sơ ACTD ASEAN (bản nháp hỗ trợ)</small></h1>
<p><i>Khung tham khảo để điền dữ liệu sản phẩm. Không phải biểu mẫu bắt buộc hoặc báo cáo đã có kết quả nghiên cứu.</i></p>

<table class="meta">
<tr><td>Tên thuốc</td><td>{_nz(h.ten_thuoc)}</td></tr>
<tr><td>Dược chất - hàm lượng</td><td>{_nz(h.duoc_chat_ham_luong)}</td></tr>
<tr><td>Dạng bào chế</td><td>{_nz(h.dang_bao_che)}</td></tr>
<tr><td>Nhà sản xuất và địa điểm sản xuất</td><td>{_nz(h.nha_san_xuat)}</td></tr>
<tr><td>Quy cách đóng gói</td><td>{_nz(h.quy_cach_dong_goi)}</td></tr>
<tr><td>Mã báo cáo/phiên bản</td><td>{_nz(h.ma_bao_cao)}</td></tr>
<tr><td>Ngày chốt dữ liệu</td><td>{_nz(h.ngay_chot_du_lieu)}</td></tr>
<tr><td>Người lập - người kiểm tra - người phê duyệt</td><td>{_nz(h.nguoi_lap_kiem_phe_duyet)}</td></tr>
</table>

<h2>P.8.1. Tóm tắt và kết luận về độ ổn định</h2>
<h3>A. Mục tiêu và thiết kế nghiên cứu</h3>
<p>{_nz(fields.muc_tieu)}</p>
<p>Đề cương áp dụng: {_nz(fields.de_cuong)}.</p>
{table(["Số lô", "Ngày sản xuất", "Cỡ lô/quy mô", "Hàm lượng", "Bao bì, quy cách", "Ngày bắt đầu"], batch_rows)}
{table(["Nghiên cứu", "Nhiệt độ/độ ẩm", "Lịch kiểm tra theo đề cương", "Thời gian đã có dữ liệu"], study_rows)}
<p>Thiết kế rút gọn, nếu áp dụng: {_nz(fields.thiet_ke_rut_gon)}.</p>

<h3>B. Tổng hợp kết quả và đánh giá</h3>
<p>Các chỉ tiêu đã kiểm tra: {_nz(fields.chi_tieu_da_kiem)}.</p>
<p>Xu hướng thay đổi theo thời gian: {_nz(fields.xu_huong)}.</p>
<p>Khác biệt giữa các lô, hàm lượng hoặc bao bì: {_nz(fields.khac_biet_lo)}.</p>
<p>Kết quả ngoài tiêu chuẩn, ngoài xu hướng hoặc sai lệch: {_nz(fields.ngoai_tieu_chuan)}.</p>
<p>Phân tích thống kê và cơ sở ngoại suy, nếu áp dụng: {_nz(fields.phan_tich_thong_ke)}.</p>
<p>Giới hạn của dữ liệu hiện có: {_nz(fields.gioi_han_du_lieu)}.</p>

<h3>C. Kết luận</h3>
<p>Tại ngày chốt dữ liệu {_nz(h.ngay_chot_du_lieu)}, đã có dữ liệu dài hạn đến {_nz(fields.dl_dai_han_thang)} tháng
và cấp tốc đến {_nz(fields.dl_cap_toc_thang)} tháng.</p>
<p>Căn cứ kết quả và đánh giá nêu trên, đề xuất:</p>
<ul>
<li>Hạn dùng: {_nz(fields.han_dung_thang)} tháng.</li>
<li>Điều kiện bảo quản ghi nhãn: {_nz(fields.dieu_kien_bao_quan)}.</li>
<li>Bao bì áp dụng: {_nz(fields.bao_bi_ap_dung if fields.bao_bi_ap_dung != PLACEHOLDER else h.quy_cach_dong_goi)}.</li>
<li>Hạn dùng sau mở nắp/pha hoàn nguyên, nếu áp dụng: {_nz(fields.han_dung_sau_mo)}.</li>
<li>Cơ sở hỗ trợ đề xuất: {_nz(fields.co_so_ho_tro)}.</li>
</ul>

<h2>P.8.2. Đề cương và cam kết độ ổn định sau phê duyệt</h2>
<h3>A. Đề cương tiếp tục nghiên cứu</h3>
{table(["Nội dung", "Kế hoạch"], [
    ["Các lô tiếp tục theo dõi", fields.lo_tiep_tuc],
    ["Các lô bổ sung", fields.lo_bo_sung],
    ["Bao bì/hàm lượng áp dụng", fields.bao_bi_ham_luong],
    ["Điều kiện bảo quản mẫu", fields.dk_bao_quan_mau],
    ["Mốc kiểm tra còn lại", fields.moc_kiem_tra_con],
    ["Chỉ tiêu và giới hạn chấp nhận", fields.chi_tieu_gioi_han],
    ["Phương pháp phân tích", fields.phuong_phap_phan_tich],
    ["Đơn vị chịu trách nhiệm", fields.don_vi_chiu_trach_nhiem],
])}
<h3>B. Cam kết</h3>
<p>{_nz(fields.cam_ket_text)}</p>
<p>Đại diện có thẩm quyền: {_nz(fields.dai_dien)}</p>
<p>Chức danh - ngày ký - chữ ký: {_nz(fields.chuc_danh_ky)}</p>

<h2>P.8.3. Dữ liệu độ ổn định</h2>
<h3>A. Tiêu chuẩn và phương pháp</h3>
<p>{_nz(fields.tieu_chuan_phuong_phap_note)}</p>
{table(["Chỉ tiêu", "Giới hạn trong hạn dùng", "Đơn vị", "Phương pháp/phiên bản", "Tham chiếu thẩm định"], spec_rows)}
<h3>B. Bảng kết quả</h3>
<p>Lập riêng cho từng lô, hàm lượng, bao bì và điều kiện nghiên cứu.</p>
{results_html}
<h3>C. Tài liệu kèm theo</h3>
<ul>
<li>Phiếu kết quả kiểm nghiệm và báo cáo nghiên cứu.</li>
<li>Đồ thị xu hướng, phân tích thống kê nếu có.</li>
<li>Hồ sơ sai lệch, điều tra và đánh giá ảnh hưởng.</li>
<li>Thông tin phương pháp phân tích và thẩm định hoặc tham chiếu đến phần tương ứng của hồ sơ.</li>
</ul>

<h2>P.8.4. Bảo đảm dây chuyền lạnh, nếu áp dụng</h2>
<p>{_nz(fields.cold_chain)}</p>

<p class="footer">Xuất lúc {now} · ICH-STABILITY-GROK · Mẫu tham khảo P.8 - ACTD ASEAN<br/>
{DISCLAIMER_VI}</p>
</body></html>
"""
    return html


def export_p8_pdf_bytes(fields: P8Fields) -> bytes:
    """PDF via reportlab (Vietnamese-capable DejaVu fonts)."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        KeepTogether,
    )

    if _BUNDLED_FONT.exists():
        pdfmetrics.registerFont(TTFont("DejaVu", str(_BUNDLED_FONT)))
        bold_name = "DejaVu"
        if _BUNDLED_FONT_BOLD.exists():
            pdfmetrics.registerFont(TTFont("DejaVuBold", str(_BUNDLED_FONT_BOLD)))
            bold_name = "DejaVuBold"
    else:
        bold_name = "Helvetica-Bold"

    font = "DejaVu" if _BUNDLED_FONT.exists() else "Helvetica"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="P.8 ACTD ASEAN (draft)",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="P8Title", fontName=bold_name, fontSize=12, leading=15, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="P8H2", fontName=bold_name, fontSize=11, leading=14, textColor=colors.HexColor("#1a4a7a"), spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="P8H3", fontName=bold_name, fontSize=10, leading=13, spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle(name="P8Body", fontName=font, fontSize=9, leading=12, alignment=TA_JUSTIFY, spaceAfter=3))
    styles.add(ParagraphStyle(name="P8Small", fontName=font, fontSize=8, leading=10, textColor=colors.HexColor("#444444")))
    styles.add(ParagraphStyle(name="P8Warn", fontName=font, fontSize=8, leading=11, backColor=colors.HexColor("#fff3cd"), borderPadding=4))
    styles.add(ParagraphStyle(name="P8Cell", fontName=font, fontSize=7.5, leading=9))

    def P(text: str, style="P8Body"):
        return Paragraph(_nz(text).replace("\n", "<br/>"), styles[style])

    def simple_table(headers: List[str], rows: List[List[str]], col_widths=None):
        data = [[Paragraph(_nz(h), styles["P8Cell"]) for h in headers]]
        if not rows:
            data.append([Paragraph(PLACEHOLDER, styles["P8Cell"])] + [""] * (len(headers) - 1))
        else:
            for r in rows:
                data.append([Paragraph(_nz(c), styles["P8Cell"]) for c in r])
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f8")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        return t

    story = []
    story.append(P(DISCLAIMER_VI + " " + DISCLAIMER_EN, "P8Warn"))
    story.append(Spacer(1, 6))
    story.append(P("P.8. ĐỘ ỔN ĐỊNH THÀNH PHẨM", "P8Title"))
    story.append(P("Mẫu báo cáo hồ sơ ACTD ASEAN (bản nháp hỗ trợ)", "P8Small"))
    story.append(Spacer(1, 6))

    h = fields.header
    meta = [
        ["Tên thuốc", _nz(h.ten_thuoc)],
        ["Dược chất - hàm lượng", _nz(h.duoc_chat_ham_luong)],
        ["Dạng bào chế", _nz(h.dang_bao_che)],
        ["Nhà sản xuất và địa điểm", _nz(h.nha_san_xuat)],
        ["Quy cách đóng gói", _nz(h.quy_cach_dong_goi)],
        ["Mã báo cáo/phiên bản", _nz(h.ma_bao_cao)],
        ["Ngày chốt dữ liệu", _nz(h.ngay_chot_du_lieu)],
        ["Người lập / kiểm / phê duyệt", _nz(h.nguoi_lap_kiem_phe_duyet)],
    ]
    story.append(simple_table(["Trường", "Giá trị"], meta, col_widths=[5.5 * cm, 11.5 * cm]))

    story.append(P("P.8.1. Tóm tắt và kết luận về độ ổn định", "P8H2"))
    story.append(P("A. Mục tiêu và thiết kế nghiên cứu", "P8H3"))
    story.append(P(fields.muc_tieu))
    story.append(P(f"Đề cương áp dụng: {_nz(fields.de_cuong)}."))
    story.append(
        simple_table(
            ["Số lô", "Ngày SX", "Cỡ lô", "Hàm lượng", "Bao bì", "Ngày BĐ"],
            [
                [r.get("so_lo"), r.get("ngay_sx"), r.get("co_lo"), r.get("ham_luong"), r.get("bao_bi"), r.get("ngay_bat_dau")]
                for r in fields.batch_design_rows
            ],
        )
    )
    story.append(
        simple_table(
            ["Nghiên cứu", "T / RH", "Lịch KT", "Đã có DL"],
            [
                [r.get("nghien_cuu"), r.get("nhiet_do_do_am"), r.get("lich_kiem_tra"), r.get("thoi_gian_du_lieu")]
                for r in fields.study_condition_rows
            ],
        )
    )
    story.append(P(f"Thiết kế rút gọn, nếu áp dụng: {_nz(fields.thiet_ke_rut_gon)}."))

    story.append(P("B. Tổng hợp kết quả và đánh giá", "P8H3"))
    for label, val in [
        ("Các chỉ tiêu đã kiểm tra", fields.chi_tieu_da_kiem),
        ("Xu hướng thay đổi theo thời gian", fields.xu_huong),
        ("Khác biệt giữa các lô / hàm lượng / bao bì", fields.khac_biet_lo),
        ("Kết quả ngoài tiêu chuẩn / OOT / sai lệch", fields.ngoai_tieu_chuan),
        ("Phân tích thống kê / ngoại suy", fields.phan_tich_thong_ke),
        ("Giới hạn của dữ liệu hiện có", fields.gioi_han_du_lieu),
    ]:
        story.append(P(f"<b>{label}:</b> {_nz(val)}"))

    story.append(P("C. Kết luận", "P8H3"))
    story.append(
        P(
            f"Tại ngày chốt dữ liệu {_nz(h.ngay_chot_du_lieu)}, đã có dữ liệu dài hạn đến "
            f"{_nz(fields.dl_dai_han_thang)} tháng và cấp tốc đến {_nz(fields.dl_cap_toc_thang)} tháng."
        )
    )
    story.append(P(f"Hạn dùng đề xuất: {_nz(fields.han_dung_thang)} tháng."))
    story.append(P(f"Điều kiện bảo quản ghi nhãn: {_nz(fields.dieu_kien_bao_quan)}."))
    story.append(
        P(
            f"Bao bì áp dụng: {_nz(fields.bao_bi_ap_dung if fields.bao_bi_ap_dung != PLACEHOLDER else h.quy_cach_dong_goi)}."
        )
    )
    story.append(P(f"Hạn dùng sau mở nắp/pha hoàn nguyên: {_nz(fields.han_dung_sau_mo)}."))
    story.append(P(f"Cơ sở hỗ trợ đề xuất: {_nz(fields.co_so_ho_tro)}."))

    story.append(P("P.8.2. Đề cương và cam kết độ ổn định sau phê duyệt", "P8H2"))
    story.append(P("A. Đề cương tiếp tục nghiên cứu", "P8H3"))
    story.append(
        simple_table(
            ["Nội dung", "Kế hoạch"],
            [
                ["Các lô tiếp tục theo dõi", fields.lo_tiep_tuc],
                ["Các lô bổ sung", fields.lo_bo_sung],
                ["Bao bì/hàm lượng áp dụng", fields.bao_bi_ham_luong],
                ["Điều kiện bảo quản mẫu", fields.dk_bao_quan_mau],
                ["Mốc kiểm tra còn lại", fields.moc_kiem_tra_con],
                ["Chỉ tiêu và giới hạn chấp nhận", fields.chi_tieu_gioi_han],
                ["Phương pháp phân tích", fields.phuong_phap_phan_tich],
                ["Đơn vị chịu trách nhiệm", fields.don_vi_chiu_trach_nhiem],
            ],
            col_widths=[6 * cm, 11 * cm],
        )
    )
    story.append(P("B. Cam kết", "P8H3"))
    story.append(P(fields.cam_ket_text))
    story.append(P(f"Đại diện có thẩm quyền: {_nz(fields.dai_dien)}"))
    story.append(P(f"Chức danh - ngày ký - chữ ký: {_nz(fields.chuc_danh_ky)}"))

    story.append(P("P.8.3. Dữ liệu độ ổn định", "P8H2"))
    story.append(P("A. Tiêu chuẩn và phương pháp", "P8H3"))
    story.append(P(fields.tieu_chuan_phuong_phap_note))
    story.append(
        simple_table(
            ["Chỉ tiêu", "Giới hạn", "Đơn vị", "PP/phiên bản", "Thẩm định"],
            [
                [r.get("chi_tieu"), r.get("gioi_han"), r.get("don_vi"), r.get("phuong_phap"), r.get("tham_chieu")]
                for r in fields.spec_method_rows
            ],
        )
    )
    story.append(P("B. Bảng kết quả", "P8H3"))
    if not fields.results_tables:
        story.append(P(PLACEHOLDER))
    else:
        for block in fields.results_tables[:12]:  # cap pages in MVP
            story.append(
                P(
                    f"Số lô: {_nz(block.get('so_lo'))} · Bao bì: {_nz(block.get('bao_bi'))} · "
                    f"Điều kiện: {_nz(block.get('dieu_kien'))}"
                )
            )
            tps = block.get("timepoints") or []
            headers = ["Chỉ tiêu", "Giới hạn"] + [f"Mốc {t}" for t in tps]
            rows = []
            for r in block.get("rows") or []:
                rows.append(
                    [r.get("chi_tieu"), r.get("gioi_han")]
                    + [r.get(f"t_{t}", PLACEHOLDER) for t in tps]
                )
            story.append(simple_table(headers, rows))
        if len(fields.results_tables) > 12:
            story.append(P(f"(… còn {len(fields.results_tables) - 12} bảng — xem bản HTML đầy đủ.)"))

    story.append(P("C. Tài liệu kèm theo", "P8H3"))
    story.append(
        P(
            "Phiếu KQKN; đồ thị xu hướng / thống kê; hồ sơ sai lệch; "
            "thông tin phương pháp / thẩm định hoặc tham chiếu hồ sơ."
        )
    )

    story.append(P("P.8.4. Bảo đảm dây chuyền lạnh, nếu áp dụng", "P8H2"))
    story.append(P(fields.cold_chain))

    story.append(Spacer(1, 10))
    story.append(
        P(
            f"Xuất lúc {datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M ICT')} · ICH-STABILITY-GROK · "
            + DISCLAIMER_VI,
            "P8Small",
        )
    )

    doc.build(story)
    return buf.getvalue()


def demo_p8_fields() -> P8Fields:
    """Demo fields for sample export without full analysis."""
    return P8Fields(
        header=P8Header(
            ten_thuoc="Thuốc demo XYZ",
            duoc_chat_ham_luong="API 500 mg",
            dang_bao_che="Viên nén bao phim",
            nha_san_xuat="Nhà máy Demo — Việt Nam",
            quy_cach_dong_goi="Chai HDPE 30 viên",
            ma_bao_cao="P8-DEMO-001 / v0.1",
            ngay_chot_du_lieu=datetime.now(VN_TZ).strftime("%Y-%m-%d"),
            nguoi_lap_kiem_phe_duyet="[…] / […] / […]",
        ),
        muc_tieu="Đánh giá độ ổn định của sản phẩm Thuốc demo XYZ trong bao bì chai HDPE nhằm hỗ trợ hạn dùng và điều kiện bảo quản đề xuất.",
        de_cuong="ĐC-STAB-001, phiên bản 01",
        thiet_ke_rut_gon="Không áp dụng (full design).",
        chi_tieu_da_kiem="Assay, Impurity",
        xu_huong="Assay giảm nhẹ theo thời gian; Impurity tăng trong giới hạn.",
        phan_tich_thong_ke="OLS + CI một phía ICH Q1E (LT-Q1E) — hỗ trợ soạn thảo.",
        dl_dai_han_thang="12",
        dl_cap_toc_thang="6",
        han_dung_thang="24",
        dieu_kien_bao_quan="Không quá 30°C, tránh ẩm",
        bao_bi_ap_dung="Chai HDPE 30 viên",
        co_so_ho_tro="Dữ liệu mẫu demo từ ICH-STABILITY-GROK.",
        cam_ket_text="Cơ sở […] cam kết thực hiện chương trình độ ổn định theo đề cương […].",
        cold_chain="Không áp dụng — dạng bào chế không yêu cầu dây chuyền lạnh.",
        batch_design_rows=[
            {
                "so_lo": "DEMO-A",
                "ngay_sx": "2025-01-15",
                "co_lo": "pilot",
                "ham_luong": "500 mg",
                "bao_bi": "HDPE",
                "ngay_bat_dau": "2025-02-01",
            }
        ],
        study_condition_rows=[
            {
                "nghien_cuu": "Dài hạn",
                "nhiet_do_do_am": "30°C/75%RH",
                "lich_kiem_tra": "0,3,6,9,12",
                "thoi_gian_du_lieu": "12",
            },
            {
                "nghien_cuu": "Lão hóa cấp tốc",
                "nhiet_do_do_am": "40°C/75%RH",
                "lich_kiem_tra": "0,3,6",
                "thoi_gian_du_lieu": "6",
            },
        ],
        spec_method_rows=[
            {
                "chi_tieu": "Assay",
                "gioi_han": "90.0–110.0%",
                "don_vi": "% LC",
                "phuong_phap": "HPLC / v02",
                "tham_chieu": PLACEHOLDER,
            },
            {
                "chi_tieu": "Impurity",
                "gioi_han": "NMT 0.5%",
                "don_vi": "%",
                "phuong_phap": "HPLC / v02",
                "tham_chieu": PLACEHOLDER,
            },
        ],
    )
