"""HTML / PDF report export for stability analysis.

Primary PDF path: reportlab (pip-only, no system libs).
Optional fallback: WeasyPrint HTML→PDF when installed with system deps.
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt

VN_TZ = timezone(timedelta(hours=7))

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FONT = _PACKAGE_ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
_BUNDLED_FONT_BOLD = _PACKAGE_ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf"

DISCLAIMER_EN = (
    "This tool provides supportive statistical analysis aligned with ICH Q1A(R2) "
    "and ICH Q1E principles. It is NOT a certified regulatory submission system "
    "and does not constitute regulatory advice."
)
DISCLAIMER_VI = (
    "Công cụ này hỗ trợ phân tích thống kê theo tinh thần ICH Q1A(R2)/Q1E — "
    "không phải hệ thống được chứng nhận cho hồ sơ đăng ký."
)
METHODS_NOTE = (
    "Methods / Phương pháp: Shelf life = time where the one-sided confidence "
    "bound of the mean meets the specification (ICH Q1E). Multi-attribute "
    "overall shelf life = minimum of attribute shelf lives (conservative). "
    "References: ICH Q1A(R2) Stability Testing of New Drug Substances and "
    "Products; ICH Q1E Evaluation of Stability Data."
)


def _fig_to_png_bytes(fig: plt.Figure, dpi: int = 120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _fig_to_base64(fig: plt.Figure) -> str:
    # Do not close the figure — Streamlit session may reuse it for PDF/UI.
    return base64.b64encode(_fig_to_png_bytes(fig)).decode("ascii")


def build_html_report(
    title: str,
    meta: Dict[str, Any],
    coef_html: str,
    anova_html: str,
    shelf_html: str,
    pool_html: str = "",
    figures: Optional[List[plt.Figure]] = None,
    notes: str = "",
    attribute_sections: Optional[List[Dict[str, Any]]] = None,
    summary_html: str = "",
) -> str:
    """Assemble a self-contained HTML report.

    Optional multi-attribute extras:
      - summary_html: overall summary table
      - attribute_sections: list of dicts with keys
        name, meta_html, coef_html, anova_html, shelf_html, figures
    """
    now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M ICT")
    fig_blocks = ""
    if figures:
        for i, fig in enumerate(figures):
            b64 = _fig_to_base64(fig)
            fig_blocks += (
                f'<div class="fig"><img src="data:image/png;base64,{b64}" '
                f'alt="figure {i+1}"/></div>\n'
            )

    meta_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in meta.items())

    attr_blocks = ""
    if attribute_sections:
        for idx, sec in enumerate(attribute_sections, start=1):
            name = sec.get("name", f"Attribute {idx}")
            sec_figs = ""
            for j, fig in enumerate(sec.get("figures") or []):
                b64 = _fig_to_base64(fig)
                sec_figs += (
                    f'<div class="fig"><img src="data:image/png;base64,{b64}" '
                    f'alt="{name} figure {j+1}"/></div>\n'
                )
            attr_blocks += f"""
<h2>{idx + 1}. Chỉ tiêu / Attribute: {name}</h2>
{sec.get('meta_html', '')}
<h3>Hệ số hồi quy / Coefficients</h3>
{sec.get('coef_html', '')}
<h3>ANOVA</h3>
{sec.get('anova_html', '')}
<h3>Shelf life / Thời hạn bảo quản</h3>
{sec.get('shelf_html', '')}
{sec_figs}
"""

    summary_block = ""
    if summary_html:
        summary_block = (
            f"<h2>1b. Tóm tắt đa chỉ tiêu / Multi-attribute summary</h2>\n"
            f"{summary_html}\n"
        )

    overall_box = ""
    if attribute_sections:
        overall_val = meta.get(
            "Overall shelf life (months)", meta.get("Shelf life (months)", "N/A")
        )
        overall_box = (
            "<div class='overall'><strong>Overall product shelf life / "
            "Thời hạn bảo quản tổng thể (min các chỉ tiêu):</strong> "
            f"{overall_val}</div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ color: #0b3d5c; }}
h2 {{ color: #125a8a; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
h3 {{ color: #1a6a9a; }}
table {{ border-collapse: collapse; margin: 12px 0; width: 100%; max-width: 900px; }}
th, td {{ border: 1px solid #bbb; padding: 6px 10px; text-align: left; }}
th {{ background: #e8f1f8; }}
.disclaimer {{ background: #fff3cd; border: 1px solid #ffc107; padding: 12px; margin: 16px 0; }}
.overall {{ background: #e8f5e9; border: 1px solid #81c784; padding: 12px; margin: 16px 0; }}
.fig img {{ max-width: 100%; height: auto; margin: 8px 0; }}
.meta table {{ max-width: 600px; }}
footer {{ margin-top: 32px; font-size: 12px; color: #666; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p><em>Generated / Tạo lúc: {now}</em></p>
<div class="disclaimer">
<strong>Disclaimer / Tuyên bố:</strong>
{DISCLAIMER_EN}
{DISCLAIMER_VI}
</div>
<h2>1. Thiết lập nghiên cứu / Study settings</h2>
<div class="meta"><table>{meta_rows}</table></div>
{summary_block}
{overall_box}
<h2>2. Hệ số hồi quy / Regression coefficients</h2>
{coef_html}
<h2>3. ANOVA</h2>
{anova_html}
<h2>4. Ước tính thời hạn bảo quản / Shelf-life estimate</h2>
{shelf_html}
{('<h2>5. Poolability (ANCOVA)</h2>' + pool_html) if pool_html else ''}
{attr_blocks}
<h2>Biểu đồ / Plots</h2>
{fig_blocks if fig_blocks else '<p><em>Plots are available in the app UI / Biểu đồ xem trong ứng dụng.</em></p>'}
<h2>Ghi chú / Notes</h2>
<p>{notes}</p>
<p>{METHODS_NOTE}</p>
<footer>
References: ICH Q1A(R2) Stability Testing of New Drug Substances and Products;
ICH Q1E Evaluation of Stability Data.
ICH-STABILITY-GROK — supportive analysis only / chỉ hỗ trợ phân tích.
</footer>
</body>
</html>
"""
    return html


def html_to_pdf_bytes(html: str) -> Optional[bytes]:
    """Optional WeasyPrint fallback; return None if unavailable."""
    try:
        from weasyprint import HTML

        return HTML(string=html).write_pdf()
    except Exception:
        return None


def dataframe_to_html(df) -> str:
    if df is None:
        return "<p>(none)</p>"
    try:
        return df.to_html(index=False, float_format=lambda x: f"{x:.6g}")
    except Exception:
        return df.to_html(index=False)


def _resolve_fonts() -> tuple[str, str]:
    """Return (regular, bold) TTF paths for Unicode / Vietnamese text."""
    candidates_reg = [
        str(_BUNDLED_FONT),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    candidates_bold = [
        str(_BUNDLED_FONT_BOLD),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    reg = next((p for p in candidates_reg if os.path.isfile(p)), None)
    bold = next((p for p in candidates_bold if os.path.isfile(p)), reg)
    if reg is None:
        raise RuntimeError(
            "No Unicode TTF font found for PDF export. "
            "Expected bundled assets/fonts/DejaVuSans.ttf."
        )
    return reg, bold or reg


def _fmt_cell(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if val != val:  # NaN
            return "—"
        return f"{val:.6g}"
    return str(val)


def _df_to_table_data(df) -> List[List[str]]:
    if df is None or getattr(df, "empty", True):
        return [["(none)"]]
    cols = [str(c) for c in df.columns]
    rows = [cols]
    for _, row in df.iterrows():
        rows.append([_fmt_cell(v) for v in row.tolist()])
    return rows


def build_pdf_report(
    title: str,
    meta: Dict[str, Any],
    *,
    summary_df=None,
    coef_df=None,
    anova_df=None,
    shelf_text: str = "",
    pool_df=None,
    attribute_sections: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[Sequence[Any]] = None,
    notes: str = "",
) -> bytes:
    """Build a native PDF report with reportlab (pip-only dependency).

    ``attribute_sections`` items may include:
      name, meta_text|meta_html, coef_df, anova_df, shelf_text|shelf_html, figures
    Figures may be matplotlib Figures or raw PNG bytes.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
    )

    font_reg, font_bold = _resolve_fonts()
    pdfmetrics.registerFont(TTFont("ReportSans", font_reg))
    pdfmetrics.registerFont(TTFont("ReportSans-Bold", font_bold))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=title,
        author="ICH-STABILITY-GROK",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1VN",
            fontName="ReportSans-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0b3d5c"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2VN",
            fontName="ReportSans-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#125a8a"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3VN",
            fontName="ReportSans-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1a6a9a"),
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyVN",
            fontName="ReportSans",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallVN",
            fontName="ReportSans",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#444444"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="DisclaimerVN",
            fontName="ReportSans",
            fontSize=8,
            leading=11,
            backColor=colors.HexColor("#fff8e1"),
            borderPadding=6,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="OverallVN",
            fontName="ReportSans-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1b5e20"),
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CenterVN",
            fontName="ReportSans",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        )
    )

    def P(text: str, style: str = "BodyVN") -> Paragraph:
        safe = (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        return Paragraph(safe, styles[style])

    def make_table(data: List[List[str]], col_widths=None) -> Table:
        # Wrap cells as Paragraphs for wrapping long headers
        wrapped = []
        for r_i, row in enumerate(data):
            wrow = []
            for cell in row:
                sty = "SmallVN"
                if r_i == 0 and len(data) > 1 and data[0] != ["(none)"]:
                    wrow.append(
                        Paragraph(
                            str(cell)
                            .replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"),
                            ParagraphStyle(
                                "th",
                                parent=styles["SmallVN"],
                                fontName="ReportSans-Bold",
                            ),
                        )
                    )
                else:
                    wrow.append(
                        Paragraph(
                            str(cell)
                            .replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"),
                            styles[sty],
                        )
                    )
            wrapped.append(wrow)
        t = Table(wrapped, colWidths=col_widths, repeatRows=1 if len(data) > 1 else 0)
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, -1), "ReportSans"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if len(data) > 1 and data[0] != ["(none)"]:
            style_cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f8")))
            style_cmds.append(("FONTNAME", (0, 0), (-1, 0), "ReportSans-Bold"))
        t.setStyle(TableStyle(style_cmds))
        return t

    def fig_flowable(fig_or_bytes: Any, max_width: float = 16 * cm) -> Optional[Image]:
        try:
            if fig_or_bytes is None:
                return None
            if isinstance(fig_or_bytes, (bytes, bytearray)):
                raw = bytes(fig_or_bytes)
            elif hasattr(fig_or_bytes, "savefig"):
                raw = _fig_to_png_bytes(fig_or_bytes)
            else:
                return None
            img_buf = io.BytesIO(raw)
            img = Image(img_buf)
            # Scale to fit page width
            iw, ih = img.imageWidth, img.imageHeight
            if iw <= 0 or ih <= 0:
                return None
            scale = min(1.0, max_width / float(iw))
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale
            return img
        except Exception:
            return None

    story: List[Any] = []
    now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M ICT")
    story.append(P(title, "H1VN"))
    story.append(P(f"Generated / Tạo lúc: {now}", "SmallVN"))
    story.append(Spacer(1, 4))
    story.append(
        P(
            f"<b>Disclaimer / Tuyên bố:</b> {DISCLAIMER_EN} {DISCLAIMER_VI}",
            "DisclaimerVN",
        )
    )

    story.append(P("1. Thiết lập nghiên cứu / Study settings", "H2VN"))
    meta_data = [["Tham số / Parameter", "Giá trị / Value"]]
    for k, v in meta.items():
        meta_data.append([str(k), _fmt_cell(v)])
    story.append(make_table(meta_data, col_widths=[8 * cm, 8 * cm]))

    if summary_df is not None and getattr(summary_df, "empty", True) is False:
        story.append(P("1b. Tóm tắt đa chỉ tiêu / Multi-attribute summary", "H2VN"))
        story.append(make_table(_df_to_table_data(summary_df)))

    overall = meta.get("Overall shelf life (months)", meta.get("Shelf life (months)"))
    if overall is not None and attribute_sections:
        story.append(
            P(
                f"Thời hạn bảo quản tổng thể / Overall product shelf life "
                f"(min các chỉ tiêu): {_fmt_cell(overall)} months",
                "OverallVN",
            )
        )

    story.append(P("2. Hệ số hồi quy / Regression coefficients", "H2VN"))
    if coef_df is not None:
        story.append(make_table(_df_to_table_data(coef_df)))
    else:
        story.append(P("(see per-attribute sections / xem mục chỉ tiêu)", "BodyVN"))

    story.append(P("3. ANOVA", "H2VN"))
    if anova_df is not None:
        story.append(make_table(_df_to_table_data(anova_df)))
    else:
        story.append(P("See per-attribute sections below. / Xem mục từng chỉ tiêu.", "BodyVN"))

    story.append(P("4. Ước tính thời hạn bảo quản / Shelf-life estimate", "H2VN"))
    story.append(P(shelf_text or "(n/a)", "BodyVN"))

    if pool_df is not None and getattr(pool_df, "empty", True) is False:
        story.append(P("5. Poolability (ANCOVA)", "H2VN"))
        story.append(make_table(_df_to_table_data(pool_df)))

    if attribute_sections:
        for idx, sec in enumerate(attribute_sections, start=1):
            name = sec.get("name", f"Attribute {idx}")
            story.append(P(f"{idx + 1}. Chỉ tiêu / Attribute: {name}", "H2VN"))
            meta_text = sec.get("meta_text") or sec.get("meta_html") or ""
            # Strip simple HTML tags if meta_html was passed
            if "<" in str(meta_text):
                import re

                meta_text = re.sub(r"<[^>]+>", " ", str(meta_text))
                meta_text = " ".join(meta_text.split())
            if meta_text:
                story.append(P(str(meta_text), "BodyVN"))

            story.append(P("Hệ số hồi quy / Coefficients", "H3VN"))
            cdf = sec.get("coef_df")
            if cdf is not None:
                story.append(make_table(_df_to_table_data(cdf)))
            elif sec.get("coef_html"):
                story.append(P("(table in HTML export)", "SmallVN"))

            story.append(P("ANOVA", "H3VN"))
            adf = sec.get("anova_df")
            if adf is not None:
                story.append(make_table(_df_to_table_data(adf)))

            story.append(P("Shelf life / Thời hạn bảo quản", "H3VN"))
            stxt = sec.get("shelf_text") or sec.get("shelf_html") or ""
            if "<" in str(stxt):
                import re

                stxt = re.sub(r"<[^>]+>", " ", str(stxt))
                stxt = " ".join(stxt.split())
            story.append(P(str(stxt) or "(n/a)", "BodyVN"))

            for j, fig in enumerate(sec.get("figures") or []):
                img = fig_flowable(fig)
                if img is not None:
                    story.append(Spacer(1, 4))
                    story.append(img)
                    story.append(P(f"Biểu đồ / Plot {j + 1} — {name}", "CenterVN"))

    story.append(P("Biểu đồ / Plots", "H2VN"))
    top_figs = list(figures or [])
    if top_figs:
        for i, fig in enumerate(top_figs):
            img = fig_flowable(fig)
            if img is not None:
                story.append(Spacer(1, 4))
                story.append(img)
                story.append(P(f"Biểu đồ / Plot {i + 1}", "CenterVN"))
    elif not attribute_sections:
        story.append(
            P(
                "Plots are available in the app UI / Biểu đồ xem trong ứng dụng.",
                "SmallVN",
            )
        )
    elif attribute_sections and not any(sec.get("figures") for sec in attribute_sections):
        story.append(
            P(
                "Plots are available in the app UI / Biểu đồ xem trong ứng dụng.",
                "SmallVN",
            )
        )

    story.append(P("Ghi chú / Notes", "H2VN"))
    if notes:
        story.append(P(notes, "BodyVN"))
    story.append(P(METHODS_NOTE, "SmallVN"))
    story.append(Spacer(1, 8))
    story.append(
        P(
            "ICH-STABILITY-GROK — supportive analysis only / chỉ hỗ trợ phân tích. "
            "Not a certified regulatory submission system.",
            "SmallVN",
        )
    )

    doc.build(story)
    pdf_bytes = buf.getvalue()
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError("reportlab produced empty or invalid PDF bytes")
    return pdf_bytes


def export_pdf_bytes(
    title: str,
    meta: Dict[str, Any],
    *,
    summary_df=None,
    coef_df=None,
    anova_df=None,
    shelf_text: str = "",
    pool_df=None,
    attribute_sections: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[Sequence[Any]] = None,
    notes: str = "",
    html: Optional[str] = None,
) -> bytes:
    """Primary: reportlab. Fallback: WeasyPrint from HTML if reportlab fails."""
    try:
        return build_pdf_report(
            title,
            meta,
            summary_df=summary_df,
            coef_df=coef_df,
            anova_df=anova_df,
            shelf_text=shelf_text,
            pool_df=pool_df,
            attribute_sections=attribute_sections,
            figures=figures,
            notes=notes,
        )
    except Exception:
        if html:
            pdf = html_to_pdf_bytes(html)
            if pdf:
                return pdf
        raise
