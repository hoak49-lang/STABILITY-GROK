"""Professional QC/lab Streamlit styles for ICH-STABILITY-GROK.

Restrained pharma aesthetic: blues/grays, dense but readable (Minitab-adjacent).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

# Accent palette (CSS variables injected once)
_CSS = """
<style>
:root {
  --ich-navy: #1e3a5f;
  --ich-blue: #2c5f8a;
  --ich-accent: #3a7ca5;
  --ich-muted: #5a6a7a;
  --ich-border: #d0d7de;
  --ich-bg-card: #f6f8fa;
  --ich-bg-soft: #eef2f6;
  --ich-ok: #1a7f4b;
  --ich-warn: #9a6700;
  --ich-danger: #a40e26;
  --ich-radius: 8px;
}

/* App chrome */
.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
  max-width: 1280px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f0f4f8 0%, #e8eef4 100%);
  border-right: 1px solid var(--ich-border);
}
section[data-testid="stSidebar"] .stRadio label {
  font-size: 0.92rem;
}

/* Brand header */
.ich-brand {
  font-family: "Segoe UI", system-ui, sans-serif;
  margin-bottom: 0.75rem;
}
.ich-brand h1 {
  font-size: 1.55rem !important;
  font-weight: 650 !important;
  color: var(--ich-navy) !important;
  margin: 0 0 0.15rem 0 !important;
  letter-spacing: -0.02em;
}
.ich-brand .subtitle {
  color: var(--ich-muted);
  font-size: 0.88rem;
  margin: 0;
}

/* Section headers */
.ich-section {
  display: flex;
  align-items: baseline;
  gap: 0.55rem;
  margin: 0.35rem 0 0.85rem 0;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid var(--ich-navy);
}
.ich-section .step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.55rem;
  height: 1.55rem;
  border-radius: 4px;
  background: var(--ich-navy);
  color: #fff;
  font-size: 0.78rem;
  font-weight: 700;
}
.ich-section .title {
  font-size: 1.12rem;
  font-weight: 650;
  color: var(--ich-navy);
}
.ich-section .en {
  font-size: 0.82rem;
  color: var(--ich-muted);
  font-weight: 500;
}

/* Metric cards row */
.ich-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.65rem;
  margin: 0.5rem 0 1rem 0;
}
.ich-card {
  background: var(--ich-bg-card);
  border: 1px solid var(--ich-border);
  border-radius: var(--ich-radius);
  padding: 0.7rem 0.85rem;
  box-shadow: 0 1px 2px rgba(30, 58, 95, 0.04);
  border-left: 3px solid var(--ich-accent);
}
.ich-card.ok { border-left-color: var(--ich-ok); }
.ich-card.warn { border-left-color: var(--ich-warn); }
.ich-card.danger { border-left-color: var(--ich-danger); }
.ich-card.neutral { border-left-color: var(--ich-muted); }
.ich-card .label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--ich-muted);
  font-weight: 600;
  margin-bottom: 0.2rem;
}
.ich-card .value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--ich-navy);
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
}
.ich-card .hint {
  font-size: 0.75rem;
  color: var(--ich-muted);
  margin-top: 0.15rem;
}

/* Empty state */
.ich-empty {
  background: var(--ich-bg-soft);
  border: 1px dashed var(--ich-border);
  border-radius: var(--ich-radius);
  padding: 1.75rem 1.25rem;
  text-align: center;
  color: var(--ich-muted);
  margin: 0.75rem 0 1rem 0;
}
.ich-empty .icon {
  font-size: 1.6rem;
  margin-bottom: 0.35rem;
}
.ich-empty .msg {
  font-size: 0.95rem;
  font-weight: 560;
  color: var(--ich-navy);
}
.ich-empty .sub {
  font-size: 0.82rem;
  margin-top: 0.25rem;
}

/* Status pill */
.ich-pill {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 650;
  letter-spacing: 0.02em;
}
.ich-pill.ok { background: #dafbe1; color: var(--ich-ok); }
.ich-pill.warn { background: #fff8c5; color: var(--ich-warn); }
.ich-pill.danger { background: #ffebe9; color: var(--ich-danger); }
.ich-pill.info { background: #ddf4ff; color: var(--ich-blue); }

/* Callout / disclaimer strip */
.ich-callout {
  background: #fff8e8;
  border: 1px solid #e6d5a8;
  border-left: 3px solid var(--ich-warn);
  border-radius: var(--ich-radius);
  padding: 0.65rem 0.85rem;
  font-size: 0.84rem;
  color: #5c4a1a;
  margin: 0.5rem 0 0.85rem 0;
}
.ich-callout.danger {
  background: #fff5f5;
  border-color: #f0c0c0;
  border-left-color: var(--ich-danger);
  color: #6e1a1a;
}
.ich-callout.info {
  background: #f0f6fb;
  border-color: #c5d6e6;
  border-left-color: var(--ich-accent);
  color: #1e3a5f;
}

/* Compact data tables feel denser */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--ich-border);
  border-radius: var(--ich-radius);
}

/* Primary button emphasis */
div.stButton > button[kind="primary"] {
  background-color: var(--ich-navy) !important;
  border-color: var(--ich-navy) !important;
}
div.stButton > button[kind="primary"]:hover {
  background-color: var(--ich-blue) !important;
  border-color: var(--ich-blue) !important;
}

/* Hide Streamlit chrome noise slightly */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""


def inject_styles() -> None:
    """Inject custom CSS once per run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section_header(step: Optional[str], title_vi: str, title_en: str) -> None:
    """Render a numbered section header (Vietnamese primary + English)."""
    step_html = f'<span class="step">{step}</span>' if step else ""
    st.markdown(
        f"""
        <div class="ich-section">
          {step_html}
          <span class="title">{title_vi}</span>
          <span class="en">/ {title_en}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def brand_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="ich-brand">
          <h1>{title}</h1>
          <p class="subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message_vi: str, message_en: str, icon: str = "📋") -> None:
    st.markdown(
        f"""
        <div class="ich-empty">
          <div class="icon">{icon}</div>
          <div class="msg">{message_vi}</div>
          <div class="sub">{message_en}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info") -> None:
    cls = {"info": "info", "warn": "", "danger": "danger"}.get(kind, "info")
    st.markdown(f'<div class="ich-callout {cls}">{text}</div>', unsafe_allow_html=True)


def metric_cards(cards: list[dict]) -> None:
    """Render a row of metric cards.

    Each card: {label, value, hint?, tone?} where tone in ok|warn|danger|neutral.
    """
    parts = ['<div class="ich-metrics">']
    for c in cards:
        tone = c.get("tone", "neutral")
        hint = c.get("hint", "")
        hint_html = f'<div class="hint">{hint}</div>' if hint else ""
        parts.append(
            f"""
            <div class="ich-card {tone}">
              <div class="label">{c.get("label", "")}</div>
              <div class="value">{c.get("value", "—")}</div>
              {hint_html}
            </div>
            """
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def pill(text: str, tone: str = "info") -> str:
    return f'<span class="ich-pill {tone}">{text}</span>'
