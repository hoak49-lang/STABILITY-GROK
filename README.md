# ICH-STABILITY-GROK

Pharmaceutical **shelf-life / stability** prediction app (Python + Streamlit), aligned with **ICH Q1A(R2)** and **ICH Q1E**.

Vietnamese UI labels + English technical terms. Similar in spirit to Minitab *Stability / Shelf Life* analysis.

Supports **multi-attribute** analysis (assay + impurity + others) in one session. Overall product shelf life = **minimum** of attribute shelf lives (conservative ICH-style).

> **Disclaimer:** Supportive statistical tool only. **NOT** a certified regulatory submission system.

---

## English — How to run

```bash
cd ICH-STABILITY-GROK
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tests:

```bash
pytest -q
```

Sample data is under `data/`. Open the app, choose a sample dataset (including **multi-attribute**), set per-attribute specs, and click **Run analysis**.

### Features
1. Import CSV/Excel or manual entry (`batch`, `time`, `response`, `condition`; optional `attribute` / `chỉ tiêu`)
2. Wide-format upload: map multiple response columns → attributes
3. OLS regression with optional log/sqrt transform (**per attribute**)
4. Shelf life = time where the **one-sided 95%** confidence bound of the **mean** meets the specification
5. Multi-batch poolability via ANCOVA (default α = 0.25); pooled vs separate; conservative minimum
6. **Multi-attribute:** independent ICH analysis per attribute; overall = **min** of attribute shelf lives
7. Minitab-like tables (coefficients, ANOVA, R²) + residual diagnostics + plots
8. Optional **Arrhenius** section (exploratory / supportive only): fit ln(k)=ln(A)−Ea/(R·T)
   from user-entered rates or |slope| proxies; selectable R (kcal/kJ); predict k + optional
   zero-order shelf-life projection at long-term T — clearly labeled SUPPORTIVE
9. HTML + **PDF** report export (reportlab, pip-only) including all attributes, overall shelf life, key plots, ICH disclaimer
10. About/Methods citing ICH Q1A(R2) and Q1E

### Push to GitHub
```bash
git init
git add .
git commit -m "Initial ICH stability shelf-life app"
git branch -M main
git remote add origin https://github.com/hoak49-lang/ICH-STABILITY-GROK.git
git push -u origin main
```

---

## Tiếng Việt — Cách chạy

```bash
cd ICH-STABILITY-GROK
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Chạy kiểm thử: `pytest -q`

### Ý nghĩa shelf life
Thời điểm biên tin cậy **một phía** của trung bình đáp ứng giao với giới hạn chất lượng (spec):
- Assay giảm → lower bound gặp lower spec
- Impurity tăng → upper bound gặp upper spec

Với **nhiều chỉ tiêu**, thời hạn bảo quản sản phẩm = **min** các shelf life chỉ tiêu.

α poolability mặc định **0.25** (có thể chỉnh), theo tinh thần ICH Q1E.

---

## Project layout

```
ICH-STABILITY-GROK/
  app.py
  requirements.txt
  README.md
  assets/fonts/     # DejaVuSans for Vietnamese PDF text
  src/
    data_io.py
    regression.py
    pooling.py
    attributes.py   # multi-attribute orchestration
    arrhenius.py
    plots.py
    report.py       # HTML + reportlab PDF (+ optional WeasyPrint)
    ui_styles.py
  data/
    sample_*.csv    # multi-attribute + sample_arrhenius_rates/conditions.csv
  tests/
    test_*.py
```

## PDF report export

Primary path: **reportlab** (listed in `requirements.txt`) — no fragile system libraries.
Report step (5) offers **Download HTML** and **Download PDF**. PDF includes:

- Vietnamese / bilingual section headers
- Multi-attribute summary + per-attribute coefficients / ANOVA / shelf life
- Overall product shelf life (min of attributes)
- Embedded key plots when available
- ICH Q1A(R2) / Q1E citation and no-certification disclaimer

Optional: install WeasyPrint separately for HTML→PDF fallback.

## Limitations
- Assumes linear kinetics on the chosen scale (ICH Q1E common case).
- Prediction interval (individual future observation) is not used for shelf life; ICH Q1E focuses on the confidence bound for the mean.
- Arrhenius module is **exploratory / SUPPORTIVE only** — not a substitute for long-term ICH Q1A/Q1E evaluation.
- PDF export uses **reportlab** (pip-only; works after `pip install -r requirements.txt`). Bundled DejaVu fonts support Vietnamese labels. WeasyPrint remains an optional HTML→PDF fallback if system libraries are available. HTML export is unchanged.

## Arrhenius module (SUPPORTIVE / exploratory)

Model: **ln(k) = ln(A) − Ea/(R·T)** with selectable gas constant R in **kcal/(mol·K)** or **kJ/(mol·K)**.

**Inputs**
- User-entered table of temperature (°C) + degradation rate k, *or*
- Multi-condition stability data → estimate k ≈ **|OLS slope|** of response vs time per condition
  (documented **zero-order rate proxy** on the modeling scale).

**Outputs**
- Ea, ln(A), R², Arrhenius plot (ln(k) vs 1/T), predicted k at a chosen long-term T (e.g. 25°C)
- Optional projected shelf life at that T via **zero-order mean crossing** (intercept → spec),
  using current attribute/spec settings when available — with a **strong disclaimer**

**Assumptions (also shown in the UI)**
1. Derived k uses |slope| as a zero-order proxy; this is not full kinetic model selection.
2. First-order rates are not fitted inside the module (enter pre-computed k if needed).
3. Humidity / packaging / non-temperature factors are ignored.
4. Extrapolated shelf life is **not** the ICH Q1E one-sided CI-of-the-mean shelf life.
5. Results are supportive / exploratory only.

Sample files: `data/sample_arrhenius_rates.csv`, `data/sample_arrhenius_conditions.csv`.
