# ICH-STABILITY-GROK

Pharmaceutical **stability / shelf-life** platform (Python + Streamlit), oriented to enterprise ICH workflows:

- **ICH Q1A(R2)** — stability testing conditions & program spirit  
- **ICH Q1D** — study design (Full / Bracketing / Matrixing) sampling plans  
- **ICH Q1E** — evaluation of stability data (OLS, one-sided CI of the mean vs spec)

Vietnamese UI labels + English ICH technical terms. Similar in spirit to Minitab *Stability / Shelf Life* analysis, extended with Study Design + Stability Program modules.

Supports **multi-attribute** analysis (assay + impurity + others). Overall product shelf life = **minimum** of attribute shelf lives (conservative ICH-style).

> **Disclaimer:** Supportive statistical / planning tool only. **NOT** a certified regulatory submission system and **does not** constitute regulatory certification or advice.

---

## ICH feature map

| Guideline | Scope in this app | Status |
|-----------|-------------------|--------|
| **ICH Q1A(R2)** | Storage conditions, climate zone context, long-term / intermediate / accelerated roles, program catalog | **Hỗ trợ một phần** (supportive framing + condition roles; not a full CTD module) |
| **ICH Q1D** | Study Design wizard: Full / Bracketing / Matrixing, sampling matrix (time × condition × strength × pack), CSV/JSON protocol | **Có** (planning MVP) |
| **ICH Q1E** | OLS shelf-life (one-sided CL vs spec), multi-batch poolability (ANCOVA), multi-attribute overall=min, plots, HTML/PDF reports | **Có** |
| Arrhenius / kinetics (exploratory) | zero/first/second order; ln(k)=ln(A)−Ea/(R·T) [+B·RH/100] [+C·light]; CI for Ea & k | **Hỗ trợ một phần** — **Arrhenius-supportive** only; see `docs/KINETICS_AUDIT.md` |
| **ACTD ASEAN P.8** | Draft P.8 HTML/PDF from app data (`[…]` placeholders) | **Hỗ trợ soạn thảo** — not approved HSĐK |
| Session JSON | Export/Import design+program+df+model+kinetics+P.8 meta | **Có** |

**Method tags (Results / Report):**

| Tag | Meaning |
|-----|---------|
| **LT-Q1E** | Long-term / điều kiện thường — **primary** shelf-life decision path |
| **LHCT** | Accelerated — supportive; optional bridging notes |
| **Arrhenius-supportive** | Exploratory kinetics — not a substitute for long-term Q1E |

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

Sample data is under `data/` (assay, impurity, multi-batch, multi-attribute, Arrhenius, **study design protocol**, **stability program**).

### Workflow (sidebar)

1. **Thiết kế / Study Design (Q1D)** — product info, Full/Bracketing/Matrixing, generate sampling matrix, export CSV/JSON  
2. **Chương trình / Stability Program** — product/study catalog, due reminders (lead 7/14/30), enter results, feed into analysis  
3. **Dữ liệu / Data** — CSV/Excel/manual or load from Program feed  
4. **Mô hình / Model** — specs, direction, transform; condition role → method tag  
5. **Gộp lô / Pooling** — ANCOVA poolability  
6. **Kết quả / Results** — Q1E shelf life + method tag  
7. **Báo cáo / Report** — Q1E HTML/PDF + **P.8 ACTD ASEAN** draft HTML/PDF  
8. **Arrhenius / Kinetics (exploratory)** — zero/first/second, T+RH/light MVP, CI — supportive only  
9. **Giới thiệu / About**  
Sidebar: **Export/Import session JSON** (persists across steps / refresh).

### How to use Study Design

1. Open **1 · Thiết kế / Study Design**.  
2. Enter product name, dosage form, strengths, packs, proposed storage, climate zone (I–IV).  
3. Choose **Full / Bracketing / Matrixing** and read wizard notes (prerequisites / risks).  
4. Set timepoints & conditions (long-term / intermediate / accelerated).  
5. Click **Generate sampling matrix** → download CSV or protocol JSON.  
6. Optionally **Send matrix to Stability Program**.

### How to use Stability Program

1. Open **2 · Chương trình / Stability Program** (or load `data/sample_stability_program.json`).  
2. Review dashboard: running studies, upcoming / due / overdue pulls.  
3. Enter results on incomplete pulls.  
4. On **Feed → Analysis**, select a study and **Load into Data step** for Q1E.  
5. Prefer a **long-term** condition for **LT-Q1E** primary decisions; treat accelerated as **LHCT**.

### Core Q1E features (unchanged engine)

1. Import CSV/Excel or manual entry (`batch`, `time`, `response`, `condition`; optional `attribute` / `chỉ tiêu`; optional `condition_type`)  
2. Wide-format upload mapping  
3. OLS with optional log/sqrt transform (**per attribute**)  
4. Shelf life = time where the **one-sided 95%** confidence bound of the **mean** meets the specification  
5. Multi-batch poolability via ANCOVA (default α = 0.25)  
6. Multi-attribute overall = **min**  
7. HTML + **PDF** report export  
8. Arrhenius exploratory with strong disclaimer  

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

### Ý nghĩa shelf life (Q1E)

Thời điểm biên tin cậy **một phía** của trung bình đáp ứng giao với giới hạn chất lượng (spec).  
Với **nhiều chỉ tiêu**, thời hạn bảo quản sản phẩm = **min** các shelf life chỉ tiêu.

**LT-Q1E** = quyết định chính từ điều kiện dài hạn. **LHCT** / **Arrhenius-supportive** chỉ mang tính hỗ trợ.

> Công cụ hỗ trợ — **không** phải chứng nhận quy định (regulatory certification).

---


---

## Session persistence (Streamlit)

**Problem fixed:** edits were lost when switching sidebar steps.

- Widgets use stable `key=` + `session_state`.
- Data step: Sample/Upload/Manual only overwrite on deliberate **Load** buttons.
- `data_editor` tables (`session_df_editor`, `manual_editor`, Arrhenius rates) persist in `session_state`.
- Sidebar **Export/Import session JSON** includes design, program, df, model settings, kinetics flags, P.8 fields, and analysis *meta* (fitted models must be re-run).

### Manual checklist

1. Step **3 · Data** → Load sample → edit a cell → **Save table edits**.  
2. Go to **4 · Model** → change spec / condition.  
3. Return to **3 · Data** → values still present (Current session data).  
4. Sidebar → Export session JSON → refresh → Import → Load session → data restored.

---

## P.8 ACTD ASEAN export

Template reference: `docs/P8.pdf` (also `docs/P8.txt`).

| P.8 section | Exporter source |
|-------------|-----------------|
| Header fields | Study Design / manual P.8 form / `[…]` |
| P.8.1 A design | Design + batch/condition tables from df |
| P.8.1 B results summary | Analysis attribute summary / trends |
| P.8.1 C conclusions | Overall shelf life, storage, coverage months |
| P.8.2 continuing protocol / commitments | Manual + placeholders |
| P.8.3 methods + results tables | Spec rows + batch×condition result tables from df |
| P.8.4 cold chain | Manual / default N/A |
| Attachments list | Static checklist |

**How to use:** run Q1E analysis → **7 · Report** → fill optional P.8 header → download **P.8 HTML/PDF**. Or click **Demo P.8** for sample-filled export.

**Disclaimer:** supportive drafting aid for HSĐK — **not** an approved dossier replacement.

---

## Kinetics upgrade

See `docs/KINETICS_AUDIT.md` (Đúng / Sai / Thiếu vs Chow 2007 Ch.2 + standard pharma kinetics).

- Orders: **zero / first / second** (second rare).  
- Arrhenius + optional **T+RH** (`B·RH/100`) and **light** term.  
- Approx. **95% CI** for Ea and predicted k (when residual df > 0).  
- Exploratory shelf-life mean crossing (+ band from k CI).  
- Remains **SUPPORTIVE** — does not replace Q1E long-term.

## Project layout

```
ICH-STABILITY-GROK/
  app.py
  requirements.txt
  README.md
  assets/fonts/          # DejaVuSans for Vietnamese PDF
  src/
    study_design.py      # ICH Q1D sampling matrix
    stability_program.py # multi-product program + due logic
    ui_design_program.py # Streamlit pages for Design/Program
    data_io.py           # + condition_type
    regression.py
    pooling.py
    attributes.py
    arrhenius.py         # kinetics + Arrhenius + T+RH/light MVP
    p8_report.py         # ACTD ASEAN P.8 HTML/PDF
    session_io.py        # session JSON export/import
    plots.py
    report.py
    ui_styles.py
  docs/
    P8.pdf / P8.txt
    kinetics_book.pdf
    KINETICS_AUDIT.md
  data/
    sample_*.csv / sample_*protocol*.json / sample_stability_program.json
  tests/
    test_*.py
```

## MVP limits

- No mandatory database (session_state + JSON/CSV upload/download).  
- Bracketing/matrixing helpers follow ICH Q1D *spirit* — scientific justification remains with the user.  
- Matrixing uses deterministic pseudo-random cell selection (seeded); not a full statistical design optimizer.  
- Due dates approximate months via mean month length.  
- Prediction interval (individual observation) is not used for shelf life (ICH Q1E mean CI).  
- Arrhenius / kinetics remain exploratory / SUPPORTIVE only.  
- T+RH and light terms are empirical MVPs (not full moisture sorption / ICH Q1B).  
- P.8 export is a drafting aid with `[…]` placeholders — not a validated CTD publisher.  
- Session JSON does not restore fitted analysis objects (re-run step 6).  

## PDF report export

Primary path: **reportlab** (pip-only). Report step offers HTML + PDF with method tags, multi-attribute summary, optional Q1D sampling plan excerpt, and ICH disclaimer.
