# Kinetics / Arrhenius audit — ICH-STABILITY-GROK

**Reference book in `docs/kinetics_book.pdf`:** Shein-Chung Chow, *Statistical Design and Analysis of Stability Studies* (Chapman & Hall/CRC, 2007), especially **Chapter 2 — Accelerated Testing** (chemical kinetic orders + Arrhenius).  
Some pages are OCR-noisy; theory below also follows standard pharmaceutical degradation kinetics (Carstensen / ICH Q1A spirit).

**Scope of this audit:** `src/arrhenius.py` (pre-upgrade vs post-upgrade).  
**Label:** all kinetics features remain **SUPPORTIVE / exploratory** — they do **not** replace ICH Q1E long-term shelf-life.

---

## Legend

| Tag | Meaning |
|-----|---------|
| **Đúng** | Matches standard theory / book formulas |
| **Sai** | Incorrect or misleading vs theory (fixed or documented) |
| **Thiếu** | Missing vs book / common practice (addressed or MVP-limited) |

---

## A. Reaction orders (Chow §2.1)

| Item | Book / theory | Pre-upgrade | Post-upgrade | Verdict |
|------|---------------|-------------|--------------|---------|
| Zero-order \(Y=Y_0-k_0 t\) | Common for some solid-state / suspension cases | Rate proxy = \|OLS slope\| of response vs time | Same; explicit `kinetics_order="zero"` | **Đúng** |
| First-order \(\ln Y=\ln Y_0-k_1 t\) | Most common for drug assay | Documented as “enter externally”; not fitted | Estimate \|slope\| on \(\ln Y\) scale; shelf-life via \(\ln(Y_0/\mathrm{spec})/k\) | **Đúng** (added) |
| Second-order \(1/Y=1/Y_0+k_2 t\) | Rare in pharma | Absent | Optional MVP estimate + mean-crossing shelf life | **Đúng** (MVP; labeled rare) |
| Multi-product / complex pathways | Book notes complexity | Out of scope | Still out of scope | **Thiếu** (accepted MVP limit) |

---

## B. Arrhenius (Chow §2.1–2.2)

| Item | Book / theory | Pre-upgrade | Post-upgrade | Verdict |
|------|---------------|-------------|--------------|---------|
| \(\ln k=\ln A-E_a/(R T)\) | Core model | Implemented via OLS of \(\ln k\) vs \(1/T\) | Same + CI | **Đúng** |
| \(E_a=-(\mathrm{slope})\cdot R\) | Standard | Correct for kcal/kJ R options | Same | **Đúng** |
| Gas constant units | kcal or kJ | `R_KCAL` / `R_KJ` selectable | Same | **Đúng** |
| Through-origin degradation rate estimators (Chow 2.23–2.24) | Alternate estimator with \(Y(0)=100\) fixed | Used intercept OLS \|slope\| proxy | Still OLS with intercept (common exploratory practice) | **Thiếu** vs book’s fixed-\(Y_0=100\) estimator — documented assumption |
| Nonlinear / weighted Arrhenius | Discussed for better inference | Not implemented | Not implemented | **Thiếu** (MVP) |
| CI for \(E_a\), predicted \(k\) | Needed for tentative dating | Absent | t-based CI on params + mean \(\ln k\) → exp | **Đúng** (added; approximate) |

---

## C. Humidity / light (beyond temperature)

| Item | Theory / practice | Pre-upgrade | Post-upgrade | Verdict |
|------|-------------------|-------------|--------------|---------|
| RH in condition strings | ICH long-term / accelerated quote %RH | Parsed T only; RH ignored | `parse_rh_percent`; optional column | **Đúng** (parse) |
| Combined T+RH model | Often \(\ln k=\ln A-E_a/(RT)+B\cdot f(\mathrm{RH})\) (empirical) | Absent | MVP: \(+B\cdot(\mathrm{RH}/100)\) | **Đúng** (MVP empiric; assumptions documented) |
| Moisture sorption / solid-state | Physical models (GAB, etc.) | Absent | Absent | **Thiếu** (out of MVP) |
| Light / photolysis (Q1B spirit) | Exposure dose affects rate | Absent | Optional additive \(C\cdot\mathrm{light}\) term + UI fields | **Đúng** (crude MVP) |
| Full Q1B photostability design | Regulatory study design | Absent | Absent | **Thiếu** (not claimed) |

---

## D. Shelf-life projection from extrapolated \(k\)

| Item | Theory | Pre-upgrade | Post-upgrade | Verdict |
|------|--------|-------------|--------------|---------|
| Mean crossing from \(k(T)\) | Exploratory only | Zero-order mean crossing | Zero / first / second + optional band from \(k\) CI | **Đúng** for supportive use |
| ICH Q1E one-sided CI of mean | Primary regulatory path | Correctly **not** claimed | Still not claimed; SUPPORTIVE labels kept | **Đúng** |
| Using Arrhenius shelf life as LT-Q1E substitute | **Incorrect** regulatoryly | Disclaimer present | Disclaimer reinforced | Was **Sai** if misused — mitigated by labels |

---

## E. Summary of changes in this upgrade

1. **Đúng giữ nguyên:** classic Arrhenius OLS, R unit choice, SUPPORTIVE labeling, Q1E path untouched.  
2. **Thiếu → bổ sung:** first/second-order rate scales; RH parse + T+RH MVP; light term MVP; CI for \(E_a\) and predicted \(k\); exploratory shelf-life band.  
3. **Sai / rủi ro:** treating Arrhenius dating as Q1E — still prevented by UI/report disclaimers (not a formula bug).  
4. **Vẫn thiếu (MVP limits):** Chow through-origin \(Y_0=100\) estimator; nonlinear Arrhenius; full moisture / Q1B designs; pooling of kinetic replicates across labs.

---

## Assumptions users must accept

1. \(k\) from stability tables is an **order-scale OLS \|slope\| proxy**, not a validated kinetic mechanism proof.  
2. T+RH / light terms are **empirical linear-in-covariate MVPs** on the \(\ln k\) scale.  
3. CIs assume classical OLS normality on \(\ln k\); with few temperatures, intervals are wide / fragile.  
4. Projected dating is **supportive drafting / exploration only**.
