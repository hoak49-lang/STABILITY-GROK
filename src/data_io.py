"""Data import/export and validation for stability studies.

Supports:
  - Single-attribute long format: batch, time, response, condition
  - Multi-attribute long format: + attribute / chỉ tiêu column
  - Wide format: multiple response columns mapped by the user
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["batch", "time", "response", "condition"]

# Generic response aliases only — attribute-specific names (assay, impurity)
# are handled separately so wide tables are not collapsed incorrectly.
COLUMN_ALIASES = {
    "batch": ["batch", "lot", "batch_id", "lo", "ma_lo", "lô"],
    "time": ["time", "month", "months", "t", "thoi_gian", "thời_gian", "time_months"],
    "response": ["response", "y", "value", "ket_qua", "kết_quả", "result"],
    "condition": ["condition", "storage", "storage_condition", "cond", "dieu_kien", "điều_kiện"],
    "attribute": [
        "attribute",
        "attr",
        "chi_tieu",
        "chỉ_tiêu",
        "chitieu",
        "parameter",
        "test",
        "test_name",
        "quality_attribute",
    ],
}

# Single-column fallbacks: if no generic "response", accept these as response
# only when exactly one is present (keeps sample/legacy CSVs working).
_SINGLE_RESPONSE_FALLBACKS = [
    "assay",
    "impurity",
    "content",
    "potency",
    "ham_luong",
    "hàm_lượng",
]

_ID_BLOCKLIST = {
    "batch",
    "lot",
    "batch_id",
    "lo",
    "ma_lo",
    "lô",
    "time",
    "month",
    "months",
    "t",
    "thoi_gian",
    "thời_gian",
    "time_months",
    "condition",
    "storage",
    "storage_condition",
    "cond",
    "dieu_kien",
    "điều_kiện",
    "attribute",
    "attr",
    "chi_tieu",
    "chỉ_tiêu",
    "chitieu",
    "parameter",
    "test",
    "test_name",
    "quality_attribute",
    "response",
    "y",
    "value",
    "ket_qua",
    "kết_quả",
    "result",
}


def _normalize_key(name: str) -> str:
    return str(name).lower().strip().replace(" ", "_")


def _normalize_columns(df: pd.DataFrame, *, map_response_fallbacks: bool = True) -> pd.DataFrame:
    """Map common column name variants to canonical names."""
    mapping = {}
    lower_map = {_normalize_key(c): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize_key(alias)
            if key in lower_map:
                mapping[lower_map[key]] = canonical
                break

    out = df.rename(columns=mapping)

    # Legacy / convenience: single assay-or-impurity column → response
    if map_response_fallbacks and "response" not in out.columns:
        present = []
        lower_map2 = {_normalize_key(c): c for c in out.columns}
        for fb in _SINGLE_RESPONSE_FALLBACKS:
            if fb in lower_map2:
                present.append(lower_map2[fb])
        if len(present) == 1:
            out = out.rename(columns={present[0]: "response"})
    return out


def detect_data_shape(df: pd.DataFrame) -> str:
    """
    Classify a raw (possibly un-normalized) dataframe.

    Returns:
      - "long_multi": has attribute column + response
      - "long_single": has response (or one fallback), no attribute
      - "wide": multiple numeric response-like columns
      - "unknown"
    """
    # Normalize WITHOUT collapsing assay/impurity → response
    norm = _normalize_columns(df.copy(), map_response_fallbacks=False)
    has_attr = "attribute" in norm.columns
    has_resp = "response" in norm.columns
    if has_attr and has_resp:
        return "long_multi"
    if has_attr and not has_resp:
        # attribute present but response missing — still multi if we can find values
        return "long_multi" if "response" in _normalize_columns(df.copy()).columns else "unknown"
    if has_resp and not has_attr:
        return "long_single"

    candidates = suggest_wide_response_columns(df)
    if len(candidates) >= 2:
        return "wide"
    if len(candidates) == 1:
        # Treat as single-attribute via fallback rename
        return "long_single"
    return "unknown"


def suggest_wide_response_columns(df: pd.DataFrame) -> List[str]:
    """Suggest columns that could be mapped as response attributes (wide format)."""
    out = []
    for c in df.columns:
        key = _normalize_key(c)
        if key in _ID_BLOCKLIST:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().sum() >= max(2, int(0.5 * len(df))):
            out.append(c)
    return out


def load_table(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel from a Streamlit UploadedFile or file path (raw)."""
    name = getattr(uploaded_file, "name", str(uploaded_file)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return df


def prepare_dataframe(df: pd.DataFrame, require_attribute: bool = False) -> pd.DataFrame:
    """Normalize columns, coerce types, drop empty rows (single- or multi-attribute long)."""
    df = _normalize_columns(df.copy(), map_response_fallbacks=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Thiếu cột bắt buộc / Missing columns: {missing}. "
            f"Cần / Need: {REQUIRED_COLUMNS}. Có / Have: {list(df.columns)}"
        )
    cols = list(REQUIRED_COLUMNS)
    has_attr = "attribute" in df.columns
    if require_attribute and not has_attr:
        raise ValueError("Thiếu cột attribute / chỉ tiêu / Missing attribute column.")
    if has_attr:
        cols = cols + ["attribute"]

    out = df[cols].copy()
    out["batch"] = out["batch"].astype(str).str.strip()
    out["condition"] = out["condition"].astype(str).str.strip()
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["response"] = pd.to_numeric(out["response"], errors="coerce")
    if has_attr:
        out["attribute"] = out["attribute"].astype(str).str.strip()
        out = out.dropna(subset=["time", "response", "batch", "condition", "attribute"])
        out = out.sort_values(["attribute", "condition", "batch", "time"]).reset_index(drop=True)
    else:
        out = out.dropna(subset=["time", "response", "batch", "condition"])
        out = out.sort_values(["condition", "batch", "time"]).reset_index(drop=True)
    return out


def wide_to_long(
    df: pd.DataFrame,
    response_columns: Union[Sequence[str], Dict[str, str]],
    *,
    batch_col: Optional[str] = None,
    time_col: Optional[str] = None,
    condition_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Melt wide columns into long format with attribute + response.

    ``response_columns``: list of column names (used as attribute labels) or
    dict {source_column: attribute_label}.
    """
    work = df.copy()
    # Resolve id columns via aliases (without collapsing assay/impurity)
    norm = _normalize_columns(work, map_response_fallbacks=False)

    if batch_col is not None:
        batch_series = work[batch_col]
    elif "batch" in norm.columns:
        batch_series = norm["batch"]
    else:
        raise ValueError("Missing batch column.")

    if time_col is not None:
        time_series = work[time_col]
    elif "time" in norm.columns:
        time_series = norm["time"]
    else:
        raise ValueError("Missing time column.")

    if condition_col is not None:
        condition_series = work[condition_col]
    elif "condition" in norm.columns:
        condition_series = norm["condition"]
    else:
        raise ValueError("Missing condition column.")

    if isinstance(response_columns, dict):
        col_map = dict(response_columns)
    else:
        col_map = {c: str(c) for c in response_columns}

    rows = []
    for src_col, attr_name in col_map.items():
        if src_col not in work.columns:
            raise ValueError(f"Response column not found: {src_col}")
        for i in range(len(work)):
            rows.append(
                {
                    "batch": batch_series.iloc[i],
                    "time": time_series.iloc[i],
                    "response": work[src_col].iloc[i],
                    "condition": condition_series.iloc[i],
                    "attribute": attr_name,
                }
            )
    long_df = pd.DataFrame(rows)
    return prepare_dataframe(long_df)


def list_attributes(df: pd.DataFrame) -> List[str]:
    """Return sorted unique attribute names; empty list if single-attribute."""
    if df is None or len(df) == 0:
        return []
    if "attribute" not in df.columns:
        return []
    return sorted(df["attribute"].astype(str).unique().tolist())


def split_by_attribute(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split a prepared dataframe into {attribute: df_without_attribute_col}.

    Single-attribute frames (no attribute column) are returned under key
    ``"(single)"``.
    """
    if df is None or len(df) == 0:
        return {}
    if "attribute" not in df.columns:
        return {"(single)": df.copy()}

    result: Dict[str, pd.DataFrame] = {}
    for name, sub in df.groupby(df["attribute"].astype(str), sort=True):
        piece = sub.drop(columns=["attribute"]).reset_index(drop=True)
        result[str(name)] = piece
    return result


def ensure_attribute_column(df: pd.DataFrame, default_name: str = "Response") -> pd.DataFrame:
    """Ensure long frame has an attribute column (for uniform multi-attr path)."""
    out = df.copy()
    if "attribute" not in out.columns:
        out["attribute"] = default_name
    return out


def validate_for_analysis(df: pd.DataFrame, min_points: int = 3) -> Tuple[bool, str]:
    """Basic checks before regression (one attribute slice)."""
    if df is None or len(df) == 0:
        return False, "Không có dữ liệu / No data."
    if len(df) < min_points:
        return False, f"Cần ít nhất {min_points} điểm / Need at least {min_points} points."
    if "time" not in df.columns or df["time"].nunique() < 2:
        return False, "Cần ít nhất 2 mốc thời gian / Need at least 2 distinct time points."
    if "response" not in df.columns:
        return False, "Thiếu cột response / Missing response column."
    return True, "OK"


def make_empty_entry_frame(n: int = 6, multi_attribute: bool = False) -> pd.DataFrame:
    """Template for manual data entry."""
    if multi_attribute:
        half = max(n // 2, 3)
        times = [0, 3, 6, 9, 12, 18, 24]
        rows_assay = {
            "batch": ["A"] * half,
            "time": (times * ((half // len(times)) + 1))[:half],
            "attribute": ["Assay"] * half,
            "response": [np.nan] * half,
            "condition": ["25C/60%RH"] * half,
        }
        rows_imp = {
            "batch": ["A"] * half,
            "time": (times * ((half // len(times)) + 1))[:half],
            "attribute": ["Impurity"] * half,
            "response": [np.nan] * half,
            "condition": ["25C/60%RH"] * half,
        }
        return pd.concat([pd.DataFrame(rows_assay), pd.DataFrame(rows_imp)], ignore_index=True)

    return pd.DataFrame(
        {
            "batch": ["A"] * n,
            "time": [0, 3, 6, 9, 12, 18][:n] + [0] * max(0, n - 6),
            "response": [np.nan] * n,
            "condition": ["25C/60%RH"] * n,
        }
    )


def filter_condition(df: pd.DataFrame, condition: Optional[str]) -> pd.DataFrame:
    if condition is None or condition == "(Tất cả / All)":
        return df.copy()
    return df[df["condition"] == condition].copy()
