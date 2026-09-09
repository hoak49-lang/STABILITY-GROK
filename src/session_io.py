"""Export / import Streamlit session JSON for ICH-STABILITY-GROK.

Persists design, program, dataframe, model settings, and analysis meta
so users can recover work across browser refreshes or share a session.
Analysis objects (fitted models, matplotlib figures) are NOT pickled —
only serializable meta is stored; user must re-run analysis after import.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

VN_TZ = timezone(timedelta(hours=7))

SESSION_SCHEMA_VERSION = 1

# Keys that hold DataFrames / nested dicts we know how to serialize
_DF_KEY = "df"
_MATRIX_KEY = "sampling_matrix"


def _df_to_records(df: Optional[pd.DataFrame]) -> Optional[List[dict]]:
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        return None
    out = df.copy()
    # Normalize timestamps / numpy types via records
    return json.loads(out.to_json(orient="records", date_format="iso", default_handler=str))


def _records_to_df(records: Optional[List[dict]]) -> Optional[pd.DataFrame]:
    if records is None:
        return None
    if isinstance(records, pd.DataFrame):
        return records
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _analysis_meta(analysis: Any) -> Optional[dict]:
    """Strip non-JSON analysis payload down to meta for re-run hints."""
    if analysis is None:
        return None
    if not isinstance(analysis, dict):
        return {"note": "non-dict analysis omitted"}
    meta = {
        "kind": analysis.get("kind"),
        "multi_attr": analysis.get("multi_attr"),
        "condition": analysis.get("condition"),
        "condition_role": analysis.get("condition_role"),
        "method_tag": analysis.get("method_tag"),
        "alpha": analysis.get("alpha"),
        "alpha_pool": analysis.get("alpha_pool"),
        "t_max": analysis.get("t_max"),
        "pool_mode": analysis.get("pool_mode"),
        "message": analysis.get("message"),
        "spec_limit": analysis.get("spec_limit"),
        "transform": analysis.get("transform"),
        "note": (
            "Fitted models/figures not restored — re-run step 6 · Kết quả after import. "
            "Mô hình đã fit không được khôi phục — chạy lại bước Kết quả."
        ),
    }
    ma = analysis.get("ma")
    if ma is not None:
        try:
            meta["overall_shelf_life"] = getattr(ma, "overall_shelf_life", None)
            meta["limiting_attribute"] = getattr(ma, "limiting_attribute", None)
            meta["n_attributes"] = getattr(ma, "n_attributes", None)
            if hasattr(ma, "summary") and ma.summary is not None:
                meta["summary_records"] = _df_to_records(ma.summary)
        except Exception:
            pass
    return meta


def export_session(state: Dict[str, Any]) -> str:
    """Serialize selected session_state keys to a JSON string."""
    payload = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "exported_at": datetime.now(VN_TZ).isoformat(),
        "app": "ICH-STABILITY-GROK",
        "design": state.get("study_design"),
        "program": state.get("stability_program"),
        "sampling_matrix": _df_to_records(state.get("sampling_matrix")),
        "df": _df_to_records(state.get("df")),
        "data_label": state.get("data_label"),
        "manual_editor_df": _df_to_records(state.get("manual_editor_df")),
        "arr_editor_df": _df_to_records(state.get("arr_editor_df")),
        "model": {
            "condition": state.get("condition"),
            "spec_limit": state.get("spec_limit"),
            "direction": state.get("direction"),
            "transform": state.get("transform"),
            "alpha": state.get("alpha"),
            "alpha_pool": state.get("alpha_pool"),
            "t_max": state.get("t_max"),
            "pool_mode": state.get("pool_mode"),
            "default_spec": state.get("default_spec"),
            "default_dir": state.get("default_dir"),
            "attr_configs": state.get("attr_configs") or {},
            "selected_attributes": state.get("selected_attributes") or [],
        },
        "kinetics": {
            "order": state.get("kinetics_order", "zero"),
            "r_unit": state.get("arr_r_unit", "kcal"),
            "predict_temp_c": state.get("arr_predict_temp", 25.0),
            "include_rh": state.get("arr_include_rh", False),
            "include_light": state.get("arr_include_light", False),
            "predict_rh": state.get("arr_predict_rh", 60.0),
            "predict_light": state.get("arr_predict_light", 0.0),
        },
        "p8": state.get("p8_fields") or {},
        "analysis_meta": _analysis_meta(state.get("analysis")),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_session(raw: str | bytes | dict) -> Tuple[Dict[str, Any], List[str]]:
    """Parse session JSON → dict of session_state updates + warning messages."""
    warnings: List[str] = []
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = dict(raw)

    ver = data.get("schema_version", 0)
    if ver != SESSION_SCHEMA_VERSION:
        warnings.append(
            f"Schema version {ver} (expected {SESSION_SCHEMA_VERSION}) — importing best-effort."
        )

    updates: Dict[str, Any] = {}
    if data.get("design") is not None:
        updates["study_design"] = data["design"]
    if data.get("program") is not None:
        updates["stability_program"] = data["program"]

    mat = _records_to_df(data.get("sampling_matrix"))
    if mat is not None:
        updates["sampling_matrix"] = mat

    df = _records_to_df(data.get("df"))
    if df is not None and len(df) > 0:
        updates["df"] = df
        updates["data_label"] = data.get("data_label") or "Imported session"
    elif data.get("df") is not None:
        updates["df"] = df if df is not None else pd.DataFrame()
        updates["data_label"] = data.get("data_label")

    med = _records_to_df(data.get("manual_editor_df"))
    if med is not None:
        updates["manual_editor_df"] = med
    aed = _records_to_df(data.get("arr_editor_df"))
    if aed is not None:
        updates["arr_editor_df"] = aed

    model = data.get("model") or {}
    for k in (
        "condition",
        "spec_limit",
        "direction",
        "transform",
        "alpha",
        "alpha_pool",
        "t_max",
        "pool_mode",
        "default_spec",
        "default_dir",
        "attr_configs",
        "selected_attributes",
    ):
        if k in model and model[k] is not None:
            updates[k] = model[k]

    kin = data.get("kinetics") or {}
    if "order" in kin:
        updates["kinetics_order"] = kin["order"]
    if "r_unit" in kin:
        updates["arr_r_unit"] = kin["r_unit"]
    if "predict_temp_c" in kin:
        updates["arr_predict_temp"] = kin["predict_temp_c"]
    if "include_rh" in kin:
        updates["arr_include_rh"] = kin["include_rh"]
    if "include_light" in kin:
        updates["arr_include_light"] = kin["include_light"]
    if "predict_rh" in kin:
        updates["arr_predict_rh"] = kin["predict_rh"]
    if "predict_light" in kin:
        updates["arr_predict_light"] = kin["predict_light"]

    if data.get("p8"):
        updates["p8_fields"] = data["p8"]

    # Clear live analysis — models/figures not restored
    updates["analysis"] = None
    if data.get("analysis_meta"):
        updates["analysis_meta_imported"] = data["analysis_meta"]
        warnings.append(
            "Analysis meta imported but fitted results cleared — re-run step 6. "
            "Đã nhập meta phân tích nhưng kết quả fit bị xóa — chạy lại bước 6."
        )

    return updates, warnings
