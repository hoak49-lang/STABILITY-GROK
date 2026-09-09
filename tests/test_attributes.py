"""Unit tests for multi-attribute aggregation and analysis."""

import numpy as np
import pandas as pd
import pytest

from src.attributes import (
    AttributeConfig,
    aggregate_overall_shelf_life,
    analyze_attributes,
    analyze_one_attribute,
    default_config_for_attribute,
)
from src.data_io import (
    detect_data_shape,
    list_attributes,
    prepare_dataframe,
    split_by_attribute,
    wide_to_long,
)


def test_aggregate_overall_is_minimum():
    assert aggregate_overall_shelf_life([24.0, 18.5, 30.0]) == 18.5
    assert aggregate_overall_shelf_life([12.0]) == 12.0
    assert aggregate_overall_shelf_life([None, 20.0, None]) == 20.0
    assert aggregate_overall_shelf_life([None, None]) is None
    assert aggregate_overall_shelf_life([]) is None


def test_aggregate_overall_with_zero():
    # Already OOS on one attribute → overall 0
    assert aggregate_overall_shelf_life([0.0, 36.0]) == 0.0


def _assay_df():
    t = [0, 3, 6, 9, 12, 18, 24]
    return pd.DataFrame(
        {
            "batch": ["A"] * len(t),
            "time": t,
            "response": [100.0 - 0.4 * x for x in t],
            "condition": ["25C/60%RH"] * len(t),
        }
    )


def _impurity_df():
    t = [0, 3, 6, 9, 12, 18, 24]
    return pd.DataFrame(
        {
            "batch": ["A"] * len(t),
            "time": t,
            "response": [0.05 + 0.02 * x for x in t],
            "condition": ["25C/60%RH"] * len(t),
        }
    )


def test_multi_attribute_overall_equals_min():
    data = {"Assay": _assay_df(), "Impurity": _impurity_df()}
    configs = [
        AttributeConfig("Assay", direction="decreasing", spec_limit=90.0),
        AttributeConfig("Impurity", direction="increasing", spec_limit=0.5),
    ]
    res = analyze_attributes(data, configs, alpha=0.05, t_max=60)
    assert res.n_attributes == 2
    lives = [a.reported_shelf_life for a in res.attributes]
    assert all(v is not None for v in lives)
    assert res.overall_shelf_life == pytest.approx(min(lives))
    assert res.limiting_attribute in ("Assay", "Impurity")
    assert len(res.summary) == 2


def test_single_attribute_path_still_works():
    data = {"Assay": _assay_df()}
    configs = [AttributeConfig("Assay", direction="decreasing", spec_limit=90.0)]
    res = analyze_attributes(data, configs, alpha=0.05, t_max=60)
    assert res.n_attributes == 1
    assert res.overall_shelf_life == res.attributes[0].reported_shelf_life
    assert res.attributes[0].kind == "single"


def test_default_config_heuristics():
    a = default_config_for_attribute("Assay")
    assert a.direction == "decreasing"
    assert a.spec_limit == 90.0
    i = default_config_for_attribute("Total Impurity")
    assert i.direction == "increasing"
    assert i.spec_limit == 0.5


def test_long_multi_attribute_csv_shape():
    raw = pd.DataFrame(
        {
            "batch": ["A", "A"],
            "time": [0, 0],
            "attribute": ["Assay", "Impurity"],
            "response": [100.0, 0.05],
            "condition": ["25C/60%RH", "25C/60%RH"],
        }
    )
    assert detect_data_shape(raw) == "long_multi"
    df = prepare_dataframe(raw)
    assert list_attributes(df) == ["Assay", "Impurity"]
    parts = split_by_attribute(df)
    assert set(parts.keys()) == {"Assay", "Impurity"}
    assert "attribute" not in parts["Assay"].columns


def test_wide_to_long_mapping():
    wide = pd.DataFrame(
        {
            "batch": ["A", "A"],
            "time": [0, 3],
            "assay": [100.0, 99.5],
            "impurity": [0.05, 0.12],
            "condition": ["25C/60%RH", "25C/60%RH"],
        }
    )
    assert detect_data_shape(wide) == "wide"
    long_df = wide_to_long(wide, {"assay": "Assay", "impurity": "Impurity"})
    assert set(list_attributes(long_df)) == {"Assay", "Impurity"}
    assert len(long_df) == 4


def test_analyze_one_attribute_multibatch():
    rows = []
    for b, off in [("A", 0.0), ("B", 0.1)]:
        for t in [0, 3, 6, 9, 12, 18, 24]:
            rows.append(
                {
                    "batch": b,
                    "time": t,
                    "response": 100.0 + off - 0.35 * t,
                    "condition": "25C/60%RH",
                }
            )
    df = pd.DataFrame(rows)
    cfg = AttributeConfig("Assay", direction="decreasing", spec_limit=90.0)
    res = analyze_one_attribute(df, cfg, alpha=0.05, alpha_pool=0.25, t_max=60)
    assert res.kind == "multi"
    assert res.reported_shelf_life is not None
