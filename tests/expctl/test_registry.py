from __future__ import annotations

from expctl.registry.experiment_registry import (
    default_row,
    infer_exp_id,
    infer_feature_set,
    upsert_registry_row,
)


def test_infer_exp_id_from_name() -> None:
    assert infer_exp_id("exp_123_my_model") == "exp_123"


def test_infer_feature_set_summarizes_text_and_derived_flags() -> None:
    config = {
        "features": {
            "numeric": ["height"],
            "categorical": ["phone_os"],
            "derived": {"numeric": ["bmi_proxy"]},
            "text": "self_intro",
            "ignore_text": False,
        },
        "preprocessing": {"use_missing_indicator": True},
    }
    assert (
        infer_feature_set(config)
        == "numeric+categorical+derived+missing_indicator+text;text_included"
    )


def test_upsert_registry_row_updates_existing_row() -> None:
    row = default_row()
    row["exp_id"] = "exp_001"
    row["exp_name"] = "exp_001_baseline"
    row["status"] = "done"

    updated = dict(row)
    updated["status"] = "promoted"

    rows, action = upsert_registry_row([row], updated)

    assert action == "updated"
    assert rows[0]["status"] == "promoted"
