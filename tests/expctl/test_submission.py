from __future__ import annotations

import pandas as pd

from expctl.evaluation.submission import validate_submission


def test_generic_submission_validator_accepts_custom_schema() -> None:
    test_df = pd.DataFrame({"row_id": ["a", "b"]})
    submission_df = pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "class": ["cat", "dog"],
        }
    )

    result = validate_submission(
        test_df=test_df,
        submission_df=submission_df,
        expected_columns=("row_id", "class"),
        id_col="row_id",
        prediction_col="class",
        valid_prediction_values=("cat", "dog"),
    )

    assert result.is_valid is True


def test_generic_submission_validator_reports_prediction_errors() -> None:
    test_df = pd.DataFrame({"row_id": ["a"]})
    submission_df = pd.DataFrame(
        {
            "row_id": ["a"],
            "class": [""],
        }
    )

    result = validate_submission(
        test_df=test_df,
        submission_df=submission_df,
        expected_columns=("row_id", "class"),
        id_col="row_id",
        prediction_col="class",
    )

    assert result.is_valid is False
    assert "prediction_values" in result.errors_by_check
