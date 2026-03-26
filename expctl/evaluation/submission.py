from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_EXPECTED_COLUMNS: tuple[str, str] = ("id", "prediction")


@dataclass(frozen=True)
class SubmissionValidationResult:
    is_valid: bool
    errors_by_check: dict[str, list[str]]

    @property
    def error_count(self) -> int:
        return sum(len(messages) for messages in self.errors_by_check.values())


def load_validation_dataframe(path: Path) -> pd.DataFrame:
    """Load CSV as string table so value-format checks stay strict and deterministic."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalize_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def check_schema(
    submission_df: pd.DataFrame,
    expected_columns: Sequence[str] = DEFAULT_EXPECTED_COLUMNS,
) -> list[str]:
    errors: list[str] = []
    expected = list(expected_columns)
    actual = list(submission_df.columns)

    if actual == expected:
        return errors

    missing = [col for col in expected if col not in actual]
    extra = [col for col in actual if col not in expected]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if extra:
        errors.append(f"Unexpected extra columns: {extra}")
    if not missing and not extra:
        errors.append(f"Column order mismatch. Expected order={expected}, actual order={actual}")
    return errors


def check_row_count(test_df: pd.DataFrame, submission_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    n_test = int(test_df.shape[0])
    n_submission = int(submission_df.shape[0])

    if n_submission != n_test:
        errors.append(f"Row count mismatch: expected {n_test}, got {n_submission}")

    duplicated_rows = int(submission_df.duplicated().sum())
    if duplicated_rows > 0:
        errors.append(f"Submission contains {duplicated_rows} duplicated rows")

    return errors


def check_id_alignment(
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    *,
    id_col: str = "id",
) -> list[str]:
    errors: list[str] = []
    if id_col not in test_df.columns:
        return [f"Test data missing id column '{id_col}'"]
    if id_col not in submission_df.columns:
        return [f"Submission missing id column '{id_col}'"]

    test_ids = test_df[id_col].map(_normalize_string)
    submission_ids = submission_df[id_col].map(_normalize_string)

    duplicated_submission_ids = submission_ids[submission_ids.duplicated(keep=False)]
    if not duplicated_submission_ids.empty:
        duplicated_values = list(dict.fromkeys(duplicated_submission_ids.tolist()))
        errors.append(f"Submission id contains duplicates, sample={duplicated_values[:5]}")

    compare_n = min(len(test_ids), len(submission_ids))
    if compare_n == 0:
        return errors

    mismatch_idx = np.flatnonzero(
        test_ids.iloc[:compare_n].to_numpy() != submission_ids.iloc[:compare_n].to_numpy()
    )
    if mismatch_idx.size > 0:
        first = int(mismatch_idx[0])
        errors.append(
            "id order/value mismatch at row "
            f"{first}: expected '{test_ids.iat[first]}', got '{submission_ids.iat[first]}'"
        )
    return errors


def check_prediction_values(
    submission_df: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    valid_values: Iterable[str] | None = None,
) -> list[str]:
    if prediction_col not in submission_df.columns:
        return [f"Submission missing prediction column '{prediction_col}'"]

    errors: list[str] = []
    normalized = submission_df[prediction_col].map(_normalize_string)

    empty_mask = normalized.eq("")
    if bool(empty_mask.any()):
        indices = normalized.index[empty_mask].tolist()[:5]
        errors.append(
            f"{prediction_col} contains empty/null values at rows {indices} (showing up to 5)"
        )

    if valid_values is None:
        return errors

    valid_set = {str(v) for v in valid_values}
    invalid_mask = ~normalized.isin(valid_set)
    if bool(invalid_mask.any()):
        invalid_values = list(dict.fromkeys(normalized[invalid_mask].tolist()))
        errors.append(
            f"{prediction_col} has invalid values {invalid_values[:5]} "
            f"(allowed={sorted(valid_set)})"
        )

    return errors


def validate_submission(
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    *,
    expected_columns: Sequence[str] = DEFAULT_EXPECTED_COLUMNS,
    id_col: str = "id",
    prediction_col: str = "prediction",
    valid_prediction_values: Iterable[str] | None = None,
) -> SubmissionValidationResult:
    errors_by_check: dict[str, list[str]] = {}

    schema_errors = check_schema(submission_df, expected_columns=expected_columns)
    if schema_errors:
        errors_by_check["schema"] = schema_errors

    row_count_errors = check_row_count(test_df, submission_df)
    if row_count_errors:
        errors_by_check["row_count"] = row_count_errors

    id_alignment_errors = check_id_alignment(test_df, submission_df, id_col=id_col)
    if id_alignment_errors:
        errors_by_check["id_alignment"] = id_alignment_errors

    prediction_errors = check_prediction_values(
        submission_df,
        prediction_col=prediction_col,
        valid_values=valid_prediction_values,
    )
    if prediction_errors:
        errors_by_check["prediction_values"] = prediction_errors

    return SubmissionValidationResult(
        is_valid=(len(errors_by_check) == 0),
        errors_by_check=errors_by_check,
    )
