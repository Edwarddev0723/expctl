from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml

REGISTRY_COLUMNS = [
    "exp_id",
    "exp_name",
    "date",
    "stage",
    "parent_exp",
    "objective",
    "hypothesis",
    "change_scope",
    "split_version",
    "config_path",
    "artifact_dir",
    "model_family",
    "feature_set",
    "cv_score_mean",
    "cv_score_std",
    "delta_vs_baseline",
    "status",
    "decision",
    "owner",
    "notes",
]


def read_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            normalized = {key: value for key, value in dict(row).items() if key is not None}
            if not any(str(value).strip() for value in normalized.values()):
                continue
            if not str(normalized.get("exp_id", "")).strip():
                continue
            rows.append(normalized)
    return rows


def write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in REGISTRY_COLUMNS})


def to_repo_rel(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def infer_exp_name(config_path: Path) -> str:
    return config_path.stem


def infer_exp_id(exp_name: str) -> str:
    match = re.match(r"^(exp_\d{3})", exp_name)
    if not match:
        raise ValueError(
            f"Cannot infer exp_id from exp_name '{exp_name}'. Expected prefix like exp_001_*"
        )
    return match.group(1)


def infer_stage(exp_name: str) -> str:
    name = exp_name.lower()
    if "baseline" in name:
        return "baseline"
    if "ablation" in name or "missing_indicator" in name:
        return "ablation"
    return "exploration"


def infer_split_version(config: dict[str, Any]) -> str:
    split_path = Path(str(config["validation"]["split_path"]))
    return f"split::{split_path.stem}"


def infer_feature_set(config: dict[str, Any]) -> str:
    feature_cfg = config["features"]
    preprocessing_cfg = config.get("preprocessing", {})
    derived_cfg = feature_cfg.get("derived", {})

    parts: list[str] = []
    if feature_cfg.get("numeric"):
        parts.append("numeric")
    if feature_cfg.get("categorical"):
        parts.append("categorical")
    if derived_cfg.get("numeric") or derived_cfg.get("categorical"):
        parts.append("derived")
    if preprocessing_cfg.get("use_missing_indicator", False):
        parts.append("missing_indicator")
    if feature_cfg.get("text") and not feature_cfg.get("ignore_text", True):
        parts.append("text")
        suffix = "text_included"
    else:
        suffix = "text_excluded"
    if not parts:
        parts.append("none")
    return f"{'+'.join(parts)};{suffix}"


def resolve_summary_path(config: dict[str, Any], *, repo_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override if override.is_absolute() else (Path.cwd() / override).resolve()
    output_cfg = config["output"]
    output_dir = repo_root / str(output_cfg["dir"])
    return output_dir / str(output_cfg["summary_file"])


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}. Run train_baseline first or pass --summary."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid summary structure: {path}")
    return data


def pick_primary_metric(config: dict[str, Any], *, override: str | None) -> str:
    if override:
        return override
    metrics = list(config["evaluation"]["metrics"])
    if not metrics:
        raise ValueError("No metrics found in config.evaluation.metrics")
    return str(metrics[0])


def metric_value(summary: dict[str, Any], primary_metric: str, suffix: str) -> str:
    key = f"{primary_metric}_{suffix}"
    if key not in summary:
        raise KeyError(f"Summary missing key: {key}")
    return str(summary[key])


def default_row() -> dict[str, str]:
    return {col: "" for col in REGISTRY_COLUMNS}


def sort_key(row: dict[str, str]) -> tuple[int, str]:
    exp_id = row.get("exp_id", "")
    match = re.match(r"exp_(\d+)", exp_id)
    if match:
        return int(match.group(1)), exp_id
    return 10**9, exp_id


def upsert_registry_row(
    rows: list[dict[str, str]],
    row: dict[str, str],
    *,
    fail_if_exists: bool = False,
) -> tuple[list[dict[str, str]], str]:
    exp_id = row.get("exp_id", "")
    existing_idx = next((i for i, current in enumerate(rows) if current.get("exp_id") == exp_id), None)
    if existing_idx is not None and fail_if_exists:
        raise ValueError(f"Experiment already exists in registry: {exp_id}")

    updated_rows = list(rows)
    if existing_idx is None:
        updated_rows.append(dict(row))
        action = "inserted"
    else:
        updated_rows[existing_idx] = dict(row)
        action = "updated"
    updated_rows = sorted(updated_rows, key=sort_key)
    return updated_rows, action
