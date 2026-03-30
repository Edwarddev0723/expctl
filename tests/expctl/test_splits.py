from __future__ import annotations

import json
from pathlib import Path

import pytest

from expctl.splits import build_stratified_folds, get_fold_indices, load_folds, save_folds


def test_build_stratified_folds_supports_shuffle_false() -> None:
    artifact = build_stratified_folds(
        [0, 1, 0, 1, 0, 1],
        protocol_name="skf_no_shuffle",
        n_splits=3,
        shuffle=False,
    )

    assert artifact["shuffle"] is False
    assert artifact["n_splits"] == 3
    assert len(artifact["folds"]) == 3

    train_idx, valid_idx = get_fold_indices(artifact, fold_id=0)
    assert len(train_idx) == 4
    assert len(valid_idx) == 2
    assert set(train_idx).isdisjoint(set(valid_idx))
    assert sorted(train_idx.tolist() + valid_idx.tolist()) == list(range(6))


def test_save_and_load_folds_round_trip(tmp_path: Path) -> None:
    artifact = build_stratified_folds(
        [0, 1, 0, 1, 0, 1, 0, 1],
        protocol_name="rskf_v1",
        n_splits=2,
        repeated=True,
        n_repeats=2,
        random_state=7,
    )
    output_path = tmp_path / "splits" / "artifact.json"

    save_folds(artifact, output_path)
    loaded = load_folds(output_path)

    assert loaded == artifact


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": 1}, "n_splits must be >= 2"),
        ({"repeated": True, "n_repeats": 1}, "repeated=True requires n_repeats >= 2"),
    ],
)
def test_build_stratified_folds_rejects_invalid_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_stratified_folds([0, 1, 0, 1], protocol_name="bad_split", **kwargs)


def test_load_folds_rejects_leaky_artifact(tmp_path: Path) -> None:
    path = tmp_path / "bad_artifact.json"
    path.write_text(
        json.dumps(
            {
                "protocol_name": "bad",
                "seed": 42,
                "n_splits": 2,
                "shuffle": True,
                "repeated": False,
                "n_repeats": 1,
                "n_samples": 4,
                "folds": [
                    {
                        "fold_id": 0,
                        "repeat_id": 0,
                        "train_indices": [0, 1, 2],
                        "valid_indices": [2, 3],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train/valid leakage detected"):
        load_folds(path)
