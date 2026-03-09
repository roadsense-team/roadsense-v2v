"""
Unit tests for eval dataset metadata helpers.
"""
import json
from pathlib import Path

import pytest

from ml import eval_dataset


def _write_dataset(
    tmp_path: Path,
    eval_scenario_ids: list[str],
) -> str:
    dataset_dir = tmp_path / "dataset"
    eval_dir = dataset_dir / "eval"
    eval_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "train_scenarios": [],
                "eval_scenarios": eval_scenario_ids,
            }
        ),
        encoding="utf-8",
    )
    return str(dataset_dir)


def test_load_eval_capabilities_reads_supported_ranks(tmp_path: Path) -> None:
    dataset_dir = Path(_write_dataset(tmp_path, ["eval_000", "eval_001"]))
    (dataset_dir / "eval_capability_audit.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "eval_000",
                        "peer_count": 2,
                        "supported_ranks_any_step": [1, 2],
                        "supported_steps_by_rank": {"1": [30], "2": [40]},
                        "failed_steps_by_rank": {},
                        "failure_reasons_by_rank": {},
                    },
                    {
                        "scenario_id": "eval_001",
                        "peer_count": 3,
                        "supported_ranks_any_step": [1, 3],
                        "supported_steps_by_rank": {"1": [30], "3": [42]},
                        "failed_steps_by_rank": {"2": [30, 31]},
                        "failure_reasons_by_rank": {
                            "2": {"target_not_found_strategy=fixed_rank_ahead": 2}
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    capabilities = eval_dataset.load_eval_capabilities(
        str(dataset_dir),
        ["eval_000", "eval_001"],
    )

    assert capabilities == {
        "eval_000": {1, 2},
        "eval_001": {1, 3},
    }


def test_load_eval_capabilities_rejects_missing_scenario_entries(tmp_path: Path) -> None:
    dataset_dir = Path(_write_dataset(tmp_path, ["eval_000", "eval_001"]))
    (dataset_dir / "eval_capability_audit.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "eval_000",
                        "peer_count": 2,
                        "supported_ranks_any_step": [1, 2],
                        "supported_steps_by_rank": {"1": [30], "2": [40]},
                        "failed_steps_by_rank": {},
                        "failure_reasons_by_rank": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing capability audit entries"):
        eval_dataset.load_eval_capabilities(
            str(dataset_dir),
            ["eval_000", "eval_001"],
        )
