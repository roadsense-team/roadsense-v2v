"""
Unit tests for ScenarioManager (Phase 2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from envs.scenario_manager import ScenarioManager


def _write_manifest(dataset_dir: Path, train_ids: list[str], eval_ids: list[str]) -> None:
    manifest = {
        "dataset_id": dataset_dir.name,
        "train_scenarios": train_ids,
        "eval_scenarios": eval_ids,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _make_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_ids = ["train_000", "train_001", "train_002"]
    eval_ids = ["eval_000", "eval_001"]
    _write_manifest(dataset_dir, train_ids, eval_ids)

    for scenario_id in train_ids:
        scenario_dir = dataset_dir / "train" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "scenario.sumocfg").write_text(
            "<configuration></configuration>",
            encoding="utf-8",
        )
    for scenario_id in eval_ids:
        scenario_dir = dataset_dir / "eval" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "scenario.sumocfg").write_text(
            "<configuration></configuration>",
            encoding="utf-8",
        )


def test_load_manifest_success(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(dataset_dir, ["train_000"], ["eval_000"])

    manager = ScenarioManager(dataset_dir=str(dataset_dir), seed=1, mode="train")

    assert manager.manifest["train_scenarios"] == ["train_000"]
    assert manager.manifest["eval_scenarios"] == ["eval_000"]


def test_load_manifest_missing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        ScenarioManager(dataset_dir=str(dataset_dir), seed=1, mode="train")


def test_train_mode_random_selection(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir)

    manager = ScenarioManager(dataset_dir=str(dataset_dir), seed=42, mode="train")
    path, scenario_id = manager.select_scenario()

    assert scenario_id in manager.train_scenario_ids
    assert path.name == "scenario.sumocfg"


def test_train_mode_deterministic_with_seed(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir)

    manager_a = ScenarioManager(dataset_dir=str(dataset_dir), seed=7, mode="train")
    manager_b = ScenarioManager(dataset_dir=str(dataset_dir), seed=7, mode="train")

    sequence_a = [manager_a.select_scenario()[1] for _ in range(5)]
    sequence_b = [manager_b.select_scenario()[1] for _ in range(5)]

    assert sequence_a == sequence_b


def test_eval_mode_sequential(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir)

    manager = ScenarioManager(dataset_dir=str(dataset_dir), seed=1, mode="eval")

    first = manager.select_scenario()[1]
    second = manager.select_scenario()[1]

    assert first == "eval_000"
    assert second == "eval_001"


def test_eval_mode_cycles(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir)

    manager = ScenarioManager(dataset_dir=str(dataset_dir), seed=1, mode="eval")
    sequence = [manager.select_scenario()[1] for _ in range(3)]

    assert sequence == ["eval_000", "eval_001", "eval_000"]


def test_get_scenario_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir)

    manager = ScenarioManager(dataset_dir=str(dataset_dir), seed=1, mode="train")
    paths = manager.get_scenario_paths("train")

    assert len(paths) == len(manager.train_scenario_ids)
    assert all(path.is_absolute() for path in paths)
    assert all(path.name == "scenario.sumocfg" for path in paths)
