import json
import os
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def scenario_path():
    """Path to base SUMO scenario."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "scenarios", "base", "scenario.sumocfg")


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a minimal dataset directory for integration tests."""
    base_dir = Path(__file__).resolve().parents[2] / "scenarios" / "base"
    dataset_root = tmp_path / "dataset_v1"
    dataset_root.mkdir(parents=True, exist_ok=True)

    train_ids = ["train_000", "train_001"]
    eval_ids = ["eval_000"]
    manifest = {
        "dataset_id": dataset_root.name,
        "train_scenarios": train_ids,
        "eval_scenarios": eval_ids,
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    def copy_scenario(scenario_id: str, split: str) -> None:
        scenario_dir = dataset_root / split / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_dir / "scenario.sumocfg", scenario_dir / "scenario.sumocfg")
        shutil.copy2(base_dir / "network.net.xml", scenario_dir / "network.net.xml")
        shutil.copy2(base_dir / "vehicles.rou.xml", scenario_dir / "vehicles.rou.xml")

    for scenario_id in train_ids:
        copy_scenario(scenario_id, "train")
    for scenario_id in eval_ids:
        copy_scenario(scenario_id, "eval")

    return dataset_root
