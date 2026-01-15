"""
Scenario management for dataset-based training.

Loads manifest from a dataset directory and provides deterministic
scenario selection for training and evaluation.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class ScenarioManager:
    """
    Manages scenario selection from a dataset directory.

    Attributes:
        dataset_dir: Path to the dataset root directory
        manifest: Loaded manifest dictionary
        train_scenario_ids: List of training scenario IDs
        eval_scenario_ids: List of evaluation scenario IDs
        rng: NumPy random generator for deterministic selection
    """

    def __init__(
        self,
        dataset_dir: str,
        seed: Optional[int] = None,
        mode: str = "train",
    ) -> None:
        """
        Initialize the scenario manager.

        Args:
            dataset_dir: Path to dataset root (contains manifest.json)
            seed: Random seed for reproducible scenario selection
            mode: "train" for randomized selection, "eval" for sequential

        Raises:
            FileNotFoundError: If manifest.json not found
            ValueError: If mode is invalid
        """
        if mode not in {"train", "eval"}:
            raise ValueError(f"Invalid mode: {mode}")

        self.dataset_dir = Path(dataset_dir).resolve()
        self.manifest = self.load_manifest(self.dataset_dir)
        self.train_scenario_ids: List[str] = self.manifest["train_scenarios"]
        self.eval_scenario_ids: List[str] = self.manifest["eval_scenarios"]
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self._eval_index = 0

    def load_manifest(self, dataset_dir: Path) -> dict:
        """
        Load and validate manifest.json.

        Args:
            dataset_dir: Path to dataset root

        Returns:
            Parsed manifest dictionary

        Raises:
            FileNotFoundError: If manifest.json missing
            json.JSONDecodeError: If manifest is invalid JSON
            KeyError: If required fields missing
        """
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")

        with manifest_path.open("r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)

        for key in ("train_scenarios", "eval_scenarios"):
            if key not in manifest:
                raise KeyError(f"Missing required manifest field: {key}")

        return manifest

    def _get_scenario_path(self, scenario_id: str) -> Path:
        if scenario_id.startswith("train_"):
            return self.dataset_dir / "train" / scenario_id / "scenario.sumocfg"
        if scenario_id.startswith("eval_"):
            return self.dataset_dir / "eval" / scenario_id / "scenario.sumocfg"
        raise ValueError(f"Unknown scenario ID format: {scenario_id}")

    def get_scenario_paths(self, mode: str) -> List[Path]:
        """
        Get list of scenario paths for the given mode.

        Args:
            mode: "train" or "eval"

        Returns:
            List of absolute paths to scenario.sumocfg files
        """
        if mode == "train":
            return [self._get_scenario_path(sid) for sid in self.train_scenario_ids]
        if mode == "eval":
            return [self._get_scenario_path(sid) for sid in self.eval_scenario_ids]
        raise ValueError(f"Invalid mode: {mode}")

    def select_scenario(self) -> Tuple[Path, str]:
        """
        Select the next scenario based on mode.

        For "train" mode: Random selection (with replacement)
        For "eval" mode: Sequential iteration (cycles after exhaustion)

        Returns:
            Tuple of (path_to_sumocfg, scenario_id)
        """
        if self.mode == "train":
            if not self.train_scenario_ids:
                raise ValueError("No train scenarios available.")
            scenario_id = str(self.rng.choice(self.train_scenario_ids))
        else:
            if not self.eval_scenario_ids:
                raise ValueError("No eval scenarios available.")
            scenario_id = self.eval_scenario_ids[self._eval_index]
            self._eval_index = (self._eval_index + 1) % len(self.eval_scenario_ids)

        return self._get_scenario_path(scenario_id), scenario_id

    def reset_eval_index(self) -> None:
        """Reset the evaluation scenario index to 0."""
        self._eval_index = 0

    def seed(self, seed: int) -> None:
        """Re-seed the random generator."""
        self.rng = np.random.default_rng(seed)
