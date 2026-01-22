#!/usr/bin/env python3
"""
Scenario Generator for Cloud Training Automation.

Generates augmented SUMO scenarios from base templates with reproducible
randomization and train/eval splits.

Usage:
    python -m ml.scripts.gen_scenarios \
        --base_dir ml/scenarios/base \
        --output_dir ml/scenarios/datasets/dataset_v1 \
        --seed 42 \
        --train_count 20 \
        --eval_count 5
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np


AUGMENTATION_RANGES = {
    "speedFactor": (0.8, 1.2),
    "sigma": (0.0, 0.5),
    "decel": (3.5, 5.5),
    "tau": (0.5, 1.5),
    "spawn_jitter_s": (0.0, 2.0),
}


@dataclass
class AugmentationConfig:
    """Configuration for scenario augmentation features."""

    peer_drop_prob: float = 0.0
    min_peers: int = 1
    route_randomize_non_ego: bool = False
    route_include_v001: bool = False


def _strip_namespaces(tree: ET.ElementTree) -> None:
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _require_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def load_base_scenario(base_dir: Path) -> Tuple[ET.ElementTree, ET.ElementTree, ET.ElementTree]:
    """
    Load base scenario XML files.

    Args:
        base_dir: Path to base scenario directory

    Returns:
        Tuple of (sumocfg_tree, network_tree, routes_tree)

    Files read:
        - {base_dir}/scenario.sumocfg
        - {base_dir}/network.net.xml
        - {base_dir}/vehicles.rou.xml
    """
    sumocfg_path = base_dir / "scenario.sumocfg"
    network_path = base_dir / "network.net.xml"
    routes_path = base_dir / "vehicles.rou.xml"

    _require_file(sumocfg_path)
    _require_file(network_path)
    _require_file(routes_path)

    sumocfg_tree = ET.parse(sumocfg_path)
    network_tree = ET.parse(network_path)
    routes_tree = ET.parse(routes_path)

    _strip_namespaces(sumocfg_tree)
    _strip_namespaces(network_tree)
    _strip_namespaces(routes_tree)

    return sumocfg_tree, network_tree, routes_tree


def augment_routes(
    routes_tree: ET.ElementTree,
    rng: np.random.Generator,
    params: Dict[str, Tuple[float, float]],
    config: Optional[AugmentationConfig] = None,
) -> Tuple[ET.ElementTree, int]:
    """
    Apply augmentation to vehicles.rou.xml.

    Args:
        routes_tree: Parsed XML tree
        rng: NumPy random generator (seeded)
        params: Dict of {param_name: (min, max)} ranges
        config: Optional augmentation config for peer dropout and route randomization

    Returns:
        Tuple of (modified XML tree, number of peers remaining)

    Modifications:
        1. Apply peer dropout (if configured)
        2. Randomize routes (if configured)
        3. Modify <vType> attributes: speedFactor, sigma, decel, tau
        4. Jitter <vehicle depart="..."> times (except V001)

    INVARIANT: V001 depart MUST remain 0.
    """
    if config is None:
        config = AugmentationConfig()

    root = routes_tree.getroot()

    peer_count = apply_peer_dropout(
        root,
        rng,
        config.peer_drop_prob,
        config.min_peers,
    )

    if config.route_randomize_non_ego:
        randomize_routes(root, rng, config.route_include_v001)

    vtype_values = {
        name: rng.uniform(*params[name])
        for name in ("speedFactor", "sigma", "decel", "tau")
    }
    for vtype in root.findall("vType"):
        for attr, value in vtype_values.items():
            vtype.set(attr, _format_float(value))

    jitter_min, jitter_max = params["spawn_jitter_s"]
    for vehicle in root.findall("vehicle"):
        vehicle_id = vehicle.get("id")
        if vehicle_id == "V001":
            continue
        depart = vehicle.get("depart")
        if depart is None:
            continue
        try:
            base_depart = float(depart)
        except ValueError as exc:
            raise ValueError(f"Invalid depart time for {vehicle_id}: {depart}") from exc
        jitter = rng.uniform(jitter_min, jitter_max)
        vehicle.set("depart", _format_float(base_depart + jitter))

    _sort_vehicles_by_depart(root)

    return routes_tree, peer_count


def _sort_vehicles_by_depart(root: ET.Element) -> None:
    children = list(root)
    vehicles = [child for child in children if child.tag == "vehicle"]
    if not vehicles:
        return

    first_vehicle_index = next(
        (idx for idx, child in enumerate(children) if child.tag == "vehicle"),
        len(children),
    )

    for vehicle in vehicles:
        root.remove(vehicle)

    def _sort_key(vehicle: ET.Element) -> Tuple[float, str]:
        depart = vehicle.get("depart")
        try:
            depart_value = float(depart) if depart is not None else float("inf")
        except ValueError:
            depart_value = float("inf")
        vehicle_id = vehicle.get("id") or ""
        return (depart_value, vehicle_id)

    for offset, vehicle in enumerate(sorted(vehicles, key=_sort_key)):
        root.insert(first_vehicle_index + offset, vehicle)


def apply_peer_dropout(
    root: ET.Element,
    rng: np.random.Generator,
    drop_prob: float,
    min_peers: int,
) -> int:
    """
    Remove non-V001 vehicles with given probability, keeping at least min_peers.

    Args:
        root: Root element of vehicles.rou.xml
        rng: NumPy random generator (seeded)
        drop_prob: Probability of dropping each peer vehicle
        min_peers: Minimum number of peers to keep (excluding V001)

    Returns:
        Number of peers remaining after dropout

    INVARIANT: V001 is NEVER dropped.
    """
    if drop_prob <= 0.0:
        vehicles = [v for v in root.findall("vehicle") if v.get("id") != "V001"]
        return len(vehicles)

    vehicles = root.findall("vehicle")
    peers = [v for v in vehicles if v.get("id") != "V001"]

    if not peers:
        return 0

    candidates_to_drop = []
    for peer in peers:
        if rng.random() < drop_prob:
            candidates_to_drop.append(peer)

    max_drops = max(0, len(peers) - min_peers)
    to_drop = candidates_to_drop[:max_drops]

    for vehicle in to_drop:
        root.remove(vehicle)

    return len(peers) - len(to_drop)


def randomize_routes(
    root: ET.Element,
    rng: np.random.Generator,
    include_v001: bool,
) -> None:
    """
    Randomly assign routes to vehicles from available route definitions.

    Args:
        root: Root element of vehicles.rou.xml
        rng: NumPy random generator (seeded)
        include_v001: If True, also randomize V001's route

    NOTE: Only works if multiple <route> elements are defined in the file.
    """
    routes = root.findall("route")
    if len(routes) < 2:
        return

    route_ids = [r.get("id") for r in routes if r.get("id")]
    if not route_ids:
        return

    for vehicle in root.findall("vehicle"):
        vehicle_id = vehicle.get("id")
        if vehicle_id == "V001" and not include_v001:
            continue
        new_route = rng.choice(route_ids)
        vehicle.set("route", new_route)


def write_scenario(
    output_dir: Path,
    scenario_id: str,
    sumocfg_tree: ET.ElementTree,
    network_tree: ET.ElementTree,
    routes_tree: ET.ElementTree,
) -> Path:
    """
    Write augmented scenario to output directory.

    Args:
        output_dir: Dataset output directory
        scenario_id: Unique scenario identifier (e.g., "train_001")
        *_tree: XML trees to write

    Returns:
        Path to written scenario.sumocfg

    Directory structure created:
        {output_dir}/{scenario_id}/
        ├── scenario.sumocfg
        ├── network.net.xml
        └── vehicles.rou.xml
    """
    scenario_dir = output_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    sumocfg_path = scenario_dir / "scenario.sumocfg"
    network_path = scenario_dir / "network.net.xml"
    routes_path = scenario_dir / "vehicles.rou.xml"

    sumocfg_tree.write(sumocfg_path, encoding="utf-8")
    network_tree.write(network_path, encoding="utf-8")
    routes_tree.write(routes_path, encoding="utf-8")

    return sumocfg_path


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()[:16]}"


def _get_sumo_version() -> str:
    try:
        result = subprocess.run(
            ["sumo", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("SUMO is required to generate scenarios.") from exc

    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise RuntimeError("Unable to detect SUMO version.")
    first_line = output.splitlines()[0]
    match = re.search(r"Version\s+([0-9.]+)", first_line)
    if match:
        return match.group(1)
    return first_line


def _get_git_commit() -> str:
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _get_container_image() -> str:
    return os.environ.get("CONTAINER_IMAGE", "unknown")


def write_manifest(
    output_dir: Path,
    seed: int,
    base_dir: Path,
    emulator_params_path: Path,
    train_scenarios: List[str],
    eval_scenarios: List[str],
    augmentation_ranges: Dict[str, Tuple[float, float]],
    augmentation_config: Optional[AugmentationConfig] = None,
    peer_count_distribution: Optional[Dict[int, int]] = None,
) -> None:
    """
    Write manifest.json with full reproducibility metadata.

    Args:
        output_dir: Dataset output directory
        seed: Master random seed used
        base_dir: Path to base scenario
        emulator_params_path: Path to emulator params JSON
        train_scenarios: List of train scenario IDs
        eval_scenarios: List of eval scenario IDs
        augmentation_ranges: Dict of augmentation parameter ranges
        augmentation_config: Optional config for peer dropout and route randomization
        peer_count_distribution: Optional dict of {peer_count: scenario_count}

    File written: {output_dir}/manifest.json
    """
    _require_file(emulator_params_path)
    _require_file(base_dir / "scenario.sumocfg")
    _require_file(base_dir / "network.net.xml")
    _require_file(base_dir / "vehicles.rou.xml")

    manifest = {
        "dataset_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "base_scenario": {
            "path": str(base_dir),
            "sumocfg_hash": compute_file_hash(base_dir / "scenario.sumocfg"),
            "network_hash": compute_file_hash(base_dir / "network.net.xml"),
            "routes_hash": compute_file_hash(base_dir / "vehicles.rou.xml"),
        },
        "augmentation_ranges": {
            key: list(value) for key, value in augmentation_ranges.items()
        },
        "augmentation_config": asdict(augmentation_config) if augmentation_config else {},
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "emulator_params": {
            "path": str(emulator_params_path),
            "hash": compute_file_hash(emulator_params_path),
        },
        "environment": {
            "sumo_version": _get_sumo_version(),
            "git_commit": _get_git_commit(),
            "container_image": _get_container_image(),
        },
    }

    if peer_count_distribution:
        manifest["peer_count_distribution"] = {
            str(k): v for k, v in sorted(peer_count_distribution.items())
        }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2, sort_keys=True)


def validate_v001_spawn(scenario_dir: Path) -> bool:
    """
    Validate that V001 departs at t=0 in the generated scenario.

    Args:
        scenario_dir: Path to scenario directory

    Returns:
        True if V001 departs at t=0, False otherwise

    Raises:
        ValueError: If V001 not found in vehicles.rou.xml

    Implementation:
        1. Parse vehicles.rou.xml
        2. Find <vehicle id="V001" ...>
        3. Assert depart="0" or depart="0.0"
    """
    routes_path = scenario_dir / "vehicles.rou.xml"
    _require_file(routes_path)
    routes_tree = ET.parse(routes_path)
    _strip_namespaces(routes_tree)
    root = routes_tree.getroot()

    for vehicle in root.findall("vehicle"):
        if vehicle.get("id") == "V001":
            depart = vehicle.get("depart")
            return depart in ("0", "0.0")

    raise ValueError("V001 not found in vehicles.rou.xml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented SUMO scenarios.")
    parser.add_argument("--base_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train_count", required=True, type=int)
    parser.add_argument("--eval_count", required=True, type=int)
    parser.add_argument(
        "--emulator_params",
        type=Path,
        default=Path("ml/espnow_emulator/emulator_params_5m.json"),
    )
    parser.add_argument(
        "--peer_drop_prob",
        type=float,
        default=0.0,
        help="Probability of dropping each non-V001 vehicle (0.0-1.0)",
    )
    parser.add_argument(
        "--min_peers",
        type=int,
        default=1,
        help="Minimum number of peers to keep after dropout",
    )
    parser.add_argument(
        "--route_randomize_non_ego",
        action="store_true",
        help="Randomize route assignment for non-V001 vehicles",
    )
    parser.add_argument(
        "--route_include_v001",
        action="store_true",
        help="Include V001 in route randomization (requires --route_randomize_non_ego)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.train_count < 0 or args.eval_count < 0:
        raise ValueError("train_count and eval_count must be non-negative.")

    if args.peer_drop_prob < 0.0 or args.peer_drop_prob > 1.0:
        raise ValueError("peer_drop_prob must be between 0.0 and 1.0.")

    if args.min_peers < 0:
        raise ValueError("min_peers must be non-negative.")

    base_dir = args.base_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sumocfg_tree, network_tree, routes_tree = load_base_scenario(base_dir)

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    aug_config = AugmentationConfig(
        peer_drop_prob=args.peer_drop_prob,
        min_peers=args.min_peers,
        route_randomize_non_ego=args.route_randomize_non_ego,
        route_include_v001=args.route_include_v001,
    )

    rng = np.random.default_rng(args.seed)
    train_scenarios: List[str] = []
    eval_scenarios: List[str] = []
    peer_count_distribution: Dict[int, int] = {}

    for idx in range(args.train_count):
        scenario_id = f"train_{idx:03d}"
        train_scenarios.append(scenario_id)
        augmented_routes, peer_count = augment_routes(
            copy.deepcopy(routes_tree),
            rng,
            AUGMENTATION_RANGES,
            aug_config,
        )
        peer_count_distribution[peer_count] = peer_count_distribution.get(peer_count, 0) + 1
        write_scenario(
            train_dir,
            scenario_id,
            copy.deepcopy(sumocfg_tree),
            copy.deepcopy(network_tree),
            augmented_routes,
        )
        scenario_dir = train_dir / scenario_id
        if not validate_v001_spawn(scenario_dir):
            raise ValueError(f"V001 depart time invalid in {scenario_dir}")

    for idx in range(args.eval_count):
        scenario_id = f"eval_{idx:03d}"
        eval_scenarios.append(scenario_id)
        augmented_routes, peer_count = augment_routes(
            copy.deepcopy(routes_tree),
            rng,
            AUGMENTATION_RANGES,
            aug_config,
        )
        peer_count_distribution[peer_count] = peer_count_distribution.get(peer_count, 0) + 1
        write_scenario(
            eval_dir,
            scenario_id,
            copy.deepcopy(sumocfg_tree),
            copy.deepcopy(network_tree),
            augmented_routes,
        )
        scenario_dir = eval_dir / scenario_id
        if not validate_v001_spawn(scenario_dir):
            raise ValueError(f"V001 depart time invalid in {scenario_dir}")

    write_manifest(
        output_dir=output_dir,
        seed=args.seed,
        base_dir=base_dir,
        emulator_params_path=args.emulator_params,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        augmentation_ranges=AUGMENTATION_RANGES,
        augmentation_config=aug_config,
        peer_count_distribution=peer_count_distribution,
    )

    print(f"Generated {len(train_scenarios)} train + {len(eval_scenarios)} eval scenarios")
    print(f"Peer count distribution: {peer_count_distribution}")


if __name__ == "__main__":
    main()
