#!/usr/bin/env python3
"""
Enforce deterministic peer-count coverage for eval scenarios.

This script rewrites eval scenarios in-place so their peer counts match the
requested targets (for example: 1,1,2,2,3,3,4,4,5,5). It only removes peers,
never V001, and updates manifest metadata accordingly.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET


# Synthetic peers are ordered farthest-to-closest from ego in base_real.
DROP_PRIORITY = ("V006", "V005", "V004")


@dataclass(frozen=True)
class EvalScenario:
    scenario_id: str
    scenario_dir: Path
    peer_count: int


@dataclass(frozen=True)
class ScenarioAssignment:
    source_scenario: EvalScenario
    target_peer_count: int
    output_scenario_id: str


def _require_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def _parse_target_counts(raw_counts: str) -> List[int]:
    values: List[int] = []
    for raw in raw_counts.split(","):
        token = raw.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Peer counts must be positive integers, got: {value}")
        values.append(value)
    if not values:
        raise ValueError("No target peer counts provided.")
    return values


def _load_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    _require_file(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)
    if "eval_scenarios" not in manifest or not isinstance(manifest["eval_scenarios"], list):
        raise KeyError("manifest.json missing required list field: eval_scenarios")
    if "train_scenarios" not in manifest or not isinstance(manifest["train_scenarios"], list):
        raise KeyError("manifest.json missing required list field: train_scenarios")
    return manifest


def _load_routes_tree(routes_path: Path) -> ET.ElementTree:
    _require_file(routes_path)
    return ET.parse(routes_path)


def _find_v001(root: ET.Element) -> ET.Element:
    for vehicle in root.findall("vehicle"):
        if vehicle.get("id") == "V001":
            return vehicle
    raise ValueError("vehicles.rou.xml missing V001 vehicle.")


def _count_peers(root: ET.Element) -> int:
    return sum(1 for vehicle in root.findall("vehicle") if vehicle.get("id") != "V001")


def _count_peers_in_scenario(scenario_dir: Path) -> int:
    routes_path = scenario_dir / "vehicles.rou.xml"
    root = _load_routes_tree(routes_path).getroot()
    _find_v001(root)
    return _count_peers(root)


def _vehicle_depart_pos(vehicle: ET.Element) -> float:
    raw = vehicle.get("departPos")
    if raw is None:
        return float("-inf")
    try:
        return float(raw)
    except ValueError:
        return float("-inf")


def _select_peers_to_drop(peers: List[ET.Element], drop_count: int) -> List[ET.Element]:
    by_id = {peer.get("id"): peer for peer in peers}
    selected: List[ET.Element] = []

    for vehicle_id in DROP_PRIORITY:
        if drop_count == 0:
            break
        peer = by_id.pop(vehicle_id, None)
        if peer is not None:
            selected.append(peer)
            drop_count -= 1

    if drop_count == 0:
        return selected

    remaining = [peer for peer in peers if peer not in selected]
    # Fallback: drop furthest-ahead peers first by departPos, then by ID.
    remaining.sort(
        key=lambda peer: (_vehicle_depart_pos(peer), peer.get("id") or ""),
        reverse=True,
    )
    selected.extend(remaining[:drop_count])
    return selected


def enforce_peer_count(root: ET.Element, target_peer_count: int) -> None:
    _find_v001(root)
    peers = [vehicle for vehicle in root.findall("vehicle") if vehicle.get("id") != "V001"]
    current_count = len(peers)

    if target_peer_count > current_count:
        raise ValueError(
            f"Cannot enforce n={target_peer_count}; scenario only has {current_count} peers."
        )

    drop_count = current_count - target_peer_count
    if drop_count <= 0:
        return

    to_drop = _select_peers_to_drop(peers, drop_count)
    for vehicle in to_drop:
        root.remove(vehicle)


def _validate_v001_depart_zero(root: ET.Element) -> None:
    v001 = _find_v001(root)
    depart = v001.get("depart")
    if depart not in {"0", "0.0"}:
        raise ValueError(f"V001 must depart at t=0, found depart={depart!r}")


def _load_eval_scenarios(dataset_dir: Path, eval_ids: List[str]) -> List[EvalScenario]:
    scenarios: List[EvalScenario] = []
    for scenario_id in eval_ids:
        scenario_dir = dataset_dir / "eval" / scenario_id
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Eval scenario directory missing: {scenario_dir}")
        peer_count = _count_peers_in_scenario(scenario_dir)
        scenarios.append(
            EvalScenario(
                scenario_id=scenario_id,
                scenario_dir=scenario_dir,
                peer_count=peer_count,
            )
        )
    return scenarios


def _build_assignments(
    scenarios: List[EvalScenario],
    target_counts: List[int],
) -> List[ScenarioAssignment]:
    if len(scenarios) != len(target_counts):
        raise ValueError(
            "Number of eval scenarios must match number of target counts: "
            f"{len(scenarios)} != {len(target_counts)}"
        )

    unassigned = scenarios.copy()
    provisional: List[tuple[EvalScenario, int]] = []

    for target in sorted(target_counts, reverse=True):
        candidates = [scenario for scenario in unassigned if scenario.peer_count >= target]
        if not candidates:
            available = sorted({scenario.peer_count for scenario in unassigned})
            raise ValueError(
                f"Unable to satisfy target n={target}. "
                f"Remaining scenario peer counts: {available}"
            )
        chosen = min(candidates, key=lambda scenario: (scenario.peer_count, scenario.scenario_id))
        provisional.append((chosen, target))
        unassigned.remove(chosen)

    counters: Dict[int, int] = {}
    assignments: List[ScenarioAssignment] = []
    for scenario, target in sorted(provisional, key=lambda item: (item[1], item[0].scenario_id)):
        index = counters.get(target, 0)
        counters[target] = index + 1
        assignments.append(
            ScenarioAssignment(
                source_scenario=scenario,
                target_peer_count=target,
                output_scenario_id=f"eval_n{target}_{index:03d}",
            )
        )
    return assignments


def _copy_required_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("scenario.sumocfg", "network.net.xml"):
        src = source_dir / filename
        _require_file(src)
        shutil.copy2(src, target_dir / filename)


def _replace_eval_directory(eval_dir: Path, rewritten_dir: Path) -> None:
    backup_dir = eval_dir.with_name(eval_dir.name + ".__pre_fix_backup__")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    eval_dir.rename(backup_dir)
    try:
        rewritten_dir.rename(eval_dir)
    except Exception:
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        backup_dir.rename(eval_dir)
        raise
    shutil.rmtree(backup_dir)


def _compute_peer_count_distribution(
    dataset_dir: Path,
    train_scenarios: List[str],
    eval_scenarios: List[str],
) -> Dict[str, int]:
    distribution: Dict[int, int] = {}

    def _accumulate(split: str, scenario_id: str) -> None:
        scenario_dir = dataset_dir / split / scenario_id
        peer_count = _count_peers_in_scenario(scenario_dir)
        distribution[peer_count] = distribution.get(peer_count, 0) + 1

    for scenario_id in train_scenarios:
        _accumulate("train", scenario_id)
    for scenario_id in eval_scenarios:
        _accumulate("eval", scenario_id)

    return {str(key): distribution[key] for key in sorted(distribution)}


def _write_manifest(
    dataset_dir: Path,
    manifest: dict,
    eval_scenarios: List[str],
) -> None:
    manifest["eval_scenarios"] = eval_scenarios
    manifest["eval_peer_counts_enforced"] = True
    manifest["peer_count_distribution"] = _compute_peer_count_distribution(
        dataset_dir=dataset_dir,
        train_scenarios=[str(value) for value in manifest["train_scenarios"]],
        eval_scenarios=eval_scenarios,
    )
    manifest["initial_speed_regime"] = "low_speed_start_from_standing"

    manifest_path = dataset_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2, sort_keys=True)


def apply_eval_peer_count_fix(
    dataset_dir: Path,
    target_counts: List[int],
    dry_run: bool = False,
) -> List[ScenarioAssignment]:
    manifest = _load_manifest(dataset_dir)
    eval_ids = [str(value) for value in manifest["eval_scenarios"]]
    scenarios = _load_eval_scenarios(dataset_dir, eval_ids)
    assignments = _build_assignments(scenarios, target_counts)

    if dry_run:
        return assignments

    eval_dir = dataset_dir / "eval"
    rewritten_dir = dataset_dir / "eval.__rewrite_tmp__"
    if rewritten_dir.exists():
        shutil.rmtree(rewritten_dir)
    rewritten_dir.mkdir(parents=True, exist_ok=True)

    try:
        for assignment in assignments:
            source_dir = assignment.source_scenario.scenario_dir
            target_dir = rewritten_dir / assignment.output_scenario_id
            _copy_required_files(source_dir, target_dir)

            routes_tree = _load_routes_tree(source_dir / "vehicles.rou.xml")
            root = routes_tree.getroot()
            enforce_peer_count(root, assignment.target_peer_count)
            _validate_v001_depart_zero(root)

            actual_count = _count_peers(root)
            if actual_count != assignment.target_peer_count:
                raise RuntimeError(
                    f"Internal error while writing {assignment.output_scenario_id}: "
                    f"expected {assignment.target_peer_count} peers, got {actual_count}"
                )

            routes_tree.write(target_dir / "vehicles.rou.xml", encoding="utf-8")

        _replace_eval_directory(eval_dir, rewritten_dir)
        _write_manifest(
            dataset_dir=dataset_dir,
            manifest=manifest,
            eval_scenarios=[item.output_scenario_id for item in assignments],
        )
    finally:
        if rewritten_dir.exists():
            shutil.rmtree(rewritten_dir)

    return assignments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite eval scenarios to deterministic peer counts and update manifest.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Dataset directory that contains manifest.json and train/eval subfolders.",
    )
    parser.add_argument(
        "--target_counts",
        type=str,
        default="1,1,2,2,3,3,4,4,5,5",
        help="Comma-separated peer counts for eval scenarios (default: 1,1,2,2,3,3,4,4,5,5).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate feasibility and print mapping without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    target_counts = _parse_target_counts(args.target_counts)
    assignments = apply_eval_peer_count_fix(
        dataset_dir=args.dataset_dir,
        target_counts=target_counts,
        dry_run=args.dry_run,
    )

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] eval peer-count assignments:")
    for assignment in assignments:
        source = assignment.source_scenario
        print(
            f"  {source.scenario_id} (n={source.peer_count}) "
            f"-> {assignment.output_scenario_id} (target n={assignment.target_peer_count})"
        )

    if not args.dry_run:
        print("Manifest updated: eval_scenarios, eval_peer_counts_enforced, peer_count_distribution.")


if __name__ == "__main__":
    main()
