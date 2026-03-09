#!/usr/bin/env python3
"""
Curate an eval subset using runtime capability audit data.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ml.eval_dataset import get_eval_capability_audit_path, load_eval_scenario_ids
from ml.eval_matrix import bucket_label, parse_peer_count_list


BucketKey = Tuple[int, int]


@dataclass(frozen=True)
class CapabilityScenario:
    scenario_id: str
    peer_count: int
    supported_buckets: Set[BucketKey]


def _required_buckets(required_peer_counts: List[int]) -> List[BucketKey]:
    return [
        (peer_count, rank_ahead)
        for peer_count in required_peer_counts
        for rank_ahead in range(1, peer_count + 1)
    ]


def _load_capability_payload(dataset_dir: Path) -> dict:
    audit_path = get_eval_capability_audit_path(str(dataset_dir))
    if not audit_path.exists():
        raise FileNotFoundError(f"Capability audit not found: {audit_path}")
    with audit_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload.get("scenarios"), list):
        raise KeyError("eval capability audit missing required list field: scenarios")
    return payload


def _load_capability_scenarios(dataset_dir: Path) -> List[CapabilityScenario]:
    eval_scenario_ids = load_eval_scenario_ids(str(dataset_dir))
    payload = _load_capability_payload(dataset_dir)
    scenario_map = {
        str(entry["scenario_id"]): entry
        for entry in payload["scenarios"]
    }

    scenarios: List[CapabilityScenario] = []
    missing = [
        scenario_id for scenario_id in eval_scenario_ids
        if scenario_id not in scenario_map
    ]
    if missing:
        raise ValueError(
            "capability audit missing eval scenarios: " + ",".join(missing)
        )

    for scenario_id in eval_scenario_ids:
        entry = scenario_map[scenario_id]
        peer_count = int(entry["peer_count"])
        if peer_count <= 0:
            raise ValueError(f"Invalid peer_count for scenario {scenario_id}: {peer_count}")

        supported_ranks = entry.get("supported_ranks_any_step", [])
        if not isinstance(supported_ranks, list):
            raise ValueError(
                f"supported_ranks_any_step must be a list for scenario {scenario_id}"
            )

        supported_buckets: Set[BucketKey] = set()
        for rank_ahead in supported_ranks:
            parsed_rank = int(rank_ahead)
            if parsed_rank <= 0 or parsed_rank > peer_count:
                raise ValueError(
                    f"Invalid supported rank {parsed_rank} for scenario {scenario_id}"
                )
            supported_buckets.add((peer_count, parsed_rank))

        scenarios.append(
            CapabilityScenario(
                scenario_id=scenario_id,
                peer_count=peer_count,
                supported_buckets=supported_buckets,
            )
        )

    return scenarios


def select_eval_scenarios_by_capability(
    dataset_dir: Path,
    required_peer_counts: List[int],
    min_scenarios_per_bucket: int = 1,
) -> List[str]:
    if min_scenarios_per_bucket <= 0:
        raise ValueError("min_scenarios_per_bucket must be > 0")

    scenarios = _load_capability_scenarios(dataset_dir)
    required_buckets = _required_buckets(required_peer_counts)

    available_counts: Dict[BucketKey, int] = defaultdict(int)
    for scenario in scenarios:
        for bucket in scenario.supported_buckets:
            if bucket in required_buckets:
                available_counts[bucket] += 1

    for bucket in required_buckets:
        available = available_counts.get(bucket, 0)
        if available < min_scenarios_per_bucket:
            raise ValueError(
                f"insufficient capability coverage for bucket {bucket_label(*bucket)}: "
                f"need {min_scenarios_per_bucket}, have {available}"
            )

    selected_ids: Set[str] = set()
    coverage_counts: Dict[BucketKey, int] = defaultdict(int)

    while any(
        coverage_counts.get(bucket, 0) < min_scenarios_per_bucket
        for bucket in required_buckets
    ):
        best_scenario = None
        best_gain = -1

        for scenario in scenarios:
            if scenario.scenario_id in selected_ids:
                continue
            gain = sum(
                1
                for bucket in scenario.supported_buckets
                if bucket in required_buckets
                and coverage_counts.get(bucket, 0) < min_scenarios_per_bucket
            )
            if gain > best_gain:
                best_scenario = scenario
                best_gain = gain

        if best_scenario is None or best_gain <= 0:
            raise RuntimeError("Unable to complete capability-based eval curation.")

        selected_ids.add(best_scenario.scenario_id)
        for bucket in best_scenario.supported_buckets:
            if (
                bucket in required_buckets
                and coverage_counts.get(bucket, 0) < min_scenarios_per_bucket
            ):
                coverage_counts[bucket] += 1

    manifest_order = load_eval_scenario_ids(str(dataset_dir))
    return [
        scenario_id for scenario_id in manifest_order
        if scenario_id in selected_ids
    ]


def _copy_dataset_structure(
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    selected_eval_ids: List[str],
) -> None:
    if output_dataset_dir.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dataset_dir}")

    output_dataset_dir.mkdir(parents=True)
    for child in source_dataset_dir.iterdir():
        if child.name == "manifest.json":
            continue
        if child.name == get_eval_capability_audit_path(str(source_dataset_dir)).name:
            continue
        if child.name == "eval":
            target_eval_dir = output_dataset_dir / "eval"
            target_eval_dir.mkdir(parents=True, exist_ok=True)
            for scenario_id in selected_eval_ids:
                shutil.copytree(
                    child / scenario_id,
                    target_eval_dir / scenario_id,
                )
            continue
        if child.is_dir():
            shutil.copytree(child, output_dataset_dir / child.name)
        else:
            shutil.copy2(child, output_dataset_dir / child.name)


def _write_curated_manifest(
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    selected_eval_ids: List[str],
    required_peer_counts: List[int],
    min_scenarios_per_bucket: int,
) -> None:
    with (source_dataset_dir / "manifest.json").open("r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)

    manifest["dataset_id"] = output_dataset_dir.name
    manifest["source_dataset_dir"] = str(source_dataset_dir)
    manifest["eval_scenarios"] = selected_eval_ids
    manifest["eval_capability_curated"] = True
    manifest["eval_capability_required_peer_counts"] = required_peer_counts
    manifest["eval_capability_min_scenarios_per_bucket"] = min_scenarios_per_bucket

    with (output_dataset_dir / "manifest.json").open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2, sort_keys=True)


def _write_filtered_audit(
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    selected_eval_ids: List[str],
) -> None:
    payload = _load_capability_payload(source_dataset_dir)
    selected_set = set(selected_eval_ids)
    payload["source_dataset_dir"] = str(source_dataset_dir)
    payload["scenarios"] = [
        scenario
        for scenario in payload["scenarios"]
        if str(scenario["scenario_id"]) in selected_set
    ]
    payload["scenarios"].sort(
        key=lambda scenario: selected_eval_ids.index(str(scenario["scenario_id"]))
    )

    with (output_dataset_dir / "eval_capability_audit.json").open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)


def curate_dataset_by_capability(
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    required_peer_counts: List[int],
    min_scenarios_per_bucket: int = 1,
) -> List[str]:
    selected_eval_ids = select_eval_scenarios_by_capability(
        dataset_dir=source_dataset_dir,
        required_peer_counts=required_peer_counts,
        min_scenarios_per_bucket=min_scenarios_per_bucket,
    )
    _copy_dataset_structure(
        source_dataset_dir=source_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        selected_eval_ids=selected_eval_ids,
    )
    _write_curated_manifest(
        source_dataset_dir=source_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        selected_eval_ids=selected_eval_ids,
        required_peer_counts=required_peer_counts,
        min_scenarios_per_bucket=min_scenarios_per_bucket,
    )
    _write_filtered_audit(
        source_dataset_dir=source_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        selected_eval_ids=selected_eval_ids,
    )
    return selected_eval_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate a capability-valid eval subset from an audited dataset."
    )
    parser.add_argument(
        "--source_dataset_dir",
        type=Path,
        required=True,
        help="Audited source dataset directory",
    )
    parser.add_argument(
        "--output_dataset_dir",
        type=Path,
        required=True,
        help="Output dataset directory for the curated subset",
    )
    parser.add_argument(
        "--required_peer_counts",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated peer counts that must be covered",
    )
    parser.add_argument(
        "--min_scenarios_per_bucket",
        type=int,
        default=1,
        help="Minimum number of supporting scenarios required per bucket",
    )
    args = parser.parse_args()

    if args.min_scenarios_per_bucket <= 0:
        parser.error("--min_scenarios_per_bucket must be > 0")

    return args


def main() -> None:
    args = parse_args()
    required_peer_counts = parse_peer_count_list(args.required_peer_counts)
    selected_eval_ids = curate_dataset_by_capability(
        source_dataset_dir=args.source_dataset_dir,
        output_dataset_dir=args.output_dataset_dir,
        required_peer_counts=required_peer_counts,
        min_scenarios_per_bucket=args.min_scenarios_per_bucket,
    )
    print(f"Selected {len(selected_eval_ids)} eval scenarios:")
    for scenario_id in selected_eval_ids:
        print(f"  {scenario_id}")
    print(f"Curated dataset written to: {args.output_dataset_dir}")


if __name__ == "__main__":
    main()
