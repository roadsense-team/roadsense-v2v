"""
Helpers for eval dataset metadata and capability audit loading.
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import xml.etree.ElementTree as ET


CAPABILITY_AUDIT_FILENAME = "eval_capability_audit.json"


def load_eval_scenario_ids(dataset_dir: str) -> List[str]:
    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in dataset_dir: {dataset_dir}")

    with manifest_path.open("r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)

    eval_scenarios = manifest.get("eval_scenarios")
    if not isinstance(eval_scenarios, list):
        raise KeyError("manifest.json missing required list field: eval_scenarios")

    return [str(scenario_id) for scenario_id in eval_scenarios]


def resolve_scenario_dir(dataset_dir: str, scenario_id: str) -> Optional[Path]:
    if not scenario_id or scenario_id == "unknown":
        return None

    dataset_root = Path(dataset_dir)
    candidates: List[Path] = []

    if scenario_id.startswith("eval_"):
        candidates.append(dataset_root / "eval" / scenario_id)
    elif scenario_id.startswith("train_"):
        candidates.append(dataset_root / "train" / scenario_id)

    candidates.extend(
        [
            dataset_root / "eval" / scenario_id,
            dataset_root / "train" / scenario_id,
        ]
    )

    for scenario_dir in candidates:
        if scenario_dir.exists():
            return scenario_dir

    return None


def resolve_sumo_cfg_path(dataset_dir: str, scenario_id: str) -> Optional[Path]:
    scenario_dir = resolve_scenario_dir(dataset_dir, scenario_id)
    if scenario_dir is None:
        return None

    sumo_cfg_path = scenario_dir / "scenario.sumocfg"
    if sumo_cfg_path.exists():
        return sumo_cfg_path

    return None


def resolve_routes_path(dataset_dir: str, scenario_id: str) -> Optional[Path]:
    scenario_dir = resolve_scenario_dir(dataset_dir, scenario_id)
    if scenario_dir is None:
        return None

    routes_path = scenario_dir / "vehicles.rou.xml"
    if routes_path.exists():
        return routes_path

    return None


def infer_peer_count(
    dataset_dir: Optional[str],
    scenario_id: str,
    cache: Dict[str, Optional[int]],
) -> Optional[int]:
    if scenario_id in cache:
        return cache[scenario_id]

    if dataset_dir is None:
        cache[scenario_id] = None
        return None

    routes_path = resolve_routes_path(dataset_dir, scenario_id)
    if routes_path is None:
        cache[scenario_id] = None
        return None

    try:
        routes_tree = ET.parse(routes_path)
    except (ET.ParseError, OSError):
        cache[scenario_id] = None
        return None

    root = routes_tree.getroot()
    peer_count = 0
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1] if isinstance(elem.tag, str) else elem.tag
        if tag == "vehicle" and elem.get("id") != "V001":
            peer_count += 1

    cache[scenario_id] = peer_count
    return peer_count


def load_eval_peer_counts(
    dataset_dir: str,
    eval_scenario_ids: List[str],
) -> Dict[str, int]:
    cache: Dict[str, Optional[int]] = {}
    peer_counts: Dict[str, int] = {}

    for scenario_id in eval_scenario_ids:
        peer_count = infer_peer_count(dataset_dir, scenario_id, cache)
        if peer_count is None:
            raise ValueError(
                f"Could not infer peer_count for eval scenario: {scenario_id}"
            )
        peer_counts[scenario_id] = int(peer_count)

    return peer_counts


def get_eval_capability_audit_path(dataset_dir: str) -> Path:
    return Path(dataset_dir) / CAPABILITY_AUDIT_FILENAME


def load_eval_capabilities(
    dataset_dir: str,
    eval_scenario_ids: Iterable[str],
    allow_missing_file: bool = False,
) -> Optional[Dict[str, Set[int]]]:
    audit_path = get_eval_capability_audit_path(dataset_dir)
    if not audit_path.exists():
        if allow_missing_file:
            return None
        raise FileNotFoundError(
            f"{CAPABILITY_AUDIT_FILENAME} not found in dataset_dir: {dataset_dir}"
        )

    with audit_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise KeyError(
            f"{CAPABILITY_AUDIT_FILENAME} missing required list field: scenarios"
        )

    capabilities_by_scenario: Dict[str, Set[int]] = {}
    for entry in scenarios:
        if not isinstance(entry, dict):
            raise TypeError("scenario capability entries must be objects")

        scenario_id = entry.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("scenario capability entry missing scenario_id")

        supported_ranks = entry.get("supported_ranks_any_step")
        if not isinstance(supported_ranks, list):
            raise ValueError(
                f"scenario capability entry missing supported_ranks_any_step: {scenario_id}"
            )

        parsed_ranks: Set[int] = set()
        for rank in supported_ranks:
            parsed_rank = int(rank)
            if parsed_rank <= 0:
                raise ValueError(
                    f"supported rank must be >= 1 for scenario {scenario_id}"
                )
            parsed_ranks.add(parsed_rank)

        capabilities_by_scenario[scenario_id] = parsed_ranks

    required_scenario_ids = [str(scenario_id) for scenario_id in eval_scenario_ids]
    missing = [
        scenario_id
        for scenario_id in required_scenario_ids
        if scenario_id not in capabilities_by_scenario
    ]
    if missing:
        missing_str = ",".join(missing)
        raise ValueError(f"missing capability audit entries: {missing_str}")

    return {
        scenario_id: set(capabilities_by_scenario[scenario_id])
        for scenario_id in required_scenario_ids
    }
