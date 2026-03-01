"""
Unit tests for deterministic eval peer-count enforcement.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ml.scripts import fix_eval_peer_counts


def _make_routes_root(peer_ids: list[str]) -> ET.Element:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        attrib={
            "id": "car",
            "accel": "2.6",
            "decel": "5.0",
            "sigma": "0.3",
            "tau": "1.0",
        },
    )
    ET.SubElement(root, "route", attrib={"id": "convoy_route", "edges": "-635191227"})

    depart_positions = {
        "V006": "82.813",
        "V005": "60.813",
        "V004": "38.813",
        "V003": "16.813",
        "V002": "10.921",
    }

    for peer_id in peer_ids:
        ET.SubElement(
            root,
            "vehicle",
            attrib={
                "id": peer_id,
                "type": "car",
                "depart": "0",
                "route": "convoy_route",
                "departPos": depart_positions.get(peer_id, "1.000"),
            },
        )

    ET.SubElement(
        root,
        "vehicle",
        attrib={
            "id": "V001",
            "type": "car",
            "depart": "0",
            "route": "convoy_route",
            "departPos": "0.000",
        },
    )
    return root


def _write_scenario(scenario_dir: Path, peer_ids: list[str]) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "scenario.sumocfg").write_text(
        "<configuration><input/></configuration>",
        encoding="utf-8",
    )
    (scenario_dir / "network.net.xml").write_text("<net/>", encoding="utf-8")
    routes_root = _make_routes_root(peer_ids)
    ET.ElementTree(routes_root).write(scenario_dir / "vehicles.rou.xml", encoding="utf-8")


def test_enforce_peer_count_drops_synthetic_farthest_first() -> None:
    root = _make_routes_root(["V006", "V005", "V004", "V003", "V002"])
    fix_eval_peer_counts.enforce_peer_count(root, target_peer_count=3)

    remaining_peers = [v.get("id") for v in root.findall("vehicle") if v.get("id") != "V001"]
    assert remaining_peers == ["V004", "V003", "V002"]

    root = _make_routes_root(["V006", "V005", "V004", "V003", "V002"])
    fix_eval_peer_counts.enforce_peer_count(root, target_peer_count=1)
    remaining_peers = [v.get("id") for v in root.findall("vehicle") if v.get("id") != "V001"]
    assert remaining_peers == ["V002"]


def test_build_assignments_errors_when_targets_not_feasible() -> None:
    scenarios = [
        fix_eval_peer_counts.EvalScenario("eval_000", Path("/tmp/eval_000"), 3),
        fix_eval_peer_counts.EvalScenario("eval_001", Path("/tmp/eval_001"), 2),
    ]
    with pytest.raises(ValueError, match="Unable to satisfy target n=5"):
        fix_eval_peer_counts._build_assignments(scenarios, [5, 1])


def test_apply_eval_peer_count_fix_rewrites_eval_and_manifest(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset_v3"
    train_dir = dataset_dir / "train"
    eval_dir = dataset_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    _write_scenario(train_dir / "train_000", ["V004", "V003", "V002"])
    _write_scenario(eval_dir / "eval_000", ["V006", "V005", "V004", "V003", "V002"])
    _write_scenario(eval_dir / "eval_001", ["V003", "V002"])

    manifest = {
        "dataset_id": "dataset_v3",
        "train_scenarios": ["train_000"],
        "eval_scenarios": ["eval_000", "eval_001"],
        "peer_count_distribution": {"2": 1, "3": 1, "5": 1},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assignments = fix_eval_peer_counts.apply_eval_peer_count_fix(
        dataset_dir=dataset_dir,
        target_counts=[1, 2],
        dry_run=False,
    )

    assert [a.output_scenario_id for a in assignments] == ["eval_n1_000", "eval_n2_000"]
    assert sorted(path.name for path in (dataset_dir / "eval").iterdir()) == [
        "eval_n1_000",
        "eval_n2_000",
    ]

    n1_routes = ET.parse(dataset_dir / "eval" / "eval_n1_000" / "vehicles.rou.xml").getroot()
    n1_peers = [v.get("id") for v in n1_routes.findall("vehicle") if v.get("id") != "V001"]
    assert n1_peers == ["V002"]

    n2_routes = ET.parse(dataset_dir / "eval" / "eval_n2_000" / "vehicles.rou.xml").getroot()
    n2_peers = [v.get("id") for v in n2_routes.findall("vehicle") if v.get("id") != "V001"]
    assert len(n2_peers) == 2

    for scenario_id in ("eval_n1_000", "eval_n2_000"):
        scenario_dir = dataset_dir / "eval" / scenario_id
        assert (scenario_dir / "scenario.sumocfg").exists()
        assert (scenario_dir / "network.net.xml").exists()

    rewritten_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert rewritten_manifest["eval_scenarios"] == ["eval_n1_000", "eval_n2_000"]
    assert rewritten_manifest["eval_peer_counts_enforced"] is True
    assert rewritten_manifest["initial_speed_regime"] == "low_speed_start_from_standing"
    assert rewritten_manifest["peer_count_distribution"] == {"1": 1, "2": 1, "3": 1}
