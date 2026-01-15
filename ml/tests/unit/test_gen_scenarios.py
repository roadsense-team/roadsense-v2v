"""
Unit tests for scenario generator (Phase 1).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ml.scripts import gen_scenarios


def _make_routes_tree() -> ET.ElementTree:
    xml = """
    <routes>
        <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" tau="1.0"/>
        <vehicle id="V003" type="car" depart="0"/>
        <vehicle id="V002" type="car" depart="0"/>
        <vehicle id="V001" type="car" depart="0"/>
    </routes>
    """
    return ET.ElementTree(ET.fromstring(xml))


def _make_base_dir(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "scenario.sumocfg").write_text(
        "<configuration><input/></configuration>",
        encoding="utf-8",
    )
    (base_dir / "network.net.xml").write_text("<net></net>", encoding="utf-8")
    (base_dir / "vehicles.rou.xml").write_text(
        """
        <routes>
            <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" tau="1.0"/>
            <vehicle id="V003" type="car" depart="0"/>
            <vehicle id="V002" type="car" depart="0"/>
            <vehicle id="V001" type="car" depart="0"/>
        </routes>
        """,
        encoding="utf-8",
    )


def _extract_routes_values(routes_tree: ET.ElementTree) -> tuple[dict, list[tuple[str, str]]]:
    root = routes_tree.getroot()
    vtype = root.find("vType")
    vtype_attrs = {
        name: vtype.get(name)
        for name in ("speedFactor", "sigma", "decel", "tau")
    }
    vehicles = [(veh.get("id"), veh.get("depart")) for veh in root.findall("vehicle")]
    return vtype_attrs, vehicles


def test_load_base_scenario_success(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    _make_base_dir(base_dir)

    sumocfg_tree, network_tree, routes_tree = gen_scenarios.load_base_scenario(base_dir)

    assert sumocfg_tree.getroot().tag == "configuration"
    assert network_tree.getroot().tag == "net"
    assert routes_tree.getroot().tag == "routes"


def test_load_base_scenario_missing_file(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "scenario.sumocfg").write_text("<configuration/>", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        gen_scenarios.load_base_scenario(base_dir)


def test_augment_routes_v001_depart_unchanged() -> None:
    routes_tree = _make_routes_tree()
    rng = np.random.default_rng(123)

    augmented = gen_scenarios.augment_routes(
        routes_tree,
        rng,
        gen_scenarios.AUGMENTATION_RANGES,
    )

    _, vehicles = _extract_routes_values(augmented)
    v001_depart = {vid: depart for vid, depart in vehicles}["V001"]
    assert v001_depart in ("0", "0.0")


def test_augment_routes_parameters_in_range() -> None:
    routes_tree = _make_routes_tree()
    rng = np.random.default_rng(42)

    augmented = gen_scenarios.augment_routes(
        routes_tree,
        rng,
        gen_scenarios.AUGMENTATION_RANGES,
    )

    vtype_attrs, vehicles = _extract_routes_values(augmented)
    for key, (low, high) in gen_scenarios.AUGMENTATION_RANGES.items():
        if key == "spawn_jitter_s":
            continue
        value = float(vtype_attrs[key])
        assert low <= value <= high

    depart_map = {vid: float(depart) for vid, depart in vehicles}
    assert 0.0 <= depart_map["V002"] <= 2.0
    assert 0.0 <= depart_map["V003"] <= 2.0


def test_augment_routes_sorts_by_depart() -> None:
    routes_tree = _make_routes_tree()
    rng = np.random.default_rng(7)

    augmented = gen_scenarios.augment_routes(
        routes_tree,
        rng,
        gen_scenarios.AUGMENTATION_RANGES,
    )

    _, vehicles = _extract_routes_values(augmented)
    departures = [float(depart) for _, depart in vehicles]
    assert departures == sorted(departures)


def test_augment_routes_deterministic() -> None:
    routes_tree = _make_routes_tree()
    rng_a = np.random.default_rng(999)
    rng_b = np.random.default_rng(999)

    augmented_a = gen_scenarios.augment_routes(
        copy.deepcopy(routes_tree),
        rng_a,
        gen_scenarios.AUGMENTATION_RANGES,
    )
    augmented_b = gen_scenarios.augment_routes(
        copy.deepcopy(routes_tree),
        rng_b,
        gen_scenarios.AUGMENTATION_RANGES,
    )

    assert _extract_routes_values(augmented_a) == _extract_routes_values(augmented_b)


def test_write_manifest_schema_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "base"
    _make_base_dir(base_dir)
    emulator_params = tmp_path / "emulator_params.json"
    emulator_params.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(gen_scenarios, "_get_sumo_version", lambda: "1.25.0")
    monkeypatch.setattr(gen_scenarios, "_get_git_commit", lambda: "abc1234")
    monkeypatch.setattr(gen_scenarios, "_get_container_image", lambda: "roadsense-ml:latest")

    output_dir = tmp_path / "dataset_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_scenarios.write_manifest(
        output_dir=output_dir,
        seed=42,
        base_dir=base_dir,
        emulator_params_path=emulator_params,
        train_scenarios=["train_000"],
        eval_scenarios=["eval_000"],
        augmentation_ranges=gen_scenarios.AUGMENTATION_RANGES,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "dataset_v1"
    assert manifest["seed"] == 42
    assert manifest["created_at"].endswith("+00:00")
    assert "base_scenario" in manifest
    assert "augmentation_ranges" in manifest
    assert "train_scenarios" in manifest
    assert "eval_scenarios" in manifest
    assert "emulator_params" in manifest
    assert "environment" in manifest
    assert manifest["environment"]["sumo_version"] == "1.25.0"
    assert manifest["environment"]["git_commit"] == "abc1234"
    assert manifest["environment"]["container_image"] == "roadsense-ml:latest"


def test_validate_v001_spawn_pass(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "vehicles.rou.xml").write_text(
        """
        <routes>
            <vehicle id="V001" depart="0"/>
        </routes>
        """,
        encoding="utf-8",
    )

    assert gen_scenarios.validate_v001_spawn(scenario_dir) is True


def test_validate_v001_spawn_fail(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "vehicles.rou.xml").write_text(
        """
        <routes>
            <vehicle id="V001" depart="1"/>
        </routes>
        """,
        encoding="utf-8",
    )

    assert gen_scenarios.validate_v001_spawn(scenario_dir) is False


def test_compute_file_hash_deterministic(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    first = gen_scenarios.compute_file_hash(file_path)
    second = gen_scenarios.compute_file_hash(file_path)

    assert first == second
