"""
Unit tests for capability-based eval dataset curation.
"""
import json
from pathlib import Path

import pytest

from ml.scripts import curate_eval_by_capability


def _write_dataset(
    tmp_path: Path,
    train_ids: list[str],
    eval_entries: list[tuple[str, int, list[int]]],
) -> Path:
    dataset_dir = tmp_path / "source_dataset"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "eval").mkdir(parents=True)

    for scenario_id in train_ids:
        scenario_dir = dataset_dir / "train" / scenario_id
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "scenario.sumocfg").write_text("<configuration/>", encoding="utf-8")
        (scenario_dir / "network.net.xml").write_text("<net/>", encoding="utf-8")
        (scenario_dir / "vehicles.rou.xml").write_text(
            "<routes><vehicle id='V001' route='r0' depart='0'/></routes>",
            encoding="utf-8",
        )

    audit_scenarios = []
    eval_ids = []
    for scenario_id, peer_count, supported_ranks in eval_entries:
        eval_ids.append(scenario_id)
        scenario_dir = dataset_dir / "eval" / scenario_id
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "scenario.sumocfg").write_text("<configuration/>", encoding="utf-8")
        (scenario_dir / "network.net.xml").write_text("<net/>", encoding="utf-8")

        vehicles = ["<routes>", "  <vehicle id='V001' route='r0' depart='0'/>"]
        for idx in range(peer_count):
            vehicles.append(
                f"  <vehicle id='V{idx + 2:03d}' route='r0' depart='{idx + 1}'/>"
            )
        vehicles.append("</routes>")
        (scenario_dir / "vehicles.rou.xml").write_text(
            "\n".join(vehicles),
            encoding="utf-8",
        )

        audit_scenarios.append(
            {
                "scenario_id": scenario_id,
                "peer_count": peer_count,
                "supported_ranks_any_step": supported_ranks,
                "supported_steps_by_rank": {
                    str(rank): [40] for rank in supported_ranks
                },
                "failed_steps_by_rank": {},
                "failure_reasons_by_rank": {},
            }
        )

    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_dir.name,
                "train_scenarios": train_ids,
                "eval_scenarios": eval_ids,
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "eval_capability_audit.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "scenarios": audit_scenarios,
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def test_select_eval_scenarios_by_capability_covers_all_required_buckets(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(
        tmp_path,
        train_ids=["train_000"],
        eval_entries=[
            ("eval_a", 1, [1]),
            ("eval_b", 2, [1, 2]),
            ("eval_c", 3, [1, 2]),
            ("eval_d", 3, [3]),
            ("eval_extra", 3, [1]),
        ],
    )

    selected = curate_eval_by_capability.select_eval_scenarios_by_capability(
        dataset_dir=dataset_dir,
        required_peer_counts=[1, 2, 3],
        min_scenarios_per_bucket=1,
    )

    assert selected == ["eval_a", "eval_b", "eval_c", "eval_d"]


def test_select_eval_scenarios_by_capability_fails_when_bucket_missing(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(
        tmp_path,
        train_ids=["train_000"],
        eval_entries=[
            ("eval_a", 1, []),
            ("eval_b", 2, [1]),
        ],
    )

    with pytest.raises(ValueError, match="insufficient capability coverage for bucket n1_rank1"):
        curate_eval_by_capability.select_eval_scenarios_by_capability(
            dataset_dir=dataset_dir,
            required_peer_counts=[1, 2],
            min_scenarios_per_bucket=1,
        )


def test_curate_dataset_by_capability_copies_selected_eval_and_filters_audit(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(
        tmp_path,
        train_ids=["train_000"],
        eval_entries=[
            ("eval_a", 1, [1]),
            ("eval_b", 2, [1, 2]),
            ("eval_c", 3, [1, 2]),
            ("eval_d", 3, [3]),
            ("eval_extra", 3, [1]),
        ],
    )
    output_dir = tmp_path / "curated_dataset"

    selected = curate_eval_by_capability.curate_dataset_by_capability(
        source_dataset_dir=dataset_dir,
        output_dataset_dir=output_dir,
        required_peer_counts=[1, 2, 3],
        min_scenarios_per_bucket=1,
    )

    assert selected == ["eval_a", "eval_b", "eval_c", "eval_d"]
    assert (output_dir / "train" / "train_000" / "scenario.sumocfg").exists()
    assert (output_dir / "eval" / "eval_a" / "scenario.sumocfg").exists()
    assert not (output_dir / "eval" / "eval_extra").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_scenarios"] == ["train_000"]
    assert manifest["eval_scenarios"] == ["eval_a", "eval_b", "eval_c", "eval_d"]
    assert manifest["eval_capability_curated"] is True
    assert manifest["eval_capability_min_scenarios_per_bucket"] == 1
    assert manifest["eval_capability_required_peer_counts"] == [1, 2, 3]

    audit = json.loads((output_dir / "eval_capability_audit.json").read_text(encoding="utf-8"))
    assert [scenario["scenario_id"] for scenario in audit["scenarios"]] == [
        "eval_a",
        "eval_b",
        "eval_c",
        "eval_d",
    ]
