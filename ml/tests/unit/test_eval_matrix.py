"""
Unit tests for deterministic eval matrix planning/coverage (H3).
"""
import pytest

from ml.eval_matrix import (
    build_deterministic_eval_plan,
    summarize_deterministic_eval_coverage,
)


def test_build_deterministic_eval_plan_covers_required_buckets():
    eval_ids = [
        "eval_n1_000",
        "eval_n2_000",
        "eval_n3_000",
        "eval_n4_000",
        "eval_n5_000",
    ]
    peer_counts = {
        "eval_n1_000": 1,
        "eval_n2_000": 2,
        "eval_n3_000": 3,
        "eval_n4_000": 4,
        "eval_n5_000": 5,
    }

    plan, target_counts = build_deterministic_eval_plan(
        eval_scenario_ids=eval_ids,
        peer_counts_by_scenario=peer_counts,
        required_peer_counts=[1, 2, 3, 4, 5],
        episodes_per_bucket=2,
    )

    assert len(target_counts) == 15
    assert all(value == 2 for value in target_counts.values())
    assert len(plan) == 50
    assert [entry.scenario_id for entry in plan[:10]] == eval_ids * 2

    n5_ranks = [entry.source_rank_ahead for entry in plan if entry.peer_count == 5][:10]
    assert n5_ranks == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]


def test_build_deterministic_eval_plan_raises_for_missing_required_peer_count():
    with pytest.raises(ValueError, match="missing required peer counts"):
        build_deterministic_eval_plan(
            eval_scenario_ids=["eval_n1_000", "eval_n2_000"],
            peer_counts_by_scenario={"eval_n1_000": 1, "eval_n2_000": 2},
            required_peer_counts=[1, 2, 3],
            episodes_per_bucket=1,
        )


def test_summarize_deterministic_eval_coverage_reports_missing_bucket():
    target_counts = {
        (1, 1): 2,
        (2, 1): 2,
        (2, 2): 2,
    }
    episodes = [
        {"peer_count": 1, "hazard_source_rank_ahead": 1, "hazard_step": 40},
        {"peer_count": 1, "hazard_source_rank_ahead": 1, "hazard_step": 42},
        {"peer_count": 2, "hazard_source_rank_ahead": 1, "hazard_step": 40},
        {"peer_count": 2, "hazard_source_rank_ahead": 1, "hazard_step": 44},
        {"peer_count": 2, "hazard_source_rank_ahead": 2, "hazard_step": 45},
        {"peer_count": 2, "hazard_source_rank_ahead": 2, "hazard_step": None},
    ]

    summary = summarize_deterministic_eval_coverage(
        episodes=episodes,
        target_counts=target_counts,
    )

    assert summary["coverage_ok"] is False
    assert summary["observed_counts"]["n2_rank2"] == 1
    assert summary["missing_buckets"] == [
        {
            "peer_count": 2,
            "source_rank_ahead": 2,
            "expected_episodes": 2,
            "observed_episodes": 1,
        }
    ]


def test_plan_distributes_rank_targets_across_scenarios_of_same_peer_count():
    eval_ids = [
        "eval_n4_a",
        "eval_n4_b",
    ]
    peer_counts = {
        "eval_n4_a": 4,
        "eval_n4_b": 4,
    }

    plan, target_counts = build_deterministic_eval_plan(
        eval_scenario_ids=eval_ids,
        peer_counts_by_scenario=peer_counts,
        required_peer_counts=[4],
        episodes_per_bucket=10,
    )

    assert target_counts == {
        (4, 1): 10,
        (4, 2): 10,
        (4, 3): 10,
        (4, 4): 10,
    }

    n4_rank4 = [entry.scenario_id for entry in plan if entry.peer_count == 4 and entry.source_rank_ahead == 4]
    assert len(n4_rank4) >= 10
    assert set(n4_rank4) == {"eval_n4_a", "eval_n4_b"}
