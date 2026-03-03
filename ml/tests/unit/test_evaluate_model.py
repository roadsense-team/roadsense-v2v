"""
Unit tests for evaluate_model aggregation helpers.
"""
import pytest

from ml.scripts import evaluate_model


def test_build_source_reaction_summary_includes_braking_signal_reception_rate():
    episodes = [
        {
            "peer_count": 4,
            "hazard_step": 40,
            "hazard_source_rank_ahead": 2,
            "hazard_source_id": "V004",
            "hazard_message_received_by_ego": True,
            "reaction_detected": True,
            "reaction_time_s": 0.2,
            "collision_post_hazard": False,
            "min_distance_post_hazard_m": 9.0,
            "hazard_any_braking_peer_received": True,
        },
        {
            "peer_count": 4,
            "hazard_step": 42,
            "hazard_source_rank_ahead": 2,
            "hazard_source_id": "V004",
            "hazard_message_received_by_ego": False,
            "reaction_detected": False,
            "reaction_time_s": None,
            "collision_post_hazard": True,
            "min_distance_post_hazard_m": 4.0,
            "hazard_any_braking_peer_received": False,
        },
    ]

    summary = evaluate_model._build_source_reaction_summary(episodes)

    bucket = summary["4"]["rank_2"]
    assert bucket["episodes"] == 2
    assert bucket["reception_rate"] == pytest.approx(0.5)
    assert bucket["reaction_rate"] == pytest.approx(1.0)
    assert bucket["avg_reaction_time_s"] == pytest.approx(0.2)
    assert bucket["collision_rate"] == pytest.approx(0.5)
    assert bucket["avg_min_distance_post_hazard_m"] == pytest.approx(6.5)
    assert bucket["braking_signal_reception_rate"] == pytest.approx(0.5)
