"""
Unit tests for smoke V2V reaction verdict helpers.
"""
import pytest

from ml.scripts import check_v2v_reaction


def test_build_peer_reaction_summary_groups_by_peer_count():
    episodes = [
        {
            "peer_count": 2,
            "hazard_injected": True,
            "hazard_message_received_by_ego": True,
            "reaction_detected": True,
            "reaction_time_s": 0.4,
        },
        {
            "peer_count": 2,
            "hazard_injected": True,
            "hazard_message_received_by_ego": True,
            "reaction_detected": False,
            "reaction_time_s": None,
        },
        {
            "peer_count": 5,
            "hazard_injected": True,
            "hazard_message_received_by_ego": False,
            "reaction_detected": False,
            "reaction_time_s": None,
        },
    ]

    summary = check_v2v_reaction._build_peer_reaction_summary(episodes)

    assert summary["2"]["episodes"] == 2
    assert summary["2"]["hazard_injected_episodes"] == 2
    assert summary["2"]["hazard_received_episodes"] == 2
    assert summary["2"]["reaction_episodes"] == 1
    assert summary["2"]["reaction_rate"] == pytest.approx(0.5)
    assert summary["2"]["avg_reaction_time_s"] == pytest.approx(0.4)

    assert summary["5"]["hazard_received_episodes"] == 0
    assert summary["5"]["reaction_rate"] == pytest.approx(0.0)


def test_build_verdict_flags_nonzero_reaction():
    episodes = [
        {
            "hazard_injected": True,
            "hazard_message_received_by_ego": True,
            "hazard_any_braking_peer_received": True,
            "reaction_detected": True,
            "reaction_time_s": 0.3,
        },
        {
            "hazard_injected": True,
            "hazard_message_received_by_ego": True,
            "hazard_any_braking_peer_received": True,
            "reaction_detected": False,
            "reaction_time_s": None,
        },
    ]

    peer_summary = {
        "3": {
            "episodes": 2,
            "hazard_injected_episodes": 2,
            "hazard_received_episodes": 2,
            "reaction_episodes": 1,
            "reception_rate": 1.0,
            "reaction_rate": 0.5,
            "avg_reaction_time_s": 0.3,
        }
    }
    source_summary = {"3": {"rank_1": {"reaction_rate": 0.5}}}

    verdict = check_v2v_reaction._build_verdict(
        episodes,
        peer_summary,
        source_summary,
    )

    assert verdict["hazard_injected_episodes"] == 2
    assert verdict["hazard_received_episodes"] == 2
    assert verdict["reaction_episodes"] == 1
    assert verdict["overall_reaction_rate"] == pytest.approx(0.5)
    assert verdict["avg_reaction_time_s"] == pytest.approx(0.3)
    assert verdict["v2v_reaction_detected"] is True
