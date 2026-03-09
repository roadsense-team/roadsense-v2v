"""
Unit tests for the eval matrix capability audit script.
"""
from typing import Dict, Tuple

from ml.scripts import audit_eval_matrix_capability


class FakeAuditEnv:
    def __init__(self, outcomes: Dict[Tuple[int, int], dict]):
        self._outcomes = outcomes
        self._current_rank = None
        self._current_step = None
        self.reset_calls = []

    def reset(self, *, options=None):
        options = options or {}
        self._current_rank = int(options["hazard_fixed_rank_ahead"])
        self._current_step = int(options["hazard_step"])
        self.reset_calls.append(dict(options))
        return "obs", {"hazard_step": self._current_step}

    def step(self, _action):
        outcome = self._outcomes[(self._current_rank, self._current_step)]
        info = {
            "step": self._current_step,
            "hazard_injection_attempted": outcome["attempted"],
            "hazard_injected": outcome["injected"],
            "hazard_injection_failed_reason": outcome["failure_reason"],
        }
        return "obs", 0.0, False, True, info


def test_probe_rank_support_records_success_and_failure():
    env = FakeAuditEnv(
        outcomes={
            (1, 30): {
                "attempted": True,
                "injected": True,
                "failure_reason": None,
            },
            (1, 31): {
                "attempted": True,
                "injected": False,
                "failure_reason": "no_front_peers",
            },
            (1, 32): {
                "attempted": False,
                "injected": False,
                "failure_reason": None,
            },
        }
    )

    results = audit_eval_matrix_capability._probe_rank_support(
        env=env,
        rank_ahead=1,
        hazard_steps=[30, 31, 32],
    )

    assert results == [
        {
            "hazard_step": 30,
            "injection_succeeded": True,
            "failure_reason": None,
        },
        {
            "hazard_step": 31,
            "injection_succeeded": False,
            "failure_reason": "no_front_peers",
        },
        {
            "hazard_step": 32,
            "injection_succeeded": False,
            "failure_reason": "hazard_not_attempted",
        },
    ]


def test_build_scenario_audit_record_summarizes_rank_support():
    scenario_record = audit_eval_matrix_capability._build_scenario_audit_record(
        scenario_id="eval_000",
        peer_count=3,
        probe_results_by_rank={
            1: [
                {
                    "hazard_step": 30,
                    "injection_succeeded": True,
                    "failure_reason": None,
                },
                {
                    "hazard_step": 31,
                    "injection_succeeded": False,
                    "failure_reason": "no_front_peers",
                },
            ],
            2: [
                {
                    "hazard_step": 30,
                    "injection_succeeded": False,
                    "failure_reason": "target_not_found_strategy=fixed_rank_ahead",
                },
            ],
            3: [],
        },
    )

    assert scenario_record["scenario_id"] == "eval_000"
    assert scenario_record["peer_count"] == 3
    assert scenario_record["supported_ranks_any_step"] == [1]
    assert scenario_record["supported_steps_by_rank"] == {
        "1": [30],
        "2": [],
        "3": [],
    }
    assert scenario_record["failed_steps_by_rank"] == {
        "1": [31],
        "2": [30],
        "3": [],
    }
    assert scenario_record["failure_reasons_by_rank"] == {
        "1": {"no_front_peers": 1},
        "2": {"target_not_found_strategy=fixed_rank_ahead": 1},
        "3": {},
    }
