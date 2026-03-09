"""
Audit which eval scenarios can support which (peer_count, rank_ahead) buckets.
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np

import ml.envs  # Registers environments
from ml.eval_dataset import (
    get_eval_capability_audit_path,
    infer_peer_count,
    load_eval_scenario_ids,
    resolve_sumo_cfg_path,
)
from ml.envs.hazard_injector import HazardInjector


def _probe_rank_support(
    env,
    rank_ahead: int,
    hazard_steps: Iterable[int],
) -> List[dict]:
    results: List[dict] = []
    for hazard_step in hazard_steps:
        _, _ = env.reset(
            options={
                "hazard_target_strategy": "fixed_rank_ahead",
                "hazard_fixed_rank_ahead": int(rank_ahead),
                "hazard_step": int(hazard_step),
                "hazard_force": True,
            }
        )

        attempted = False
        injected = False
        failure_reason: Optional[str] = None
        terminated = False
        truncated = False

        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.0], dtype=np.float32)
            )
            if info.get("hazard_injection_attempted", False):
                attempted = True
                injected = bool(info.get("hazard_injected", False))
                failure_reason = info.get("hazard_injection_failed_reason")
                break

            step_value = info.get("step")
            if step_value is not None and int(step_value) > int(hazard_step):
                break

        if not attempted:
            failure_reason = "hazard_not_attempted"

        results.append(
            {
                "hazard_step": int(hazard_step),
                "injection_succeeded": bool(attempted and injected),
                "failure_reason": None if attempted and injected else failure_reason,
            }
        )

    return results


def _build_scenario_audit_record(
    scenario_id: str,
    peer_count: int,
    probe_results_by_rank: Dict[int, List[dict]],
) -> dict:
    supported_ranks_any_step: List[int] = []
    supported_steps_by_rank: Dict[str, List[int]] = {}
    failed_steps_by_rank: Dict[str, List[int]] = {}
    failure_reasons_by_rank: Dict[str, Dict[str, int]] = {}

    for rank_ahead in range(1, int(peer_count) + 1):
        probe_results = list(probe_results_by_rank.get(rank_ahead, []))
        supported_steps = [
            int(result["hazard_step"])
            for result in probe_results
            if bool(result.get("injection_succeeded", False))
        ]
        failed_steps = [
            int(result["hazard_step"])
            for result in probe_results
            if not bool(result.get("injection_succeeded", False))
        ]
        failure_reason_counts: Dict[str, int] = defaultdict(int)
        for result in probe_results:
            if bool(result.get("injection_succeeded", False)):
                continue
            failure_reason = result.get("failure_reason") or "unknown_failure"
            failure_reason_counts[str(failure_reason)] += 1

        if supported_steps:
            supported_ranks_any_step.append(rank_ahead)

        rank_key = str(rank_ahead)
        supported_steps_by_rank[rank_key] = supported_steps
        failed_steps_by_rank[rank_key] = failed_steps
        failure_reasons_by_rank[rank_key] = dict(failure_reason_counts)

    return {
        "scenario_id": scenario_id,
        "peer_count": int(peer_count),
        "supported_ranks_any_step": supported_ranks_any_step,
        "supported_steps_by_rank": supported_steps_by_rank,
        "failed_steps_by_rank": failed_steps_by_rank,
        "failure_reasons_by_rank": failure_reasons_by_rank,
    }


def audit_dataset_capability(
    dataset_dir: str,
    emulator_params: str,
    seed: int,
    scenario_ids: Optional[List[str]] = None,
    hazard_steps: Optional[Iterable[int]] = None,
) -> dict:
    eval_scenario_ids = (
        list(scenario_ids) if scenario_ids is not None else load_eval_scenario_ids(dataset_dir)
    )
    peer_count_cache: Dict[str, Optional[int]] = {}
    hazard_steps_list = list(hazard_steps) if hazard_steps is not None else list(
        range(
            HazardInjector.HAZARD_WINDOW_START,
            HazardInjector.HAZARD_WINDOW_END + 1,
        )
    )

    scenarios: List[dict] = []
    for scenario_id in eval_scenario_ids:
        peer_count = infer_peer_count(dataset_dir, scenario_id, peer_count_cache)
        if peer_count is None:
            raise ValueError(f"Could not infer peer_count for eval scenario: {scenario_id}")

        sumo_cfg_path = resolve_sumo_cfg_path(dataset_dir, scenario_id)
        if sumo_cfg_path is None:
            raise FileNotFoundError(
                f"scenario.sumocfg not found for eval scenario: {scenario_id}"
            )

        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=str(sumo_cfg_path),
            emulator_params_path=emulator_params,
            hazard_injection=True,
            scenario_seed=seed,
        )

        try:
            probe_results_by_rank = {
                rank_ahead: _probe_rank_support(
                    env=env,
                    rank_ahead=rank_ahead,
                    hazard_steps=hazard_steps_list,
                )
                for rank_ahead in range(1, peer_count + 1)
            }
        finally:
            env.close()

        scenarios.append(
            _build_scenario_audit_record(
                scenario_id=scenario_id,
                peer_count=peer_count,
                probe_results_by_rank=probe_results_by_rank,
            )
        )

    return {
        "dataset_dir": dataset_dir,
        "emulator_params": emulator_params,
        "seed": int(seed),
        "hazard_window_start": int(hazard_steps_list[0]),
        "hazard_window_end": int(hazard_steps_list[-1]),
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit eval scenario capability for deterministic matrix coverage."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing manifest.json",
    )
    parser.add_argument(
        "--emulator_params",
        type=str,
        default="ml/espnow_emulator/emulator_params_measured.json",
        help="Path to emulator params JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic probing",
    )
    parser.add_argument(
        "--scenario_ids",
        type=str,
        default=None,
        help="Optional comma-separated subset of eval scenario IDs to audit",
    )
    parser.add_argument(
        "--hazard_window_start",
        type=int,
        default=HazardInjector.HAZARD_WINDOW_START,
        help="First hazard step to probe",
    )
    parser.add_argument(
        "--hazard_window_end",
        type=int,
        default=HazardInjector.HAZARD_WINDOW_END,
        help="Last hazard step to probe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <dataset_dir>/eval_capability_audit.json",
    )
    args = parser.parse_args()

    if args.hazard_window_start > args.hazard_window_end:
        parser.error("--hazard_window_start must be <= --hazard_window_end")

    return args


def main() -> None:
    args = parse_args()
    scenario_ids = None
    if args.scenario_ids:
        scenario_ids = [
            scenario_id.strip()
            for scenario_id in args.scenario_ids.split(",")
            if scenario_id.strip()
        ]

    audit = audit_dataset_capability(
        dataset_dir=args.dataset_dir,
        emulator_params=args.emulator_params,
        seed=args.seed,
        scenario_ids=scenario_ids,
        hazard_steps=range(args.hazard_window_start, args.hazard_window_end + 1),
    )

    output_path = args.output or str(get_eval_capability_audit_path(args.dataset_dir))
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(audit, file_handle, indent=2, sort_keys=True)

    print(f"Wrote eval capability audit: {output_path}")


if __name__ == "__main__":
    main()
