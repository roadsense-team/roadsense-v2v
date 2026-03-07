"""
Standalone evaluation script for trained RoadSense models.

Usage (from ml/ directory, inside Docker):
    python -m ml.scripts.evaluate_model --model_path ml/models/runs/cloud_prod_001/model_final.zip
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import ml.envs  # Registers environments
from ml.eval_matrix import (
    build_deterministic_eval_plan,
    parse_peer_count_list,
    summarize_deterministic_eval_coverage,
)

REACTION_DECEL_THRESHOLD = 0.5
SIM_STEP_SECONDS = 0.1
UNSAFE_DIST_THRESHOLD_M = 10.0
BEHAVIORAL_SUCCESS_REWARD_THRESHOLD = -1000.0


def _load_eval_scenario_ids(dataset_dir: str) -> List[str]:
    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in dataset_dir: {dataset_dir}")

    with manifest_path.open("r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)

    eval_scenarios = manifest.get("eval_scenarios")
    if not isinstance(eval_scenarios, list):
        raise KeyError("manifest.json missing required list field: eval_scenarios")

    return [str(scenario_id) for scenario_id in eval_scenarios]


def _resolve_routes_path(dataset_dir: str, scenario_id: str) -> Optional[Path]:
    if not scenario_id or scenario_id == "unknown":
        return None

    dataset_root = Path(dataset_dir)
    candidates: List[Path] = []

    if scenario_id.startswith("eval_"):
        candidates.append(dataset_root / "eval" / scenario_id / "vehicles.rou.xml")
    elif scenario_id.startswith("train_"):
        candidates.append(dataset_root / "train" / scenario_id / "vehicles.rou.xml")

    candidates.extend(
        [
            dataset_root / "eval" / scenario_id / "vehicles.rou.xml",
            dataset_root / "train" / scenario_id / "vehicles.rou.xml",
        ]
    )

    for route_path in candidates:
        if route_path.exists():
            return route_path

    return None


def _infer_peer_count(
    dataset_dir: str,
    scenario_id: str,
    cache: Dict[str, Optional[int]],
) -> Optional[int]:
    if scenario_id in cache:
        return cache[scenario_id]

    routes_path = _resolve_routes_path(dataset_dir, scenario_id)
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


def _load_eval_peer_counts(
    dataset_dir: str,
    eval_scenario_ids: List[str],
) -> Dict[str, int]:
    """
    Load peer counts for eval scenarios, failing if any count cannot be inferred.
    """
    cache: Dict[str, Optional[int]] = {}
    peer_counts: Dict[str, int] = {}
    for scenario_id in eval_scenario_ids:
        peer_count = _infer_peer_count(dataset_dir, scenario_id, cache)
        if peer_count is None:
            raise ValueError(
                f"Could not infer peer_count for eval scenario: {scenario_id}"
            )
        peer_counts[scenario_id] = int(peer_count)
    return peer_counts


def _summarize_episode_group(episodes: List[dict]) -> dict:
    episode_count = len(episodes)
    if episode_count == 0:
        return {
            "episodes": 0,
            "collisions": 0,
            "collision_rate": 0.0,
            "truncations": 0,
            "truncation_rate": 0.0,
            "success_rate": 0.0,
            "behavioral_success_rate": 0.0,
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "avg_length": 0.0,
            "avg_pct_time_unsafe": 0.0,
        }

    rewards = [float(episode["reward"]) for episode in episodes]
    lengths = [int(episode["length"]) for episode in episodes]
    collisions = sum(1 for episode in episodes if episode["collision"])
    truncations = sum(1 for episode in episodes if episode["truncated"])
    behavioral_successes = sum(
        1 for episode in episodes
        if not episode["collision"] and episode["reward"] > BEHAVIORAL_SUCCESS_REWARD_THRESHOLD
    )
    pct_unsafe_values = [
        float(episode.get("pct_time_unsafe", 0.0)) for episode in episodes
    ]

    return {
        "episodes": episode_count,
        "collisions": collisions,
        "collision_rate": collisions / episode_count,
        "truncations": truncations,
        "truncation_rate": truncations / episode_count,
        "success_rate": (episode_count - collisions) / episode_count,
        "behavioral_success_rate": behavioral_successes / episode_count,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "avg_length": float(np.mean(lengths)),
        "avg_pct_time_unsafe": float(np.mean(pct_unsafe_values)),
    }


def _build_source_reaction_summary(episodes: List[dict]) -> Dict[str, Dict[str, dict]]:
    buckets: Dict[Tuple[str, str], List[dict]] = {}
    for episode in episodes:
        if episode.get("hazard_step") is None:
            continue

        peer_key = "unknown" if episode.get("peer_count") is None else str(episode["peer_count"])
        rank_ahead = episode.get("hazard_source_rank_ahead")
        source_id = episode.get("hazard_source_id")
        if rank_ahead is not None:
            source_key = f"rank_{int(rank_ahead)}"
        elif source_id:
            source_key = str(source_id)
        else:
            source_key = "unknown_source"

        buckets.setdefault((peer_key, source_key), []).append(episode)

    summary: Dict[str, Dict[str, dict]] = {}
    for (peer_key, source_key), bucket in buckets.items():
        episodes_count = len(bucket)
        reception_count = sum(
            1 for ep in bucket if bool(ep.get("hazard_message_received_by_ego", False))
        )
        reaction_count = sum(
            1
            for ep in bucket
            if bool(ep.get("hazard_message_received_by_ego", False))
            and bool(ep.get("reaction_detected", False))
        )
        reaction_times = [
            float(ep["reaction_time_s"])
            for ep in bucket
            if ep.get("hazard_message_received_by_ego", False)
            and ep.get("reaction_time_s") is not None
        ]
        min_distances = [
            float(ep["min_distance_post_hazard_m"])
            for ep in bucket
            if ep.get("min_distance_post_hazard_m") is not None
        ]
        collisions = sum(1 for ep in bucket if bool(ep.get("collision_post_hazard", False)))
        braking_reception_count = sum(
            1 for ep in bucket if bool(ep.get("hazard_any_braking_peer_received", False))
        )

        summary.setdefault(peer_key, {})[source_key] = {
            "episodes": episodes_count,
            "reception_rate": reception_count / episodes_count,
            "reaction_rate": (
                reaction_count / reception_count if reception_count > 0 else 0.0
            ),
            "avg_reaction_time_s": (
                float(np.mean(reaction_times)) if reaction_times else None
            ),
            "collision_rate": collisions / episodes_count,
            "avg_min_distance_post_hazard_m": (
                float(np.mean(min_distances)) if min_distances else None
            ),
            "braking_signal_reception_rate": braking_reception_count / episodes_count,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RoadSense model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="ml/scenarios/datasets/dataset_v1",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--emulator_params",
        type=str,
        default="ml/espnow_emulator/emulator_params_5m.json",
        help="Path to emulator params JSON",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--episodes_per_scenario",
        type=int,
        default=None,
        help=(
            "If set, overrides --episodes with "
            "(episodes_per_scenario * number_of_eval_scenarios)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )
    parser.add_argument(
        "--no_hazard_injection",
        action="store_true",
        help=(
            "Disable hazard injection during evaluation. "
            "By default hazard injection is ENABLED."
        ),
    )
    parser.add_argument(
        "--hazard_target_strategy",
        type=str,
        default=None,
        help=(
            "Optional hazard source strategy override "
            "(nearest|uniform_front_peers|fixed_vehicle_id|fixed_rank_ahead)."
        ),
    )
    parser.add_argument(
        "--hazard_fixed_vehicle_id",
        type=str,
        default=None,
        help="Optional fixed hazard source vehicle ID (e.g., V003).",
    )
    parser.add_argument(
        "--hazard_fixed_rank_ahead",
        type=int,
        default=None,
        help="Optional fixed hazard source rank ahead (1-based).",
    )
    parser.add_argument(
        "--hazard_step",
        type=int,
        default=None,
        help="Optional forced hazard step inside injector window.",
    )
    parser.add_argument(
        "--force_hazard",
        action="store_true",
        help="Force hazard injection every eval episode.",
    )
    parser.add_argument(
        "--use_deterministic_matrix",
        action="store_true",
        help=(
            "Enable deterministic H3 eval matrix coverage over "
            "(peer_count, source_rank_ahead) buckets."
        ),
    )
    parser.add_argument(
        "--matrix_peer_counts",
        type=str,
        default="1,2,3,4,5",
        help=(
            "Comma-separated peer counts that MUST be covered by deterministic "
            "eval matrix mode (default: 1,2,3,4,5)."
        ),
    )
    parser.add_argument(
        "--matrix_episodes_per_bucket",
        type=int,
        default=10,
        help=(
            "Minimum episodes per (peer_count, source_rank_ahead) bucket in "
            "deterministic eval matrix mode (default: 10)."
        ),
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be > 0")
    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        parser.error("--episodes_per_scenario must be > 0")
    if args.hazard_fixed_rank_ahead is not None and args.hazard_fixed_rank_ahead <= 0:
        parser.error("--hazard_fixed_rank_ahead must be >= 1")
    if args.matrix_episodes_per_bucket <= 0:
        parser.error("--matrix_episodes_per_bucket must be > 0")
    if args.use_deterministic_matrix and args.no_hazard_injection:
        parser.error("deterministic matrix requires hazard injection")

    total_episodes = args.episodes
    eval_matrix_plan = None
    eval_matrix_target_counts = None
    eval_matrix_required_peer_counts = None

    if args.use_deterministic_matrix:
        if args.hazard_target_strategy is not None:
            raise ValueError(
                "--use_deterministic_matrix is incompatible with --hazard_target_strategy."
            )
        if args.hazard_fixed_vehicle_id is not None:
            raise ValueError(
                "--use_deterministic_matrix is incompatible with --hazard_fixed_vehicle_id."
            )
        if args.hazard_fixed_rank_ahead is not None:
            raise ValueError(
                "--use_deterministic_matrix is incompatible with --hazard_fixed_rank_ahead."
            )

        eval_scenario_ids = _load_eval_scenario_ids(args.dataset_dir)
        if not eval_scenario_ids:
            raise ValueError("Dataset eval_scenarios is empty; cannot evaluate.")

        eval_matrix_required_peer_counts = parse_peer_count_list(args.matrix_peer_counts)
        peer_counts_by_scenario = _load_eval_peer_counts(
            args.dataset_dir,
            eval_scenario_ids,
        )
        eval_matrix_plan, eval_matrix_target_counts = build_deterministic_eval_plan(
            eval_scenario_ids=eval_scenario_ids,
            peer_counts_by_scenario=peer_counts_by_scenario,
            required_peer_counts=eval_matrix_required_peer_counts,
            episodes_per_bucket=args.matrix_episodes_per_bucket,
        )
        total_episodes = len(eval_matrix_plan)
    elif args.episodes_per_scenario is not None:
        eval_scenario_ids = _load_eval_scenario_ids(args.dataset_dir)
        if not eval_scenario_ids:
            raise ValueError("Dataset eval_scenarios is empty; cannot evaluate.")
        total_episodes = args.episodes_per_scenario * len(eval_scenario_ids)

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path)

    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=None,
        dataset_dir=args.dataset_dir,
        scenario_mode="eval",
        emulator_params_path=args.emulator_params,
        hazard_injection=not args.no_hazard_injection,
        scenario_seed=args.seed,
    )

    print(f"\nRunning {total_episodes} evaluation episodes...\n")

    episode_details = []
    scenario_ids = []
    peer_count_cache: Dict[str, Optional[int]] = {}

    for ep in range(total_episodes):
        plan_entry = eval_matrix_plan[ep] if eval_matrix_plan is not None else None
        reset_options = {}
        if plan_entry is not None:
            reset_options["hazard_target_strategy"] = "fixed_rank_ahead"
            reset_options["hazard_fixed_rank_ahead"] = plan_entry.source_rank_ahead
            reset_options["hazard_force"] = True
        else:
            if args.hazard_target_strategy is not None:
                reset_options["hazard_target_strategy"] = args.hazard_target_strategy
            if args.hazard_fixed_vehicle_id is not None:
                reset_options["hazard_fixed_vehicle_id"] = args.hazard_fixed_vehicle_id
            if args.hazard_fixed_rank_ahead is not None:
                reset_options["hazard_fixed_rank_ahead"] = args.hazard_fixed_rank_ahead
        if args.hazard_step is not None:
            reset_options["hazard_step"] = args.hazard_step
        if args.force_hazard and plan_entry is None:
            reset_options["hazard_force"] = True

        obs, info = env.reset(options=reset_options or None)
        scenario_id = info.get("scenario_id", "unknown")
        if plan_entry is not None and scenario_id != plan_entry.scenario_id:
            raise RuntimeError(
                "Deterministic eval matrix scenario mismatch: "
                f"expected {plan_entry.scenario_id}, got {scenario_id}"
            )
        scenario_ids.append(scenario_id)

        total_reward = 0.0
        steps = 0
        unsafe_steps = 0
        hazard_step = info.get("hazard_step")
        hazard_source_id = None
        hazard_source_rank_ahead = None
        hazard_injected = False
        hazard_injection_attempted = False
        hazard_injection_failed_reason = None
        hazard_message_received_by_ego = False
        hazard_any_braking_peer_received = False
        reaction_detected = False
        reaction_step = None
        min_distance_post_hazard_m = None
        unsafe_steps_post_hazard = 0
        post_hazard_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            distance = float(info.get("distance", 1000.0))
            if distance < UNSAFE_DIST_THRESHOLD_M:
                unsafe_steps += 1

            if info.get("hazard_injection_attempted", False):
                hazard_injection_attempted = True
                hazard_injection_failed_reason = info.get(
                    "hazard_injection_failed_reason"
                )

            if info.get("hazard_injected", False):
                hazard_injected = True
                hazard_source_id = info.get("hazard_source_id")
                hazard_step = info.get("hazard_step", info.get("step"))
                hazard_source_rank_ahead = info.get("hazard_source_rank_ahead")

            if hazard_injected:
                post_hazard_steps += 1
                if min_distance_post_hazard_m is None:
                    min_distance_post_hazard_m = distance
                else:
                    min_distance_post_hazard_m = min(min_distance_post_hazard_m, distance)

                if distance < UNSAFE_DIST_THRESHOLD_M:
                    unsafe_steps_post_hazard += 1

                mesh_ids = info.get("mesh_received_source_ids", [])
                if (
                    hazard_source_id is not None
                    and hazard_source_id in mesh_ids
                ):
                    hazard_message_received_by_ego = True

                if bool(info.get("mesh_any_braking_peer_received", False)):
                    hazard_any_braking_peer_received = True

                if (
                    not reaction_detected
                    # Intentional: reaction metric tracks RL policy brake command only.
                    # Car-following decel while released is excluded by design.
                    and float(info.get("deceleration", 0.0)) > REACTION_DECEL_THRESHOLD
                ):
                    reaction_detected = True
                    reaction_step = info.get("step")

        pct_time_unsafe = unsafe_steps / steps if steps > 0 else 0.0
        pct_time_unsafe_post_hazard = (
            unsafe_steps_post_hazard / post_hazard_steps
            if post_hazard_steps > 0
            else 0.0
        )
        reaction_time_s = None
        if (
            reaction_detected
            and reaction_step is not None
            and hazard_step is not None
        ):
            reaction_time_s = max(
                0.0, (int(reaction_step) - int(hazard_step)) * SIM_STEP_SECONDS
            )

        peer_count = _infer_peer_count(args.dataset_dir, scenario_id, peer_count_cache)
        collision = bool(terminated)
        truncation = bool(truncated)
        outcome = "collision" if collision else "truncated" if truncation else "other"
        collision_post_hazard = bool(collision and hazard_injected)
        episode_details.append(
            {
                "episode": ep + 1,
                "scenario_id": scenario_id,
                "peer_count": peer_count,
                "reward": float(total_reward),
                "length": steps,
                "collision": collision,
                "truncated": truncation,
                "success": not collision,
                "outcome": outcome,
                "pct_time_unsafe": float(pct_time_unsafe),
                "hazard_source_id": hazard_source_id,
                "hazard_step": hazard_step,
                "hazard_source_rank_ahead": hazard_source_rank_ahead,
                "hazard_injection_attempted": bool(hazard_injection_attempted),
                "hazard_injection_failed_reason": hazard_injection_failed_reason,
                "hazard_message_received_by_ego": bool(hazard_message_received_by_ego),
                "hazard_any_braking_peer_received": bool(hazard_any_braking_peer_received),
                "reaction_detected": bool(reaction_detected),
                "reaction_time_s": reaction_time_s,
                "min_distance_post_hazard_m": min_distance_post_hazard_m,
                "collision_post_hazard": collision_post_hazard,
                "pct_time_unsafe_post_hazard": float(pct_time_unsafe_post_hazard),
                "matrix_target_source_rank_ahead": (
                    plan_entry.source_rank_ahead if plan_entry is not None else None
                ),
            }
        )

        print(
            f"  Episode {ep + 1:3d}/{total_episodes}: reward={total_reward:7.2f}, "
            f"steps={steps:4d}, scenario={scenario_id}, "
            f"peers={peer_count if peer_count is not None else 'unknown'}, "
            f"outcome={outcome}"
        )

    env.close()

    overall_summary = _summarize_episode_group(episode_details)

    scenario_groups: Dict[str, List[dict]] = {}
    for episode in episode_details:
        scenario_groups.setdefault(episode["scenario_id"], []).append(episode)
    scenario_summary = {
        scenario_id: _summarize_episode_group(group_episodes)
        for scenario_id, group_episodes in sorted(scenario_groups.items())
    }

    peer_count_groups: Dict[str, List[dict]] = {}
    for episode in episode_details:
        peer_key = "unknown" if episode["peer_count"] is None else str(episode["peer_count"])
        peer_count_groups.setdefault(peer_key, []).append(episode)

    def _peer_sort_key(item: Tuple[str, List[dict]]) -> Tuple[int, int]:
        key = item[0]
        if key == "unknown":
            return (1, 0)
        return (0, int(key))

    peer_count_summary = {
        peer_key: _summarize_episode_group(group_episodes)
        for peer_key, group_episodes in sorted(peer_count_groups.items(), key=_peer_sort_key)
    }
    source_reaction_summary = _build_source_reaction_summary(episode_details)
    eval_matrix_summary = None
    eval_matrix_coverage_error = None
    if eval_matrix_target_counts is not None:
        eval_matrix_summary = summarize_deterministic_eval_coverage(
            episodes=episode_details,
            target_counts=eval_matrix_target_counts,
        )
        eval_matrix_summary["required_peer_counts"] = eval_matrix_required_peer_counts
        eval_matrix_summary["episodes_per_bucket"] = args.matrix_episodes_per_bucket
        eval_matrix_summary["planned_episodes"] = len(eval_matrix_plan or [])

        if not eval_matrix_summary["coverage_ok"]:
            eval_matrix_coverage_error = (
                "Deterministic eval matrix coverage failed: "
                f"{eval_matrix_summary['missing_buckets']}"
            )

    results = {
        "model_path": args.model_path,
        "episodes": total_episodes,
        "episodes_per_scenario": args.episodes_per_scenario,
        "hazard_injection": not args.no_hazard_injection,
        "deterministic_eval_matrix": bool(args.use_deterministic_matrix),
        "matrix_peer_counts": (
            eval_matrix_required_peer_counts if args.use_deterministic_matrix else None
        ),
        "matrix_episodes_per_bucket": (
            args.matrix_episodes_per_bucket if args.use_deterministic_matrix else None
        ),
        "hazard_target_strategy": args.hazard_target_strategy,
        "hazard_fixed_vehicle_id": args.hazard_fixed_vehicle_id,
        "hazard_fixed_rank_ahead": args.hazard_fixed_rank_ahead,
        "hazard_step": args.hazard_step,
        "force_hazard": args.force_hazard,
        "dataset_dir": args.dataset_dir,
        "emulator_params": args.emulator_params,
        "seed": args.seed,
        "avg_reward": overall_summary["avg_reward"],
        "std_reward": overall_summary["std_reward"],
        "min_reward": overall_summary["min_reward"],
        "max_reward": overall_summary["max_reward"],
        "avg_length": overall_summary["avg_length"],
        "collisions": overall_summary["collisions"],
        "collision_rate": overall_summary["collision_rate"],
        "truncations": overall_summary["truncations"],
        "truncation_rate": overall_summary["truncation_rate"],
        "success_rate": overall_summary["success_rate"],
        "behavioral_success_rate": overall_summary["behavioral_success_rate"],
        "avg_pct_time_unsafe": overall_summary["avg_pct_time_unsafe"],
        "scenarios_evaluated": scenario_ids,
        "scenario_summary": scenario_summary,
        "peer_count_summary": peer_count_summary,
        "source_reaction_summary": source_reaction_summary,
        "eval_matrix": eval_matrix_summary,
        "eval_matrix_coverage_error": eval_matrix_coverage_error,
        "episode_details": episode_details,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:       {total_episodes}")
    print(f"  Avg Reward:     {results['avg_reward']:.2f} (+/- {results['std_reward']:.2f})")
    print(f"  Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Avg Length:     {results['avg_length']:.1f} steps")
    print(
        f"  Collisions:     {results['collisions']}/{total_episodes} "
        f"({results['collision_rate']*100:.1f}%)"
    )
    print(f"  Success Rate:   {results['success_rate']*100:.1f}%")
    print(f"  Hazard Injection: {results['hazard_injection']}")
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    if eval_matrix_coverage_error is not None:
        raise RuntimeError(str(eval_matrix_coverage_error))


if __name__ == "__main__":
    main()
