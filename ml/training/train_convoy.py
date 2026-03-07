"""
Train PPO on RoadSense-Convoy-v0 using Deep Sets feature extractor.

Run from repo root:
    python -m ml.training.train_convoy
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ml.eval_matrix import (
    build_deterministic_eval_plan,
    parse_peer_count_list,
    summarize_deterministic_eval_coverage,
)
from ml.policies.deep_set_policy import create_deep_set_policy_kwargs
import ml.envs  # Registers environments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO on RoadSense-Convoy environment with Deep Sets."
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "Path to dataset directory (contains manifest.json). "
            "If not provided, uses single scenario from --sumo_cfg."
        ),
    )
    parser.add_argument(
        "--sumo_cfg",
        type=str,
        default=None,
        help="Path to single scenario.sumocfg (mutually exclusive with --dataset_dir).",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training (default: 10).",
    )
    parser.add_argument(
        "--episodes_per_scenario",
        type=int,
        default=None,
        help=(
            "If set, overrides --eval_episodes with "
            "(episodes_per_scenario * number_of_eval_scenarios)."
        ),
    )

    # Emulator configuration
    parser.add_argument(
        "--emulator_params",
        type=str,
        default=None,
        help="Path to emulator params JSON. If not provided, uses defaults.",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training and scenario selection (default: 42).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100,000).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=4096,
        help="Steps per rollout (default: 4096).",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.0,
        help="Entropy coefficient (default: 0.0). With log_std_init=-0.5 the learnable "
        "std already provides exploration; ent_coef>0 fights convergence in late training.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for model and metrics. "
            "If not provided, uses ml/models/runs/<run_id>/."
        ),
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier. If not provided, generates from timestamp.",
    )

    # Flags
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training.",
    )
    parser.add_argument(
        "--no_eval_hazard_injection",
        action="store_true",
        help=(
            "Disable hazard injection during post-training evaluation. "
            "By default hazard injection is ENABLED for evaluation."
        ),
    )
    parser.add_argument(
        "--eval_hazard_target_strategy",
        type=str,
        default=None,
        help=(
            "Optional eval-only hazard source strategy override "
            "(nearest|uniform_front_peers|fixed_vehicle_id|fixed_rank_ahead)."
        ),
    )
    parser.add_argument(
        "--eval_hazard_fixed_vehicle_id",
        type=str,
        default=None,
        help="Optional eval-only fixed hazard source vehicle ID (e.g., V003).",
    )
    parser.add_argument(
        "--eval_hazard_fixed_rank_ahead",
        type=int,
        default=None,
        help="Optional eval-only fixed hazard source rank ahead (1-based).",
    )
    parser.add_argument(
        "--eval_hazard_step",
        type=int,
        default=None,
        help="Optional eval-only forced hazard step inside injector window.",
    )
    parser.add_argument(
        "--eval_force_hazard",
        action="store_true",
        help="Force hazard injection every eval episode.",
    )
    parser.add_argument(
        "--eval_use_deterministic_matrix",
        action="store_true",
        help=(
            "Enable deterministic H3 eval matrix coverage over "
            "(peer_count, source_rank_ahead) buckets."
        ),
    )
    parser.add_argument(
        "--eval_matrix_peer_counts",
        type=str,
        default="1,2,3,4,5",
        help=(
            "Comma-separated peer counts that MUST be covered by deterministic "
            "eval matrix mode (default: 1,2,3,4,5)."
        ),
    )
    parser.add_argument(
        "--eval_matrix_episodes_per_bucket",
        type=int,
        default=10,
        help=(
            "Minimum episodes per (peer_count, source_rank_ahead) bucket in "
            "deterministic eval matrix mode (default: 10)."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO with GUI (for debugging).",
    )

    args = parser.parse_args()

    # Validation
    if args.dataset_dir is None and args.sumo_cfg is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.sumo_cfg = os.path.join(base_dir, "scenarios", "base", "scenario.sumocfg")

    if args.dataset_dir is not None and args.sumo_cfg is not None:
        parser.error("Cannot specify both --dataset_dir and --sumo_cfg")

    if args.eval_episodes <= 0:
        parser.error("--eval_episodes must be > 0")

    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        parser.error("--episodes_per_scenario must be > 0")

    if (
        args.eval_hazard_fixed_rank_ahead is not None
        and args.eval_hazard_fixed_rank_ahead <= 0
    ):
        parser.error("--eval_hazard_fixed_rank_ahead must be >= 1")
    if args.eval_matrix_episodes_per_bucket <= 0:
        parser.error("--eval_matrix_episodes_per_bucket must be > 0")
    if args.eval_use_deterministic_matrix and args.no_eval_hazard_injection:
        parser.error("deterministic matrix requires hazard injection")

    return args


def _load_eval_scenario_ids(dataset_dir: str) -> List[str]:
    """Load eval scenario IDs from dataset manifest."""
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
    """Resolve vehicles.rou.xml path for a scenario ID."""
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
    dataset_dir: Optional[str],
    scenario_id: str,
    cache: Dict[str, Optional[int]],
) -> Optional[int]:
    """
    Infer peer count from vehicles.rou.xml by counting non-V001 vehicles.
    """
    if scenario_id in cache:
        return cache[scenario_id]

    if dataset_dir is None:
        cache[scenario_id] = None
        return None

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


BEHAVIORAL_SUCCESS_REWARD_THRESHOLD = -1000.0
REACTION_DECEL_THRESHOLD = 0.5
SIM_STEP_SECONDS = 0.1
UNSAFE_DIST_THRESHOLD_M = 10.0


def _summarize_episode_group(episodes: List[dict]) -> dict:
    """Build aggregate metrics for a list of episode details."""
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
        1 for ep in episodes
        if not ep["collision"] and ep["reward"] > BEHAVIORAL_SUCCESS_REWARD_THRESHOLD
    )

    pct_unsafe_values = [
        float(ep.get("pct_time_unsafe", 0.0)) for ep in episodes
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
    """
    Aggregate hazard-source reaction metrics by peer-count and source bucket.
    """
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


def train(args: argparse.Namespace) -> Tuple[str, Dict[str, int]]:
    """
    Train the model and return output path and metrics.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (model_path, metrics_dict)
    """
    if args.run_id is None:
        args.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    if args.output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output_dir = os.path.join(base_dir, "models", "runs", args.run_id)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    env_kwargs = {
        "render_mode": "human" if args.gui else None,
        "gui": args.gui,
        "emulator_params_path": args.emulator_params,
        "scenario_seed": args.seed,
    }

    if args.dataset_dir is not None:
        env_kwargs["dataset_dir"] = args.dataset_dir
        env_kwargs["scenario_mode"] = "train"
        env_kwargs["sumo_cfg"] = None
    else:
        env_kwargs["sumo_cfg"] = args.sumo_cfg

    env = gym.make("RoadSense-Convoy-v0", **env_kwargs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy_kwargs = create_deep_set_policy_kwargs(peer_embed_dim=32)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="deep_sets",
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
        seed=args.seed,
    )

    print(f"Starting training: {args.total_timesteps} timesteps")
    print(f"Output directory: {args.output_dir}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
        )
    finally:
        env.close()

    model_path = os.path.join(args.output_dir, "model_final.zip")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    return model_path, {"training_timesteps": args.total_timesteps}


def evaluate(model_path: str, args: argparse.Namespace) -> dict:
    """
    Evaluate a trained model on the eval scenario set.

    Args:
        model_path: Path to saved model
        args: Parsed arguments (for dataset_dir, emulator_params, etc.)

    Returns:
        Dictionary of evaluation metrics
    """
    eval_episodes = args.eval_episodes
    eval_matrix_plan = None
    eval_matrix_target_counts = None
    eval_matrix_required_peer_counts = None
    use_eval_matrix = bool(getattr(args, "eval_use_deterministic_matrix", False))

    if use_eval_matrix:
        if args.dataset_dir is None:
            raise ValueError(
                "--eval_use_deterministic_matrix requires --dataset_dir."
            )
        if args.eval_hazard_target_strategy is not None:
            raise ValueError(
                "--eval_use_deterministic_matrix is incompatible with "
                "--eval_hazard_target_strategy."
            )
        if args.eval_hazard_fixed_vehicle_id is not None:
            raise ValueError(
                "--eval_use_deterministic_matrix is incompatible with "
                "--eval_hazard_fixed_vehicle_id."
            )
        if args.eval_hazard_fixed_rank_ahead is not None:
            raise ValueError(
                "--eval_use_deterministic_matrix is incompatible with "
                "--eval_hazard_fixed_rank_ahead."
            )

        eval_scenario_ids = _load_eval_scenario_ids(args.dataset_dir)
        if not eval_scenario_ids:
            raise ValueError("Dataset eval_scenarios is empty; cannot evaluate.")

        eval_matrix_required_peer_counts = parse_peer_count_list(
            args.eval_matrix_peer_counts
        )
        peer_counts_by_scenario = _load_eval_peer_counts(
            args.dataset_dir,
            eval_scenario_ids,
        )
        eval_matrix_plan, eval_matrix_target_counts = build_deterministic_eval_plan(
            eval_scenario_ids=eval_scenario_ids,
            peer_counts_by_scenario=peer_counts_by_scenario,
            required_peer_counts=eval_matrix_required_peer_counts,
            episodes_per_bucket=args.eval_matrix_episodes_per_bucket,
        )
        eval_episodes = len(eval_matrix_plan)
    elif args.episodes_per_scenario is not None:
        if args.dataset_dir is None:
            eval_episodes = args.episodes_per_scenario
        else:
            eval_scenario_ids = _load_eval_scenario_ids(args.dataset_dir)
            if not eval_scenario_ids:
                raise ValueError("Dataset eval_scenarios is empty; cannot evaluate.")
            eval_episodes = args.episodes_per_scenario * len(eval_scenario_ids)

    print(f"\nStarting evaluation: {eval_episodes} episodes")

    model = PPO.load(model_path)

    env_kwargs = {
        "render_mode": "human" if args.gui else None,
        "gui": args.gui,
        "hazard_injection": not args.no_eval_hazard_injection,
        "emulator_params_path": args.emulator_params,
        "scenario_seed": args.seed,
    }

    if args.dataset_dir is not None:
        env_kwargs["dataset_dir"] = args.dataset_dir
        env_kwargs["scenario_mode"] = "eval"
        env_kwargs["sumo_cfg"] = None
    else:
        env_kwargs["sumo_cfg"] = args.sumo_cfg

    env = gym.make("RoadSense-Convoy-v0", **env_kwargs)

    episode_details = []
    scenario_ids = []
    peer_count_cache: Dict[str, Optional[int]] = {}

    try:
        for ep in range(eval_episodes):
            plan_entry = eval_matrix_plan[ep] if eval_matrix_plan is not None else None
            reset_options = {}
            if plan_entry is not None:
                reset_options["hazard_target_strategy"] = "fixed_rank_ahead"
                reset_options["hazard_fixed_rank_ahead"] = plan_entry.source_rank_ahead
                reset_options["hazard_force"] = True
            else:
                if args.eval_hazard_target_strategy is not None:
                    reset_options["hazard_target_strategy"] = args.eval_hazard_target_strategy
                if args.eval_hazard_fixed_vehicle_id is not None:
                    reset_options["hazard_fixed_vehicle_id"] = args.eval_hazard_fixed_vehicle_id
                if args.eval_hazard_fixed_rank_ahead is not None:
                    reset_options["hazard_fixed_rank_ahead"] = args.eval_hazard_fixed_rank_ahead
            if args.eval_hazard_step is not None:
                reset_options["hazard_step"] = args.eval_hazard_step
            if args.eval_force_hazard and plan_entry is None:
                reset_options["hazard_force"] = True

            obs, info = env.reset(options=reset_options or None)
            scenario_id = info.get("scenario_id", "unknown")
            if plan_entry is not None and scenario_id != plan_entry.scenario_id:
                raise RuntimeError(
                    "Deterministic eval matrix scenario mismatch: "
                    f"expected {plan_entry.scenario_id}, got {scenario_id}"
                )
            scenario_ids.append(scenario_id)

            episode_reward = 0.0
            episode_length = 0
            unsafe_steps = 0
            hazard_step = info.get("hazard_step")
            hazard_source_id = None
            hazard_source_rank_ahead = None
            hazard_injected = False
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
                episode_reward += reward
                episode_length += 1
                distance = float(info.get("distance", 1000.0))
                if distance < UNSAFE_DIST_THRESHOLD_M:
                    unsafe_steps += 1

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

            pct_time_unsafe = (
                unsafe_steps / episode_length if episode_length > 0 else 0.0
            )
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
                    "reward": float(episode_reward),
                    "length": episode_length,
                    "collision": collision,
                    "truncated": truncation,
                    "success": not collision,
                    "outcome": outcome,
                    "pct_time_unsafe": float(pct_time_unsafe),
                    "hazard_source_id": hazard_source_id,
                    "hazard_step": hazard_step,
                    "hazard_source_rank_ahead": hazard_source_rank_ahead,
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
                f"  Episode {ep + 1}/{eval_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}, "
                f"scenario={scenario_id}, "
                f"peers={peer_count if peer_count is not None else 'unknown'}, "
                f"outcome={outcome}"
            )
    finally:
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
        eval_matrix_summary["episodes_per_bucket"] = args.eval_matrix_episodes_per_bucket
        eval_matrix_summary["planned_episodes"] = len(eval_matrix_plan or [])

        if not eval_matrix_summary["coverage_ok"]:
            eval_matrix_coverage_error = (
                "Deterministic eval matrix coverage failed: "
                f"{eval_matrix_summary['missing_buckets']}"
            )

    metrics = {
        "eval_episodes": eval_episodes,
        "episodes_per_scenario": args.episodes_per_scenario,
        "hazard_injection": not args.no_eval_hazard_injection,
        "deterministic_eval_matrix": use_eval_matrix,
        "eval_matrix_peer_counts": (
            eval_matrix_required_peer_counts if use_eval_matrix else None
        ),
        "eval_matrix_episodes_per_bucket": (
            args.eval_matrix_episodes_per_bucket if use_eval_matrix else None
        ),
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

    print("\nEvaluation Summary:")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f} (+/- {metrics['std_reward']:.2f})")
    print(f"  Avg Length: {metrics['avg_length']:.1f}")
    print(
        f"  Collisions: {metrics['collisions']}/{eval_episodes} "
        f"({metrics['collision_rate'] * 100:.1f}%)"
    )
    print(f"  Success Rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"  Hazard Injection: {metrics['hazard_injection']}")

    return metrics


def save_metrics(
    output_dir: str,
    training_metrics: dict,
    eval_metrics: Optional[dict],
    args: argparse.Namespace,
) -> str:
    """
    Save combined metrics to JSON file.

    Args:
        output_dir: Output directory
        training_metrics: Metrics from training
        eval_metrics: Metrics from evaluation (or None if skipped)
        args: Original arguments (for reproducibility)

    Returns:
        Path to saved metrics file
    """
    metrics = {
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset_dir": args.dataset_dir,
            "sumo_cfg": args.sumo_cfg,
            "emulator_params": args.emulator_params,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "eval_episodes": args.eval_episodes,
            "episodes_per_scenario": args.episodes_per_scenario,
            "eval_hazard_injection": not args.no_eval_hazard_injection,
            "eval_hazard_target_strategy": args.eval_hazard_target_strategy,
            "eval_hazard_fixed_vehicle_id": args.eval_hazard_fixed_vehicle_id,
            "eval_hazard_fixed_rank_ahead": args.eval_hazard_fixed_rank_ahead,
            "eval_hazard_step": args.eval_hazard_step,
            "eval_force_hazard": args.eval_force_hazard,
            "eval_use_deterministic_matrix": args.eval_use_deterministic_matrix,
            "eval_matrix_peer_counts": args.eval_matrix_peer_counts,
            "eval_matrix_episodes_per_bucket": args.eval_matrix_episodes_per_bucket,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "ent_coef": args.ent_coef,
        },
        "training": training_metrics,
        "evaluation": eval_metrics,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    print(f"Metrics saved to: {metrics_path}")
    return metrics_path


def main() -> None:
    """Main entry point."""
    args = parse_args()

    model_path, training_metrics = train(args)

    eval_metrics = None
    if not args.skip_eval:
        eval_metrics = evaluate(model_path, args)

    save_metrics(args.output_dir, training_metrics, eval_metrics, args)

    if eval_metrics is not None and eval_metrics.get("eval_matrix_coverage_error"):
        raise RuntimeError(str(eval_metrics["eval_matrix_coverage_error"]))

    print(f"\nRun complete: {args.run_id}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
