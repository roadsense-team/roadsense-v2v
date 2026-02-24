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
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Steps per rollout (default: 2048).",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient (default: 0.01).",
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
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "avg_length": 0.0,
        }

    rewards = [float(episode["reward"]) for episode in episodes]
    lengths = [int(episode["length"]) for episode in episodes]
    collisions = sum(1 for episode in episodes if episode["collision"])
    truncations = sum(1 for episode in episodes if episode["truncated"])

    return {
        "episodes": episode_count,
        "collisions": collisions,
        "collision_rate": collisions / episode_count,
        "truncations": truncations,
        "truncation_rate": truncations / episode_count,
        "success_rate": (episode_count - collisions) / episode_count,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "avg_length": float(np.mean(lengths)),
    }


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
    if args.episodes_per_scenario is not None:
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
            obs, info = env.reset()
            scenario_id = info.get("scenario_id", "unknown")
            scenario_ids.append(scenario_id)

            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            peer_count = _infer_peer_count(args.dataset_dir, scenario_id, peer_count_cache)
            collision = bool(terminated)
            truncation = bool(truncated)
            outcome = "collision" if collision else "truncated" if truncation else "other"
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

    metrics = {
        "eval_episodes": eval_episodes,
        "episodes_per_scenario": args.episodes_per_scenario,
        "hazard_injection": not args.no_eval_hazard_injection,
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
        "scenarios_evaluated": scenario_ids,
        "scenario_summary": scenario_summary,
        "peer_count_summary": peer_count_summary,
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

    print(f"\nRun complete: {args.run_id}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
