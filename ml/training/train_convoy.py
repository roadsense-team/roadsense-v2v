"""
Train PPO on RoadSense-Convoy-v0 using Deep Sets feature extractor.

Run from repo root:
    python -m ml.training.train_convoy
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

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

    return args


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
    print(f"\nStarting evaluation: {args.eval_episodes} episodes")

    model = PPO.load(model_path)

    env_kwargs = {
        "render_mode": "human" if args.gui else None,
        "gui": args.gui,
        "hazard_injection": False,
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

    episode_rewards = []
    episode_lengths = []
    collisions = 0
    truncations = 0
    scenario_ids = []

    try:
        for ep in range(args.eval_episodes):
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

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if terminated:
                collisions += 1
            if truncated:
                truncations += 1

            print(
                f"  Episode {ep + 1}/{args.eval_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}, "
                f"scenario={scenario_id}"
            )
    finally:
        env.close()

    metrics = {
        "eval_episodes": args.eval_episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "avg_length": float(np.mean(episode_lengths)),
        "collisions": collisions,
        "collision_rate": collisions / args.eval_episodes,
        "truncations": truncations,
        "truncation_rate": truncations / args.eval_episodes,
        "scenarios_evaluated": scenario_ids,
    }

    print("\nEvaluation Summary:")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f} (+/- {metrics['std_reward']:.2f})")
    print(f"  Avg Length: {metrics['avg_length']:.1f}")
    print(
        f"  Collisions: {metrics['collisions']}/{args.eval_episodes} "
        f"({metrics['collision_rate'] * 100:.1f}%)"
    )

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
