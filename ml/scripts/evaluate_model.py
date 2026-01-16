"""
Standalone evaluation script for trained RoadSense models.

Usage (from ml/ directory, inside Docker):
    python -m ml.scripts.evaluate_model --model_path ml/models/runs/cloud_prod_001/model_final.zip
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import ml.envs  # Registers environments


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
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path)

    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=None,
        dataset_dir=args.dataset_dir,
        scenario_mode="eval",
        emulator_params_path=args.emulator_params,
        hazard_injection=False,
        scenario_seed=args.seed,
    )

    print(f"\nRunning {args.episodes} evaluation episodes...\n")

    episode_rewards = []
    episode_lengths = []
    collisions = 0
    truncations = 0

    for ep in range(args.episodes):
        obs, info = env.reset()
        scenario_id = info.get("scenario_id", "unknown")

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if terminated:
            collisions += 1
        if truncated:
            truncations += 1

        status = "COLLISION" if terminated else "OK"
        print(f"  Episode {ep + 1:3d}/{args.episodes}: reward={total_reward:7.2f}, steps={steps:4d}, scenario={scenario_id}, status={status}")

    env.close()

    results = {
        "model_path": args.model_path,
        "episodes": args.episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "avg_length": float(np.mean(episode_lengths)),
        "collisions": collisions,
        "collision_rate": collisions / args.episodes,
        "truncations": truncations,
        "success_rate": (args.episodes - collisions) / args.episodes,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:       {args.episodes}")
    print(f"  Avg Reward:     {results['avg_reward']:.2f} (+/- {results['std_reward']:.2f})")
    print(f"  Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Avg Length:     {results['avg_length']:.1f} steps")
    print(f"  Collisions:     {collisions}/{args.episodes} ({results['collision_rate']*100:.1f}%)")
    print(f"  Success Rate:   {results['success_rate']*100:.1f}%")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
