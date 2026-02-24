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
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be > 0")
    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        parser.error("--episodes_per_scenario must be > 0")

    total_episodes = args.episodes
    if args.episodes_per_scenario is not None:
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
        obs, info = env.reset()
        scenario_id = info.get("scenario_id", "unknown")
        scenario_ids.append(scenario_id)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        peer_count = _infer_peer_count(args.dataset_dir, scenario_id, peer_count_cache)
        collision = bool(terminated)
        truncation = bool(truncated)
        outcome = "collision" if collision else "truncated" if truncation else "other"
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

    results = {
        "model_path": args.model_path,
        "episodes": total_episodes,
        "episodes_per_scenario": args.episodes_per_scenario,
        "hazard_injection": not args.no_hazard_injection,
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
        "scenarios_evaluated": scenario_ids,
        "scenario_summary": scenario_summary,
        "peer_count_summary": peer_count_summary,
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


if __name__ == "__main__":
    main()
