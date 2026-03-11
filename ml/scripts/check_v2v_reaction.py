"""
Standalone smoke-check for V2V reaction on a saved model.

Usage (from ml/ directory, inside Docker):
    python -m ml.scripts.check_v2v_reaction \
        --model_path results/smoke_014_v6_100k/model_final.zip \
        --dataset_dir ml/scenarios/datasets/dataset_v6_formation_fix
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import ml.envs  # Registers environments
from ml.eval_dataset import infer_peer_count, load_eval_scenario_ids
from ml.scripts import evaluate_model

REACTION_DECEL_THRESHOLD = 0.5
SIM_STEP_SECONDS = 0.1
UNSAFE_DIST_THRESHOLD_M = 10.0


def _build_peer_reaction_summary(episodes: List[dict]) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for episode in episodes:
        peer_key = "unknown" if episode.get("peer_count") is None else str(episode["peer_count"])
        grouped.setdefault(peer_key, []).append(episode)

    def _sort_key(item: tuple[str, List[dict]]) -> tuple[int, int]:
        key = item[0]
        if key == "unknown":
            return (1, 0)
        return (0, int(key))

    summary: Dict[str, dict] = {}
    for peer_key, bucket in sorted(grouped.items(), key=_sort_key):
        episodes_count = len(bucket)
        injected_count = sum(1 for ep in bucket if bool(ep.get("hazard_injected", False)))
        received_count = sum(
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

        summary[peer_key] = {
            "episodes": episodes_count,
            "hazard_injected_episodes": injected_count,
            "hazard_received_episodes": received_count,
            "reaction_episodes": reaction_count,
            "reception_rate": (
                received_count / injected_count if injected_count > 0 else 0.0
            ),
            "reaction_rate": (
                reaction_count / received_count if received_count > 0 else 0.0
            ),
            "avg_reaction_time_s": (
                float(np.mean(reaction_times)) if reaction_times else None
            ),
        }

    return summary


def _build_verdict(
    episodes: List[dict],
    peer_summary: Dict[str, dict],
    source_summary: Dict[str, Dict[str, dict]],
) -> dict:
    total_episodes = len(episodes)
    injected_count = sum(1 for ep in episodes if bool(ep.get("hazard_injected", False)))
    received_count = sum(
        1 for ep in episodes if bool(ep.get("hazard_message_received_by_ego", False))
    )
    braking_signal_count = sum(
        1 for ep in episodes if bool(ep.get("hazard_any_braking_peer_received", False))
    )
    reaction_count = sum(
        1
        for ep in episodes
        if bool(ep.get("hazard_message_received_by_ego", False))
        and bool(ep.get("reaction_detected", False))
    )
    reaction_times = [
        float(ep["reaction_time_s"])
        for ep in episodes
        if ep.get("hazard_message_received_by_ego", False)
        and ep.get("reaction_time_s") is not None
    ]

    return {
        "episodes": total_episodes,
        "hazard_injected_episodes": injected_count,
        "hazard_received_episodes": received_count,
        "braking_signal_received_episodes": braking_signal_count,
        "reaction_episodes": reaction_count,
        "hazard_injection_rate": (
            injected_count / total_episodes if total_episodes > 0 else 0.0
        ),
        "hazard_reception_rate": (
            received_count / injected_count if injected_count > 0 else 0.0
        ),
        "braking_signal_reception_rate": (
            braking_signal_count / injected_count if injected_count > 0 else 0.0
        ),
        "overall_reaction_rate": (
            reaction_count / received_count if received_count > 0 else 0.0
        ),
        "avg_reaction_time_s": (
            float(np.mean(reaction_times)) if reaction_times else None
        ),
        "v2v_reaction_detected": reaction_count > 0,
        "peer_count_summary": peer_summary,
        "source_reaction_summary": source_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model and report V2V reaction verdict."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model .zip file.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory to evaluate.",
    )
    parser.add_argument(
        "--emulator_params",
        type=str,
        default="ml/espnow_emulator/emulator_params_measured.json",
        help="Path to emulator params JSON.",
    )
    parser.add_argument(
        "--episodes_per_scenario",
        type=int,
        default=1,
        help="Number of sequential eval passes per scenario (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--fail_if_no_reaction",
        action="store_true",
        help="Exit non-zero if no V2V reaction is detected.",
    )
    args = parser.parse_args()

    if args.episodes_per_scenario <= 0:
        parser.error("--episodes_per_scenario must be > 0")

    return args


def main() -> None:
    args = parse_args()

    eval_scenario_ids = load_eval_scenario_ids(args.dataset_dir)
    if not eval_scenario_ids:
        raise ValueError("Dataset eval_scenarios is empty; cannot evaluate.")
    total_episodes = len(eval_scenario_ids) * args.episodes_per_scenario

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path)
    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=None,
        dataset_dir=args.dataset_dir,
        scenario_mode="eval",
        emulator_params_path=args.emulator_params,
        hazard_injection=True,
        scenario_seed=args.seed,
    )

    print(f"\nRunning {total_episodes} smoke-eval episodes...\n")

    episode_details = []
    peer_count_cache: Dict[str, Optional[int]] = {}

    try:
        for ep in range(total_episodes):
            obs, info = env.reset(options={"hazard_force": True})
            scenario_id = info.get("scenario_id", "unknown")

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
                    if hazard_source_id is not None and hazard_source_id in mesh_ids:
                        hazard_message_received_by_ego = True

                    if bool(info.get("mesh_any_braking_peer_received", False)):
                        hazard_any_braking_peer_received = True

                    if (
                        not reaction_detected
                        and float(info.get("deceleration", 0.0)) > REACTION_DECEL_THRESHOLD
                    ):
                        reaction_detected = True
                        reaction_step = info.get("step")

            reaction_time_s = None
            if (
                reaction_detected
                and reaction_step is not None
                and hazard_step is not None
            ):
                reaction_time_s = max(
                    0.0, (int(reaction_step) - int(hazard_step)) * SIM_STEP_SECONDS
                )

            peer_count = infer_peer_count(args.dataset_dir, scenario_id, peer_count_cache)
            episode_details.append(
                {
                    "episode": ep + 1,
                    "scenario_id": scenario_id,
                    "peer_count": peer_count,
                    "reward": float(total_reward),
                    "length": steps,
                    "collision": bool(terminated),
                    "truncated": bool(truncated),
                    "pct_time_unsafe": (unsafe_steps / steps if steps > 0 else 0.0),
                    "hazard_injected": bool(hazard_injected),
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
                    "collision_post_hazard": bool(terminated and hazard_injected),
                    "pct_time_unsafe_post_hazard": (
                        unsafe_steps_post_hazard / post_hazard_steps
                        if post_hazard_steps > 0
                        else 0.0
                    ),
                }
            )

            print(
                f"  Episode {ep + 1:3d}/{total_episodes}: "
                f"scenario={scenario_id}, "
                f"peers={peer_count if peer_count is not None else 'unknown'}, "
                f"hazard_received={hazard_message_received_by_ego}, "
                f"reaction={reaction_detected}"
            )
    finally:
        env.close()

    peer_summary = _build_peer_reaction_summary(episode_details)
    source_summary = evaluate_model._build_source_reaction_summary(episode_details)
    verdict = _build_verdict(episode_details, peer_summary, source_summary)

    print("\n" + "=" * 60)
    print("SMOKE V2V REACTION CHECK")
    print("=" * 60)
    print(f"  Episodes:                  {verdict['episodes']}")
    print(
        "  Hazard Injected:           "
        f"{verdict['hazard_injected_episodes']}/{verdict['episodes']} "
        f"({verdict['hazard_injection_rate'] * 100:.1f}%)"
    )
    print(
        "  Hazard Source Received:    "
        f"{verdict['hazard_received_episodes']}/{verdict['hazard_injected_episodes']} "
        f"({verdict['hazard_reception_rate'] * 100:.1f}%)"
    )
    print(
        "  Braking Signal Received:   "
        f"{verdict['braking_signal_received_episodes']}/{verdict['hazard_injected_episodes']} "
        f"({verdict['braking_signal_reception_rate'] * 100:.1f}%)"
    )
    print(
        "  RL Reactions After Receipt:"
        f" {verdict['reaction_episodes']}/{verdict['hazard_received_episodes']} "
        f"({verdict['overall_reaction_rate'] * 100:.1f}%)"
    )
    if verdict["avg_reaction_time_s"] is not None:
        print(f"  Avg Reaction Time:         {verdict['avg_reaction_time_s']:.2f}s")
    else:
        print("  Avg Reaction Time:         n/a")
    print(
        "  V2V Reaction Detected:     "
        f"{'YES' if verdict['v2v_reaction_detected'] else 'NO'}"
    )
    print("  By Peer Count:")
    for peer_key, bucket in verdict["peer_count_summary"].items():
        print(
            f"    n={peer_key}: "
            f"received {bucket['hazard_received_episodes']}/{bucket['hazard_injected_episodes']}, "
            f"reacted {bucket['reaction_episodes']}/{bucket['hazard_received_episodes']} "
            f"({bucket['reaction_rate'] * 100:.1f}%)"
        )
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "model_path": args.model_path,
                    "dataset_dir": args.dataset_dir,
                    "episodes_per_scenario": args.episodes_per_scenario,
                    "seed": args.seed,
                    "verdict": verdict,
                    "episode_details": episode_details,
                },
                file_handle,
                indent=2,
            )
        print(f"\nResults saved to: {output_path}")

    if args.fail_if_no_reaction and not verdict["v2v_reaction_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
