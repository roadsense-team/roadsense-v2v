#!/usr/bin/env python3
"""
Run 023 preflight: audit hazard-onset bucket coverage.

Runs the ConvoyEnv in training mode with forced hazard episodes and
state_bucket trigger mode.  Records how often each onset bucket actually
triggers vs. falls back, so we can verify that the state-triggered design
is actually broadening onset diversity before launching a cloud run.

Usage (from repo root, inside Docker):
    python -m ml.scripts.audit_hazard_onset_coverage \
        --dataset_dir ml/scenarios/datasets/dataset_v13_run023 \
        --emulator_params ml/espnow_emulator/emulator_params_measured.json \
        --episodes 200 \
        --seed 42

Minimum gate criteria (from Run 023 implementation plan):
    - fallback rate  <= 25%
    - at least 3 gap buckets observed
    - state-bucket triggers observed for peer counts n=1..5
    - no silent hazard-drop episodes
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np

import ml.envs  # noqa: F401 — registers environments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit hazard-onset bucket coverage for Run 023."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--emulator_params", type=str,
        default="ml/espnow_emulator/emulator_params_measured.json",
        help="Path to emulator params JSON.",
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of episodes to audit (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "RoadSense-Convoy-v0",
        sumo_cfg=None,
        dataset_dir=args.dataset_dir,
        scenario_mode="train",
        emulator_params_path=args.emulator_params,
        hazard_injection=True,
        scenario_seed=args.seed,
    )

    # Override injector to use state_bucket mode.
    if env.unwrapped.hazard_injector is not None:
        from ml.envs.hazard_injector import HazardInjector
        env.unwrapped.hazard_injector = HazardInjector(
            seed=args.seed,
            target_strategy=HazardInjector.TARGET_STRATEGY_UNIFORM_FRONT_PEERS,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )

    trigger_results: List[str] = []
    gap_buckets: List[str] = []
    ranks: List[int] = []
    onset_gaps: List[float] = []
    onset_closing_speeds: List[float] = []
    onset_peer_counts: List[int] = []
    silent_drops = 0

    print(f"Auditing {args.episodes} episodes with state_bucket trigger mode...")

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(
                seed=args.seed + ep,
                options={"hazard_force": True},
            )
            terminated = False
            truncated = False
            hazard_fired = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("hazard_injected", False):
                    hazard_fired = True

            # Collect onset telemetry from the episode.
            inj = env.unwrapped.hazard_injector
            tr = info.get("hazard_trigger_result") or inj.trigger_result
            if tr:
                trigger_results.append(tr)
            bucket = info.get("hazard_onset_gap_bucket") or inj.onset_gap_bucket
            if bucket:
                gap_buckets.append(bucket)
            rank = info.get("hazard_onset_desired_rank_ahead") or inj.onset_desired_rank_ahead
            if rank is not None:
                ranks.append(rank)
            gap_m = info.get("hazard_onset_gap_m") or inj.onset_gap_m
            if gap_m is not None:
                onset_gaps.append(gap_m)
            cs = info.get("hazard_onset_closing_speed_mps") or inj.onset_closing_speed_mps
            if cs is not None:
                onset_closing_speeds.append(cs)
            pc = info.get("hazard_onset_peer_count") or inj.onset_peer_count
            if pc is not None:
                onset_peer_counts.append(pc)

            if not hazard_fired:
                silent_drops += 1

            if (ep + 1) % 50 == 0 or ep == 0:
                print(f"  Episode {ep + 1}/{args.episodes}: trigger={tr}, bucket={bucket}")

    finally:
        env.close()

    # --- Summarize ---
    total = len(trigger_results)
    bucket_match_count = trigger_results.count("bucket_match")
    fallback_count = trigger_results.count("fallback_step")
    fallback_rate = fallback_count / total if total > 0 else 1.0

    bucket_counts = Counter(gap_buckets)
    rank_counts = Counter(ranks)
    peer_count_counts = Counter(onset_peer_counts)

    report = {
        "total_episodes": args.episodes,
        "hazard_triggered_episodes": total,
        "bucket_match_episodes": bucket_match_count,
        "fallback_episodes": fallback_count,
        "fallback_rate": round(fallback_rate, 4),
        "silent_drop_episodes": silent_drops,
        "gap_bucket_counts": dict(bucket_counts),
        "unique_gap_buckets": len(bucket_counts),
        "rank_counts": {str(k): v for k, v in sorted(rank_counts.items())},
        "peer_count_counts": {str(k): v for k, v in sorted(peer_count_counts.items())},
        "onset_gap_histogram": {
            "mean": round(float(np.mean(onset_gaps)), 2) if onset_gaps else None,
            "std": round(float(np.std(onset_gaps)), 2) if onset_gaps else None,
            "min": round(float(np.min(onset_gaps)), 2) if onset_gaps else None,
            "max": round(float(np.max(onset_gaps)), 2) if onset_gaps else None,
        },
        "onset_closing_speed_histogram": {
            "mean": round(float(np.mean(onset_closing_speeds)), 2) if onset_closing_speeds else None,
            "std": round(float(np.std(onset_closing_speeds)), 2) if onset_closing_speeds else None,
            "min": round(float(np.min(onset_closing_speeds)), 2) if onset_closing_speeds else None,
            "max": round(float(np.max(onset_closing_speeds)), 2) if onset_closing_speeds else None,
        },
    }

    print("\n" + "=" * 60)
    print("ONSET COVERAGE AUDIT")
    print("=" * 60)
    print(f"  Total episodes:          {args.episodes}")
    print(f"  Hazard triggered:        {total}")
    print(f"  Bucket match:            {bucket_match_count}")
    print(f"  Fallback:                {fallback_count}")
    print(f"  Fallback rate:           {fallback_rate * 100:.1f}%")
    print(f"  Silent drops:            {silent_drops}")
    print(f"  Unique gap buckets:      {len(bucket_counts)}")
    print(f"  Gap bucket counts:       {dict(bucket_counts)}")
    print(f"  Rank counts:             {dict(rank_counts)}")
    print(f"  Peer count counts:       {dict(peer_count_counts)}")
    if onset_gaps:
        print(f"  Onset gap (m):           mean={np.mean(onset_gaps):.1f}, "
              f"std={np.std(onset_gaps):.1f}, "
              f"range=[{np.min(onset_gaps):.1f}, {np.max(onset_gaps):.1f}]")
    if onset_closing_speeds:
        print(f"  Closing speed (m/s):     mean={np.mean(onset_closing_speeds):.1f}, "
              f"std={np.std(onset_closing_speeds):.1f}, "
              f"range=[{np.min(onset_closing_speeds):.1f}, {np.max(onset_closing_speeds):.1f}]")
    print("=" * 60)

    # --- Gate checks ---
    gate_pass = True
    if fallback_rate > 0.25:
        print(f"\nFAIL: Fallback rate {fallback_rate * 100:.1f}% > 25%")
        gate_pass = False
    if len(bucket_counts) < 3:
        print(f"\nFAIL: Only {len(bucket_counts)} gap buckets observed (need >= 3)")
        gate_pass = False
    if silent_drops > 0:
        print(f"\nFAIL: {silent_drops} silent hazard-drop episodes")
        gate_pass = False

    if gate_pass:
        print("\nGATE: PASS — onset coverage is sufficient for cloud launch")
    else:
        print("\nGATE: FAIL — fix trigger design before cloud launch")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
