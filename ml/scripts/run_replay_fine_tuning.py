#!/usr/bin/env python3
"""
Run 024 — Fine-tune a SUMO-trained model on real-data replay.

Usage:
    cd roadsense-v2v
    source ml/venv/bin/activate
    python -m ml.scripts.run_replay_fine_tuning \
        --model_path ml/models/runs/cloud_prod_023/model_final.zip \
        --vecnormalize_path ml/models/runs/cloud_prod_023/vecnormalize.pkl \
        --recordings_dir ml/data/recordings \
        --output_dir ml/results/run_024_replay \
        --total_timesteps 500000 \
        --learning_rate 1e-5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from ml.envs.replay_convoy_env import ReplayConvoyEnv


class MetricsCallback(BaseCallback):
    """Log fine-tuning metrics periodically."""

    def __init__(self, log_interval: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        # Collect episode info from monitor
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        if self.num_timesteps % self.log_interval == 0 and self._episode_rewards:
            mean_r = np.mean(self._episode_rewards[-100:])
            mean_l = np.mean(self._episode_lengths[-100:])
            print(
                f"  [{self.num_timesteps:>8d}] "
                f"ep_rew_mean={mean_r:+.1f}  "
                f"ep_len_mean={mean_l:.0f}  "
                f"episodes={len(self._episode_rewards)}"
            )
        return True


def make_linear_schedule(initial_value: float, final_value: float):
    """Linear schedule over SB3's progress_remaining in [1.0, 0.0]."""
    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return schedule


def build_replay_reward_config(
    ignore_hazard_threshold: float,
    ignore_danger_distance: float,
    ignore_danger_closing_rate: float,
    ignore_require_danger_geometry: bool,
) -> dict:
    """Build replay-only reward config for ReplayConvoyEnv."""
    return {
        "ignoring_hazard_threshold": ignore_hazard_threshold,
        "ignoring_danger_distance": ignore_danger_distance,
        "ignoring_danger_closing_rate": ignore_danger_closing_rate,
        "ignoring_require_danger_geometry": ignore_require_danger_geometry,
        "ignoring_use_any_braking_peer": True,
        "early_reaction_threshold": 0.01,
    }


def make_env(
    recordings_dir: str,
    augment: bool,
    seed: int,
    use_recorded_ego: bool = True,
    use_shadow_reward_geometry: bool = False,
    reward_config: dict | None = None,
    ego_stack_frames: int = 1,
):
    """Factory for ReplayConvoyEnv wrapped with Monitor."""
    def _init():
        env = ReplayConvoyEnv(
            recordings_dir=recordings_dir,
            augment=augment,
            max_steps=500,
            seed=seed,
            use_recorded_ego=use_recorded_ego,
            use_shadow_reward_geometry=use_shadow_reward_geometry,
            random_start=True,
            reward_config=reward_config,
            ego_stack_frames=ego_stack_frames,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Run 024: Replay Fine-Tuning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SUMO-trained model .zip")
    parser.add_argument("--vecnormalize_path", type=str, default=None,
                        help="Path to VecNormalize .pkl from SUMO training")
    parser.add_argument("--recordings_dir", type=str, required=True,
                        help="Path to recordings/ directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--final_learning_rate", type=float, default=None,
                        help="Optional final LR for linear decay. "
                             "If omitted, LR stays constant.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable augmentation (deterministic replay)")
    parser.add_argument("--use_kinematic_ego", action="store_true",
                        help="Use kinematic ego instead of recorded ego "
                             "(default: use recorded ego to match validation)")
    parser.add_argument("--use_shadow_reward_geometry", action="store_true",
                        help="In recorded-ego mode, drive reward geometry from "
                             "a shadow EgoKinematics trajectory while keeping "
                             "observations on the recorded ego.")
    parser.add_argument("--reset_log_std", type=float, default=None,
                        help="Reset policy log_std before fine-tuning "
                             "(e.g. -1.0 → std≈0.37). Needed to explore "
                             "when base model policy is too tight (std<0.1).")
    parser.add_argument("--ignore_hazard_threshold", type=float, default=0.15,
                        help="Replay-only ignoring-hazard gate on braking_received.")
    parser.add_argument("--ignore_danger_distance", type=float, default=20.0,
                        help="Replay-only ignoring-hazard geometry gate: penalty "
                             "fires when distance is below this threshold.")
    parser.add_argument("--ignore_danger_closing_rate", type=float, default=0.5,
                        help="Replay-only ignoring-hazard geometry gate: penalty "
                             "fires when closing_rate exceeds this threshold.")
    parser.add_argument("--ignore_require_danger_geometry", action="store_true",
                        help="Require danger geometry for replay ignoring penalty. "
                             "Default is OFF for recorded-ego fine-tuning.")
    parser.add_argument("--ego_stack_frames", type=int, default=1,
                        help="Number of ego observation frames to stack (Run 025). "
                             "Default 1 = no stacking. Use 3 for temporal context.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RUN 024 — REPLAY FINE-TUNING")
    print("=" * 70)
    print(f"  Model:        {args.model_path}")
    print(f"  VecNormalize: {args.vecnormalize_path}")
    print(f"  Recordings:   {args.recordings_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Timesteps:    {args.total_timesteps:,}")
    print(f"  LR:           {args.learning_rate}")
    print(f"  Final LR:     {args.final_learning_rate}")
    use_recorded_ego = not args.use_kinematic_ego
    use_shadow_reward_geometry = args.use_shadow_reward_geometry
    print(f"  Augment:      {not args.no_augment}")
    print(f"  Recorded ego: {use_recorded_ego}")
    print(f"  Shadow geom:  {use_shadow_reward_geometry}")
    print(f"  Random start: True")
    print(f"  Ignore thr:   {args.ignore_hazard_threshold}")
    print(f"  Danger dist:  {args.ignore_danger_distance}")
    print(f"  Danger rate:  {args.ignore_danger_closing_rate}")
    print(f"  Use geom:     {args.ignore_require_danger_geometry}")
    print(f"  Ego stack:    {args.ego_stack_frames}")
    print("=" * 70)

    reward_config = build_replay_reward_config(
        ignore_hazard_threshold=args.ignore_hazard_threshold,
        ignore_danger_distance=args.ignore_danger_distance,
        ignore_danger_closing_rate=args.ignore_danger_closing_rate,
        ignore_require_danger_geometry=args.ignore_require_danger_geometry,
    )

    # Create vectorized env (Monitor-wrapped for episode stats)
    vec_env = DummyVecEnv([
        make_env(
            args.recordings_dir, not args.no_augment, args.seed,
            use_recorded_ego=use_recorded_ego,
            use_shadow_reward_geometry=use_shadow_reward_geometry,
            reward_config=reward_config,
            ego_stack_frames=args.ego_stack_frames,
        )
    ])

    # Load VecNormalize if available
    if args.vecnormalize_path and Path(args.vecnormalize_path).exists():
        print(f"\nLoading VecNormalize from {args.vecnormalize_path}")
        vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print(f"  Reward running mean: {vec_env.ret_rms.mean:.2f}")
        print(f"  Reward running var:  {vec_env.ret_rms.var:.2f}")
    else:
        print("\nNo VecNormalize — using raw rewards")
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    # Load pre-trained model
    print(f"\nLoading model from {args.model_path}")
    model = PPO.load(args.model_path, env=vec_env)

    # Override tensorboard log to local path (cloud model has /work/... baked in)
    model.tensorboard_log = str(output_dir / "tensorboard")

    # Override learning rate for fine-tuning
    if args.final_learning_rate is None:
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
    else:
        model.learning_rate = args.learning_rate
        model.lr_schedule = make_linear_schedule(
            args.learning_rate,
            args.final_learning_rate,
        )
    print(f"  LR set to: {model.learning_rate}")
    print(f"  Policy std (before): {model.policy.log_std.exp().detach().numpy()}")

    # Optionally reset log_std to increase exploration.
    # SUMO model converges to std~0.06, which means P(action>0.1)~7%.
    # For recorded-ego fine-tuning the model must DISCOVER that braking
    # during hazard yields higher reward, so more exploration is essential.
    if args.reset_log_std is not None:
        import torch
        with torch.no_grad():
            model.policy.log_std.fill_(args.reset_log_std)
        print(f"  Policy std (reset): {model.policy.log_std.exp().detach().numpy()}")

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="replay_ft",
    )
    metrics_cb = MetricsCallback(log_interval=5000)

    # Fine-tune
    print(f"\nStarting fine-tuning for {args.total_timesteps:,} timesteps...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, metrics_cb],
        reset_num_timesteps=True,
        progress_bar=False,
    )
    elapsed = time.time() - t0

    # Save
    model_path = str(output_dir / "model_final.zip")
    model.save(model_path)
    vec_env.save(str(output_dir / "vecnormalize.pkl"))
    print(f"\nModel saved to {model_path}")
    print(f"Training time: {elapsed / 60:.1f} minutes")

    # Summary
    summary = {
        "run_id": "run_024_replay",
        "base_model": args.model_path,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "final_learning_rate": args.final_learning_rate,
        "augment": not args.no_augment,
        "use_recorded_ego": use_recorded_ego,
        "use_shadow_reward_geometry": use_shadow_reward_geometry,
        "random_start": True,
        "reset_log_std": args.reset_log_std,
        "reward_config": reward_config,
        "training_time_s": elapsed,
        "seed": args.seed,
        "ego_stack_frames": args.ego_stack_frames,
        "episodes_completed": len(metrics_cb._episode_rewards),
        "final_ep_rew_mean": float(np.mean(metrics_cb._episode_rewards[-100:]))
        if metrics_cb._episode_rewards else None,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
