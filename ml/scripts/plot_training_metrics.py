"""Plot Run 025 PPO training metrics from TensorBoard event files.

Produces a single 6-panel figure covering both training phases on a shared
timeline (Phase 1: SUMO 2M, Phase 2: replay fine-tune 500k).

Usage:
    cd roadsense-v2v
    source ml/venv/bin/activate
    python -m ml.scripts.plot_training_metrics
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]
SUMO_TB = REPO / "ml/models/runs/cloud_prod_025/tensorboard/PPO_1"
REPLAY_TB = REPO / "ml/results/run_025_replay_v1/tensorboard/PPO_1"
OUT_PATH = REPO / "ml/results/run_025_replay_v1/training_metrics.png"

PANELS = [
    ("rollout/ep_rew_mean", "Episode Reward (mean)", "reward"),
    ("train/explained_variance", "Explained Variance (value function)", "fraction"),
    ("train/std", "Policy Std (exploration)", "std"),
    ("train/value_loss", "Value Loss", "loss"),
    ("train/approx_kl", "Approx KL (PPO trust region)", "KL"),
    ("train/clip_fraction", "Clip Fraction", "fraction"),
]


def load_scalars(tb_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.float64)
        values = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, values)
    return out


def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    print(f"Loading SUMO TB: {SUMO_TB}")
    sumo = load_scalars(SUMO_TB)
    print(f"Loading replay TB: {REPLAY_TB}")
    replay = load_scalars(REPLAY_TB)

    sumo_max_step = max(s.max() for s, _ in sumo.values())
    print(f"Phase 1 (SUMO) max step: {sumo_max_step:,.0f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(
        "Run 025 — PPO Training Metrics\n"
        "Phase 1: SUMO simulation (2M steps)   |   Phase 2: Real-data replay fine-tuning (500k steps)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (tag, title, ylabel) in zip(axes.flat, PANELS):
        if tag not in sumo or tag not in replay:
            ax.set_title(f"{title}\n(missing tag)")
            continue

        s_steps, s_vals = sumo[tag]
        r_steps, r_vals = replay[tag]

        # Offset replay onto a shared timeline so the prof sees one continuous story.
        r_steps_shifted = r_steps + sumo_max_step

        # Raw values faintly, smoothed values bold.
        ax.plot(s_steps, s_vals, color="#1f77b4", alpha=0.25, linewidth=0.8)
        ax.plot(s_steps, smooth(s_vals), color="#1f77b4", linewidth=1.8, label="Phase 1: SUMO")
        ax.plot(r_steps_shifted, r_vals, color="#d62728", alpha=0.25, linewidth=0.8)
        ax.plot(
            r_steps_shifted,
            smooth(r_vals),
            color="#d62728",
            linewidth=1.8,
            label="Phase 2: Replay",
        )

        ax.axvline(sumo_max_step, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    for ax in axes[-1]:
        ax.set_xlabel("Training Timesteps")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k")
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
