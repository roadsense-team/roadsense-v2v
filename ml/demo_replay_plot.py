#!/usr/bin/env python3
"""
Real-Data Replay Visualization — PoC Presentation Plot
=======================================================

Generates a publication-quality multi-panel timeline showing the Run 025
model's reactions on real convoy recording data (Recording #2).

Highlights:
- 3/3 real braking events detected (100%)
- Model reacts 2-4 seconds BEFORE the ego driver brakes (V2V early warning)
- 11.5% false positive rate on calm driving segments

This script loads pre-computed validation timeseries (NPZ) and report (JSON)
from the validation output — no model inference needed.

Usage:
    cd roadsense-v2v
    source ml/venv/bin/activate

    # Default: uses Run 025 500k fixed validation data
    python -m ml.demo_replay_plot

    # Custom paths:
    python -m ml.demo_replay_plot \
        --npz ml/results/run_025_replay_v1/validation/rec02_500000_fixed/timeseries_recording_02_fixed.npz \
        --report ml/results/run_025_replay_v1/validation/rec02_500000_fixed/validation_report_recording_02_fixed.json \
        --output demo_poc_plot.png

    # Show interactively (no save):
    python -m ml.demo_replay_plot --show
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

# Default paths (Run 025, 500k checkpoint, fixed validator)
DEFAULT_NPZ = (
    "ml/results/run_025_replay_v1/validation/rec02_500000_fixed"
    "/timeseries_recording_02_fixed.npz"
)
DEFAULT_REPORT = (
    "ml/results/run_025_replay_v1/validation/rec02_500000_fixed"
    "/validation_report_recording_02_fixed.json"
)

MAX_DECEL = 8.0  # m/s²
MODEL_ACTION_THRESHOLD = 0.1


def load_data(npz_path: Path, report_path: Path):
    """Load timeseries and validation report."""
    data = np.load(npz_path)
    with open(report_path) as f:
        report = json.load(f)
    return data, report


def make_plot(data, report, output_path: Path = None, show: bool = False):
    """Generate a focused PoC presentation plot.

    Layout: 3 side-by-side event panels, each zoomed to ±8s around one
    braking event.  Both driver deceleration and model response are overlaid
    so the early-warning gap is immediately visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Extract timeseries
    timestamps_ms = data["timestamps_ms"]
    ego_accel = data["ego_accel"]
    model_action = data["model_action"]

    t0 = timestamps_ms[0]
    t_sec = (timestamps_ms - t0) / 1000.0
    model_decel = model_action * MAX_DECEL

    events = report["braking_events_detail"]

    # Colors
    C_DRIVER = "#DC2626"     # red  — driver braking
    C_MODEL = "#2563EB"      # blue — model output
    C_EVENT = "#FEE2E2"      # light red — event background
    C_EARLY = "#DCFCE7"      # light green — early warning background

    event_titles = [
        "Event 1 — Medium Brake",
        "Event 2 — Hard Brake",
        "Event 3 — Emergency Stop",
    ]
    WINDOW_S = 8  # seconds before/after event start to show

    # ---- Figure: 3 columns ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    fig.suptitle(
        "RoadSense V2V Early Warning — Real Convoy Data",
        fontsize=16, fontweight="bold", y=1.02,
    )

    for i, (evt, ax) in enumerate(zip(events, axes)):
        evt_start_s = (evt["start_ms"] - t0) / 1000.0
        evt_end_s = (evt["end_ms"] - t0) / 1000.0

        reaction_ms = evt["reaction_time_ms"]
        early_s = abs(reaction_ms / 1000) if reaction_ms is not None and reaction_ms < 0 else 0
        model_react_s = evt_start_s + reaction_ms / 1000.0 if reaction_ms is not None else evt_start_s

        # Window bounds
        win_lo = evt_start_s - WINDOW_S
        win_hi = evt_start_s + WINDOW_S
        mask = (t_sec >= win_lo) & (t_sec <= win_hi)
        t_win = t_sec[mask]

        # ---- Background shading ----
        # Green: model reacted early (before driver braking)
        if reaction_ms is not None and reaction_ms < 0:
            ax.axvspan(model_react_s, evt_start_s, alpha=0.30, color=C_EARLY, zorder=0)
        # Red: actual driver braking
        ax.axvspan(evt_start_s, min(evt_end_s, win_hi), alpha=0.25, color=C_EVENT, zorder=0)

        # ---- Driver deceleration (inverted: positive = braking) ----
        driver_decel = np.clip(-ego_accel[mask], 0, None)
        ax.fill_between(t_win, driver_decel, alpha=0.25, color=C_DRIVER)
        ax.plot(t_win, driver_decel, color=C_DRIVER, linewidth=1.5, label="Driver braking")

        # ---- Model deceleration (raw, no smoothing — truth) ----
        mdl_win = model_decel[mask]
        ax.plot(t_win, mdl_win, color=C_MODEL, linewidth=2.0, label="Model response")

        # ---- Vertical marker lines ----
        ax.axvline(evt_start_s, color=C_DRIVER, linewidth=1.2, linestyle="--", alpha=0.7)
        if reaction_ms is not None and reaction_ms < 0:
            ax.axvline(model_react_s, color=C_MODEL, linewidth=1.2, linestyle="--", alpha=0.7)

        # ---- Early warning arrow ----
        if early_s > 0:
            arrow_y = 9.2
            ax.annotate(
                "", xy=(evt_start_s, arrow_y), xytext=(model_react_s, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#15803D", lw=2.0),
            )
            ax.text(
                (model_react_s + evt_start_s) / 2, arrow_y + 0.3,
                f"{early_s:.1f}s early",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#15803D",
            )

        # ---- Event info box ----
        peak_decel = abs(evt["real_min_accel"])
        model_peak = evt["max_model_action"] * MAX_DECEL
        info = (
            f"Driver peak: {peak_decel:.1f} m/s²\n"
            f"Model peak:  {model_peak:.1f} m/s²"
        )
        ax.text(
            0.97, 0.03, info,
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )

        # ---- Formatting ----
        ax.set_title(event_titles[i], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_xlim(win_lo, win_hi)
        ax.set_ylim(-0.5, 11)
        ax.grid(axis="both", alpha=0.2)

        # X-axis: show absolute times
        tick_start = int(win_lo)
        ax.set_xticks(range(tick_start, int(win_hi) + 1, 2))

    axes[0].set_ylabel("Deceleration (m/s²)", fontsize=12)

    # ---- Shared legend ----
    legend_elements = [
        Line2D([0], [0], color=C_DRIVER, linewidth=2, label="Driver braking (IMU)"),
        Line2D([0], [0], color=C_MODEL, linewidth=2, label="Model response (V2V)"),
        Patch(facecolor=C_EARLY, alpha=0.30, edgecolor="#15803D", label="Early warning window"),
        Patch(facecolor=C_EVENT, alpha=0.25, edgecolor=C_DRIVER, label="Driver braking period"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=4, fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # ---- Summary text ----
    sensitivity = report["sensitivity"]
    specificity = report["specificity"]
    summary = (
        f"Detection: {sensitivity['events_detected']}/{sensitivity['events_total']} "
        f"({sensitivity['detection_rate'] * 100:.0f}%)    |    "
        f"False Positive Rate: {specificity['false_positive_rate'] * 100:.1f}%    |    "
        f"Recording: {report['duration_s']:.0f}s real convoy driving"
    )
    fig.text(
        0.5, 0.94, summary,
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FDF4",
                  edgecolor="#16A34A", alpha=0.95),
    )

    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.08)

    if output_path:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        print(f"Plot saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(output_path) if output_path else None


def main():
    parser = argparse.ArgumentParser(
        description="RoadSense V2V PoC — Real-Data Replay Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--npz", type=Path, default=None,
        help=f"Path to timeseries NPZ file (default: {DEFAULT_NPZ})",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help=f"Path to validation report JSON (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("demo_poc_recording02.png"),
        help="Output PNG path (default: demo_poc_recording02.png)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show interactive matplotlib window instead of saving",
    )
    args = parser.parse_args()

    # Resolve defaults relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    npz_path = args.npz or (repo_root / DEFAULT_NPZ)
    report_path = args.report or (repo_root / DEFAULT_REPORT)

    for p, label in [(npz_path, "NPZ timeseries"), (report_path, "JSON report")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # Use non-interactive backend if not showing
    if not args.show:
        matplotlib.use("Agg")

    print(f"Loading timeseries: {npz_path}")
    print(f"Loading report:     {report_path}")
    data, report = load_data(npz_path, report_path)

    print(f"Recording: {report['recording']}")
    print(f"  Duration: {report['duration_s']:.1f}s, {report['total_steps']} steps")
    print(f"  Events: {report['sensitivity']['events_detected']}/{report['sensitivity']['events_total']} detected")
    print(f"  FP Rate: {report['specificity']['false_positive_rate'] * 100:.1f}%")
    print()

    output = args.output if not args.show else None
    make_plot(data, report, output_path=output, show=args.show)

    if not args.show:
        print("\nDone. Use this plot in your PoC presentation slides.")


if __name__ == "__main__":
    main()
