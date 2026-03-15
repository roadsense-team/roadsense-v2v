#!/usr/bin/env python3
"""
H5 Sim-to-Real Validation — Offline Replay of Real Convoy Data Through Trained Model.

Feeds real V2V recording data (TX + RX CSVs) through the trained RL model's
observation pipeline and records what actions the model would have taken.
Compares against the real driver's behavior to validate sim-to-real transfer.

Usage:
    cd roadsense-v2v
    source ml/venv/bin/activate
    python -m ml.scripts.validate_against_real_data \
        --tx_csv /path/to/V001_tx_004.csv \
        --rx_csv /path/to/V001_rx_004.csv \
        --model_path ml/results/cloud_prod_019/model_final.zip \
        --output_dir ml/data/sim_to_real_validation \
        --forward_axis y

    # Replay-side braking signal synthesis (Run 020):
    #   --braking_received_mode decay     # default — exponential decay (0.95/step)
    #   --braking_received_mode latched   # legacy sticky latch (regression analysis)
    #   --braking_received_mode instant   # per-step only, no memory
    #   --braking_received_mode off       # force ego[4] = 0 throughout replay
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so SB3 can resolve custom extractors
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.envs.observation_builder import ObservationBuilder
from ml.envs.sumo_connection import VehicleState

# ---------------------------------------------------------------------------
# Constants (must match training pipeline)
# ---------------------------------------------------------------------------
METERS_PER_DEG_LAT = 111_000.0  # Same as ESPNOWEmulator.METERS_PER_DEG_LAT
STEP_MS = 100  # 10 Hz replay rate (matches sensor broadcast rate)
STALENESS_THRESHOLD_MS = 500.0
BRAKING_THRESHOLD_MS2 = -1.0  # accel below this = braking event (analysis)
V2V_BRAKING_ACCEL_THRESHOLD = -2.5  # matches ConvoyEnv.BRAKING_ACCEL_THRESHOLD
BRAKING_DECAY = 0.95  # matches ConvoyEnv.BRAKING_DECAY
MODEL_ACTION_THRESHOLD = 0.1  # model action above this = model brakes
MAX_DECEL = 8.0  # m/s², matches training

# Current ego observation layout (Run 022+):
# [speed/30, accel/10, peer_count/8, min_peer_accel/10, braking_received]
EGO_SPEED_IDX = 0
EGO_ACCEL_IDX = 1
EGO_PEER_COUNT_IDX = 2
EGO_MIN_PEER_ACCEL_IDX = 3
EGO_BRAKING_RECEIVED_IDX = 4


@dataclass
class SensorRow:
    """One row from a TX or RX CSV."""
    timestamp_local_ms: int
    msg_timestamp: int
    vehicle_id: str
    lat: float
    lon: float
    speed: float  # m/s
    heading: float  # degrees, 0=North clockwise
    accel_fwd: float  # forward-axis acceleration (m/s²)
    hop_count: int


def parse_csv(path: Path, forward_axis: str = "y") -> List[SensorRow]:
    """Parse a TX or RX CSV into SensorRow list, sorted by timestamp_local_ms."""
    rows: List[SensorRow] = []
    accel_col = "accel_y" if forward_axis == "y" else "accel_x"

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = r.get("vehicle_id") or r.get("from_vehicle_id", "")
            rows.append(SensorRow(
                timestamp_local_ms=int(r["timestamp_local_ms"]),
                msg_timestamp=int(r["msg_timestamp"]),
                vehicle_id=vid.strip(),
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                speed=float(r["speed"]),
                heading=float(r["heading"]),
                accel_fwd=float(r[accel_col]),
                hop_count=int(r.get("hop_count", 0)),
            ))

    rows.sort(key=lambda r: r.timestamp_local_ms)
    return rows


def gps_to_meters(lat: float, lon: float) -> Tuple[float, float]:
    """Convert GPS lat/lon to meters (x=East, y=North) using training constant."""
    x = lon * METERS_PER_DEG_LAT
    y = lat * METERS_PER_DEG_LAT
    return x, y


def build_ego_state(row: SensorRow) -> Tuple[VehicleState, Tuple[float, float]]:
    """Create VehicleState + position from a TX CSV row."""
    x, y = gps_to_meters(row.lat, row.lon)
    state = VehicleState(
        vehicle_id=row.vehicle_id,
        x=x,
        y=y,
        speed=max(0.0, row.speed),
        acceleration=row.accel_fwd,
        heading=row.heading,
        lane_position=0.0,
    )
    return state, (x, y)


def build_peer_observations(
    rx_rows: List[SensorRow],
    current_time_ms: int,
) -> List[Dict[str, float]]:
    """
    Build peer observation dicts from RX rows within the staleness window.

    For each source vehicle, keep only the freshest message.
    """
    # Group by vehicle, keep freshest
    freshest: Dict[str, SensorRow] = {}
    for r in rx_rows:
        age = current_time_ms - r.timestamp_local_ms
        if age < 0 or age > STALENESS_THRESHOLD_MS:
            continue
        prev = freshest.get(r.vehicle_id)
        if prev is None or r.timestamp_local_ms > prev.timestamp_local_ms:
            freshest[r.vehicle_id] = r

    peers = []
    for vid, r in freshest.items():
        x, y = gps_to_meters(r.lat, r.lon)
        age_ms = float(current_time_ms - r.timestamp_local_ms)
        peers.append({
            "x": x,
            "y": y,
            "speed": max(0.0, r.speed),
            "heading": r.heading,
            "accel": r.accel_fwd,
            "age_ms": max(0.0, age_ms),
            "valid": True,
        })
    return peers


def detect_braking_events(
    tx_rows: List[SensorRow],
    threshold: float = BRAKING_THRESHOLD_MS2,
    min_duration_ms: int = 300,
) -> List[Dict]:
    """
    Detect contiguous braking events in ego TX data.

    Returns list of {start_ms, end_ms, duration_ms, min_accel, avg_accel}.
    """
    events = []
    in_event = False
    start_ms = 0
    accels: List[float] = []

    for r in tx_rows:
        if r.accel_fwd < threshold:
            if not in_event:
                in_event = True
                start_ms = r.timestamp_local_ms
                accels = []
            accels.append(r.accel_fwd)
        else:
            if in_event:
                end_ms = r.timestamp_local_ms
                duration = end_ms - start_ms
                if duration >= min_duration_ms:
                    events.append({
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": duration,
                        "min_accel": float(min(accels)),
                        "avg_accel": float(np.mean(accels)),
                    })
                in_event = False

    # Close trailing event
    if in_event and tx_rows:
        end_ms = tx_rows[-1].timestamp_local_ms
        duration = end_ms - start_ms
        if duration >= min_duration_ms:
            events.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration,
                "min_accel": float(min(accels)),
                "avg_accel": float(np.mean(accels)),
            })

    return events


def _update_braking_received_decay(
    any_braking_this_step: bool,
    current_decay: float,
    mode: str,
) -> float:
    """
    Update replay-side braking_received signal.

    Modes:
      - decay: exponential decay (default, deployment-canonical)
      - latched: sticky forever (legacy, for regression analysis)
      - instant: true only on steps with current braking evidence
      - off: always 0.0

    Returns:
        Updated braking_received value in [0.0, 1.0].
    """
    if mode == "off":
        return 0.0

    if mode == "instant":
        return 1.0 if any_braking_this_step else 0.0

    if mode == "latched":
        if any_braking_this_step:
            return 1.0
        return current_decay  # stays at whatever it was (sticky)

    if mode != "decay":
        raise ValueError(f"Unknown braking_received_mode: {mode}")

    # Decay mode: reset to 1.0 on trigger, else multiply by decay factor
    if any_braking_this_step:
        return 1.0
    return current_decay * BRAKING_DECAY


def run_replay(
    tx_rows: List[SensorRow],
    rx_rows: List[SensorRow],
    model,
    output_dir: Path,
    recording_name: str,
    braking_received_mode: str = "decay",
) -> Dict:
    """
    Replay real data through the model and generate validation report.
    """
    obs_builder = ObservationBuilder()

    # Build time range
    t_start = tx_rows[0].timestamp_local_ms
    t_end = tx_rows[-1].timestamp_local_ms
    steps = list(range(t_start, t_end + 1, STEP_MS))

    # Index TX rows by timestamp for O(1) lookup
    tx_by_time: Dict[int, SensorRow] = {}
    for r in tx_rows:
        # Snap to nearest step
        snapped = round(r.timestamp_local_ms / STEP_MS) * STEP_MS
        if snapped not in tx_by_time or abs(r.timestamp_local_ms - snapped) < abs(tx_by_time[snapped].timestamp_local_ms - snapped):
            tx_by_time[snapped] = r

    # Index RX rows sorted by time for efficient windowing
    rx_sorted = sorted(rx_rows, key=lambda r: r.timestamp_local_ms)

    # Results storage
    results = {
        "timestamps_ms": [],
        "ego_speed": [],
        "ego_accel": [],
        "ego_heading": [],
        "model_action": [],
        "model_decel_ms2": [],
        "peer_count": [],
        "min_peer_accel": [],
        "braking_received": [],
        "any_braking_peer": [],
    }

    rx_window_start = 0
    braking_received_value = 0.0

    total_replay_steps = len(steps)
    print(f"Replaying {total_replay_steps} steps ({(t_end - t_start) / 1000:.1f}s) ...")

    for step_idx, step_t in enumerate(steps):
        # Find ego state: use latest TX row <= step_t
        ego_row = tx_by_time.get(step_t)
        if ego_row is None:
            # Find nearest TX row before this step
            for r in tx_rows:
                if r.timestamp_local_ms <= step_t:
                    ego_row = r
                else:
                    break
        if ego_row is None:
            continue

        ego_state, ego_pos = build_ego_state(ego_row)

        # Advance RX window: collect messages in [step_t - staleness, step_t]
        while rx_window_start < len(rx_sorted) and rx_sorted[rx_window_start].timestamp_local_ms < step_t - STALENESS_THRESHOLD_MS:
            rx_window_start += 1

        rx_window = []
        for i in range(rx_window_start, len(rx_sorted)):
            r = rx_sorted[i]
            if r.timestamp_local_ms > step_t:
                break
            rx_window.append(r)

        peer_obs = build_peer_observations(rx_window, step_t)
        cone_filtered_peers = obs_builder.filter_observable_peers(
            ego_heading_deg=ego_state.heading,
            ego_pos=ego_pos,
            peer_observations=peer_obs,
        )
        any_braking_this_step = any(
            p.get("accel", 0.0) <= V2V_BRAKING_ACCEL_THRESHOLD
            for p in cone_filtered_peers
        )
        braking_received_value = _update_braking_received_decay(
            any_braking_this_step=any_braking_this_step,
            current_decay=braking_received_value,
            mode=braking_received_mode,
        )

        observation = obs_builder.build(
            ego_state=ego_state,
            peer_observations=peer_obs,
            ego_pos=ego_pos,
            braking_received=braking_received_value,
        )

        # Model inference
        action, _ = model.predict(observation, deterministic=True)
        action_val = float(action.item()) if hasattr(action, "item") else float(action)
        decel_ms2 = action_val * MAX_DECEL

        # Count visible peers from observation
        visible_peers = int(np.sum(observation["peer_mask"]))
        min_pa = float(observation["ego"][EGO_MIN_PEER_ACCEL_IDX]) * obs_builder.MAX_ACCEL
        braking_received_obs = float(observation["ego"][EGO_BRAKING_RECEIVED_IDX])

        results["timestamps_ms"].append(step_t)
        results["ego_speed"].append(float(ego_row.speed))
        results["ego_accel"].append(float(ego_row.accel_fwd))
        results["ego_heading"].append(float(ego_row.heading))
        results["model_action"].append(action_val)
        results["model_decel_ms2"].append(decel_ms2)
        results["peer_count"].append(visible_peers)
        results["min_peer_accel"].append(min_pa)
        results["braking_received"].append(braking_received_obs)
        results["any_braking_peer"].append(any_braking_this_step)

    print(f"  Completed: {len(results['timestamps_ms'])} steps with model output")

    # ------ Analysis ------
    timestamps = np.array(results["timestamps_ms"])
    ego_accel = np.array(results["ego_accel"])
    model_action = np.array(results["model_action"])
    model_decel = np.array(results["model_decel_ms2"])
    peer_count = np.array(results["peer_count"])

    # Detect real braking events
    braking_events = detect_braking_events(tx_rows)

    # Model braking episodes (action > threshold)
    model_braking_mask = model_action > MODEL_ACTION_THRESHOLD

    # Time with peers visible
    has_peers_mask = peer_count > 0
    pct_time_with_peers = float(np.mean(has_peers_mask)) * 100

    # Braking correlation: during real braking events, does model brake?
    event_analysis = []
    for evt in braking_events:
        mask = (timestamps >= evt["start_ms"]) & (timestamps <= evt["end_ms"])
        if np.sum(mask) == 0:
            continue

        evt_actions = model_action[mask]
        evt_peers = peer_count[mask]
        model_reacted = bool(np.any(evt_actions > MODEL_ACTION_THRESHOLD))
        max_model_action = float(np.max(evt_actions))
        max_model_decel = max_model_action * MAX_DECEL

        # Reaction time: time from event start to first model action > threshold
        reaction_time_ms = None
        if model_reacted:
            evt_timestamps = timestamps[mask]
            for t, a in zip(evt_timestamps, evt_actions):
                if a > MODEL_ACTION_THRESHOLD:
                    reaction_time_ms = int(t - evt["start_ms"])
                    break

        event_analysis.append({
            "start_ms": evt["start_ms"],
            "end_ms": evt["end_ms"],
            "duration_ms": evt["duration_ms"],
            "real_min_accel": evt["min_accel"],
            "real_avg_accel": evt["avg_accel"],
            "model_reacted": model_reacted,
            "max_model_action": max_model_action,
            "max_model_decel_ms2": max_model_decel,
            "reaction_time_ms": reaction_time_ms,
            "avg_peers_visible": float(np.mean(evt_peers)),
        })

    # False positive analysis: model brakes when real driver didn't
    # (outside braking events, with peers visible)
    outside_events_mask = np.ones(len(timestamps), dtype=bool)
    for evt in braking_events:
        outside_events_mask &= ~((timestamps >= evt["start_ms"]) & (timestamps <= evt["end_ms"]))

    calm_with_peers = outside_events_mask & has_peers_mask
    if np.sum(calm_with_peers) > 0:
        false_positive_rate = float(np.mean(model_action[calm_with_peers] > MODEL_ACTION_THRESHOLD))
        avg_calm_action = float(np.mean(model_action[calm_with_peers]))
        max_calm_action = float(np.max(model_action[calm_with_peers]))
    else:
        false_positive_rate = float("nan")
        avg_calm_action = float("nan")
        max_calm_action = float("nan")

    # Overall action distribution
    action_hist, bin_edges = np.histogram(model_action, bins=10, range=(0, 1))

    # Build report
    report = {
        "recording": recording_name,
        "duration_s": float((t_end - t_start) / 1000),
        "total_steps": len(results["timestamps_ms"]),
        "step_interval_ms": STEP_MS,
        "pct_time_with_peers": round(pct_time_with_peers, 1),
        "real_braking_events": len(braking_events),
        "braking_events_detail": event_analysis,
        "sensitivity": {
            "events_detected": sum(1 for e in event_analysis if e["model_reacted"]),
            "events_total": len(event_analysis),
            "detection_rate": (
                float(sum(1 for e in event_analysis if e["model_reacted"]) / len(event_analysis))
                if event_analysis else float("nan")
            ),
        },
        "specificity": {
            "false_positive_rate": round(false_positive_rate, 4),
            "avg_calm_action": round(avg_calm_action, 4),
            "max_calm_action": round(max_calm_action, 4),
            "calm_steps_with_peers": int(np.sum(calm_with_peers)),
        },
        "action_distribution": {
            "histogram_counts": action_hist.tolist(),
            "bin_edges": [round(float(e), 1) for e in bin_edges],
            "mean": round(float(np.mean(model_action)), 4),
            "std": round(float(np.std(model_action)), 4),
            "median": round(float(np.median(model_action)), 4),
            "pct_above_threshold": round(float(np.mean(model_braking_mask)) * 100, 1),
        },
        "model_action_threshold": MODEL_ACTION_THRESHOLD,
        "max_decel_ms2": MAX_DECEL,
        "braking_received_mode": braking_received_mode,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"validation_report_{recording_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    # Save raw timeseries for plotting
    timeseries_path = output_dir / f"timeseries_{recording_name}.npz"
    np.savez_compressed(
        timeseries_path,
        timestamps_ms=timestamps,
        ego_speed=np.array(results["ego_speed"]),
        ego_accel=ego_accel,
        ego_heading=np.array(results["ego_heading"]),
        model_action=model_action,
        model_decel_ms2=model_decel,
        peer_count=peer_count,
        min_peer_accel=np.array(results["min_peer_accel"]),
        braking_received=np.array(results["braking_received"], dtype=np.float32),
        any_braking_peer=np.array(results["any_braking_peer"], dtype=bool),
    )
    print(f"  Timeseries saved: {timeseries_path}")

    return report


def print_report(report: Dict) -> None:
    """Print human-readable summary of validation results."""
    print()
    print("=" * 72)
    print(f"  SIM-TO-REAL VALIDATION: {report['recording']}")
    print("=" * 72)
    print(f"  Duration:        {report['duration_s']:.1f}s ({report['total_steps']} steps)")
    print(f"  Peers visible:   {report['pct_time_with_peers']:.1f}% of time")
    print(f"  Braking signal:  {report['braking_received_mode']}")
    print()

    # Sensitivity
    s = report["sensitivity"]
    print(f"  SENSITIVITY (does model brake when real driver braked?)")
    print(f"    Braking events detected:  {s['events_detected']}/{s['events_total']}")
    if not math.isnan(s["detection_rate"]):
        print(f"    Detection rate:           {s['detection_rate'] * 100:.1f}%")
    print()

    for i, evt in enumerate(report["braking_events_detail"]):
        marker = "DETECTED" if evt["model_reacted"] else "MISSED"
        rt = f"{evt['reaction_time_ms']}ms" if evt["reaction_time_ms"] is not None else "N/A"
        print(f"    Event {i + 1}: [{marker}] real_accel={evt['real_min_accel']:.2f} m/s², "
              f"model_max_action={evt['max_model_action']:.3f} ({evt['max_model_decel_ms2']:.1f} m/s²), "
              f"reaction={rt}, peers={evt['avg_peers_visible']:.1f}")
    print()

    # Specificity
    sp = report["specificity"]
    print(f"  SPECIFICITY (does model avoid braking during calm driving?)")
    print(f"    False positive rate:      {sp['false_positive_rate'] * 100:.2f}%")
    print(f"    Avg calm action:          {sp['avg_calm_action']:.4f}")
    print(f"    Max calm action:          {sp['max_calm_action']:.4f}")
    print(f"    Calm steps (with peers):  {sp['calm_steps_with_peers']}")
    print()

    # Action distribution
    ad = report["action_distribution"]
    print(f"  ACTION DISTRIBUTION")
    print(f"    Mean:   {ad['mean']:.4f}  Std: {ad['std']:.4f}  Median: {ad['median']:.4f}")
    print(f"    Steps with action > {report['model_action_threshold']}: {ad['pct_above_threshold']:.1f}%")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="H5 Sim-to-Real Validation: replay real convoy data through trained model"
    )
    parser.add_argument("--tx_csv", type=Path, required=True,
                        help="Path to ego TX CSV (V001_tx_*.csv)")
    parser.add_argument("--rx_csv", type=Path, required=True,
                        help="Path to ego RX CSV (V001_rx_*.csv)")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to trained model (model_final.zip)")
    parser.add_argument("--output_dir", type=Path, default=Path("ml/data/sim_to_real_validation"),
                        help="Output directory for reports")
    parser.add_argument("--forward_axis", choices=["x", "y"], default="y",
                        help="Which accelerometer axis is forward (default: y)")
    parser.add_argument("--recording_name", type=str, default=None,
                        help="Label for this recording (auto-detected from filename)")
    parser.add_argument(
        "--braking_received_mode",
        choices=["decay", "latched", "instant", "off"],
        default="decay",
        help="How to synthesize replay-side braking_received (ego[4]). "
             "'decay' (default) matches Run 020 training: exponential decay "
             "with factor 0.95/step. 'latched'/'instant'/'off' kept for "
             "regression analysis against earlier runs.",
    )
    args = parser.parse_args()

    # Validate inputs
    for p, label in [(args.tx_csv, "TX CSV"), (args.rx_csv, "RX CSV"), (args.model_path, "Model")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    recording_name = args.recording_name
    if recording_name is None:
        recording_name = args.tx_csv.parent.name

    print(f"Loading TX: {args.tx_csv}")
    tx_rows = parse_csv(args.tx_csv, args.forward_axis)
    print(f"  {len(tx_rows)} rows, {(tx_rows[-1].timestamp_local_ms - tx_rows[0].timestamp_local_ms) / 1000:.1f}s")

    print(f"Loading RX: {args.rx_csv}")
    rx_rows = parse_csv(args.rx_csv, args.forward_axis)
    print(f"  {len(rx_rows)} rows, {(rx_rows[-1].timestamp_local_ms - rx_rows[0].timestamp_local_ms) / 1000:.1f}s")
    unique_peers = set(r.vehicle_id for r in rx_rows)
    print(f"  Peers: {sorted(unique_peers)}")

    print(f"Braking signal mode: {args.braking_received_mode}")

    print(f"Loading model: {args.model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(str(args.model_path))
    print("  Model loaded successfully")

    report = run_replay(
        tx_rows, rx_rows, model, args.output_dir, recording_name,
        braking_received_mode=args.braking_received_mode,
    )
    print_report(report)

    # Print verdict
    print()
    s = report["sensitivity"]
    sp = report["specificity"]
    has_events = s["events_total"] > 0
    if has_events:
        all_detected = s["events_detected"] == s["events_total"]
        low_fp = sp["false_positive_rate"] < 0.10
        if all_detected and low_fp:
            print("VERDICT: PASS — Model detects real braking events with low false positives")
        elif all_detected:
            print(f"VERDICT: WARN — Model detects events but FP rate is {sp['false_positive_rate'] * 100:.1f}%")
        else:
            missed = s["events_total"] - s["events_detected"]
            print(f"VERDICT: FAIL — Model missed {missed}/{s['events_total']} braking events")
    else:
        low_fp = sp["false_positive_rate"] < 0.10
        if low_fp:
            print("VERDICT: PASS — No braking events in recording, model stays calm")
        else:
            print(f"VERDICT: WARN — No braking events, but FP rate is {sp['false_positive_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
