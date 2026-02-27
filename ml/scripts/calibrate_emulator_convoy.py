#!/usr/bin/env python3
"""
Calibrate ESP-NOW emulator parameters from convoy recording data.

Uses ONLY healthy links (excludes V001<->V002 due to placement-induced
packet loss). Produces emulator_params_convoy.json suitable for training.

Usage:
    cd roadsense-v2v/ml
    source venv/bin/activate
    python scripts/calibrate_emulator_convoy.py \
        --input_dir /home/amirkhalifa/RoadSense2/Convoy_recording_02212026 \
        --output ml/espnow_emulator/emulator_params_convoy.json

Author: Amir Khalifa / Claude
Date: February 24, 2026
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────
HEALTHY_LINKS = [
    ("V001", "V003"),
    ("V002", "V003"),
    ("V003", "V001"),
    ("V003", "V002"),
]
EXCLUDED_LINKS = [
    ("V001", "V002"),
    ("V002", "V001"),
]
VEHICLES = ["V001", "V002", "V003"]
DISTANCE_BINS = [
    ("0_20m", 0.0, 20.0),
    ("20_50m", 20.0, 50.0),
    ("50_100m", 50.0, 100.0),
    ("100m_plus", 100.0, float("inf")),
]
GPS_M_PER_DEG_LAT = 110540.0
GPS_M_PER_DEG_LON_EQUATOR = 111320.0
STALENESS_MS = 500


# ── CSV Parsing ────────────────────────────────────────────────────────
COLUMNS = [
    "timestamp_local_ms", "msg_timestamp", "vehicle_id",
    "lat", "lon", "speed", "heading",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
]
FLOAT_COLS = COLUMNS[3:]  # everything after vehicle_id
INT_COLS = ["timestamp_local_ms", "msg_timestamp"]
SENDER_COLUMN_CANDIDATES = ("from_vehicle_id", "vehicle_id")
MIN_EXPECTED_COLUMNS = 16


def parse_csv(path: Path) -> Dict[str, np.ndarray]:
    """Parse a convoy CSV into dict of numpy arrays."""
    def _empty() -> Dict[str, np.ndarray]:
        out = {}
        for c in INT_COLS:
            out[c] = np.array([], dtype=np.int64)
        out["vehicle_id"] = np.array([], dtype=object)
        for c in FLOAT_COLS:
            out[c] = np.array([], dtype=np.float64)
        return out

    rows = {c: [] for c in COLUMNS}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return _empty()

        if len(header) < MIN_EXPECTED_COLUMNS:
            return _empty()

        header_map = {name.strip(): idx for idx, name in enumerate(header)}
        sender_col = next((name for name in SENDER_COLUMN_CANDIDATES if name in header_map), None)
        if sender_col is None:
            return _empty()

        required = ("timestamp_local_ms", "msg_timestamp", "lat", "lon", "speed", "heading",
                    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
                    "mag_x", "mag_y", "mag_z")
        if any(col not in header_map for col in required):
            return _empty()

        for parts in reader:
            if len(parts) != len(header):
                continue
            try:
                rows["timestamp_local_ms"].append(int(float(parts[header_map["timestamp_local_ms"]])))
                rows["msg_timestamp"].append(int(float(parts[header_map["msg_timestamp"]])))
                rows["vehicle_id"].append(parts[header_map[sender_col]].strip())
                for col in FLOAT_COLS:
                    rows[col].append(float(parts[header_map[col]]))
            except (ValueError, IndexError):
                continue

    out = {}
    for c in INT_COLS:
        out[c] = np.array(rows[c], dtype=np.int64)
    out["vehicle_id"] = np.array(rows["vehicle_id"], dtype=object)
    for c in FLOAT_COLS:
        out[c] = np.array(rows[c], dtype=np.float64)
    return out


def find_csv(input_dir: Path, vehicle: str, log_type: str) -> Optional[Path]:
    """Find the CSV file for a given vehicle and type (tx/rx)."""
    pattern = f"{vehicle}_{log_type}_*.csv"
    matches = sorted(input_dir.rglob(pattern))
    if not matches:
        # Try case variations
        pattern2 = f"{vehicle.lower()}_{log_type}_*.csv"
        matches = sorted(input_dir.rglob(pattern2))
    return matches[0] if matches else None


def load_all_logs(input_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Load all 6 CSV files."""
    logs = {}
    for v in VEHICLES:
        logs[v] = {}
        for lt in ("tx", "rx"):
            path = find_csv(input_dir, v, lt)
            if path is None:
                print(f"WARNING: Could not find {v}_{lt} CSV in {input_dir}")
                logs[v][lt] = None
                continue
            print(f"  Loading {path.name} ... ", end="")
            data = parse_csv(path)
            n = len(data["timestamp_local_ms"])
            print(f"{n} rows")
            logs[v][lt] = data
    return logs


# ── Clock Offset Estimation ───────────────────────────────────────────
def estimate_clock_offset(logs, sender: str, receiver: str) -> Optional[Dict]:
    """Estimate clock offset between sender and receiver using RX timestamps."""
    rx = logs[receiver]["rx"]
    if rx is None:
        return None
    mask = rx["vehicle_id"] == sender
    if np.sum(mask) < 10:
        return None
    delta = rx["timestamp_local_ms"][mask].astype(float) - rx["msg_timestamp"][mask].astype(float)
    return {
        "median_ms": float(np.median(delta)),
        "std_ms": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
        "samples": int(delta.size),
    }


# ── Haversine Distance ────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# ── Packet Delivery Analysis (Healthy Links Only) ─────────────────────
def compute_link_stats(logs, sender: str, receiver: str) -> Dict:
    """Compute PDR, latency, burst stats for one link."""
    tx = logs[sender]["tx"]
    rx = logs[receiver]["rx"]
    if tx is None or rx is None:
        return {"link": f"{sender}->{receiver}", "valid": False}

    tx_msg = np.unique(tx["msg_timestamp"])
    rx_mask = rx["vehicle_id"] == sender
    rx_msg = np.unique(rx["msg_timestamp"][rx_mask])

    if tx_msg.size == 0 or rx_msg.size == 0:
        return {"link": f"{sender}->{receiver}", "valid": False}

    # Overlap window
    start = max(int(tx_msg.min()), int(rx_msg.min()))
    end = min(int(tx_msg.max()), int(rx_msg.max()))
    if end <= start:
        return {"link": f"{sender}->{receiver}", "valid": False}

    tx_ov = tx_msg[(tx_msg >= start) & (tx_msg <= end)]
    rx_set = set(rx_msg.tolist())

    # PDR
    matched = sum(1 for t in tx_ov if int(t) in rx_set)
    total = tx_ov.size
    pdr = matched / total if total > 0 else 0.0
    loss_rate = 1.0 - pdr

    # Loss flags for burst analysis
    loss_flags = np.array([0 if int(t) in rx_set else 1 for t in tx_ov], dtype=np.int8)

    # Burst lengths
    bursts = []
    run = 0
    for flag in loss_flags:
        if flag:
            run += 1
        elif run > 0:
            bursts.append(run)
            run = 0
    if run > 0:
        bursts.append(run)

    mean_burst = float(np.mean(bursts)) if bursts else 1.0
    max_burst = int(np.max(bursts)) if bursts else 0

    # Relative latency (remove clock offset via median subtraction)
    rx_all = rx["timestamp_local_ms"][rx_mask].astype(float)
    rx_msg_ts = rx["msg_timestamp"][rx_mask].astype(float)
    raw_delta = rx_all - rx_msg_ts
    median_offset = np.median(raw_delta)
    relative_latency = raw_delta - median_offset
    # Shift so minimum is ~0
    p1 = float(np.percentile(relative_latency, 1))
    relative_latency = relative_latency - p1

    return {
        "link": f"{sender}->{receiver}",
        "valid": True,
        "tx_in_overlap": int(total),
        "matched": int(matched),
        "pdr": float(pdr),
        "loss_rate": float(loss_rate),
        "mean_burst_length": float(mean_burst),
        "max_burst_length": int(max_burst),
        "burst_count": len(bursts),
        "latency_mean_ms": float(np.mean(relative_latency)),
        "latency_median_ms": float(np.median(relative_latency)),
        "latency_std_ms": float(np.std(relative_latency, ddof=1)) if relative_latency.size > 1 else 0.0,
        "latency_p95_ms": float(np.percentile(relative_latency, 95)),
        "latency_p99_ms": float(np.percentile(relative_latency, 99)),
        "clock_offset_std_ms": float(np.std(raw_delta, ddof=1)) if raw_delta.size > 1 else 0.0,
    }


def interpolate_distance(logs, sender: str, receiver: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get distance between sender and receiver at each TX timestamp.

    Uses reverse-link clock offset to align receiver's TX positions
    into sender's time domain, then interpolates.

    Returns (tx_timestamps, distances, received_flags) or None.
    """
    tx = logs[sender]["tx"]
    recv_tx = logs[receiver]["tx"]
    if tx is None or recv_tx is None:
        return None

    # Clock offset: receiver→sender (how to map receiver times to sender times)
    offset_info = estimate_clock_offset(logs, receiver, sender)
    if offset_info is None or offset_info["samples"] < 10:
        return None

    recv_offset = offset_info["median_ms"]
    recv_t_in_sender = recv_tx["msg_timestamp"].astype(float) + recv_offset

    sender_t = tx["msg_timestamp"].astype(float)
    if sender_t.size < 2 or recv_t_in_sender.size < 2:
        return None

    # Filter valid GPS
    s_valid = (tx["lat"] != 0) & (tx["lon"] != 0)
    r_valid = (recv_tx["lat"] != 0) & (recv_tx["lon"] != 0)
    if np.sum(s_valid) < 5 or np.sum(r_valid) < 5:
        return None

    # Sort receiver by aligned time
    order = np.argsort(recv_t_in_sender)
    r_t = recv_t_in_sender[order]
    r_lat = recv_tx["lat"][order]
    r_lon = recv_tx["lon"][order]
    r_v = r_valid[order]

    r_t_valid = r_t[r_v]
    r_lat_valid = r_lat[r_v]
    r_lon_valid = r_lon[r_v]

    # Only interpolate within receiver's time range
    in_range = s_valid & (sender_t >= r_t_valid.min()) & (sender_t <= r_t_valid.max())
    if np.sum(in_range) < 5:
        return None

    interp_lat = np.interp(sender_t[in_range], r_t_valid, r_lat_valid)
    interp_lon = np.interp(sender_t[in_range], r_t_valid, r_lon_valid)

    dist = haversine_m(tx["lat"][in_range], tx["lon"][in_range], interp_lat, interp_lon)

    # Received flags
    rx = logs[receiver]["rx"]
    if rx is None:
        return None
    rx_mask = rx["vehicle_id"] == sender
    rx_set = set(rx["msg_timestamp"][rx_mask].astype(np.int64).tolist())
    tx_ts_in_range = tx["msg_timestamp"][in_range].astype(np.int64)
    received = np.array([int(t) in rx_set for t in tx_ts_in_range], dtype=bool)

    return tx_ts_in_range, dist, received


def compute_distance_binned_loss_all_healthy(logs) -> Dict[str, Dict]:
    """Compute distance-binned loss rates across all healthy links."""
    all_dist = []
    all_received = []

    for sender, receiver in HEALTHY_LINKS:
        result = interpolate_distance(logs, sender, receiver)
        if result is None:
            continue
        _, dist, received = result
        all_dist.append(dist)
        all_received.append(received)

    if not all_dist:
        return {}

    dist_all = np.concatenate(all_dist)
    recv_all = np.concatenate(all_received)

    bins = {}
    for name, lo, hi in DISTANCE_BINS:
        mask = (dist_all >= lo) & (dist_all < hi)
        total = int(np.sum(mask))
        if total < 10:
            bins[name] = {"count": total, "pdr": None, "loss_rate": None}
            continue
        matched = int(np.sum(recv_all[mask]))
        pdr = matched / total
        bins[name] = {"count": total, "pdr": float(pdr), "loss_rate": float(1.0 - pdr)}

    return bins


# ── Sensor Noise (Stationary Tail + Driving) ──────────────────────────
def moving_average(arr, window=21):
    """Simple centered moving average for detrending."""
    if arr.size < window:
        return arr.copy()
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:arr.size]


def circular_std_deg(angles_deg: np.ndarray) -> Optional[float]:
    """Circular standard deviation for angular data."""
    if angles_deg.size < 2:
        return None
    rad = np.radians(angles_deg)
    S = np.mean(np.sin(rad))
    C = np.mean(np.cos(rad))
    R = math.sqrt(S ** 2 + C ** 2)
    if R >= 1.0:
        return 0.0
    return float(math.degrees(math.sqrt(-2.0 * math.log(R))))


def compute_noise_for_segment(data: Dict, mask: np.ndarray, detrend: bool) -> Dict[str, float]:
    """Compute sensor noise stats for a boolean mask of samples."""
    out = {"samples": int(np.sum(mask))}

    # GPS noise (residual from smoothed trajectory)
    gps_valid = mask & (data["lat"] != 0) & (data["lon"] != 0)
    if np.sum(gps_valid) > 10:
        lat_smooth = moving_average(data["lat"][gps_valid])
        lon_smooth = moving_average(data["lon"][gps_valid])
        lat_res = data["lat"][gps_valid] - lat_smooth
        lon_res = data["lon"][gps_valid] - lon_smooth
        mean_lat = float(np.mean(data["lat"][gps_valid]))
        lat_std_m = float(np.std(lat_res, ddof=1)) * GPS_M_PER_DEG_LAT
        lon_std_m = float(np.std(lon_res, ddof=1)) * GPS_M_PER_DEG_LON_EQUATOR * math.cos(math.radians(mean_lat))
        out["gps_std_m"] = float(math.sqrt((lat_std_m ** 2 + lon_std_m ** 2) / 2.0))

    # Accel noise
    def residual_std(arr):
        if arr.size < 3:
            return float("nan")
        if detrend:
            arr = arr - moving_average(arr)
        return float(np.std(arr, ddof=1))

    accel_stds = [residual_std(data[c][mask]) for c in ("accel_x", "accel_y", "accel_z")]
    out["accel_std_ms2"] = float(np.nanmean(accel_stds))

    gyro_stds = [residual_std(data[c][mask]) for c in ("gyro_x", "gyro_y", "gyro_z")]
    out["gyro_std_rad_s"] = float(np.nanmean(gyro_stds))

    # Mag noise (no detrend — mag should be stable in a fixed orientation)
    mag_stds = [residual_std(data[c][mask]) for c in ("mag_x", "mag_y", "mag_z")]
    out["mag_std_ut"] = float(np.nanmean(mag_stds))

    # Speed noise
    speed_arr = data["speed"][mask]
    if speed_arr.size >= 3:
        if detrend:
            speed_arr = speed_arr - moving_average(speed_arr)
        out["speed_std_ms"] = float(np.std(speed_arr, ddof=1))

    # Heading noise (from magnetometer)
    heading_mag = np.degrees(np.arctan2(data["mag_y"][mask], data["mag_x"][mask]))
    heading_std = circular_std_deg(heading_mag)
    if heading_std is not None:
        out["heading_std_deg"] = float(heading_std)

    return out


def find_stationary_tail(speed: np.ndarray, min_duration_samples: int = 100) -> Optional[np.ndarray]:
    """Find the FINAL contiguous stationary segment (the parked tail).

    We only want the last stretch where the vehicle stopped and stayed stopped,
    NOT brief stops at traffic lights earlier in the drive.

    Returns a boolean mask or None if no suitable tail found.
    """
    stationary = speed < 0.5
    n = len(stationary)
    if np.sum(stationary) < min_duration_samples:
        return None

    # Walk backwards from the end to find the last contiguous stationary block
    # Allow small gaps (up to 3 samples) where speed briefly blips above threshold
    end_idx = n - 1
    # Find last stationary sample
    while end_idx >= 0 and not stationary[end_idx]:
        end_idx -= 1
    if end_idx < min_duration_samples:
        return None

    start_idx = end_idx
    gap_count = 0
    while start_idx > 0:
        if stationary[start_idx - 1]:
            start_idx -= 1
            gap_count = 0
        elif gap_count < 3:
            # Allow small gap (GPS speed noise can briefly exceed 0.5)
            start_idx -= 1
            gap_count += 1
        else:
            break

    tail_len = end_idx - start_idx + 1
    if tail_len < min_duration_samples:
        return None

    mask = np.zeros(n, dtype=bool)
    mask[start_idx:end_idx + 1] = True
    # Within the tail, only use truly stationary samples
    mask = mask & stationary
    return mask


def compute_sensor_noise(logs) -> Dict[str, Dict]:
    """Compute sensor noise across all vehicles for stationary and cruising segments."""
    stationary_results = []
    cruising_results = []

    for v in VEHICLES:
        tx = logs[v]["tx"]
        if tx is None:
            continue

        speed = tx["speed"]
        accel_x = tx["accel_x"]

        # Stationary: ONLY the final parked tail (~42s at end of recording)
        # This avoids mixing in brief stops at different locations which
        # inflates GPS noise artificially.
        tail_mask = find_stationary_tail(speed)
        if tail_mask is not None and np.sum(tail_mask) >= 20:
            result = compute_noise_for_segment(tx, tail_mask, detrend=False)
            stationary_results.append(result)
            print(f"  {v} stationary tail: {result['samples']} samples")
        else:
            print(f"  {v} stationary tail: not found or too short")

        # Cruising: speed > 3 m/s, |accel_x| < 1.0 m/s²
        cruising_mask = (speed > 3.0) & (np.abs(accel_x) < 1.0)
        if np.sum(cruising_mask) >= 20:
            result = compute_noise_for_segment(tx, cruising_mask, detrend=True)
            cruising_results.append(result)
            print(f"  {v} cruising: {result['samples']} samples")

    def avg_results(results: List[Dict], keys: List[str]) -> Dict[str, float]:
        out = {}
        for k in keys:
            vals = [r[k] for r in results if k in r and not math.isnan(r[k])]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    noise_keys = ["gps_std_m", "speed_std_ms", "accel_std_ms2", "gyro_std_rad_s",
                  "heading_std_deg", "mag_std_ut"]

    return {
        "stationary": avg_results(stationary_results, noise_keys),
        "cruising": avg_results(cruising_results, noise_keys),
    }


# ── Main Calibration ──────────────────────────────────────────────────
def calibrate(input_dir: Path, output_path: Path):
    print("=" * 60)
    print("ESP-NOW Emulator Calibration from Convoy Recording")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_path}")
    print()

    # ── Load data ──
    print("Loading CSV files...")
    logs = load_all_logs(input_dir)
    print()

    # ── Link analysis (healthy only) ──
    print("Analyzing healthy links...")
    link_stats = {}
    for sender, receiver in HEALTHY_LINKS:
        stats = compute_link_stats(logs, sender, receiver)
        link_stats[f"{sender}->{receiver}"] = stats
        if stats["valid"]:
            print(f"  {sender}->{receiver}: PDR={stats['pdr']:.1%}, "
                  f"loss={stats['loss_rate']:.1%}, "
                  f"burst_mean={stats['mean_burst_length']:.2f}, "
                  f"latency_std={stats['latency_std_ms']:.2f}ms")
        else:
            print(f"  {sender}->{receiver}: INVALID (insufficient data)")

    print()
    print("Excluded links (V001<->V002, broken due to placement):")
    for sender, receiver in EXCLUDED_LINKS:
        stats = compute_link_stats(logs, sender, receiver)
        if stats["valid"]:
            print(f"  {sender}->{receiver}: PDR={stats['pdr']:.1%} (excluded from calibration)")
    print()

    # ── Aggregate network stats from healthy links ──
    valid_stats = [s for s in link_stats.values() if s.get("valid")]
    if not valid_stats:
        print("ERROR: No valid healthy links found!")
        sys.exit(1)

    # Overall loss rate (weighted by TX count)
    total_tx = sum(s["tx_in_overlap"] for s in valid_stats)
    total_matched = sum(s["matched"] for s in valid_stats)
    overall_loss = 1.0 - (total_matched / total_tx) if total_tx > 0 else 0.0
    overall_pdr = total_matched / total_tx if total_tx > 0 else 0.0

    # Latency (aggregate from healthy links, exclude V003->V002 if std is anomalous)
    latency_stds = []
    for name, s in link_stats.items():
        if s.get("valid") and s["clock_offset_std_ms"] < 10.0:
            latency_stds.append(s["latency_std_ms"])
            print(f"  Using {name} for latency: std={s['latency_std_ms']:.3f}ms "
                  f"(clock_offset_std={s['clock_offset_std_ms']:.2f}ms)")
        elif s.get("valid"):
            print(f"  SKIPPING {name} for latency: clock_offset_std={s['clock_offset_std_ms']:.2f}ms (anomalous)")

    jitter_std = float(np.mean(latency_stds)) if latency_stds else 1.0
    print()

    # Burst (from healthy links)
    burst_lengths_all = [s["mean_burst_length"] for s in valid_stats if s.get("valid")]
    mean_burst = float(np.mean(burst_lengths_all)) if burst_lengths_all else 1.0
    max_burst = max((s["max_burst_length"] for s in valid_stats if s.get("valid")), default=1)

    # ── Distance-binned loss ──
    print("Computing distance-binned loss (healthy links)...")
    dist_bins = compute_distance_binned_loss_all_healthy(logs)
    for name, stats in dist_bins.items():
        if stats["loss_rate"] is not None:
            print(f"  {name}: {stats['count']} packets, loss={stats['loss_rate']:.1%}")
        else:
            print(f"  {name}: {stats['count']} packets (insufficient data)")
    print()

    # ── Sensor noise ──
    print("Computing sensor noise...")
    noise = compute_sensor_noise(logs)
    print()

    stat_noise = noise["stationary"]
    drive_noise = noise["cruising"]

    print("Stationary noise:")
    for k, v in stat_noise.items():
        print(f"  {k}: {v:.4f}")
    print("Cruising noise:")
    for k, v in drive_noise.items():
        print(f"  {k}: {v:.4f}")
    print()

    # ── Build calibrated params ──
    # Latency: keep RTT-measured base (can't determine absolute from convoy),
    # update jitter from convoy measurement
    base_ms = 3.84  # from RTT characterization (Feb 20)

    # Packet loss: use distance-binned values from healthy links
    base_rate = dist_bins.get("0_20m", {}).get("loss_rate")
    rate_20_50 = dist_bins.get("20_50m", {}).get("loss_rate")
    rate_50_100 = dist_bins.get("50_100m", {}).get("loss_rate")

    # Fallback to overall if bins are sparse
    if base_rate is None:
        base_rate = overall_loss
    if rate_20_50 is None:
        rate_20_50 = overall_loss
    if rate_50_100 is None:
        rate_50_100 = min(overall_loss * 2, 0.95)

    # Sensor noise: use CRUISING values as the emulator base.
    # Rationale: the model operates during driving, not while parked.
    # Cruising noise captures real vibration + measurement noise.
    # Stationary values are kept in metadata for reference.
    #
    # For sensors where stationary is a cleaner measurement (mag, gyro),
    # use the max of stationary and cruising to be conservative.
    # For GPS/speed/accel, use cruising directly (stationary can be
    # unrealistically low — NEO-6M reports 0.0 speed when parked).
    sensor_noise = {
        "gps_std_m": max(drive_noise.get("gps_std_m", 1.5), 0.5),
        "speed_std_ms": max(drive_noise.get("speed_std_ms", 0.05), 0.02),
        "accel_std_ms2": max(drive_noise.get("accel_std_ms2", 0.2),
                             stat_noise.get("accel_std_ms2", 0.07)),
        "gyro_std_rad_s": max(drive_noise.get("gyro_std_rad_s", 0.01),
                              stat_noise.get("gyro_std_rad_s", 0.006)),
        "heading_std_deg": max(drive_noise.get("heading_std_deg", 1.0),
                               stat_noise.get("heading_std_deg", 0.4)),
        "mag_std_ut": max(drive_noise.get("mag_std_ut", 0.6),
                          stat_noise.get("mag_std_ut", 0.4)),
    }

    # Domain randomization: ranges from stationary floor to cruising * 1.5
    # Trains the model to handle both calm and noisy conditions.
    dr_gps_lo = max(0.5, stat_noise.get("gps_std_m", 0.5))
    dr_gps_hi = max(sensor_noise["gps_std_m"] * 1.5, dr_gps_lo + 1.0)
    dr_jitter_lo = max(0.3, jitter_std * 0.5)
    dr_jitter_hi = max(jitter_std * 2.0, dr_jitter_lo + 0.5)
    dr_latency_lo = max(1.0, base_ms * 0.5)
    dr_latency_hi = base_ms * 2.0
    dr_loss_lo = max(0.0, base_rate * 0.5)
    dr_loss_hi = min(1.0, max(rate_20_50, base_rate) * 2.0)

    # Burst loss: compute multiplier from burst behavior
    # loss_multiplier = burst_loss_rate / base_loss_rate (approximation)
    loss_multiplier = 2.5  # conservative default
    max_loss_cap = min(0.8, overall_loss * 3)

    params = {
        "latency": {
            "base_ms": round(base_ms, 2),
            "distance_factor": 0.0,
            "jitter_std_ms": round(jitter_std, 3),
        },
        "packet_loss": {
            "base_rate": round(base_rate, 4),
            "distance_threshold_1": 50,
            "distance_threshold_2": 100,
            "rate_tier_1": round(rate_20_50, 4),
            "rate_tier_2": round(rate_50_100, 4),
            "rate_tier_3": round(min(rate_50_100 * 1.5, 0.95), 4),
        },
        "burst_loss": {
            "enabled": True,
            "mean_burst_length": round(mean_burst, 3),
            "loss_multiplier": round(loss_multiplier, 2),
            "max_loss_cap": round(max_loss_cap, 4),
        },
        "sensor_noise": {k: round(v, 6) for k, v in sensor_noise.items()},
        "observation": {
            "staleness_threshold_ms": STALENESS_MS,
        },
        "domain_randomization": {
            "latency_range_ms": [round(dr_latency_lo, 2), round(dr_latency_hi, 2)],
            "loss_rate_range": [round(dr_loss_lo, 4), round(dr_loss_hi, 4)],
            "jitter_std_range_ms": [round(dr_jitter_lo, 2), round(dr_jitter_hi, 2)],
            "gps_noise_range_m": [round(dr_gps_lo, 3), round(dr_gps_hi, 3)],
        },
        "_metadata": {
            "source": "convoy_recording_02212026",
            "healthy_links": [f"{s}->{r}" for s, r in HEALTHY_LINKS],
            "excluded_links": [f"{s}->{r}" for s, r in EXCLUDED_LINKS],
            "exclusion_reason": "V001<->V002 had ~85% loss due to board placement inside car door",
            "overall_pdr_healthy": round(overall_pdr, 4),
            "overall_loss_healthy": round(overall_loss, 4),
            "stationary_noise": {k: round(v, 6) for k, v in stat_noise.items()},
            "driving_noise": {k: round(v, 6) for k, v in drive_noise.items()},
            "distance_bins": dist_bins,
            "calibration_date": "2026-02-24",
        },
    }

    # ── Write output ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"Calibrated params written to: {output_path}")

    # ── Summary ──
    print()
    print("=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print()
    print("Network (healthy links only):")
    print(f"  Overall PDR:        {overall_pdr:.1%}")
    print(f"  Overall loss:       {overall_loss:.1%}")
    print(f"  Latency base:       {base_ms:.2f} ms (from RTT)")
    print(f"  Jitter std:         {jitter_std:.3f} ms (convoy-measured)")
    print(f"  Mean burst length:  {mean_burst:.2f}")
    print()
    print("Packet loss by distance:")
    print(f"  0-20m:    {base_rate:.1%}")
    print(f"  20-50m:   {rate_20_50:.1%}")
    print(f"  50-100m:  {rate_50_100:.1%}")
    print()
    print("Sensor noise (stationary → base, cruising → DR upper bound):")
    for k in ["accel_std_ms2", "gps_std_m", "gyro_std_rad_s", "heading_std_deg", "mag_std_ut", "speed_std_ms"]:
        s = stat_noise.get(k, float("nan"))
        d = drive_noise.get(k, float("nan"))
        print(f"  {k:20s}: stationary={s:.4f}  cruising={d:.4f}")
    print()
    print("Comparison with previous RTT params:")
    prev = {
        "accel_std_ms2": 0.059, "gps_std_m": 1.506, "gyro_std_rad_s": 0.01,
        "heading_std_deg": 0.58, "mag_std_ut": 0.612, "speed_std_ms": 0.021,
    }
    for k, prev_v in prev.items():
        new_v = sensor_noise.get(k, float("nan"))
        ratio = new_v / prev_v if prev_v > 0 else float("inf")
        print(f"  {k:20s}: RTT={prev_v:.4f}  convoy_stat={new_v:.4f}  ratio={ratio:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate emulator from convoy recording")
    parser.add_argument("--input_dir", type=str,
                        default="/home/amirkhalifa/RoadSense2/Convoy_recording_02212026",
                        help="Directory containing convoy CSV files")
    parser.add_argument("--output", type=str,
                        default="/home/amirkhalifa/RoadSense2/roadsense-v2v/ml/espnow_emulator/emulator_params_convoy.json",
                        help="Output JSON path")
    args = parser.parse_args()
    calibrate(Path(args.input_dir), Path(args.output))
