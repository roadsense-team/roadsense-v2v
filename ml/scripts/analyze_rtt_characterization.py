#!/usr/bin/env python3
"""
RTT characterization analysis for ESP-NOW field logs.

Outputs:
  - Summary stats printed to stdout
  - emulator_params.json tuned from measurements
  - Plots (RTT distribution, loss over time, GPS track)
  - loss_by_segment.csv + analysis_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


REQUIRED_COLUMNS = [
    "sequence",
    "send_time_ms",
    "recv_time_ms",
    "rtt_ms",
    "lat",
    "lon",
    "speed",
    "heading",
    "accel_x",
    "accel_y",
    "accel_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "lost",
]

GPS_METERS_PER_DEG_LAT = 111000.0
DEFAULT_STALENESS_MS = 500
DEFAULT_MONITORED_VEHICLES = ["V002", "V003"]
LOSS_MULTIPLIER_MIN = 1.5
LOSS_MULTIPLIER_MAX = 5.0
LOSS_CAP_MIN = 0.50
LOSS_CAP_MAX = 0.95
DEFAULT_DISTANCE_THRESHOLDS_M = (50, 100)


@dataclass(frozen=True)
class RTTRecord:
    sequence: int
    send_time_ms: int
    recv_time_ms: int
    rtt_ms: float
    lat: float
    lon: float
    speed: float
    heading: float
    accel_x: float
    accel_y: float
    accel_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    lost: int


def _parse_int(value: str) -> int:
    return int(float(value))


def _parse_float(value: str) -> float:
    return float(value)


def load_records(path: Path) -> List[RTTRecord]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        records: List[RTTRecord] = []
        for row in reader:
            try:
                record = RTTRecord(
                    sequence=_parse_int(row["sequence"]),
                    send_time_ms=_parse_int(row["send_time_ms"]),
                    recv_time_ms=_parse_int(row["recv_time_ms"]),
                    rtt_ms=_parse_float(row["rtt_ms"]),
                    lat=_parse_float(row["lat"]),
                    lon=_parse_float(row["lon"]),
                    speed=_parse_float(row["speed"]),
                    heading=_parse_float(row["heading"]),
                    accel_x=_parse_float(row["accel_x"]),
                    accel_y=_parse_float(row["accel_y"]),
                    accel_z=_parse_float(row["accel_z"]),
                    mag_x=_parse_float(row["mag_x"]),
                    mag_y=_parse_float(row["mag_y"]),
                    mag_z=_parse_float(row["mag_z"]),
                    lost=_parse_int(row["lost"]),
                )
            except (TypeError, ValueError):
                continue
            records.append(record)

    return records


def sort_records(records: Iterable[RTTRecord]) -> List[RTTRecord]:
    return sorted(records, key=lambda r: (r.send_time_ms, r.sequence))


def valid_gps_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (np.abs(lat) > 1e-6) & (np.abs(lon) > 1e-6)


def compute_percentiles(values: np.ndarray, percentiles: Iterable[float]) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{int(p)}": float("nan") for p in percentiles}
    result = np.percentile(values, list(percentiles))
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, result)}


def compute_basic_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_loss_segments(
    records: List[RTTRecord],
    segment_seconds: int,
) -> Tuple[List[Dict[str, float]], float]:
    if not records:
        return [], 0.0

    send_times = np.array([r.send_time_ms for r in records], dtype=np.int64)
    lost_flags = np.array([r.lost for r in records], dtype=np.int8)
    start_ms = int(send_times.min())
    time_s = (send_times - start_ms) / 1000.0

    segment_idx = (time_s // segment_seconds).astype(int)
    max_idx = int(segment_idx.max())

    segments: List[Dict[str, float]] = []
    for idx in range(max_idx + 1):
        mask = segment_idx == idx
        count = int(mask.sum())
        if count == 0:
            continue
        lost_count = int(lost_flags[mask].sum())
        loss_rate = lost_count / count
        segment = {
            "index": idx,
            "start_s": float(idx * segment_seconds),
            "end_s": float((idx + 1) * segment_seconds),
            "count": count,
            "lost_count": lost_count,
            "loss_rate": loss_rate,
        }
        segments.append(segment)

    duration_s = float(time_s.max() if time_s.size else 0.0)
    return segments, duration_s


def summarize_loss_segments(segments: List[Dict[str, float]]) -> Dict[str, float]:
    if not segments:
        return {
            "segment_count": 0,
            "baseline_rate": float("nan"),
            "high_loss_threshold": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    loss_rates = np.array([seg["loss_rate"] for seg in segments], dtype=float)
    p50, p90, p95 = np.percentile(loss_rates, [50, 90, 95])
    baseline = float(np.percentile(loss_rates, 25))
    high_threshold = float(max(p90, baseline + 0.1))
    return {
        "segment_count": int(len(segments)),
        "baseline_rate": baseline,
        "high_loss_threshold": high_threshold,
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
    }


def find_high_loss_zones(
    records: List[RTTRecord],
    segments: List[Dict[str, float]],
    summary: Dict[str, float],
    segment_seconds: int,
) -> List[Dict[str, float]]:
    if not segments:
        return []

    high_threshold = summary["high_loss_threshold"]
    high_segments = [seg for seg in segments if seg["loss_rate"] >= high_threshold]
    if not high_segments:
        return []

    high_segments = sorted(high_segments, key=lambda s: s["index"])
    zones: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = [high_segments[0]]
    for seg in high_segments[1:]:
        if seg["index"] == current[-1]["index"] + 1:
            current.append(seg)
        else:
            zones.append(current)
            current = [seg]
    zones.append(current)

    send_times = np.array([r.send_time_ms for r in records], dtype=np.int64)
    start_ms = int(send_times.min())
    lats = np.array([r.lat for r in records], dtype=float)
    lons = np.array([r.lon for r in records], dtype=float)
    gps_valid = valid_gps_mask(lats, lons)

    zone_summaries: List[Dict[str, float]] = []
    for zone in zones:
        zone_start_s = zone[0]["start_s"]
        zone_end_s = zone[-1]["end_s"]
        zone_loss_rates = [seg["loss_rate"] for seg in zone]
        zone_indices = {seg["index"] for seg in zone}

        time_s = (send_times - start_ms) / 1000.0
        segment_idx = (time_s // segment_seconds).astype(int)
        in_zone = np.isin(segment_idx, list(zone_indices))

        zone_lat = lats[in_zone & gps_valid]
        zone_lon = lons[in_zone & gps_valid]

        zone_summary = {
            "start_s": float(zone_start_s),
            "end_s": float(zone_end_s),
            "duration_s": float(zone_end_s - zone_start_s),
            "loss_rate_avg": float(np.mean(zone_loss_rates)),
            "loss_rate_max": float(np.max(zone_loss_rates)),
            "segment_count": int(len(zone)),
        }

        if zone_lat.size > 0:
            zone_summary.update(
                {
                    "lat_min": float(np.min(zone_lat)),
                    "lat_max": float(np.max(zone_lat)),
                    "lon_min": float(np.min(zone_lon)),
                    "lon_max": float(np.max(zone_lon)),
                }
            )

        zone_summaries.append(zone_summary)

    return zone_summaries


def compute_burst_stats(records: List[RTTRecord]) -> Dict[str, float]:
    if not records:
        return {"mean_burst_length": 1.0, "max_burst_length": 0.0, "burst_count": 0}

    # NOTE: Records are sorted by send_time_ms. Because lost packets are logged
    # after the timeout window, consecutive lost sequences still cluster in this
    # order. If logging behavior changes, burst estimates may need revisiting.
    lost_flags = [r.lost for r in records]
    bursts: List[int] = []
    current = 0
    for lost in lost_flags:
        if lost:
            current += 1
        elif current > 0:
            bursts.append(current)
            current = 0
    if current > 0:
        bursts.append(current)

    if not bursts:
        return {"mean_burst_length": 1.0, "max_burst_length": 0.0, "burst_count": 0}

    return {
        "mean_burst_length": float(np.mean(bursts)),
        "max_burst_length": float(np.max(bursts)),
        "burst_count": int(len(bursts)),
    }


def compute_imu_noise(
    records: List[RTTRecord],
    stationary_speed: float,
) -> Dict[str, float]:
    if not records:
        return {}

    speeds = np.array([r.speed for r in records], dtype=float)
    mask = speeds <= stationary_speed
    if mask.sum() < 5:
        return {}

    ax = np.array([r.accel_x for r in records], dtype=float)[mask]
    ay = np.array([r.accel_y for r in records], dtype=float)[mask]
    az = np.array([r.accel_z for r in records], dtype=float)[mask]
    amag = np.sqrt(ax**2 + ay**2 + az**2)

    return {
        "samples": int(mask.sum()),
        "accel_x_std": float(np.std(ax, ddof=1)) if ax.size > 1 else 0.0,
        "accel_y_std": float(np.std(ay, ddof=1)) if ay.size > 1 else 0.0,
        "accel_z_std": float(np.std(az, ddof=1)) if az.size > 1 else 0.0,
        "accel_mag_mean": float(np.mean(amag)),
        "accel_mag_std": float(np.std(amag, ddof=1)) if amag.size > 1 else 0.0,
    }


def compute_gps_noise(
    records: List[RTTRecord],
    stationary_speed: float,
) -> Dict[str, float]:
    speeds = np.array([r.speed for r in records], dtype=float)
    lats = np.array([r.lat for r in records], dtype=float)
    lons = np.array([r.lon for r in records], dtype=float)
    mask = (speeds <= stationary_speed) & valid_gps_mask(lats, lons)
    if mask.sum() < 5:
        return {}

    lat = lats[mask]
    lon = lons[mask]
    mean_lat = float(np.mean(lat))
    lat_std_m = float(np.std(lat, ddof=1) * GPS_METERS_PER_DEG_LAT)
    lon_std_m = float(np.std(lon, ddof=1) * GPS_METERS_PER_DEG_LAT * math.cos(math.radians(mean_lat)))
    gps_std_m = float(math.sqrt((lat_std_m**2 + lon_std_m**2) / 2.0))
    return {
        "samples": int(mask.sum()),
        "lat_std_m": lat_std_m,
        "lon_std_m": lon_std_m,
        "gps_std_m": gps_std_m,
    }


def compute_mag_noise(
    records: List[RTTRecord],
    stationary_speed: float,
) -> Dict[str, float]:
    if not records:
        return {}

    speeds = np.array([r.speed for r in records], dtype=float)
    mask = speeds <= stationary_speed
    if mask.sum() < 5:
        return {}

    mx = np.array([r.mag_x for r in records], dtype=float)[mask]
    my = np.array([r.mag_y for r in records], dtype=float)[mask]
    mz = np.array([r.mag_z for r in records], dtype=float)[mask]
    if mx.size < 5:
        return {}

    std_x = float(np.std(mx, ddof=1)) if mx.size > 1 else 0.0
    std_y = float(np.std(my, ddof=1)) if my.size > 1 else 0.0
    std_z = float(np.std(mz, ddof=1)) if mz.size > 1 else 0.0
    mag_std_ut = float(np.mean([std_x, std_y, std_z]))

    heading_from_mag_deg = np.degrees(np.arctan2(my, mx))
    heading_std_deg = circular_std_deg(heading_from_mag_deg)

    return {
        "samples": int(mask.sum()),
        "mag_std_x": std_x,
        "mag_std_y": std_y,
        "mag_std_z": std_z,
        "mag_std_ut": mag_std_ut,
        "heading_std_deg": heading_std_deg if heading_std_deg is not None else float("nan"),
    }


def circular_std_deg(angles_deg: np.ndarray) -> Optional[float]:
    if angles_deg.size < 5:
        return None
    angles_rad = np.deg2rad(angles_deg)
    sin_mean = float(np.mean(np.sin(angles_rad)))
    cos_mean = float(np.mean(np.cos(angles_rad)))
    r = math.sqrt(sin_mean**2 + cos_mean**2)
    if r <= 0:
        return None
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def compute_heading_noise(records: List[RTTRecord], min_speed: float) -> Optional[float]:
    speeds = np.array([r.speed for r in records], dtype=float)
    headings = np.array([r.heading for r in records], dtype=float)
    mask = speeds >= min_speed
    if mask.sum() < 5:
        return None
    return circular_std_deg(headings[mask])


def write_loss_segments_csv(out_dir: Path, segments: List[Dict[str, float]]):
    if not segments:
        return
    out_path = out_dir / "loss_by_segment.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment_index", "start_s", "end_s", "count", "lost_count", "loss_rate"])
        for seg in segments:
            writer.writerow(
                [
                    seg["index"],
                    f"{seg['start_s']:.2f}",
                    f"{seg['end_s']:.2f}",
                    seg["count"],
                    seg["lost_count"],
                    f"{seg['loss_rate']:.6f}",
                ]
            )


def plot_rtt_distribution(out_dir: Path, rtt_ms: np.ndarray):
    if plt is None or rtt_ms.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rtt_ms, bins=40, color="#4c78a8", alpha=0.85)
    ax.set_title("RTT Distribution")
    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "rtt_distribution.png", dpi=150)
    plt.close(fig)


def plot_loss_over_time(out_dir: Path, segments: List[Dict[str, float]], threshold: Optional[float]):
    if plt is None or not segments:
        return
    times = [seg["start_s"] + (seg["end_s"] - seg["start_s"]) / 2.0 for seg in segments]
    rates = [seg["loss_rate"] * 100.0 for seg in segments]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, rates, color="#e45756", linewidth=1.5)
    ax.set_title("Packet Loss Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loss Rate (%)")
    if threshold is not None:
        ax.axhline(threshold * 100.0, color="#999999", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_over_time.png", dpi=150)
    plt.close(fig)


def plot_gps_track(out_dir: Path, records: List[RTTRecord]):
    if plt is None or not records:
        return
    lats = np.array([r.lat for r in records], dtype=float)
    lons = np.array([r.lon for r in records], dtype=float)
    lost = np.array([r.lost for r in records], dtype=int)
    mask = valid_gps_mask(lats, lons)
    if mask.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(lons[mask & (lost == 0)], lats[mask & (lost == 0)], s=6, color="#4c78a8", alpha=0.7, label="Received")
    ax.scatter(lons[mask & (lost == 1)], lats[mask & (lost == 1)], s=8, color="#f58518", alpha=0.8, label="Lost")
    ax.set_title("GPS Track")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "gps_track.png", dpi=150)
    plt.close(fig)


def build_emulator_params(
    one_way_ms: np.ndarray,
    loss_summary: Dict[str, float],
    loss_segments: List[Dict[str, float]],
    burst_stats: Dict[str, float],
    accel_noise: Dict[str, float],
    mag_noise: Dict[str, float],
    gps_noise: Dict[str, float],
    speed_std: Optional[float],
    heading_std: Optional[float],
    monitored_vehicles: List[str],
) -> Dict[str, Dict[str, float]]:
    if one_way_ms.size > 0:
        trimmed = one_way_ms
        if one_way_ms.size >= 20:
            p5, p95 = np.percentile(one_way_ms, [5, 95])
            trimmed = one_way_ms[(one_way_ms >= p5) & (one_way_ms <= p95)]
        base_ms = float(np.mean(trimmed))
        jitter_std_ms = float(np.std(trimmed, ddof=1)) if trimmed.size > 1 else 0.0
    else:
        base_ms = 15.0
        jitter_std_ms = 8.0

    baseline_rate = loss_summary.get("baseline_rate")
    p90_rate = loss_summary.get("p90")
    p95_rate = loss_summary.get("p95")
    if baseline_rate is None or math.isnan(baseline_rate):
        baseline_rate = 0.02
    if p90_rate is None or math.isnan(p90_rate):
        p90_rate = min(0.15, max(0.05, baseline_rate * 2))
    if p95_rate is None or math.isnan(p95_rate):
        p95_rate = min(0.30, max(0.10, baseline_rate * 3))

    burst_mean = burst_stats.get("mean_burst_length", 1.0)
    burst_max = burst_stats.get("max_burst_length", 0.0)
    burst_enabled = bool(burst_mean >= 1.5 or burst_max >= 3)
    loss_multiplier = 3.0
    if baseline_rate > 0:
        loss_multiplier = max(LOSS_MULTIPLIER_MIN, min(LOSS_MULTIPLIER_MAX, p90_rate / baseline_rate))
    max_loss_cap = min(LOSS_CAP_MAX, max(LOSS_CAP_MIN, p95_rate * 1.1))

    accel_std = accel_noise.get("accel_x_std")
    if accel_std is None:
        accel_std = 0.2
    accel_std = float(np.mean([
        accel_noise.get("accel_x_std", accel_std),
        accel_noise.get("accel_y_std", accel_std),
        accel_noise.get("accel_z_std", accel_std),
    ]))

    gps_std_m = gps_noise.get("gps_std_m", 5.0)
    mag_std_ut = mag_noise.get("mag_std_ut", 1.0)
    speed_std_ms = speed_std if speed_std is not None else 0.5
    heading_std_deg = heading_std if heading_std is not None else 3.0

    latency_range = [max(1.0, base_ms * 0.5), max(2.0, base_ms * 2.0)]
    jitter_range = [max(0.5, jitter_std_ms * 0.5), max(1.0, jitter_std_ms * 2.0)]
    loss_range = [max(0.0, baseline_rate * 0.5), min(0.6, baseline_rate * 2.0)]
    gps_range = [max(0.5, gps_std_m * 0.5), max(1.0, gps_std_m * 2.0)]

    threshold_1, threshold_2 = DEFAULT_DISTANCE_THRESHOLDS_M

    return {
        "latency": {
            "base_ms": base_ms,
            "distance_factor": 0.0,
            "jitter_std_ms": jitter_std_ms,
        },
        "packet_loss": {
            "base_rate": baseline_rate,
            "distance_threshold_1": threshold_1,
            "distance_threshold_2": threshold_2,
            "rate_tier_1": float(loss_summary.get("p50", baseline_rate)),
            "rate_tier_2": float(p90_rate),
            "rate_tier_3": float(p95_rate),
        },
        "burst_loss": {
            "enabled": burst_enabled,
            "mean_burst_length": float(burst_mean if burst_mean > 0 else 1.0),
            "loss_multiplier": float(loss_multiplier),
            "max_loss_cap": float(max_loss_cap),
        },
        "sensor_noise": {
            "gps_std_m": float(gps_std_m),
            "speed_std_ms": float(speed_std_ms),
            "accel_std_ms2": float(accel_std),
            "mag_std_ut": float(mag_std_ut),
            "heading_std_deg": float(heading_std_deg),
            "gyro_std_rad_s": 0.01,
        },
        "observation": {
            "staleness_threshold_ms": DEFAULT_STALENESS_MS,
            "monitored_vehicles": monitored_vehicles,
        },
        "domain_randomization": {
            "latency_range_ms": latency_range,
            "loss_rate_range": loss_range,
            "jitter_std_range_ms": jitter_range,
            "gps_noise_range_m": gps_range,
        },
    }


def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def compute_sequence_stats(records: List[RTTRecord]) -> Dict[str, int]:
    if not records:
        return {"unique_sequences": 0, "missing_sequences": 0}
    sequences = [r.sequence for r in records]
    unique = len(set(sequences))
    expected = max(sequences) - min(sequences) + 1
    missing = max(0, expected - unique)
    return {"unique_sequences": unique, "missing_sequences": missing}


def default_input_path() -> Path:
    cwd = Path.cwd()
    local = cwd / "rtt_log.csv"
    if local.exists():
        return local
    return (Path(__file__).resolve().parent / "../../../rtt_log.csv").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RTT characterization CSV and build emulator params.")
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Path to rtt_log.csv")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "../data/rtt_analysis")
    parser.add_argument("--params-out", type=Path, default=Path(__file__).resolve().parent / "../espnow_emulator/emulator_params_measured.json")
    parser.add_argument("--segment-seconds", type=int, default=10, help="Time bin size for loss analysis (seconds)")
    parser.add_argument("--stationary-speed-ms", type=float, default=0.5, help="Speed threshold for stationary noise")
    parser.add_argument("--heading-min-speed-ms", type=float, default=2.0, help="Speed threshold for heading noise")
    parser.add_argument("--monitored-vehicles", type=str, default=",".join(DEFAULT_MONITORED_VEHICLES), help="Comma-separated vehicle IDs")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    records = sort_records(load_records(args.input))
    if not records:
        print("No valid records loaded from CSV.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    send_times = np.array([r.send_time_ms for r in records], dtype=np.int64)
    lost_flags = np.array([r.lost for r in records], dtype=np.int8)
    rtt_ms = np.array([r.rtt_ms for r in records], dtype=float)
    received_mask = (lost_flags == 0) & (rtt_ms >= 0)
    rtt_ms = rtt_ms[received_mask]
    # Assumes symmetric RTT; see analysis_summary.json for this assumption.
    one_way_ms = rtt_ms / 2.0

    rtt_stats = compute_basic_stats(rtt_ms)
    rtt_stats.update(compute_percentiles(rtt_ms, [5, 25, 50, 75, 95, 99]))
    one_way_stats = compute_basic_stats(one_way_ms)
    one_way_stats.update(compute_percentiles(one_way_ms, [5, 50, 95]))

    loss_segments, duration_s = compute_loss_segments(records, args.segment_seconds)
    loss_summary = summarize_loss_segments(loss_segments)
    loss_rate = float(lost_flags.mean())
    loss_summary["overall_rate"] = loss_rate
    loss_summary["overall_lost"] = int(lost_flags.sum())
    loss_summary["overall_count"] = int(lost_flags.size)

    burst_stats = compute_burst_stats(records)
    imu_noise = compute_imu_noise(records, args.stationary_speed_ms)
    mag_noise = compute_mag_noise(records, args.stationary_speed_ms)
    gps_noise = compute_gps_noise(records, args.stationary_speed_ms)
    speed_noise = None
    if imu_noise:
        speeds = np.array([r.speed for r in records], dtype=float)
        speed_noise = float(np.std(speeds[speeds <= args.stationary_speed_ms], ddof=1)) if (speeds <= args.stationary_speed_ms).sum() > 1 else 0.0
    heading_noise = mag_noise.get("heading_std_deg") if mag_noise else None
    if heading_noise is not None and math.isnan(heading_noise):
        heading_noise = None
    if heading_noise is None:
        heading_noise = compute_heading_noise(records, args.heading_min_speed_ms)

    high_loss_zones = find_high_loss_zones(records, loss_segments, loss_summary, args.segment_seconds)

    monitored_vehicles = [v.strip() for v in args.monitored_vehicles.split(",") if v.strip()]
    if not monitored_vehicles:
        monitored_vehicles = DEFAULT_MONITORED_VEHICLES

    params = build_emulator_params(
        one_way_ms=one_way_ms,
        loss_summary=loss_summary,
        loss_segments=loss_segments,
        burst_stats=burst_stats,
        accel_noise=imu_noise,
        mag_noise=mag_noise,
        gps_noise=gps_noise,
        speed_std=speed_noise,
        heading_std=heading_noise,
        monitored_vehicles=monitored_vehicles,
    )

    write_json(args.params_out, params)
    write_loss_segments_csv(args.out_dir, loss_segments)

    summary = {
        "input": str(args.input),
        "duration_s": duration_s,
        "rtt_ms": rtt_stats,
        "one_way_latency_ms": one_way_stats,
        "loss": loss_summary,
        "burst_loss": burst_stats,
        "imu_noise": imu_noise,
        "mag_noise": mag_noise,
        "gps_noise": gps_noise,
        "speed_noise_std_ms": speed_noise,
        "heading_noise_std_deg": heading_noise,
        "high_loss_zones": high_loss_zones,
        "sequence": compute_sequence_stats(records),
        "assumptions": {
            "one_way_latency_symmetric": True,
            "distance_factor_zero": True,
        },
    }
    write_json(args.out_dir / "analysis_summary.json", summary)

    if not args.no_plots:
        plot_rtt_distribution(args.out_dir, rtt_ms)
        plot_loss_over_time(args.out_dir, loss_segments, loss_summary.get("high_loss_threshold"))
        plot_gps_track(args.out_dir, records)

    duration = (send_times.max() - send_times.min()) / 1000.0 if send_times.size else 0.0
    dt_ms = np.diff(send_times) if send_times.size > 1 else np.array([])
    median_dt = float(np.median(dt_ms)) if dt_ms.size else 0.0
    seq_stats = compute_sequence_stats(records)

    print("RTT Analysis Summary")
    print(f"  Rows: {lost_flags.size} | Received: {int(received_mask.sum())} | Lost: {int(lost_flags.sum())} ({loss_rate:.1%})")
    print(f"  Duration: {duration:.1f}s | Median send interval: {median_dt:.1f}ms")
    print(
        "  RTT ms: mean={mean:.2f}, std={std:.2f}, p50={p50:.2f}, p95={p95:.2f}, max={max:.2f}".format(
            **rtt_stats
        )
    )
    print(
        "  One-way ms: mean={mean:.2f}, std={std:.2f}, p50={p50:.2f}, p95={p95:.2f}".format(
            **one_way_stats
        )
    )
    if imu_noise:
        print(
            "  Accel noise std (m/s^2): ax={accel_x_std:.3f}, ay={accel_y_std:.3f}, az={accel_z_std:.3f}".format(
                **imu_noise
            )
        )
    if mag_noise:
        print(
            "  Mag noise std (uT): mx={mag_std_x:.3f}, my={mag_std_y:.3f}, mz={mag_std_z:.3f}".format(
                **mag_noise
            )
        )
    if gps_noise:
        print(f"  GPS noise std (m): {gps_noise['gps_std_m']:.2f}")
    if heading_noise is not None:
        print(f"  Heading noise std (deg): {heading_noise:.2f}")
    if high_loss_zones:
        print("  High-loss zones:")
        for zone in high_loss_zones:
            print(
                "    - {start_s:.0f}-{end_s:.0f}s | avg loss {loss_rate_avg:.1%} | max {loss_rate_max:.1%}".format(
                    **zone
                )
            )
    if seq_stats["missing_sequences"] > 0:
        print(
            f"  Warning: {seq_stats['missing_sequences']} sequences missing from log (potential additional loss).",
            file=sys.stderr,
        )

    if plt is None and not args.no_plots:
        print("  Plots skipped: matplotlib not available.")

    print(f"  Emulator params: {args.params_out}")
    print(f"  Outputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
