#!/usr/bin/env python3
"""
Analyze a 3-vehicle convoy recording (V001, V002, V003) and produce:
  - convoy_analysis_report.md
  - convoy_emulator_params.json
  - convoy_observations.npz
  - convoy_events.json
  - data_quality_summary.json
  - figures/*.png

Usage (from roadsense-v2v/ml with venv activated):
  python scripts/analyze_convoy_recording.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


VEHICLES = ("V001", "V002", "V003")
MIN_EXPECTED_COLUMNS = 16
SENDER_COLUMN_CANDIDATES = ("from_vehicle_id", "vehicle_id")
REQUIRED_NUMERIC_COLUMNS = (
    "timestamp_local_ms",
    "msg_timestamp",
    "lat",
    "lon",
    "speed",
    "heading",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
)
GPS_METERS_PER_DEG_LAT = 110540.0
GPS_METERS_PER_DEG_LON_AT_EQUATOR = 111320.0
MAX_PEERS = 8
STALENESS_MS = 500.0
PDR_WINDOW_MS = 10_000
PDR_STEP_MS = 1_000

CURRENT_EMULATOR_PARAMS = {
    "latency": {"base_ms": 3.84, "jitter_std_ms": 0.971},
    "packet_loss": {"base_rate": 0.21, "rate_tier_1": 0.31, "rate_tier_2": 0.57},
    "burst_loss": {"mean_burst_length": 2.106, "max_loss_cap": 0.717},
    "sensor_noise": {
        "gps_std_m": 1.506,
        "speed_std_ms": 0.021,
        "accel_std_ms2": 0.059,
        "gyro_std_rad_s": 0.01,
        "heading_std_deg": 0.580,
        "mag_std_ut": 0.612,
    },
}


@dataclass
class ParsedCSV:
    path: Path
    vehicle: str
    log_type: str
    third_col_name: str
    rows_total: int
    rows_valid: int
    malformed_rows: int
    missing_rows: int
    data: Dict[str, np.ndarray]


def _safe_int(value: str) -> int:
    return int(float(value))


def _safe_float(value: str) -> float:
    return float(value)


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def circular_std_deg(angles_deg: np.ndarray) -> Optional[float]:
    if angles_deg.size < 5:
        return None
    angles_rad = np.deg2rad(angles_deg)
    s = float(np.mean(np.sin(angles_rad)))
    c = float(np.mean(np.cos(angles_rad)))
    r = math.sqrt(s * s + c * c)
    if r <= 0.0:
        return None
    return float(math.degrees(math.sqrt(-2.0 * math.log(r))))


def basic_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    out = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    out["p50"] = float(p50)
    out["p95"] = float(p95)
    out["p99"] = float(p99)
    return out


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    win = max(3, int(window))
    if win % 2 == 0:
        win += 1
    if win >= values.size:
        return np.full_like(values, float(np.mean(values)))
    kernel = np.ones(win, dtype=float) / float(win)
    padded = np.pad(values, (win // 2, win // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_series(values: np.ndarray) -> np.ndarray:
    if values.size < 7:
        return moving_average(values, 5)
    win = min(31, values.size if values.size % 2 == 1 else values.size - 1)
    win = max(7, win)
    if win % 2 == 0:
        win -= 1
    try:
        return savgol_filter(values, window_length=win, polyorder=2, mode="interp")
    except Exception:
        return moving_average(values, 11)


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6_371_000.0
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.maximum(a, 0.0)))


def valid_gps(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (np.abs(lat) > 1e-9) & (np.abs(lon) > 1e-9)


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    lat_rad = math.radians(ref_lat)
    x = (lon - ref_lon) * GPS_METERS_PER_DEG_LON_AT_EQUATOR * math.cos(lat_rad)
    y = (lat - ref_lat) * GPS_METERS_PER_DEG_LAT
    return x, y


def write_json(path: Path, payload: Dict) -> None:
    def _sanitize(value):
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        if isinstance(value, tuple):
            return [_sanitize(v) for v in value]
        if isinstance(value, np.ndarray):
            return _sanitize(value.tolist())
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return None
            return float(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(payload), handle, indent=2, sort_keys=True, allow_nan=False)


def find_single_file(root: Path, vehicle: str, log_type: str) -> Path:
    pattern = f"{vehicle}_{log_type}_*.csv"
    matches = sorted((root / vehicle).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {log_type} CSV for {vehicle} in {root / vehicle}")
    return matches[-1]


def parse_convoy_csv(path: Path, vehicle: str, log_type: str) -> ParsedCSV:
    rows_total = 0
    rows_valid = 0
    malformed_rows = 0
    missing_rows = 0

    cols: Dict[str, List] = {
        "timestamp_local_ms": [],
        "msg_timestamp": [],
        "sender_id": [],
        "lat": [],
        "lon": [],
        "speed": [],
        "heading": [],
        "accel_x": [],
        "accel_y": [],
        "accel_z": [],
        "gyro_x": [],
        "gyro_y": [],
        "gyro_z": [],
        "mag_x": [],
        "mag_y": [],
        "mag_z": [],
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} has no header")
        if len(header) < MIN_EXPECTED_COLUMNS:
            raise ValueError(f"{path} header has {len(header)} columns, expected at least {MIN_EXPECTED_COLUMNS}")

        header_map = {name.strip(): idx for idx, name in enumerate(header)}
        sender_col_name = next((name for name in SENDER_COLUMN_CANDIDATES if name in header_map), None)
        if sender_col_name is None:
            raise ValueError(f"{path} missing sender id column (expected one of {SENDER_COLUMN_CANDIDATES})")

        missing_required = [name for name in REQUIRED_NUMERIC_COLUMNS if name not in header_map]
        if missing_required:
            raise ValueError(f"{path} missing required columns: {missing_required}")

        third_col_name = header[2]
        row_width = len(header)

        for row in reader:
            rows_total += 1
            if len(row) != row_width:
                malformed_rows += 1
                continue

            required_indexes = (
                header_map["timestamp_local_ms"],
                header_map["msg_timestamp"],
                header_map[sender_col_name],
                header_map["lat"],
                header_map["lon"],
                header_map["speed"],
                header_map["heading"],
                header_map["accel_x"],
                header_map["accel_y"],
                header_map["accel_z"],
                header_map["gyro_x"],
                header_map["gyro_y"],
                header_map["gyro_z"],
                header_map["mag_x"],
                header_map["mag_y"],
                header_map["mag_z"],
            )
            if any(row[idx].strip() == "" for idx in required_indexes):
                missing_rows += 1
                continue

            try:
                cols["timestamp_local_ms"].append(_safe_int(row[header_map["timestamp_local_ms"]]))
                cols["msg_timestamp"].append(_safe_int(row[header_map["msg_timestamp"]]))
                cols["sender_id"].append(row[header_map[sender_col_name]].strip())
                cols["lat"].append(_safe_float(row[header_map["lat"]]))
                cols["lon"].append(_safe_float(row[header_map["lon"]]))
                cols["speed"].append(_safe_float(row[header_map["speed"]]))
                cols["heading"].append(_safe_float(row[header_map["heading"]]))
                cols["accel_x"].append(_safe_float(row[header_map["accel_x"]]))
                cols["accel_y"].append(_safe_float(row[header_map["accel_y"]]))
                cols["accel_z"].append(_safe_float(row[header_map["accel_z"]]))
                cols["gyro_x"].append(_safe_float(row[header_map["gyro_x"]]))
                cols["gyro_y"].append(_safe_float(row[header_map["gyro_y"]]))
                cols["gyro_z"].append(_safe_float(row[header_map["gyro_z"]]))
                cols["mag_x"].append(_safe_float(row[header_map["mag_x"]]))
                cols["mag_y"].append(_safe_float(row[header_map["mag_y"]]))
                cols["mag_z"].append(_safe_float(row[header_map["mag_z"]]))
            except ValueError:
                malformed_rows += 1
                continue

            rows_valid += 1

    data = {
        "timestamp_local_ms": np.asarray(cols["timestamp_local_ms"], dtype=np.int64),
        "msg_timestamp": np.asarray(cols["msg_timestamp"], dtype=np.int64),
        "sender_id": np.asarray(cols["sender_id"], dtype=object),
        "lat": np.asarray(cols["lat"], dtype=float),
        "lon": np.asarray(cols["lon"], dtype=float),
        "speed": np.asarray(cols["speed"], dtype=float),
        "heading": np.asarray(cols["heading"], dtype=float),
        "accel_x": np.asarray(cols["accel_x"], dtype=float),
        "accel_y": np.asarray(cols["accel_y"], dtype=float),
        "accel_z": np.asarray(cols["accel_z"], dtype=float),
        "gyro_x": np.asarray(cols["gyro_x"], dtype=float),
        "gyro_y": np.asarray(cols["gyro_y"], dtype=float),
        "gyro_z": np.asarray(cols["gyro_z"], dtype=float),
        "mag_x": np.asarray(cols["mag_x"], dtype=float),
        "mag_y": np.asarray(cols["mag_y"], dtype=float),
        "mag_z": np.asarray(cols["mag_z"], dtype=float),
    }

    if data["timestamp_local_ms"].size > 1:
        order = np.argsort(data["timestamp_local_ms"])
        for key in list(data.keys()):
            data[key] = data[key][order]

    return ParsedCSV(
        path=path,
        vehicle=vehicle,
        log_type=log_type,
        third_col_name=third_col_name,
        rows_total=rows_total,
        rows_valid=rows_valid,
        malformed_rows=malformed_rows,
        missing_rows=missing_rows,
        data=data,
    )


def build_file_validation(parsed: ParsedCSV) -> Dict:
    d = parsed.data
    local_t = d["timestamp_local_ms"]
    lat = d["lat"]
    lon = d["lon"]
    speed = d["speed"]
    accel = np.vstack([d["accel_x"], d["accel_y"], d["accel_z"]]).T
    gyro = np.vstack([d["gyro_x"], d["gyro_y"], d["gyro_z"]]).T
    mag = np.vstack([d["mag_x"], d["mag_y"], d["mag_z"]]).T

    dup_ts = int(local_t.size - np.unique(local_t).size)
    dt = np.diff(local_t)
    gaps_gt_500 = int(np.sum(dt > 500)) if dt.size else 0

    gps_mask = valid_gps(lat, lon)
    gps_fix_pct = float(100.0 * gps_mask.mean()) if gps_mask.size else 0.0
    gps_jump_count = 0
    gps_jump_max_m = 0.0
    if gps_mask.sum() > 1:
        lat_v = lat[gps_mask]
        lon_v = lon[gps_mask]
        jumps = haversine_m(lat_v[:-1], lon_v[:-1], lat_v[1:], lon_v[1:])
        if jumps.size:
            gps_jump_count = int(np.sum(jumps > 50.0))
            gps_jump_max_m = float(np.max(jumps))

    accel_std = float(np.mean(np.std(accel, axis=0, ddof=1))) if accel.shape[0] > 1 else 0.0
    gyro_std = float(np.mean(np.std(gyro, axis=0, ddof=1))) if gyro.shape[0] > 1 else 0.0
    mag_std = float(np.mean(np.std(mag, axis=0, ddof=1))) if mag.shape[0] > 1 else 0.0

    imu_alive = bool(accel_std > 1e-3 and gyro_std > 1e-4)
    mag_alive = bool(np.any(np.abs(mag) > 1e-3) and mag_std > 1e-3)

    speed_gps_corr = None
    if local_t.size > 2 and gps_mask.sum() > 5:
        dt_s = np.diff(local_t) / 1000.0
        dist_m = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        valid_step = (dt_s > 1e-3) & valid_gps(lat[:-1], lon[:-1]) & valid_gps(lat[1:], lon[1:])
        if np.sum(valid_step) > 5:
            gps_speed = dist_m[valid_step] / dt_s[valid_step]
            speed_mid = speed[1:][valid_step]
            if np.std(gps_speed) > 1e-6 and np.std(speed_mid) > 1e-6:
                speed_gps_corr = float(np.corrcoef(gps_speed, speed_mid)[0, 1])

    if imu_alive and mag_alive and (speed_gps_corr is None or speed_gps_corr > 0.4):
        sensor_status = "OK"
    elif imu_alive or mag_alive:
        sensor_status = "WARN"
    else:
        sensor_status = "FAIL"

    return {
        "file": str(parsed.path),
        "vehicle": parsed.vehicle,
        "type": parsed.log_type,
        "rows_total": parsed.rows_total,
        "rows_valid": parsed.rows_valid,
        "malformed_rows": parsed.malformed_rows,
        "missing_rows": parsed.missing_rows,
        "duplicate_timestamps": dup_ts,
        "gaps_gt_500ms": gaps_gt_500,
        "time_range_ms": [int(local_t.min()) if local_t.size else None, int(local_t.max()) if local_t.size else None],
        "duration_s": float((local_t.max() - local_t.min()) / 1000.0) if local_t.size > 1 else 0.0,
        "gps_fix_pct": gps_fix_pct,
        "gps_jumps_gt_50m": gps_jump_count,
        "gps_jump_max_m": gps_jump_max_m,
        "accel_std_ms2": accel_std,
        "gyro_std_rad_s": gyro_std,
        "mag_std_ut": mag_std,
        "speed_gps_corr": speed_gps_corr,
        "sensor_status": sensor_status,
    }


def offset_sender_to_receiver(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> Optional[Dict[str, float]]:
    rx = logs[receiver]["rx"].data
    mask = rx["sender_id"] == sender
    if np.sum(mask) < 10:
        return None
    delta = rx["timestamp_local_ms"][mask].astype(float) - rx["msg_timestamp"][mask].astype(float)
    slope, intercept = np.polyfit(rx["msg_timestamp"][mask].astype(float), rx["timestamp_local_ms"][mask].astype(float), 1)
    return {
        "median_ms": float(np.median(delta)),
        "mean_ms": float(np.mean(delta)),
        "std_ms": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
        "drift_ppm": float((slope - 1.0) * 1e6),
        "fit_intercept_ms": float(intercept),
        "samples": int(delta.size),
    }


def align_vehicle_windows_to_v002(logs: Dict[str, Dict[str, ParsedCSV]]) -> Dict:
    offsets = {"V002": 0.0}
    for vehicle in ("V001", "V003"):
        o = offset_sender_to_receiver(logs, vehicle, "V002")
        offsets[vehicle] = float(o["median_ms"]) if o is not None else 0.0

    ranges = {}
    for vehicle in VEHICLES:
        tx = logs[vehicle]["tx"].data
        if tx["msg_timestamp"].size == 0:
            ranges[vehicle] = {"start_v002_ms": None, "end_v002_ms": None}
            continue
        if vehicle == "V002":
            start = float(tx["timestamp_local_ms"].min())
            end = float(tx["timestamp_local_ms"].max())
        else:
            start = float(tx["msg_timestamp"].min() + offsets[vehicle])
            end = float(tx["msg_timestamp"].max() + offsets[vehicle])
        ranges[vehicle] = {"start_v002_ms": start, "end_v002_ms": end}

    starts = [v["start_v002_ms"] for v in ranges.values() if v["start_v002_ms"] is not None]
    ends = [v["end_v002_ms"] for v in ranges.values() if v["end_v002_ms"] is not None]
    overlap_start = float(max(starts)) if starts else None
    overlap_end = float(min(ends)) if ends else None
    overlap_duration_s = 0.0
    if overlap_start is not None and overlap_end is not None and overlap_end > overlap_start:
        overlap_duration_s = (overlap_end - overlap_start) / 1000.0

    return {
        "offsets_to_v002_ms": offsets,
        "ranges_in_v002_clock": ranges,
        "overlap_window_v002_ms": [overlap_start, overlap_end],
        "overlap_duration_s": overlap_duration_s,
    }


def link_received_set(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> np.ndarray:
    rx = logs[receiver]["rx"].data
    mask = rx["sender_id"] == sender
    if np.sum(mask) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(rx["msg_timestamp"][mask].astype(np.int64))


def compute_link_pdr(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> Dict:
    tx_msg = logs[sender]["tx"].data["msg_timestamp"].astype(np.int64)
    tx_msg = np.unique(tx_msg)
    rx_msg = link_received_set(logs, sender, receiver)

    if tx_msg.size == 0 or rx_msg.size == 0:
        return {
            "link": f"{sender}->{receiver}",
            "sender": sender,
            "receiver": receiver,
            "overlap_sender_ms": [None, None],
            "tx_in_overlap": int(tx_msg.size),
            "rx_in_overlap": int(rx_msg.size),
            "matched": 0,
            "pdr": 0.0 if tx_msg.size > 0 else float("nan"),
            "loss_rate": 1.0 if tx_msg.size > 0 else float("nan"),
            "missing": int(tx_msg.size),
        }

    overlap_start = int(max(tx_msg.min(), rx_msg.min()))
    overlap_end = int(min(tx_msg.max(), rx_msg.max()))
    if overlap_end < overlap_start:
        return {
            "link": f"{sender}->{receiver}",
            "sender": sender,
            "receiver": receiver,
            "overlap_sender_ms": [None, None],
            "tx_in_overlap": 0,
            "rx_in_overlap": 0,
            "matched": 0,
            "pdr": float("nan"),
            "loss_rate": float("nan"),
            "missing": 0,
        }

    tx_overlap = tx_msg[(tx_msg >= overlap_start) & (tx_msg <= overlap_end)]
    rx_overlap = rx_msg[(rx_msg >= overlap_start) & (rx_msg <= overlap_end)]
    matched_mask = np.isin(tx_overlap, rx_overlap)
    matched = int(np.sum(matched_mask))
    total = int(tx_overlap.size)
    pdr = float(matched / total) if total > 0 else float("nan")

    return {
        "link": f"{sender}->{receiver}",
        "sender": sender,
        "receiver": receiver,
        "overlap_sender_ms": [overlap_start, overlap_end],
        "tx_in_overlap": total,
        "rx_in_overlap": int(rx_overlap.size),
        "matched": matched,
        "pdr": pdr,
        "loss_rate": float(1.0 - pdr) if total > 0 else float("nan"),
        "missing": int(total - matched),
    }


def compute_pdr_windows(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> Dict[str, np.ndarray]:
    tx_msg = np.unique(logs[sender]["tx"].data["msg_timestamp"].astype(np.int64))
    rx_msg = link_received_set(logs, sender, receiver)
    if tx_msg.size == 0 or rx_msg.size == 0:
        return {"t_s": np.array([], dtype=float), "pdr": np.array([], dtype=float)}

    start = max(int(tx_msg.min()), int(rx_msg.min()))
    end = min(int(tx_msg.max()), int(rx_msg.max()))
    if end <= start:
        return {"t_s": np.array([], dtype=float), "pdr": np.array([], dtype=float)}

    centers = np.arange(start + PDR_WINDOW_MS // 2, end - PDR_WINDOW_MS // 2 + 1, PDR_STEP_MS, dtype=np.int64)
    pdr_values: List[float] = []
    t_values: List[float] = []
    rx_set = set(int(v) for v in rx_msg.tolist())

    for center in centers:
        lo = center - PDR_WINDOW_MS // 2
        hi = center + PDR_WINDOW_MS // 2
        tx_win = tx_msg[(tx_msg >= lo) & (tx_msg <= hi)]
        if tx_win.size == 0:
            continue
        matched = sum(1 for t in tx_win.tolist() if int(t) in rx_set)
        pdr_values.append(matched / float(tx_win.size))
        t_values.append((center - start) / 1000.0)

    return {"t_s": np.asarray(t_values, dtype=float), "pdr": np.asarray(pdr_values, dtype=float)}


def compute_latency_by_link(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> Dict:
    rx = logs[receiver]["rx"].data
    mask = rx["sender_id"] == sender
    if np.sum(mask) == 0:
        empty = np.array([], dtype=float)
        return {
            "link": f"{sender}->{receiver}",
            "raw_ms": basic_stats(empty),
            "relative_ms": basic_stats(empty),
            "relative_ms_series": empty,
            "raw_ms_series": empty,
        }
    raw = rx["timestamp_local_ms"][mask].astype(float) - rx["msg_timestamp"][mask].astype(float)
    # Relative latency removes constant clock offset per link.
    relative = raw - np.median(raw)
    # Keep a non-negative shifted view for emulator base/jitter estimation.
    relative = relative - float(np.percentile(relative, 1))
    return {
        "link": f"{sender}->{receiver}",
        "raw_ms": basic_stats(raw),
        "relative_ms": basic_stats(relative),
        "relative_ms_series": relative,
        "raw_ms_series": raw,
    }


def burst_lengths(loss_flags: np.ndarray) -> List[int]:
    bursts: List[int] = []
    run = 0
    for lost in loss_flags.astype(int).tolist():
        if lost:
            run += 1
        elif run > 0:
            bursts.append(run)
            run = 0
    if run > 0:
        bursts.append(run)
    return bursts


def compute_burst_by_link(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> Dict:
    pdr = compute_link_pdr(logs, sender, receiver)
    ov = pdr["overlap_sender_ms"]
    tx_msg = np.unique(logs[sender]["tx"].data["msg_timestamp"].astype(np.int64))
    if ov[0] is None or ov[1] is None:
        return {
            "link": f"{sender}->{receiver}",
            "burst_count": 0,
            "mean_burst_length": 0.0,
            "max_burst_length": 0,
            "loss_sequence_len": 0,
        }
    tx_overlap = tx_msg[(tx_msg >= ov[0]) & (tx_msg <= ov[1])]
    rx_set = set(link_received_set(logs, sender, receiver).tolist())
    lost = np.array([1 if int(t) not in rx_set else 0 for t in tx_overlap], dtype=np.int8)
    bursts = burst_lengths(lost)
    return {
        "link": f"{sender}->{receiver}",
        "burst_count": int(len(bursts)),
        "mean_burst_length": float(np.mean(bursts)) if bursts else 0.0,
        "max_burst_length": int(np.max(bursts)) if bursts else 0,
        "loss_sequence_len": int(lost.size),
    }


def interpolate_receiver_position_in_sender_clock(
    logs: Dict[str, Dict[str, ParsedCSV]],
    sender: str,
    receiver: str,
) -> Optional[Dict[str, np.ndarray]]:
    sender_tx = logs[sender]["tx"].data
    recv_tx = logs[receiver]["tx"].data

    # Convert receiver clock -> sender clock using reverse-link offset.
    reverse = offset_sender_to_receiver(logs, receiver, sender)
    if reverse is None:
        return None
    recv_to_sender_ms = reverse["median_ms"]
    recv_shifted = recv_tx["msg_timestamp"].astype(float) + float(recv_to_sender_ms)

    sender_t = sender_tx["msg_timestamp"].astype(float)
    if sender_t.size < 2 or recv_shifted.size < 2:
        return None

    recv_valid = valid_gps(recv_tx["lat"], recv_tx["lon"])
    send_valid = valid_gps(sender_tx["lat"], sender_tx["lon"])
    if recv_valid.sum() < 2 or send_valid.sum() < 2:
        return None

    order = np.argsort(recv_shifted)
    recv_shifted = recv_shifted[order]
    recv_lat = recv_tx["lat"][order]
    recv_lon = recv_tx["lon"][order]
    recv_valid = recv_valid[order]

    valid_recv_time = recv_shifted[recv_valid]
    valid_recv_lat = recv_lat[recv_valid]
    valid_recv_lon = recv_lon[recv_valid]
    if valid_recv_time.size < 2:
        return None

    in_range = (sender_t >= valid_recv_time.min()) & (sender_t <= valid_recv_time.max()) & send_valid
    if np.sum(in_range) < 5:
        return None

    interp_lat = np.interp(sender_t[in_range], valid_recv_time, valid_recv_lat)
    interp_lon = np.interp(sender_t[in_range], valid_recv_time, valid_recv_lon)
    dist = haversine_m(
        sender_tx["lat"][in_range],
        sender_tx["lon"][in_range],
        interp_lat,
        interp_lon,
    )
    return {
        "sender_msg_timestamp": sender_t[in_range].astype(np.int64),
        "distance_m": dist.astype(float),
    }


def compute_distance_binned_loss(logs: Dict[str, Dict[str, ParsedCSV]], sender: str, receiver: str) -> Dict:
    interp = interpolate_receiver_position_in_sender_clock(logs, sender, receiver)
    if interp is None:
        return {"link": f"{sender}->{receiver}", "bins": {}}

    tx_ts = interp["sender_msg_timestamp"]
    dist = interp["distance_m"]
    rx_set = set(link_received_set(logs, sender, receiver).tolist())
    received = np.array([int(ts) in rx_set for ts in tx_ts], dtype=bool)

    bins = [
        ("0_20m", 0.0, 20.0),
        ("20_50m", 20.0, 50.0),
        ("50_100m", 50.0, 100.0),
        ("100m_plus", 100.0, float("inf")),
    ]
    out = {}
    for name, lo, hi in bins:
        mask = (dist >= lo) & (dist < hi)
        total = int(np.sum(mask))
        if total == 0:
            out[name] = {"count": 0, "pdr": float("nan"), "loss_rate": float("nan")}
            continue
        matched = int(np.sum(received[mask]))
        pdr = matched / float(total)
        out[name] = {"count": total, "pdr": pdr, "loss_rate": 1.0 - pdr}

    return {"link": f"{sender}->{receiver}", "bins": out}


def classify_segments(speed: np.ndarray, accel_x: np.ndarray) -> np.ndarray:
    labels = np.full(speed.shape[0], "other", dtype=object)
    labels[accel_x < -1.0] = "braking"
    labels[accel_x > 1.0] = "accelerating"
    labels[speed < 0.5] = "stationary"
    cruise_mask = (speed > 5.0) & (np.abs(accel_x) < 0.5)
    labels[cruise_mask] = "cruising"
    # Keep braking/acceleration precedence.
    labels[accel_x < -1.0] = "braking"
    labels[accel_x > 1.0] = "accelerating"
    return labels


def residual_std(values: np.ndarray, detrend: bool) -> float:
    if values.size < 2:
        return float("nan")
    if detrend:
        values = values - moving_average(values, 21)
    return float(np.std(values, ddof=1))


def compute_sensor_noise_for_vehicle(parsed_tx: ParsedCSV) -> Dict:
    d = parsed_tx.data
    labels = classify_segments(d["speed"], d["accel_x"])
    out: Dict[str, Dict[str, float]] = {}
    gps_mask = valid_gps(d["lat"], d["lon"])

    # Smoothed trajectory for GPS residual computation.
    lat_smooth = np.full_like(d["lat"], np.nan, dtype=float)
    lon_smooth = np.full_like(d["lon"], np.nan, dtype=float)
    if np.sum(gps_mask) > 6:
        lat_smooth[gps_mask] = smooth_series(d["lat"][gps_mask])
        lon_smooth[gps_mask] = smooth_series(d["lon"][gps_mask])

    for seg in ("stationary", "cruising", "braking", "accelerating"):
        mask = labels == seg
        if np.sum(mask) < 5:
            continue

        seg_dict: Dict[str, float] = {"samples": int(np.sum(mask))}

        gps_seg = mask & gps_mask & np.isfinite(lat_smooth) & np.isfinite(lon_smooth)
        if np.sum(gps_seg) >= 5:
            lat_res = d["lat"][gps_seg] - lat_smooth[gps_seg]
            lon_res = d["lon"][gps_seg] - lon_smooth[gps_seg]
            mean_lat = float(np.mean(d["lat"][gps_seg]))
            lat_std_m = float(np.std(lat_res, ddof=1) * GPS_METERS_PER_DEG_LAT)
            lon_std_m = float(
                np.std(lon_res, ddof=1)
                * GPS_METERS_PER_DEG_LON_AT_EQUATOR
                * math.cos(math.radians(mean_lat))
            )
            seg_dict["gps_std_m"] = float(math.sqrt((lat_std_m * lat_std_m + lon_std_m * lon_std_m) / 2.0))

        detrend = seg in ("cruising", "braking", "accelerating")
        seg_dict["accel_std_ms2"] = float(
            np.nanmean(
                [
                    residual_std(d["accel_x"][mask], detrend=detrend),
                    residual_std(d["accel_y"][mask], detrend=detrend),
                    residual_std(d["accel_z"][mask], detrend=detrend),
                ]
            )
        )
        seg_dict["gyro_std_rad_s"] = float(
            np.nanmean(
                [
                    residual_std(d["gyro_x"][mask], detrend=detrend),
                    residual_std(d["gyro_y"][mask], detrend=detrend),
                    residual_std(d["gyro_z"][mask], detrend=detrend),
                ]
            )
        )
        seg_dict["mag_std_ut"] = float(
            np.nanmean(
                [
                    residual_std(d["mag_x"][mask], detrend=False),
                    residual_std(d["mag_y"][mask], detrend=False),
                    residual_std(d["mag_z"][mask], detrend=False),
                ]
            )
        )
        seg_dict["speed_std_ms"] = residual_std(d["speed"][mask], detrend=detrend)

        heading_mag_deg = np.degrees(np.arctan2(d["mag_y"][mask], d["mag_x"][mask]))
        mag_heading_std = circular_std_deg(heading_mag_deg)
        if mag_heading_std is not None:
            seg_dict["heading_std_deg"] = float(mag_heading_std)

        out[seg] = seg_dict

    return out


def aggregate_sensor_noise(per_vehicle: Dict[str, Dict]) -> Dict:
    per_segment: Dict[str, Dict[str, List[float]]] = {}
    for vehicle_dict in per_vehicle.values():
        for segment, seg_stats in vehicle_dict.items():
            if segment not in per_segment:
                per_segment[segment] = {}
            for key, value in seg_stats.items():
                if key == "samples":
                    continue
                if np.isnan(value):
                    continue
                per_segment[segment].setdefault(key, []).append(float(value))

    agg: Dict[str, Dict[str, float]] = {}
    for segment, metrics in per_segment.items():
        agg[segment] = {}
        for key, values in metrics.items():
            agg[segment][key] = float(np.mean(values)) if values else float("nan")
    return agg


def detect_braking_events(v001_tx: ParsedCSV, max_light_events: int = 2) -> Dict:
    d = v001_tx.data
    t = d["msg_timestamp"].astype(np.int64)
    accel = moving_average(d["accel_x"], 5)
    speed = d["speed"]
    if t.size == 0:
        return {"events": [], "hard_event": None, "stationary_tail": None}

    braking_mask = accel < -0.8
    idx = np.flatnonzero(braking_mask)
    segments: List[Tuple[int, int]] = []
    if idx.size > 0:
        start = int(idx[0])
        prev = int(idx[0])
        for val in idx[1:]:
            cur = int(val)
            if cur <= prev + 6:  # merge small gaps (<~0.6s at 10Hz)
                prev = cur
            else:
                segments.append((start, prev))
                start = cur
                prev = cur
        segments.append((start, prev))

    events: List[Dict] = []
    for s, e in segments:
        if e - s + 1 < 3:
            continue
        t0 = int(t[s])
        t1 = int(t[e])
        dt_s = max(0.0, (t1 - t0) / 1000.0)
        min_ax = float(np.min(accel[s : e + 1]))
        mean_ax = float(np.mean(accel[s : e + 1]))
        speed_drop = float(speed[s] - speed[e]) if e > s else 0.0
        events.append(
            {
                "start_msg_ms": t0,
                "end_msg_ms": t1,
                "duration_s": dt_s,
                "min_accel_x_ms2": min_ax,
                "mean_accel_x_ms2": mean_ax,
                "speed_drop_ms": speed_drop,
            }
        )

    if not events:
        return {"events": [], "hard_event": None, "stationary_tail": None}

    hard_idx = int(np.argmin([e["min_accel_x_ms2"] for e in events]))
    hard_event = dict(events[hard_idx])
    hard_event["event_type"] = "hard_braking"

    hard_start = hard_event["start_msg_ms"]
    light_candidates = [
        e for i, e in enumerate(events) if i != hard_idx and e["start_msg_ms"] < hard_start and e["min_accel_x_ms2"] < -0.8
    ]
    light_candidates = sorted(light_candidates, key=lambda x: x["start_msg_ms"])[:max_light_events]
    for e in light_candidates:
        e["event_type"] = "light_braking"

    final_events = light_candidates + [hard_event]
    final_events = sorted(final_events, key=lambda x: x["start_msg_ms"])

    # Stationary tail near the end.
    stationary = speed < 0.5
    stationary_tail = None
    if np.sum(stationary) > 5:
        rev_idx = np.flatnonzero(stationary[::-1])
        if rev_idx.size > 0:
            end_idx = int(len(stationary) - 1 - rev_idx[0])
            start_idx = end_idx
            while start_idx > 0 and stationary[start_idx - 1]:
                start_idx -= 1
            duration_s = max(0.0, (int(t[end_idx]) - int(t[start_idx])) / 1000.0)
            if duration_s >= 20.0:
                stationary_tail = {
                    "start_msg_ms": int(t[start_idx]),
                    "end_msg_ms": int(t[end_idx]),
                    "duration_s": duration_s,
                }

    return {
        "events": final_events,
        "hard_event": hard_event,
        "stationary_tail": stationary_tail,
        "all_candidates": events,
    }


def build_observations_v002(
    logs: Dict[str, Dict[str, ParsedCSV]],
    ref_lat: float,
    ref_lon: float,
) -> Dict[str, np.ndarray]:
    ego_tx = logs["V002"]["tx"].data
    rx = logs["V002"]["rx"].data

    ego_t = ego_tx["timestamp_local_ms"].astype(np.int64)
    n = ego_t.size
    ego_lat = ego_tx["lat"]
    ego_lon = ego_tx["lon"]
    ego_speed = ego_tx["speed"]
    ego_heading_deg = ego_tx["heading"]
    ego_accel_x = ego_tx["accel_x"]
    ego_x, ego_y = latlon_to_xy(ego_lat, ego_lon, ref_lat, ref_lon)

    peer_msgs: Dict[str, Dict[str, np.ndarray]] = {}
    for sender in ("V001", "V003"):
        m = rx["sender_id"] == sender
        order = np.argsort(rx["timestamp_local_ms"][m])
        peer_msgs[sender] = {
            "rx_t": rx["timestamp_local_ms"][m][order].astype(np.int64),
            "lat": rx["lat"][m][order].astype(float),
            "lon": rx["lon"][m][order].astype(float),
            "speed": rx["speed"][m][order].astype(float),
            "heading": rx["heading"][m][order].astype(float),
            "accel_x": rx["accel_x"][m][order].astype(float),
        }

    ego = np.zeros((n, 4), dtype=np.float32)
    peers = np.zeros((n, MAX_PEERS, 6), dtype=np.float32)
    peer_mask = np.zeros((n, MAX_PEERS), dtype=np.float32)
    peer_count = np.zeros((n,), dtype=np.int32)

    for i in range(n):
        t = int(ego_t[i])
        ego_heading_rad = wrap_angle_rad(np.array([math.radians(ego_heading_deg[i])], dtype=float))[0]
        ego[i, 0] = float(ego_speed[i] / 30.0)
        ego[i, 1] = float(ego_accel_x[i] / 10.0)
        ego[i, 2] = float(ego_heading_rad / math.pi)

        valid_peer_idx = 0
        for sender in ("V001", "V003"):
            msg = peer_msgs[sender]
            if msg["rx_t"].size == 0:
                continue
            idx = int(np.searchsorted(msg["rx_t"], t, side="right") - 1)
            if idx < 0:
                continue
            age = float(t - int(msg["rx_t"][idx]))
            if age > STALENESS_MS:
                continue

            px, py = latlon_to_xy(
                np.array([msg["lat"][idx]], dtype=float),
                np.array([msg["lon"][idx]], dtype=float),
                ref_lat,
                ref_lon,
            )
            dx = float(px[0] - ego_x[i])
            dy = float(py[0] - ego_y[i])

            cos_h = math.cos(-ego_heading_rad)
            sin_h = math.sin(-ego_heading_rad)
            rel_x = dx * cos_h - dy * sin_h
            rel_y = dx * sin_h + dy * cos_h

            rel_speed = float(msg["speed"][idx] - ego_speed[i])
            peer_heading_rad = math.radians(float(msg["heading"][idx]))
            rel_heading = wrap_angle_rad(np.array([peer_heading_rad - ego_heading_rad], dtype=float))[0]

            if valid_peer_idx < MAX_PEERS:
                peers[i, valid_peer_idx, 0] = float(rel_x / 100.0)
                peers[i, valid_peer_idx, 1] = float(rel_y / 100.0)
                peers[i, valid_peer_idx, 2] = float(rel_speed / 30.0)
                peers[i, valid_peer_idx, 3] = float(rel_heading / math.pi)
                peers[i, valid_peer_idx, 4] = float(msg["accel_x"][idx] / 10.0)
                peers[i, valid_peer_idx, 5] = float(age / STALENESS_MS)
                peer_mask[i, valid_peer_idx] = 1.0
                valid_peer_idx += 1

        peer_count[i] = valid_peer_idx
        ego[i, 3] = float(valid_peer_idx / MAX_PEERS)

    return {
        "timestamps_local_ms": ego_t,
        "ego": ego,
        "peers": peers,
        "peer_mask": peer_mask,
        "peer_count": peer_count,
    }


def summarize_observation_ranges(obs: Dict[str, np.ndarray]) -> Dict:
    ego = obs["ego"]
    peers = obs["peers"]
    mask = obs["peer_mask"] > 0.5

    fields = {
        "ego_speed_norm": ego[:, 0],
        "ego_accel_norm": ego[:, 1],
        "ego_heading_norm": ego[:, 2],
        "peer_count_norm": ego[:, 3],
    }

    peer_fields = {
        "peer_rel_x_norm": peers[:, :, 0][mask],
        "peer_rel_y_norm": peers[:, :, 1][mask],
        "peer_rel_speed_norm": peers[:, :, 2][mask],
        "peer_rel_heading_norm": peers[:, :, 3][mask],
        "peer_accel_norm": peers[:, :, 4][mask],
        "peer_age_norm": peers[:, :, 5][mask],
    }
    fields.update(peer_fields)

    ranges = {}
    for name, vals in fields.items():
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            ranges[name] = {"count": 0}
            continue
        ranges[name] = {
            "count": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "pct_outside_minus1_to_1": float(np.mean((vals < -1.0) | (vals > 1.0)) * 100.0),
            "pct_outside_minus1p5_to_1p5": float(np.mean((vals < -1.5) | (vals > 1.5)) * 100.0),
        }
    return ranges


def compute_zero_peer_windows(obs: Dict[str, np.ndarray]) -> Dict:
    t = obs["timestamps_local_ms"]
    zero = obs["peer_count"] == 0
    if t.size == 0:
        return {"count": 0, "windows": []}
    windows = []
    in_run = False
    start_i = 0
    for i in range(zero.size):
        if zero[i] and not in_run:
            in_run = True
            start_i = i
        if in_run and (i == zero.size - 1 or not zero[i + 1]):
            end_i = i
            duration_ms = int(t[end_i] - t[start_i])
            if duration_ms > STALENESS_MS:
                windows.append(
                    {
                        "start_local_ms": int(t[start_i]),
                        "end_local_ms": int(t[end_i]),
                        "duration_ms": duration_ms,
                    }
                )
            in_run = False
    return {"count": len(windows), "windows": windows}


def discover_default_model(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "ml/models/runs/run_20260115_210446/model_final.zip",
        repo_root / "ml/models/runs/cloud_prod_001/model_final.zip",
        repo_root / "ml/models/runs/rs_dataset_test/model_final.zip",
        repo_root / "ml/models/runs/rs_smoke/model_final.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checkpoints = sorted((repo_root / "ml/models/runs/cloud_prod_001/checkpoints").glob("deep_sets_*_steps.zip"))
    if checkpoints:
        def step_num(path: Path) -> int:
            stem = path.stem
            parts = stem.split("_")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0

        return sorted(checkpoints, key=step_num)[-1]
    return None


def run_optional_model_inference(
    model_path: Optional[Path],
    obs: Dict[str, np.ndarray],
    hard_event_v002_ms: Optional[Tuple[float, float]],
) -> Dict:
    if model_path is None or not model_path.exists():
        return {
            "model_path": None,
            "loaded": False,
            "reason": "model_not_found",
        }

    # Ensure `import ml.*` works when loading SB3 models saved with custom extractors.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        return {
            "model_path": str(model_path),
            "loaded": False,
            "reason": f"sb3_import_failed: {exc}",
        }

    try:
        import torch
    except Exception:
        torch = None

    try:
        model = PPO.load(str(model_path))
    except Exception as exc:
        return {
            "model_path": str(model_path),
            "loaded": False,
            "reason": f"model_load_failed: {exc}",
        }

    n = obs["ego"].shape[0]
    actions = np.zeros((n,), dtype=np.int32)
    probs = np.full((n, 4), np.nan, dtype=float)

    for i in range(n):
        single_obs = {
            "ego": obs["ego"][i],
            "peers": obs["peers"][i],
            "peer_mask": obs["peer_mask"][i],
        }
        action, _ = model.predict(single_obs, deterministic=True)
        action_i = int(action.item()) if hasattr(action, "item") else int(action)
        actions[i] = action_i

        if torch is not None:
            try:
                obs_tensor, _ = model.policy.obs_to_tensor(single_obs)
                with torch.no_grad():
                    dist = model.policy.get_distribution(obs_tensor)
                    p = dist.distribution.probs.detach().cpu().numpy()
                if p.ndim == 2 and p.shape[1] == 4:
                    probs[i, :] = p[0]
            except Exception:
                pass

    counts = {str(i): int(np.sum(actions == i)) for i in range(4)}
    fractions = {k: (v / float(n) if n > 0 else float("nan")) for k, v in counts.items()}

    event_summary = None
    if hard_event_v002_ms is not None:
        t = obs["timestamps_local_ms"].astype(float)
        lo, hi = hard_event_v002_ms
        mask = (t >= lo) & (t <= hi)
        if np.sum(mask) > 0:
            ev_actions = actions[mask]
            event_summary = {
                "samples": int(np.sum(mask)),
                "action_counts": {str(i): int(np.sum(ev_actions == i)) for i in range(4)},
                "maintain_rate": float(np.mean(ev_actions == 0)),
                "brake_or_higher_rate": float(np.mean(ev_actions >= 2)),
                "emergency_rate": float(np.mean(ev_actions == 3)),
            }

    return {
        "model_path": str(model_path),
        "loaded": True,
        "action_counts": counts,
        "action_fraction": fractions,
        "hard_event_window_summary": event_summary,
        "actions": actions,
        "probs": probs,
    }


def extract_trajectories(
    logs: Dict[str, Dict[str, ParsedCSV]],
    out_dir: Path,
    ref_lat: float,
    ref_lon: float,
) -> Dict[str, Dict]:
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict] = {}

    for vehicle in VEHICLES:
        tx = logs[vehicle]["tx"].data
        t = tx["timestamp_local_ms"].astype(np.int64)
        if t.size < 2:
            continue
        t0 = int(t.min())
        t1 = int(t.max())
        t_uniform = np.arange(t0, t1 + 1, 100, dtype=np.int64)

        lat = np.interp(t_uniform, t, tx["lat"])
        lon = np.interp(t_uniform, t, tx["lon"])
        speed = np.interp(t_uniform, t, tx["speed"])
        heading = np.interp(t_uniform, t, tx["heading"])

        lat_s = smooth_series(lat)
        lon_s = smooth_series(lon)
        speed_s = moving_average(speed, 9)
        heading_s = moving_average(heading, 9)

        x, y = latlon_to_xy(lat_s, lon_s, ref_lat, ref_lon)
        path = traj_dir / f"{vehicle}_trajectory.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_local_ms", "t_rel_s", "lat", "lon", "x_m", "y_m", "speed_ms", "heading_deg"])
            for i in range(t_uniform.size):
                writer.writerow(
                    [
                        int(t_uniform[i]),
                        f"{(t_uniform[i] - t0) / 1000.0:.3f}",
                        f"{lat_s[i]:.7f}",
                        f"{lon_s[i]:.7f}",
                        f"{x[i]:.3f}",
                        f"{y[i]:.3f}",
                        f"{speed_s[i]:.3f}",
                        f"{heading_s[i]:.3f}",
                    ]
                )

        out[vehicle] = {
            "csv": str(path),
            "samples": int(t_uniform.size),
            "duration_s": float((t1 - t0) / 1000.0),
            "avg_speed_ms": float(np.mean(speed_s)),
        }
    return out


def formation_analysis(
    logs: Dict[str, Dict[str, ParsedCSV]],
    offsets_to_v002: Dict[str, float],
) -> Dict:
    v2 = logs["V002"]["tx"].data
    t2 = v2["timestamp_local_ms"].astype(float)
    if t2.size < 2:
        return {"stats": {}, "series": {}}

    def interp_vehicle(vehicle: str) -> Tuple[np.ndarray, np.ndarray]:
        tx = logs[vehicle]["tx"].data
        if vehicle == "V002":
            t = tx["timestamp_local_ms"].astype(float)
        else:
            t = tx["msg_timestamp"].astype(float) + float(offsets_to_v002.get(vehicle, 0.0))
        lat = np.interp(t2, t, tx["lat"], left=np.nan, right=np.nan)
        lon = np.interp(t2, t, tx["lon"], left=np.nan, right=np.nan)
        return lat, lon

    lat1, lon1 = interp_vehicle("V001")
    lat2, lon2 = interp_vehicle("V002")
    lat3, lon3 = interp_vehicle("V003")

    d12 = haversine_m(lat1, lon1, lat2, lon2)
    d23 = haversine_m(lat2, lon2, lat3, lon3)
    d13 = haversine_m(lat1, lon1, lat3, lon3)

    def series_stats(values: np.ndarray) -> Dict[str, float]:
        v = values[np.isfinite(values)]
        if v.size == 0:
            return {"count": 0}
        return {
            "count": int(v.size),
            "mean_m": float(np.mean(v)),
            "median_m": float(np.median(v)),
            "p95_m": float(np.percentile(v, 95)),
            "min_m": float(np.min(v)),
            "max_m": float(np.max(v)),
        }

    stats = {
        "V001_V002": series_stats(d12),
        "V002_V003": series_stats(d23),
        "V001_V003": series_stats(d13),
    }

    pair_stack = np.vstack([d12, d23])
    valid_cols = np.any(np.isfinite(pair_stack), axis=0)
    if np.any(valid_cols):
        convoy_mean = np.nanmean(pair_stack[:, valid_cols], axis=0)
        avg_spacing = float(np.nanmean(convoy_mean))
    else:
        avg_spacing = float("nan")
    if avg_spacing <= 15.0:
        formation_class = "tight"
    elif avg_spacing <= 30.0:
        formation_class = "moderate"
    else:
        formation_class = "loose"

    return {
        "stats": stats,
        "formation_class": formation_class,
        "avg_spacing_m": avg_spacing,
        "series": {
            "t_rel_s": (t2 - t2[0]) / 1000.0,
            "d12_m": d12,
            "d23_m": d23,
            "d13_m": d13,
        },
    }


def plot_figures(
    out_dir: Path,
    logs: Dict[str, Dict[str, ParsedCSV]],
    pdr_windows: Dict[str, Dict[str, np.ndarray]],
    latency_by_link: Dict[str, Dict],
    formation: Dict,
    sensor_agg: Dict,
    obs: Dict[str, np.ndarray],
    hard_event: Optional[Dict],
    offsets_to_v002: Dict[str, float],
) -> List[str]:
    if plt is None:
        return []

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    colors = {
        "V001": "#d1495b",
        "V002": "#00798c",
        "V003": "#edae49",
        "V001->V002": "#2a6f97",
        "V001->V003": "#d1495b",
        "V002->V001": "#4d908e",
        "V002->V003": "#f4a261",
        "V003->V001": "#9c6644",
        "V003->V002": "#6d597a",
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )

    # 1) GPS Track Map
    fig, ax = plt.subplots(figsize=(7, 6))
    for vehicle in VEHICLES:
        tx = logs[vehicle]["tx"].data
        gps = valid_gps(tx["lat"], tx["lon"])
        if np.sum(gps) == 0:
            continue
        ax.plot(tx["lon"][gps], tx["lat"][gps], linewidth=1.8, color=colors[vehicle], label=vehicle, alpha=0.9)
        ax.scatter(tx["lon"][gps][0], tx["lat"][gps][0], color=colors[vehicle], marker="o", s=30)
        ax.scatter(tx["lon"][gps][-1], tx["lat"][gps][-1], color=colors[vehicle], marker="X", s=40)
    ax.set_title("Convoy GPS Track Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    p = fig_dir / "gps_track_map.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 2) Speed Profiles
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for vehicle in VEHICLES:
        tx = logs[vehicle]["tx"].data
        t = (tx["timestamp_local_ms"] - tx["timestamp_local_ms"][0]) / 1000.0
        ax.plot(t, tx["speed"], color=colors[vehicle], linewidth=1.5, label=vehicle)
    ax.set_title("Vehicle Speed Profiles")
    ax.set_xlabel("Time Since Vehicle Start (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend(loc="best")
    fig.tight_layout()
    p = fig_dir / "speed_profiles.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 3) Packet Delivery Over Time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for link, series in pdr_windows.items():
        if series["t_s"].size == 0:
            continue
        ax.plot(series["t_s"], series["pdr"] * 100.0, label=link, linewidth=1.2, color=colors.get(link))
    ax.set_title("PDR Over Time (10s Sliding Window)")
    ax.set_xlabel("Sender-Relative Time (s)")
    ax.set_ylabel("PDR (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    p = fig_dir / "packet_delivery_over_time.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 4) Latency Distribution (Histogram + CDF)
    rel_all = np.concatenate(
        [v["relative_ms_series"] for v in latency_by_link.values() if v["relative_ms_series"].size > 0],
        dtype=float,
    ) if latency_by_link else np.array([], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if rel_all.size > 0:
        axes[0].hist(rel_all, bins=50, color="#2a9d8f", alpha=0.85)
        axes[0].set_xlabel("Relative Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Latency Histogram")

        s = np.sort(rel_all)
        y = np.arange(1, s.size + 1) / float(s.size)
        axes[1].plot(s, y, color="#e76f51", linewidth=1.8)
        axes[1].set_xlabel("Relative Latency (ms)")
        axes[1].set_ylabel("CDF")
        axes[1].set_title("Latency CDF")
    fig.tight_layout()
    p = fig_dir / "latency_distribution.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 5) Inter-Vehicle Distance
    series = formation.get("series", {})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if series:
        ax.plot(series["t_rel_s"], series["d12_m"], label="V001-V002", color="#d1495b", linewidth=1.5)
        ax.plot(series["t_rel_s"], series["d23_m"], label="V002-V003", color="#00798c", linewidth=1.5)
        ax.plot(series["t_rel_s"], series["d13_m"], label="V001-V003", color="#edae49", linewidth=1.5)
    ax.set_title("Inter-Vehicle Distance Over Time")
    ax.set_xlabel("Time (V002 clock, s)")
    ax.set_ylabel("Distance (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    p = fig_dir / "inter_vehicle_distance.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 6) Braking Event Detail (lead = V001, ego receiver = V002)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    v1_tx = logs["V001"]["tx"].data
    v2_rx = logs["V002"]["rx"].data
    if hard_event is not None:
        center = 0.5 * (hard_event["start_msg_ms"] + hard_event["end_msg_ms"])
        lo = center - 10_000
        hi = center + 10_000
        t_rel = (v1_tx["msg_timestamp"] - center) / 1000.0
        m = (v1_tx["msg_timestamp"] >= lo) & (v1_tx["msg_timestamp"] <= hi)
        axes[0].plot(t_rel[m], v1_tx["accel_x"][m], color="#d1495b", linewidth=1.8)
        axes[0].axhline(-2.0, color="#666666", linestyle="--", linewidth=1.0)
        axes[0].set_ylabel("Lead accel_x")
        axes[0].set_title("Hard Braking Event Detail")

        rx_m = v2_rx["sender_id"] == "V001"
        rx_msg = v2_rx["msg_timestamp"][rx_m]
        rx_local = v2_rx["timestamp_local_ms"][rx_m]
        rx_time_in_v1 = rx_local - float(offsets_to_v002.get("V001", 0.0))
        in_win_rx = (rx_time_in_v1 >= lo) & (rx_time_in_v1 <= hi)
        axes[1].scatter((rx_time_in_v1[in_win_rx] - center) / 1000.0, np.ones(np.sum(in_win_rx)), s=15, color="#2a9d8f", label="Received")

        tx_win = v1_tx["msg_timestamp"][(v1_tx["msg_timestamp"] >= lo) & (v1_tx["msg_timestamp"] <= hi)]
        rx_set = set(int(v) for v in rx_msg.tolist())
        miss = np.array([int(ts) not in rx_set for ts in tx_win], dtype=bool)
        if np.sum(miss) > 0:
            axes[1].scatter((tx_win[miss] - center) / 1000.0, np.zeros(np.sum(miss)), marker="x", color="#e76f51", s=25, label="Missing")
        axes[1].set_yticks([0, 1], ["Missing", "Rx"])
        axes[1].legend(loc="upper right")

        lat_ms = rx_local[in_win_rx].astype(float) - rx_msg[in_win_rx].astype(float)
        axes[2].scatter((rx_time_in_v1[in_win_rx] - center) / 1000.0, lat_ms, s=15, color="#264653")
        axes[2].set_ylabel("Raw latency term (ms)")
        axes[2].set_xlabel("Time around hard brake (s)")

    fig.tight_layout()
    p = fig_dir / "braking_event_detail.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 7) Sensor Noise Comparison
    metrics = ["gps_std_m", "accel_std_ms2", "gyro_std_rad_s", "heading_std_deg", "mag_std_ut"]
    stationary = sensor_agg.get("stationary", {})
    driving = sensor_agg.get("cruising", {})
    current = CURRENT_EMULATOR_PARAMS["sensor_noise"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.8))
    for i, metric in enumerate(metrics):
        vals = [
            stationary.get(metric, float("nan")),
            driving.get(metric, float("nan")),
            current.get(metric, float("nan")),
        ]
        axes[i].bar([0, 1, 2], vals, color=["#8ecae6", "#219ebc", "#ffb703"], width=0.72)
        axes[i].set_xticks([0, 1, 2], ["Stationary", "Driving", "Current"], rotation=25)
        axes[i].set_title(metric)
    fig.suptitle("Sensor Noise: Stationary vs Driving vs Current Params")
    fig.tight_layout()
    p = fig_dir / "sensor_noise_comparison.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # 8) Observation Value Distributions
    ego = obs["ego"]
    peers = obs["peers"]
    mask = obs["peer_mask"] > 0.5
    dist_fields = {
        "ego_speed_norm": ego[:, 0],
        "ego_accel_norm": ego[:, 1],
        "ego_heading_norm": ego[:, 2],
        "peer_count_norm": ego[:, 3],
        "peer_rel_x_norm": peers[:, :, 0][mask],
        "peer_age_norm": peers[:, :, 5][mask],
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, values) in zip(axes.flatten(), dist_fields.items()):
        if values.size == 0:
            ax.set_title(name)
            continue
        ax.hist(values, bins=40, color="#577590", alpha=0.85)
        ax.axvline(-1.0, color="#e63946", linestyle="--", linewidth=1.0)
        ax.axvline(1.0, color="#e63946", linestyle="--", linewidth=1.0)
        ax.set_title(name)
    fig.suptitle("Observation Value Distributions vs Training Range [-1, 1]")
    fig.tight_layout()
    p = fig_dir / "observation_value_distributions.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    return paths


def build_emulator_params_from_convoy(
    pdr_links: Dict[str, Dict],
    latency_by_link: Dict[str, Dict],
    burst_by_link: Dict[str, Dict],
    distance_loss_by_link: Dict[str, Dict],
    sensor_agg: Dict,
) -> Dict:
    rel_latency = np.concatenate(
        [v["relative_ms_series"] for v in latency_by_link.values() if v["relative_ms_series"].size > 0],
        dtype=float,
    ) if latency_by_link else np.array([], dtype=float)
    if rel_latency.size == 0:
        base_ms = 5.0
        jitter_ms = 1.0
    else:
        base_ms = float(np.median(rel_latency))
        jitter_ms = float(np.std(rel_latency, ddof=1)) if rel_latency.size > 1 else 0.0

    pdr_vals = [v["pdr"] for v in pdr_links.values() if not np.isnan(v["pdr"])]
    overall_pdr = float(np.mean(pdr_vals)) if pdr_vals else 0.0
    base_loss = 1.0 - overall_pdr

    # Aggregate distance bins across links.
    bin_acc: Dict[str, List[float]] = {}
    for link_data in distance_loss_by_link.values():
        bins = link_data.get("bins", {})
        for name, stats in bins.items():
            if stats.get("count", 0) > 0 and not np.isnan(stats.get("loss_rate", float("nan"))):
                bin_acc.setdefault(name, []).append(float(stats["loss_rate"]))
    loss_20_50 = float(np.mean(bin_acc.get("20_50m", [base_loss])))
    loss_50_100 = float(np.mean(bin_acc.get("50_100m", [max(base_loss, loss_20_50)])))
    loss_100_plus = float(np.mean(bin_acc.get("100m_plus", [max(loss_50_100, loss_20_50)])))

    burst_means = [v["mean_burst_length"] for v in burst_by_link.values() if v["mean_burst_length"] > 0]
    burst_max = [v["max_burst_length"] for v in burst_by_link.values()]
    mean_burst = float(np.mean(burst_means)) if burst_means else 1.0
    max_burst_length = int(np.max(burst_max)) if burst_max else 1
    max_loss_cap = float(min(0.95, max(0.4, base_loss * 2.2)))

    stationary = sensor_agg.get("stationary", {})
    driving = sensor_agg.get("cruising", {})

    sensor_noise = {
        "gps_std_m": float(driving.get("gps_std_m", stationary.get("gps_std_m", CURRENT_EMULATOR_PARAMS["sensor_noise"]["gps_std_m"]))),
        "speed_std_ms": float(driving.get("speed_std_ms", stationary.get("speed_std_ms", CURRENT_EMULATOR_PARAMS["sensor_noise"]["speed_std_ms"]))),
        "accel_std_ms2": float(driving.get("accel_std_ms2", stationary.get("accel_std_ms2", CURRENT_EMULATOR_PARAMS["sensor_noise"]["accel_std_ms2"]))),
        "gyro_std_rad_s": float(driving.get("gyro_std_rad_s", stationary.get("gyro_std_rad_s", CURRENT_EMULATOR_PARAMS["sensor_noise"]["gyro_std_rad_s"]))),
        "heading_std_deg": float(driving.get("heading_std_deg", stationary.get("heading_std_deg", CURRENT_EMULATOR_PARAMS["sensor_noise"]["heading_std_deg"]))),
        "mag_std_ut": float(driving.get("mag_std_ut", stationary.get("mag_std_ut", CURRENT_EMULATOR_PARAMS["sensor_noise"]["mag_std_ut"]))),
    }

    return {
        "latency": {
            "base_ms": base_ms,
            "jitter_std_ms": jitter_ms,
            "notes": "Relative latency after per-link clock offset correction.",
        },
        "packet_loss": {
            "base_rate": float(base_loss),
            "distance_threshold_1": 50,
            "distance_threshold_2": 100,
            "rate_tier_1": loss_20_50,
            "rate_tier_2": loss_50_100,
            "rate_tier_3": loss_100_plus,
        },
        "burst_loss": {
            "enabled": bool(mean_burst >= 1.5 or max_burst_length >= 3),
            "mean_burst_length": mean_burst,
            "max_burst_length": max_burst_length,
            "max_loss_cap": max_loss_cap,
        },
        "sensor_noise": sensor_noise,
        "sensor_noise_stationary": stationary,
        "sensor_noise_driving": driving,
    }


def markdown_report(
    out_path: Path,
    data_quality: Dict,
    temporal: Dict,
    pdr_links: Dict[str, Dict],
    latency_by_link: Dict[str, Dict],
    burst_by_link: Dict[str, Dict],
    distance_by_link: Dict[str, Dict],
    events: Dict,
    obs_ranges: Dict,
    zero_peer_windows: Dict,
    inference: Dict,
    formation: Dict,
    trajectory_outputs: Dict,
    figures: List[str],
) -> None:
    pdr_items = sorted(pdr_links.values(), key=lambda x: x["link"])
    latency_rel_all = np.concatenate(
        [v["relative_ms_series"] for v in latency_by_link.values() if v["relative_ms_series"].size > 0],
        dtype=float,
    ) if latency_by_link else np.array([], dtype=float)

    lines: List[str] = []
    lines.append("# Convoy Analysis Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Dataset: `Convoy_recording_02212026` (V001, V002, V003).")
    lines.append("- Ego perspective for model validation: **V002**.")
    lines.append("- Ground-truth protocol expected: 2 light braking events, 1 hard braking event, final stationary window.")
    lines.append("")
    lines.append("## Task 1 - Data Validation")
    lines.append("")
    lines.append("| File | Rows (valid/total) | GPS fix % | Duration (s) | Duplicates | Gaps >500ms | Sensor status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for rec in data_quality["files"]:
        lines.append(
            f"| `{Path(rec['file']).name}` | {rec['rows_valid']}/{rec['rows_total']} | "
            f"{rec['gps_fix_pct']:.1f} | {rec['duration_s']:.1f} | {rec['duplicate_timestamps']} | "
            f"{rec['gaps_gt_500ms']} | {rec['sensor_status']} |"
        )
    lines.append("")
    lines.append(
        f"- Estimated 3-vehicle overlap window (mapped to V002 clock): "
        f"{temporal['overlap_duration_s']:.1f} s."
    )
    lines.append("")
    lines.append("## Task 2 - Packet Delivery / Latency")
    lines.append("")
    lines.append("| Link | TX in overlap | Matched RX | PDR | Loss | Mean relative latency (ms) | p95 relative latency (ms) | Mean burst length |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for pdr in pdr_items:
        link = pdr["link"]
        lat = latency_by_link[link]["relative_ms"]
        burst = burst_by_link[link]
        lines.append(
            f"| {link} | {pdr['tx_in_overlap']} | {pdr['matched']} | {pdr['pdr']:.3f} | "
            f"{pdr['loss_rate']:.3f} | {lat['mean']:.2f} | {lat['p95']:.2f} | {burst['mean_burst_length']:.2f} |"
        )
    if latency_rel_all.size > 0:
        lat_stats = basic_stats(latency_rel_all)
        lines.append("")
        lines.append(
            f"- Overall relative latency: mean {lat_stats['mean']:.2f} ms, "
            f"p95 {lat_stats['p95']:.2f} ms, p99 {lat_stats['p99']:.2f} ms."
        )
    lines.append("")
    lines.append("## Task 3 - Sensor Noise")
    lines.append("")
    lines.append("- Segment aggregation is across all three TX logs.")
    lines.append("- `stationary` and `cruising` are the primary references for emulator calibration.")
    lines.append("")
    lines.append("## Task 4 - Offline Observation / Inference")
    lines.append("")
    lines.append("- Observation arrays exported in `convoy_observations.npz` (ego, peers, peer_mask, timestamps).")
    lines.append(f"- Zero-peer windows (>500ms): {zero_peer_windows['count']}.")
    outlier_fields = [
        (name, vals["pct_outside_minus1p5_to_1p5"])
        for name, vals in obs_ranges.items()
        if vals.get("count", 0) > 0 and vals.get("pct_outside_minus1p5_to_1p5", 0.0) > 0.0
    ]
    if outlier_fields:
        lines.append("- Out-of-distribution candidates (outside [-1.5, 1.5]):")
        for name, pct in sorted(outlier_fields, key=lambda x: -x[1]):
            lines.append(f"  - {name}: {pct:.2f}%")
    else:
        lines.append("- No fields exceeded [-1.5, 1.5].")

    if inference.get("loaded"):
        lines.append(f"- Model inference executed using `{inference['model_path']}`.")
        lines.append(f"- Action fractions: {inference['action_fraction']}.")
        if inference.get("hard_event_window_summary") is not None:
            lines.append(f"- Hard-event window action summary: {inference['hard_event_window_summary']}.")
    else:
        lines.append(f"- Model inference skipped: {inference.get('reason', 'unknown')}.")

    lines.append("")
    lines.append("## Task 5 - Trajectory / Formation")
    lines.append("")
    lines.append(f"- Formation classification: **{formation.get('formation_class', 'unknown')}**.")
    lines.append(f"- Average convoy spacing: {formation.get('avg_spacing_m', float('nan')):.2f} m.")
    lines.append("- Extracted trajectories:")
    for vehicle, info in trajectory_outputs.items():
        lines.append(f"  - {vehicle}: `{Path(info['csv']).name}` ({info['samples']} samples, {info['duration_s']:.1f}s)")

    lines.append("")
    lines.append("## Event Detection")
    lines.append("")
    lines.append(f"- Selected braking events: {len(events.get('events', []))}.")
    for ev in events.get("events", []):
        lines.append(
            f"  - {ev.get('event_type', 'braking')}: start={ev['start_msg_ms']} ms, "
            f"end={ev['end_msg_ms']} ms, min_accel={ev['min_accel_x_ms2']:.2f} m/s^2"
        )
    if events.get("stationary_tail") is not None:
        st = events["stationary_tail"]
        lines.append(
            f"- Final stationary tail: start={st['start_msg_ms']} ms, end={st['end_msg_ms']} ms, "
            f"duration={st['duration_s']:.1f}s"
        )

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{Path(fig).name}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze 3-car convoy field recording.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "Convoy_recording_02212026",
        help="Path to Convoy_recording_02212026 directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "../data/convoy_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional PPO model zip for offline inference.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    if not args.input_root.exists():
        print(f"Input root not found: {args.input_root}", file=sys.stderr)
        return 1

    logs: Dict[str, Dict[str, ParsedCSV]] = {}
    try:
        for vehicle in VEHICLES:
            tx_path = find_single_file(args.input_root, vehicle, "tx")
            rx_path = find_single_file(args.input_root, vehicle, "rx")
            logs[vehicle] = {
                "tx": parse_convoy_csv(tx_path, vehicle, "tx"),
                "rx": parse_convoy_csv(rx_path, vehicle, "rx"),
            }
    except Exception as exc:
        print(f"Failed loading convoy CSVs: {exc}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    file_summaries = [
        build_file_validation(logs[v][typ])
        for v in VEHICLES
        for typ in ("tx", "rx")
    ]

    temporal = align_vehicle_windows_to_v002(logs)
    clock_links = {}
    for sender in VEHICLES:
        for receiver in VEHICLES:
            if sender == receiver:
                continue
            est = offset_sender_to_receiver(logs, sender, receiver)
            if est is not None:
                clock_links[f"{sender}->{receiver}"] = est

    links = [
        ("V001", "V002"),
        ("V001", "V003"),
        ("V002", "V001"),
        ("V002", "V003"),
        ("V003", "V001"),
        ("V003", "V002"),
    ]
    pdr_links = {f"{s}->{r}": compute_link_pdr(logs, s, r) for s, r in links}
    pdr_windows = {f"{s}->{r}": compute_pdr_windows(logs, s, r) for s, r in links}
    latency_by_link = {f"{s}->{r}": compute_latency_by_link(logs, s, r) for s, r in links}
    burst_by_link = {f"{s}->{r}": compute_burst_by_link(logs, s, r) for s, r in links}
    distance_by_link = {f"{s}->{r}": compute_distance_binned_loss(logs, s, r) for s, r in links}

    per_vehicle_sensor = {v: compute_sensor_noise_for_vehicle(logs[v]["tx"]) for v in VEHICLES}
    sensor_agg = aggregate_sensor_noise(per_vehicle_sensor)

    # V002 reference point for local XY conversion.
    v2_tx = logs["V002"]["tx"].data
    valid_v2 = valid_gps(v2_tx["lat"], v2_tx["lon"])
    if np.sum(valid_v2) == 0:
        print("V002 has no valid GPS fixes; cannot build observation/trajectory geometry.", file=sys.stderr)
        return 1
    ref_lat = float(v2_tx["lat"][valid_v2][0])
    ref_lon = float(v2_tx["lon"][valid_v2][0])

    events = detect_braking_events(logs["V001"]["tx"])
    hard_event = events.get("hard_event")
    hard_event_v002_window: Optional[Tuple[float, float]] = None
    if hard_event is not None:
        shift = float(temporal["offsets_to_v002_ms"].get("V001", 0.0))
        hard_event_v002_window = (
            hard_event["start_msg_ms"] + shift,
            hard_event["end_msg_ms"] + shift,
        )

    obs = build_observations_v002(logs, ref_lat, ref_lon)
    obs_ranges = summarize_observation_ranges(obs)
    zero_peer = compute_zero_peer_windows(obs)

    repo_root = Path(__file__).resolve().parents[2]
    model_path = args.model_path if args.model_path is not None else discover_default_model(repo_root)
    inference = run_optional_model_inference(model_path, obs, hard_event_v002_window)

    # Save NPZ for downstream offline work.
    obs_npz_path = args.out_dir / "convoy_observations.npz"
    np.savez_compressed(
        obs_npz_path,
        timestamps_local_ms=obs["timestamps_local_ms"],
        ego=obs["ego"],
        peers=obs["peers"],
        peer_mask=obs["peer_mask"],
    )

    # Extract trajectories and formation stats.
    traj_outputs = extract_trajectories(logs, args.out_dir, ref_lat, ref_lon)
    formation = formation_analysis(logs, temporal["offsets_to_v002_ms"])

    figures: List[str] = []
    if not args.no_plots and plt is not None:
        figures = plot_figures(
            out_dir=args.out_dir,
            logs=logs,
            pdr_windows=pdr_windows,
            latency_by_link=latency_by_link,
            formation=formation,
            sensor_agg=sensor_agg,
            obs=obs,
            hard_event=hard_event,
            offsets_to_v002=temporal["offsets_to_v002_ms"],
        )

    emulator_params = build_emulator_params_from_convoy(
        pdr_links=pdr_links,
        latency_by_link=latency_by_link,
        burst_by_link=burst_by_link,
        distance_loss_by_link=distance_by_link,
        sensor_agg=sensor_agg,
    )

    data_quality_summary = {
        "input_root": str(args.input_root),
        "files": file_summaries,
        "temporal_alignment": temporal,
        "clock_link_estimates": clock_links,
    }

    events_out = {
        "expected_protocol": {
            "light_braking_events": 2,
            "hard_braking_events": 1,
            "final_stationary_window_expected_s": 60,
            "ego_vehicle": "V002",
        },
        "detected": events,
        "mapped_to_v002_clock": {
            "hard_event_v002_ms": list(hard_event_v002_window) if hard_event_v002_window is not None else None,
        },
    }

    analysis_summary = {
        "pdr_by_link": pdr_links,
        "latency_by_link": {
            k: {
                "raw_ms": v["raw_ms"],
                "relative_ms": v["relative_ms"],
            }
            for k, v in latency_by_link.items()
        },
        "burst_by_link": burst_by_link,
        "distance_loss_by_link": distance_by_link,
        "sensor_noise_per_vehicle": per_vehicle_sensor,
        "sensor_noise_aggregate": sensor_agg,
        "observation_ranges": obs_ranges,
        "zero_peer_windows": zero_peer,
        "inference": {
            k: v
            for k, v in inference.items()
            if k not in ("actions", "probs")
        },
        "formation": {
            "stats": formation.get("stats", {}),
            "formation_class": formation.get("formation_class"),
            "avg_spacing_m": formation.get("avg_spacing_m"),
        },
        "trajectories": traj_outputs,
    }

    write_json(args.out_dir / "data_quality_summary.json", data_quality_summary)
    write_json(args.out_dir / "convoy_events.json", events_out)
    write_json(args.out_dir / "convoy_emulator_params.json", emulator_params)
    write_json(args.out_dir / "analysis_summary.json", analysis_summary)

    markdown_report(
        out_path=args.out_dir / "convoy_analysis_report.md",
        data_quality=data_quality_summary,
        temporal=temporal,
        pdr_links=pdr_links,
        latency_by_link=latency_by_link,
        burst_by_link=burst_by_link,
        distance_by_link=distance_by_link,
        events=events,
        obs_ranges=obs_ranges,
        zero_peer_windows=zero_peer,
        inference=inference,
        formation=formation,
        trajectory_outputs=traj_outputs,
        figures=figures,
    )

    print("Convoy analysis complete")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Report: {args.out_dir / 'convoy_analysis_report.md'}")
    print(f"  Emulator params: {args.out_dir / 'convoy_emulator_params.json'}")
    print(f"  Observations: {obs_npz_path}")
    print(f"  Events: {args.out_dir / 'convoy_events.json'}")
    print(f"  Data quality: {args.out_dir / 'data_quality_summary.json'}")
    if figures:
        print(f"  Figures: {args.out_dir / 'figures'} ({len(figures)} files)")
    else:
        print("  Figures: skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
