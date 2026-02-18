#!/usr/bin/env python3
"""
Validate recorded CSV logs from field runs.

Understands three formats produced by this repo:
1) RTT sender logs (rtt_log.csv):
   sequence,send_time_ms,recv_time_ms,rtt_ms,lat,lon,speed,heading,accel_x,accel_y,accel_z,mag_x,mag_y,mag_z,lost
2) Characterization TX/RX logs (hardware main):
   TX: timestamp_local_ms,msg_timestamp,vehicle_id,lat,lon,speed,heading,accel_*,gyro_*,mag_*
   RX: timestamp_local_ms,msg_timestamp,from_vehicle_id,lat,lon,speed,heading,accel_*,gyro_*,mag_*
3) Training session logs (single file):
   timestamp_ms,vehicle_id,lat,lon,alt,speed,heading,long_accel,lat_accel,accel_*,gyro_*,mag_*,gps_valid,gps_age_ms

The script prints a PASS/FAIL report per file and sets a non-zero exit code
if any validation fails.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# --- File type detection ----------------------------------------------------


class FileType:
    RTT = "rtt"
    TX = "tx"
    RX = "rx"
    TRAINING = "training"
    UNKNOWN = "unknown"


def _lower_keys(d: Iterable[str]) -> Dict[str, str]:
    return {k.lower(): k for k in d}


def detect_type(header: List[str]) -> str:
    if not header:
        return FileType.UNKNOWN
    cols = {h.lower() for h in header}
    if {"sequence", "send_time_ms", "recv_time_ms", "rtt_ms", "lat", "lon", "lost"}.issubset(cols):
        return FileType.RTT
    if {"timestamp_local_ms", "msg_timestamp", "vehicle_id", "lat", "lon"}.issubset(cols):
        return FileType.TX
    if {"timestamp_local_ms", "msg_timestamp", "from_vehicle_id", "lat", "lon"}.issubset(cols):
        return FileType.RX
    if {"timestamp_ms", "vehicle_id", "lat", "lon", "gps_valid", "gps_age_ms"}.issubset(cols):
        return FileType.TRAINING
    return FileType.UNKNOWN


# --- Validation helpers -----------------------------------------------------


@dataclass
class Result:
    file: Path
    kind: str
    passed: bool
    messages: List[str]


def _is_valid_lat_lon(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (
        abs(lat) < 1e-9 and abs(lon) < 1e-9
    )


def _check_monotonic(values: List[int]) -> bool:
    last = -math.inf
    for v in values:
        if v < last:
            return False
        last = v
    return True


def validate_file(path: Path, min_rows: int, min_gps_fraction: float, require_gps: bool) -> Result:
    messages: List[str] = []
    try:
        with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return Result(path, FileType.UNKNOWN, False, ["Missing header row"])
            kind = detect_type(reader.fieldnames)
            if kind == FileType.UNKNOWN:
                return Result(path, kind, False, [f"Unrecognized header: {reader.fieldnames}"])

            rows = list(reader)
            row_count = len(rows)
            if row_count < min_rows:
                messages.append(f"Too few rows: {row_count} < {min_rows}")

            # Common checks: lat/lon validity fraction
            header_map = _lower_keys(reader.fieldnames)
            lat_key = header_map.get("lat")
            lon_key = header_map.get("lon")
            valid_gps = 0
            for r in rows:
                try:
                    lat = float(r[lat_key]) if lat_key else math.nan
                    lon = float(r[lon_key]) if lon_key else math.nan
                except (TypeError, ValueError):
                    lat = lon = math.nan
                if not math.isnan(lat) and not math.isnan(lon) and _is_valid_lat_lon(lat, lon):
                    valid_gps += 1
            gps_fraction = (valid_gps / row_count) if row_count else 0.0
            if require_gps and gps_fraction < min_gps_fraction:
                messages.append(f"Low valid GPS fraction: {gps_fraction:.2f} < {min_gps_fraction:.2f}")

            # Type-specific checks
            if kind == FileType.RTT:
                # recv_time_ms >= send_time_ms for received rows, rtt_ms matches
                bad_time = 0
                bad_rtt = 0
                sends: List[int] = []
                for r in rows:
                    try:
                        s = int(float(r[header_map["send_time_ms"]]))
                        rc = int(float(r[header_map["recv_time_ms"]]))
                        rtt = int(float(r[header_map["rtt_ms"]]))
                        lost = int(float(r[header_map["lost"]]))
                    except Exception:
                        continue
                    sends.append(s)
                    if lost == 0 and rc < s:
                        bad_time += 1
                    if lost == 0 and rtt != (rc - s):
                        bad_rtt += 1
                if bad_time:
                    messages.append(f"{bad_time} rows have recv_time_ms < send_time_ms")
                if bad_rtt:
                    messages.append(f"{bad_rtt} rows have inconsistent rtt_ms")
                if sends and not _check_monotonic(sends):
                    messages.append("send_time_ms not monotonic non-decreasing")

            elif kind in (FileType.TX, FileType.RX):
                # Timestamps monotonic, speed non-negative, heading range
                ts_key = header_map.get("timestamp_local_ms")
                msg_ts_key = header_map.get("msg_timestamp")
                speeds: List[float] = []
                headings_bad = 0
                ts_vals: List[int] = []
                msg_ts_vals: List[int] = []
                for r in rows:
                    try:
                        speeds.append(max(0.0, float(r[header_map["speed"]])))
                    except Exception:
                        pass
                    try:
                        hd = float(r[header_map["heading"]])
                        if not (0.0 <= hd <= 360.0):
                            headings_bad += 1
                    except Exception:
                        headings_bad += 1
                    try:
                        if ts_key:
                            ts_vals.append(int(float(r[ts_key])))
                        if msg_ts_key:
                            msg_ts_vals.append(int(float(r[msg_ts_key])))
                    except Exception:
                        pass
                if ts_vals and not _check_monotonic(ts_vals):
                    messages.append("timestamp_local_ms not monotonic non-decreasing")
                if msg_ts_vals and not _check_monotonic(msg_ts_vals):
                    messages.append("msg_timestamp not monotonic non-decreasing")
                if headings_bad > row_count * 0.1:
                    messages.append(f"Many invalid headings: {headings_bad}/{row_count}")

            elif kind == FileType.TRAINING:
                # gps_valid and gps_age_ms checks; heading and speed range
                bad_age = 0
                bad_heading = 0
                bad_speed = 0
                ages: List[int] = []
                for r in rows:
                    try:
                        gv = int(float(r[header_map["gps_valid"]]))
                        ga = int(float(r[header_map["gps_age_ms"]]))
                        ages.append(ga)
                        if gv not in (0, 1):
                            messages.append("gps_valid field contains non {0,1}")
                        if gv == 1 and ga > 30000:
                            bad_age += 1
                    except Exception:
                        pass
                    try:
                        spd = float(r[header_map["speed"]])
                        if spd < -0.5:
                            bad_speed += 1
                    except Exception:
                        pass
                    try:
                        hd = float(r[header_map["heading"]])
                        if not (0.0 <= hd <= 360.0):
                            bad_heading += 1
                    except Exception:
                        bad_heading += 1
                if ages and not _check_monotonic(ages):
                    # Age should generally be non-decreasing until reset; warn only
                    messages.append("gps_age_ms not monotonic (ok if fix refreshed)")
                if bad_age:
                    messages.append(f"gps_age_ms > 30000 for {bad_age} valid-fix rows")
                if bad_speed:
                    messages.append(f"Negative speeds in {bad_speed} rows")
                if bad_heading > row_count * 0.1:
                    messages.append(f"Many invalid headings: {bad_heading}/{row_count}")

            passed = len(messages) == 0
            if passed:
                messages.append(f"OK: {row_count} rows; valid GPS fraction {gps_fraction:.2f}")
            return Result(path, kind, passed, messages)

    except Exception as e:
        return Result(path, FileType.UNKNOWN, False, [f"Exception: {e}"])


def gather_inputs(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.csv")))
        elif p.suffix.lower() == ".csv":
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate field recording CSV logs")
    ap.add_argument("inputs", nargs="+", type=Path, help="CSV file(s) or directory(ies)")
    ap.add_argument("--min-rows", type=int, default=50, help="Minimum rows required")
    ap.add_argument("--min-gps-fraction", type=float, default=0.5, help="Min fraction of rows with valid lat/lon")
    ap.add_argument("--no-require-gps", action="store_true", help="Do not fail on low GPS fraction")
    args = ap.parse_args()

    inputs = gather_inputs([Path(x) for x in args.inputs])
    if not inputs:
        print("No CSV files found in inputs", file=sys.stderr)
        sys.exit(2)

    results: List[Result] = []
    for f in inputs:
        res = validate_file(
            f,
            min_rows=args.min_rows,
            min_gps_fraction=args.min_gps_fraction,
            require_gps=not args.no_require_gps,
        )
        results.append(res)

    any_fail = False
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.kind:9} :: {r.file}")
        for m in r.messages:
            print(f"  - {m}")

        if not r.passed:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

