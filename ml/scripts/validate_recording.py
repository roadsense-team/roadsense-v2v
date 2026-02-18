#!/usr/bin/env python3
"""
Post-Recording Validator

Validates raw logs from RTT and Convoy recordings and emits JSON + Markdown
reports. Auto-detects mode, checks headers, timestamps, GPS, IMU, and (when
possible) RTT/loss. Exits non-zero on failures (and on warnings if --strict).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class Mode:
    RTT = "rtt"
    CONVOY = "convoy"
    COMBINED = "combined"
    AUTO = "auto"


class FileType:
    RTT = "rtt"
    TX = "tx"
    RX = "rx"
    TRAINING = "training"
    COMBINED = "combined"
    UNKNOWN = "unknown"


def _lower_keys(d: Iterable[str]) -> Dict[str, str]:
    return {k.lower(): k for k in d}


def detect_file_type(header: List[str], name: str) -> str:
    cols = {h.strip().lower() for h in (header or [])}
    base = name.lower()
    if {"sequence", "send_time_ms", "recv_time_ms", "rtt_ms", "lat", "lon", "lost"}.issubset(cols):
        return FileType.RTT
    if {"timestamp_local_ms", "msg_timestamp", "vehicle_id", "lat", "lon"}.issubset(cols):
        return FileType.TX
    if {"timestamp_local_ms", "msg_timestamp", "from_vehicle_id", "lat", "lon"}.issubset(cols):
        return FileType.RX
    if {"timestamp_ms", "vehicle_id", "lat", "lon", "gps_valid", "gps_age_ms"}.issubset(cols):
        return FileType.TRAINING
    if base.startswith("scenario_") or base.endswith("scenario.csv") or "v002_lat" in cols:
        return FileType.COMBINED
    return FileType.UNKNOWN


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [h.strip() for h in header]


def list_csvs(root: Path) -> List[Path]:
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def detect_mode(paths: List[Path]) -> str:
    types = []
    for p in paths:
        t = detect_file_type(read_header(p), p.name)
        types.append(t)
    if any(t == FileType.RTT for t in types):
        return Mode.RTT
    # Convoy when multiple v00X_tx/rx present
    names = [p.name.lower() for p in paths]
    if any("v001_tx_" in n for n in names) and any("v001_rx_" in n for n in names):
        if any("v002_tx_" in n for n in names) and any("v003_tx_" in n for n in names):
            return Mode.CONVOY
        # Fallback to convoy with partial set
        return Mode.CONVOY
    if any(t in (FileType.TRAINING, FileType.COMBINED) for t in types):
        return Mode.COMBINED
    return Mode.CONVOY  # default fallback


def _is_valid_lat_lon(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (
        abs(lat) < 1e-9 and abs(lon) < 1e-9
    )


def _monotonic(values: List[int]) -> bool:
    last = -math.inf
    for v in values:
        if v < last:
            return False
        last = v
    return True


def _median_delta_ms(values: List[int]) -> Optional[float]:
    if len(values) < 3:
        return None
    diffs = [b - a for a, b in zip(values[:-1], values[1:])]
    diffs = [d for d in diffs if d >= 0]
    if not diffs:
        return None
    return float(statistics.median(diffs))


@dataclass
class FileReport:
    path: str
    type: str
    rows: int
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


@dataclass
class SessionReport:
    mode: str
    files: List[FileReport]
    summary: Dict[str, float]
    status: str  # pass|warn|fail
    reasons: List[str]


def _parse_float(row: Dict[str, str], key_map: Dict[str, str], key: str) -> Optional[float]:
    k = key_map.get(key)
    if not k:
        return None
    try:
        return float(row[k])
    except Exception:
        return None


def validate_rtt_csv(path: Path, sample_rate_hz: int) -> FileReport:
    issues: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, float] = {}

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        hdr = [h.strip() for h in (reader.fieldnames or [])]
        need = [
            "sequence","send_time_ms","recv_time_ms","rtt_ms","lat","lon","speed","heading","accel_x","accel_y","accel_z","mag_x","mag_y","mag_z","lost"
        ]
        if hdr != need:
            if set(hdr) >= set(need):
                warnings.append("Header order differs from expected")
            else:
                issues.append("Missing required columns for RTT")

        key = _lower_keys(hdr)
        seqs: List[int] = []
        sends: List[int] = []
        rtts: List[float] = []
        losts: List[int] = []
        lat_valid = 0
        rows = 0
        for row in reader:
            rows += 1
            s = _parse_float(row, key, "send_time_ms")
            rc = _parse_float(row, key, "recv_time_ms")
            rtt = _parse_float(row, key, "rtt_ms")
            lost = int(_parse_float(row, key, "lost") or 0)
            seq = int(_parse_float(row, key, "sequence") or 0)
            lt = _parse_float(row, key, "lat")
            ln = _parse_float(row, key, "lon")

            if s is not None:
                sends.append(int(s))
            seqs.append(seq)
            losts.append(lost)
            if lost == 0 and rtt is not None:
                if rtt <= 0:
                    issues.append("Non-positive RTT detected")
                rtts.append(float(rtt))
                if rc is not None and s is not None and rc < s:
                    issues.append("recv_time_ms < send_time_ms")
            if lt is not None and ln is not None and _is_valid_lat_lon(lt, ln):
                lat_valid += 1

        gps_frac = (lat_valid / rows) if rows else 0.0
        metrics.update({"rows": rows, "gps_fraction": gps_frac})
        if rows < 60:
            warnings.append("Too few rows (<60)")
        if gps_frac < 0.95:
            warnings.append("GPS valid fraction < 0.95")
        if sends and not _monotonic(sends):
            issues.append("send_time_ms not monotonic")
        md = _median_delta_ms(sends)
        if md is not None:
            metrics["median_dt_ms"] = md
            if abs(md - (1000 / sample_rate_hz)) > 50:
                warnings.append("Median dt not near expected sample rate")

        # Loss via explicit flag or via sequence gaps
        if rows:
            loss_explicit = sum(losts) / rows
            metrics["loss_rate"] = loss_explicit
            if loss_explicit > 0.6:
                issues.append("Loss rate > 60%")
            elif loss_explicit > 0.3:
                warnings.append("High loss rate (>30%)")
        if seqs:
            seq_gaps = (max(seqs) - min(seqs) + 1) - len(seqs)
            if max(seqs) > min(seqs) and seq_gaps >= 0:
                est_loss = seq_gaps / (max(seqs) - min(seqs) + 1)
                metrics["loss_rate_seq"] = est_loss

        if rtts:
            arr = np.array(rtts, dtype=float)
            metrics.update({
                "rtt_mean": float(np.mean(arr)),
                "rtt_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "rtt_p50": float(np.percentile(arr, 50)),
                "rtt_p95": float(np.percentile(arr, 95)),
                "rtt_p99": float(np.percentile(arr, 99)),
            })

    return FileReport(str(path), FileType.RTT, metrics.get("rows", 0), issues, warnings, metrics)


def _read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        hdr = [h.strip() for h in (reader.fieldnames or [])]
        rows = list(reader)
    return hdr, rows


def validate_tx_rx_csv(path: Path, expect_tx: bool, sample_rate_hz: int) -> FileReport:
    issues: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    hdr, rows = _read_rows(path)
    cols_tx = ["timestamp_local_ms","msg_timestamp","vehicle_id","lat","lon","speed","heading","accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]
    cols_rx = ["timestamp_local_ms","msg_timestamp","from_vehicle_id","lat","lon","speed","heading","accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]
    need = cols_tx if expect_tx else cols_rx
    if hdr != need:
        if set(hdr) >= set(need):
            warnings.append("Header order differs from expected")
        else:
            issues.append("Missing required columns for TX/RX")
    km = _lower_keys(hdr)
    n = len(rows)
    metrics["rows"] = n
    if n < 60:
        warnings.append("Too few rows (<60)")

    # GPS validity fraction
    valid_gps = 0
    ts_local: List[int] = []
    msg_ts: List[int] = []
    heading_bad = 0
    speeds_neg = 0
    for r in rows:
        try:
            lt = float(r[km["lat"]]); ln = float(r[km["lon"]])
            if _is_valid_lat_lon(lt, ln):
                valid_gps += 1
        except Exception:
            pass
        try:
            ts_local.append(int(float(r[km["timestamp_local_ms"]])))
            msg_ts.append(int(float(r[km["msg_timestamp"]])))
        except Exception:
            pass
        try:
            hd = float(r[km["heading"]]);
            if not (0.0 <= hd <= 360.0):
                heading_bad += 1
        except Exception:
            heading_bad += 1
        try:
            spd = float(r[km["speed"]])
            if spd < -0.5:
                speeds_neg += 1
        except Exception:
            pass

    gps_frac = (valid_gps / n) if n else 0.0
    metrics["gps_fraction"] = gps_frac
    if gps_frac < 0.95:
        warnings.append("GPS valid fraction < 0.95")
    if heading_bad > n * 0.1:
        warnings.append(f"Many invalid headings {heading_bad}/{n}")
    if speeds_neg:
        warnings.append(f"Negative speeds in {speeds_neg} rows")
    if ts_local and not _monotonic(ts_local):
        issues.append("timestamp_local_ms not monotonic")
    md = _median_delta_ms(ts_local)
    if md is not None:
        metrics["median_dt_ms"] = md
        if abs(md - (1000 / sample_rate_hz)) > 50:
            warnings.append("Median dt not near expected sample rate")

    return FileReport(str(path), FileType.TX if expect_tx else FileType.RX, n, issues, warnings, metrics)


def validate_training_csv(path: Path, sample_rate_hz: int, warmup_s: int) -> FileReport:
    issues: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    hdr, rows = _read_rows(path)
    need = [
        "timestamp_ms","vehicle_id","lat","lon","alt","speed","heading","long_accel","lat_accel",
        "accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z","gps_valid","gps_age_ms"
    ]
    if set(need) - set(hdr):
        warnings.append("Training CSV missing some expected columns")
    km = _lower_keys(hdr)
    n = len(rows)
    metrics["rows"] = n
    valid_gps = 0
    ages: List[int] = []
    accel_x: List[float] = []
    accel_y: List[float] = []
    accel_z: List[float] = []
    ts: List[int] = []
    for r in rows:
        try:
            lt = float(r[km["lat"]]); ln = float(r[km["lon"]])
            if _is_valid_lat_lon(lt, ln):
                valid_gps += 1
        except Exception:
            pass
        try:
            ages.append(int(float(r[km["gps_age_ms"]])))
        except Exception:
            pass
        for key, arr in (("accel_x", accel_x),("accel_y", accel_y),("accel_z", accel_z)):
            try:
                arr.append(float(r[km[key]]))
            except Exception:
                pass
        try:
            ts.append(int(float(r[km["timestamp_ms"]])))
        except Exception:
            pass
    gps_frac = (valid_gps / n) if n else 0.0
    metrics["gps_fraction"] = gps_frac
    if gps_frac < 0.95:
        warnings.append("GPS valid fraction < 0.95")
    if ts and not _monotonic(ts):
        issues.append("timestamp_ms not monotonic")
    md = _median_delta_ms(ts)
    if md is not None:
        metrics["median_dt_ms"] = md
        if abs(md - (1000 / sample_rate_hz)) > 50:
            warnings.append("Median dt not near expected sample rate")
    # IMU baseline over warmup window
    if ts and accel_z:
        t0 = ts[0]
        cutoff = t0 + warmup_s * 1000
        mask = [t <= cutoff for t in ts]
        def _stats(vals: List[float], m: List[bool]) -> Tuple[float, float]:
            data = [v for v, keep in zip(vals, m) if keep]
            if not data:
                return (float("nan"), float("nan"))
            return float(np.mean(data)), float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        mean_z, std_z = _stats(accel_z, mask)
        mean_x, std_x = _stats(accel_x, mask)
        mean_y, std_y = _stats(accel_y, mask)
        metrics.update({
            "imu_z_mean": mean_z, "imu_z_std": std_z,
            "imu_x_std": std_x, "imu_y_std": std_y,
        })
        if not (6.8 <= mean_z <= 12.8):
            warnings.append("accel_z mean not near 9.81±3.0 in warmup")
        if std_x > 0.5 or std_y > 0.5:
            warnings.append("High accel std during warmup (>0.5 m/s^2)")

    return FileReport(str(path), FileType.TRAINING, n, issues, warnings, metrics)


def summarize_convoy_presence(paths: List[Path]) -> Tuple[bool, List[str]]:
    names = [p.name.lower() for p in paths]
    need = ["v001_tx_", "v001_rx_", "v002_tx_", "v002_rx_", "v003_tx_", "v003_rx_"]
    missing = [tag for tag in need if not any(tag in n for n in names)]
    return (len(missing) == 0, missing)


def build_markdown(report: SessionReport) -> str:
    lines = []
    lines.append(f"# Recording Validation Summary\n")
    lines.append(f"Mode: {report.mode}\n")
    lines.append(f"Status: {report.status}\n")
    if report.reasons:
        lines.append("\n## Reasons\n")
        for r in report.reasons:
            lines.append(f"- {r}\n")
    lines.append("\n## Files\n")
    lines.append("| File | Type | Rows | Issues | Warnings |\n")
    lines.append("|---|---:|---:|---|---|\n")
    for f in report.files:
        issues = "; ".join(f.issues) if f.issues else ""
        warns = "; ".join(f.warnings) if f.warnings else ""
        lines.append(f"| {Path(f.path).name} | {f.type} | {f.rows} | {issues} | {warns} |\n")
    if report.summary:
        lines.append("\n## Metrics\n")
        for k, v in sorted(report.summary.items()):
            lines.append(f"- {k}: {v}\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate field recording logs and emit reports")
    ap.add_argument("--input", required=True, type=Path, help="Session directory or CSV file")
    ap.add_argument("--out-dir", required=True, type=Path, help="Directory for reports")
    ap.add_argument("--mode", default=Mode.AUTO, choices=[Mode.AUTO, Mode.RTT, Mode.CONVOY, Mode.COMBINED])
    ap.add_argument("--sample-rate-hz", type=int, default=10)
    ap.add_argument("--stationary-warmup-s", type=int, default=60)
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = ap.parse_args()

    csvs = list_csvs(args.input)
    if not csvs:
        print("No CSV files found", file=sys.stderr)
        sys.exit(2)

    mode = args.mode if args.mode != Mode.AUTO else detect_mode(csvs)

    file_reports: List[FileReport] = []
    reasons: List[str] = []
    summary: Dict[str, float] = {}

    if mode == Mode.RTT:
        # Prefer files named rtt_log.csv; else validate any RTT-typed files
        rtt_files = [p for p in csvs if detect_file_type(read_header(p), p.name) == FileType.RTT or p.name.lower().endswith("rtt_log.csv")]
        if not rtt_files:
            reasons.append("No RTT CSV files detected")
        for p in rtt_files:
            file_reports.append(validate_rtt_csv(p, args.sample_rate_hz))
        # Aggregate summary
        rtts = [fr.metrics.get("rtt_mean") for fr in file_reports if fr.type == FileType.RTT and fr.metrics.get("rtt_mean") is not None]
        if rtts:
            summary["rtt_mean_avg"] = float(np.nanmean(np.array(rtts, dtype=float)))
        losses = [fr.metrics.get("loss_rate") for fr in file_reports if fr.metrics.get("loss_rate") is not None]
        if losses:
            summary["loss_rate_avg"] = float(np.nanmean(np.array(losses, dtype=float)))

    elif mode == Mode.CONVOY:
        ok, miss = summarize_convoy_presence(csvs)
        if not ok:
            reasons.append("Missing expected convoy files: " + ", ".join(miss))
        for p in csvs:
            ft = detect_file_type(read_header(p), p.name)
            if ft in (FileType.TX, FileType.RX):
                file_reports.append(validate_tx_rx_csv(p, expect_tx=(ft==FileType.TX), sample_rate_hz=args.sample_rate_hz))
        # Simple loss estimate per vehicle via TX vs RX counts
        by_vehicle: Dict[str, Dict[str, int]] = {}
        for fr in file_reports:
            name = Path(fr.path).name.lower()
            vid = "vxxx"
            for tag in ("v001","v002","v003"):
                if tag in name:
                    vid = tag
                    break
            kind = "tx" if fr.type == FileType.TX else "rx"
            by_vehicle.setdefault(vid, {}).setdefault(kind, 0)
            by_vehicle[vid][kind] += fr.rows
        for vid, counts in by_vehicle.items():
            tx = counts.get("tx", 0)
            rx = counts.get("rx", 0)
            if tx > 0:
                loss = max(0.0, 1.0 - (rx / tx))
                summary[f"{vid}_loss_estimate"] = round(loss, 3)
                if loss > 0.6:
                    reasons.append(f"{vid} loss > 60%")
                elif loss > 0.3:
                    # warning level aggregated later
                    pass

    else:  # COMBINED
        combined = [p for p in csvs if p.name.lower().startswith("scenario_") or detect_file_type(read_header(p), p.name) in (FileType.TRAINING, FileType.COMBINED)]
        if not combined:
            reasons.append("No combined/training CSV detected")
        for p in combined:
            file_reports.append(validate_training_csv(p, args.sample_rate_hz, args.stationary_warmup_s))

    # Determine status
    any_issue = any(fr.issues for fr in file_reports) or bool(reasons)
    any_warn = any(fr.warnings for fr in file_reports)
    status = "pass"
    if any_issue:
        status = "fail"
    elif any_warn:
        status = "warn"

    # Strict mode escalates warns to fail
    if args.strict and status == "warn":
        status = "fail"
        reasons.append("Strict mode: warnings treated as failures")

    session = SessionReport(mode=mode, files=file_reports, summary=summary, status=status, reasons=reasons)

    # Write reports
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "validation_report.json"
    md_path = out_dir / "validation_summary.md"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump({
            "mode": session.mode,
            "status": session.status,
            "reasons": session.reasons,
            "summary": session.summary,
            "files": [
                {
                    "path": f.path,
                    "type": f.type,
                    "rows": f.rows,
                    "issues": f.issues,
                    "warnings": f.warnings,
                    "metrics": f.metrics,
                } for f in session.files
            ],
        }, jf, indent=2, sort_keys=True)
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write(build_markdown(session))

    # Exit code
    if session.status == "fail":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
