#!/usr/bin/env python3
"""
Quick GPS track visualization for RoadSense logs.

Supports CSVs produced by:
- hardware main logger (TX/RX characterization files with lat/lon columns)
- RTT sender logs (rtt_log.csv)
- Training session logs (single CSV per vehicle)

Outputs a PNG per input with the path plotted (lon vs lat),
highlighting start (green) and end (red).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_lat_lon(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        # Column detection (case-insensitive)
        cols = {c.lower(): c for c in reader.fieldnames}
        lat_key = cols.get("lat") or cols.get("latitude")
        lon_key = cols.get("lon") or cols.get("longitude")
        gps_valid_key = cols.get("gps_valid")

        if not lat_key or not lon_key:
            raise ValueError(
                f"Missing lat/lon columns in {path}. Found: {reader.fieldnames}"
            )

        lats: List[float] = []
        lons: List[float] = []
        for row in reader:
            try:
                lat = float(row[lat_key])
                lon = float(row[lon_key])
            except (KeyError, TypeError, ValueError):
                continue

            # Filter invalid coordinates
            if abs(lat) < 1e-6 and abs(lon) < 1e-6:
                continue

            # If present, honor gps_valid == 1
            if gps_valid_key is not None:
                try:
                    if int(float(row[gps_valid_key])) != 1:
                        continue
                except (TypeError, ValueError):
                    pass

            lats.append(lat)
            lons.append(lon)

        return np.array(lats, dtype=float), np.array(lons, dtype=float)


def plot_track(lat: np.ndarray, lon: np.ndarray, title: str, out_path: Path) -> None:
    if lat.size == 0:
        raise ValueError("No valid GPS points to plot")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lon, lat, "-", color="#1f77b4", linewidth=1.5, alpha=0.9)
    # Start/end markers
    ax.scatter(lon[0], lat[0], c="#2ca02c", s=40, label="start")
    ax.scatter(lon[-1], lat[-1], c="#d62728", s=40, label="end")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GPS track(s) from CSV log(s)")
    p.add_argument("inputs", nargs="+", type=Path, help="Input CSV file(s)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to each input's parent)",
    )
    p.add_argument(
        "--suffix",
        default="_track",
        help="Suffix appended to basename before .png (default: _track)",
    )
    args = p.parse_args()

    for inp in args.inputs:
        if not inp.exists():
            print(f"[WARN] Missing file: {inp}")
            continue
        try:
            lat, lon = read_lat_lon(inp)
            title = f"GPS Track: {inp.name} ({lat.size} pts)"
            out_dir = args.out_dir if args.out_dir else inp.parent
            out_path = out_dir / (inp.stem + args.suffix + ".png")
            plot_track(lat, lon, title, out_path)
            print(f"[OK] Saved {out_path}")
        except Exception as e:
            print(f"[ERROR] {inp}: {e}")


if __name__ == "__main__":
    main()

