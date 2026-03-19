"""
Trajectory loader and augmentor for real-data replay training.

Parses V001 TX/RX CSVs into time-indexed snapshots at 10 Hz,
converts GPS → meters, and applies augmentations for diverse
training episodes.
"""

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


METERS_PER_DEG_LAT = 111_000.0
STEP_MS = 100  # 10 Hz
STALENESS_THRESHOLD_MS = 500.0
SELF_RX_VEHICLE_ID = "V001"


@dataclass
class PeerSnapshot:
    """One peer observation at one timestep."""
    vehicle_id: str
    x: float       # meters
    y: float       # meters
    speed: float   # m/s
    heading: float  # degrees, SUMO convention
    accel: float   # m/s², forward axis
    age_ms: float  # staleness


@dataclass
class TrajectorySnapshot:
    """One timestep of a recorded episode."""
    time_ms: int
    ego_x: float       # meters
    ego_y: float       # meters
    ego_speed: float   # m/s
    ego_heading: float  # degrees, SUMO convention
    ego_accel: float   # m/s², forward axis
    peers: List[PeerSnapshot] = field(default_factory=list)


@dataclass
class AugmentConfig:
    """Configuration for trajectory augmentation."""
    accel_scale: Tuple[float, float] = (1.0, 1.0)
    speed_scale: Tuple[float, float] = (1.0, 1.0)
    gps_noise_m: Tuple[float, float] = (0.0, 0.0)
    drop_rate: Tuple[float, float] = (0.0, 0.0)
    ego_offset_m: Tuple[float, float] = (0.0, 0.0)
    inject_brake: bool = False
    inject_brake_accel: Tuple[float, float] = (-9.0, -4.0)
    inject_brake_duration_s: Tuple[float, float] = (0.5, 2.0)


def gps_to_meters(lat: float, lon: float) -> Tuple[float, float]:
    """Convert GPS lat/lon to meters (x=East, y=North)."""
    return lon * METERS_PER_DEG_LAT, lat * METERS_PER_DEG_LAT


class ReplayTrajectory:
    """Loads and augments a single recording."""

    def __init__(
        self,
        tx_path: str,
        rx_path: str,
        forward_axis: str = "y",
    ):
        self.tx_path = Path(tx_path)
        self.rx_path = Path(rx_path)
        self.forward_axis = forward_axis
        self._snapshots: Optional[List[TrajectorySnapshot]] = None

    def load(self) -> List[TrajectorySnapshot]:
        """Parse CSVs into time-indexed snapshots at 10 Hz."""
        if self._snapshots is not None:
            return self._snapshots

        accel_col = "accel_y" if self.forward_axis == "y" else "accel_x"

        # Parse TX (ego)
        tx_by_time: Dict[int, dict] = {}
        with open(self.tx_path, newline="") as f:
            for r in csv.DictReader(f):
                t = int(r["timestamp_local_ms"])
                tx_by_time[t] = r

        # Parse RX (peers), grouped by time
        rx_by_time: Dict[int, List[dict]] = {}
        with open(self.rx_path, newline="") as f:
            for r in csv.DictReader(f):
                vid = r.get("from_vehicle_id", r.get("vehicle_id", "")).strip()
                if vid == SELF_RX_VEHICLE_ID:
                    continue
                t = int(r["timestamp_local_ms"])
                rx_by_time.setdefault(t, []).append(r)

        if not tx_by_time:
            self._snapshots = []
            return self._snapshots

        # Build 10 Hz grid
        t_min = min(tx_by_time.keys())
        t_max = max(tx_by_time.keys())
        grid_times = list(range(t_min, t_max + 1, STEP_MS))

        snapshots: List[TrajectorySnapshot] = []
        sorted_tx_times = sorted(tx_by_time.keys())
        sorted_rx_times = sorted(rx_by_time.keys())

        for grid_t in grid_times:
            # Find nearest TX row
            tx_row = self._nearest_row(sorted_tx_times, tx_by_time, grid_t)
            if tx_row is None:
                continue

            ego_x, ego_y = gps_to_meters(
                float(tx_row["lat"]), float(tx_row["lon"])
            )

            # Gather RX peers within staleness window
            peers = self._gather_peers(
                rx_by_time, sorted_rx_times, grid_t, accel_col
            )

            snapshots.append(TrajectorySnapshot(
                time_ms=grid_t,
                ego_x=ego_x,
                ego_y=ego_y,
                ego_speed=max(0.0, float(tx_row["speed"])),
                ego_heading=float(tx_row["heading"]),
                ego_accel=float(tx_row[accel_col]),
                peers=peers,
            ))

        self._snapshots = snapshots
        return snapshots

    def _nearest_row(
        self,
        sorted_times: List[int],
        by_time: Dict[int, dict],
        target: int,
    ) -> Optional[dict]:
        """Find the TX row with timestamp closest to target."""
        if not sorted_times:
            return None
        idx = self._bisect_closest(sorted_times, target)
        t = sorted_times[idx]
        if abs(t - target) > STEP_MS:
            return None
        return by_time[t]

    def _gather_peers(
        self,
        rx_by_time: Dict[int, List[dict]],
        sorted_rx_times: List[int],
        grid_t: int,
        accel_col: str,
    ) -> List[PeerSnapshot]:
        """Gather freshest peer per vehicle within staleness window."""
        freshest: Dict[str, Tuple[dict, int]] = {}  # vid → (row, rx_time)

        for rx_t in sorted_rx_times:
            age = grid_t - rx_t
            if age < 0 or age > STALENESS_THRESHOLD_MS:
                continue
            for r in rx_by_time.get(rx_t, []):
                vid = r.get("from_vehicle_id", r.get("vehicle_id", "")).strip()
                if vid == SELF_RX_VEHICLE_ID:
                    continue
                prev = freshest.get(vid)
                if prev is None or rx_t > prev[1]:
                    freshest[vid] = (r, rx_t)

        peers = []
        for vid, (r, rx_t) in freshest.items():
            px, py = gps_to_meters(float(r["lat"]), float(r["lon"]))
            peers.append(PeerSnapshot(
                vehicle_id=vid,
                x=px,
                y=py,
                speed=max(0.0, float(r["speed"])),
                heading=float(r["heading"]),
                accel=float(r[accel_col]),
                age_ms=float(grid_t - rx_t),
            ))
        return peers

    @staticmethod
    def _bisect_closest(sorted_list: List[int], target: int) -> int:
        """Binary search for the index of the closest value."""
        lo, hi = 0, len(sorted_list) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_list[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(sorted_list[lo - 1] - target) < abs(sorted_list[lo] - target):
            return lo - 1
        return lo

    def augment(
        self,
        rng: np.random.Generator,
        config: AugmentConfig,
    ) -> List[TrajectorySnapshot]:
        """Return an augmented copy of the trajectory."""
        base = self.load()
        if not base:
            return []

        # Sample augmentation parameters
        accel_factor = rng.uniform(*config.accel_scale)
        speed_factor = rng.uniform(*config.speed_scale)
        noise_std = rng.uniform(*config.gps_noise_m)
        drop_rate = rng.uniform(*config.drop_rate)
        ego_offset = rng.uniform(*config.ego_offset_m)

        # Compute offset vector along initial heading
        h_rad = math.radians(90.0 - base[0].ego_heading)
        off_x = ego_offset * math.cos(h_rad)
        off_y = ego_offset * math.sin(h_rad)

        result: List[TrajectorySnapshot] = []
        for snap in base:
            # Augment peers
            aug_peers: List[PeerSnapshot] = []
            for p in snap.peers:
                if drop_rate > 0 and rng.random() < drop_rate:
                    continue
                nx = p.x + (rng.normal(0, noise_std) if noise_std > 0 else 0)
                ny = p.y + (rng.normal(0, noise_std) if noise_std > 0 else 0)
                aug_peers.append(PeerSnapshot(
                    vehicle_id=p.vehicle_id,
                    x=nx,
                    y=ny,
                    speed=max(0.0, p.speed * speed_factor),
                    heading=p.heading,
                    accel=p.accel * accel_factor,
                    age_ms=p.age_ms,
                ))

            result.append(TrajectorySnapshot(
                time_ms=snap.time_ms,
                ego_x=snap.ego_x + off_x,
                ego_y=snap.ego_y + off_y,
                ego_speed=max(0.0, snap.ego_speed * speed_factor),
                ego_heading=snap.ego_heading,
                ego_accel=snap.ego_accel * accel_factor,
                peers=aug_peers,
            ))

        # Synthetic hazard injection
        if config.inject_brake and result:
            result = self._inject_synthetic_hazard(rng, result, config)

        return result

    def _inject_synthetic_hazard(
        self,
        rng: np.random.Generator,
        snapshots: List[TrajectorySnapshot],
        config: AugmentConfig,
    ) -> List[TrajectorySnapshot]:
        """Insert a braking event into a calm recording."""
        n = len(snapshots)
        # Pick onset in [30%, 70%] of episode
        onset_idx = rng.integers(int(n * 0.3), int(n * 0.7))

        # Find all unique peer IDs at onset
        peers_at_onset = snapshots[onset_idx].peers
        if not peers_at_onset:
            return snapshots

        target_vid = rng.choice([p.vehicle_id for p in peers_at_onset])

        # Braking profile parameters
        peak_accel = rng.uniform(*config.inject_brake_accel)
        duration_s = rng.uniform(*config.inject_brake_duration_s)
        duration_steps = max(1, int(duration_s / (STEP_MS / 1000.0)))

        for i in range(onset_idx, min(n, onset_idx + duration_steps)):
            frac = (i - onset_idx) / max(1, duration_steps - 1)
            brake_accel = peak_accel * min(1.0, frac * 2.0)  # ramp up

            snap = snapshots[i]
            new_peers = []
            for p in snap.peers:
                if p.vehicle_id == target_vid:
                    # Apply braking: decelerate speed
                    speed_loss = abs(brake_accel) * (STEP_MS / 1000.0)
                    new_speed = max(0.0, p.speed - speed_loss * (i - onset_idx))
                    new_peers.append(PeerSnapshot(
                        vehicle_id=p.vehicle_id,
                        x=p.x,
                        y=p.y,
                        speed=new_speed,
                        heading=p.heading,
                        accel=brake_accel,
                        age_ms=p.age_ms,
                    ))
                else:
                    new_peers.append(p)
            snapshots[i] = TrajectorySnapshot(
                time_ms=snap.time_ms,
                ego_x=snap.ego_x,
                ego_y=snap.ego_y,
                ego_speed=snap.ego_speed,
                ego_heading=snap.ego_heading,
                ego_accel=snap.ego_accel,
                peers=new_peers,
            )

        return snapshots
