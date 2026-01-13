"""
Observation builder for ConvoyEnv.
"""

import math
from typing import Dict, List

import numpy as np

from envs.sumo_connection import VehicleState
from espnow_emulator.espnow_emulator import ESPNOWEmulator


class ObservationBuilder:
    """
    Builds normalized observation array from emulator output.

    Observation space (11 features):
    [0]    ego_speed (normalized by max_speed)
    [1-5]  v002: rel_dist, rel_speed, accel, age_norm, valid
    [6-10] v003: rel_dist, rel_speed, accel, age_norm, valid
    """

    MAX_SPEED = 30.0
    MAX_DISTANCE = 100.0
    MAX_ACCEL = 10.0
    STALENESS_THRESHOLD = 500.0
    MISSING_MESSAGE_AGE_MS = 9999

    def build(self, ego_state: VehicleState, emulator_obs: Dict) -> np.ndarray:
        """
        Build observation array from ego state and emulator output.

        Returns:
            np.ndarray of shape (11,) with dtype float32
        """
        ego_speed_norm = ego_state.speed / self.MAX_SPEED
        v002 = self._build_peer_features("v002", ego_state, emulator_obs)
        v003 = self._build_peer_features("v003", ego_state, emulator_obs)

        return np.array([ego_speed_norm] + v002 + v003, dtype=np.float32)

    def _build_peer_features(
        self, vehicle_key: str, ego_state: VehicleState, emulator_obs: Dict
    ) -> List[float]:
        prefix = vehicle_key.lower()
        lat = emulator_obs.get(f"{prefix}_lat", 0.0)
        lon = emulator_obs.get(f"{prefix}_lon", 0.0)
        speed = emulator_obs.get(f"{prefix}_speed", 0.0)
        accel_x = emulator_obs.get(f"{prefix}_accel_x", 0.0)
        age_ms = emulator_obs.get(f"{prefix}_age_ms", self.MISSING_MESSAGE_AGE_MS)
        valid = bool(emulator_obs.get(f"{prefix}_valid", False))

        if not valid and age_ms >= self.MISSING_MESSAGE_AGE_MS:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        peer_x = lon * ESPNOWEmulator.METERS_PER_DEG_LAT
        peer_y = lat * ESPNOWEmulator.METERS_PER_DEG_LAT
        rel_dist = math.hypot(ego_state.x - peer_x, ego_state.y - peer_y)
        rel_speed = speed - ego_state.speed

        rel_dist_norm = rel_dist / self.MAX_DISTANCE
        rel_speed_norm = rel_speed / self.MAX_SPEED
        accel_norm = accel_x / self.MAX_ACCEL
        age_norm = age_ms / self.STALENESS_THRESHOLD
        valid_flag = 1.0 if valid else 0.0

        return [rel_dist_norm, rel_speed_norm, accel_norm, age_norm, valid_flag]
