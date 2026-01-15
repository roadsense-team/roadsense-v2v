"""
Observation builder for ConvoyEnv.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

from .sumo_connection import VehicleState


class ObservationBuilder:
    """
    Builds observation dict for variable-n peer environments.

    Keys:
        - ego: [speed/30, accel/10, heading/pi, peer_count/8]
        - peers: (MAX_PEERS, 6) peer features
        - peer_mask: (MAX_PEERS,) 1.0 for valid peers, 0.0 for padding
    """

    MAX_PEERS = 8
    MAX_SPEED = 30.0
    MAX_DISTANCE = 100.0
    MAX_ACCEL = 10.0
    STALENESS_THRESHOLD = 500.0

    def build(
        self,
        ego_state: VehicleState,
        peer_observations: List[Dict[str, float]],
        ego_pos: Tuple[float, float],
    ) -> Dict[str, np.ndarray]:
        """
        Build observation dict from ego state and variable peer list.

        Returns:
            Dict with 'ego', 'peers', and 'peer_mask' arrays
        """
        ego_heading_rad = math.radians(ego_state.heading)
        ego_heading_rad = (ego_heading_rad + math.pi) % (2 * math.pi) - math.pi

        peers = np.zeros((self.MAX_PEERS, 6), dtype=np.float32)
        peer_mask = np.zeros((self.MAX_PEERS,), dtype=np.float32)

        valid_count = 0
        for peer in peer_observations[: self.MAX_PEERS]:
            if not peer.get("valid", True):
                continue

            dx = peer["x"] - ego_pos[0]
            dy = peer["y"] - ego_pos[1]

            cos_h = math.cos(-ego_heading_rad)
            sin_h = math.sin(-ego_heading_rad)
            rel_x = dx * cos_h - dy * sin_h
            rel_y = dx * sin_h + dy * cos_h

            rel_speed = peer["speed"] - ego_state.speed
            peer_heading_rad = math.radians(peer["heading"])
            rel_heading = peer_heading_rad - ego_heading_rad
            rel_heading = (rel_heading + math.pi) % (2 * math.pi) - math.pi

            peers[valid_count] = np.array(
                [
                    rel_x / self.MAX_DISTANCE,
                    rel_y / self.MAX_DISTANCE,
                    rel_speed / self.MAX_SPEED,
                    rel_heading / math.pi,
                    peer["accel"] / self.MAX_ACCEL,
                    peer["age_ms"] / self.STALENESS_THRESHOLD,
                ],
                dtype=np.float32,
            )
            peer_mask[valid_count] = 1.0
            valid_count += 1

        ego = np.array(
            [
                ego_state.speed / self.MAX_SPEED,
                ego_state.acceleration / self.MAX_ACCEL,
                ego_heading_rad / math.pi,
                valid_count / self.MAX_PEERS,
            ],
            dtype=np.float32,
        )

        return {
            "ego": ego,
            "peers": peers,
            "peer_mask": peer_mask,
        }
