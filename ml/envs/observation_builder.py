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

    def is_in_cone(
        self,
        ego_heading_deg: float,
        ego_pos: Tuple[float, float],
        peer_x: float,
        peer_y: float,
        half_angle_deg: float = 45.0,
    ) -> bool:
        """
        Return True when peer lies inside ego's forward cone.
        
        Uses SUMO coordinates (0=North, clockwise).
        """
        dx = peer_x - ego_pos[0]
        dy = peer_y - ego_pos[1]
        
        # Bearing in standard Cartesian (0=East, counter-clockwise)
        bearing_rad = math.atan2(dy, dx)
        bearing_deg = math.degrees(bearing_rad)
        
        # Convert Cartesian bearing to SUMO angle (0=North, clockwise)
        # SUMO_angle = 90 - Cartesian_angle
        peer_sumo_angle = (90.0 - bearing_deg + 360.0) % 360.0
        
        # Calculate angular difference in SUMO frame
        diff = (peer_sumo_angle - ego_heading_deg + 180.0) % 360.0 - 180.0
        return abs(diff) <= half_angle_deg

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
        ego_heading_deg = ego_state.heading
        
        filtered_peers = []
        for peer in peer_observations:
            if not peer.get("valid", True):
                continue
            if self.is_in_cone(ego_heading_deg, ego_pos, peer["x"], peer["y"]):
                filtered_peers.append(peer)

        peers = np.zeros((self.MAX_PEERS, 6), dtype=np.float32)
        peer_mask = np.zeros((self.MAX_PEERS,), dtype=np.float32)

        valid_count = 0
        # Ego heading in Cartesian for rotation
        ego_phi_rad = math.radians(90.0 - ego_heading_deg)

        for peer in filtered_peers[: self.MAX_PEERS]:
            dx = peer["x"] - ego_pos[0]
            dy = peer["y"] - ego_pos[1]

            # Rotate to ego's local frame (rel_x = forward, rel_y = left)
            cos_phi = math.cos(ego_phi_rad)
            sin_phi = math.sin(ego_phi_rad)
            rel_x = dx * cos_phi + dy * sin_phi
            rel_y = -dx * sin_phi + dy * cos_phi

            rel_speed = peer["speed"] - ego_state.speed
            rel_heading = (peer["heading"] - ego_heading_deg + 180.0) % 360.0 - 180.0

            peers[valid_count] = np.array(
                [
                    rel_x / self.MAX_DISTANCE,
                    rel_y / self.MAX_DISTANCE,
                    rel_speed / self.MAX_SPEED,
                    rel_heading / 180.0,  # Normalized to [-1, 1]
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
                ((ego_heading_deg + 180.0) % 360.0 - 180.0) / 180.0,
                valid_count / self.MAX_PEERS,
            ],
            dtype=np.float32,
        )

        return {
            "ego": ego,
            "peers": peers,
            "peer_mask": peer_mask,
        }
