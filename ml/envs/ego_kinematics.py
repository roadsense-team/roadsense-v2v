"""
Simple 2D kinematic ego model for replay-based training.

Replaces SUMO + ActionApplicator for ReplayConvoyEnv.  The ego follows
the recorded road heading but its speed is controlled by the RL agent.
"""

import math

from .sumo_connection import VehicleState


class EgoKinematics:
    """
    Kinematic ego model for replay training.

    The RL action controls braking only (0 = coast, 1 = full brake).
    Position advances along the heading provided each step (road direction).
    """

    MAX_DECEL = 8.0   # m/s², matches ActionApplicator
    MAX_SPEED = 30.0  # m/s
    DT = 0.1          # 10 Hz, matches SUMO step

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        initial_speed: float,
        initial_heading: float,
    ):
        self.x = initial_x
        self.y = initial_y
        self.speed = max(0.0, initial_speed)
        self.heading = initial_heading
        self.acceleration = 0.0

    def step(self, action_value: float, heading_deg: float) -> float:
        """
        Apply RL action and advance one timestep.

        Args:
            action_value: RL output in [0, 1]. 0 = coast, 1 = full brake.
            heading_deg: Road heading at this timestep (from TX recording).

        Returns:
            Actual deceleration applied (m/s², positive = slowing down).
        """
        clamped = max(0.0, min(1.0, float(action_value)))
        requested_decel = clamped * self.MAX_DECEL

        old_speed = self.speed
        self.speed = max(0.0, self.speed - requested_decel * self.DT)
        self.acceleration = (self.speed - old_speed) / self.DT

        # Advance position along road heading
        self.heading = heading_deg
        heading_rad = math.radians(90.0 - heading_deg)  # SUMO → Cartesian
        self.x += self.speed * math.cos(heading_rad) * self.DT
        self.y += self.speed * math.sin(heading_rad) * self.DT

        actual_decel = (old_speed - self.speed) / self.DT
        return actual_decel

    def to_vehicle_state(self) -> VehicleState:
        """Convert to VehicleState for ObservationBuilder compatibility."""
        return VehicleState(
            vehicle_id="V001",
            x=self.x,
            y=self.y,
            speed=self.speed,
            acceleration=self.acceleration,
            heading=self.heading,
            lane_position=0.0,
        )
