"""
Action applicator for ConvoyEnv.
"""


class ActionApplicator:
    """
    Translates continuous RL action to SUMO vehicle speed control.

    Action semantics:
    - 0.0: no deceleration
    - 1.0: full deceleration (MAX_DECEL)
    """

    MAX_DECEL = 5.0
    STEP_DT = 0.1
    EGO_VEHICLE_ID = "V001"

    def __init__(self) -> None:
        """Initialize ActionApplicator."""
        pass

    def get_deceleration(self, action_value: float) -> float:
        """Map action value to requested deceleration in m/s^2."""
        fraction = max(0.0, min(1.0, float(action_value)))
        return fraction * self.MAX_DECEL

    def apply(self, sumo: "SUMOConnection", action_value: float) -> float:
        """
        Apply action to ego vehicle.

        Returns actual deceleration applied in m/s^2 after speed floor clipping.
        """
        requested_decel = self.get_deceleration(action_value)

        current_state = sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_speed = current_state.speed

        speed_delta = requested_decel * self.STEP_DT
        new_speed = max(0.0, current_speed - speed_delta)
        sumo.set_vehicle_speed(self.EGO_VEHICLE_ID, new_speed)

        actual_speed_delta = current_speed - new_speed
        return actual_speed_delta / self.STEP_DT
