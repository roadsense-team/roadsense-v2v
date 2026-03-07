"""
Action applicator for ConvoyEnv.
"""


class ActionApplicator:
    """
    Translates continuous RL action to SUMO vehicle speed control.

    Action semantics:
    - 0.0: release to car-following model (no intervention)
    - 1.0: full deceleration (MAX_DECEL)
    """

    MAX_DECEL = 8.0
    STEP_DT = 0.1
    RELEASE_THRESHOLD = 0.02
    EGO_VEHICLE_ID = "V001"

    def __init__(self) -> None:
        """Initialize ActionApplicator."""
        pass

    def get_deceleration(self, action_value: float) -> float:
        """Map action value to requested deceleration in m/s^2."""
        fraction = max(0.0, min(1.0, float(action_value)))
        return fraction * self.MAX_DECEL

    def apply(
        self,
        sumo: "SUMOConnection",
        action_value: float,
        cf_override: bool = False,
    ) -> float:
        """
        Apply action to ego vehicle.

        When action is near zero, releases speed control to SUMO's
        car-following model so the vehicle can accelerate naturally.

        When cf_override is True (hazard active), low actions hold current
        speed instead of releasing to CF, forcing the RL model to be the
        sole source of deceleration.

        Returns actual deceleration applied in m/s^2 (0.0 when released/held).
        """
        clamped = max(0.0, min(1.0, float(action_value)))

        if clamped <= self.RELEASE_THRESHOLD:
            if cf_override:
                current_state = sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
                sumo.set_vehicle_speed(self.EGO_VEHICLE_ID, current_state.speed)
            else:
                sumo.release_vehicle_speed(self.EGO_VEHICLE_ID)
            return 0.0

        requested_decel = clamped * self.MAX_DECEL

        current_state = sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_speed = current_state.speed

        speed_delta = requested_decel * self.STEP_DT
        new_speed = max(0.0, current_speed - speed_delta)
        sumo.set_vehicle_speed(self.EGO_VEHICLE_ID, new_speed)

        actual_speed_delta = current_speed - new_speed
        return actual_speed_delta / self.STEP_DT
