"""
Action applicator for ConvoyEnv.
"""


class ActionApplicator:
    """
    Translates discrete RL actions to SUMO vehicle speed control.

    The ego vehicle (V001) receives speed reduction commands based on
    the RL agent's action choice. Actions represent warning/response levels.
    """

    MAINTAIN = 0
    CAUTION = 1
    BRAKE = 2
    EMERGENCY = 3

    ACTION_DECEL = {
        MAINTAIN: 0.0,
        CAUTION: 0.5,
        BRAKE: 2.0,
        EMERGENCY: 4.5,
    }

    EGO_VEHICLE_ID = "V001"

    def __init__(self) -> None:
        """Initialize ActionApplicator."""
        pass

    def get_deceleration(self, action: int) -> float:
        """
        Get the speed reduction for a given action.

        Args:
            action: Integer action [0, 3]

        Returns:
            Speed reduction in m/s

        Raises:
            ValueError: If action not in [0, 3]
        """
        if action not in self.ACTION_DECEL:
            raise ValueError(f"Invalid action {action}. Must be in [0, 1, 2, 3]")
        return self.ACTION_DECEL[action]

    def apply(self, sumo: "SUMOConnection", action: int) -> float:
        """
        Apply action to ego vehicle.

        Reduces ego vehicle speed by the action's deceleration value.
        Speed is floored at 0.0 (cannot go negative).

        Args:
            sumo: SUMOConnection instance
            action: Integer action [0, 3]

        Returns:
            Actual deceleration applied (for reward calculation)
        """
        decel = self.get_deceleration(action)

        current_state = sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_speed = current_state.speed

        new_speed = max(0.0, current_speed - decel)
        sumo.set_vehicle_speed(self.EGO_VEHICLE_ID, new_speed)

        return current_speed - new_speed
