"""
Hazard injection utilities for ConvoyEnv.
"""
from typing import Optional
import random


class HazardInjector:
    """
    Injects hazard events mid-episode using TraCI.
    """

    HAZARD_PROBABILITY = 0.3
    HAZARD_WINDOW_START = 30
    HAZARD_WINDOW_END = 80

    EMERGENCY_BRAKE = "emergency_brake"
    HAZARD_TARGET_VEHICLE = "V002"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._hazard_step: Optional[int] = None
        self._hazard_injected = False
        self._episode_will_have_hazard = False
        self._reset_state()

    def seed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self._episode_will_have_hazard = (
            self._rng.random() < self.HAZARD_PROBABILITY
        )

        if self._episode_will_have_hazard:
            self._hazard_step = self._rng.randint(
                self.HAZARD_WINDOW_START,
                self.HAZARD_WINDOW_END,
            )
        else:
            self._hazard_step = None

        self._hazard_injected = False

    def reset(self) -> None:
        self._reset_state()

    def maybe_inject(self, step: int, sumo: "SUMOConnection") -> bool:
        if self._hazard_injected:
            return False
        if not self._episode_will_have_hazard:
            return False
        if self._hazard_step is None or step != self._hazard_step:
            return False
        if not sumo.is_vehicle_active(self.HAZARD_TARGET_VEHICLE):
            return False

        sumo.set_vehicle_speed(self.HAZARD_TARGET_VEHICLE, 0.0)
        self._hazard_injected = True

        return True

    def is_in_hazard_window(self, step: int) -> bool:
        return self.HAZARD_WINDOW_START <= step <= self.HAZARD_WINDOW_END

    @property
    def hazard_injected(self) -> bool:
        return self._hazard_injected
