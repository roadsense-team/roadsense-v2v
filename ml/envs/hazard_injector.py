"""
Hazard injection utilities for ConvoyEnv.
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple


class HazardInjector:
    """
    Injects hazard events mid-episode using TraCI with configurable source
    selection strategies.
    """

    # Run 017 diagnostic: deterministic hazard every episode at fixed step.
    HAZARD_PROBABILITY = 1.0
    HAZARD_WINDOW_START = 150
    HAZARD_WINDOW_END = 350
    DEFAULT_HAZARD_STEP = 200

    STEP_LENGTH = 0.1  # seconds per simulation step
    BRAKING_DURATION_MIN = 2.0  # seconds
    BRAKING_DURATION_MAX = 4.0  # seconds

    EMERGENCY_BRAKE = "emergency_brake"
    EGO_VEHICLE_ID = "V001"
    TARGET_STRATEGY_NEAREST = "nearest"
    TARGET_STRATEGY_UNIFORM_FRONT_PEERS = "uniform_front_peers"
    TARGET_STRATEGY_FIXED_VEHICLE_ID = "fixed_vehicle_id"
    TARGET_STRATEGY_FIXED_RANK_AHEAD = "fixed_rank_ahead"
    ALLOWED_TARGET_STRATEGIES = {
        TARGET_STRATEGY_NEAREST,
        TARGET_STRATEGY_UNIFORM_FRONT_PEERS,
        TARGET_STRATEGY_FIXED_VEHICLE_ID,
        TARGET_STRATEGY_FIXED_RANK_AHEAD,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        target_strategy: str = TARGET_STRATEGY_NEAREST,
        fixed_vehicle_id: Optional[str] = None,
        fixed_rank_ahead: Optional[int] = None,
    ) -> None:
        self._validate_target_strategy(target_strategy)
        self._rng = random.Random(seed)
        self._default_target_strategy = target_strategy
        self._default_fixed_vehicle_id = fixed_vehicle_id
        self._default_fixed_rank_ahead = fixed_rank_ahead

        self._hazard_step: Optional[int] = None
        self._hazard_injected = False
        self._episode_will_have_hazard = False
        self._hazard_target: Optional[str] = None
        self._hazard_source_rank_ahead: Optional[int] = None
        self._reset_state()

    def seed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _validate_target_strategy(self, target_strategy: str) -> None:
        if target_strategy not in self.ALLOWED_TARGET_STRATEGIES:
            raise ValueError(
                f"Invalid target_strategy '{target_strategy}'. "
                f"Expected one of {sorted(self.ALLOWED_TARGET_STRATEGIES)}"
            )

    def _reset_state(self, options: Optional[Dict[str, Any]] = None) -> None:
        opts = options or {}

        target_strategy = str(
            opts.get("target_strategy", self._default_target_strategy)
        )
        self._validate_target_strategy(target_strategy)
        self._target_strategy = target_strategy

        fixed_vehicle_id = opts.get("fixed_vehicle_id", self._default_fixed_vehicle_id)
        self._fixed_vehicle_id = (
            str(fixed_vehicle_id) if fixed_vehicle_id is not None else None
        )

        fixed_rank_ahead = opts.get("fixed_rank_ahead", self._default_fixed_rank_ahead)
        self._fixed_rank_ahead = (
            int(fixed_rank_ahead) if fixed_rank_ahead is not None else None
        )

        force_hazard_opt = opts.get("force_hazard", None)
        if force_hazard_opt is None:
            self._episode_will_have_hazard = (
                self._rng.random() < self.HAZARD_PROBABILITY
            )
        else:
            self._episode_will_have_hazard = bool(force_hazard_opt)

        hazard_step_opt = opts.get("hazard_step", None)
        if not self._episode_will_have_hazard:
            self._hazard_step = None
        elif hazard_step_opt is not None:
            hazard_step = int(hazard_step_opt)
            if not self.is_in_hazard_window(hazard_step):
                raise ValueError(
                    f"hazard_step {hazard_step} outside injection window "
                    f"[{self.HAZARD_WINDOW_START}, {self.HAZARD_WINDOW_END}]"
                )
            self._hazard_step = hazard_step
        else:
            self._hazard_step = self.DEFAULT_HAZARD_STEP

        self._hazard_injected = False
        self._hazard_target = None
        self._hazard_source_rank_ahead = None
        self._hazard_injection_attempted = False
        self._hazard_injection_failed = False
        self._hazard_injection_failed_reason: Optional[str] = None
        self._braking_duration: Optional[float] = None
        self._slowdown_end_step: Optional[int] = None

    def reset(self, options: Optional[Dict[str, Any]] = None) -> None:
        self._reset_state(options=options)

    def _front_peers_with_rank(
        self,
        sumo: "SUMOConnection",
    ) -> List[Tuple[str, int]]:
        """
        Return front peers sorted nearest-first with 1-based rank ahead.
        """
        vehicle_ids = sumo.get_active_vehicle_ids()
        if self.EGO_VEHICLE_ID not in vehicle_ids:
            return []

        ego_state = sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        heading_rad = math.radians(float(ego_state.heading))
        forward_x = math.sin(heading_rad)
        forward_y = math.cos(heading_rad)
        front_peers: List[Tuple[str, float]] = []

        for vid in vehicle_ids:
            if vid == self.EGO_VEHICLE_ID:
                continue

            peer_state = sumo.get_vehicle_state(vid)
            dx = peer_state.x - ego_state.x
            dy = peer_state.y - ego_state.y
            longitudinal = dx * forward_x + dy * forward_y
            if longitudinal <= 0.0:
                continue
            dist = math.hypot(dx, dy)
            front_peers.append((vid, dist))

        front_peers.sort(key=lambda item: item[1])
        return [(vehicle_id, rank + 1) for rank, (vehicle_id, _) in enumerate(front_peers)]

    def _select_target(
        self,
        front_peers_with_rank: List[Tuple[str, int]],
    ) -> Tuple[Optional[str], Optional[int]]:
        if not front_peers_with_rank:
            return None, None

        if self._target_strategy == self.TARGET_STRATEGY_NEAREST:
            return front_peers_with_rank[0]

        if self._target_strategy == self.TARGET_STRATEGY_UNIFORM_FRONT_PEERS:
            return self._rng.choice(front_peers_with_rank)

        if self._target_strategy == self.TARGET_STRATEGY_FIXED_VEHICLE_ID:
            if self._fixed_vehicle_id is None:
                return None, None
            for vehicle_id, rank_ahead in front_peers_with_rank:
                if vehicle_id == self._fixed_vehicle_id:
                    return vehicle_id, rank_ahead
            return None, None

        if self._target_strategy == self.TARGET_STRATEGY_FIXED_RANK_AHEAD:
            if self._fixed_rank_ahead is None or self._fixed_rank_ahead <= 0:
                return None, None
            for vehicle_id, rank_ahead in front_peers_with_rank:
                if rank_ahead == self._fixed_rank_ahead:
                    return vehicle_id, rank_ahead
            return None, None

        return None, None

    def maybe_inject(self, step: int, sumo: "SUMOConnection") -> bool:
        if self._hazard_injected:
            return False
        if not self._episode_will_have_hazard:
            return False
        if self._hazard_step is None or step != self._hazard_step:
            return False

        self._hazard_injection_attempted = True

        front_peers_with_rank = self._front_peers_with_rank(sumo)
        target, source_rank_ahead = self._select_target(front_peers_with_rank)
        if target is None or source_rank_ahead is None:
            self._hazard_injection_failed = True
            if not front_peers_with_rank:
                self._hazard_injection_failed_reason = "no_front_peers"
            else:
                self._hazard_injection_failed_reason = (
                    f"target_not_found_strategy={self._target_strategy}"
                )
            return False

        braking_duration = self._rng.uniform(
            self.BRAKING_DURATION_MIN, self.BRAKING_DURATION_MAX
        )
        sumo.slow_down(target, 0.0, braking_duration)
        self._braking_duration = braking_duration
        self._slowdown_end_step = step + int(
            round(braking_duration / self.STEP_LENGTH)
        )
        self._hazard_injected = True
        self._hazard_target = target
        self._hazard_source_rank_ahead = source_rank_ahead

        return True

    def maintain_hazard(self, step: int, sumo: "SUMOConnection") -> None:
        """Pin hazard target at speed 0 once the gradual slowdown completes.

        Must be called every step from the environment. After the slowDown
        duration expires, SUMO would return the vehicle to CF control and
        it would accelerate again. This pins speed=0 to keep the hazard
        active for the rest of the episode (same reward dynamics as before).
        """
        if (
            self._hazard_injected
            and self._hazard_target is not None
            and self._slowdown_end_step is not None
            and step >= self._slowdown_end_step
        ):
            sumo.set_vehicle_speed(self._hazard_target, 0.0)
            self._slowdown_end_step = None  # Only pin once

    def is_in_hazard_window(self, step: int) -> bool:
        return self.HAZARD_WINDOW_START <= step <= self.HAZARD_WINDOW_END

    @property
    def hazard_injected(self) -> bool:
        return self._hazard_injected

    @property
    def hazard_target(self) -> Optional[str]:
        return self._hazard_target

    @property
    def hazard_step(self) -> Optional[int]:
        return self._hazard_step

    @property
    def hazard_source_rank_ahead(self) -> Optional[int]:
        return self._hazard_source_rank_ahead

    @property
    def hazard_injection_attempted(self) -> bool:
        return self._hazard_injection_attempted

    @property
    def hazard_injection_failed(self) -> bool:
        return self._hazard_injection_failed

    @property
    def hazard_injection_failed_reason(self) -> Optional[str]:
        return self._hazard_injection_failed_reason

    @property
    def braking_duration(self) -> Optional[float]:
        return self._braking_duration

    @property
    def target_strategy(self) -> str:
        return self._target_strategy
