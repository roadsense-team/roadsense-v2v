"""
Hazard injection utilities for ConvoyEnv.

Run 023: adds state-triggered onset mode (``state_bucket``) alongside the
original ``fixed_step`` mode.  In ``state_bucket`` mode the injector
waits until a chosen front peer reaches a sampled rank/gap onset bucket
before firing, broadening the diversity of relative convoy states at
hazard onset.
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
    # Run 018: shorter braking produces accel signal ~-10 m/s² (clamped),
    # clearly distinguishable from normal CF adjustments (-0.3 to -0.5).
    # At 13.9 m/s: 0.5s → -27.8 (clamped to -10), 1.5s → -9.3 m/s².
    BRAKING_DURATION_MIN = 0.5  # seconds
    BRAKING_DURATION_MAX = 1.5  # seconds

    # Run 021: domain-randomize braking intensity to cover the real-data
    # distribution (mostly -2.5 to -5.0 m/s²).  Desired decel is drawn
    # uniformly and converted to a target speed based on the target's
    # current speed at injection time.  HAZARD_DECEL_MIN=3.0 stays well
    # above normal CF adjustments (-0.3 to -0.5), so the critic can still
    # distinguish hazard from calm episodes.
    HAZARD_DECEL_MIN = 3.0   # m/s² — moderate braking
    HAZARD_DECEL_MAX = 10.0  # m/s² — emergency (clamped by SUMO physics)

    # Run 021: resolved-hazard episodes.  Some fraction of hazards clear
    # after a short delay — the target resumes normal driving under CF.
    # Teaches the policy to react to transient braking (real-world pattern)
    # and to release the brake when the scene normalizes.
    HAZARD_RESOLVE_PROB = 0.4       # 40% of hazards resolve
    RESOLVE_DELAY_MIN = 20          # steps after slowdown ends (2.0s)
    RESOLVE_DELAY_MAX = 50          # steps after slowdown ends (5.0s)

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

    # --- Run 023: trigger modes -----------------------------------------------
    TRIGGER_MODE_FIXED_STEP = "fixed_step"
    TRIGGER_MODE_STATE_BUCKET = "state_bucket"

    # Gap buckets (longitudinal ego-to-source, metres).
    GAP_BUCKETS: Dict[str, Tuple[float, float]] = {
        "close":    (12.0, 22.0),
        "medium":   (22.0, 35.0),
        "far":      (35.0, 55.0),
        "very_far": (55.0, 85.0),
    }

    # Rank-conditioned eligible gap buckets.
    RANK_GAP_BUCKETS: Dict[int, List[str]] = {
        1: ["close", "medium", "far"],
        2: ["medium", "far", "very_far"],
    }
    # Rank 3+ gets "far" and "very_far".
    RANK_GAP_BUCKETS_DEFAULT = ["far", "very_far"]

    # Search window for state_bucket mode (training only).
    STATE_BUCKET_SEARCH_START = HAZARD_WINDOW_START   # step 150
    STATE_BUCKET_SEARCH_END = HAZARD_WINDOW_END       # step 300
    STATE_BUCKET_FALLBACK_STEP = HAZARD_WINDOW_END    # hard fallback

    def __init__(
        self,
        seed: Optional[int] = None,
        target_strategy: str = TARGET_STRATEGY_NEAREST,
        fixed_vehicle_id: Optional[str] = None,
        fixed_rank_ahead: Optional[int] = None,
        trigger_mode: str = TRIGGER_MODE_FIXED_STEP,
    ) -> None:
        self._validate_target_strategy(target_strategy)
        self._rng = random.Random(seed)
        self._default_target_strategy = target_strategy
        self._default_fixed_vehicle_id = fixed_vehicle_id
        self._default_fixed_rank_ahead = fixed_rank_ahead
        self._default_trigger_mode = trigger_mode

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

    # ------------------------------------------------------------------
    #  Gap-bucket helpers
    # ------------------------------------------------------------------

    @classmethod
    def eligible_gap_buckets(cls, rank: int) -> List[str]:
        """Return gap-bucket names eligible for the given rank ahead."""
        return cls.RANK_GAP_BUCKETS.get(rank, cls.RANK_GAP_BUCKETS_DEFAULT)

    def _sample_onset_bucket(self, rank: int) -> str:
        """Sample a gap bucket uniformly from rank-eligible set."""
        buckets = self.eligible_gap_buckets(rank)
        return self._rng.choice(buckets)

    @classmethod
    def gap_in_bucket(cls, gap_m: float, bucket_name: str) -> bool:
        """Return True if *gap_m* falls inside *bucket_name*."""
        lo, hi = cls.GAP_BUCKETS[bucket_name]
        return lo <= gap_m < hi

    # ------------------------------------------------------------------
    #  Longitudinal-gap computation
    # ------------------------------------------------------------------

    @staticmethod
    def _longitudinal_gap(
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        target_x: float,
        target_y: float,
    ) -> float:
        """
        Signed longitudinal gap from ego to target along ego's heading.

        Positive means target is ahead.
        """
        heading_rad = math.radians(float(ego_heading))
        forward_x = math.sin(heading_rad)
        forward_y = math.cos(heading_rad)
        dx = target_x - ego_x
        dy = target_y - ego_y
        return dx * forward_x + dy * forward_y

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

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

        # --- trigger mode ---
        trigger_mode = str(
            opts.get("trigger_mode", self._default_trigger_mode)
        )
        self._trigger_mode = trigger_mode

        hazard_step_opt = opts.get("hazard_step", None)
        if not self._episode_will_have_hazard:
            self._hazard_step = None
        elif hazard_step_opt is not None:
            # Explicit hazard_step forces fixed_step mode regardless.
            hazard_step = int(hazard_step_opt)
            if not self.is_in_hazard_window(hazard_step):
                raise ValueError(
                    f"hazard_step {hazard_step} outside injection window "
                    f"[{self.HAZARD_WINDOW_START}, {self.HAZARD_WINDOW_END}]"
                )
            self._hazard_step = hazard_step
            self._trigger_mode = self.TRIGGER_MODE_FIXED_STEP
        elif self._trigger_mode == self.TRIGGER_MODE_STATE_BUCKET:
            # State-bucket mode: hazard_step is determined at runtime.
            self._hazard_step = None
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
        self._hazard_target_speed: Optional[float] = None
        self._desired_decel: Optional[float] = None
        self._should_resolve: bool = False
        self._resolve_step: Optional[int] = None
        self._hazard_resolved: bool = False

        # Run 023: onset-bucket telemetry.
        self._onset_trigger_result: Optional[str] = None
        self._onset_gap_bucket: Optional[str] = None
        self._onset_gap_m: Optional[float] = None
        self._onset_closing_speed_mps: Optional[float] = None
        self._onset_peer_count: Optional[int] = None
        self._onset_trigger_step: Optional[int] = None
        self._onset_desired_rank_ahead: Optional[int] = None

        # Pre-sample state-bucket parameters for this episode.
        self._sb_target_vid: Optional[str] = None
        self._sb_target_rank: Optional[int] = None
        self._sb_desired_bucket: Optional[str] = None
        self._sb_search_active: bool = False

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

    # ------------------------------------------------------------------
    #  Injection (supports both fixed_step and state_bucket)
    # ------------------------------------------------------------------

    def maybe_inject(self, step: int, sumo: "SUMOConnection") -> bool:
        if self._hazard_injected:
            return False
        if not self._episode_will_have_hazard:
            return False

        if self._trigger_mode == self.TRIGGER_MODE_STATE_BUCKET:
            return self._maybe_inject_state_bucket(step, sumo)

        # fixed_step mode: inject at the exact scheduled step.
        if self._hazard_step is None or step != self._hazard_step:
            return False
        return self._do_inject(step, sumo, trigger_result="fixed_step")

    def _maybe_inject_state_bucket(
        self, step: int, sumo: "SUMOConnection"
    ) -> bool:
        """State-bucket trigger: search for onset condition each step."""
        if step < self.STATE_BUCKET_SEARCH_START:
            return False

        # ---- First time entering search window: pick target + bucket ----
        if not self._sb_search_active:
            front_peers = self._front_peers_with_rank(sumo)
            if not front_peers:
                # No front peers yet — wait until one appears or fallback.
                if step >= self.STATE_BUCKET_FALLBACK_STEP:
                    return self._do_inject(
                        step, sumo, trigger_result="fallback_step"
                    )
                return False

            # Pick a target using the configured strategy.
            target, rank = self._select_target(front_peers)
            if target is None or rank is None:
                if step >= self.STATE_BUCKET_FALLBACK_STEP:
                    return self._do_inject(
                        step, sumo, trigger_result="fallback_step"
                    )
                return False

            self._sb_target_vid = target
            self._sb_target_rank = rank
            self._sb_desired_bucket = self._sample_onset_bucket(rank)
            self._sb_search_active = True
            self._onset_desired_rank_ahead = rank

        # ---- Check onset condition each step ----
        ego_state = sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        if not sumo.is_vehicle_active(self._sb_target_vid):
            # Target left — fall back.
            if step >= self.STATE_BUCKET_FALLBACK_STEP:
                return self._do_inject(
                    step, sumo, trigger_result="fallback_step"
                )
            return False

        target_state = sumo.get_vehicle_state(self._sb_target_vid)
        gap = self._longitudinal_gap(
            ego_state.x, ego_state.y, ego_state.heading,
            target_state.x, target_state.y,
        )

        if gap > 0 and self.gap_in_bucket(gap, self._sb_desired_bucket):
            # Record onset telemetry before injection.
            self._onset_gap_bucket = self._sb_desired_bucket
            self._onset_gap_m = gap
            self._onset_closing_speed_mps = ego_state.speed - target_state.speed
            front_peers = self._front_peers_with_rank(sumo)
            self._onset_peer_count = len(front_peers)
            self._onset_trigger_step = step

            # Force the pre-selected target/rank into _select_target by
            # temporarily overriding strategy to fixed_rank_ahead.
            saved_strategy = self._target_strategy
            saved_rank = self._fixed_rank_ahead
            self._target_strategy = self.TARGET_STRATEGY_FIXED_RANK_AHEAD
            self._fixed_rank_ahead = self._sb_target_rank
            result = self._do_inject(step, sumo, trigger_result="bucket_match")
            self._target_strategy = saved_strategy
            self._fixed_rank_ahead = saved_rank
            return result

        # ---- Fallback at end of search window ----
        if step >= self.STATE_BUCKET_FALLBACK_STEP:
            # Record what we had at fallback.
            self._onset_gap_m = gap if gap > 0 else None
            self._onset_closing_speed_mps = (
                ego_state.speed - target_state.speed if gap > 0 else None
            )
            front_peers = self._front_peers_with_rank(sumo)
            self._onset_peer_count = len(front_peers)
            self._onset_trigger_step = step
            self._onset_desired_rank_ahead = self._sb_target_rank
            return self._do_inject(step, sumo, trigger_result="fallback_step")

        return False

    def _do_inject(
        self,
        step: int,
        sumo: "SUMOConnection",
        trigger_result: str = "fixed_step",
    ) -> bool:
        """Common injection path for both trigger modes."""
        self._hazard_injection_attempted = True
        self._onset_trigger_result = trigger_result

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

        # Run 021: randomize desired deceleration and compute target speed
        # from the target's current speed at injection time.
        desired_decel = self._rng.uniform(
            self.HAZARD_DECEL_MIN, self.HAZARD_DECEL_MAX
        )
        target_state = sumo.get_vehicle_state(target)
        target_speed = max(0.0, target_state.speed - desired_decel * braking_duration)
        sumo.slow_down(target, target_speed, braking_duration)

        self._braking_duration = braking_duration
        self._desired_decel = desired_decel
        self._hazard_target_speed = target_speed
        self._slowdown_end_step = step + int(
            round(braking_duration / self.STEP_LENGTH)
        )
        self._hazard_injected = True
        self._hazard_target = target
        self._hazard_source_rank_ahead = source_rank_ahead
        self._hazard_step = step  # record actual injection step

        # Onset telemetry for fixed_step mode (state_bucket fills these
        # in _maybe_inject_state_bucket before calling _do_inject).
        if self._onset_trigger_step is None:
            self._onset_trigger_step = step
        if self._onset_peer_count is None:
            self._onset_peer_count = len(front_peers_with_rank)

        # Run 021: decide whether this hazard resolves (target resumes CF).
        self._should_resolve = self._rng.random() < self.HAZARD_RESOLVE_PROB
        if self._should_resolve:
            resolve_delay = self._rng.randint(
                self.RESOLVE_DELAY_MIN, self.RESOLVE_DELAY_MAX
            )
            self._resolve_step = self._slowdown_end_step + resolve_delay
        else:
            self._resolve_step = None

        return True

    def maintain_hazard(self, step: int, sumo: "SUMOConnection") -> None:
        """Manage hazard target after initial slowdown.

        Must be called every step from the environment.

        Phase 1 — pin: Once the slowDown duration expires, SUMO would
        return the vehicle to CF control.  Pin the target at the chosen
        hazard target speed to keep the obstruction active.

        Phase 2 — resolve (Run 021): For resolved-hazard episodes, release
        the target back to CF after a short delay so it resumes normal
        driving.  This teaches the policy to react to transient braking.
        """
        if not self._hazard_injected or self._hazard_target is None:
            return

        # Phase 1: pin at target speed once slowdown completes
        if self._slowdown_end_step is not None and step >= self._slowdown_end_step:
            target_speed = self._hazard_target_speed if self._hazard_target_speed is not None else 0.0
            sumo.set_vehicle_speed(self._hazard_target, target_speed)
            self._slowdown_end_step = None  # Only pin once

        # Phase 2: release to CF for resolved hazards
        if (
            self._should_resolve
            and self._resolve_step is not None
            and step >= self._resolve_step
        ):
            sumo.release_vehicle_speed(self._hazard_target)
            self._should_resolve = False  # Only release once
            self._hazard_resolved = True

    def is_in_hazard_window(self, step: int) -> bool:
        return self.HAZARD_WINDOW_START <= step <= self.HAZARD_WINDOW_END

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

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
    def desired_decel(self) -> Optional[float]:
        return self._desired_decel

    @property
    def hazard_target_speed(self) -> Optional[float]:
        return self._hazard_target_speed

    @property
    def hazard_resolved(self) -> bool:
        return self._hazard_resolved

    @property
    def target_strategy(self) -> str:
        return self._target_strategy

    # --- Run 023: onset metadata properties ---

    @property
    def trigger_mode(self) -> str:
        return self._trigger_mode

    @property
    def trigger_result(self) -> Optional[str]:
        return self._onset_trigger_result

    @property
    def onset_gap_bucket(self) -> Optional[str]:
        return self._onset_gap_bucket

    @property
    def onset_gap_m(self) -> Optional[float]:
        return self._onset_gap_m

    @property
    def onset_closing_speed_mps(self) -> Optional[float]:
        return self._onset_closing_speed_mps

    @property
    def onset_peer_count(self) -> Optional[int]:
        return self._onset_peer_count

    @property
    def onset_trigger_step(self) -> Optional[int]:
        return self._onset_trigger_step

    @property
    def onset_desired_rank_ahead(self) -> Optional[int]:
        return self._onset_desired_rank_ahead
