"""
Reward calculation utilities for ConvoyEnv.

Run 020 — Linear Ramp + Decay-Scaled V2V Response Terms + Speed-Gated Penalty.
Builds on Run 009 ramp structure.  Run 020 changes vs Run 017:
  - braking_received is now a float in [0, 1] (exponential decay, half-life
    ~1.35s at 10 Hz).  Reward terms scale by this value so the shaping fades
    smoothly — no obs/reward mismatch.
  - progress feature removed (deployment-incompatible).
  - PENALTY_IGNORING_HAZARD suppressed when ego is stopped (speed <= 0.5 m/s).
  - REWARD_EARLY_REACTION: bonus when ego brakes while braking signal is active
    AND distance is still > EARLY_REACTION_DIST.
"""

from typing import Dict, Tuple


class RewardCalculator:
    """
    Calculates reward based on safety, comfort, and appropriateness.

    Reward structure (Run 009 — Linear Ramp + Active V2V):
    - Collision (<5m): -100 (terminal)
    - Ramp zone (5-20m): linear from -5 to +4 (steeper gradient)
    - Safe following (20-35m): +4 (plateau)
    - Far (>35m): -2 (stronger anti-laziness)
    - Comfort penalty scales with decel, graduated by distance (min mult 0.1)
    - Missed warning (<10m, closing, no brake) penalized
    """

    COLLISION_DIST = 5.0
    RAMP_END = 20.0
    SAFE_DIST_MAX = 35.0

    RAMP_LOW = -5.0
    RAMP_HIGH = 4.0
    RAMP_SPAN = RAMP_HIGH - RAMP_LOW  # 9.0

    REWARD_COLLISION = -100.0
    REWARD_SAFE = 4.0
    REWARD_FAR = -2.0

    COMFORT_MIN_MULTIPLIER = 0.1

    GENTLE_DECEL_THRESHOLD = 0.5
    UNCOMFORTABLE_BRAKE_THRESHOLD = 3.0
    HARSH_BRAKE_THRESHOLD = 4.5
    PENALTY_UNCOMFORTABLE = -2.0
    PENALTY_HARSH_BRAKE = -5.0

    PENALTY_UNNECESSARY_ALERT = -2.0
    PENALTY_UNNECESSARY_MAX_BRAKE = -4.0
    PENALTY_MISSED_WARNING = -3.0

    MISSED_WARNING_DIST = 10.0
    CLOSING_RATE_THRESHOLD = 0.5

    PENALTY_IGNORING_HAZARD = -5.0
    REWARD_EARLY_REACTION = 2.0
    EARLY_REACTION_DIST = 15.0
    STOPPED_SPEED_THRESHOLD = 0.5

    def _safety_reward(self, distance: float) -> float:
        """Calculate safety reward with continuous linear ramp."""
        if distance < self.COLLISION_DIST:
            return self.REWARD_COLLISION
        if distance < self.RAMP_END:
            # Linear ramp: -5 at 5m -> +3 at 20m
            t = (distance - self.COLLISION_DIST) / (
                self.RAMP_END - self.COLLISION_DIST
            )
            return self.RAMP_LOW + self.RAMP_SPAN * t
        if distance <= self.SAFE_DIST_MAX:
            return self.REWARD_SAFE
        return self.REWARD_FAR

    def _comfort_penalty(self, deceleration: float) -> float:
        """Calculate base comfort penalty (before distance scaling)."""
        decel_abs = abs(deceleration)

        if decel_abs <= self.GENTLE_DECEL_THRESHOLD:
            return 0.0

        if decel_abs <= self.UNCOMFORTABLE_BRAKE_THRESHOLD:
            span = self.UNCOMFORTABLE_BRAKE_THRESHOLD - self.GENTLE_DECEL_THRESHOLD
            ratio = (decel_abs - self.GENTLE_DECEL_THRESHOLD) / span
            return ratio * self.PENALTY_UNCOMFORTABLE

        if decel_abs <= self.HARSH_BRAKE_THRESHOLD:
            span = self.HARSH_BRAKE_THRESHOLD - self.UNCOMFORTABLE_BRAKE_THRESHOLD
            ratio = (decel_abs - self.UNCOMFORTABLE_BRAKE_THRESHOLD) / span
            return self.PENALTY_UNCOMFORTABLE + ratio * (
                self.PENALTY_HARSH_BRAKE - self.PENALTY_UNCOMFORTABLE
            )

        return self.PENALTY_HARSH_BRAKE

    def _comfort_multiplier(self, distance: float) -> float:
        """Distance-scaled comfort suppression. Braking is cheaper when close."""
        if distance >= self.RAMP_END:
            return 1.0
        raw = max(
            0.0,
            (distance - self.COLLISION_DIST)
            / (self.RAMP_END - self.COLLISION_DIST),
        )
        return max(self.COMFORT_MIN_MULTIPLIER, raw)

    def _appropriateness_reward(
        self, distance: float, deceleration: float, closing_rate: float
    ) -> float:
        """Calculate appropriateness penalty for continuous control."""
        decel_abs = abs(deceleration)

        if distance > self.SAFE_DIST_MAX:
            if decel_abs >= self.HARSH_BRAKE_THRESHOLD:
                return self.PENALTY_UNNECESSARY_MAX_BRAKE
            if decel_abs > self.GENTLE_DECEL_THRESHOLD:
                return self.PENALTY_UNNECESSARY_ALERT

        if (
            distance < self.MISSED_WARNING_DIST
            and closing_rate > self.CLOSING_RATE_THRESHOLD
            and decel_abs <= self.GENTLE_DECEL_THRESHOLD
        ):
            return self.PENALTY_MISSED_WARNING

        return 0.0

    def calculate(
        self,
        distance: float,
        action_value: float,
        deceleration: float,
        closing_rate: float = 0.0,
        any_braking_peer: bool = False,
        braking_received: float = 0.0,
        ego_speed: float = 10.0,
    ) -> Tuple[float, Dict]:
        """Calculate total reward for current step.

        braking_received: exponential-decay signal in [0, 1] — same value
            that drives ego[4].  Reward terms are scaled by this value so
            the shaping fades smoothly with the observation signal.
        ego_speed: current ego speed in m/s.  Used to suppress the ignoring
            penalty when ego is already stopped.
        """
        safety = self._safety_reward(distance)
        base_comfort = self._comfort_penalty(deceleration)
        multiplier = self._comfort_multiplier(distance)
        comfort = base_comfort * multiplier
        appropriateness = self._appropriateness_reward(
            distance, deceleration, closing_rate
        )

        early_reaction = 0.0
        ignoring_hazard = 0.0
        ignoring_penalty_suppressed_for_stop = False

        if braking_received > 0.01:
            decel_abs = abs(deceleration)
            is_braking = decel_abs > self.GENTLE_DECEL_THRESHOLD
            is_stopped = ego_speed <= self.STOPPED_SPEED_THRESHOLD

            if not is_braking:
                if is_stopped:
                    ignoring_penalty_suppressed_for_stop = True
                else:
                    ignoring_hazard = self.PENALTY_IGNORING_HAZARD * braking_received
            elif distance > self.EARLY_REACTION_DIST:
                early_reaction = self.REWARD_EARLY_REACTION * braking_received

        total = (
            safety + comfort + appropriateness
            + early_reaction + ignoring_hazard
        )

        info = {
            "reward_safety": safety,
            "reward_comfort": comfort,
            "reward_appropriateness": appropriateness,
            "reward_early_reaction": early_reaction,
            "reward_ignoring_hazard": ignoring_hazard,
            "reward_total": total,
            "distance": distance,
            "action_value": action_value,
            "deceleration": deceleration,
            "closing_rate": closing_rate,
            "any_braking_peer": any_braking_peer,
            "braking_received": braking_received,
            "ego_speed": ego_speed,
            "ignoring_penalty_suppressed_for_stop": ignoring_penalty_suppressed_for_stop,
        }

        return total, info
