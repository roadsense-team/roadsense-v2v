"""
Reward calculation utilities for ConvoyEnv.

Run 008 — Linear Ramp reward (Grok-style fix for poverty trap).
Replaces discrete zone boundaries with a continuous gradient from 5m to 20m,
plus distance-scaled comfort suppression so braking is cheap when it matters.
"""

from typing import Dict, Tuple


class RewardCalculator:
    """
    Calculates reward based on safety, comfort, and appropriateness.

    Reward structure (Run 008 — Linear Ramp):
    - Collision (<5m): -100 (terminal)
    - Ramp zone (5-20m): linear from -5 to +3 (continuous gradient)
    - Safe following (20-35m): +3 (plateau)
    - Far (>35m): -1 (anti-laziness)
    - Comfort penalty scales with decel, graduated by distance (min mult 0.1)
    - Missed warning (<10m, closing, no brake) penalized
    """

    COLLISION_DIST = 5.0
    RAMP_END = 20.0
    SAFE_DIST_MAX = 35.0

    RAMP_LOW = -5.0
    RAMP_HIGH = 3.0
    RAMP_SPAN = RAMP_HIGH - RAMP_LOW  # 8.0

    REWARD_COLLISION = -100.0
    REWARD_SAFE = 3.0
    REWARD_FAR = -1.0

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
    ) -> Tuple[float, Dict]:
        """Calculate total reward for current step."""
        safety = self._safety_reward(distance)
        base_comfort = self._comfort_penalty(deceleration)
        multiplier = self._comfort_multiplier(distance)
        comfort = base_comfort * multiplier
        appropriateness = self._appropriateness_reward(
            distance, deceleration, closing_rate
        )
        early_reaction = 0.0
        ignoring_hazard = 0.0

        total = safety + comfort + appropriateness

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
        }

        return total, info
