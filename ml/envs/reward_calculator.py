"""
Reward calculation utilities for ConvoyEnv.
"""

from typing import Dict, Tuple


class RewardCalculator:
    """
    Calculates reward based on safety, comfort, and appropriateness.

    Reward structure from ARCHITECTURE_V2_MINIMAL_RL.md:
    - Collision: -100
    - Unsafe proximity (<15m): -5
    - Safe following (20-40m): +1
    - Far (>40m): +0.5
    - Comfort penalty scales with deceleration magnitude
    - Unnecessary strong braking at far distance is penalized
    - Missed warning (<15m with near-zero braking) is penalized
    """

    COLLISION_DIST = 5.0
    UNSAFE_DIST = 15.0
    SAFE_DIST_MIN = 20.0
    SAFE_DIST_MAX = 40.0

    REWARD_COLLISION = -100.0
    REWARD_UNSAFE = -5.0
    REWARD_SAFE = 1.0
    REWARD_FAR = 0.5

    GENTLE_DECEL_THRESHOLD = 0.5
    UNCOMFORTABLE_BRAKE_THRESHOLD = 3.0
    HARSH_BRAKE_THRESHOLD = 4.5
    PENALTY_UNCOMFORTABLE = -2.0
    PENALTY_HARSH_BRAKE = -10.0

    PENALTY_UNNECESSARY_ALERT = -2.0
    PENALTY_UNNECESSARY_MAX_BRAKE = -4.0
    PENALTY_MISSED_WARNING = -3.0

    def _safety_reward(self, distance: float) -> float:
        """Calculate safety component of reward."""
        if distance < self.COLLISION_DIST:
            return self.REWARD_COLLISION
        if distance < self.UNSAFE_DIST:
            return self.REWARD_UNSAFE
        if self.SAFE_DIST_MIN <= distance <= self.SAFE_DIST_MAX:
            return self.REWARD_SAFE
        if distance > self.SAFE_DIST_MAX:
            return self.REWARD_FAR
        return 0.0

    def _comfort_penalty(self, deceleration: float) -> float:
        """Calculate comfort penalty with continuous scaling."""
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

    def _appropriateness_reward(self, distance: float, deceleration: float) -> float:
        """Calculate appropriateness penalty for continuous control."""
        decel_abs = abs(deceleration)

        if distance > self.SAFE_DIST_MAX:
            if decel_abs >= self.HARSH_BRAKE_THRESHOLD:
                return self.PENALTY_UNNECESSARY_MAX_BRAKE
            if decel_abs > self.GENTLE_DECEL_THRESHOLD:
                return self.PENALTY_UNNECESSARY_ALERT

        if distance < self.UNSAFE_DIST and decel_abs <= self.GENTLE_DECEL_THRESHOLD:
            return self.PENALTY_MISSED_WARNING

        return 0.0

    def calculate(
        self, distance: float, action_value: float, deceleration: float
    ) -> Tuple[float, Dict]:
        """Calculate total reward for current step."""
        safety = self._safety_reward(distance)
        comfort = self._comfort_penalty(deceleration)
        appropriateness = self._appropriateness_reward(distance, deceleration)

        total = safety + comfort + appropriateness

        info = {
            "reward_safety": safety,
            "reward_comfort": comfort,
            "reward_appropriateness": appropriateness,
            "reward_total": total,
            "distance": distance,
            "action_value": action_value,
            "deceleration": deceleration,
        }

        return total, info
