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
    - Harsh braking (>4.5 m/s²): -10
    - Uncomfortable braking (>3.0 m/s²): -2
    - Unnecessary alert (>40m, action>1): -2
    - Missed warning (<15m, action==0): -3
    """

    COLLISION_DIST = 5.0
    UNSAFE_DIST = 15.0
    SAFE_DIST_MIN = 20.0
    SAFE_DIST_MAX = 40.0

    REWARD_COLLISION = -100.0
    REWARD_UNSAFE = -5.0
    REWARD_SAFE = 1.0
    REWARD_FAR = 0.5

    HARSH_BRAKE_THRESHOLD = 4.5
    UNCOMFORTABLE_BRAKE_THRESHOLD = 3.0
    PENALTY_HARSH_BRAKE = -10.0
    PENALTY_UNCOMFORTABLE = -2.0

    PENALTY_UNNECESSARY_ALERT = -2.0
    PENALTY_MISSED_WARNING = -3.0

    def _safety_reward(self, distance: float) -> float:
        """
        Calculate safety component of reward.

        Args:
            distance: Distance to nearest lead vehicle in meters

        Returns:
            Safety reward component
        """
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
        """
        Calculate comfort penalty based on braking harshness.

        Args:
            deceleration: Actual deceleration applied (positive = braking)

        Returns:
            Comfort penalty (negative or zero)
        """
        decel_abs = abs(deceleration)
        if decel_abs > self.HARSH_BRAKE_THRESHOLD:
            return self.PENALTY_HARSH_BRAKE
        if decel_abs > self.UNCOMFORTABLE_BRAKE_THRESHOLD:
            return self.PENALTY_UNCOMFORTABLE
        return 0.0

    def _appropriateness_reward(self, distance: float, action: int) -> float:
        """
        Calculate appropriateness of action given situation.

        Args:
            distance: Distance to lead vehicle
            action: Action taken (0=Maintain, 1=Caution, 2=Brake, 3=Emergency)

        Returns:
            Appropriateness penalty (negative or zero)
        """
        if distance > self.SAFE_DIST_MAX and action > 1:
            return self.PENALTY_UNNECESSARY_ALERT
        if distance < self.UNSAFE_DIST and action == 0:
            return self.PENALTY_MISSED_WARNING
        return 0.0

    def calculate(
        self, distance: float, action: int, deceleration: float
    ) -> Tuple[float, Dict]:
        """
        Calculate total reward for current step.

        Args:
            distance: Distance to nearest lead vehicle (meters)
            action: Action taken (0-3)
            deceleration: Actual deceleration applied (m/s)

        Returns:
            Tuple of (total_reward, info_dict)
            info_dict contains component breakdown for debugging
        """
        safety = self._safety_reward(distance)
        comfort = self._comfort_penalty(deceleration)
        appropriateness = self._appropriateness_reward(distance, action)

        total = safety + comfort + appropriateness

        info = {
            "reward_safety": safety,
            "reward_comfort": comfort,
            "reward_appropriateness": appropriateness,
            "reward_total": total,
            "distance": distance,
            "action": action,
            "deceleration": deceleration,
        }

        return total, info
