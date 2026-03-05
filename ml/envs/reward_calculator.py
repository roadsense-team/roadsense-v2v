"""
Reward calculation utilities for ConvoyEnv.
"""

from typing import Dict, Tuple


class RewardCalculator:
    """
    Calculates reward based on safety, comfort, and appropriateness.

    Reward structure:
    - Collision (<5m): -100
    - Unsafe proximity (<10m): -5
    - Neutral zone (10-15m): 0
    - Safe following (15-35m): +1
    - Far (>35m): +0.5
    - Comfort penalty scales with deceleration magnitude
    - Unnecessary strong braking at far distance is penalized
    - Missed warning (<10m, closing, no brake) is penalized
    """

    COLLISION_DIST = 5.0
    UNSAFE_DIST = 10.0
    SAFE_DIST_MIN = 15.0
    SAFE_DIST_MAX = 35.0

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

    CLOSING_RATE_THRESHOLD = 0.5

    REWARD_EARLY_REACTION = 2.0
    EARLY_REACTION_DECEL_THRESHOLD = 0.5

    PENALTY_IGNORING_HAZARD = -5.0
    IGNORING_HAZARD_DIST_THRESHOLD = 30.0
    IGNORING_HAZARD_DECEL_THRESHOLD = 0.5

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
            distance < self.UNSAFE_DIST
            and closing_rate > self.CLOSING_RATE_THRESHOLD
            and decel_abs <= self.GENTLE_DECEL_THRESHOLD
        ):
            return self.PENALTY_MISSED_WARNING

        return 0.0

    def _early_reaction_bonus(
        self, distance: float, deceleration: float, any_braking_peer: bool
    ) -> float:
        """Bonus for proactive braking before distance becomes unsafe."""
        if not any_braking_peer:
            return 0.0
        if distance <= self.UNSAFE_DIST:
            return 0.0
        if abs(deceleration) < self.EARLY_REACTION_DECEL_THRESHOLD:
            return 0.0
        return self.REWARD_EARLY_REACTION

    def _ignoring_hazard_penalty(
        self, distance: float, deceleration: float, any_braking_peer: bool
    ) -> float:
        """Penalty for not braking when a peer is braking and close."""
        if not any_braking_peer:
            return 0.0
        if distance > self.IGNORING_HAZARD_DIST_THRESHOLD:
            return 0.0
        if abs(deceleration) >= self.IGNORING_HAZARD_DECEL_THRESHOLD:
            return 0.0
        return self.PENALTY_IGNORING_HAZARD

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
        comfort = self._comfort_penalty(deceleration)
        appropriateness = self._appropriateness_reward(
            distance, deceleration, closing_rate
        )
        early_reaction = self._early_reaction_bonus(
            distance, deceleration, any_braking_peer
        )
        ignoring_hazard = self._ignoring_hazard_penalty(
            distance, deceleration, any_braking_peer
        )

        # Zero comfort penalty when braking during a detected hazard.
        # Braking in response to a braking peer should never be punished.
        if any_braking_peer and abs(deceleration) >= self.IGNORING_HAZARD_DECEL_THRESHOLD:
            comfort = 0.0

        total = safety + comfort + appropriateness + early_reaction + ignoring_hazard

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
