"""
Unit tests for RewardCalculator.

Run 009 — Linear ramp reward structure:
  collision(<5m), ramp(5-20m linear -5→+4), safe(20-35m +4), far(>35m -2)
  Comfort graduated by distance (min multiplier 0.1).
"""

import pytest

from envs.reward_calculator import RewardCalculator


# ── Safety reward (linear ramp) ──────────────────────────────────────────


def test_safety_collision_returns_neg_100():
    calc = RewardCalculator()
    assert calc._safety_reward(4.0) == -100.0


def test_safety_at_ramp_start_5m_returns_neg_5():
    calc = RewardCalculator()
    assert calc._safety_reward(5.0) == pytest.approx(-5.0)


def test_safety_at_spawn_8m_returns_ramp_value():
    calc = RewardCalculator()
    # -5 + 9 * (3/15) = -5 + 1.8 = -3.2
    assert calc._safety_reward(8.0) == pytest.approx(-3.2, abs=0.01)


def test_safety_at_10m_returns_ramp_value():
    calc = RewardCalculator()
    # -5 + 9 * (5/15) = -5 + 3.0 = -2.0
    assert calc._safety_reward(10.0) == pytest.approx(-2.0, abs=0.01)


def test_safety_at_12_5m_returns_ramp_midpoint():
    calc = RewardCalculator()
    # -5 + 9 * (7.5/15) = -5 + 4.5 = -0.5
    assert calc._safety_reward(12.5) == pytest.approx(-0.5, abs=0.01)


def test_safety_at_15m_returns_positive_ramp():
    calc = RewardCalculator()
    # -5 + 9 * (10/15) = -5 + 6.0 = +1.0
    assert calc._safety_reward(15.0) == pytest.approx(1.0, abs=0.01)


def test_safety_at_ramp_end_20m_returns_plus_4():
    calc = RewardCalculator()
    assert calc._safety_reward(20.0) == pytest.approx(4.0)


def test_safety_in_safe_plateau_returns_plus_4():
    calc = RewardCalculator()
    assert calc._safety_reward(25.0) == pytest.approx(4.0)
    assert calc._safety_reward(35.0) == pytest.approx(4.0)


def test_safety_far_returns_neg_2():
    calc = RewardCalculator()
    assert calc._safety_reward(40.0) == pytest.approx(-2.0)


def test_safety_ramp_is_monotonically_increasing():
    calc = RewardCalculator()
    distances = [5.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.5, 19.9]
    rewards = [calc._safety_reward(d) for d in distances]
    for i in range(1, len(rewards)):
        assert rewards[i] > rewards[i - 1], (
            f"Ramp not increasing at d={distances[i]}: "
            f"{rewards[i]} <= {rewards[i-1]}"
        )


# ── Comfort penalty (base, before distance scaling) ─────────────────────


def test_comfort_no_penalty_for_gentle_decel():
    calc = RewardCalculator()
    assert calc._comfort_penalty(0.2) == 0.0
    assert calc._comfort_penalty(0.5) == 0.0


def test_comfort_scales_with_decel_magnitude():
    calc = RewardCalculator()
    low = calc._comfort_penalty(1.5)
    mid = calc._comfort_penalty(3.0)
    high = calc._comfort_penalty(4.5)
    assert low < 0.0
    assert low > mid
    assert mid > high


def test_comfort_harsh_brake_penalty_is_neg_5():
    calc = RewardCalculator()
    assert calc._comfort_penalty(5.0) == pytest.approx(-5.0)


def test_comfort_just_above_gentle_is_negative():
    calc = RewardCalculator()
    assert calc._comfort_penalty(0.6) < 0.0


# ── Comfort multiplier (distance-scaled suppression) ────────────────────


def test_multiplier_at_5m_is_min_cap():
    calc = RewardCalculator()
    assert calc._comfort_multiplier(5.0) == pytest.approx(0.1)


def test_multiplier_at_8m():
    calc = RewardCalculator()
    # raw = 3/15 = 0.2 > 0.1 min
    assert calc._comfort_multiplier(8.0) == pytest.approx(0.2, abs=0.01)


def test_multiplier_at_12_5m():
    calc = RewardCalculator()
    assert calc._comfort_multiplier(12.5) == pytest.approx(0.5, abs=0.01)


def test_multiplier_at_20m_is_full():
    calc = RewardCalculator()
    assert calc._comfort_multiplier(20.0) == pytest.approx(1.0)


def test_multiplier_in_safe_zone_is_full():
    calc = RewardCalculator()
    assert calc._comfort_multiplier(25.0) == pytest.approx(1.0)


def test_multiplier_below_5m_uses_min_cap():
    calc = RewardCalculator()
    assert calc._comfort_multiplier(4.5) == pytest.approx(0.1)


# ── Appropriateness penalty ─────────────────────────────────────────────


def test_missed_warning_close_and_no_brake_while_closing():
    calc = RewardCalculator()
    assert calc._appropriateness_reward(7.0, 0.0, 2.0) == -3.0


def test_no_missed_warning_when_gap_opening():
    calc = RewardCalculator()
    assert calc._appropriateness_reward(7.0, 0.0, -1.0) == 0.0


def test_no_missed_warning_when_gap_stable():
    calc = RewardCalculator()
    assert calc._appropriateness_reward(7.0, 0.0, 0.2) == 0.0


def test_missed_warning_threshold():
    calc = RewardCalculator()
    assert calc._appropriateness_reward(7.0, 0.0, 0.4) == 0.0
    assert calc._appropriateness_reward(7.0, 0.0, 0.6) == -3.0


def test_no_missed_warning_above_10m():
    calc = RewardCalculator()
    assert calc._appropriateness_reward(12.0, 0.0, 2.0) == 0.0


def test_unnecessary_braking_far():
    calc = RewardCalculator()
    assert calc._appropriateness_reward(40.0, 5.0, 0.0) <= -2.0


# ── Integrated calculate() ──────────────────────────────────────────────


def test_calculate_safe_no_brake():
    calc = RewardCalculator()
    total, info = calc.calculate(distance=25.0, action_value=0.0, deceleration=0.2)
    assert total == pytest.approx(4.0)
    assert info["reward_safety"] == pytest.approx(4.0)
    assert info["reward_comfort"] == pytest.approx(0.0)
    assert info["reward_appropriateness"] == pytest.approx(0.0)


def test_calculate_returns_all_info_fields():
    calc = RewardCalculator()
    _, info = calc.calculate(distance=20.0, action_value=0.4, deceleration=2.0)
    required = [
        "reward_safety", "reward_comfort", "reward_appropriateness",
        "reward_early_reaction", "reward_ignoring_hazard", "reward_total",
        "distance", "action_value", "deceleration", "closing_rate",
        "any_braking_peer",
    ]
    for key in required:
        assert key in info


def test_calculate_legacy_fields_are_zero():
    calc = RewardCalculator()
    _, info = calc.calculate(
        distance=25.0, action_value=0.1, deceleration=1.0,
        closing_rate=0.0, any_braking_peer=True,
    )
    assert info["reward_early_reaction"] == 0.0
    assert info["reward_ignoring_hazard"] == 0.0


def test_calculate_total_equals_sum():
    calc = RewardCalculator()
    total, info = calc.calculate(
        distance=25.0, action_value=0.1, deceleration=1.0,
        closing_rate=0.0, any_braking_peer=True,
    )
    expected = (
        info["reward_safety"]
        + info["reward_comfort"]
        + info["reward_appropriateness"]
    )
    assert total == pytest.approx(expected)


def test_calculate_comfort_graduated_in_ramp_zone():
    """At 8m, harsh braking costs much less than at 25m (graduated suppression)."""
    calc = RewardCalculator()
    _, info_close = calc.calculate(distance=8.0, action_value=0.8, deceleration=6.0)
    _, info_safe = calc.calculate(distance=25.0, action_value=0.8, deceleration=6.0)
    # At 8m, multiplier ~0.2; at 25m, multiplier 1.0
    assert abs(info_close["reward_comfort"]) < abs(info_safe["reward_comfort"])


def test_calculate_comfort_not_zero_in_ramp_zone():
    """Even at low distance, comfort has a small cost (min multiplier 0.1)."""
    calc = RewardCalculator()
    _, info = calc.calculate(distance=5.5, action_value=0.8, deceleration=6.0)
    assert info["reward_comfort"] < 0.0


def test_calculate_comfort_full_in_safe_zone():
    calc = RewardCalculator()
    _, info = calc.calculate(distance=25.0, action_value=0.5, deceleration=4.0)
    assert info["reward_comfort"] < 0.0
    assert info["reward_safety"] == pytest.approx(4.0)


def test_calculate_braking_better_than_not_braking_when_close_and_closing():
    """In ramp zone + closing, braking must be better than not braking."""
    calc = RewardCalculator()
    total_no_brake, _ = calc.calculate(
        distance=7.0, action_value=0.0, deceleration=0.0, closing_rate=2.0
    )
    total_brake, _ = calc.calculate(
        distance=7.0, action_value=0.8, deceleration=6.0, closing_rate=2.0
    )
    assert total_brake >= total_no_brake


def test_calculate_any_braking_peer_does_not_change_reward():
    calc = RewardCalculator()
    total_f, _ = calc.calculate(
        distance=22.0, action_value=0.3, deceleration=2.2,
        closing_rate=0.4, any_braking_peer=False,
    )
    total_t, _ = calc.calculate(
        distance=22.0, action_value=0.3, deceleration=2.2,
        closing_rate=0.4, any_braking_peer=True,
    )
    assert total_t == pytest.approx(total_f)


def test_calculate_passes_closing_rate():
    calc = RewardCalculator()
    _, info = calc.calculate(
        distance=7.0, action_value=0.0, deceleration=0.0, closing_rate=3.0
    )
    assert info["closing_rate"] == 3.0
    assert info["reward_appropriateness"] == -3.0


# ── Economic sanity checks (the whole point of the ramp) ────────────────


def test_ramp_provides_gradient_from_spawn():
    """Agent at 8m gets better reward per meter gained — no dead zones."""
    calc = RewardCalculator()
    r8 = calc._safety_reward(8.0)
    r10 = calc._safety_reward(10.0)
    r12 = calc._safety_reward(12.0)
    r15 = calc._safety_reward(15.0)
    assert r10 > r8, "Moving from 8m to 10m must improve reward"
    assert r12 > r10, "Moving from 10m to 12m must improve reward"
    assert r15 > r12, "Moving from 12m to 15m must improve reward"


def test_far_zone_is_worse_than_safe_zone():
    """Anti-laziness: falling behind to >35m is worse than staying at 20-35m."""
    calc = RewardCalculator()
    assert calc._safety_reward(40.0) < calc._safety_reward(25.0)


def test_random_policy_reward_at_spawn_is_not_dominated_by_comfort():
    """At spawn (8m), moderate braking (action~0.5, decel~4.0) should have
    manageable comfort cost relative to safety signal."""
    calc = RewardCalculator()
    _, info = calc.calculate(distance=8.0, action_value=0.5, deceleration=4.0)
    assert abs(info["reward_comfort"]) < abs(info["reward_safety"]), (
        "Comfort must not drown safety signal at spawn"
    )


def test_safe_plateau_beats_passive_do_nothing_at_ramp():
    """Active braking to safe zone (+4/step) clearly beats passive at ramp mid."""
    calc = RewardCalculator()
    safe_reward = calc._safety_reward(25.0)   # +4
    ramp_mid = calc._safety_reward(12.5)      # -0.5
    assert safe_reward - ramp_mid > 4.0, (
        "Safe zone must dominate mid-ramp by enough to justify braking cost"
    )
