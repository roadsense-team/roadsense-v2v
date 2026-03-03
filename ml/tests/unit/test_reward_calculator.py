"""
Unit tests for RewardCalculator.

Distance bands: collision(<5m), unsafe(5-10m), neutral(10-15m), safe(15-35m), far(>35m)
"""

from envs.reward_calculator import RewardCalculator


def test_reward_collision_distance_under_5m_returns_neg_100():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=4.0)
    assert reward == -100.0


def test_reward_unsafe_distance_returns_neg_5():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=7.0)
    assert reward == -5.0


def test_reward_neutral_zone_returns_zero():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=12.0)
    assert reward == 0.0


def test_reward_safe_distance_15_to_35m_returns_pos_1():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=25.0)
    assert reward == 1.0


def test_reward_far_distance_returns_pos_half():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=40.0)
    assert reward == 0.5


def test_reward_comfort_penalty_scales_with_decel_magnitude():
    calc = RewardCalculator()

    low = calc._comfort_penalty(deceleration=1.5)
    mid = calc._comfort_penalty(deceleration=3.0)
    high = calc._comfort_penalty(deceleration=4.5)

    assert low <= 0.0
    assert low > mid
    assert mid > high


def test_reward_no_penalty_for_gentle_decel():
    calc = RewardCalculator()
    penalty = calc._comfort_penalty(deceleration=0.2)
    assert penalty == 0.0


def test_reward_comfort_at_gentle_threshold_is_zero():
    calc = RewardCalculator()
    penalty = calc._comfort_penalty(deceleration=0.5)
    assert penalty == 0.0


def test_reward_comfort_just_above_gentle_is_negative():
    calc = RewardCalculator()
    penalty = calc._comfort_penalty(deceleration=0.6)
    assert penalty < 0.0


def test_reward_high_penalty_for_max_decel_when_unnecessary():
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=40.0, deceleration=5.0, closing_rate=0.0)
    assert penalty <= -2.0


def test_reward_missed_warning_close_and_no_brake_when_closing():
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=7.0, deceleration=0.0, closing_rate=2.0)
    assert penalty == -3.0


def test_reward_no_missed_warning_when_gap_opening():
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=7.0, deceleration=0.0, closing_rate=-1.0)
    assert penalty == 0.0


def test_reward_no_missed_warning_when_gap_stable():
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=7.0, deceleration=0.0, closing_rate=0.2)
    assert penalty == 0.0


def test_reward_missed_warning_only_above_closing_threshold():
    calc = RewardCalculator()
    below = calc._appropriateness_reward(distance=7.0, deceleration=0.0, closing_rate=0.4)
    above = calc._appropriateness_reward(distance=7.0, deceleration=0.0, closing_rate=0.6)
    assert below == 0.0
    assert above == -3.0


def test_reward_no_missed_warning_in_neutral_zone():
    """Distance in neutral zone (10-15m) should not trigger missed warning."""
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=12.0, deceleration=0.0, closing_rate=2.0)
    assert penalty == 0.0


def test_reward_calculate_combines_all_components():
    calc = RewardCalculator()
    total, info = calc.calculate(distance=25.0, action_value=0.0, deceleration=0.2)

    assert total == 1.0
    assert info["reward_safety"] == 1.0
    assert info["reward_comfort"] == 0.0
    assert info["reward_appropriateness"] == 0.0


def test_reward_calculate_returns_info_dict():
    calc = RewardCalculator()
    _, info = calc.calculate(distance=20.0, action_value=0.4, deceleration=2.0)

    assert "reward_safety" in info
    assert "reward_comfort" in info
    assert "reward_appropriateness" in info
    assert "reward_total" in info
    assert "distance" in info
    assert "action_value" in info
    assert "deceleration" in info
    assert "closing_rate" in info


def test_reward_calculate_passes_closing_rate():
    calc = RewardCalculator()
    _, info = calc.calculate(
        distance=7.0, action_value=0.0, deceleration=0.0, closing_rate=3.0
    )
    assert info["closing_rate"] == 3.0
    assert info["reward_appropriateness"] == -3.0
