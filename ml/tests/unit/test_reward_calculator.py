"""
Unit tests for RewardCalculator continuous deceleration semantics (Phase B).
"""

from envs.reward_calculator import RewardCalculator


def test_reward_collision_distance_under_5m_returns_neg_100():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=4.0)
    assert reward == -100.0


def test_reward_safe_distance_20_to_40m_returns_pos_1():
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=30.0)
    assert reward == 1.0


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


def test_reward_high_penalty_for_max_decel_when_unnecessary():
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=50.0, deceleration=5.0)
    assert penalty <= -2.0


def test_reward_missed_warning_close_and_no_brake_returns_neg_3():
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=10.0, deceleration=0.0)
    assert penalty == -3.0


def test_reward_calculate_combines_all_components():
    calc = RewardCalculator()
    total, info = calc.calculate(distance=30.0, action_value=0.0, deceleration=0.2)

    assert total == 1.0
    assert info["reward_safety"] == 1.0
    assert info["reward_comfort"] == 0.0
    assert info["reward_appropriateness"] == 0.0


def test_reward_calculate_returns_info_dict():
    calc = RewardCalculator()
    _, info = calc.calculate(distance=25.0, action_value=0.4, deceleration=2.0)

    assert "reward_safety" in info
    assert "reward_comfort" in info
    assert "reward_appropriateness" in info
    assert "reward_total" in info
    assert "distance" in info
    assert "action_value" in info
    assert "deceleration" in info
