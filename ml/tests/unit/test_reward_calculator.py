"""
Unit tests for RewardCalculator (Phase 5).
"""

from envs.reward_calculator import RewardCalculator


def test_reward_collision_distance_under_5m_returns_neg_100():
    """d < 5m -> reward = -100."""
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=4.0)
    assert reward == -100.0


def test_reward_unsafe_distance_under_15m_returns_neg_5():
    """5m <= d < 15m -> reward includes -5."""
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=10.0)
    assert reward == -5.0


def test_reward_safe_distance_20_to_40m_returns_pos_1():
    """20m <= d <= 40m -> reward includes +1."""
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=30.0)
    assert reward == 1.0


def test_reward_far_distance_over_40m_returns_pos_0_5():
    """d > 40m -> reward includes +0.5."""
    calc = RewardCalculator()
    reward = calc._safety_reward(distance=50.0)
    assert reward == 0.5


def test_reward_harsh_brake_over_4_5_returns_neg_10():
    """|decel| > 4.5 m/s² -> reward includes -10."""
    calc = RewardCalculator()
    penalty = calc._comfort_penalty(deceleration=5.0)
    assert penalty == -10.0


def test_reward_uncomfortable_brake_over_3_returns_neg_2():
    """3.0 < |decel| <= 4.5 -> reward includes -2."""
    calc = RewardCalculator()
    penalty = calc._comfort_penalty(deceleration=3.5)
    assert penalty == -2.0


def test_reward_gentle_brake_under_3_no_penalty():
    """|decel| <= 3.0 -> no comfort penalty."""
    calc = RewardCalculator()
    penalty = calc._comfort_penalty(deceleration=2.0)
    assert penalty == 0.0


def test_reward_unnecessary_alert_far_and_braking_returns_neg_2():
    """d > 40m AND action > 1 -> reward includes -2."""
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=50.0, action=2)
    assert penalty == -2.0


def test_reward_missed_warning_close_and_maintain_returns_neg_3():
    """d < 15m AND action == 0 -> reward includes -3."""
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=10.0, action=0)
    assert penalty == -3.0


def test_reward_appropriate_action_no_penalty():
    """Correct action for situation -> no appropriateness penalty."""
    calc = RewardCalculator()
    penalty = calc._appropriateness_reward(distance=30.0, action=0)
    assert penalty == 0.0


def test_reward_calculate_combines_all_components():
    """Total reward = safety + comfort + appropriateness."""
    calc = RewardCalculator()
    total, info = calc.calculate(distance=30.0, action=0, deceleration=1.0)
    assert total == 1.0
    assert info["reward_safety"] == 1.0
    assert info["reward_comfort"] == 0.0
    assert info["reward_appropriateness"] == 0.0


def test_reward_calculate_returns_info_dict():
    """Returns (reward, info) where info has component breakdown."""
    calc = RewardCalculator()
    total, info = calc.calculate(distance=25.0, action=1, deceleration=0.5)

    assert "reward_safety" in info
    assert "reward_comfort" in info
    assert "reward_appropriateness" in info
    assert "reward_total" in info
    assert "distance" in info
    assert "action" in info
    assert "deceleration" in info
