"""
Unit tests for HazardInjector.
"""
from unittest.mock import Mock

import pytest

from envs.hazard_injector import HazardInjector
from envs.sumo_connection import VehicleState


def _make_vehicle_state(vehicle_id, x, y, speed=10.0, heading=0.0):
    return VehicleState(
        vehicle_id=vehicle_id, x=x, y=y,
        speed=speed, acceleration=0.0, heading=heading, lane_position=0.0,
    )


def _make_sumo(vehicle_states):
    """Mock SUMOConnection with given vehicles."""
    mock = Mock()
    ids = [vs.vehicle_id for vs in vehicle_states]
    state_map = {vs.vehicle_id: vs for vs in vehicle_states}

    mock.get_active_vehicle_ids.return_value = ids
    mock.is_vehicle_active.side_effect = lambda vid: vid in ids
    mock.get_vehicle_state.side_effect = lambda vid: state_map[vid]
    mock.set_vehicle_speed = Mock()
    mock.slow_down = Mock()
    mock.release_vehicle_speed = Mock()
    return mock


@pytest.fixture
def mock_sumo():
    """Mock SUMOConnection with V001 ego and V002 nearest front peer."""
    return _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
    ])


def test_hazard_injector_respects_probability():
    """Run 017: HAZARD_PROBABILITY=1.0 — all episodes get a hazard."""
    hazard_count = 0

    for i in range(100):
        inj = HazardInjector(seed=i)
        inj.reset()
        if inj._episode_will_have_hazard:
            hazard_count += 1

    assert hazard_count == 100, f"Got {hazard_count}, expected 100"


def test_hazard_injector_default_step_is_fixed():
    """Run 017: default hazard_step is fixed at 200 (not randomized)."""
    for seed in range(50):
        inj = HazardInjector(seed=seed)
        inj.reset()
        assert inj.hazard_step == 200, f"Seed {seed}: got {inj.hazard_step}, expected 200"


def test_hazard_injector_only_injects_in_window(mock_sumo):
    """Hazards only injected between steps 150-350."""
    inj = HazardInjector(seed=42)

    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    assert inj.maybe_inject(step=10, sumo=mock_sumo) is False
    assert inj.maybe_inject(step=149, sumo=mock_sumo) is False
    assert inj.maybe_inject(step=351, sumo=mock_sumo) is False

    assert inj.maybe_inject(step=200, sumo=mock_sumo) is True


def test_hazard_targets_nearest_peer(mock_sumo):
    """Nearest strategy targets nearest front peer with gradual slowdown."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=mock_sumo)

    mock_sumo.slow_down.assert_called_once()
    call_args = mock_sumo.slow_down.call_args
    assert call_args[0][0] == "V002"
    # Run 021: target speed is computed from desired decel × duration,
    # so it is no longer always 0.0 — just check it is non-negative.
    assert call_args[0][1] >= 0.0
    duration = call_args[0][2]
    assert 0.5 <= duration <= 1.5
    assert inj.hazard_target == "V002"


def test_hazard_targets_closest_among_multiple_peers():
    """With multiple peers, targets the nearest one."""
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=30.0),
        _make_vehicle_state("V003", x=0.0, y=12.0),
        _make_vehicle_state("V004", x=0.0, y=50.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=sumo)

    sumo.slow_down.assert_called_once()
    assert sumo.slow_down.call_args[0][0] == "V003"
    assert inj.hazard_target == "V003"
    assert inj.hazard_source_rank_ahead == 1


def test_hazard_skips_when_no_peers():
    """No injection if ego is the only vehicle."""
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is False
    sumo.slow_down.assert_not_called()


def test_hazard_skips_when_ego_not_active():
    """No injection if ego is not in the simulation."""
    sumo = _make_sumo([
        _make_vehicle_state("V002", x=10.0, y=0.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is False
    sumo.slow_down.assert_not_called()


def test_hazard_injector_returns_true_when_injected(mock_sumo):
    """maybe_inject() returns True if hazard occurred."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    result_before = inj.maybe_inject(step=49, sumo=mock_sumo)
    assert result_before is False

    result_at = inj.maybe_inject(step=50, sumo=mock_sumo)
    assert result_at is True

    result_after = inj.maybe_inject(step=51, sumo=mock_sumo)
    assert result_after is False


def test_hazard_injector_only_fires_once(mock_sumo):
    """Hazard is injected at most once per episode."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50
    inj._hazard_injected = False

    inj.maybe_inject(step=50, sumo=mock_sumo)
    inj.maybe_inject(step=50, sumo=mock_sumo)

    assert mock_sumo.slow_down.call_count == 1


def test_hazard_reset_clears_target():
    """Reset clears the previous target."""
    inj = HazardInjector(seed=42)
    inj._hazard_target = "V003"
    inj.reset()
    assert inj.hazard_target is None


def test_hazard_ignores_peers_behind_ego():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=-10.0),
    ])

    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is False
    sumo.slow_down.assert_not_called()


def test_uniform_front_peers_samples_multiple_sources():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
        _make_vehicle_state("V004", x=0.0, y=30.0),
    ])

    sampled = set()
    for seed in range(80):
        inj = HazardInjector(
            seed=seed,
            target_strategy=HazardInjector.TARGET_STRATEGY_UNIFORM_FRONT_PEERS,
        )
        inj._episode_will_have_hazard = True
        inj._hazard_step = 50
        inj.maybe_inject(step=50, sumo=sumo)
        sampled.add(inj.hazard_target)

    assert sampled >= {"V002", "V003", "V004"}


def test_fixed_vehicle_id_targets_requested_front_vehicle():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        fixed_vehicle_id="V003",
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is True
    sumo.slow_down.assert_called_once()
    assert sumo.slow_down.call_args[0][0] == "V003"
    assert inj.hazard_source_rank_ahead == 2


def test_fixed_vehicle_id_missing_returns_false():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        fixed_vehicle_id="V099",
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    assert inj.maybe_inject(step=50, sumo=sumo) is False
    sumo.slow_down.assert_not_called()


def test_fixed_vehicle_id_behind_ego_returns_false():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=-10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        fixed_vehicle_id="V002",
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    assert inj.maybe_inject(step=50, sumo=sumo) is False
    sumo.slow_down.assert_not_called()


def test_fixed_rank_ahead_targets_requested_rank():
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
        _make_vehicle_state("V003", x=0.0, y=20.0),
        _make_vehicle_state("V004", x=0.0, y=30.0),
    ])

    inj = HazardInjector(
        seed=42,
        target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_RANK_AHEAD,
        fixed_rank_ahead=2,
    )
    inj._episode_will_have_hazard = True
    inj._hazard_step = 50

    result = inj.maybe_inject(step=50, sumo=sumo)

    assert result is True
    sumo.slow_down.assert_called_once()
    assert sumo.slow_down.call_args[0][0] == "V003"
    assert inj.hazard_source_rank_ahead == 2


def test_reset_options_override_target_strategy_and_schedule():
    inj = HazardInjector(seed=1)

    inj.reset(
        options={
            "target_strategy": HazardInjector.TARGET_STRATEGY_FIXED_RANK_AHEAD,
            "fixed_rank_ahead": 3,
            "force_hazard": True,
            "hazard_step": 160,
        }
    )

    assert inj.target_strategy == HazardInjector.TARGET_STRATEGY_FIXED_RANK_AHEAD
    assert inj.hazard_step == 160
    assert inj._episode_will_have_hazard is True


# --- New tests for gradual hazard injection ---


def test_braking_duration_randomized_in_range(mock_sumo):
    """Braking duration is randomized between 0.5 and 1.5 seconds."""
    durations = set()
    for seed in range(100):
        inj = HazardInjector(seed=seed)
        inj._episode_will_have_hazard = True
        inj._hazard_step = 200
        inj._hazard_injected = False
        inj.maybe_inject(step=200, sumo=mock_sumo)
        if inj.braking_duration is not None:
            durations.add(round(inj.braking_duration, 2))
            assert 0.5 <= inj.braking_duration <= 1.5

    # Should have variety (not all the same value)
    assert len(durations) > 10


def test_slowdown_end_step_computed_correctly(mock_sumo):
    """_slowdown_end_step = injection_step + round(duration / 0.1)."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    inj.maybe_inject(step=200, sumo=mock_sumo)

    expected_end = 200 + int(round(inj.braking_duration / 0.1))
    assert inj._slowdown_end_step == expected_end


def test_maintain_hazard_pins_speed_after_slowdown(mock_sumo):
    """After slowdown duration, maintain_hazard pins target at hazard target speed."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    inj.maybe_inject(step=200, sumo=mock_sumo)
    end_step = inj._slowdown_end_step
    expected_speed = inj.hazard_target_speed

    # Before slowdown ends: no setSpeed call
    inj.maintain_hazard(step=end_step - 1, sumo=mock_sumo)
    mock_sumo.set_vehicle_speed.assert_not_called()

    # At slowdown end: pins at the computed target speed
    inj.maintain_hazard(step=end_step, sumo=mock_sumo)
    mock_sumo.set_vehicle_speed.assert_called_once_with("V002", expected_speed)


def test_maintain_hazard_only_pins_once(mock_sumo):
    """After pinning, maintain_hazard does not call setSpeed again."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    inj.maybe_inject(step=200, sumo=mock_sumo)
    end_step = inj._slowdown_end_step

    inj.maintain_hazard(step=end_step, sumo=mock_sumo)
    inj.maintain_hazard(step=end_step + 1, sumo=mock_sumo)
    inj.maintain_hazard(step=end_step + 10, sumo=mock_sumo)

    assert mock_sumo.set_vehicle_speed.call_count == 1


def test_maintain_hazard_noop_before_injection(mock_sumo):
    """maintain_hazard is safe to call before any hazard injection."""
    inj = HazardInjector(seed=42)
    inj.reset()

    # Should not raise or call anything
    inj.maintain_hazard(step=100, sumo=mock_sumo)
    mock_sumo.set_vehicle_speed.assert_not_called()
    mock_sumo.slow_down.assert_not_called()


def test_reset_clears_braking_duration_and_slowdown_end():
    """Reset clears gradual-braking tracking fields."""
    inj = HazardInjector(seed=42)
    inj._braking_duration = 3.0
    inj._slowdown_end_step = 250
    inj._hazard_target_speed = 5.0
    inj._desired_decel = 7.0
    inj._should_resolve = True
    inj._resolve_step = 300
    inj._hazard_resolved = True
    inj.reset()
    assert inj.braking_duration is None
    assert inj._slowdown_end_step is None
    assert inj.hazard_target_speed is None
    assert inj.desired_decel is None
    assert inj._should_resolve is False
    assert inj._resolve_step is None
    assert inj.hazard_resolved is False


def test_no_set_vehicle_speed_on_injection(mock_sumo):
    """maybe_inject uses slow_down, NOT set_vehicle_speed."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    inj.maybe_inject(step=200, sumo=mock_sumo)

    mock_sumo.slow_down.assert_called_once()
    mock_sumo.set_vehicle_speed.assert_not_called()


# --- Run 021: domain-randomized braking intensity tests ---


def test_desired_decel_randomized_in_range(mock_sumo):
    """Run 021: desired decel is randomized between HAZARD_DECEL_MIN and MAX."""
    decels = set()
    for seed in range(100):
        inj = HazardInjector(seed=seed)
        inj._episode_will_have_hazard = True
        inj._hazard_step = 200
        inj._hazard_injected = False
        inj.maybe_inject(step=200, sumo=mock_sumo)
        if inj.desired_decel is not None:
            decels.add(round(inj.desired_decel, 1))
            assert HazardInjector.HAZARD_DECEL_MIN <= inj.desired_decel <= HazardInjector.HAZARD_DECEL_MAX

    # Should have variety
    assert len(decels) > 5


def test_target_speed_computed_from_current_speed(mock_sumo):
    """Run 021: target speed = max(0, current_speed - decel * duration)."""
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    inj.maybe_inject(step=200, sumo=mock_sumo)

    # V002 speed=10.0 in mock_sumo
    expected = max(0.0, 10.0 - inj.desired_decel * inj.braking_duration)
    assert inj.hazard_target_speed == pytest.approx(expected, abs=1e-9)

    # slow_down was called with computed target speed
    call_args = mock_sumo.slow_down.call_args
    assert call_args[0][1] == pytest.approx(expected, abs=1e-9)


def test_target_speed_clamped_to_zero():
    """Run 021: target speed never goes negative even with high decel."""
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0, speed=3.0),  # slow target
    ])

    # Force maximum decel and duration for a guaranteed clamp
    inj = HazardInjector(seed=42)
    inj._episode_will_have_hazard = True
    inj._hazard_step = 200
    inj._hazard_injected = False

    inj.maybe_inject(step=200, sumo=sumo)

    assert inj.hazard_target_speed >= 0.0
    call_args = sumo.slow_down.call_args
    assert call_args[0][1] >= 0.0


# --- Run 021: resolved-hazard episode tests ---


def test_resolved_hazard_fraction():
    """Run 021: ~40% of hazards should resolve (statistical check)."""
    sumo = _make_sumo([
        _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0),
        _make_vehicle_state("V002", x=0.0, y=10.0),
    ])

    resolve_count = 0
    trials = 500
    for seed in range(trials):
        inj = HazardInjector(seed=seed)
        inj._episode_will_have_hazard = True
        inj._hazard_step = 200
        inj._hazard_injected = False
        inj.maybe_inject(step=200, sumo=sumo)
        if inj._should_resolve:
            resolve_count += 1

    ratio = resolve_count / trials
    # 40% ± 6% margin (generous for 500 trials)
    assert 0.30 <= ratio <= 0.50, f"Resolve ratio {ratio:.2f} outside [0.30, 0.50]"


def test_resolve_step_in_expected_range(mock_sumo):
    """Run 021: resolve step is slowdown_end + [20, 50]."""
    for seed in range(200):
        inj = HazardInjector(seed=seed)
        inj._episode_will_have_hazard = True
        inj._hazard_step = 200
        inj._hazard_injected = False
        inj.maybe_inject(step=200, sumo=mock_sumo)

        if inj._should_resolve:
            end = 200 + int(round(inj.braking_duration / 0.1))
            delay = inj._resolve_step - end
            assert HazardInjector.RESOLVE_DELAY_MIN <= delay <= HazardInjector.RESOLVE_DELAY_MAX


def test_resolved_hazard_releases_to_cf(mock_sumo):
    """Run 021: resolved hazard calls release_vehicle_speed after delay."""
    # Find a seed that produces a resolved hazard
    for seed in range(100):
        inj = HazardInjector(seed=seed)
        inj._episode_will_have_hazard = True
        inj._hazard_step = 200
        inj._hazard_injected = False
        mock_sumo.set_vehicle_speed.reset_mock()
        mock_sumo.release_vehicle_speed.reset_mock()

        inj.maybe_inject(step=200, sumo=mock_sumo)
        if not inj._should_resolve:
            continue

        end_step = 200 + int(round(inj.braking_duration / 0.1))
        resolve_step = inj._resolve_step

        # Phase 1: pin at target speed
        inj.maintain_hazard(step=end_step, sumo=mock_sumo)
        mock_sumo.set_vehicle_speed.assert_called_once()
        mock_sumo.release_vehicle_speed.assert_not_called()

        # Before resolve: no release
        inj.maintain_hazard(step=resolve_step - 1, sumo=mock_sumo)
        mock_sumo.release_vehicle_speed.assert_not_called()

        # At resolve step: release to CF
        inj.maintain_hazard(step=resolve_step, sumo=mock_sumo)
        mock_sumo.release_vehicle_speed.assert_called_once_with("V002")
        assert inj.hazard_resolved is True

        # Subsequent calls: no additional release
        mock_sumo.release_vehicle_speed.reset_mock()
        inj.maintain_hazard(step=resolve_step + 1, sumo=mock_sumo)
        mock_sumo.release_vehicle_speed.assert_not_called()
        return

    pytest.fail("No seed produced a resolved hazard in 100 trials")


def test_persistent_hazard_never_releases(mock_sumo):
    """Run 021: persistent hazard (no resolve) never calls release_vehicle_speed."""
    # Find a seed that produces a persistent hazard
    for seed in range(100):
        inj = HazardInjector(seed=seed)
        inj._episode_will_have_hazard = True
        inj._hazard_step = 200
        inj._hazard_injected = False
        mock_sumo.release_vehicle_speed.reset_mock()

        inj.maybe_inject(step=200, sumo=mock_sumo)
        if inj._should_resolve:
            continue

        end_step = 200 + int(round(inj.braking_duration / 0.1))

        # Pin, then run many more steps
        for s in range(end_step, end_step + 200):
            inj.maintain_hazard(step=s, sumo=mock_sumo)

        mock_sumo.release_vehicle_speed.assert_not_called()
        assert inj.hazard_resolved is False
        return

    pytest.fail("No seed produced a persistent hazard in 100 trials")


# =============================================================================
# Run 023: state-triggered onset tests
# =============================================================================


class TestStateBucketTrigger:
    """Tests for state_bucket trigger mode (Run 023 H1)."""

    def _make_convoy_sumo(self, gap_to_v002=30.0, gap_to_v003=60.0):
        """Convoy heading North: V001 at origin, V002 and V003 ahead."""
        return _make_sumo([
            _make_vehicle_state("V001", x=0.0, y=0.0, heading=0.0, speed=13.9),
            _make_vehicle_state("V002", x=0.0, y=gap_to_v002, heading=0.0, speed=13.9),
            _make_vehicle_state("V003", x=0.0, y=gap_to_v003, heading=0.0, speed=13.9),
        ])

    def test_state_bucket_mode_sets_trigger_mode(self):
        """Trigger mode is recorded correctly."""
        inj = HazardInjector(
            seed=42,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        inj.reset()
        assert inj.trigger_mode == HazardInjector.TRIGGER_MODE_STATE_BUCKET

    def test_state_bucket_does_not_fire_before_search_window(self):
        """No injection before STATE_BUCKET_SEARCH_START."""
        sumo = self._make_convoy_sumo(gap_to_v002=25.0)
        inj = HazardInjector(
            seed=42,
            target_strategy=HazardInjector.TARGET_STRATEGY_UNIFORM_FRONT_PEERS,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        inj.reset(options={"force_hazard": True})

        for step in range(HazardInjector.STATE_BUCKET_SEARCH_START):
            assert inj.maybe_inject(step=step, sumo=sumo) is False

    def test_state_bucket_fires_on_bucket_match(self):
        """Injection occurs when gap enters the sampled bucket."""
        # Place V002 at 25m (inside "medium" bucket [22, 35)).
        sumo = self._make_convoy_sumo(gap_to_v002=25.0)
        inj = HazardInjector(
            seed=42,
            target_strategy=HazardInjector.TARGET_STRATEGY_NEAREST,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        inj.reset(options={"force_hazard": True})

        # Manually set the desired bucket to match the gap.
        # We iterate until the search window starts, then check.
        injected = False
        for step in range(150, 351):
            # After first call at step 150, the search activates.
            if inj._sb_search_active and inj._sb_desired_bucket:
                # Force desired bucket to one that matches 25m.
                inj._sb_desired_bucket = "medium"
            if inj.maybe_inject(step=step, sumo=sumo):
                injected = True
                break

        assert injected is True
        assert inj.trigger_result == "bucket_match"
        assert inj.onset_gap_bucket == "medium"
        assert inj.onset_gap_m is not None
        assert 22.0 <= inj.onset_gap_m < 35.0

    def test_state_bucket_falls_back_at_search_end(self):
        """If bucket never matches, fallback fires at step 350."""
        # Place V002 at 10m — below all bucket ranges.
        sumo = self._make_convoy_sumo(gap_to_v002=10.0)
        inj = HazardInjector(
            seed=42,
            target_strategy=HazardInjector.TARGET_STRATEGY_NEAREST,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        inj.reset(options={"force_hazard": True})

        injected = False
        for step in range(150, 351):
            if inj.maybe_inject(step=step, sumo=sumo):
                injected = True
                break

        assert injected is True
        assert inj.trigger_result == "fallback_step"
        assert inj.onset_trigger_step == HazardInjector.STATE_BUCKET_FALLBACK_STEP

    def test_state_bucket_records_onset_metadata(self):
        """Onset telemetry fields are populated after injection."""
        sumo = self._make_convoy_sumo(gap_to_v002=30.0)
        inj = HazardInjector(
            seed=42,
            target_strategy=HazardInjector.TARGET_STRATEGY_NEAREST,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        inj.reset(options={"force_hazard": True})

        for step in range(150, 351):
            if inj._sb_search_active and inj._sb_desired_bucket:
                inj._sb_desired_bucket = "medium"  # 30m is in [22, 35)
            if inj.maybe_inject(step=step, sumo=sumo):
                break

        assert inj.onset_trigger_step is not None
        assert inj.onset_peer_count is not None
        assert inj.onset_peer_count >= 1
        assert inj.onset_closing_speed_mps is not None
        assert inj.onset_desired_rank_ahead is not None

    def test_explicit_hazard_step_overrides_state_bucket(self):
        """Passing hazard_step forces fixed_step even if default is state_bucket."""
        inj = HazardInjector(
            seed=42,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        inj.reset(options={"force_hazard": True, "hazard_step": 200})

        assert inj.trigger_mode == HazardInjector.TRIGGER_MODE_FIXED_STEP
        assert inj.hazard_step == 200

    def test_reset_clears_onset_metadata(self):
        """Reset clears all Run 023 onset fields."""
        inj = HazardInjector(
            seed=42,
            trigger_mode=HazardInjector.TRIGGER_MODE_STATE_BUCKET,
        )
        # Simulate some onset data.
        inj._onset_gap_bucket = "close"
        inj._onset_gap_m = 15.0
        inj._onset_trigger_step = 180
        inj._sb_search_active = True

        inj.reset()

        assert inj.onset_gap_bucket is None
        assert inj.onset_gap_m is None
        assert inj.onset_trigger_step is None
        assert inj._sb_search_active is False


class TestGapBucketHelpers:
    """Tests for gap bucket classification and sampling."""

    def test_gap_in_bucket_close(self):
        assert HazardInjector.gap_in_bucket(12.0, "close") is True
        assert HazardInjector.gap_in_bucket(21.9, "close") is True
        assert HazardInjector.gap_in_bucket(22.0, "close") is False
        assert HazardInjector.gap_in_bucket(11.9, "close") is False

    def test_gap_in_bucket_medium(self):
        assert HazardInjector.gap_in_bucket(22.0, "medium") is True
        assert HazardInjector.gap_in_bucket(34.9, "medium") is True
        assert HazardInjector.gap_in_bucket(35.0, "medium") is False

    def test_gap_in_bucket_far(self):
        assert HazardInjector.gap_in_bucket(35.0, "far") is True
        assert HazardInjector.gap_in_bucket(54.9, "far") is True
        assert HazardInjector.gap_in_bucket(55.0, "far") is False

    def test_gap_in_bucket_very_far(self):
        assert HazardInjector.gap_in_bucket(55.0, "very_far") is True
        assert HazardInjector.gap_in_bucket(84.9, "very_far") is True
        assert HazardInjector.gap_in_bucket(85.0, "very_far") is False

    def test_eligible_gap_buckets_rank_1(self):
        assert HazardInjector.eligible_gap_buckets(1) == ["close", "medium", "far"]

    def test_eligible_gap_buckets_rank_2(self):
        assert HazardInjector.eligible_gap_buckets(2) == ["medium", "far", "very_far"]

    def test_eligible_gap_buckets_rank_3_plus(self):
        assert HazardInjector.eligible_gap_buckets(3) == ["far", "very_far"]
        assert HazardInjector.eligible_gap_buckets(5) == ["far", "very_far"]

    def test_sample_onset_bucket_returns_valid(self):
        inj = HazardInjector(seed=42)
        for rank in [1, 2, 3, 4, 5]:
            eligible = HazardInjector.eligible_gap_buckets(rank)
            for _ in range(20):
                bucket = inj._sample_onset_bucket(rank)
                assert bucket in eligible


class TestLongitudinalGap:
    """Tests for the longitudinal gap computation."""

    def test_gap_heading_north(self):
        """Target ahead (north) of ego heading north → positive gap."""
        gap = HazardInjector._longitudinal_gap(0, 0, 0.0, 0, 30)
        assert gap == pytest.approx(30.0, abs=0.1)

    def test_gap_heading_north_behind(self):
        """Target behind ego → negative gap."""
        gap = HazardInjector._longitudinal_gap(0, 0, 0.0, 0, -10)
        assert gap == pytest.approx(-10.0, abs=0.1)

    def test_gap_heading_east(self):
        """Target east of ego heading east (90°) → positive gap."""
        gap = HazardInjector._longitudinal_gap(0, 0, 90.0, 30, 0)
        assert gap == pytest.approx(30.0, abs=0.1)
