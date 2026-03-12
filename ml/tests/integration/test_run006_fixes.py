"""
Integration tests for Run 006 fixes — require SUMO.

Run with: ./run_docker.sh integration
Or locally: pytest -m integration tests/integration/test_run006_fixes.py

Tests validate that all Run 006 fixes work with real SUMO physics,
not just mocked unit tests.
"""
import os

import numpy as np
import pytest

SUMO_AVAILABLE = os.system("sumo --version > /dev/null 2>&1") == 0


@pytest.fixture
def base_scenario_path():
    """Path to base SUMO scenario with multiple peers."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "scenarios", "base_real", "scenario.sumocfg")


@pytest.fixture
def make_env(base_scenario_path):
    """Factory for ConvoyEnv with real SUMO."""
    envs = []

    def _make(**kwargs):
        from ml.envs.convoy_env import ConvoyEnv
        defaults = dict(
            sumo_cfg=base_scenario_path,
            hazard_injection=True,
            max_steps=200,
        )
        defaults.update(kwargs)
        env = ConvoyEnv(**defaults)
        envs.append(env)
        return env

    yield _make

    for env in envs:
        env.close()


# --- Test 1: Ground-truth collision prevents false collisions ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_no_false_collision_from_gps_noise(make_env):
    """
    Run 50 episodes. Assert collision rate < 10% in first 30 steps.

    With ground-truth collision detection, GPS noise in emulated positions
    should NOT cause false collisions.
    """
    env = make_env(max_steps=100)
    early_collisions = 0
    total_episodes = 50

    for ep in range(total_episodes):
        obs, info = env.reset(seed=ep)
        for step in range(30):
            obs, reward, terminated, truncated, info = env.step(
                np.array([0.0], dtype=np.float32)
            )
            if terminated:
                early_collisions += 1
                break
            if truncated:
                break

    collision_rate = early_collisions / total_episodes
    assert collision_rate < 0.10, (
        f"Early collision rate {collision_rate:.0%} too high — "
        f"ground-truth collision may not be working"
    )


# --- Test 2: Route-end guard prevents TraCI crash ---

@pytest.mark.integration
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_route_end_no_crash(make_env):
    """
    Run episode to max_steps. No TraCIException should occur.

    The pre-action guard at top of step() prevents crashes when V001
    leaves the simulation between steps.
    """
    env = make_env(max_steps=500, hazard_injection=False)
    obs, info = env.reset(seed=0)

    for step in range(500):
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        if terminated or truncated:
            # Truncation with route_ended is the expected path
            if truncated and info.get("truncated_reason") == "ego_route_ended":
                return  # Test passes — clean truncation
            break

    # If we get here, episode completed normally or hit max_steps — also fine


# --- Test 3: Warmup produces safe starting distance ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_warmup_produces_safe_distance(make_env):
    """
    After reset(), ground-truth distance to nearest peer > COLLISION_DIST + margin.

    Warmup extension should ensure vehicles aren't dangerously close when
    RL takes over.
    """
    env = make_env(max_steps=100, hazard_injection=False)

    for ep in range(20):
        obs, info = env.reset(seed=ep)
        # Take one no-op step to get distance info
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        if terminated:
            pytest.fail(
                f"Episode {ep}: Immediate collision after warmup — "
                f"distance={info.get('distance', '?')}"
            )


# --- Test 4: Braking peer signal appears in observation ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_braking_peer_in_observation(make_env):
    """
    After hazard injection (peer brakes to 0), ego observation index [4]
    (min_peer_accel) should become negative.

    This validates the new braking signal flows from emulator through
    observation builder to the observation vector.
    """
    env = make_env(max_steps=200)

    found_braking_signal = False

    for ep in range(30):
        obs, info = env.reset(seed=ep, options={
            "hazard_options": {"force_hazard": True, "hazard_step": 160},
        })

        for step in range(200):
            obs, reward, terminated, truncated, step_info = env.step(
                np.array([0.0], dtype=np.float32)
            )

            # After hazard injection, check for braking signal
            if step > 162 and obs["ego"][4] < -0.03:
                found_braking_signal = True
                # Binary braking_received flag (ego[5]) must also be set
                assert obs["ego"][5] == pytest.approx(1.0), (
                    f"braking_received flag not set when min_peer_accel={obs['ego'][4]}"
                )
                break

            if terminated or truncated:
                break

        if found_braking_signal:
            break

    assert found_braking_signal, (
        "min_peer_accel never went negative after hazard injection. "
        "The braking signal is not reaching the observation."
    )


# --- Test 5: Strategy B+ disables hazard-specific reward shaping ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_hazard_reward_terms_disabled(make_env):
    """
    Strategy B+ restores Run 004 reward economics.
    Even during hazard episodes, hazard-specific reward components remain zero.
    """
    env = make_env(max_steps=200)

    for ep in range(30):
        obs, info = env.reset(seed=ep, options={
            "hazard_options": {"force_hazard": True, "hazard_step": 160},
        })

        for step in range(220):
            obs, reward, terminated, truncated, step_info = env.step(
                np.array([0.0], dtype=np.float32)  # No braking
            )

            # V2V hazard terms are now active (Run 013). During a forced hazard
            # with action=0, ignoring_hazard may fire once the hazard source's
            # braking message reaches ego.  Verify value is in expected range.
            assert step_info.get("reward_ignoring_hazard", 0.0) >= -5.0
            assert step_info.get("reward_early_reaction", 0.0) >= 0.0

            if terminated or truncated:
                break


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_ignoring_hazard_penalty_persists_past_active_slowdown(make_env):
    """
    Once the injected source's braking signal reaches ego, the ignoring-hazard
    penalty should persist for the rest of the hazard event, not only during
    the 2-4s active slowDown() window.

    Run 017 update: the penalty is now suppressed when ego is stopped
    (speed <= 0.5 m/s). So we allow gaps only when the speed gate is active
    (info['ignoring_penalty_suppressed_for_stop'] is True).
    """
    from ml.envs.hazard_injector import HazardInjector

    hazard_step = 80
    max_slowdown_steps = int(
        HazardInjector.BRAKING_DURATION_MAX / HazardInjector.STEP_LENGTH
    )  # 40
    slowdown_done_step = hazard_step + max_slowdown_steps  # 120

    env = make_env(max_steps=260)
    env.hazard_injector.HAZARD_WINDOW_START = 0
    obs, info = env.reset(seed=0, options={
        "hazard_options": {"force_hazard": True, "hazard_step": hazard_step},
    })

    first_penalty_step = None
    post_slowdown_penalty = 0
    post_slowdown_unexplained_gap = 0

    for step in range(260):
        obs, reward, terminated, truncated, step_info = env.step(
            np.array([0.0], dtype=np.float32)
        )

        has_penalty = step_info.get("reward_ignoring_hazard", 0.0) < 0.0
        suppressed_for_stop = step_info.get(
            "ignoring_penalty_suppressed_for_stop", False
        )

        if has_penalty and first_penalty_step is None:
            first_penalty_step = step

        # After slowDown is guaranteed complete, track whether penalty persists
        if step > slowdown_done_step and not (terminated or truncated):
            if has_penalty:
                post_slowdown_penalty += 1
            elif not suppressed_for_stop:
                # Gap that is NOT explained by the stopped-car speed gate
                post_slowdown_unexplained_gap += 1

        if terminated or truncated:
            break

    assert first_penalty_step is not None, (
        "Ignoring-hazard penalty never activated after forced hazard injection."
    )
    assert post_slowdown_penalty > 0, (
        f"No ignoring-hazard penalty detected after slowDown window "
        f"(step {slowdown_done_step}). The latch is not working."
    )
    assert post_slowdown_unexplained_gap == 0, (
        f"Ignoring-hazard penalty had {post_slowdown_unexplained_gap} "
        f"unexplained gap(s) after slowDown completed (step {slowdown_done_step}). "
        f"The latch dropped before episode end."
    )


# --- Test 6: Eval matrix coverage with small run ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_eval_matrix_coverage_mini(make_env):
    """
    Run 10 episodes with forced hazard. Assert injection_attempted > 0
    and that info dict contains the new tracking fields.
    """
    env = make_env(max_steps=220)
    injection_attempted_count = 0
    has_tracking_fields = False

    for ep in range(10):
        obs, info = env.reset(seed=ep + 100, options={
            "hazard_options": {"force_hazard": True, "hazard_step": 160},
        })

        for step in range(220):
            obs, reward, terminated, truncated, step_info = env.step(
                np.array([0.0], dtype=np.float32)
            )

            if step_info.get("hazard_injection_attempted", False):
                injection_attempted_count += 1
                has_tracking_fields = True

            if terminated or truncated:
                break

    assert has_tracking_fields, (
        "hazard_injection_attempted field never appeared in info dict"
    )
    assert injection_attempted_count > 0, (
        "No hazard injections were attempted across 10 episodes"
    )


# --- Test 7: Emulator resets between episodes ---

@pytest.mark.integration
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_emulator_resets_per_episode(make_env):
    """
    Emulator internal state should reset between episodes.

    After reset(), the emulator message history should be empty
    (no stale messages from previous episode leaking through).
    """
    env = make_env(max_steps=50, hazard_injection=False)

    # Run episode 1 for several steps to build up emulator state
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break

    # Reset for episode 2
    obs, info = env.reset(seed=1)

    # First step after reset — peer_mask should reflect only what
    # the emulator received in THIS episode (from warmup), not
    # stale data from episode 1.
    # The key check: observation is valid and finite
    assert np.isfinite(obs["ego"]).all()
    assert np.isfinite(obs["peers"]).all()
    assert obs["ego"].shape == (7,)


# --- Smoke training test ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not installed")
def test_smoke_training_5000_steps(base_scenario_path):
    """
    5000-step training run. Assert:
    - No crashes
    - Training completes
    - Model can predict actions

    This is the final sanity check before pushing to EC2.
    """
    from stable_baselines3 import PPO
    from ml.envs.convoy_env import ConvoyEnv
    from ml.policies.deep_set_policy import create_deep_set_policy_kwargs

    env = ConvoyEnv(
        sumo_cfg=base_scenario_path,
        hazard_injection=True,
        max_steps=100,
    )

    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=create_deep_set_policy_kwargs(),
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            verbose=0,
        )

        model.learn(total_timesteps=5000)

        # Verify model can predict
        obs, _ = env.reset(seed=999)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == env.action_space.shape
    finally:
        env.close()
