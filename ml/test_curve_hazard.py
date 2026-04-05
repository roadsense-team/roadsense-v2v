#!/usr/bin/env python3
"""Quick diagnostic: does hazard injection fire on the curve scenario?"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from envs.convoy_env import ConvoyEnv
from envs.hazard_injector import HazardInjector

sumo_cfg = os.path.join(SCRIPT_DIR, "scenarios", "base_curve", "scenario.sumocfg")
emulator_params = os.path.join(SCRIPT_DIR, "espnow_emulator", "emulator_params_measured.json")

# Same overrides as the demo
HazardInjector.HAZARD_DECEL_MIN = 10.0
HazardInjector.HAZARD_DECEL_MAX = 10.0
HazardInjector.BRAKING_DURATION_MIN = 1.5
HazardInjector.BRAKING_DURATION_MAX = 1.5
HazardInjector.HAZARD_RESOLVE_PROB = 0.0

env = ConvoyEnv(
    sumo_cfg=sumo_cfg,
    emulator_params_path=emulator_params,
    max_steps=500,
    hazard_injection=True,
    hazard_target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
    hazard_fixed_vehicle_id="V006",
    gui=False,
    ego_stack_frames=3,
)

obs, info = env.reset(seed=42)
print(f"Reset OK. Scheduled hazard_step={info.get('hazard_step')}")
print(f"Hazard injector state:")
hi = env.hazard_injector
print(f"  _episode_will_have_hazard = {hi._episode_will_have_hazard}")
print(f"  _hazard_step = {hi._hazard_step}")
print(f"  _fixed_vehicle_id = {hi._fixed_vehicle_id}")
print(f"  _target_strategy = {hi._target_strategy}")

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if step in (195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205):
        active = env.sumo.get_active_vehicle_ids()
        print(f"\nStep {step}: active_vehicles={active}")
        if info.get("hazard_injected"):
            print(f"  >>> HAZARD INJECTED! target={info.get('hazard_source_id')}")
        if hi._hazard_injection_failed:
            print(f"  >>> HAZARD FAILED! reason={hi._hazard_injection_failed_reason}")

    if info.get("hazard_injected") and step > 205:
        # Only print once after initial window
        break

    if terminated or truncated:
        print(f"\nEpisode ended at step {step}: terminated={terminated}, truncated={truncated}")
        break

if not hi._hazard_injected:
    print(f"\n*** HAZARD NEVER INJECTED ***")
    print(f"  _hazard_injection_attempted = {hi._hazard_injection_attempted}")
    print(f"  _hazard_injection_failed = {hi._hazard_injection_failed}")
    print(f"  _hazard_injection_failed_reason = {hi._hazard_injection_failed_reason}")
else:
    print(f"\nHazard injected at step {hi._hazard_step}, target={hi._hazard_target}")

env.close()
