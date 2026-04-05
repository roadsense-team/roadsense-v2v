#!/usr/bin/env python3
"""
ConvoyEnv GUI Demo
==================

Watch the ConvoyEnv simulation in real-time with SUMO-GUI.
See vehicles move, hazard injection events, and collision avoidance.

What you'll see in SUMO-GUI:
- V003: Lead vehicle (front of convoy)
- V002: Middle vehicle
- V001: Ego vehicle (rear, the one being controlled)

What's happening internally (not visible):
- ESP-NOW Emulator adding latency/packet loss to V2V messages
- Observation builder creating state vector for RL agent
- RL model (PPO) deciding braking intensity from V2V signals

Usage (from inside Docker):
    # With trained Run 025 model (recommended for PoC demo):
    python3 demo_convoy_gui.py \
        --model_path ml/results/run_025_replay_v1/checkpoints/replay_ft_500000_steps.zip \
        --ego_stack_frames 3

    # Random actions (testing only):
    python3 demo_convoy_gui.py --scenario var_tight_convoy --delay 0.2
"""

import argparse
import os
import sys
import time

# Ensure imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MAX_DECEL = 8.0  # m/s², matches training


def list_scenarios():
    """List available scenarios."""
    scenarios_dir = os.path.join(SCRIPT_DIR, "scenarios")
    available = []
    if os.path.isdir(scenarios_dir):
        for name in sorted(os.listdir(scenarios_dir)):
            cfg = os.path.join(scenarios_dir, name, "scenario.sumocfg")
            if os.path.exists(cfg):
                available.append(name)
    return available


def _decel_bar(action_val: float, width: int = 10) -> str:
    """Render a visual brake intensity bar: [####......] 3.2 m/s²."""
    filled = int(round(action_val * width))
    bar = "#" * filled + "." * (width - filled)
    decel = action_val * MAX_DECEL
    return f"[{bar}] {decel:4.1f}"


def _load_model(model_path: str):
    """Load a trained PPO model. Returns (model, True) or (None, False)."""
    if not model_path:
        return None, False
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    return model, True


def run_demo(
    scenario: str,
    delay: float,
    episodes: int,
    seed: int,
    model_path: str = None,
    ego_stack_frames: int = 1,
    hazard_vehicle: str = "V003",
    cone_half_angle_deg: float = 45.0,
    hazard_step: int = None,
    max_relay_hops: int = 3,
):
    """Run the visualization demo."""
    from envs.convoy_env import ConvoyEnv

    # Find scenario config
    sumo_cfg = os.path.join(SCRIPT_DIR, "scenarios", scenario, "scenario.sumocfg")
    if not os.path.exists(sumo_cfg):
        print(f"ERROR: Scenario not found: {sumo_cfg}")
        print(f"Available scenarios: {', '.join(list_scenarios())}")
        sys.exit(1)

    # Prefer measured emulator params (convoy-calibrated)
    emulator_params = os.path.join(SCRIPT_DIR, "espnow_emulator", "emulator_params_measured.json")
    if not os.path.exists(emulator_params):
        emulator_params = os.path.join(SCRIPT_DIR, "espnow_emulator", "emulator_params_5m.json")
    if not os.path.exists(emulator_params):
        emulator_params = None
        print("NOTE: Using default emulator params")

    # Load model
    model, using_model = _load_model(model_path)

    # Print header
    print("=" * 70)
    print("ConvoyEnv GUI Demo — RoadSense V2V PoC")
    print("=" * 70)
    print(f"  Scenario:         {scenario}")
    print(f"  SUMO Config:      {sumo_cfg}")
    print(f"  Emulator:         {emulator_params or 'default'}")
    print(f"  Agent:            {'PPO Model (' + os.path.basename(model_path) + ')' if using_model else 'RANDOM (no model loaded)'}")
    print(f"  Ego Stack Frames: {ego_stack_frames}")
    print(f"  Step Delay:       {delay}s")
    print(f"  Episodes:         {episodes}")
    print("=" * 70)
    print()
    print("WHAT TO WATCH IN SUMO-GUI:")
    print("  - Three vehicles in convoy on a road")
    print("  - V001 (ego) follows V002 and V003")
    print("  - Watch for sudden braking (HazardInjector)")
    print("  - Watch gap distance shrinking/growing")
    print()
    print("CONSOLE OUTPUT KEY:")
    print("  dist     = Gap to nearest vehicle (collision if < 5m)")
    print("  action   = Brake intensity [0.0-1.0] → deceleration m/s²")
    print("  reward   = Positive=good, Negative=bad")
    print("  [HAZARD] = HazardInjector triggered lead vehicle braking")
    print()
    print("=" * 70)
    input("Press ENTER to start (SUMO-GUI window will open)...")
    print()

    # Create environment with GUI
    # Force V003 (lead) as hazard target so the braking chain is visible:
    # V003 brakes → V2V signal → V001 (ego) reacts before V002 even slows
    from envs.hazard_injector import HazardInjector

    # DEMO OVERRIDE: force hazard vehicle to FULL emergency stop.
    # At ~22 m/s cruise, need duration >= 2.2s at decel=10 to reach 0.
    # Use 3.0s to guarantee full stop from any speed in the scenario.
    HazardInjector.HAZARD_DECEL_MIN = 25.0   # ensures target_speed=0 even with short duration
    HazardInjector.HAZARD_DECEL_MAX = 25.0
    HazardInjector.BRAKING_DURATION_MIN = 1.0  # abrupt emergency stop (~20 m/s² actual)
    HazardInjector.BRAKING_DURATION_MAX = 1.0
    HazardInjector.HAZARD_RESOLVE_PROB = 0.0   # hazard vehicle stays stopped
    HazardInjector.HAZARD_WINDOW_START = 50    # allow early injection for curve demo

    print(f"DEBUG: hazard_vehicle={hazard_vehicle!r}, strategy={HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID}")

    # Build emulator with max_relay_hops override for curve scenarios (5+ vehicles)
    from ml.espnow_emulator.espnow_emulator import ESPNOWEmulator
    emulator = ESPNOWEmulator(params_file=emulator_params) if emulator_params else ESPNOWEmulator()
    emulator.params.setdefault('mesh', {})['max_relay_hops'] = max_relay_hops

    env = ConvoyEnv(
        sumo_cfg=sumo_cfg,
        emulator=emulator,
        max_steps=500,
        hazard_injection=True,
        hazard_target_strategy=HazardInjector.TARGET_STRATEGY_FIXED_VEHICLE_ID,
        hazard_fixed_vehicle_id=hazard_vehicle,
        gui=True,
        ego_stack_frames=ego_stack_frames,
        cone_half_angle_deg=cone_half_angle_deg,
    )

    try:
        for episode in range(episodes):
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1}/{episodes}")
            print(f"{'='*70}")
            print(f"{'Step':>4} | {'Distance':>8} | {'Brake Intensity':>20} | {'Reward':>7} | {'EgoSpd':>6} | Event")
            print("-" * 75)

            reset_opts = {}
            if hazard_step is not None:
                reset_opts["hazard_step"] = hazard_step
            obs, info = env.reset(seed=seed + episode, options=reset_opts if reset_opts else None)
            hi = env.hazard_injector
            print(f"DEBUG reset: hazard_step={hi._hazard_step}, will_have={hi._episode_will_have_hazard}, fixed_vid={hi._fixed_vehicle_id}")
            done = False
            total_reward = 0.0
            step = 0

            while not done:
                if using_model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                # Extract scalar action value
                action_val = float(action.item()) if hasattr(action, "item") else float(action)

                # Use ground truth distance (stable, no V2V packet loss flicker)
                gt_dist = info.get("gt_distance", info["distance"])

                # Debug around hazard step
                if 199 <= step <= 205:
                    print(f"DEBUG step={step} env._step_count={env._step_count} hi._hazard_injected={hi._hazard_injected} hi._hazard_injection_failed={hi._hazard_injection_failed} hi._hazard_injection_attempted={hi._hazard_injection_attempted} reason={hi._hazard_injection_failed_reason}")

                # Build event string
                events = []
                if info.get("hazard_injected"):
                    events.append("[HAZARD]")
                if gt_dist < 10:
                    events.append("[CLOSE!]")
                if terminated:
                    events.append("[COLLISION]")
                event_str = " ".join(events)

                # Ego speed from observation (first 6 dims = current frame)
                ego_speed = obs["ego"][0] * 30.0

                print(
                    f"{step:4d} | {info['distance']:7.1f}m | "
                    f"{_decel_bar(action_val)} m/s² | {reward:+7.1f} | "
                    f"{ego_speed:5.1f} | {event_str}"
                )

                # Delay for visualization
                time.sleep(delay)

            # Episode summary
            print("-" * 75)
            outcome = "COLLISION" if terminated else "COMPLETED"
            print(f"Episode {episode + 1} finished: {outcome}")
            print(f"  Steps:        {step}")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Final Gap:    {info['distance']:.1f}m")

            if episode < episodes - 1:
                print()
                input("Press ENTER for next episode (or Ctrl+C to quit)...")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    finally:
        env.close()
        print("\nSUMO closed. Demo complete.")


def main():
    parser = argparse.ArgumentParser(
        description="ConvoyEnv GUI Demo - Watch the simulation in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available scenarios: {', '.join(list_scenarios())}",
    )
    parser.add_argument(
        "--scenario", "-s",
        default="demo_poc",
        help="Scenario to run (default: demo_poc)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.1,
        help="Delay between steps in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default=None,
        help="Path to trained PPO model (.zip). If not provided, uses random actions.",
    )
    parser.add_argument(
        "--ego_stack_frames",
        type=int,
        default=3,
        help="Number of ego observation frames to stack (default: 3 for Run 025+)",
    )
    parser.add_argument(
        "--hazard_vehicle",
        type=str,
        default="V003",
        help="Vehicle ID for hazard injection (default: V003, use V006 for curve scenario)",
    )
    parser.add_argument(
        "--cone_half_angle",
        type=float,
        default=45.0,
        help="Observation cone half-angle in degrees (default: 45, use 90 for curve scenarios)",
    )
    parser.add_argument(
        "--hazard_step",
        type=int,
        default=None,
        help="Override hazard injection step (default: 200, use 150 for earlier injection)",
    )
    parser.add_argument(
        "--max_relay_hops",
        type=int,
        default=3,
        help="Max mesh relay hops (default: 3, use 5 for 6-vehicle curve scenario)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scenarios and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for s in list_scenarios():
            print(f"  - {s}")
        sys.exit(0)

    run_demo(
        scenario=args.scenario,
        delay=args.delay,
        episodes=args.episodes,
        seed=args.seed,
        model_path=args.model_path,
        ego_stack_frames=args.ego_stack_frames,
        hazard_vehicle=args.hazard_vehicle,
        cone_half_angle_deg=args.cone_half_angle,
        hazard_step=args.hazard_step,
        max_relay_hops=args.max_relay_hops,
    )


if __name__ == "__main__":
    main()
