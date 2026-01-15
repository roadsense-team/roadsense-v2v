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
- Reward calculator scoring each action

Usage (from inside Docker):
    python3 demo_convoy_gui.py
    python3 demo_convoy_gui.py --scenario var_tight_convoy --delay 0.2
"""

import argparse
import os
import sys
import time

# Ensure imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


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


def run_demo(scenario: str, delay: float, episodes: int, seed: int):
    """Run the visualization demo."""
    from envs.convoy_env import ConvoyEnv

    # Find scenario config
    sumo_cfg = os.path.join(SCRIPT_DIR, "scenarios", scenario, "scenario.sumocfg")
    if not os.path.exists(sumo_cfg):
        print(f"ERROR: Scenario not found: {sumo_cfg}")
        print(f"Available scenarios: {', '.join(list_scenarios())}")
        sys.exit(1)

    # Find emulator params (optional)
    emulator_params = os.path.join(SCRIPT_DIR, "espnow_emulator", "emulator_params_5m.json")
    if not os.path.exists(emulator_params):
        emulator_params = None
        print("NOTE: Using default emulator params")

    # Print header
    print("=" * 70)
    print("ConvoyEnv GUI Demo")
    print("=" * 70)
    print(f"  Scenario:      {scenario}")
    print(f"  SUMO Config:   {sumo_cfg}")
    print(f"  Emulator:      {emulator_params or 'default'}")
    print(f"  Step Delay:    {delay}s (slower = easier to watch)")
    print(f"  Episodes:      {episodes}")
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
    print("  action   = MAINTAIN(0), CAUTION(1), BRAKE(2), EMERGENCY(3)")
    print("  reward   = Positive=good, Negative=bad")
    print("  [HAZARD] = HazardInjector triggered lead vehicle braking")
    print()
    print("=" * 70)
    input("Press ENTER to start (SUMO-GUI window will open)...")
    print()

    # Create environment with GUI
    env = ConvoyEnv(
        sumo_cfg=sumo_cfg,
        emulator_params_path=emulator_params,
        max_steps=200,  # Shorter for demo
        hazard_injection=True,
        gui=True,  # Enable SUMO-GUI
    )

    ACTION_NAMES = ["MAINTAIN ", "CAUTION  ", "BRAKE    ", "EMERGENCY"]

    try:
        for episode in range(episodes):
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1}/{episodes}")
            print(f"{'='*70}")
            print(f"{'Step':>4} | {'Distance':>8} | {'Action':>9} | {'Reward':>7} | {'EgoSpd':>6} | Event")
            print("-" * 70)

            obs, info = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            step = 0

            while not done:
                # Random action (replace with trained agent)
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                # Build event string
                events = []
                if info.get("hazard_injected"):
                    events.append("[HAZARD]")
                if info["distance"] < 10:
                    events.append("[CLOSE!]")
                if terminated:
                    events.append("[COLLISION]")
                event_str = " ".join(events)

                # Ego speed from observation (normalized by 30)
                ego_speed = obs["ego"][0] * 30.0

                print(
                    f"{step:4d} | {info['distance']:7.1f}m | "
                    f"{ACTION_NAMES[action]} | {reward:+7.1f} | "
                    f"{ego_speed:5.1f} | {event_str}"
                )

                # Delay for visualization
                time.sleep(delay)

            # Episode summary
            print("-" * 70)
            outcome = "COLLISION" if terminated else "TIMEOUT/EXIT"
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
        default="base",
        help="Scenario to run (default: base)",
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
        default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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
    )


if __name__ == "__main__":
    main()
