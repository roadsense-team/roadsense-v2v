#!/usr/bin/env python3
"""
ESP-NOW Emulator Demonstration Script

This script demonstrates the full lifecycle of the ESPNOWEmulator:
1. Initialize emulator with custom parameters
2. Simulate a 3-car convoy scenario
3. Show "Real vs Observed" state side-by-side (latency/noise effects)
4. Demonstrate domain randomization across episodes
5. Show packet loss and staleness handling

Run with:
    python -m examples.demo_emulator

Or from the ml/ directory:
    python examples/demo_emulator.py

Author: Amir Khalifa
Date: December 29, 2025
"""

import sys
import os

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espnow_emulator import ESPNOWEmulator, V2VMessage


def create_convoy_message(vehicle_id: str, position_m: float,
                          speed_mps: float, accel: float,
                          timestamp_ms: int) -> V2VMessage:
    """
    Create a V2VMessage for a vehicle in the convoy.

    Args:
        vehicle_id: Vehicle identifier (e.g., 'V002', 'V003')
        position_m: Position along road in meters
        speed_mps: Speed in m/s
        accel: Longitudinal acceleration in m/s^2
        timestamp_ms: Message timestamp in milliseconds

    Returns:
        V2VMessage instance
    """
    # Convert position to lat/lon (simplified: 1m ≈ 0.00001 degrees)
    base_lat = 32.085  # Tel Aviv area
    base_lon = 34.781
    lat = base_lat + (position_m * 0.00001)

    return V2VMessage(
        vehicle_id=vehicle_id,
        lat=lat,
        lon=base_lon,
        speed=speed_mps,
        heading=90.0,  # Heading east
        accel_x=accel,
        accel_y=0.0,
        accel_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        timestamp_ms=timestamp_ms
    )


def print_comparison(real_msg: V2VMessage, obs: dict, vehicle_id: str):
    """Print side-by-side comparison of real vs observed state."""
    vid_lower = vehicle_id.lower()

    print(f"\n  {vehicle_id} State Comparison:")
    print(f"  {'Field':<12} {'Real':>12} {'Observed':>12} {'Diff':>10}")
    print(f"  {'-'*50}")

    # Speed comparison
    real_speed = real_msg.speed
    obs_speed = obs.get(f'{vid_lower}_speed', 0.0)
    print(f"  {'Speed (m/s)':<12} {real_speed:>12.2f} {obs_speed:>12.2f} "
          f"{obs_speed - real_speed:>+10.2f}")

    # Acceleration comparison
    real_accel = real_msg.accel_x
    obs_accel = obs.get(f'{vid_lower}_accel_x', 0.0)
    print(f"  {'Accel (m/s2)':<12} {real_accel:>12.2f} {obs_accel:>12.2f} "
          f"{obs_accel - real_accel:>+10.2f}")

    # Message age
    age_ms = obs.get(f'{vid_lower}_age_ms', 9999)
    valid = obs.get(f'{vid_lower}_valid', False)
    status = "VALID" if valid else "STALE"
    print(f"  {'Age (ms)':<12} {'N/A':>12} {age_ms:>12} {status:>10}")


def demo_basic_transmission():
    """Demonstrate basic transmission with latency and noise effects."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Transmission with Communication Effects")
    print("="*60)

    # Initialize emulator with fixed seed for reproducibility
    emulator = ESPNOWEmulator(domain_randomization=False, seed=42)

    print("\nScenario: 3-car convoy (V003 -> V002 -> V001)")
    print("V001 (ego) is at position 0m, receiving messages from V002 and V003")

    # Positions: V003 at 50m, V002 at 25m, V001 at 0m (ego)
    v001_pos = (0.0, 0.0)
    v002_pos = (25.0, 0.0)
    v003_pos = (50.0, 0.0)

    # Create messages (perfect SUMO data)
    t = 1000  # Start time in ms
    v002_msg = create_convoy_message('V002', 25.0, 15.0, 0.0, t)
    v003_msg = create_convoy_message('V003', 50.0, 15.0, 0.0, t)

    print(f"\nTime T={t}ms: Transmitting messages...")

    # Transmit messages
    result_v002 = emulator.transmit(v002_msg, v002_pos, v001_pos, t)
    result_v003 = emulator.transmit(v003_msg, v003_pos, v001_pos, t)

    if result_v002:
        print(f"  V002 -> V001: SUCCESS (latency={result_v002.age_ms}ms, "
              f"arrives at T={result_v002.received_at_ms}ms)")
    else:
        print("  V002 -> V001: PACKET LOST")

    if result_v003:
        print(f"  V003 -> V001: SUCCESS (latency={result_v003.age_ms}ms, "
              f"arrives at T={result_v003.received_at_ms}ms)")
    else:
        print("  V003 -> V001: PACKET LOST")

    # Show observations at different times (causality demonstration)
    print(f"\nTime T={t}ms: Checking observation (before arrival)...")
    obs_before = emulator.get_observation(ego_speed=15.0, current_time_ms=t)
    print(f"  V002 visible: {obs_before['v002_valid']} (age={obs_before['v002_age_ms']}ms)")
    print(f"  V003 visible: {obs_before['v003_valid']} (age={obs_before['v003_age_ms']}ms)")

    # Wait for arrival
    t_after = t + 100  # 100ms later
    print(f"\nTime T={t_after}ms: Checking observation (after arrival)...")
    obs_after = emulator.get_observation(ego_speed=15.0, current_time_ms=t_after)

    if result_v002:
        print_comparison(v002_msg, obs_after, 'V002')
    if result_v003:
        print_comparison(v003_msg, obs_after, 'V003')


def demo_convoy_simulation():
    """Simulate a convoy scenario with hard braking."""
    print("\n" + "="*60)
    print("DEMO 2: Convoy Simulation with Emergency Braking")
    print("="*60)

    emulator = ESPNOWEmulator(domain_randomization=False, seed=123)

    print("\nScenario: V003 performs emergency braking at T=2000ms")
    print("Initial: All vehicles at 15 m/s, gaps of 25m")

    # Initial positions
    v002_x = 25.0
    v003_x = 50.0

    # Initial speeds
    v002_speed = 15.0
    v003_speed = 15.0

    # Simulation parameters
    dt = 100  # 100ms timestep (10 Hz)
    brake_time = 2000  # V003 brakes at T=2000ms
    brake_decel = -5.0  # 5 m/s^2 deceleration

    stats = {'received': 0, 'lost': 0, 'total_latency': 0}

    print(f"\n{'T(ms)':<8} {'V003 Speed':>12} {'V002 Speed':>12} "
          f"{'Obs V003 Spd':>14} {'Obs V003 Age':>14} {'Status':>10}")
    print("-" * 75)

    for t in range(0, 4000, dt):
        # Update V003 dynamics (emergency brake at T=2000ms)
        if t >= brake_time:
            v003_speed = max(0, 15.0 + brake_decel * (t - brake_time) / 1000.0)
        else:
            v003_speed = 15.0

        # V002 maintains speed (doesn't see V003's brake yet)
        v002_speed = 15.0

        # Update positions (simplified)
        v003_x += v003_speed * (dt / 1000.0)
        v002_x += v002_speed * (dt / 1000.0)

        # Create and transmit messages
        v003_msg = create_convoy_message('V003', v003_x, v003_speed,
                                          brake_decel if t >= brake_time else 0.0, t)
        v002_msg = create_convoy_message('V002', v002_x, v002_speed, 0.0, t)

        v001_pos = (0.0, 0.0)
        v002_pos = (v002_x, 0.0)
        v003_pos = (v003_x, 0.0)

        result_v003 = emulator.transmit(v003_msg, v003_pos, v001_pos, t)
        result_v002 = emulator.transmit(v002_msg, v002_pos, v001_pos, t)

        if result_v003:
            stats['received'] += 1
            stats['total_latency'] += result_v003.age_ms
        else:
            stats['lost'] += 1

        # Get observation
        obs = emulator.get_observation(ego_speed=15.0, current_time_ms=t)
        obs_v003_speed = obs['v003_speed']
        obs_v003_age = obs['v003_age_ms']
        obs_v003_valid = obs['v003_valid']

        # Print every 500ms
        if t % 500 == 0:
            status = "VALID" if obs_v003_valid else "STALE"
            print(f"{t:<8} {v003_speed:>12.1f} {v002_speed:>12.1f} "
                  f"{obs_v003_speed:>14.1f} {obs_v003_age:>14} {status:>10}")

    # Print statistics
    total = stats['received'] + stats['lost']
    loss_rate = stats['lost'] / total * 100 if total > 0 else 0
    avg_latency = stats['total_latency'] / stats['received'] if stats['received'] > 0 else 0

    print(f"\n--- Statistics ---")
    print(f"Total transmissions: {total}")
    print(f"Received: {stats['received']} ({100-loss_rate:.1f}%)")
    print(f"Lost: {stats['lost']} ({loss_rate:.1f}%)")
    print(f"Average latency: {avg_latency:.1f}ms")


def demo_domain_randomization():
    """Demonstrate domain randomization across episodes."""
    print("\n" + "="*60)
    print("DEMO 3: Domain Randomization Across Episodes")
    print("="*60)

    emulator = ESPNOWEmulator(domain_randomization=True, seed=42)

    print("\nDomain randomization enabled: Parameters vary per episode")
    print("This trains RL agents to be robust to communication variations.\n")

    print(f"{'Episode':<10} {'Latency Base':>15} {'Loss Base':>12} "
          f"{'Jitter Std':>12} {'GPS Noise':>12}")
    print("-" * 65)

    for episode in range(1, 11):
        # Reset triggers new randomization
        emulator.reset()

        print(f"{episode:<10} {emulator.episode_latency_base:>15.1f}ms "
              f"{emulator.episode_loss_base:>12.3f} "
              f"{emulator.episode_jitter_std:>12.1f}ms "
              f"{emulator.episode_gps_noise_std:>12.1f}m")

    print("\nNote: Each episode has different parameters within configured ranges:")
    print(f"  - Latency base: {emulator.params['domain_randomization']['latency_range_ms']} ms")
    print(f"  - Loss rate: {emulator.params['domain_randomization']['loss_rate_range']}")
    print(f"  - Jitter std: {emulator.params['domain_randomization']['jitter_std_range_ms']} ms")
    print(f"  - GPS noise: {emulator.params['domain_randomization']['gps_noise_range_m']} m")


def demo_reproducibility():
    """Demonstrate reproducible results with seeding."""
    print("\n" + "="*60)
    print("DEMO 4: Reproducibility with Seeding")
    print("="*60)

    print("\nSame seed produces identical results (critical for debugging):\n")

    for run in range(1, 3):
        print(f"--- Run {run} (seed=42) ---")
        emulator = ESPNOWEmulator(domain_randomization=True, seed=42)

        # Perform some transmissions
        msg = create_convoy_message('V002', 30.0, 15.0, 0.0, 1000)

        latencies = []
        for t in range(0, 500, 100):
            result = emulator.transmit(msg, (30.0, 0.0), (0.0, 0.0), t)
            if result:
                latencies.append(result.age_ms)

        print(f"  Episode params: latency_base={emulator.episode_latency_base:.1f}ms")
        print(f"  Latencies: {latencies}")
        print()


def demo_staleness():
    """Demonstrate message staleness handling."""
    print("\n" + "="*60)
    print("DEMO 5: Message Staleness Detection")
    print("="*60)

    emulator = ESPNOWEmulator(domain_randomization=False, seed=42)

    print("\nMessages older than 500ms are marked as invalid (stale)")
    print("This prevents the agent from acting on outdated information.\n")

    # Transmit a message at T=0
    msg = create_convoy_message('V002', 25.0, 15.0, -3.0, 0)
    emulator.transmit(msg, (25.0, 0.0), (0.0, 0.0), 0)

    # Check at various times
    check_times = [0, 100, 300, 500, 600, 800, 1000]

    print(f"{'Time (ms)':<12} {'V002 Age (ms)':>15} {'Valid':>10}")
    print("-" * 40)

    for t in check_times:
        obs = emulator.get_observation(ego_speed=15.0, current_time_ms=t)
        age = obs['v002_age_ms']
        valid = obs['v002_valid']
        status = "YES" if valid else "NO (stale)"
        print(f"{t:<12} {age:>15} {status:>10}")

    print("\nNote: After 500ms, the message is marked invalid.")
    print("The agent should request fresh data or use fallback behavior.")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*60)
    print("# ESP-NOW EMULATOR DEMONSTRATION")
    print("# RoadSense V2V Safety Project")
    print("#"*60)

    demo_basic_transmission()
    demo_convoy_simulation()
    demo_domain_randomization()
    demo_reproducibility()
    demo_staleness()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nFor more information, see the module docstring:")
    print("  python -c \"from espnow_emulator import ESPNOWEmulator; help(ESPNOWEmulator)\"")
    print()


if __name__ == "__main__":
    main()
