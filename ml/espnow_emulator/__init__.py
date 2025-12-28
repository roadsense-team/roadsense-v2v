"""
ESP-NOW Communication Emulator

This module provides a Python emulator for ESP-NOW communication characteristics
based on measurements from real ESP32 hardware. It models latency, packet loss,
jitter, and optional sensor noise to enable Sim2Real transfer for RL training.

Key components:
- V2VMessage: Dataclass representing vehicle-to-vehicle messages
- ReceivedMessage: Dataclass for messages with communication effects applied
- ESPNOWEmulator: Main emulator class that applies realistic communication effects

Usage:
    from espnow_emulator import ESPNOWEmulator, V2VMessage

    # Initialize with measured parameters
    emulator = ESPNOWEmulator(params_file='emulator_params.json')

    # Create message
    msg = V2VMessage(
        vehicle_id='V002',
        lat=32.085,
        lon=34.781,
        speed=15.0,
        heading=90.0,
        accel_x=0.5,
        accel_y=0.0,
        accel_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.1,
        timestamp_ms=1000
    )

    # Transmit with communication effects
    received = emulator.transmit(
        sender_msg=msg,
        sender_pos=(10, 5),
        receiver_pos=(0, 0),
        current_time_ms=1000
    )

    if received:
        print(f"Message received with {received.age_ms}ms latency")
    else:
        print("Packet lost")
"""

from .espnow_emulator import (
    V2VMessage,
    ReceivedMessage,
    ESPNOWEmulator
)

__version__ = "1.0.0"
__all__ = [
    "V2VMessage",
    "ReceivedMessage",
    "ESPNOWEmulator"
]
