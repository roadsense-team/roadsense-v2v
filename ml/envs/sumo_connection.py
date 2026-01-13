"""
SUMO/TraCI connection utilities.
"""

from dataclasses import dataclass
from espnow_emulator.espnow_emulator import V2VMessage, ESPNOWEmulator


import traci
import os

@dataclass(frozen=True)
class VehicleState:
    """Vehicle state extracted from SUMO via TraCI."""
    vehicle_id: str
    x: float              # SUMO x coordinate (meters)
    y: float              # SUMO y coordinate (meters)
    speed: float          # Speed in m/s
    acceleration: float   # Acceleration in m/s²
    heading: float        # Heading in degrees (SUMO angle)
    lane_position: float  # Position along lane

    def __post_init__(self):
        # Clamping speed to 0.0 if negative
        if self.speed < 0:
            object.__setattr__(self, "speed", 0.0)

    def to_v2v_message(self, timestamp_ms: int) -> V2VMessage:
        """
        Convert to V2VMessage for emulator.

        Maps SUMO state to V2V message fields:
        - x -> lon (scaled)
        - y -> lat (scaled)
        - acceleration -> accel_x (longitudinal)
        - heading -> heading
        - speed -> speed
        """
        meters_per_deg = ESPNOWEmulator.METERS_PER_DEG_LAT
        
        return V2VMessage(
            vehicle_id=self.vehicle_id,
            lat=self.y / meters_per_deg,
            lon=self.x / meters_per_deg,
            speed=self.speed,
            heading=self.heading,
            accel_x=self.acceleration,
            accel_y=0.0,
            accel_z=9.81,  # Standard gravity
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=timestamp_ms
        )


class SUMOConnection:
    """
    Manages SUMO simulation via TraCI.
    """
    def __init__(self, sumo_cfg: str, gui: bool = False, sumo_binary: str = None, port: int = None):
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.sumo_binary = sumo_binary
        self.port = port

    def start(self):
        """Start SUMO simulation."""
        binary = self.sumo_binary
        if binary is None:
            binary = "sumo-gui" if self.gui else "sumo"
        
        cmd = [binary, "-c", self.sumo_cfg, "--step-length", "0.1"]
        
        if self.port is not None:
            traci.start(cmd, port=self.port)
        else:
            traci.start(cmd)

    def get_vehicle_state(self, vehicle_id: str) -> VehicleState:
        """Get current state of a vehicle."""
        x, y = traci.vehicle.getPosition(vehicle_id)
        speed = traci.vehicle.getSpeed(vehicle_id)
        accel = traci.vehicle.getAcceleration(vehicle_id)
        angle = traci.vehicle.getAngle(vehicle_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        
        return VehicleState(
            vehicle_id=vehicle_id,
            x=x,
            y=y,
            speed=speed,
            acceleration=accel,
            heading=angle,
            lane_position=lane_pos
        )

    def set_vehicle_speed(self, vehicle_id: str, speed: float) -> None:
        """Set target speed for a vehicle."""
        if speed < 0:
            speed = 0.0
        traci.vehicle.setSpeed(vehicle_id, speed)

    def stop(self) -> None:
        """Stop SUMO simulation."""
        traci.close()

    def step(self) -> None:
        """Advance simulation by one timestep."""
        traci.simulationStep()

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return traci.simulation.getTime()

    def is_vehicle_active(self, vehicle_id: str) -> bool:
        """Check if vehicle is still in simulation."""
        return vehicle_id in traci.vehicle.getIDList()
