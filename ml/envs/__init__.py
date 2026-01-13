"""
Gymnasium environments for RoadSense V2V.

Registers:
- RoadSense-Convoy-v0: 3-vehicle convoy following with ESP-NOW emulation
"""

import os
from gymnasium.envs.registration import register

_SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "..", "scenarios")
_DEFAULT_SCENARIO = os.path.join(_SCENARIOS_DIR, "base", "scenario.sumocfg")

register(
    id="RoadSense-Convoy-v0",
    entry_point="envs.convoy_env:ConvoyEnv",
    kwargs={
        "sumo_cfg": _DEFAULT_SCENARIO,
    },
)

from envs.convoy_env import ConvoyEnv
from envs.sumo_connection import SUMOConnection, VehicleState
from envs.action_applicator import ActionApplicator
from envs.reward_calculator import RewardCalculator
from envs.observation_builder import ObservationBuilder
from envs.hazard_injector import HazardInjector

__all__ = [
    "ConvoyEnv",
    "SUMOConnection",
    "VehicleState",
    "ActionApplicator",
    "RewardCalculator",
    "ObservationBuilder",
    "HazardInjector",
]
