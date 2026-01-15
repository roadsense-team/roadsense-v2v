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
    entry_point="ml.envs.convoy_env:ConvoyEnv",
    kwargs={
        "sumo_cfg": _DEFAULT_SCENARIO,
    },
)

from .convoy_env import ConvoyEnv
from .scenario_manager import ScenarioManager
from .sumo_connection import SUMOConnection, VehicleState
from .action_applicator import ActionApplicator
from .reward_calculator import RewardCalculator
from .observation_builder import ObservationBuilder
from .hazard_injector import HazardInjector

__all__ = [
    "ConvoyEnv",
    "ScenarioManager",
    "SUMOConnection",
    "VehicleState",
    "ActionApplicator",
    "RewardCalculator",
    "ObservationBuilder",
    "HazardInjector",
]
