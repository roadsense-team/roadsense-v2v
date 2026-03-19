"""
Gymnasium environments for RoadSense V2V.

Registers:
- RoadSense-Convoy-v0: 3-vehicle convoy following with ESP-NOW emulation
- RoadSense-ReplayConvoy-v0: Real-data replay for sim-to-real fine-tuning
"""

import os
from gymnasium.envs.registration import register

_SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "..", "scenarios")
_DEFAULT_SCENARIO = os.path.join(_SCENARIOS_DIR, "base", "scenario.sumocfg")
_RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "recordings")

register(
    id="RoadSense-Convoy-v0",
    entry_point="ml.envs.convoy_env:ConvoyEnv",
    kwargs={
        "sumo_cfg": _DEFAULT_SCENARIO,
    },
)

register(
    id="RoadSense-ReplayConvoy-v0",
    entry_point="ml.envs.replay_convoy_env:ReplayConvoyEnv",
    kwargs={
        "recordings_dir": _RECORDINGS_DIR,
    },
)

from .convoy_env import ConvoyEnv
from .scenario_manager import ScenarioManager
from .sumo_connection import SUMOConnection, VehicleState
from .action_applicator import ActionApplicator
from .reward_calculator import RewardCalculator
from .observation_builder import ObservationBuilder
from .hazard_injector import HazardInjector
from .replay_convoy_env import ReplayConvoyEnv
from .ego_kinematics import EgoKinematics

__all__ = [
    "ConvoyEnv",
    "ReplayConvoyEnv",
    "EgoKinematics",
    "ScenarioManager",
    "SUMOConnection",
    "VehicleState",
    "ActionApplicator",
    "RewardCalculator",
    "ObservationBuilder",
    "HazardInjector",
]
