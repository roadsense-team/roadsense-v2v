"""
ConvoyEnv Gymnasium environment.
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.action_applicator import ActionApplicator
from envs.hazard_injector import HazardInjector
from envs.observation_builder import ObservationBuilder
from envs.reward_calculator import RewardCalculator
from envs.sumo_connection import SUMOConnection, VehicleState
from espnow_emulator.espnow_emulator import ESPNOWEmulator


class ConvoyEnv(gym.Env):
    """
    RoadSense V2V Convoy Environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    EGO_VEHICLE_ID = "V001"
    LEAD_VEHICLE_IDS = ["V002", "V003"]

    DEFAULT_MAX_STEPS = 1000
    COLLISION_DIST = 5.0

    def __init__(
        self,
        sumo_cfg: str,
        emulator: Optional[ESPNOWEmulator] = None,
        emulator_params_path: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        hazard_injection: bool = True,
        render_mode: Optional[str] = None,
        gui: bool = False,
    ) -> None:
        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.hazard_injection_enabled = hazard_injection
        self.render_mode = render_mode
        self.gui = gui or (render_mode == "human")

        self.sumo = SUMOConnection(sumo_cfg, gui=self.gui)

        if emulator is not None:
            self.emulator = emulator
        elif emulator_params_path is not None:
            self.emulator = ESPNOWEmulator(params_file=emulator_params_path)
        else:
            self.emulator = ESPNOWEmulator()

        self.obs_builder = ObservationBuilder()
        self.action_applicator = ActionApplicator()
        self.reward_calculator = RewardCalculator()
        self.hazard_injector = HazardInjector() if hazard_injection else None

        self._step_count = 0
        self._sumo_started = False

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if hasattr(self.emulator, "clear"):
            self.emulator.clear()
        else:
            self.emulator.reset(seed=seed)

        self._step_count = 0

        if self._sumo_started:
            self.sumo.stop()
        self.sumo.start()
        self._sumo_started = True

        if self.hazard_injector is not None:
            if seed is not None:
                self.hazard_injector.seed(seed)
            else:
                self.hazard_injector.reset()

        ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_time_ms = int(self.sumo.get_simulation_time() * 1000)

        observation = self.obs_builder.build(
            ego_state=ego_state,
            emulator_obs=self.emulator.get_observation(
                ego_speed=ego_state.speed,
                current_time_ms=current_time_ms,
            ),
        )

        info = {
            "step": 0,
            "simulation_time": self.sumo.get_simulation_time(),
        }

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        actual_decel = self.action_applicator.apply(self.sumo, action)

        hazard_injected = False
        if self.hazard_injector is not None:
            hazard_injected = self.hazard_injector.maybe_inject(
                step=self._step_count,
                sumo=self.sumo,
            )

        self.sumo.step()
        self._step_count += 1

        ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_time_ms = int(self.sumo.get_simulation_time() * 1000)

        peer_states = []
        for vid in self.LEAD_VEHICLE_IDS:
            if self.sumo.is_vehicle_active(vid):
                peer_states.append(self.sumo.get_vehicle_state(vid))

        self._transmit_vehicle_states(ego_state, peer_states, current_time_ms)

        distance = self._calculate_min_distance(ego_state, peer_states)

        terminated = False
        truncated = False

        if distance < self.COLLISION_DIST:
            terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        if not self._all_vehicles_active():
            truncated = True

        observation = self.obs_builder.build(
            ego_state=ego_state,
            emulator_obs=self.emulator.get_observation(
                ego_speed=ego_state.speed,
                current_time_ms=current_time_ms,
            ),
        )

        reward, reward_info = self.reward_calculator.calculate(
            distance=distance,
            action=action,
            deceleration=actual_decel,
        )

        info = {
            "step": self._step_count,
            "simulation_time": self.sumo.get_simulation_time(),
            "distance": distance,
            "hazard_injected": hazard_injected,
            **reward_info,
        }

        return observation, reward, terminated, truncated, info

    def _transmit_vehicle_states(
        self,
        ego_state: VehicleState,
        peer_states: Iterable[VehicleState],
        current_time_ms: int,
    ) -> None:
        for state in peer_states:
            msg = state.to_v2v_message(timestamp_ms=current_time_ms)
            self.emulator.transmit(
                sender_msg=msg,
                sender_pos=(state.x, state.y),
                receiver_pos=(ego_state.x, ego_state.y),
                current_time_ms=current_time_ms,
            )

    def _calculate_min_distance(
        self,
        ego_state: VehicleState,
        peer_states: Iterable[VehicleState],
    ) -> float:
        if not peer_states:
            return 1000.0

        min_dist = float("inf")
        for peer in peer_states:
            dx = peer.x - ego_state.x
            dy = peer.y - ego_state.y
            dist = (dx**2 + dy**2) ** 0.5
            min_dist = min(min_dist, dist)

        return min_dist

    def _all_vehicles_active(self) -> bool:
        if not self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID):
            return False
        for vid in self.LEAD_VEHICLE_IDS:
            if not self.sumo.is_vehicle_active(vid):
                return False
        return True

    def render(self) -> None:
        pass

    def close(self) -> None:
        if self._sumo_started:
            self.sumo.stop()
            self._sumo_started = False
