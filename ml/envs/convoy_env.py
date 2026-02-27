"""
ConvoyEnv Gymnasium environment.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

from .action_applicator import ActionApplicator
from .hazard_injector import HazardInjector
from .observation_builder import ObservationBuilder
from .reward_calculator import RewardCalculator
from .scenario_manager import ScenarioManager
from .sumo_connection import SUMOConnection, VehicleState
from ml.espnow_emulator.espnow_emulator import ESPNOWEmulator


class ConvoyEnv(gym.Env):
    """
    RoadSense V2V Convoy Environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    EGO_VEHICLE_ID = "V001"
    MAX_PEERS = ObservationBuilder.MAX_PEERS

    DEFAULT_MAX_STEPS = 1000
    COLLISION_DIST = 5.0
    MAX_STARTUP_STEPS = 100

    def __init__(
        self,
        sumo_cfg: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        scenario_mode: str = "train",
        scenario_seed: Optional[int] = None,
        emulator: Optional[ESPNOWEmulator] = None,
        emulator_params_path: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        hazard_injection: bool = True,
        render_mode: Optional[str] = None,
        gui: bool = False,
    ) -> None:
        super().__init__()

        if sumo_cfg is None and dataset_dir is None:
            raise ValueError("Must provide either sumo_cfg or dataset_dir")
        if sumo_cfg is not None and dataset_dir is not None:
            raise ValueError("Cannot provide both sumo_cfg and dataset_dir")

        if dataset_dir is not None:
            self.scenario_manager = ScenarioManager(
                dataset_dir=dataset_dir,
                seed=scenario_seed,
                mode=scenario_mode,
            )
            self._initial_sumo_cfg = None
        else:
            self.scenario_manager = None
            self._initial_sumo_cfg = sumo_cfg

        self._dataset_dir = dataset_dir
        self._scenario_seed = scenario_seed

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.hazard_injection_enabled = hazard_injection
        self.render_mode = render_mode
        self.gui = gui or (render_mode == "human")

        initial_cfg = sumo_cfg or ""
        self.sumo = SUMOConnection(initial_cfg, gui=self.gui)

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

        self.observation_space = spaces.Dict({
            "ego": spaces.Box(
                low=np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "peers": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.MAX_PEERS, 6),
                dtype=np.float32,
            ),
            "peer_mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.MAX_PEERS,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _parse_action_value(self, action: Any) -> float:
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Expected scalar action, got shape {action.shape}")
            return float(action.item())

        if hasattr(action, "item"):
            return float(action.item())

        if isinstance(action, (list, tuple)):
            if len(action) != 1:
                raise ValueError(f"Expected scalar action, got {len(action)} values")
            return float(action[0])

        return float(action)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self.emulator.clear()

        self._step_count = 0

        scenario_id = None
        if self.scenario_manager is not None:
            new_cfg, scenario_id = self.scenario_manager.select_scenario()
            self.sumo.set_config(str(new_cfg))
        elif self._initial_sumo_cfg is not None and not self._sumo_started:
            self.sumo.set_config(self._initial_sumo_cfg)

        if self._sumo_started:
            self.sumo.stop()
            self._sumo_started = False
        try:
            self.sumo.start()
            self._sumo_started = True
        except Exception:
            self._sumo_started = False
            raise

        # Wait for ego vehicle to enter the simulation before querying state.
        for _ in range(self.MAX_STARTUP_STEPS):
            if self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID):
                break
            self.sumo.step()
        else:
            self.sumo.stop()
            self._sumo_started = False
            raise RuntimeError(
                f"Timeout: {self.EGO_VEHICLE_ID} failed to spawn."
            )

        if self.hazard_injector is not None:
            if seed is not None:
                self.hazard_injector.seed(seed)
            else:
                self.hazard_injector.reset()

        ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_time_ms = int(self.sumo.get_simulation_time() * 1000)

        observation, _ = self._step_espnow(ego_state, current_time_ms)

        info = {
            "step": 0,
            "simulation_time": self.sumo.get_simulation_time(),
            "scenario_id": scenario_id,
        }

        return observation, info

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action_value = self._parse_action_value(action)

        actual_decel = self.action_applicator.apply(self.sumo, action_value)

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

        observation, peer_states = self._step_espnow(ego_state, current_time_ms)
        distance = self._calculate_min_distance(ego_state, peer_states)

        terminated = False
        truncated = False

        if distance < self.COLLISION_DIST:
            terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        if not self._all_vehicles_active():
            truncated = True

        reward, reward_info = self.reward_calculator.calculate(
            distance=distance,
            action_value=action_value,
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

    def _step_espnow(
        self,
        ego_state: VehicleState,
        current_time_ms: int,
    ) -> Tuple[Dict[str, np.ndarray], List[VehicleState]]:
        ego_pos = (ego_state.x, ego_state.y)
        vehicle_states = {
            vehicle_id: self.sumo.get_vehicle_state(vehicle_id)
            for vehicle_id in traci.vehicle.getIDList()
        }
        peer_states = [
            state for vehicle_id, state in vehicle_states.items()
            if vehicle_id != self.EGO_VEHICLE_ID
        ]

        received_map = self.emulator.simulate_mesh_step(
            vehicle_states=vehicle_states,
            ego_id=self.EGO_VEHICLE_ID,
            current_time_ms=current_time_ms,
        )

        meters_per_deg = ESPNOWEmulator.METERS_PER_DEG_LAT
        peer_observations = []
        staleness_threshold = self.obs_builder.STALENESS_THRESHOLD
        params = getattr(self.emulator, "params", None)
        if isinstance(params, dict):
            staleness_threshold = params.get("observation", {}).get(
                "staleness_threshold_ms",
                staleness_threshold,
            )

        for received in received_map.values():
            msg = received.message
            age_ms = float(received.age_ms)
            valid = age_ms < staleness_threshold
            peer_observations.append({
                "x": msg.lon * meters_per_deg,
                "y": msg.lat * meters_per_deg,
                "speed": msg.speed,
                "heading": msg.heading,
                "accel": msg.accel_x,
                "age_ms": age_ms,
                "valid": valid,
            })

        observation = self.obs_builder.build(
            ego_state=ego_state,
            peer_observations=peer_observations,
            ego_pos=ego_pos,
        )

        return observation, peer_states

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
        return self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID)

    def render(self) -> None:
        pass

    def close(self) -> None:
        if self._sumo_started:
            self.sumo.stop()
            self._sumo_started = False

    def set_eval_mode(self) -> None:
        """
        Switch to evaluation mode (sequential scenario iteration).

        Raises:
            RuntimeError: If not using dataset-based configuration
        """
        if self.scenario_manager is None:
            raise RuntimeError("set_eval_mode() requires dataset_dir configuration")
        self.scenario_manager = ScenarioManager(
            dataset_dir=self._dataset_dir,
            seed=self._scenario_seed,
            mode="eval",
        )

    def set_train_mode(self) -> None:
        """
        Switch to training mode (random scenario selection).

        Raises:
            RuntimeError: If not using dataset-based configuration
        """
        if self.scenario_manager is None:
            raise RuntimeError("set_train_mode() requires dataset_dir configuration")
        self.scenario_manager = ScenarioManager(
            dataset_dir=self._dataset_dir,
            seed=self._scenario_seed,
            mode="train",
        )

    def get_scenario_count(self) -> Tuple[int, int]:
        """
        Get the number of scenarios in train and eval sets.

        Returns:
            Tuple of (train_count, eval_count)

        Raises:
            RuntimeError: If not using dataset-based configuration
        """
        if self.scenario_manager is None:
            raise RuntimeError("get_scenario_count() requires dataset_dir configuration")
        return (
            len(self.scenario_manager.manifest["train_scenarios"]),
            len(self.scenario_manager.manifest["eval_scenarios"]),
        )
