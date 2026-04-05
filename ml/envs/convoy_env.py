"""
ConvoyEnv Gymnasium environment.
"""
from collections import deque
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

    DEFAULT_MAX_STEPS = 500
    COLLISION_DIST = 5.0
    MAX_STARTUP_STEPS = 100
    BRAKING_ACCEL_THRESHOLD = -2.5
    BRAKING_DECAY = 0.95  # per-step at 10Hz → half-life ~1.35s
    CF_OVERRIDE_GRACE_STEPS = 3

    # Single-frame ego observation bounds (6-dim).
    _EGO_LOW = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
    _EGO_HIGH = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    EGO_SINGLE_FRAME_DIM = 6

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
        hazard_target_strategy: str = HazardInjector.TARGET_STRATEGY_UNIFORM_FRONT_PEERS,
        hazard_fixed_vehicle_id: Optional[str] = None,
        hazard_fixed_rank_ahead: Optional[int] = None,
        render_mode: Optional[str] = None,
        gui: bool = False,
        ego_stack_frames: int = 1,
        cone_half_angle_deg: float = 45.0,
    ) -> None:
        super().__init__()

        if ego_stack_frames < 1:
            raise ValueError(f"ego_stack_frames must be >= 1, got {ego_stack_frames}")
        self.ego_stack_frames = ego_stack_frames

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

        self.cone_half_angle_deg = cone_half_angle_deg
        self.obs_builder = ObservationBuilder()
        self.action_applicator = ActionApplicator()
        self.reward_calculator = RewardCalculator()
        self.hazard_injector = (
            HazardInjector(
                target_strategy=hazard_target_strategy,
                fixed_vehicle_id=hazard_fixed_vehicle_id,
                fixed_rank_ahead=hazard_fixed_rank_ahead,
            )
            if hazard_injection
            else None
        )

        self._step_count = 0
        self._sumo_started = False
        self._hazard_injection_step: Optional[int] = None
        self._cf_override_active = False
        self._hazard_source_braking_latched = False
        self._braking_received_decay = 0.0
        # Ego frame stacking history (Run 025)
        self._ego_history: deque = deque(maxlen=self.ego_stack_frames)

        ego_low = np.tile(self._EGO_LOW, self.ego_stack_frames)
        ego_high = np.tile(self._EGO_HIGH, self.ego_stack_frames)

        self.observation_space = spaces.Dict({
            "ego": spaces.Box(
                low=ego_low, high=ego_high, dtype=np.float32,
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

    def _extract_hazard_options(
        self,
        options: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if options is None:
            return None

        nested = options.get("hazard_options")
        if isinstance(nested, dict):
            return dict(nested)

        mapped: Dict[str, Any] = {}
        key_map = {
            "hazard_target_strategy": "target_strategy",
            "hazard_fixed_vehicle_id": "fixed_vehicle_id",
            "hazard_fixed_rank_ahead": "fixed_rank_ahead",
            "hazard_step": "hazard_step",
            "hazard_force": "force_hazard",
        }
        for source_key, dest_key in key_map.items():
            if source_key in options:
                mapped[dest_key] = options[source_key]

        return mapped if mapped else None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self.emulator.reset()

        self._step_count = 0
        self._hazard_injection_step = None
        self._cf_override_active = False
        self._hazard_source_braking_latched = False
        self._braking_received_decay = 0.0

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

        # Warmup: let SUMO's CF model stabilize the convoy before RL takes over.
        # Prevents spawn-time collisions when vehicles start too close together.
        WARMUP_STEPS = 30
        WARMUP_EXTENSION_MAX = 30
        WARMUP_SAFE_MARGIN = 3.0
        for _ in range(WARMUP_STEPS):
            self.sumo.step()
            if not self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID):
                self.sumo.stop()
                self._sumo_started = False
                raise RuntimeError("V001 left simulation during warmup.")
            # Feed emulator so it has message history when RL starts
            ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
            current_time_ms = int(self.sumo.get_simulation_time() * 1000)
            self._step_espnow(
                ego_state,
                current_time_ms,
                update_braking_latch=False,
            )

        # Extend warmup if ground-truth distance is too close
        ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        for _ in range(WARMUP_EXTENSION_MAX):
            gt_dist = self._calculate_ground_truth_distance(ego_state)
            if gt_dist > self.COLLISION_DIST + WARMUP_SAFE_MARGIN:
                break
            self.sumo.step()
            if not self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID):
                self.sumo.stop()
                self._sumo_started = False
                raise RuntimeError("V001 left simulation during warmup extension.")
            ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
            current_time_ms = int(self.sumo.get_simulation_time() * 1000)
            self._step_espnow(
                ego_state,
                current_time_ms,
                update_braking_latch=False,
            )

        # Warmup is pre-episode stabilization. Do not carry warmup braking
        # events into the RL episode decay.
        self._braking_received_decay = 0.0

        if self.hazard_injector is not None:
            hazard_options = self._extract_hazard_options(options)
            if seed is not None:
                self.hazard_injector.seed(seed)
            self.hazard_injector.reset(options=hazard_options)

        ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_time_ms = int(self.sumo.get_simulation_time() * 1000)

        observation, _, _, _ = self._step_espnow(ego_state, current_time_ms)

        # Initialize ego history with N copies of the first ego vector (Run 025)
        self._ego_history.clear()
        for _ in range(self.ego_stack_frames):
            self._ego_history.append(observation["ego"].copy())
        observation["ego"] = self._stack_ego()

        info = {
            "step": 0,
            "simulation_time": self.sumo.get_simulation_time(),
            "scenario_id": scenario_id,
            "hazard_step": (
                self.hazard_injector.hazard_step if self.hazard_injector else None
            ),
            "hazard_target_strategy": (
                self.hazard_injector.target_strategy if self.hazard_injector else None
            ),
        }

        return observation, info

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Pre-action guard: V001 may have left at end of previous step.
        # Must check BEFORE any TraCI calls (apply/inject) to avoid crash.
        if not self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID):
            empty_obs = {
                "ego": np.zeros(self.ego_stack_frames * self.EGO_SINGLE_FRAME_DIM, dtype=np.float32),
                "peers": np.zeros((self.MAX_PEERS, 6), dtype=np.float32),
                "peer_mask": np.zeros(self.MAX_PEERS, dtype=np.float32),
            }
            return empty_obs, 0.0, False, True, {
                "step": self._step_count,
                "simulation_time": self.sumo.get_simulation_time(),
                "distance": 1000.0,
                "truncated_reason": "ego_route_ended",
            }

        action_value = self._parse_action_value(action)

        # Activate CF override after grace period following hazard injection.
        # This forces the RL model to be the sole source of deceleration —
        # the SUMO CF model cannot brake for free during hazard events.
        if (
            self._hazard_injection_step is not None
            and not self._cf_override_active
            and self._step_count >= self._hazard_injection_step + self.CF_OVERRIDE_GRACE_STEPS
        ):
            self._cf_override_active = True

        actual_decel = self.action_applicator.apply(
            self.sumo, action_value, cf_override=self._cf_override_active
        )

        hazard_injected = False
        if self.hazard_injector is not None:
            hazard_injected = self.hazard_injector.maybe_inject(
                step=self._step_count,
                sumo=self.sumo,
            )
            if hazard_injected and self._hazard_injection_step is None:
                self._hazard_injection_step = self._step_count
            self.hazard_injector.maintain_hazard(
                step=self._step_count, sumo=self.sumo,
            )

        self.sumo.step()
        self._step_count += 1

        # Post-step guard: V001 may have left during this sumo.step().
        if not self.sumo.is_vehicle_active(self.EGO_VEHICLE_ID):
            empty_obs = {
                "ego": np.zeros(self.ego_stack_frames * self.EGO_SINGLE_FRAME_DIM, dtype=np.float32),
                "peers": np.zeros((self.MAX_PEERS, 6), dtype=np.float32),
                "peer_mask": np.zeros(self.MAX_PEERS, dtype=np.float32),
            }
            return empty_obs, 0.0, False, True, {
                "step": self._step_count,
                "simulation_time": self.sumo.get_simulation_time(),
                "distance": 1000.0,
                "truncated_reason": "ego_route_ended",
            }

        ego_state = self.sumo.get_vehicle_state(self.EGO_VEHICLE_ID)
        current_time_ms = int(self.sumo.get_simulation_time() * 1000)

        (
            observation,
            peer_states,
            received_map,
            any_braking_peer_received,
        ) = self._step_espnow(ego_state, current_time_ms)

        # Ego frame stacking (Run 025)
        self._ego_history.append(observation["ego"].copy())
        observation["ego"] = self._stack_ego()

        distance, closing_rate = self._calculate_min_distance_and_closing_rate(
            ego_state, peer_states
        )

        # Use SUMO ground truth for collision/termination (not GPS-noisy mesh)
        gt_distance = self._calculate_ground_truth_distance(ego_state)

        terminated = False
        truncated = False

        if gt_distance < self.COLLISION_DIST:
            terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        hazard_source_id = None
        hazard_step = None
        hazard_source_rank_ahead = None
        hazard_injection_attempted = False
        hazard_injection_failed_reason = None
        if self.hazard_injector is not None:
            hazard_source_id = self.hazard_injector.hazard_target
            hazard_step = self.hazard_injector.hazard_step
            hazard_source_rank_ahead = self.hazard_injector.hazard_source_rank_ahead
            hazard_injection_attempted = self.hazard_injector.hazard_injection_attempted
            hazard_injection_failed_reason = self.hazard_injector.hazard_injection_failed_reason

        # Check whether the *injected* hazard source's braking message reached
        # ego this step.  Gated to the specific target — never true during
        # normal driving or if no hazard is active.
        hazard_source_braking_received = self._hazard_source_braking_latched
        if (
            not self._hazard_source_braking_latched
            and self.hazard_injector is not None
            and self.hazard_injector._hazard_injected
            and hazard_source_id is not None
        ):
            for received in received_map.values():
                msg = received.message
                source = msg.source_id or msg.vehicle_id
                if source == hazard_source_id:
                    if float(msg.accel_x) <= self.BRAKING_ACCEL_THRESHOLD:
                        self._hazard_source_braking_latched = True
                        hazard_source_braking_received = True
                    break
        elif self._hazard_source_braking_latched:
            hazard_source_braking_received = True

        if (
            self.hazard_injector is not None
            and not self.hazard_injector._hazard_injected
        ):
            self._hazard_source_braking_latched = False
            hazard_source_braking_received = False

        ego_speed = ego_state.speed

        # Run 020: reward uses the SAME decaying braking signal that
        # drives ego[4].  Reward terms scale by the decay value so the
        # shaping fades smoothly — no obs/reward mismatch.
        reward, reward_info = self.reward_calculator.calculate(
            distance=distance,
            action_value=action_value,
            deceleration=actual_decel,
            closing_rate=closing_rate,
            any_braking_peer=any_braking_peer_received,
            braking_received=self._braking_received_decay,
            ego_speed=ego_speed,
        )

        mesh_received_source_ids = sorted(received_map.keys())

        info = {
            "step": self._step_count,
            "simulation_time": self.sumo.get_simulation_time(),
            "distance": distance,
            "hazard_injected": hazard_injected,
            "hazard_source_id": hazard_source_id,
            "hazard_step": hazard_step,
            "hazard_source_rank_ahead": hazard_source_rank_ahead,
            "hazard_injection_attempted": hazard_injection_attempted,
            "hazard_injection_failed_reason": hazard_injection_failed_reason,
            "hazard_desired_decel": (
                self.hazard_injector.desired_decel
                if self.hazard_injector is not None else None
            ),
            "hazard_target_speed": (
                self.hazard_injector.hazard_target_speed
                if self.hazard_injector is not None else None
            ),
            "hazard_resolved": (
                self.hazard_injector.hazard_resolved
                if self.hazard_injector is not None else False
            ),
            "hazard_trigger_mode": (
                self.hazard_injector.trigger_mode
                if self.hazard_injector is not None else None
            ),
            "hazard_trigger_result": (
                self.hazard_injector.trigger_result
                if self.hazard_injector is not None else None
            ),
            "hazard_onset_gap_bucket": (
                self.hazard_injector.onset_gap_bucket
                if self.hazard_injector is not None else None
            ),
            "hazard_onset_gap_m": (
                self.hazard_injector.onset_gap_m
                if self.hazard_injector is not None else None
            ),
            "hazard_onset_closing_speed_mps": (
                self.hazard_injector.onset_closing_speed_mps
                if self.hazard_injector is not None else None
            ),
            "hazard_onset_peer_count": (
                self.hazard_injector.onset_peer_count
                if self.hazard_injector is not None else None
            ),
            "hazard_onset_trigger_step": (
                self.hazard_injector.onset_trigger_step
                if self.hazard_injector is not None else None
            ),
            "hazard_onset_desired_rank_ahead": (
                self.hazard_injector.onset_desired_rank_ahead
                if self.hazard_injector is not None else None
            ),
            "mesh_received_source_ids": mesh_received_source_ids,
            "mesh_any_braking_peer_received": any_braking_peer_received,
            "braking_received_decay": self._braking_received_decay,
            "hazard_source_braking_latched": self._hazard_source_braking_latched,
            "ego_speed": ego_speed,
            **reward_info,
        }

        return observation, reward, terminated, truncated, info

    def _step_espnow(
        self,
        ego_state: VehicleState,
        current_time_ms: int,
        update_braking_latch: bool = True,
    ) -> Tuple[
        Dict[str, np.ndarray],
        List[VehicleState],
        Dict[str, Any],
        bool,
    ]:
        ego_pos = (ego_state.x, ego_state.y)
        vehicle_states = {
            vehicle_id: self.sumo.get_vehicle_state(vehicle_id)
            for vehicle_id in traci.vehicle.getIDList()
        }

        received_map = self.emulator.simulate_mesh_step(
            vehicle_states=vehicle_states,
            ego_id=self.EGO_VEHICLE_ID,
            current_time_ms=current_time_ms,
            cone_half_angle_deg=self.cone_half_angle_deg,
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

        mesh_visible_peer_states: List[VehicleState] = []

        for received in received_map.values():
            msg = received.message
            age_ms = float(received.age_ms)
            valid = age_ms < staleness_threshold
            source_id = msg.source_id or msg.vehicle_id
            peer_observations.append({
                "x": msg.lon * meters_per_deg,
                "y": msg.lat * meters_per_deg,
                "speed": msg.speed,
                "heading": msg.heading,
                "accel": msg.accel_x,
                "age_ms": age_ms,
                "valid": valid,
            })
            if not valid:
                continue

            mesh_visible_peer_states.append(
                VehicleState(
                    vehicle_id=source_id,
                    x=msg.lon * meters_per_deg,
                    y=msg.lat * meters_per_deg,
                    speed=max(0.0, float(msg.speed)),
                    acceleration=float(msg.accel_x),
                    heading=float(msg.heading),
                    lane_position=0.0,
                )
            )

        cone_filtered_peers = self.obs_builder.filter_observable_peers(
            ego_heading_deg=ego_state.heading,
            ego_pos=ego_pos,
            peer_observations=peer_observations,
            half_angle_deg=self.cone_half_angle_deg,
        )
        any_braking_peer_received = any(
            float(peer["accel"]) <= self.BRAKING_ACCEL_THRESHOLD
            for peer in cone_filtered_peers
        )

        if update_braking_latch:
            if any_braking_peer_received:
                self._braking_received_decay = 1.0
            else:
                self._braking_received_decay *= self.BRAKING_DECAY

        observation = self.obs_builder.build(
            ego_state=ego_state,
            peer_observations=peer_observations,
            ego_pos=ego_pos,
            braking_received=self._braking_received_decay,
            half_angle_deg=self.cone_half_angle_deg,
        )

        return (
            observation,
            mesh_visible_peer_states,
            dict(received_map),
            any_braking_peer_received,
        )

    def _stack_ego(self) -> np.ndarray:
        """Flatten ego history deque into a single vector.

        Order: [ego_t, ego_{t-1}, ..., ego_{t-N+1}] — most recent first.
        When ego_stack_frames=1, returns the single (6,) vector unchanged.
        """
        return np.concatenate(list(reversed(self._ego_history)), dtype=np.float32)

    def _calculate_min_distance_and_closing_rate(
        self,
        ego_state: VehicleState,
        peer_states: Iterable[VehicleState],
    ) -> Tuple[float, float]:
        if not peer_states:
            return 1000.0, 0.0

        min_dist = float("inf")
        nearest_peer = None
        for peer in peer_states:
            dx = peer.x - ego_state.x
            dy = peer.y - ego_state.y
            dist = (dx**2 + dy**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_peer = peer

        closing_rate = 0.0
        if nearest_peer is not None:
            closing_rate = ego_state.speed - nearest_peer.speed

        return min_dist, closing_rate

    def _calculate_ground_truth_distance(self, ego_state: VehicleState) -> float:
        """
        Calculate min distance to any active peer using SUMO ground truth.

        Used ONLY for collision/termination check — avoids false collisions
        from GPS-noisy emulated positions.
        """
        active_ids = self.sumo.get_active_vehicle_ids()
        min_dist = float("inf")
        for vid in active_ids:
            if vid == self.EGO_VEHICLE_ID:
                continue
            peer_state = self.sumo.get_vehicle_state(vid)
            dx = peer_state.x - ego_state.x
            dy = peer_state.y - ego_state.y
            dist = (dx**2 + dy**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
        return min_dist if min_dist < float("inf") else 1000.0

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
