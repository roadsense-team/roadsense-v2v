"""
Real-data replay environment for sim-to-real fine-tuning.

Replays recorded peer trajectories while simulating ego with kinematics
(default) or using the recorded ego state directly (use_recorded_ego=True).

Observation and action spaces are IDENTICAL to ConvoyEnv, allowing
direct weight transfer from SUMO-trained models.

When use_recorded_ego=True the observation pipeline matches
validate_against_real_data.py exactly — ego position, speed, and
acceleration come from the TX recording each step, so the model is
fine-tuned on the same observation distribution it will see during
replay validation.  Optionally, reward geometry can instead be driven
by a shadow EgoKinematics instance so the reward reflects the
counterfactual trajectory implied by the policy's actions.
"""

import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ego_kinematics import EgoKinematics
from .observation_builder import ObservationBuilder
from .replay_trajectory import (
    AugmentConfig,
    PeerSnapshot,
    ReplayTrajectory,
    TrajectorySnapshot,
)
from .reward_calculator import RewardCalculator
from .sumo_connection import VehicleState

# Minimum ego speed (m/s) to consider a snapshot "moving".
_MIN_MOVING_SPEED = 0.5


class ReplayConvoyEnv(gym.Env):
    """
    Real-data replay environment for sim-to-real fine-tuning.

    Replays recorded peer trajectories while simulating ego with kinematics
    (default) or recorded ego state (use_recorded_ego=True).
    Observation and action spaces are IDENTICAL to ConvoyEnv.
    """

    metadata = {"render_modes": []}

    MAX_PEERS = 8
    COLLISION_DIST = 5.0
    BRAKING_ACCEL_THRESHOLD = -2.5
    BRAKING_DECAY = 0.95
    MAX_DECEL = 8.0
    DEFAULT_REWARD_CONFIG = {
        "early_reaction_threshold": 0.01,
        "ignoring_hazard_threshold": 0.15,
        "ignoring_require_danger_geometry": False,
        "ignoring_danger_distance": 20.0,
        "ignoring_danger_closing_rate": 0.5,
        "ignoring_use_any_braking_peer": True,
    }

    # Single-frame ego observation bounds (6-dim).
    _EGO_LOW = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
    _EGO_HIGH = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    EGO_SINGLE_FRAME_DIM = 6

    def __init__(
        self,
        recordings_dir: str,
        augment: bool = True,
        augment_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 500,
        seed: int = 42,
        forward_axis: str = "y",
        use_recorded_ego: bool = False,
        use_shadow_reward_geometry: bool = False,
        random_start: bool = False,
        reward_config: Optional[Dict[str, Any]] = None,
        ego_stack_frames: int = 1,
    ):
        super().__init__()

        if ego_stack_frames < 1:
            raise ValueError(f"ego_stack_frames must be >= 1, got {ego_stack_frames}")
        self.ego_stack_frames = ego_stack_frames

        # Observation space: ego bounds are tiled for stacking
        ego_low = np.tile(self._EGO_LOW, ego_stack_frames)
        ego_high = np.tile(self._EGO_HIGH, ego_stack_frames)

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

        self.obs_builder = ObservationBuilder()
        self.reward_config = dict(self.DEFAULT_REWARD_CONFIG)
        if reward_config:
            self.reward_config.update(reward_config)
        self.reward_calculator = RewardCalculator(**self.reward_config)
        self.max_steps = max_steps
        self.do_augment = augment
        self.forward_axis = forward_axis
        self.use_recorded_ego = use_recorded_ego
        self.use_shadow_reward_geometry = (
            use_shadow_reward_geometry and use_recorded_ego
        )
        self.random_start = random_start

        # Build augment config
        if augment_config:
            self.augment_config = AugmentConfig(**augment_config)
        else:
            self.augment_config = AugmentConfig(
                accel_scale=(0.6, 1.4),
                speed_scale=(0.8, 1.2),
                gps_noise_m=(0.5, 3.0),
                drop_rate=(0.0, 0.2),
                ego_offset_m=(-15.0, 15.0),
                inject_brake=True,
                inject_brake_accel=(-9.0, -4.0),
                inject_brake_duration_s=(0.5, 2.0),
            )

        # Load trajectories from recordings dir
        self.trajectories: List[ReplayTrajectory] = []
        rec_dir = Path(recordings_dir)
        if rec_dir.is_dir():
            for sub in sorted(rec_dir.iterdir()):
                if sub.is_dir():
                    tx = sub / "V001_tx.csv"
                    rx = sub / "V001_rx.csv"
                    if tx.exists() and rx.exists():
                        self.trajectories.append(
                            ReplayTrajectory(str(tx), str(rx), forward_axis)
                        )

        # Episode state
        self._rng = np.random.default_rng(seed)
        self._ego: Optional[EgoKinematics] = None
        self._snapshots: List[TrajectorySnapshot] = []
        self._step_idx = 0
        self._start_offset = 0
        self._episode_steps = 0
        self._braking_received_decay = 0.0
        self._last_action = 0.0
        self._last_decel = 0.0
        # Ego frame stacking history (Run 025)
        self._ego_history: deque = deque(maxlen=self.ego_stack_frames)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self.trajectories:
            raise RuntimeError("No recordings found in recordings_dir")

        # Pick random recording
        traj_idx = int(self._rng.integers(0, len(self.trajectories)))
        traj = self.trajectories[traj_idx]

        # Load or augment
        if self.do_augment:
            self._snapshots = traj.augment(self._rng, self.augment_config)
        else:
            self._snapshots = list(traj.load())

        if not self._snapshots:
            raise RuntimeError("Recording produced no snapshots")

        # Determine start offset — skip stationary startup, optionally
        # randomize into the recording for more diverse episodes.
        self._start_offset = self._pick_start_offset()

        # Init ego from start snapshot
        s0 = self._snapshots[self._start_offset]
        self._ego = EgoKinematics(s0.ego_x, s0.ego_y, s0.ego_speed, s0.ego_heading)
        self._step_idx = self._start_offset
        self._episode_steps = 0
        self._braking_received_decay = 0.0
        self._last_action = 0.0
        self._last_decel = 0.0

        obs = self._build_observation(self._snapshots[self._start_offset])
        # Initialize ego history with N copies of the first ego vector
        self._ego_history.clear()
        for _ in range(self.ego_stack_frames):
            self._ego_history.append(obs["ego"].copy())
        obs["ego"] = self._stack_ego()
        info = {
            "recording_index": traj_idx,
            "num_snapshots": len(self._snapshots),
            "start_offset": self._start_offset,
            "augmented": self.do_augment,
        }
        return obs, info

    def _pick_start_offset(self) -> int:
        """Choose episode start index.

        Always skips the stationary startup (speed < 0.5 m/s).
        When ``random_start=True``, picks a random offset within the
        first half of the moving portion so the episode still has room
        to run.
        """
        # Find first moving snapshot
        first_moving = 0
        for i, snap in enumerate(self._snapshots):
            if snap.ego_speed >= _MIN_MOVING_SPEED:
                first_moving = i
                break

        if not self.random_start:
            return first_moving

        # Random start within first half of remaining recording
        remaining = len(self._snapshots) - first_moving
        max_offset = max(0, remaining // 2)
        if max_offset <= 1:
            return first_moving
        return first_moving + int(self._rng.integers(0, max_offset))

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Parse action
        if hasattr(action, "__len__"):
            action_value = float(action[0])
        else:
            action_value = float(action)
        action_value = max(0.0, min(1.0, action_value))

        self._step_idx += 1
        self._episode_steps += 1

        # Check bounds
        truncated = False
        if self._step_idx >= len(self._snapshots):
            truncated = True
            self._step_idx = len(self._snapshots) - 1
        if self._episode_steps >= self.max_steps:
            truncated = True

        snap = self._snapshots[self._step_idx]

        # --- Ego state: recorded for observations, optional shadow kinematics
        # for reward geometry ---
        reward_geometry_source = "kinematic"
        reward_ego_state = None
        reward_ego_pos = None
        if self.use_recorded_ego:
            ego_state, ego_pos = self._recorded_ego_state(snap)

            # Default recorded-ego replay semantics: observation geometry and
            # reward geometry both follow the recorded human trajectory.
            actual_decel = action_value * self.MAX_DECEL
            reward_ego_state = ego_state
            reward_ego_pos = ego_pos
            reward_geometry_source = "recorded"

            if self.use_shadow_reward_geometry:
                actual_decel = self._ego.step(action_value, snap.ego_heading)
                reward_ego_state = self._ego.to_vehicle_state()
                reward_ego_pos = (reward_ego_state.x, reward_ego_state.y)
                reward_geometry_source = "shadow"
        else:
            actual_decel = self._ego.step(action_value, snap.ego_heading)
            ego_state = self._ego.to_vehicle_state()
            ego_pos = (self._ego.x, self._ego.y)
            reward_ego_state = ego_state
            reward_ego_pos = ego_pos

        self._last_action = action_value
        self._last_decel = actual_decel

        # Build peer observations
        peer_obs = self._build_peer_obs(snap.peers, ego_pos, ego_state)

        # Cone-filter peers first (matches ConvoyEnv and validator),
        # then detect braking among the visible set.
        cone_filtered_peers = self.obs_builder.filter_observable_peers(
            ego_heading_deg=ego_state.heading,
            ego_pos=ego_pos,
            peer_observations=peer_obs,
        )
        any_braking_peer = any(
            p["accel"] <= self.BRAKING_ACCEL_THRESHOLD
            for p in cone_filtered_peers
        )

        # Update braking decay
        if any_braking_peer:
            self._braking_received_decay = 1.0
        else:
            self._braking_received_decay *= self.BRAKING_DECAY

        # Build observation via ObservationBuilder (applies cone filter internally)
        obs = self.obs_builder.build(
            ego_state=ego_state,
            peer_observations=peer_obs,
            ego_pos=ego_pos,
            braking_received=self._braking_received_decay,
        )

        # Ego frame stacking (Run 025)
        self._ego_history.append(obs["ego"].copy())
        obs["ego"] = self._stack_ego()

        recorded_distance = self._nearest_peer_distance(snap.peers, ego_pos)
        recorded_closing_rate = self._closing_rate(snap.peers, ego_state, ego_pos)
        reward_distance = self._nearest_peer_distance(snap.peers, reward_ego_pos)
        reward_closing_rate = self._closing_rate(
            snap.peers, reward_ego_state, reward_ego_pos
        )

        # Reward — deceleration sign: positive means slowing down for
        # recorded-ego mode, signed for kinematics mode.
        reward_decel = actual_decel if self.use_recorded_ego else -actual_decel
        reward_ego_speed = reward_ego_state.speed

        reward, reward_info = self.reward_calculator.calculate(
            distance=reward_distance,
            action_value=action_value,
            deceleration=reward_decel,
            closing_rate=reward_closing_rate,
            any_braking_peer=any_braking_peer,
            braking_received=self._braking_received_decay,
            ego_speed=reward_ego_speed,
        )

        # Collision check — only meaningful in kinematics mode
        terminated = (
            (not self.use_recorded_ego)
            and reward_distance < self.COLLISION_DIST
        )

        info = {
            "distance": reward_distance,
            "closing_rate": reward_closing_rate,
            "recorded_distance": recorded_distance,
            "recorded_closing_rate": recorded_closing_rate,
            "reward_distance": reward_distance,
            "reward_closing_rate": reward_closing_rate,
            "reward_geometry_source": reward_geometry_source,
            "deceleration": actual_decel,
            "braking_received_decay": self._braking_received_decay,
            "any_braking_peer": any_braking_peer,
            "ego_speed": reward_ego_speed,
            "observation_ego_speed": ego_state.speed,
            "step_idx": self._step_idx,
            "action_value": action_value,
            "reward_config": self.reward_config,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def _build_peer_obs(
        self,
        peers: List[PeerSnapshot],
        ego_pos: Tuple[float, float],
        ego_state: VehicleState,
    ) -> List[Dict[str, float]]:
        """Convert PeerSnapshots to observation dicts for ObservationBuilder."""
        result = []
        for p in peers:
            result.append({
                "x": p.x,
                "y": p.y,
                "speed": p.speed,
                "heading": p.heading,
                "accel": p.accel,
                "age_ms": p.age_ms,
                "valid": True,
            })
        return result

    def _nearest_peer_distance(
        self,
        peers: List[PeerSnapshot],
        ego_pos: Tuple[float, float],
    ) -> float:
        """Euclidean distance to nearest peer."""
        if not peers:
            return 100.0  # No peers → far away

        min_dist = float("inf")
        for p in peers:
            dx = p.x - ego_pos[0]
            dy = p.y - ego_pos[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
        return min_dist

    def _closing_rate(
        self,
        peers: List[PeerSnapshot],
        ego_state: VehicleState,
        ego_pos: Tuple[float, float],
    ) -> float:
        """Closing rate to nearest front-cone peer."""
        if not peers:
            return 0.0

        # Find nearest peer in front cone
        best_dist = float("inf")
        best_speed = ego_state.speed
        for p in peers:
            if self.obs_builder.is_in_cone(
                ego_state.heading, ego_pos, p.x, p.y
            ):
                dx = p.x - ego_pos[0]
                dy = p.y - ego_pos[1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_dist:
                    best_dist = d
                    best_speed = p.speed

        return max(0.0, ego_state.speed - best_speed)

    def _build_observation(
        self, snap: TrajectorySnapshot
    ) -> Dict[str, np.ndarray]:
        """Build initial observation from snapshot."""
        if self.use_recorded_ego:
            ego_state, ego_pos = self._recorded_ego_state(snap)
        else:
            ego_state = self._ego.to_vehicle_state()
            ego_pos = (self._ego.x, self._ego.y)
        peer_obs = self._build_peer_obs(snap.peers, ego_pos, ego_state)
        return self.obs_builder.build(
            ego_state=ego_state,
            peer_observations=peer_obs,
            ego_pos=ego_pos,
            braking_received=self._braking_received_decay,
        )

    def _recorded_ego_state(
        self, snap: TrajectorySnapshot
    ) -> Tuple[VehicleState, Tuple[float, float]]:
        """Return the ego state exactly as recorded in the TX CSV."""
        ego_state = VehicleState(
            vehicle_id="V001",
            x=snap.ego_x,
            y=snap.ego_y,
            speed=max(0.0, snap.ego_speed),
            acceleration=snap.ego_accel,
            heading=snap.ego_heading,
            lane_position=0.0,
        )
        return ego_state, (snap.ego_x, snap.ego_y)

    def _stack_ego(self) -> np.ndarray:
        """Flatten ego history deque into a single vector.

        Order: [ego_t, ego_{t-1}, ..., ego_{t-N+1}] — most recent first.
        When ego_stack_frames=1, returns the single (6,) vector unchanged.
        """
        # deque has most-recent at the right end; reverse so index 0:6 = current
        return np.concatenate(list(reversed(self._ego_history)), dtype=np.float32)
