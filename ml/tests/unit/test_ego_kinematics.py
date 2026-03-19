"""Unit tests for EgoKinematics."""

import math
import pytest
from ml.envs.ego_kinematics import EgoKinematics
from ml.envs.sumo_connection import VehicleState


class TestEgoKinematicsInit:
    def test_initial_state_matches_constructor_args(self):
        ego = EgoKinematics(100.0, 200.0, 13.9, 45.0)
        assert ego.x == 100.0
        assert ego.y == 200.0
        assert ego.speed == 13.9
        assert ego.heading == 45.0
        assert ego.acceleration == 0.0

    def test_negative_speed_clamped_to_zero(self):
        ego = EgoKinematics(0.0, 0.0, -5.0, 0.0)
        assert ego.speed == 0.0


class TestEgoKinematicsStep:
    def test_step_zero_action_maintains_speed(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        decel = ego.step(0.0, 0.0)
        assert ego.speed == 10.0
        assert decel == 0.0

    def test_step_full_action_decelerates_by_max_decel(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        decel = ego.step(1.0, 0.0)
        # 10.0 - 8.0 * 0.1 = 9.2
        assert abs(ego.speed - 9.2) < 1e-6
        assert abs(decel - 8.0) < 1e-6

    def test_step_half_action(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        decel = ego.step(0.5, 0.0)
        # 10.0 - 4.0 * 0.1 = 9.6
        assert abs(ego.speed - 9.6) < 1e-6
        assert abs(decel - 4.0) < 1e-6

    def test_step_clamps_speed_to_zero(self):
        ego = EgoKinematics(0.0, 0.0, 0.5, 0.0)
        decel = ego.step(1.0, 0.0)
        # 0.5 - 8.0 * 0.1 = -0.3 → clamped to 0
        assert ego.speed == 0.0
        # Actual decel = (0.5 - 0.0) / 0.1 = 5.0
        assert abs(decel - 5.0) < 1e-6

    def test_step_clamps_action_above_one(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        ego.step(1.5, 0.0)
        # Should clamp to 1.0: 10.0 - 8.0 * 0.1 = 9.2
        assert abs(ego.speed - 9.2) < 1e-6

    def test_step_clamps_action_below_zero(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        ego.step(-0.5, 0.0)
        assert ego.speed == 10.0

    def test_step_updates_position_heading_north(self):
        """heading=0 (North in SUMO) → y increases."""
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        ego.step(0.0, 0.0)
        # Heading 0° SUMO → Cartesian 90° → cos=0, sin=1
        assert abs(ego.x) < 1e-6
        assert abs(ego.y - 1.0) < 1e-6  # 10.0 * 0.1 = 1.0

    def test_step_updates_position_heading_east(self):
        """heading=90 (East in SUMO) → x increases."""
        ego = EgoKinematics(0.0, 0.0, 10.0, 90.0)
        ego.step(0.0, 90.0)
        # Heading 90° SUMO → Cartesian 0° → cos=1, sin=0
        assert abs(ego.x - 1.0) < 1e-6
        assert abs(ego.y) < 1e-6

    def test_step_returns_actual_deceleration(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        decel = ego.step(0.3, 0.0)
        # requested = 0.3 * 8.0 = 2.4 m/s²
        expected_decel = 2.4
        assert abs(decel - expected_decel) < 1e-6

    def test_acceleration_tracks_speed_change(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        ego.step(1.0, 0.0)
        # speed: 10.0 → 9.2, accel = (9.2 - 10.0) / 0.1 = -8.0
        assert abs(ego.acceleration - (-8.0)) < 1e-6

    def test_heading_updated_from_argument(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        ego.step(0.0, 135.0)
        assert ego.heading == 135.0


class TestToVehicleState:
    def test_to_vehicle_state_produces_valid_object(self):
        ego = EgoKinematics(100.0, 200.0, 13.9, 45.0)
        vs = ego.to_vehicle_state()
        assert isinstance(vs, VehicleState)
        assert vs.vehicle_id == "V001"
        assert vs.x == 100.0
        assert vs.y == 200.0
        assert vs.speed == 13.9
        assert vs.heading == 45.0
        assert vs.lane_position == 0.0

    def test_to_vehicle_state_reflects_step(self):
        ego = EgoKinematics(0.0, 0.0, 10.0, 0.0)
        ego.step(1.0, 90.0)
        vs = ego.to_vehicle_state()
        assert abs(vs.speed - 9.2) < 1e-6
        assert abs(vs.acceleration - (-8.0)) < 1e-6
        assert vs.heading == 90.0
