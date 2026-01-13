"""
Unit tests for SUMOConnection (Phase 2).
"""
import pytest
from unittest.mock import patch, MagicMock
import traci
from envs.sumo_connection import SUMOConnection, VehicleState

def test_sumo_connection_init_stores_config_path():
    """Constructor stores sumo_cfg path for later use."""
    conn = SUMOConnection(sumo_cfg="test.sumocfg")
    assert conn.sumo_cfg == "test.sumocfg"

def test_sumo_connection_init_default_port_is_none():
    """Default port is None (auto-select)."""
    conn = SUMOConnection(sumo_cfg="test.sumocfg")
    assert conn.port is None

def test_sumo_connection_start_raises_error_if_traci_fails():
    """If traci.start fails, exception is propagated."""
    with patch('traci.start', side_effect=RuntimeError("TraCI failed")):
        conn = SUMOConnection(sumo_cfg="test.sumocfg")
        with pytest.raises(RuntimeError):
            conn.start()

def test_get_vehicle_state_returns_vehicle_state_dataclass():
    """Returns VehicleState, not dict or tuple."""
    conn = SUMOConnection("cfg")
    
    with patch('traci.vehicle.getPosition', return_value=(10.0, 20.0)), \
         patch('traci.vehicle.getSpeed', return_value=15.0), \
         patch('traci.vehicle.getAcceleration', return_value=1.5), \
         patch('traci.vehicle.getAngle', return_value=90.0), \
         patch('traci.vehicle.getLanePosition', return_value=50.0):
        
        state = conn.get_vehicle_state("V001")
        assert isinstance(state, VehicleState)
        assert state.vehicle_id == "V001"

def test_get_vehicle_state_extracts_position_correctly():
    """x, y match TraCI getPosition()."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.getPosition', return_value=(123.0, 456.0)), \
         patch('traci.vehicle.getSpeed', return_value=0), \
         patch('traci.vehicle.getAcceleration', return_value=0), \
         patch('traci.vehicle.getAngle', return_value=0), \
         patch('traci.vehicle.getLanePosition', return_value=0):
        
        state = conn.get_vehicle_state("V002")
        assert state.x == 123.0
        assert state.y == 456.0

def test_get_vehicle_state_extracts_speed_correctly():
    """speed matches TraCI getSpeed()."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.getPosition', return_value=(0,0)), \
         patch('traci.vehicle.getSpeed', return_value=22.5), \
         patch('traci.vehicle.getAcceleration', return_value=0), \
         patch('traci.vehicle.getAngle', return_value=0), \
         patch('traci.vehicle.getLanePosition', return_value=0):
        
        state = conn.get_vehicle_state("V003")
        assert state.speed == 22.5

def test_get_vehicle_state_extracts_acceleration():
    """acceleration matches TraCI getAcceleration()."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.getPosition', return_value=(0,0)), \
         patch('traci.vehicle.getSpeed', return_value=0), \
         patch('traci.vehicle.getAcceleration', return_value=-2.0), \
         patch('traci.vehicle.getAngle', return_value=0), \
         patch('traci.vehicle.getLanePosition', return_value=0):
        
        state = conn.get_vehicle_state("V004")
        assert state.acceleration == -2.0

def test_get_vehicle_state_nonexistent_vehicle_raises():
    """Requesting unknown vehicle_id raises KeyError or similar."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.getPosition', side_effect=traci.exceptions.TraCIException("Vehicle 'X' does not exist")):
        with pytest.raises(traci.exceptions.TraCIException):
            conn.get_vehicle_state("X")

def test_set_vehicle_speed_changes_target_speed():
    """After set_vehicle_speed(10), vehicle approaches 10 m/s."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.setSpeed') as mock_set_speed:
        conn.set_vehicle_speed("V001", 10.0)
        mock_set_speed.assert_called_with("V001", 10.0)

def test_set_vehicle_speed_zero_stops_vehicle():
    """Setting speed=0 brings vehicle to stop."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.setSpeed') as mock_set_speed:
        conn.set_vehicle_speed("V001", 0.0)
        mock_set_speed.assert_called_with("V001", 0.0)

def test_set_vehicle_speed_negative_clamps_to_zero():
    """Negative speed is clamped to 0."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.setSpeed') as mock_set_speed:
        conn.set_vehicle_speed("V001", -5.0)
        mock_set_speed.assert_called_with("V001", 0.0)

def test_step_advances_simulation():
    """step() calls traci.simulationStep()."""
    conn = SUMOConnection("cfg")
    with patch('traci.simulationStep') as mock_step:
        conn.step()
        mock_step.assert_called_once()

def test_stop_closes_connection():
    """stop() calls traci.close()."""
    conn = SUMOConnection("cfg")
    with patch('traci.close') as mock_close:
        conn.stop()
        mock_close.assert_called_once()

def test_get_simulation_time_returns_float():
    """get_simulation_time() returns current time."""
    conn = SUMOConnection("cfg")
    with patch('traci.simulation.getTime', return_value=12.3):
        assert conn.get_simulation_time() == 12.3

def test_is_vehicle_active_returns_bool():
    """is_vehicle_active checks if vehicle in ID list."""
    conn = SUMOConnection("cfg")
    with patch('traci.vehicle.getIDList', return_value=["V001", "V002"]):
        assert conn.is_vehicle_active("V001") is True
        assert conn.is_vehicle_active("V003") is False
