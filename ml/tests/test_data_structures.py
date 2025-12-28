"""
Unit tests for ESP-NOW Emulator data structures.

Tests V2VMessage and ReceivedMessage dataclasses to ensure they:
- Store all required fields correctly
- Have correct types
- Are immutable (frozen=True)
- Can be serialized/deserialized if needed
"""

import pytest
from dataclasses import fields


# We'll import these once implemented
# from espnow_emulator import V2VMessage, ReceivedMessage


class TestV2VMessage:
    """Test suite for V2VMessage dataclass"""

    @pytest.mark.unit
    def test_v2v_message_creation(self):
        """Test that V2VMessage can be created with all required fields"""
        from espnow_emulator import V2VMessage

        msg = V2VMessage(
            vehicle_id="V002",
            lat=32.085123,
            lon=34.781234,
            speed=15.5,
            heading=90.0,
            accel_x=0.5,
            accel_y=0.1,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.15,
            timestamp_ms=1000
        )

        assert msg.vehicle_id == "V002"
        assert msg.lat == 32.085123
        assert msg.lon == 34.781234
        assert msg.speed == 15.5
        assert msg.heading == 90.0
        assert msg.accel_x == 0.5
        assert msg.accel_y == 0.1
        assert msg.accel_z == 9.81
        assert msg.gyro_x == 0.0
        assert msg.gyro_y == 0.0
        assert msg.gyro_z == 0.15
        assert msg.timestamp_ms == 1000

    @pytest.mark.unit
    def test_v2v_message_field_types(self):
        """Test that V2VMessage has correct field types"""
        from espnow_emulator import V2VMessage

        msg_fields = {f.name: f.type for f in fields(V2VMessage)}

        assert msg_fields['vehicle_id'] == str
        assert msg_fields['lat'] == float
        assert msg_fields['lon'] == float
        assert msg_fields['speed'] == float
        assert msg_fields['heading'] == float
        assert msg_fields['accel_x'] == float
        assert msg_fields['accel_y'] == float
        assert msg_fields['accel_z'] == float
        assert msg_fields['gyro_x'] == float
        assert msg_fields['gyro_y'] == float
        assert msg_fields['gyro_z'] == float
        assert msg_fields['timestamp_ms'] == int

    @pytest.mark.unit
    def test_v2v_message_immutable(self):
        """Test that V2VMessage is immutable (frozen)"""
        from espnow_emulator import V2VMessage

        msg = V2VMessage(
            vehicle_id="V002",
            lat=32.085,
            lon=34.781,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        # Attempt to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # FrozenInstanceError from dataclasses
            msg.speed = 20.0

    @pytest.mark.unit
    def test_v2v_message_equality(self):
        """Test that two V2VMessage instances with same data are equal"""
        from espnow_emulator import V2VMessage

        msg1 = V2VMessage(
            vehicle_id="V002",
            lat=32.085,
            lon=34.781,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        msg2 = V2VMessage(
            vehicle_id="V002",
            lat=32.085,
            lon=34.781,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        assert msg1 == msg2


class TestReceivedMessage:
    """Test suite for ReceivedMessage dataclass"""

    @pytest.mark.unit
    def test_received_message_creation(self):
        """Test that ReceivedMessage can be created with V2VMessage and metadata"""
        from espnow_emulator import V2VMessage, ReceivedMessage

        v2v_msg = V2VMessage(
            vehicle_id="V002",
            lat=32.085,
            lon=34.781,
            speed=15.0,
            heading=90.0,
            accel_x=0.5,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.1,
            timestamp_ms=1000
        )

        received = ReceivedMessage(
            message=v2v_msg,
            age_ms=15,
            received_at_ms=1015
        )

        assert received.message == v2v_msg
        assert received.age_ms == 15
        assert received.received_at_ms == 1015

    @pytest.mark.unit
    def test_received_message_field_types(self):
        """Test that ReceivedMessage has correct field types"""
        from espnow_emulator import V2VMessage, ReceivedMessage

        recv_fields = {f.name: f.type for f in fields(ReceivedMessage)}

        assert recv_fields['message'] == V2VMessage
        assert recv_fields['age_ms'] == int
        assert recv_fields['received_at_ms'] == int

    @pytest.mark.unit
    def test_received_message_immutable(self):
        """Test that ReceivedMessage is immutable"""
        from espnow_emulator import V2VMessage, ReceivedMessage

        v2v_msg = V2VMessage(
            vehicle_id="V002",
            lat=32.085,
            lon=34.781,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        received = ReceivedMessage(
            message=v2v_msg,
            age_ms=15,
            received_at_ms=1015
        )

        # Attempt to modify should raise error
        with pytest.raises(Exception):
            received.age_ms = 20

    @pytest.mark.unit
    def test_received_message_access_nested_fields(self):
        """Test that we can access V2VMessage fields through ReceivedMessage"""
        from espnow_emulator import V2VMessage, ReceivedMessage

        v2v_msg = V2VMessage(
            vehicle_id="V002",
            lat=32.085,
            lon=34.781,
            speed=15.5,
            heading=90.0,
            accel_x=0.5,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.1,
            timestamp_ms=1000
        )

        received = ReceivedMessage(
            message=v2v_msg,
            age_ms=15,
            received_at_ms=1015
        )

        # Access nested fields
        assert received.message.vehicle_id == "V002"
        assert received.message.speed == 15.5
        assert received.message.timestamp_ms == 1000
