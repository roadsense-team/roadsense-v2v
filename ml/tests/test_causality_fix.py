"""
Test to verify causality bug is fixed.

This test ensures that messages sent with latency are NOT visible
until their arrival time (preventing Sim2Real failure).
"""

import pytest
from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage


class TestCausalityFix:
    """Verify that the causality bug is fixed."""

    def test_message_not_visible_before_arrival(self):
        """
        CRITICAL TEST: Messages should NOT be visible before arrival time.

        This test verifies the fix for the causality violation bug:
        - Message sent at t=1000 with 50ms latency
        - Should NOT be visible at t=1001 (before arrival)
        - Should BE visible at t=1050+ (after arrival)
        """
        # Create emulator with NO domain randomization (deterministic)
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Override latency to fixed 50ms for test
        emulator.params['latency']['base_ms'] = 50
        emulator.params['latency']['distance_factor'] = 0
        emulator.params['latency']['jitter_std_ms'] = 0

        # Disable packet loss
        emulator.params['packet_loss']['base_rate'] = 0

        # Disable sensor noise for deterministic test
        emulator.params['sensor_noise']['gps_std_m'] = 0.0
        emulator.params['sensor_noise']['speed_std_ms'] = 0.0
        emulator.params['sensor_noise']['accel_std_ms2'] = 0.0
        emulator.params['sensor_noise']['heading_std_deg'] = 0.0

        # Create test message
        msg = V2VMessage(
            vehicle_id='V002',
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        # Send message at t=1000
        received = emulator.transmit(
            sender_msg=msg,
            sender_pos=(10, 0),
            receiver_pos=(0, 0),
            current_time_ms=1000
        )

        # Verify message was sent (not lost)
        assert received is not None
        assert received.received_at_ms == 1050  # Should arrive at t=1050

        # ===== CRITICAL TEST: Message should NOT be visible at t=1001 =====
        obs_early = emulator.get_observation(ego_speed=10.0, current_time_ms=1001)

        # Message should NOT be in observation (hasn't arrived yet)
        assert obs_early['v002_valid'] is False
        assert obs_early['v002_lat'] == 0.0  # Default value

        # ===== CRITICAL TEST: Message SHOULD be visible at t=1050 =====
        obs_arrived = emulator.get_observation(ego_speed=10.0, current_time_ms=1050)

        # Message should NOW be visible
        assert obs_arrived['v002_valid'] is True
        assert obs_arrived['v002_lat'] == 32.0
        assert obs_arrived['v002_lon'] == 34.0
        assert obs_arrived['v002_speed'] == 15.0

    def test_multiple_messages_delivered_in_order(self):
        """
        Test that multiple messages are delivered in arrival order.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Fixed latency
        emulator.params['latency']['base_ms'] = 50
        emulator.params['latency']['distance_factor'] = 0
        emulator.params['latency']['jitter_std_ms'] = 0
        emulator.params['packet_loss']['base_rate'] = 0

        # Disable sensor noise for deterministic test
        emulator.params['sensor_noise']['gps_std_m'] = 0.0
        emulator.params['sensor_noise']['speed_std_ms'] = 0.0
        emulator.params['sensor_noise']['accel_std_ms2'] = 0.0
        emulator.params['sensor_noise']['heading_std_deg'] = 0.0

        # Send 3 messages at different times
        for i in range(3):
            msg = V2VMessage(
                vehicle_id='V002',
                lat=32.0 + i * 0.001,  # Increment lat
                lon=34.0,
                speed=15.0,
                heading=90.0,
                accel_x=0.0,
                accel_y=0.0,
                accel_z=9.8,
                gyro_x=0.0,
                gyro_y=0.0,
                gyro_z=0.0,
                timestamp_ms=1000 + i * 100
            )

            emulator.transmit(
                sender_msg=msg,
                sender_pos=(10, 0),
                receiver_pos=(0, 0),
                current_time_ms=1000 + i * 100
            )

        # At t=1025: No messages should be visible
        obs = emulator.get_observation(ego_speed=10.0, current_time_ms=1025)
        assert obs['v002_valid'] is False

        # At t=1050: First message visible
        obs = emulator.get_observation(ego_speed=10.0, current_time_ms=1050)
        assert obs['v002_valid'] is True
        assert obs['v002_lat'] == 32.0

        # At t=1150: Second message visible (overwrites first)
        obs = emulator.get_observation(ego_speed=10.0, current_time_ms=1150)
        assert obs['v002_valid'] is True
        assert obs['v002_lat'] == 32.001

        # At t=1250: Third message visible
        obs = emulator.get_observation(ego_speed=10.0, current_time_ms=1250)
        assert obs['v002_valid'] is True
        assert obs['v002_lat'] == 32.002


class TestBurstLossFix:
    """Verify that burst loss tracking is fixed."""

    def test_burst_loss_state_tracking(self):
        """
        Test that burst loss state is actually tracked.

        Previous bug: last_loss_state was never updated.
        Fix: transmit() now updates last_loss_state.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Enable burst loss
        emulator.params['burst_loss']['enabled'] = True

        # Force packet loss (100% loss rate)
        emulator.params['packet_loss']['base_rate'] = 1.0

        msg = V2VMessage(
            vehicle_id='V002',
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        # Send message (will be lost due to 100% loss rate)
        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(10, 0),
            receiver_pos=(0, 0),
            current_time_ms=1000
        )

        # Verify packet was lost
        assert result is None

        # CRITICAL: Verify that loss state was tracked
        assert 'V002' in emulator.last_loss_state
        assert emulator.last_loss_state['V002'] is True

    def test_burst_loss_not_triggered_after_success(self):
        """
        Test that burst loss is not triggered after successful transmission.
        """
        emulator = ESPNOWEmulator(domain_randomization=False)

        # Enable burst loss
        emulator.params['burst_loss']['enabled'] = True

        # No packet loss
        emulator.params['packet_loss']['base_rate'] = 0.0

        msg = V2VMessage(
            vehicle_id='V002',
            lat=32.0,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=1000
        )

        # Send message (will succeed due to 0% loss rate)
        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(10, 0),
            receiver_pos=(0, 0),
            current_time_ms=1000
        )

        # Verify packet was successful
        assert result is not None

        # Verify that loss state shows success
        assert emulator.last_loss_state['V002'] is False
