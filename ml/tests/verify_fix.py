from espnow_emulator.espnow_emulator import ESPNOWEmulator, V2VMessage

def create_msg(vid, timestamp):
    return V2VMessage(
        vehicle_id=vid, lat=0, lon=0, speed=0, heading=0,
        accel_x=0, accel_y=0, accel_z=0, gyro_x=0, gyro_y=0, gyro_z=0,
        timestamp_ms=timestamp
    )

def test_collision():
    print("Testing concurrent arrival collision...")
    emulator = ESPNOWEmulator(domain_randomization=False)
    
    # Disable loss and jitter to control arrival time exactly
    emulator.params['packet_loss']['base_rate'] = 0.0
    emulator.params['latency']['base_ms'] = 50
    emulator.params['latency']['distance_factor'] = 0
    emulator.params['latency']['jitter_std_ms'] = 0

    # Message 1: Sent T=0, arrives T=50
    msg1 = create_msg("V002", 0)
    emulator.transmit(msg1, (0,0), (0,0), 0)

    # Message 2: Sent T=0, arrives T=50 (Same arrival time!)
    # In previous version, this would crash because PriorityQueue 
    # would try to compare ReceivedMessage objects.
    # Now it should use the sequence counter.
    msg2 = create_msg("V002", 0) # Same vehicle ID too, to force deep comparison
    emulator.transmit(msg2, (0,0), (0,0), 0)

    print("Transmission complete. Checking observation...")
    
    # Should not crash here
    obs = emulator.get_observation(10.0, 50)
    print("Observation retrieved successfully.")
    print(f"V002 valid: {obs['v002_valid']}")

if __name__ == "__main__":
    try:
        test_collision()
        print("PASS: No crash detected.")
    except Exception as e:
        print(f"FAIL: Crashed with error: {e}")
        import traceback
        traceback.print_exc()
