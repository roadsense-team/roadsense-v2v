# Convoy Analysis Report

## Scope
- Dataset: `Convoy_recording_02212026` (V001, V002, V003).
- Ego perspective for model validation: **V002**.
- Ground-truth protocol expected: 2 light braking events, 1 hard braking event, final stationary window.

## Task 1 - Data Validation

| File | Rows (valid/total) | GPS fix % | Duration (s) | Duplicates | Gaps >500ms | Sensor status |
|---|---:|---:|---:|---:|---:|---|
| `V001_tx_002.csv` | 2087/2087 | 100.0 | 208.6 | 0 | 0 | WARN |
| `V001_rx_002.csv` | 2220/2220 | 100.0 | 208.5 | 0 | 0 | WARN |
| `V002_tx_004.csv` | 2024/2024 | 100.0 | 204.2 | 0 | 2 | WARN |
| `V002_rx_004.csv` | 2318/2318 | 100.0 | 204.2 | 5 | 1 | WARN |
| `V003_tx_002.csv` | 2078/2078 | 100.0 | 207.7 | 0 | 0 | WARN |
| `V003_rx_002.csv` | 3812/3812 | 100.0 | 207.7 | 0 | 0 | WARN |

- Estimated 3-vehicle overlap window (mapped to V002 clock): 200.7 s.

## Task 2 - Packet Delivery / Latency

| Link | TX in overlap | Matched RX | PDR | Loss | Mean relative latency (ms) | p95 relative latency (ms) | Mean burst length |
|---|---:|---:|---:|---:|---:|---:|---:|
| V001->V002 | 1999 | 317 | 0.159 | 0.841 | 1.29 | 4.00 | 14.75 |
| V001->V003 | 2067 | 1920 | 0.929 | 0.071 | 1.22 | 3.00 | 1.34 |
| V002->V001 | 1938 | 269 | 0.139 | 0.861 | 1.48 | 2.00 | 14.26 |
| V002->V003 | 2000 | 1824 | 0.912 | 0.088 | 1.92 | 4.00 | 1.39 |
| V003->V001 | 2066 | 1927 | 0.933 | 0.067 | 1.71 | 3.00 | 1.13 |
| V003->V002 | 2020 | 1976 | 0.978 | 0.022 | 5.01 | 3.00 | 1.13 |

- Overall relative latency: mean 2.41 ms, p95 3.00 ms, p99 5.00 ms.

## Task 3 - Sensor Noise

- Segment aggregation is across all three TX logs.
- `stationary` and `cruising` are the primary references for emulator calibration.

## Task 4 - Offline Observation / Inference

- Observation arrays exported in `convoy_observations.npz` (ego, peers, peer_mask, timestamps).
- Zero-peer windows (>500ms): 0.
- No fields exceeded [-1.5, 1.5].
- Model inference executed using `/home/amirkhalifa/RoadSense2/roadsense-v2v/ml/models/runs/run_20260115_210446/model_final.zip`.
- Action fractions: {'0': 1.0, '1': 0.0, '2': 0.0, '3': 0.0}.
- Hard-event window action summary: {'samples': 15, 'action_counts': {'0': 15, '1': 0, '2': 0, '3': 0}, 'maintain_rate': 1.0, 'brake_or_higher_rate': 0.0, 'emergency_rate': 0.0}.

## Task 5 - Trajectory / Formation

- Formation classification: **moderate**.
- Average convoy spacing: 16.73 m.
- Extracted trajectories:
  - V001: `V001_trajectory.csv` (2087 samples, 208.6s)
  - V002: `V002_trajectory.csv` (2042 samples, 204.2s)
  - V003: `V003_trajectory.csv` (2077 samples, 207.7s)

## Event Detection

- Selected braking events: 3.
  - light_braking: start=913812 ms, end=922512 ms, min_accel=-1.34 m/s^2
  - light_braking: start=925712 ms, end=950612 ms, min_accel=-1.39 m/s^2
  - hard_braking: start=1063412 ms, end=1064912 ms, min_accel=-1.44 m/s^2
- Final stationary tail: start=1080612 ms, end=1122414 ms, duration=41.8s

## Figures

- `gps_track_map.png`
- `speed_profiles.png`
- `packet_delivery_over_time.png`
- `latency_distribution.png`
- `inter_vehicle_distance.png`
- `braking_event_detail.png`
- `sensor_noise_comparison.png`
- `observation_value_distributions.png`