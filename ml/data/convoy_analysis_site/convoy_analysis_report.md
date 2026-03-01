# Convoy Analysis Report (Ego-Only Mode)

## Scope
- Dataset mode: **ego-only mesh logging** (V001 tx/rx).
- Peer-side tx/rx files are not required in this mode.
- Link quality metrics are estimated from V001 RX message spacing.

## Data Validation

| File | Rows (valid/total) | GPS fix % | Duration (s) | Duplicates | Gaps >500ms | Sensor status |
|---|---:|---:|---:|---:|---:|---|
| `V001_tx_004.csv` | 4984/4984 | 100.0 | 195.5 | 0 | 0 | WARN |
| `V001_rx_004.csv` | 3768/3768 | 100.0 | 195.5 | 0 | 2 | WARN |

- TX/RX overlap window on V001 clock: 195.5 s

## Link Health (Estimated)

| Link | RX packets | Estimated PDR | Estimated Loss | Mean inter-arrival (ms) | p95 inter-arrival (ms) |
|---|---:|---:|---:|---:|---:|
| V002->V001 | 1667 | 0.852 | 0.148 | 117.35 | 200.00 |
| V003->V001 | 1471 | 0.752 | 0.248 | 132.99 | 300.00 |

## Latency / Burst (Estimated)

- V002->V001: rel latency mean=0.66 ms, p95=3.00 ms, mean burst=1.97, max burst=23
- V003->V001: rel latency mean=35.23 ms, p95=80.00 ms, mean burst=2.17, max burst=14

## Event / Observation Checks

- Selected braking events in V001 TX: 3.
  - light_braking: start=115971 ms, end=121071 ms, min_accel=-1.45 m/s^2
  - light_braking: start=121971 ms, end=122371 ms, min_accel=-0.98 m/s^2
  - hard_braking: start=234071 ms, end=237371 ms, min_accel=-4.11 m/s^2
- Zero-peer windows (>500ms): 0
- Formation class (ego-based estimate): tight
- Average spacing (ego-based estimate): 13.92 m

## Observation Range Summary

- ego_accel_norm: min=-0.864, max=0.794, outside[-1.5,1.5]=0.00%
- ego_heading_norm: min=-1.000, max=0.999, outside[-1.5,1.5]=0.00%
- ego_speed_norm: min=0.000, max=0.399, outside[-1.5,1.5]=0.00%
- peer_accel_norm: min=-0.864, max=0.794, outside[-1.5,1.5]=0.00%
- peer_age_norm: min=0.068, max=0.970, outside[-1.5,1.5]=0.00%
- peer_count_norm: min=0.000, max=0.250, outside[-1.5,1.5]=0.00%
- peer_rel_heading_norm: min=-0.205, max=0.766, outside[-1.5,1.5]=0.00%
- peer_rel_speed_norm: min=-0.304, max=0.257, outside[-1.5,1.5]=0.00%
- peer_rel_x_norm: min=-0.744, max=0.328, outside[-1.5,1.5]=0.00%
- peer_rel_y_norm: min=-0.324, max=0.506, outside[-1.5,1.5]=0.00%

## Trajectories

- V001: `V001_trajectory.csv` (1956 samples, 195.5s)
- V002: `V002_trajectory.csv` (1956 samples, 195.5s)
- V003: `V003_trajectory.csv` (1956 samples, 195.5s)