# Convoy Analysis Report (Ego-Only Mode)

## Scope
- Dataset mode: **ego-only mesh logging** (V001 tx/rx).
- Peer-side tx/rx files are not required in this mode.
- Link quality metrics are estimated from V001 RX message spacing.

## Data Validation

| File | Rows (valid/total) | GPS fix % | Duration (s) | Duplicates | Gaps >500ms | Sensor status |
|---|---:|---:|---:|---:|---:|---|
| `V001_tx_005.csv` | 14899/14899 | 100.0 | 581.6 | 0 | 3 | WARN |
| `V001_rx_005.csv` | 12092/12092 | 100.0 | 581.6 | 35 | 7 | WARN |

- TX/RX overlap window on V001 clock: 581.6 s

## Link Health (Estimated)

| Link | RX packets | Estimated PDR | Estimated Loss | Mean inter-arrival (ms) | p95 inter-arrival (ms) |
|---|---:|---:|---:|---:|---:|
| V002->V001 | 5114 | 0.881 | 0.119 | 113.48 | 200.00 |
| V003->V001 | 4500 | 0.774 | 0.226 | 129.27 | 300.00 |

## Latency / Burst (Estimated)

- V002->V001: rel latency mean=4.76 ms, p95=5.00 ms, mean burst=1.65, max burst=23
- V003->V001: rel latency mean=39.79 ms, p95=71.00 ms, mean burst=2.13, max burst=32

## Event / Observation Checks

- Selected braking events in V001 TX: 3.
  - light_braking: start=63348 ms, end=73348 ms, min_accel=-1.41 m/s^2
  - light_braking: start=93748 ms, end=94248 ms, min_accel=-2.01 m/s^2
  - hard_braking: start=248251 ms, end=940424 ms, min_accel=-4.58 m/s^2
- Zero-peer windows (>500ms): 1
- Formation class (ego-based estimate): moderate
- Average spacing (ego-based estimate): 21.22 m

## Observation Range Summary

- ego_accel_norm: min=-0.768, max=0.819, outside[-1.5,1.5]=0.00%
- ego_heading_norm: min=-0.996, max=0.998, outside[-1.5,1.5]=0.00%
- ego_speed_norm: min=0.000, max=0.675, outside[-1.5,1.5]=0.00%
- peer_accel_norm: min=-0.604, max=0.819, outside[-1.5,1.5]=0.00%
- peer_age_norm: min=0.006, max=0.964, outside[-1.5,1.5]=0.00%
- peer_count_norm: min=0.000, max=0.250, outside[-1.5,1.5]=0.00%
- peer_rel_heading_norm: min=-0.630, max=0.598, outside[-1.5,1.5]=0.00%
- peer_rel_speed_norm: min=-0.343, max=0.239, outside[-1.5,1.5]=0.00%
- peer_rel_x_norm: min=-0.919, max=0.883, outside[-1.5,1.5]=0.00%
- peer_rel_y_norm: min=-1.123, max=1.006, outside[-1.5,1.5]=0.00%

## Trajectories

- V001: `V001_trajectory.csv` (5817 samples, 581.6s)