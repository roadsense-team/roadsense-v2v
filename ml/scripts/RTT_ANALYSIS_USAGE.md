# RTT Analysis Script Usage

This document explains how to run the RTT analysis script and what each flag
does, with guidance on when you should use them.

## Quick Start (Recommended)

Run with defaults (uses `/home/amirkhalifa/RoadSense2/rtt_log.csv` and writes
outputs to `roadsense-v2v/ml/data/rtt_analysis/`):

```bash
/home/amirkhalifa/RoadSense2/roadsense-v2v/ml/venv/bin/python \
  /home/amirkhalifa/RoadSense2/roadsense-v2v/ml/scripts/analyze_rtt_characterization.py
```

## Enable Matplotlib Plots

The script will generate PNG plots if `matplotlib` is installed in the ML venv.

Install it once:

```bash
/home/amirkhalifa/RoadSense2/roadsense-v2v/ml/venv/bin/python -m pip install matplotlib
```

Then run the script normally. Plot files are written to:

```
roadsense-v2v/ml/data/rtt_analysis/
```

## Flags Reference (What and When)

### `--input /path/to/rtt_log.csv`
**What it does:** Tells the script where your CSV log file is.
**When to use:** Only if your log is not the default `/home/amirkhalifa/RoadSense2/rtt_log.csv`.

### `--out-dir /path/to/output_folder`
**What it does:** Sets where the summary JSON, plots, and loss CSV are written.
**When to use:** If you want outputs in a different folder.

### `--params-out /path/to/emulator_params.json`
**What it does:** Sets the output path for emulator parameters.
**When to use:** If you want to keep multiple parameter files or save elsewhere.

### `--segment-seconds 10`
**What it does:** Sets the time bin size for packet loss over time.
**When to use:**
- Smaller value (e.g., 5): more detail, noisier.
- Larger value (e.g., 30): smoother, less detail.

### `--stationary-speed-ms 0.5`
**What it does:** Defines "stationary" for IMU and GPS noise calculations.
**When to use:**
- Raise it if the vehicle never fully stops (e.g., 1.0 m/s).
- Lower it if you have clean stationary data.

### `--heading-min-speed-ms 2.0`
**What it does:** Heading noise is only meaningful when moving; this sets the
minimum speed to include in heading noise stats.
**When to use:** Increase if you want heading noise from faster motion only.

### `--monitored-vehicles V002,V003`
**What it does:** Sets the list of vehicle IDs written into emulator params.
**When to use:** If you want a different set of vehicles in the emulator config.

### `--no-plots`
**What it does:** Skips PNG generation.
**When to use:** If you are running headless or do not need graphs.

## Example: Custom Input + Smoother Loss Curve

```bash
/home/amirkhalifa/RoadSense2/roadsense-v2v/ml/venv/bin/python \
  /home/amirkhalifa/RoadSense2/roadsense-v2v/ml/scripts/analyze_rtt_characterization.py \
  --input /path/to/another_log.csv \
  --segment-seconds 30
```

## Output Files

Default output folder:

```
roadsense-v2v/ml/data/rtt_analysis/
```

Key outputs:
- `analysis_summary.json` (main metrics and assumptions)
- `loss_by_segment.csv` (loss over time bins)
- `rtt_distribution.png`, `loss_over_time.png`, `gps_track.png`
- `roadsense-v2v/ml/espnow_emulator/emulator_params.json`
