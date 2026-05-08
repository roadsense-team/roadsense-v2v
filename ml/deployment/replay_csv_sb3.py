"""
Replay real CSV recording through the SB3 (PyTorch) model and compare
to the confidence values logged by the ESP32.

Usage (inside Docker, from /work):
  python -m ml.deployment.replay_csv_sb3 \
    --tx_csv ml/data/V003_tx_005.csv \
    --rx_csv ml/data/V003_rx_005.csv \
    --model  ml/results/run_025_replay_v1/model_final.zip
"""
import argparse
import os
import shutil

import numpy as np
from stable_baselines3 import PPO

import ml.envs  # registers ReplayConvoyEnv
from ml.envs.replay_convoy_env import ReplayConvoyEnv
from ml.envs.replay_trajectory import ReplayTrajectory


def load_esp32_confidence(tx_csv: str) -> dict:
    """Return {timestamp_ms: confidence} for hop_count=0 rows that have a confidence column."""
    import csv
    result = {}
    with open(tx_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "confidence" not in (reader.fieldnames or []):
            return result
        for row in reader:
            if row.get("hop_count", "0").strip() != "0":
                continue
            try:
                result[int(row["timestamp_local_ms"])] = float(row["confidence"])
            except (ValueError, KeyError):
                pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx_csv", required=True)
    ap.add_argument("--rx_csv", required=True)
    ap.add_argument("--model",  default="ml/results/run_025_replay_v1/model_final.zip")
    args = ap.parse_args()

    # ReplayConvoyEnv looks for V001_tx/rx.csv inside a subfolder
    tmp = "ml/deployment/artifacts/_replay_sb3_tmp/rec1"
    os.makedirs(tmp, exist_ok=True)
    shutil.copyfile(args.tx_csv, os.path.join(tmp, "V001_tx.csv"))
    shutil.copyfile(args.rx_csv, os.path.join(tmp, "V001_rx.csv"))
    recordings_dir = os.path.dirname(tmp)

    # Load ESP32-logged confidence values keyed by timestamp
    esp32_conf = load_esp32_confidence(args.tx_csv)

    # Load the trajectory so we can read timestamps per step
    traj = ReplayTrajectory(
        os.path.join(tmp, "V001_tx.csv"),
        os.path.join(tmp, "V001_rx.csv"),
    )
    snapshots = traj.load()
    timestamps = [s.time_ms for s in snapshots]

    model = PPO.load(args.model, device="cpu")

    env = ReplayConvoyEnv(
        recordings_dir=recordings_dir,
        augment=False,
        use_recorded_ego=True,
        random_start=False,
        ego_stack_frames=3,
    )

    obs, _ = env.reset(seed=0)
    done = False
    truncated = False
    step = 0

    py_confs = []
    esp_confs = []

    print(f"\n{'Step':>5}  {'t(ms)':>8}  {'Python':>8}  {'ESP32':>8}  {'Diff':>8}")
    print("-" * 48)

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        py_conf = float(np.clip(action[0], 0.0, 1.0))

        t_ms = timestamps[step] if step < len(timestamps) else -1
        esp_conf = esp32_conf.get(t_ms, float("nan"))

        diff = abs(py_conf - esp_conf) if not np.isnan(esp_conf) else float("nan")
        flag = "  <!" if (not np.isnan(diff) and diff > 0.15) else ""

        print(f"{step:>5}  {t_ms:>8}  {py_conf:>8.3f}  {esp_conf:>8.3f}  "
              f"{'n/a':>8}" if np.isnan(diff) else
              f"{step:>5}  {t_ms:>8}  {py_conf:>8.3f}  {esp_conf:>8.3f}  {diff:>8.3f}{flag}")

        obs, _, done, truncated, _ = env.step(action)
        py_confs.append(py_conf)
        if not np.isnan(esp_conf):
            esp_confs.append((py_conf, esp_conf))
        step += 1

    print("-" * 48)
    print(f"Total steps: {step}")

    if esp_confs:
        py_arr  = np.array([x[0] for x in esp_confs])
        esp_arr = np.array([x[1] for x in esp_confs])
        mae = float(np.mean(np.abs(py_arr - esp_arr)))
        corr = float(np.corrcoef(py_arr, esp_arr)[0, 1])
        print(f"Compared steps: {len(esp_confs)}")
        print(f"MAE  (mean abs error): {mae:.4f}")
        print(f"Corr (Pearson):        {corr:.4f}")
    else:
        print("No ESP32 confidence column found — printing Python output only.")

    shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)


if __name__ == "__main__":
    main()
