"""
Replay A/B evaluation on real recordings: Keras vs TFLite.

Runs ReplayConvoyEnv on a recordings directory (subfolders with V001_tx.csv +
V001_rx.csv) and compares Keras and TFLite policies episode-by-episode.

Usage (inside Docker):
  python -m ml.deployment.step10_eval_replay_tflite_ab \
    --recordings_dir /path/to/recordings_root \
    --episodes 20 \
    --ego_stack_frames 3 \
    --savedmodel ml/deployment/artifacts/model_float_savedmodel \
    --tflite ml/deployment/artifacts/model_int8_dr.tflite \
    --use_recorded_ego
"""
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import gymnasium as gym
import ml.envs  # registers ReplayConvoyEnv

from ml.envs.replay_convoy_env import ReplayConvoyEnv


DEFAULT_SAVEDMODEL = "ml/deployment/artifacts/model_float_savedmodel"
DEFAULT_TFLITE     = "ml/deployment/artifacts/model_int8_dr.tflite"


class KerasPolicy:
    def __init__(self, savedmodel_path: str) -> None:
        self._model = tf.saved_model.load(savedmodel_path)
        self._infer = self._model.signatures["serving_default"]
        sig_inputs = self._infer.structured_input_signature[1]
        self._name_for_key: Dict[str, str] = {}
        for name, spec in sig_inputs.items():
            s = tuple(spec.shape.as_list()[1:])
            if s == (18,):
                self._name_for_key["ego"] = name
            elif s == (8, 6):
                self._name_for_key["peers"] = name
            elif s == (8,):
                self._name_for_key["peer_mask"] = name
        if set(self._name_for_key.keys()) != {"ego", "peers", "peer_mask"}:
            raise RuntimeError("SavedModel inputs do not match expected shapes")

    def predict(self, obs: Dict[str, np.ndarray]) -> float:
        feed = {
            self._name_for_key["ego"]: tf.constant(obs["ego"][None, :], dtype=tf.float32),
            self._name_for_key["peers"]: tf.constant(obs["peers"][None, :, :], dtype=tf.float32),
            self._name_for_key["peer_mask"]: tf.constant(obs["peer_mask"][None, :], dtype=tf.float32),
        }
        out = self._infer(**feed)
        val = list(out.values())[0].numpy().reshape(-1)[0]
        return float(np.clip(val, 0.0, 1.0))


class TFLitePolicy:
    def __init__(self, tflite_path: str) -> None:
        with open(tflite_path, "rb") as f:
            content = f.read()
        self._interp = tf.lite.Interpreter(
            model_content=content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_delegates=[],
        )
        self._interp.allocate_tensors()
        self._inps = self._interp.get_input_details()
        self._outs = self._interp.get_output_details()
        self._index: Dict[str, int] = {}
        for d in self._inps:
            shape = tuple(d["shape"].tolist())
            if shape == (1, 18):
                self._index["ego"] = d["index"]
            elif shape == (1, 8, 6):
                self._index["peers"] = d["index"]
            elif shape == (1, 8):
                self._index["peer_mask"] = d["index"]
        if set(self._index.keys()) != {"ego", "peers", "peer_mask"}:
            raise RuntimeError(f"Unexpected TFLite input shapes: {[d['shape'].tolist() for d in self._inps]}")

    def predict(self, obs: Dict[str, np.ndarray]) -> float:
        self._interp.set_tensor(self._index["ego"], obs["ego"][None, :].astype(np.float32))
        self._interp.set_tensor(self._index["peers"], obs["peers"][None, :, :].astype(np.float32))
        self._interp.set_tensor(self._index["peer_mask"], obs["peer_mask"][None, :].astype(np.float32))
        self._interp.invoke()
        val = float(self._interp.get_tensor(self._outs[0]["index"]).reshape(-1)[0])
        return float(np.clip(val, 0.0, 1.0))


@dataclass
class EpisodeStats:
    length: int
    reward_sum: float


def run_episode(env: gym.Env, policy) -> EpisodeStats:
    obs, _ = env.reset()
    done = False
    truncated = False
    length = 0
    reward_sum = 0.0
    while not (done or truncated):
        action = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(np.array([action], dtype=np.float32))
        length += 1
        reward_sum += float(reward)
    return EpisodeStats(length=length, reward_sum=reward_sum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings_dir", required=False, help="Root containing subfolders with V001_tx.csv + V001_rx.csv")
    ap.add_argument("--tx_csv", required=False, help="Path to a single recording's V001_tx.csv")
    ap.add_argument("--rx_csv", required=False, help="Path to a single recording's V001_rx.csv")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ego_stack_frames", type=int, default=3)
    ap.add_argument("--use_recorded_ego", action="store_true")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--random_start", action="store_true")
    ap.add_argument("--savedmodel", default=DEFAULT_SAVEDMODEL)
    ap.add_argument("--tflite", default=DEFAULT_TFLITE)
    args = ap.parse_args()

    keras_policy = KerasPolicy(args.savedmodel)
    tflite_policy = TFLitePolicy(args.tflite)

    # Resolve recordings root. Support either a directory of subfolders or a single pair of CSVs.
    recordings_root = args.recordings_dir
    if (args.tx_csv and not args.rx_csv) or (args.rx_csv and not args.tx_csv):
        raise SystemExit("Provide both --tx_csv and --rx_csv, or use --recordings_dir")
    if args.tx_csv and args.rx_csv:
        import os, shutil
        tmp_root = "ml/deployment/artifacts/replay_eval_tmp"
        rec_dir = os.path.join(tmp_root, "rec1")
        os.makedirs(rec_dir, exist_ok=True)
        shutil.copyfile(args.tx_csv, os.path.join(rec_dir, "V001_tx.csv"))
        shutil.copyfile(args.rx_csv, os.path.join(rec_dir, "V001_rx.csv"))
        recordings_root = tmp_root
    if not recordings_root:
        raise SystemExit("Must provide --recordings_dir or (--tx_csv and --rx_csv)")

    keras_stats = []
    tflite_stats = []

    for i in range(args.episodes):
        # Baseline (Keras)
        env = ReplayConvoyEnv(
            recordings_dir=recordings_root,
            augment=not args.no_augment,
            use_recorded_ego=args.use_recorded_ego,
            random_start=args.random_start,
            ego_stack_frames=args.ego_stack_frames,
        )
        env.reset(seed=args.seed + i)
        ks = run_episode(env, keras_policy)
        env.close()
        keras_stats.append(ks)

        # Candidate (TFLite)
        env = ReplayConvoyEnv(
            recordings_dir=recordings_root,
            augment=not args.no_augment,
            use_recorded_ego=args.use_recorded_ego,
            random_start=args.random_start,
            ego_stack_frames=args.ego_stack_frames,
        )
        env.reset(seed=args.seed + i)
        ts = run_episode(env, tflite_policy)
        env.close()
        tflite_stats.append(ts)

    def summarize(stats):
        L = np.array([s.length for s in stats], dtype=np.float32)
        R = np.array([s.reward_sum for s in stats], dtype=np.float32)
        return float(L.mean()), float(R.mean())

    kL, kR = summarize(keras_stats)
    tL, tR = summarize(tflite_stats)

    print("\n=== Replay A/B Summary (mean over episodes) ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mode: recorded_ego={'on' if args.use_recorded_ego else 'off'}  augment={'off' if args.no_augment else 'on'}  random_start={'on' if args.random_start else 'off'}")
    print(f"Keras    → len={kL:.1f}  reward={kR:.3f}")
    print(f"TFLite   → len={tL:.1f}  reward={tR:.3f}")
    print(f"Deltas   → len={tL-kL:+.1f}  reward={tR-kR:+.3f}")


if __name__ == "__main__":
    main()
