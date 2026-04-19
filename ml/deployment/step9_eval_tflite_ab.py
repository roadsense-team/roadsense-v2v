"""
A/B evaluation: Keras (baseline) vs TFLite policy in ConvoyEnv.

Runs deterministic episodes with identical seeds and compares episode metrics:
 - collision rate (terminated due to ground-truth distance threshold)
 - average reward
 - episode length (steps)

Supports selecting the exact SUMO scenario config (e.g., base_real) and ESP-NOW
emulator parameters to match training/previous validation.

Usage (inside Docker):
  # Run on base_real with measured emulator params (no hazards)
  python -m ml.deployment.step9_eval_tflite_ab \
    --episodes 20 \
    --ego_stack_frames 3 \
    --scenario base_real \
    --emulator_params ml/espnow_emulator/emulator_params_measured.json \
    --hazard_injection 0 \
    --savedmodel ml/deployment/artifacts/model_float_savedmodel \
    --tflite ml/deployment/artifacts/model_int8_dr.tflite
"""
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import gymnasium as gym
import ml.envs  # registers gym envs
import os


DEFAULT_SAVEDMODEL = "ml/deployment/artifacts/model_float_savedmodel"
DEFAULT_TFLITE     = "ml/deployment/artifacts/model_int8_dr.tflite"


class KerasPolicy:
    def __init__(self, savedmodel_path: str) -> None:
        self._model = tf.saved_model.load(savedmodel_path)
        self._infer = self._model.signatures["serving_default"]
        # Build a name→key map by shape (drop batch dim)
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
        # Reference kernels to avoid delegate differences
        self._interp = tf.lite.Interpreter(
            model_content=content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_delegates=[],
        )
        self._interp.allocate_tensors()
        self._inps = self._interp.get_input_details()
        self._outs = self._interp.get_output_details()
        # Map input indices by shape
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


class SB3Policy:
    def __init__(self, zip_path: str) -> None:
        from stable_baselines3 import PPO
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"SB3 model not found: {zip_path}")
        self._model = PPO.load(zip_path)

    def predict(self, obs: Dict[str, np.ndarray]) -> float:
        # SB3 expects dict of arrays; ConvoyEnv already returns correct shapes
        action, _ = self._model.predict(obs, deterministic=True)
        # Action is array([val])
        return float(np.clip(float(action[0]), 0.0, 1.0))


@dataclass
class EpisodeStats:
    length: int
    reward_sum: float
    collided: bool


def run_episode(env: gym.Env, policy) -> EpisodeStats:
    obs, _ = env.reset()
    done = False
    truncated = False
    length = 0
    reward_sum = 0.0
    collided = False

    while not (done or truncated):
        action = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(np.array([action], dtype=np.float32))
        length += 1
        reward_sum += float(reward)
        # Collision is signaled by terminated=True when GT distance < threshold
        if done:
            collided = True
            # but distinguish normal stopping cases if needed later via info

    return EpisodeStats(length=length, reward_sum=reward_sum, collided=collided)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ego_stack_frames", type=int, default=3)
    ap.add_argument("--savedmodel", default=DEFAULT_SAVEDMODEL)
    ap.add_argument("--sb3_model", default=None, help="Optional: path to SB3 PPO zip to use as baseline instead of SavedModel")
    ap.add_argument("--tflite", default=DEFAULT_TFLITE)
    ap.add_argument("--sumo_cfg", default=None, help="Path to a SUMO scenario.sumocfg")
    ap.add_argument("--scenario", default=None, help="Scenario folder under ml/scenarios (e.g., base_real)")
    ap.add_argument("--emulator_params", default=None, help="Path to emulator params JSON")
    ap.add_argument("--hazard_injection", type=int, default=1, help="1 to enable, 0 to disable")
    ap.add_argument("--cone_half_angle_deg", type=float, default=45.0)
    args = ap.parse_args()

    # Baseline policy selection: SB3 zip (if provided) else SavedModel
    if args.sb3_model:
        keras_policy = SB3Policy(args.sb3_model)
    else:
        keras_policy = KerasPolicy(args.savedmodel)
    tflite_policy = TFLitePolicy(args.tflite)

    keras_stats: list[EpisodeStats] = []
    tflite_stats: list[EpisodeStats] = []

    # Resolve SUMO config path
    sumo_cfg = None
    if args.sumo_cfg:
        sumo_cfg = args.sumo_cfg
    elif args.scenario:
        sumo_cfg = os.path.join(os.path.dirname(__file__), "..", "scenarios", args.scenario, "scenario.sumocfg")
        sumo_cfg = os.path.normpath(sumo_cfg)
        if not os.path.exists(sumo_cfg):
            raise SystemExit(f"Scenario not found: {sumo_cfg}")
    else:
        # Default to base_real
        sumo_cfg = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "scenarios", "base_real", "scenario.sumocfg"))

    for i in range(args.episodes):
        # Baseline (Keras)
        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=sumo_cfg,
            emulator_params_path=args.emulator_params,
            hazard_injection=bool(args.hazard_injection),
            ego_stack_frames=args.ego_stack_frames,
            cone_half_angle_deg=args.cone_half_angle_deg,
        )
        env.reset(seed=args.seed + i)
        ks = run_episode(env, keras_policy)
        env.close()
        keras_stats.append(ks)

        # Candidate (TFLite)
        env = gym.make(
            "RoadSense-Convoy-v0",
            sumo_cfg=sumo_cfg,
            emulator_params_path=args.emulator_params,
            hazard_injection=bool(args.hazard_injection),
            ego_stack_frames=args.ego_stack_frames,
            cone_half_angle_deg=args.cone_half_angle_deg,
        )
        env.reset(seed=args.seed + i)
        ts = run_episode(env, tflite_policy)
        env.close()
        tflite_stats.append(ts)

    def summarize(stats: list[EpisodeStats]) -> Tuple[float, float, float]:
        L = np.array([s.length for s in stats], dtype=np.float32)
        R = np.array([s.reward_sum for s in stats], dtype=np.float32)
        C = np.array([1.0 if s.collided else 0.0 for s in stats], dtype=np.float32)
        return float(L.mean()), float(R.mean()), float(C.mean())

    kL, kR, kC = summarize(keras_stats)
    tL, tR, tC = summarize(tflite_stats)

    print("\n=== A/B Summary (mean over episodes) ===")
    print(f"Episodes: {args.episodes}")
    print(f"SUMO: {sumo_cfg}")
    print(f"Emulator: {args.emulator_params or 'default'}  Hazards: {'on' if args.hazard_injection else 'off'}  Cone: {args.cone_half_angle_deg} deg")
    print(f"Keras    → len={kL:.1f}  reward={kR:.3f}  collision_rate={kC:.2%}")
    print(f"TFLite   → len={tL:.1f}  reward={tR:.3f}  collision_rate={tC:.2%}")
    print(f"Deltas   → len={tL-kL:+.1f}  reward={tR-kR:+.3f}  collision_rate={tC-kC:+.2%}")


if __name__ == "__main__":
    main()
