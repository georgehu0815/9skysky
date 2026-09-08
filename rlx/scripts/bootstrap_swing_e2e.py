"""Bounded teacher-assisted BC/DAgger initialization, never PPO training."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT.parent / "docs/remaining-scenarios-e2e/recipes/swing-teacher.json"
DEFAULT_OUTPUT = ROOT / "artifacts/scenarios-e2e-20260907/swing-bootstrap"
STARTS = ((0.0, 0.0), (2.0, 0.05), (5.0, 0.1), (10.0, 0.2))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def teacher_action(parameters: np.ndarray, angle: float, rate: float) -> np.ndarray:
    phase = math.tanh(float(parameters[0] + parameters[1] * angle + parameters[2] * rate))
    sagittal = np.clip(parameters[3:8] + parameters[8:13] * phase, -1.0, 1.0)
    action = np.zeros(14, dtype=np.float32)
    action[2:7] = sagittal
    action[11:14] = -sagittal[:3]
    return action


def label_observation(env, parameters: np.ndarray) -> np.ndarray:
    state = env.unwrapped._swing_state()
    return teacher_action(parameters, state["angle"], state["angle_rate"])


class ObservationActor:
    """Inference accepts observations only; no environment or teacher reference."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def batch(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 61 or not np.isfinite(values).all():
            raise ValueError("actor requires finite [batch,61] observations")
        actions = self.session.run([self.output_name], {self.input_name: values})[0]
        if actions.shape != (len(values), 14) or not np.isfinite(actions).all():
            raise ValueError("actor returned invalid actions")
        return actions

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return self.batch(observation[None, :])[0]


def make_env(angle: float = 0.0, rate: float = 0.0):
    from rlx.environments.microduck_recipes import make_single_recipe_env

    return make_single_recipe_env(
        "swing", actuator="xml", domain_rand=False, obs_noise=False,
        action_delay=False, random_yaw=False, max_episode_s=24.0,
        swing_initial_angle_deg=angle, swing_initial_rate_rad_s=rate,
        swing_planar_actions=True,
    )


def collect(parameters, actor, *, seed, remaining, deadline, episodes=8):
    observations, labels, provenance = [], [], []
    for episode in range(episodes):
        if len(labels) >= remaining or time.monotonic() >= deadline:
            break
        angle, rate = STARTS[episode % len(STARTS)]
        env = make_env(angle, rate)
        episode_start = len(labels)
        try:
            observation, _ = env.reset(seed=seed + episode)
            for _ in range(1200):
                if len(labels) >= remaining or time.monotonic() >= deadline:
                    break
                label = label_observation(env, parameters)
                if not np.isfinite(observation).all():
                    raise ValueError("nonfinite demonstration observation")
                observations.append(observation.copy())
                labels.append(label)
                action = label if actor is None else actor(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            provenance.append({
                "seed": seed + episode, "angle_bound_deg": angle,
                "rate_bound_rad_s": rate, "samples": len(labels) - episode_start,
                "behavior": "teacher" if actor is None else "learned_61d_onnx",
                "labels": "privileged teacher at pre-action learner state",
            })
        finally:
            env.close()
    return np.asarray(observations, np.float32), np.asarray(labels, np.float32), provenance


def evaluate_actor(actor, *, seeds=(1701, 1702)):
    from rlx.environments.swing_evaluation import SwingEvaluation, SwingEvaluationPlan

    evaluator = SwingEvaluation(len(seeds), SwingEvaluationPlan())
    traces, initial_states = [], []
    for lane, seed in enumerate(seeds):
        env = make_env()
        try:
            observation, _ = env.reset(seed=seed)
            initial_states.append({"seed": seed, **env.unwrapped.recipe_metrics()})
            for step in range(1200):
                action = actor(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                metrics = info["recipe_metrics"]
                evaluator.observe(lane, metrics, terminated=terminated, truncated=truncated)
                traces.append({"lane": lane, "step": step + 1, "reward": float(reward), **metrics})
                if terminated or truncated:
                    break
        finally:
            env.close()
    report = evaluator.report()
    report.update({
        "initial_states": initial_states, "seeds": list(seeds),
        "input": "61D observations only", "policy_kind": "teacher-assisted BC/DAgger, not PPO",
        "seed_limit": "Nominal deterministic resets are identical; multiple seeds do not prove robustness.",
    })
    return report, traces


def selection_key(report: dict) -> tuple:
    episodes = report["episodes"]
    return (
        int(report["passed"]),
        min(episode["valid_geometry_fraction"] for episode in episodes),
        min(episode["both_strings_tensioned_fraction"] for episode in episodes),
        min(episode["bidirectional_span_deg"] for episode in episodes),
    )


def observation_statistics(observations: np.ndarray):
    floors = np.asarray([0.03] * 3 + [0.01] * 3 + [0.01] * 14 + [0.25] * 14
                        + [0.05] * 14 + [0.01] * 13, np.float32)
    return observations.mean(axis=0), np.maximum(observations.var(axis=0), floors ** 2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--train-seconds", type=float, default=280.0)
    parser.add_argument("--total-seconds", type=float, default=570.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args(argv)
    if not 0 <= args.iterations <= 5 or not 1200 <= args.max_samples <= 60000:
        parser.error("requires 0..5 DAgger iterations and 1200..60000 samples")
    if not 0 < args.train_seconds <= 300 or not 0 < args.total_seconds <= 600:
        parser.error("training <=300s and combined work <=600s required")
    if args.epochs < 1 or not math.isfinite(args.train_seconds + args.total_seconds):
        parser.error("positive epochs and finite budgets required")
    return args


def main(argv=None):
    args = parse_args(argv)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from rlx.export.microduck_onnx import export_deterministic_actor
    from rlx.models.microduck import create_actor_critic, save_checkpoint

    mx.eval(mx.zeros(1))
    started = time.monotonic()
    deadline = started + args.total_seconds
    source_hash = digest(args.source)
    evidence = json.loads(args.source.read_text())
    parameters = np.asarray(evidence["search"]["parameters"]["best"], dtype=np.float64)
    if parameters.shape != (13,) or not np.isfinite(parameters).all():
        raise ValueError("expected 13 finite privileged-controller parameters")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifacts: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = create_actor_critic(initial_std=0.1)
    optimizer = optim.Adam(learning_rate=3e-4)
    loss_grad = nn.value_and_grad(
        model.actor_mean, lambda network, inputs, targets: mx.mean((network(inputs) - targets) ** 2)
    )
    observations, labels, batches = collect(
        parameters, None, seed=args.seed, remaining=args.max_samples, deadline=deadline - 30
    )
    if not len(labels):
        raise RuntimeError("no demonstrations collected within budget")
    mean, variance = observation_statistics(observations)
    normalizer_count = len(labels)
    history, train_seconds, best = [], 0.0, None
    for stage in range(args.iterations + 1):
        if stage:
            if len(labels) >= args.max_samples or time.monotonic() >= deadline - 30:
                break
            new_obs, new_labels, new_batches = collect(
                parameters, actor, seed=args.seed + stage * 100,
                remaining=args.max_samples - len(labels), deadline=deadline - 30,
            )
            if not len(new_labels):
                break
            observations = np.concatenate((observations, new_obs))
            labels = np.concatenate((labels, new_labels))
            batches.extend(new_batches)
        normalized = np.clip((observations - mean) / np.sqrt(variance + 1e-8), -10, 10)
        inputs, targets = mx.array(normalized), mx.array(labels)
        mx.eval(inputs, targets)
        stage_budget = max(0, args.train_seconds - train_seconds) / (args.iterations + 1 - stage)
        train_start = time.monotonic()
        train_deadline = min(train_start + stage_budget, deadline - 30)
        losses, updates, completed_epochs = [], 0, 0
        for epoch in range(args.epochs):
            if time.monotonic() >= train_deadline:
                break
            permutation = rng.permutation(len(labels))
            for offset in range(0, len(labels), 512):
                if time.monotonic() >= train_deadline:
                    break
                indices = mx.array(permutation[offset:offset + 512].astype(np.int32))
                loss, gradients = loss_grad(model.actor_mean, inputs[indices], targets[indices])
                optimizer.update(model.actor_mean, gradients)
                mx.eval(model.parameters(), optimizer.state, loss)
                scalar = float(np.asarray(loss))
                if not math.isfinite(scalar):
                    raise ValueError("nonfinite BC loss")
                losses.append(scalar)
                updates += 1
            completed_epochs = epoch + 1
        duration = time.monotonic() - train_start
        train_seconds += duration
        stage_dir = args.output / f"stage-{stage:02d}"
        stage_dir.mkdir()
        checkpoint = stage_dir / "swing.safetensors"
        metadata = {
            "recipe": "swing", "steps": 0, "ppo_steps": 0, "teacher_assisted": True,
            "bootstrap": "BC + on-policy DAgger teacher labeling; no PPO updates",
            "dagger_iteration": stage, "teacher_samples": len(labels),
            "teacher_source": str(args.source.resolve()), "teacher_source_sha256": source_hash,
            "teacher_parameters": parameters.tolist(), "critic": "random, not pretrained",
            "deployment_input": "standard 61D observation; no privileged inference state",
            "normalizer": "frozen initial teacher dataset with sensor-scale variance floors",
            "actuator": "xml", "max_episode_s": 24.0,
            "recipe_options": {"swing_planar_actions": True, "swing_initial_angle_deg": 0,
                               "swing_initial_rate_rad_s": 0},
            "randomization": dict(domain_rand=False, obs_noise=False, action_delay=False, random_yaw=False),
            "initial_std": 0.1, "reward_weights": {},
        }
        save_checkpoint(checkpoint, model, mean, variance, normalizer_count,
                        return_mean=np.array(0.), return_variance=np.array(1.), return_count=1e-4,
                        metadata=metadata)
        onnx = export_deterministic_actor(checkpoint, checkpoint.with_suffix(".onnx"))
        actor = ObservationActor(onnx)
        parity_inputs = observations[rng.choice(len(labels), min(256, len(labels)), replace=False)]
        expected = np.clip(np.asarray(model.deterministic(mx.array(np.clip(
            (parity_inputs - mean) / np.sqrt(variance + 1e-8), -10, 10)))), -1, 1)
        actual = actor.batch(parity_inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        evaluation, traces = evaluate_actor(actor)
        write_json(stage_dir / "strict-evaluation.json", evaluation)
        write_json(stage_dir / "strict-trace.json", {"samples": traces})
        row = {
            "stage": stage, "samples": len(labels), "epochs_started": completed_epochs,
            "optimizer_updates": updates, "train_seconds": duration,
            "mean_bc_loss": float(np.mean(losses)) if losses else None,
            "last_bc_loss": losses[-1] if losses else None,
            "onnx_parity_max_error": float(np.max(np.abs(actual - expected))),
            "strict_passed": evaluation["passed"], "selection_key": selection_key(evaluation),
            "strict_episodes": evaluation["episodes"],
        }
        history.append(row)
        write_json(stage_dir / "training.json", row)
        if best is None or selection_key(evaluation) > best[0]:
            best = (selection_key(evaluation), stage_dir, stage)
        print(json.dumps(row, allow_nan=False), flush=True)
        write_json(args.output / "progress.json", {"stages": history, "best_stage": best[2]})
        if time.monotonic() >= deadline - 30 or train_seconds >= args.train_seconds:
            break
    for name in ("swing.safetensors", "swing.safetensors.json", "swing.onnx", "strict-evaluation.json"):
        shutil.copyfile(best[1] / name, args.output / name)
    np.savez_compressed(args.output / "demonstrations.npz", observations=observations, labels=labels)
    source_unchanged = digest(args.source) == source_hash
    if not source_unchanged:
        raise RuntimeError("teacher evidence changed during bootstrap")
    write_json(args.output / "provenance.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "script_sha256": digest(Path(__file__)), "teacher_source_sha256": source_hash,
        "source_unchanged": source_unchanged, "teacher_assisted": True, "ppo_steps": 0,
        "selected_stage": best[2], "selection": "strict pass, geometry, tension, symmetric span",
        "stages": history, "collection_batches": batches, "samples": len(labels),
        "training_seconds": train_seconds, "combined_seconds": time.monotonic() - started,
        "limits": dict(iterations=args.iterations, samples=args.max_samples,
                       training_seconds=args.train_seconds, combined_seconds=args.total_seconds),
        "normalizer_frozen_sample_count": normalizer_count,
        "artifacts_sha256": {name: digest(args.output / name) for name in (
            "swing.safetensors", "swing.safetensors.json", "swing.onnx", "demonstrations.npz")},
        "limitations": ["XML simulation only", "No PPO learning claim", "Critic is untrained",
                        "Evaluation seeds share identical nominal reset dynamics",
                        "50 Hz tension sampling is not continuous-time or hardware safety"],
    })
    print(json.dumps({"output": str(args.output), "selected_stage": best[2],
                      "strict_passed": bool(best[0][0]), "samples": len(labels)}), flush=True)


if __name__ == "__main__":
    main()
