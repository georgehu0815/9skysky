"""Evaluate saved PPO policies against controls and produce reproducible evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def studio_module():
    spec = importlib.util.spec_from_file_location("dance_audit_studio", ROOT / "examples/ppo_microduck_studio.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rollout(infer, reference, seconds, weights, seed, *, freeze_phase=False, dance_pose_sigma=None):
    from rlx.environments.dance_evaluation import DanceEvaluation
    from rlx.environments.microduck_recipes import make_single_recipe_env

    env = make_single_recipe_env(
        "dance", dance_clip=reference, seed=seed, domain_rand=False,
        obs_noise=False, action_delay=False, random_yaw=False,
        max_episode_s=seconds, weight_overrides=weights,
        dance_pose_sigma=dance_pose_sigma,
    )
    assessor = DanceEvaluation(1, round(seconds * 50))
    trace = {name: [] for name in ("observations", "actions", "rewards", "current", "target", "qpos", "upright", "height")}
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(round(seconds * 50)):
            policy_observation = observation.copy()
            if freeze_phase:
                policy_observation[59:61] = [0, 1]
            action = np.asarray(infer(policy_observation), dtype=np.float32)
            trace["observations"].append(observation.copy())
            observation, reward, terminated, truncated, info = env.step(action)
            state = info["dance_state"]
            assessor.observe(0, state, terminated, truncated)
            trace["actions"].append(action)
            trace["rewards"].append(reward)
            trace["current"].append(state["current_joints"])
            trace["target"].append(state["target_joints"])
            trace["qpos"].append(env.unwrapped.data.qpos.copy())
            trace["upright"].append(state["upright"])
            trace["height"].append(state["height_m"])
            if terminated or truncated:
                break
    finally:
        env.close()
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    report = assessor.report()
    report.update(seed=seed, raw_return=float(arrays["rewards"].sum()),
                  reward_per_step=float(arrays["rewards"].mean()),
                  finite=all(bool(np.isfinite(values).all()) for values in arrays.values()))
    return report, arrays


def plots(run, output, history, final_trace, zero_trace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from microduck_local.contract import JOINT_NAMES

    events = [json.loads(line) for line in (run / "training-metrics.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["phase"] == "update"]
    collections = [event for event in events if event["phase"] == "collection"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot([entry["steps"] for entry in history], [entry["raw_return"] for entry in history], "o-", label="Deterministic checkpoint, held-out seed 101")
    axes[0, 0].set(title="Raw evaluation return (not normalized)", ylabel="8-second return; early falls end rollout")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot([entry["steps"] for entry in history], [entry["episodes"][0]["pose_rmse_rad"] for entry in history], "o-", label="Pose RMSE")
    axes[0, 1].axhline(.15, color="red", linestyle="--", label="Predeclared 0.15-rad ceiling")
    axes[0, 1].set(title="Tracking accuracy", ylabel="Joint RMSE, radians")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot([event["env_steps"] for event in collections], [event["mean_reward"] for event in collections], linewidth=.8)
    axes[1, 0].set(title="Training reward after running-return normalization", ylabel="Mean reward per transition (scale changes)")
    episodes = [event for event in events if event["phase"] == "episodes"]
    if episodes:
        axes[1, 1].plot([event["env_steps"] for event in episodes], [event["mean_raw_return"] for event in episodes], alpha=.35, linewidth=.6)
        axes[1, 1].set(title="Raw stochastic training episode return", ylabel="Unsmoothed completed-episode return")
    else:
        axes[1, 1].plot([entry["steps"] for entry in history], [entry["episodes"][0]["dynamic_gain"] for entry in history], "o-")
        axes[1, 1].axhline(.2, color="red", linestyle="--")
        axes[1, 1].set(title="Dynamic tracking gain vs constant reference mean", ylabel="1 − tracking RMSE / static-baseline RMSE")
    for axis in axes.flat:
        axis.set_xlabel("PPO environment transitions")
        axis.grid(alpha=.2)
    figure.suptitle("Dance PPO: reward is evidence of optimization, not proof of choreography")
    figure.savefig(output / "reward-learning.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, key, title in zip(axes.flat,
        ("mean_loss", "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction"),
        ("Weighted PPO objective", "Clipped policy loss", "Clipped Huber critic loss", "Gaussian differential entropy", "Approximate KL", "Clip fraction")):
        axis.plot([event["env_steps"] for event in updates], [event[key] for event in updates], linewidth=.8)
        axis.set(title=title, xlabel="Environment transitions", ylabel=key)
        axis.grid(alpha=.2)
    figure.suptitle("Actual minibatch metrics; no smoothing or invented monotonic loss")
    figure.savefig(output / "ppo-losses.png", dpi=170)
    plt.close(figure)
    later_updates = [event for event in updates if event["env_steps"] >= 100000]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, key in zip(axes.flat, ("approximate_kl", "clip_fraction", "explained_variance", "value_loss")):
        axis.plot([event["env_steps"] for event in later_updates], [event[key] for event in later_updates], linewidth=.8)
        axis.set(title=key, xlabel="Environment transitions")
        axis.grid(alpha=.2)
    figure.suptitle("Post-100k detail — startup spikes remain visible in the full-history loss figure")
    figure.savefig(output / "ppo-stability.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for axis, joint in zip(axes.flat, (1, 3, 4, 7, 10, 11, 12, 13)):
        times = (np.arange(len(final_trace["current"])) + 1) / 50
        axis.plot(times, final_trace["target"][:, joint], label="Authored reference", color="black", linestyle="--")
        axis.plot(times, final_trace["current"][:, joint], label="PPO ONNX", color="#126daa")
        axis.plot((np.arange(len(zero_trace["current"])) + 1) / 50, zero_trace["current"][:, joint], label="Zero-action control", color="#bd6338", alpha=.7)
        axis.set(title=JOINT_NAMES[joint], xlabel="Seconds", ylabel="Radians")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Time-aligned physical joint tracking — no fitted lag or phase alignment")
    figure.savefig(output / "joint-tracking.png", dpi=170)
    plt.close(figure)


def comparison_video(reference, seconds, output, trained, zero):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    from microduck_local.render_rollout import build_sheet, sheet_indices
    from rlx.environments.microduck_recipes import make_single_recipe_env

    env = make_single_recipe_env("dance", dance_clip=reference, domain_rand=False, obs_noise=False, action_delay=False, random_yaw=False)
    env.reset(seed=101)
    model = env.unwrapped.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = studio_module()._camera("dance", "three-quarter", .72)
    stand = env.unwrapped.data.qpos.copy()
    frames = []
    titles = ("Authored target (KINEMATIC, not learned)", "PPO policy (ONNX, physical simulation)", "Zero action (physical null control)")
    try:
        with imageio.get_writer(output / "comparison.mp4", fps=25, codec="libx264", macro_block_size=None) as writer:
            for step in range(0, round(seconds * 50), 2):
                panels = []
                for index, trace in enumerate((None, trained, zero)):
                    if trace is None:
                        data.qpos[:] = stand
                        target, _ = env.unwrapped.clip.at(step + 1)
                        data.qpos[env.unwrapped.joint_qpos_adr] = target
                    else:
                        data.qpos[:] = trace["qpos"][min(step, len(trace["qpos"]) - 1)]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.xpos[env.unwrapped.trunk_body_id]
                    renderer.update_scene(data, camera=camera)
                    image = Image.new("RGB", (480, 412), "#101c2c")
                    image.paste(Image.fromarray(renderer.render().copy()), (0, 52))
                    draw = ImageDraw.Draw(image)
                    draw.text((10, 8), titles[index], fill="white")
                    note = f"t={(step + 1) / 50:.2f}s"
                    if trace is not None:
                        measured = min(step, len(trace["current"]) - 1)
                        error = np.sqrt(np.mean((trace["current"][measured] - trace["target"][measured]) ** 2))
                        note += f" | joint RMSE {error:.3f} rad"
                        if step >= len(trace["current"]):
                            note += " | TERMINATED (last frame)"
                    draw.text((10, 28), note, fill="#c3d7f0")
                    panels.append(np.asarray(image))
                frame = np.concatenate(panels, axis=1)
                writer.append_data(frame)
                frames.append(frame)
        selected = sheet_indices(len(frames), 8)
        build_sheet([frames[index] for index in selected], [[f"t={(index * 2 + 1) / 50:.2f}s"] for index in selected],
                    ["Bachata excerpt: reference vs PPO vs zero-action control", "Reference is kinematic visualization only; never applied to policy simulation."],
                    [], [False] * len(selected), output / "comparison_sheet.png")
    finally:
        renderer.close()
        env.close()


def audit(args):
    import mlx.core as mx
    import onnx
    from rlx.export.microduck_onnx import export_deterministic_actor

    args.output.mkdir(parents=True, exist_ok=True)
    studio = studio_module()
    recipe = json.loads(args.recipe.read_text())
    reference = Path(recipe["danceClip"])
    seconds = recipe["maxEpisodeS"]
    weights = recipe["rewardWeights"]
    final = args.run / "dance.onnx"
    if not final.is_file():
        export_deterministic_actor(args.run / "dance.safetensors", final)
    infer = studio._render_policy_source(final, "policy")
    reports = {}
    traces = {}
    for name, policy, frozen in (("trained", infer, False), ("zero", lambda observation: np.zeros(14, np.float32), False), ("frozen_phase", infer, True)):
        reports[name] = []
        for seed in args.seeds:
            report, trace = rollout(policy, reference, seconds, weights, seed, freeze_phase=frozen, dance_pose_sigma=recipe.get("dancePoseSigma"))
            reports[name].append(report)
            np.savez_compressed(args.output / f"{name}-seed{seed}.npz", **trace)
            if seed == args.seeds[0]:
                traces[name] = trace
    history = []
    checkpoints = sorted((args.run / "checkpoints").glob("*.safetensors"))
    selected = checkpoints[::args.history_stride]
    final_steps = json.loads((args.run / "dance.safetensors.json").read_text())["metadata"]["steps"]
    selected = [checkpoint for checkpoint in selected if json.loads(checkpoint.with_suffix(".safetensors.json").read_text())["metadata"]["steps"] < final_steps]
    selected = [*selected, args.run / "dance.safetensors"]
    for checkpoint in selected:
        metadata = json.loads(checkpoint.with_suffix(".safetensors.json").read_text())
        policy = studio._render_policy_source(checkpoint, "checkpoint")
        report, trace = rollout(policy, reference, seconds, weights, args.seeds[0], dance_pose_sigma=recipe.get("dancePoseSigma"))
        report.update(steps=metadata["metadata"]["steps"], checkpoint=str(checkpoint))
        history.append(report)
        if report["steps"] == 0:
            reports["untrained"] = [report]
            np.savez_compressed(args.output / "untrained.npz", **trace)
    observations = traces["trained"]["observations"].astype(np.float32)
    native = np.asarray(studio._policy_from_checkpoint(args.run / "dance.safetensors")(observations))
    onnx_actions = np.asarray(studio._policy_from_onnx(final)(observations))
    max_error = float(np.max(np.abs(native - onnx_actions)))
    initial = args.initial_checkpoint or checkpoints[0]
    original_weights = mx.load(str(initial))
    trained_weights = mx.load(str(args.run / "dance.safetensors"))
    parameter_changes = [np.asarray(trained_weights[name]) - np.asarray(value)
                         for name, value in original_weights.items() if name.startswith("model.")]
    parameter_l2_change = float(np.sqrt(sum(float(np.square(change).sum()) for change in parameter_changes)))
    onnx_model = onnx.load(str(final))
    onnx.checker.check_model(onnx_model)
    input_dims = [dimension.dim_value or dimension.dim_param for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    output_dims = [dimension.dim_value or dimension.dim_param for dimension in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    contract_passed = input_dims == ["batch", 61] and output_dims == ["batch", 14]
    events = [json.loads(line) for line in (args.run / "training-metrics.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["phase"] == "update"]
    collections = [event for event in events if event["phase"] == "collection"]
    training_finite = bool(updates) and all(np.isfinite([value for value in event.values() if isinstance(value, (int, float))]).all() for event in updates)
    observed_steps = sum(event["steps"] for event in collections)
    training_evidence = {"finite": training_finite, "updates": len(updates),
                         "optimizer_steps": sum(event["optimizer_steps"] for event in updates),
                         "observed_transitions": observed_steps,
                         "collection_seconds": sum(event["seconds"] for event in collections),
                         "update_seconds": sum(event["seconds"] for event in updates),
                         "last_update": updates[-1] if updates else None}
    result = {"reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "policy_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
              "recipe": recipe, "controls": reports, "history": history,
              "onnx_parity_max_abs_error": max_error, "onnx_parity_passed": max_error < 1e-4,
              "onnx_contract": {"input": input_dims, "output": output_dims, "passed": contract_passed},
              "initial_checkpoint": str(initial), "parameter_l2_change": parameter_l2_change,
              "training": training_evidence,
              "passed": all(report["passed"] and report["finite"] for report in reports["trained"]) and max_error < 1e-4 and contract_passed and parameter_l2_change > 0 and training_finite and observed_steps >= recipe["totalTimesteps"]}
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plots(args.run, args.output, history, traces["trained"], traces["zero"])
    if args.render:
        comparison_video(reference, seconds, args.output, traces["trained"], traces["zero"])
    print(json.dumps({"passed": result["passed"], "onnx_parity": max_error, "output": str(args.output)}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--history-stride", type=int, default=2)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.history_stride < 1 or not args.seeds:
        parser.error("positive history stride and at least one evaluation seed are required")
    if not audit(args)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
