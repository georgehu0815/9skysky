"""Audit native Studio locomotion/Swing PPO checkpoints and retain physical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_dance import studio_module


def environment_options(recipe):
    return {
        "actuator": "xml",
        "domain_rand": False, "obs_noise": False, "action_delay": False,
        "random_yaw": False, "max_episode_s": recipe["maxEpisodeS"],
        "weight_overrides": recipe.get("rewardWeights", {}),
        "stilt_height_cm": recipe.get("stiltHeightCm", 2),
        "stilt_blend": recipe.get("stiltBlend", 0),
        "stilt_mass_kg": recipe.get("stiltMassKg"),
        "swing_initial_angle_deg": 0, "swing_initial_rate_rad_s": 0,
        "swing_planar_actions": recipe.get("swingPlanarActions", False),
        "locomotion_forward_command": recipe.get("locomotionForwardCommand"),
    }


def rollout(infer, recipe, seed):
    from rlx.environments.microduck_recipes import make_single_recipe_env
    from rlx.environments.swing_evaluation import SwingEvaluation, SwingEvaluationPlan
    from rlx.environments.locomotion_evaluation import LocomotionEvaluation

    scenario = recipe["experimentId"]
    steps = round(recipe["maxEpisodeS"] * 50)
    assessor = (
        SwingEvaluation(1, SwingEvaluationPlan(required_steps=max(1200, steps)))
        if scenario == "swing" else LocomotionEvaluation(scenario, 1, steps)
    )
    env = make_single_recipe_env(scenario, seed=seed, **environment_options(recipe))
    trace = {name: [] for name in ("observations", "actions", "rewards", "qpos")}
    measurements = {}
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(steps):
            action = np.asarray(infer(observation), dtype=np.float32)
            trace["observations"].append(observation.copy())
            observation, reward, terminated, truncated, info = env.step(action)
            measured = info["recipe_metrics"]
            assessor.observe(0, measured, terminated=terminated, truncated=truncated)
            trace["actions"].append(action)
            trace["rewards"].append(reward)
            trace["qpos"].append(env.unwrapped.data.qpos.copy())
            for name, value in measured.items():
                measurements.setdefault(name, []).append(value)
            if terminated or truncated:
                break
    finally:
        env.close()
    arrays = {name: np.asarray(values) for name, values in {**trace, **measurements}.items()}
    report = assessor.report()
    report.update(seed=seed, raw_return=float(arrays["rewards"].sum()),
                  steps=len(arrays["rewards"]),
                  finite=all(bool(np.isfinite(values).all()) for values in arrays.values()),
                  metric_means={name: float(np.mean(values)) for name, values in measurements.items()})
    return report, arrays


def training_evidence(run, requested_steps):
    events = [json.loads(line) for line in (run / "training-metrics.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["phase"] == "update"]
    collections = [event for event in events if event["phase"] == "collection"]
    required = ("mean_loss", "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction", "explained_variance")
    finite = bool(updates) and all(
        all(key in event and np.isfinite(event[key]) for key in required) for event in updates
    )
    steps = sum(event["steps"] for event in collections)
    monotonic = all(current["env_steps"] > previous["env_steps"] for previous, current in zip(updates, updates[1:]))
    summary = {
        "finite": bool(finite), "updates": len(updates),
        "optimizer_steps": sum(event["optimizer_steps"] for event in updates),
        "observed_transitions": steps, "requested_transitions": requested_steps,
        "monotonic_update_steps": monotonic,
        "passed": bool(finite and monotonic and steps >= requested_steps),
        "last_100_means": {
            key: float(np.mean([event[key] for event in updates[-100:]]))
            if updates and all(key in event and np.isfinite(event[key]) for event in updates[-100:]) else None
            for key in required
        },
        "max_approximate_kl": max((event["approximate_kl"] for event in updates
                                   if "approximate_kl" in event and np.isfinite(event["approximate_kl"])), default=None),
    }
    return events, summary


def plots(output, scenario, events, history, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = [event for event in events if event["phase"] == "update"]
    collections = [event for event in events if event["phase"] == "collection"]
    episodes = [event for event in events if event["phase"] == "episodes"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot([entry["training_steps"] for entry in history], [entry["raw_return"] for entry in history], "o-")
    axes[0, 0].set(title="Deterministic checkpoint raw return", ylabel="Unnormalized return; falls end episodes")
    axes[0, 1].plot([entry["training_steps"] for entry in history], [entry["steps"] / 50 for entry in history], "o-")
    axes[0, 1].set(title="Deterministic survival", ylabel="Seconds")
    axes[1, 0].plot([event["env_steps"] for event in collections], [event["mean_reward"] for event in collections], linewidth=.7)
    axes[1, 0].set(title="Normalized stochastic collection reward", ylabel="Mean per transition; normalization changes")
    axes[1, 1].plot([event["env_steps"] for event in episodes], [event["mean_raw_return"] for event in episodes], linewidth=.7)
    axes[1, 1].set(title="Raw stochastic episode return", ylabel="Unsmoothed mean completed-episode return")
    for axis in axes.flat:
        axis.set_xlabel("PPO environment transitions")
        axis.grid(alpha=.2)
    figure.suptitle(f"{scenario}: optimization evidence, not a substitute for skill acceptance")
    figure.savefig(output / "reward-learning.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    keys = ("mean_loss", "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction", "explained_variance", "seconds")
    for axis, key in zip(axes.flat, keys):
        axis.plot([event["env_steps"] for event in updates], [event[key] for event in updates], linewidth=.7)
        axis.set(title=key, xlabel="PPO environment transitions")
        axis.grid(alpha=.2)
    figure.suptitle("Actual PPO minibatch aggregates; full history including startup spikes")
    figure.savefig(output / "ppo-losses.png", dpi=160)
    plt.close(figure)
    metric_keys = (
        ("swing_angle_deg", "string_left_tension_n", "string_right_tension_n", "lateral_offset_m")
        if scenario == "swing" else
        ("forward_speed_m_s", "upright", "left_foot_contact", "right_foot_contact")
    )
    columns = 2 if scenario == "swing" else 3
    figure, axes = plt.subplots(2, columns, figsize=(12, 8), constrained_layout=True)
    for axis, key in zip(axes.flat, metric_keys):
        for label, trace in traces.items():
            if key in trace:
                axis.plot((np.arange(len(trace[key])) + 1) / 50, trace[key], label=label, linewidth=.8)
        axis.set(title=key, xlabel="Seconds")
        axis.grid(alpha=.2)
        axis.legend()
    if scenario != "swing":
        path_axis, air_axis = list(axes.flat)[4:]
        for label, trace in traces.items():
            path_axis.plot(trace["world_x_m"], trace["world_y_m"], label=label)
            aerial = (trace["left_foot_contact"] == 0) & (trace["right_foot_contact"] == 0)
            air_axis.plot((np.arange(len(aerial)) + 1) / 50, aerial.astype(float), label=label, linewidth=.8)
        path_axis.set(title="World path (circling is not forward progress)", xlabel="World x (m)", ylabel="World y (m)", aspect="equal")
        air_axis.set(title="Both feet airborne", xlabel="Seconds", ylabel="Airborne indicator")
        for axis in (path_axis, air_axis):
            axis.grid(alpha=.2)
            axis.legend()
    figure.suptitle(f"{scenario}: held-out seed physical measurements, PPO versus zero action")
    figure.savefig(output / "physical-tracking.png", dpi=160)
    plt.close(figure)


def comparison_video(output, recipe, traces, reports):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    from microduck_local.render_rollout import build_sheet, sheet_indices
    from rlx.environments.microduck_recipes import make_single_recipe_env

    scenario = recipe["experimentId"]
    env = make_single_recipe_env(scenario, **environment_options(recipe))
    env.reset(seed=101)
    model = env.unwrapped.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = studio_module()._camera(scenario, "three-quarter", 1.3 if scenario == "swing" else .8)
    frames = []
    try:
        with imageio.get_writer(output / "comparison.mp4", fps=25, codec="libx264", macro_block_size=None) as writer:
            for step in range(0, round(recipe["maxEpisodeS"] * 50), 2):
                panels = []
                for name, trace in traces.items():
                    position = min(step, len(trace["qpos"]) - 1)
                    data.qpos[:] = trace["qpos"][position]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.xpos[env.unwrapped.trunk_body_id]
                    renderer.update_scene(data, camera=camera)
                    image = Image.new("RGB", (480, 412), "#101c2c")
                    image.paste(Image.fromarray(renderer.render().copy()), (0, 52))
                    draw = ImageDraw.Draw(image)
                    verdict = "PASS" if reports[name][0]["passed"] else "FAIL"
                    draw.text((10, 8), f"{scenario}: {name} | skill {verdict} | simulation", fill="white")
                    note = f"t={(step + 1) / 50:.2f}s"
                    if step >= len(trace["qpos"]):
                        note += " | TERMINATED; last frame held, no reset"
                    draw.text((10, 28), note, fill="#c3d7f0")
                    panels.append(np.asarray(image))
                frame = np.concatenate(panels, axis=1)
                writer.append_data(frame)
                frames.append(frame)
        selected = sheet_indices(len(frames), 8)
        build_sheet([frames[index] for index in selected],
                    [[f"t={(index * 2 + 1) / 50:.2f}s"] for index in selected],
                    [f"{scenario}: trained ONNX versus zero action", "No resets, assistance, or kinematic policy replacement."],
                    [], [False] * len(selected), output / "comparison_sheet.png")
    finally:
        renderer.close()
        env.close()


def audit(args):
    import mlx.core as mx
    import onnx
    from rlx.export.microduck_onnx import parity_error

    recipe = json.loads(args.recipe_json.read_text())
    scenario = recipe["experimentId"]
    args.output.mkdir(parents=True, exist_ok=True)
    studio = studio_module()
    checkpoint = args.run / f"{scenario}.safetensors"
    policy = args.run / f"{scenario}.onnx"
    reports, traces = {}, {}
    for name, infer in (("trained", studio._render_policy_source(policy, "policy")),
                        ("zero", lambda observation: np.zeros(14, np.float32))):
        reports[name] = []
        for seed in args.seeds:
            report, trace = rollout(infer, recipe, seed)
            reports[name].append(report)
            np.savez_compressed(args.output / f"{name}-seed{seed}.npz", **trace)
            if seed == args.seeds[0]:
                traces[name] = trace
    checkpoints = sorted((args.run / "checkpoints").glob("*.safetensors"))
    initial = checkpoints[0]
    initial_metadata = json.loads(initial.with_suffix(".safetensors.json").read_text())["metadata"]
    prior_steps = initial_metadata["steps"]
    initialization = initial_metadata.get("initialization", {"kind": "random"})
    pretrained = initialization.get("kind") == "checkpoint"
    if pretrained:
        from rlx.models.microduck import create_actor_critic, load_checkpoint, normalize_observations

        loaded_initial = load_checkpoint(initial)
        mx.random.seed(recipe.get("seed", 7))
        random_model = create_actor_critic(initial_std=recipe.get("initialStd", .1))

        def random_infer(observation):
            normalized = normalize_observations(
                mx.array(observation[None], dtype=mx.float32),
                loaded_initial["mean"], loaded_initial["variance"],
                epsilon=loaded_initial["epsilon"], clip=loaded_initial["clip"],
            )
            return np.asarray(random_model.deterministic(normalized)[0])

        reports["random"] = []
        for seed in args.seeds:
            report, trace = rollout(random_infer, recipe, seed)
            reports["random"].append(report)
            np.savez_compressed(args.output / f"random-seed{seed}.npz", **trace)
    final_steps = json.loads(checkpoint.with_suffix(".safetensors.json").read_text())["metadata"]["steps"]
    selected = [entry for entry in checkpoints[::args.history_stride]
                if json.loads(entry.with_suffix(".safetensors.json").read_text())["metadata"]["steps"] < final_steps]
    history = []
    for entry in [*selected, checkpoint]:
        infer = studio._render_policy_source(entry, "checkpoint")
        report, trace = rollout(infer, recipe, args.seeds[0])
        report.update(training_steps=json.loads(entry.with_suffix(".safetensors.json").read_text())["metadata"]["steps"] - prior_steps, checkpoint=str(entry))
        history.append(report)
        if entry == initial:
            reports["initial"] = [report]
            np.savez_compressed(args.output / "initial.npz", **trace)
    observations = traces["trained"]["observations"].astype(np.float32)
    policy_error = parity_error(checkpoint, policy, observations)
    original, trained = mx.load(str(initial)), mx.load(str(checkpoint))
    deltas = [np.asarray(trained[name]) - np.asarray(value) for name, value in original.items() if name.startswith("model.")]
    parameter_change = float(np.sqrt(sum(float(np.square(delta).sum()) for delta in deltas)))
    model = onnx.load(str(policy))
    onnx.checker.check_model(model)
    input_dims = [dimension.dim_value or dimension.dim_param for dimension in model.graph.input[0].type.tensor_type.shape.dim]
    output_dims = [dimension.dim_value or dimension.dim_param for dimension in model.graph.output[0].type.tensor_type.shape.dim]
    contract_passed = input_dims == ["batch", 61] and output_dims == ["batch", 14]
    events, training = training_evidence(args.run, recipe["totalTimesteps"])
    negative_names = ("zero", "random") if pretrained else ("zero", "initial")
    controls_failed = all(not report["passed"] for name in negative_names for report in reports[name])
    passed = all(report["passed"] and report["finite"] for report in reports["trained"])
    result = {
        "recipe": recipe, "effective_audit_environment": environment_options(recipe),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "controls": reports, "history": history, "training": training,
        "initial_checkpoint": str(initial), "parameter_l2_change": parameter_change,
        "initialization": initialization,
        "prior_training_transitions": prior_steps,
        "initial_policy_role": "pretrained baseline, not a null control" if pretrained else "random-initialization null control",
        "negative_control_names": list(negative_names),
        "comparison_seed": args.seeds[0],
        "ppo_raw_return_change": reports["trained"][0]["raw_return"] - reports["initial"][0]["raw_return"],
        "onnx_parity_max_abs_error": policy_error,
        "onnx_parity_scope": "Exporter contract: normalized deterministic actor, including Swing's required [-1, 1] output clip.",
        "onnx_contract": {"input": input_dims, "output": output_dims, "passed": contract_passed},
        "negative_controls_failed": controls_failed,
        "passed": bool(passed and controls_failed and policy_error < 1e-4 and contract_passed and parameter_change > 0 and training["passed"]),
        "scope": "Nominal local simulation only; no hardware certification. No assisted Swing starts.",
    }
    (args.output / "audit.json").write_text(json.dumps(studio._finite_json(result), indent=2, allow_nan=False) + "\n")
    plots(args.output, scenario, events, history, traces)
    if args.render:
        comparison_video(args.output, recipe, traces, reports)
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "held_out": [report["passed"] for report in reports["trained"]]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--history-stride", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.history_stride < 1:
        parser.error("--history-stride must be positive")
    raise SystemExit(0 if audit(args)["passed"] else 2)


if __name__ == "__main__":
    main()
