"""Create a deployable Swing PPO warm start from a simulator-only teacher."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from rlx.environments.microduck_recipes import make_single_recipe_env


TEACHER_PARAMETERS = np.array(
    [
        0.16855159179301582,
        -5.978105843503098,
        -0.9914264524266221,
        -0.3787501101535562,
        -0.38920486336216165,
        -0.8471990875948665,
        0.7109162708565924,
        -0.9891212934189859,
    ],
    dtype=np.float32,
)


def teacher_action(angle: float, rate: float) -> np.ndarray:
    """Return a bounded sagittal action from privileged training state."""
    phase = math.tanh(
        float(
            TEACHER_PARAMETERS[0]
            + TEACHER_PARAMETERS[1] * angle
            + TEACHER_PARAMETERS[2] * rate
        )
    )
    action = np.zeros(14, dtype=np.float32)
    action[2:7] = TEACHER_PARAMETERS[3:8] * phase
    action[11:14] = -TEACHER_PARAMETERS[3:6] * phase
    return np.clip(action, -1.0, 1.0)


def collect_demonstrations(episodes: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    env = make_single_recipe_env(
        "swing",
        actuator="xml",
        seed=1,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=steps / 50,
        swing_planar_actions=True,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=episode + 1)
            for _ in range(steps):
                state = env.unwrapped._swing_state()
                action = teacher_action(state["angle"], state["angle_rate"])
                observations.append(observation.copy())
                actions.append(action)
                observation, _, _, _, _ = env.step(action)
    finally:
        env.close()
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/studio/swing/swing-bootstrap/swing.safetensors"),
    )
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from rlx.export.microduck_onnx import export_deterministic_actor
    from rlx.models.microduck import create_actor_critic, save_checkpoint

    observations, actions = collect_demonstrations(args.episodes, args.steps)
    mean = observations.mean(axis=0)
    variance = observations.var(axis=0)
    normalized = np.clip(
        (observations - mean) / np.sqrt(variance + 1e-8),
        -10.0,
        10.0,
    ).astype(np.float32)

    mx.random.seed(2)
    model = create_actor_critic(initial_std=0.1)
    optimizer = optim.Adam(learning_rate=3e-4)

    def loss_fn(network, inputs, targets):
        return mx.mean(mx.square(network.deterministic(inputs) - targets))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(2)
    for epoch in range(args.epochs):
        losses = []
        for _ in range(0, len(normalized), args.batch_size):
            indices = rng.choice(len(normalized), args.batch_size, replace=False)
            loss, gradients = loss_and_grad(
                model,
                mx.array(normalized[indices]),
                mx.array(actions[indices]),
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(float(np.asarray(loss)))
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(
                {"epoch": epoch + 1, "mean_imitation_loss": float(np.mean(losses))},
                flush=True,
            )

    save_checkpoint(
        args.output,
        model,
        mean,
        variance,
        float(len(observations)),
        return_mean=np.array(0.0),
        return_variance=np.array(1.0),
        return_count=1e-4,
        metadata={
            "recipe": "swing",
            "steps": 0,
            "bootstrap": "privileged simulator teacher distilled to 61D actor",
            "teacher_samples": len(observations),
            "teacher_parameters": TEACHER_PARAMETERS.tolist(),
        },
    )
    onnx = export_deterministic_actor(args.output, args.output.with_suffix(".onnx"))
    print({"checkpoint": str(args.output), "onnx": str(onnx)})


if __name__ == "__main__":
    main()
