"""Hello World proof that each visible duck owns independent MuJoCo state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import contract as C
from .viz_server import Duck
from .walk_env import MicroduckWalkEnv


@dataclass(frozen=True)
class DuckMeasurement:
    """Measured object identities and trunk positions for one real Duck."""

    duck_id: str
    seed: int
    env_id: int
    data_id: int
    model_id: int
    initial_trunk: tuple[float, float, float]
    final_trunk: tuple[float, float, float]


@dataclass(frozen=True)
class DemoResult:
    """Verified measurements produced by the demo."""

    ducks: tuple[DuckMeasurement, ...]
    isolated_qpos_unchanged: bool
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]


def zero_policy(observation: np.ndarray) -> np.ndarray:
    """Return the safe zero action after checking the deployment observation contract."""
    if observation.shape != (C.OBS_DIM,):
        raise ValueError(
            f"expected observation shape ({C.OBS_DIM},), got {observation.shape}"
        )
    return np.zeros(C.NUM_JOINTS, dtype=np.float32)


def _trunk_position(duck: Duck) -> tuple[float, float, float]:
    position = duck.env.data.xpos[duck.env.trunk_body_id]
    return tuple(float(value) for value in position)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"hello-ducks invariant failed: {message}")


def run_demo(duck_count: int = 3, steps: int = 20) -> DemoResult:
    """Create real Ducks, prove model sharing and state isolation, then close them."""
    if not 2 <= duck_count <= 20:
        raise ValueError("duck_count must be between 2 and 20")
    if steps <= 0:
        raise ValueError("steps must be positive")

    ducks: list[Duck] = []
    try:
        for index in range(duck_count):
            ducks.append(
                Duck(
                    f"duck-{index + 1}",
                    f"Hello Duck {index + 1}",
                    zero_policy,
                    seed=10_001 + index,
                    env_kwargs={"actuator_force": "xml"},
                )
            )

        initial_positions = [_trunk_position(duck) for duck in ducks]
        env_ids = [id(duck.env) for duck in ducks]
        data_ids = [id(duck.env.data) for duck in ducks]
        model_ids = [id(duck.env.model) for duck in ducks]

        _require(len(set(env_ids)) == duck_count, "each Duck must own a distinct env")
        _require(
            all(isinstance(duck.env, MicroduckWalkEnv) for duck in ducks),
            "each Duck must own a MicroduckWalkEnv",
        )
        _require(len(set(data_ids)) == duck_count, "each env must own distinct MjData")
        _require(len(set(model_ids)) == 1, "XML ducks must share one compiled MjModel")
        _require(
            all(duck.env._model_shared for duck in ducks),
            "the shared model must be marked for safe sibling use",
        )

        observation_shape = tuple(ducks[0].obs.shape)
        action_shape = tuple(zero_policy(ducks[0].obs).shape)
        _require(observation_shape == (C.OBS_DIM,), "observation shape must be (61,)")
        _require(action_shape == (C.NUM_JOINTS,), "action shape must be (14,)")
        _require(
            all(tuple(duck.obs.shape) == observation_shape for duck in ducks),
            "all ducks must use the same observation contract",
        )

        duck_2_qpos = ducks[1].env.data.qpos.tobytes()
        ducks[0].tick()
        isolated_qpos_unchanged = ducks[1].env.data.qpos.tobytes() == duck_2_qpos
        _require(
            isolated_qpos_unchanged,
            "stepping duck-1 changed duck-2 qpos bytes",
        )

        # Advance one duck at a time. Every sibling's state must remain untouched
        # until its own turn, even though all ducks share the same compiled model.
        for duck in ducks:
            sibling_qpos = {
                sibling.id: sibling.env.data.qpos.tobytes()
                for sibling in ducks
                if sibling is not duck
            }
            old_time = float(duck.env.data.time)
            for _ in range(steps):
                duck.tick()
            _require(float(duck.env.data.time) > old_time, f"{duck.id} did not advance")
            for sibling in ducks:
                if sibling is not duck:
                    _require(
                        sibling.env.data.qpos.tobytes() == sibling_qpos[sibling.id],
                        f"stepping {duck.id} changed {sibling.id} qpos bytes",
                    )

        measurements = tuple(
            DuckMeasurement(
                duck_id=duck.id,
                seed=duck.seed,
                env_id=id(duck.env),
                data_id=id(duck.env.data),
                model_id=id(duck.env.model),
                initial_trunk=initial,
                final_trunk=_trunk_position(duck),
            )
            for duck, initial in zip(ducks, initial_positions, strict=True)
        )
        return DemoResult(
            ducks=measurements,
            isolated_qpos_unchanged=isolated_qpos_unchanged,
            observation_shape=observation_shape,
            action_shape=action_shape,
        )
    finally:
        for duck in ducks:
            close = getattr(duck.env, "close", None)
            if callable(close):
                close()


def _bounded_ducks(value: str) -> int:
    count = int(value)
    if not 2 <= count <= 20:
        raise argparse.ArgumentTypeError("must be between 2 and 20")
    return count


def _positive_steps(value: str) -> int:
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prove that visible Duck Lab ducks have independent MuJoCo state."
    )
    parser.add_argument("--ducks", type=_bounded_ducks, default=3)
    parser.add_argument("--steps", type=_positive_steps, default=20)
    return parser


def _format_position(position: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{value:+.5f}" for value in position) + ")"


def _print_result(result: DemoResult) -> None:
    headers = ("Duck", "Measurement", "Measured value")
    rows: list[tuple[str, str, str]] = []
    for duck in result.ducks:
        rows.extend(
            (
                (duck.duck_id, "seed", str(duck.seed)),
                ("", "MicroduckWalkEnv identity", hex(duck.env_id)),
                ("", "private MjData identity", hex(duck.data_id)),
                ("", "shared MjModel identity", hex(duck.model_id)),
                ("", "initial trunk (x, y, z)", _format_position(duck.initial_trunk)),
                ("", "final trunk (x, y, z)", _format_position(duck.final_trunk)),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    rule = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def print_row(row: tuple[str, str, str]) -> None:
        cells = (value.ljust(widths[index]) for index, value in enumerate(row))
        print("| " + " | ".join(cells) + " |")

    print(rule)
    print_row(headers)
    print(rule)
    for row in rows:
        print_row(row)
    print(rule)
    print(
        f"Measured {len(result.ducks)} distinct envs, {len(result.ducks)} distinct MjData "
        f"objects, and {len({duck.model_id for duck in result.ducks})} shared MjModel."
    )
    print(
        "English: Each visible Duck owns an independent environment and MjData state; "
        "XML ducks share one read-only compiled MjModel."
    )
    print(
        "简体中文：每只可见鸭子拥有独立的环境和 MjData 状态；"
        "同一场景中的 XML 鸭子共享一个只读的已编译 MjModel。"
    )
    print(
        f"Contracts: observation={result.observation_shape}, action={result.action_shape}; "
        f"duck-2 qpos unchanged={result.isolated_qpos_unchanged}."
    )
    print("PASS: one environment and private simulation state per visible duck")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_demo(args.ducks, args.steps)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
