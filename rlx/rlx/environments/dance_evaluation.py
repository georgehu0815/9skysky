"""Strict per-episode acceptance checks for time-aligned dance imitation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np


JOINT_COUNT = 14
LEG_JOINT_INDICES = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
REQUIRED_STATE = (
    "phase_step",
    "current_joints",
    "target_joints",
    "upright",
    "height_m",
)


@dataclass(frozen=True)
class DanceEvaluationCriteria:
    version: int = 1
    min_upright: float = 0.9
    min_upright_fraction: float = 0.95
    max_pose_rmse_rad: float = 0.15
    min_dynamic_gain: float = 0.20
    min_leg_dynamic_gain: float = 0.0
    moving_joint_std_rad: float = 0.03
    min_moving_leg_joints: int = 2
    all_episodes_required: bool = True

    def __post_init__(self) -> None:
        bounded = {
            "min_upright": self.min_upright,
            "min_upright_fraction": self.min_upright_fraction,
        }
        for name, value in bounded.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        positive = {
            "max_pose_rmse_rad": self.max_pose_rmse_rad,
            "moving_joint_std_rad": self.moving_joint_std_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.min_dynamic_gain)
            or not 0.0 <= self.min_dynamic_gain <= 1.0
        ):
            raise ValueError("min_dynamic_gain must be finite and in [0, 1]")
        if (
            not math.isfinite(self.min_leg_dynamic_gain)
            or not 0.0 <= self.min_leg_dynamic_gain < 1.0
        ):
            raise ValueError(
                "min_leg_dynamic_gain must be finite and in [0, 1)"
            )
        if self.min_moving_leg_joints < 1:
            raise ValueError("min_moving_leg_joints must be positive")
        if self.all_episodes_required is not True:
            raise ValueError("all_episodes_required must remain true")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _dynamic_statistics(
    current: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if not indices.size:
        return None, None, None
    selected_current = current[:, indices]
    selected_target = target[:, indices]
    baseline = _rmse(selected_target - selected_target.mean(axis=0))
    tracking = _rmse(selected_current - selected_target)

    correlations: list[float] = []
    for column in range(selected_target.shape[1]):
        target_centered = selected_target[:, column] - selected_target[:, column].mean()
        current_centered = (
            selected_current[:, column] - selected_current[:, column].mean()
        )
        denominator = float(
            np.linalg.norm(target_centered) * np.linalg.norm(current_centered)
        )
        correlations.append(
            0.0
            if denominator <= 0.0
            else float(np.dot(target_centered, current_centered) / denominator)
        )
    return tracking, baseline, float(np.mean(correlations))


@dataclass
class _Episode:
    env_index: int
    index: int
    steps: int = 0
    upright_steps: int = 0
    phase_steps: list[int] = field(default_factory=list)
    current_joints: list[np.ndarray] = field(default_factory=list)
    target_joints: list[np.ndarray] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    failures: set[str] = field(default_factory=set)

    def observe(
        self,
        state: dict[str, Any],
        criteria: DanceEvaluationCriteria,
    ) -> None:
        self.steps += 1
        if not isinstance(state, dict) or any(
            key not in state for key in REQUIRED_STATE
        ):
            self.failures.add("missing or non-finite dance state")
            return
        try:
            phase_value = float(state["phase_step"])
            current = np.asarray(state["current_joints"], dtype=np.float64)
            target = np.asarray(state["target_joints"], dtype=np.float64)
            upright = float(state["upright"])
            height = float(state["height_m"])
        except (TypeError, ValueError):
            self.failures.add("missing or non-finite dance state")
            return
        if (
            current.shape != (JOINT_COUNT,)
            or target.shape != (JOINT_COUNT,)
            or not np.isfinite(current).all()
            or not np.isfinite(target).all()
            or not all(math.isfinite(value) for value in (phase_value, upright, height))
            or not phase_value.is_integer()
        ):
            self.failures.add("missing or non-finite dance state")
            return

        self.phase_steps.append(int(phase_value))
        self.current_joints.append(current.copy())
        self.target_joints.append(target.copy())
        self.heights.append(height)
        self.upright_steps += int(upright >= criteria.min_upright)

    def finish(
        self,
        criteria: DanceEvaluationCriteria,
        required_steps: int,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        failures = set(self.failures)
        measured_steps = len(self.current_joints)
        complete = truncated and not terminated and self.steps >= required_steps
        if terminated:
            failures.add("episode terminated or fell")
        if not complete:
            failures.add("episode did not cover the requested clip horizon")
        if self.steps == 0 or measured_steps != self.steps:
            failures.add("incomplete dance state coverage")
        phase_progression_valid = bool(self.phase_steps) and all(
            current == previous + 1
            for previous, current in zip(self.phase_steps, self.phase_steps[1:])
        )
        if measured_steps and not phase_progression_valid:
            failures.add("non-contiguous dance phase evidence")

        result: dict[str, Any] = {
            "env_index": self.env_index,
            "episode_index": self.index,
            "steps": self.steps,
            "measured_steps": measured_steps,
            "terminated": terminated,
            "truncated": truncated,
            "complete": complete,
            "upright_fraction": (
                self.upright_steps / self.steps if self.steps else 0.0
            ),
            "height_m": {
                "mean": float(np.mean(self.heights)) if self.heights else None,
                "min": float(np.min(self.heights)) if self.heights else None,
            },
            "phase_step": {
                "start": self.phase_steps[0] if self.phase_steps else None,
                "end": self.phase_steps[-1] if self.phase_steps else None,
                "unique": len(set(self.phase_steps)),
                "contiguous": phase_progression_valid,
            },
            "pose_rmse_rad": None,
            "leg_pose_rmse_rad": None,
            "moving_joint_count": 0,
            "moving_leg_joint_count": 0,
            "dynamic_tracking_rmse_rad": None,
            "constant_mean_baseline_rmse_rad": None,
            "dynamic_gain": None,
            "mean_joint_correlation": None,
            "leg_dynamic_tracking_rmse_rad": None,
            "leg_constant_mean_baseline_rmse_rad": None,
            "leg_dynamic_gain": None,
            "leg_mean_joint_correlation": None,
        }

        if measured_steps:
            current = np.stack(self.current_joints)
            target = np.stack(self.target_joints)
            errors = current - target
            leg_indices = np.asarray(LEG_JOINT_INDICES)
            reference_std = target.std(axis=0)
            moving = np.flatnonzero(reference_std >= criteria.moving_joint_std_rad)
            moving_set = set(moving)
            moving_legs = np.asarray(
                [index for index in LEG_JOINT_INDICES if index in moving_set]
            )
            tracking, baseline, correlation = _dynamic_statistics(
                current, target, moving
            )
            leg_tracking, leg_baseline, leg_correlation = _dynamic_statistics(
                current, target, moving_legs
            )
            dynamic_gain = (
                None if tracking is None or baseline is None
                else 1.0 - tracking / baseline
            )
            leg_dynamic_gain = (
                None if leg_tracking is None or leg_baseline is None
                else 1.0 - leg_tracking / leg_baseline
            )
            pose_rmse = _rmse(errors)
            leg_pose_rmse = _rmse(errors[:, leg_indices])
            result.update(
                pose_rmse_rad=pose_rmse,
                leg_pose_rmse_rad=leg_pose_rmse,
                moving_joint_count=int(moving.size),
                moving_leg_joint_count=int(moving_legs.size),
                dynamic_tracking_rmse_rad=tracking,
                constant_mean_baseline_rmse_rad=baseline,
                dynamic_gain=dynamic_gain,
                mean_joint_correlation=correlation,
                leg_dynamic_tracking_rmse_rad=leg_tracking,
                leg_constant_mean_baseline_rmse_rad=leg_baseline,
                leg_dynamic_gain=leg_dynamic_gain,
                leg_mean_joint_correlation=leg_correlation,
            )
            if result["upright_fraction"] < criteria.min_upright_fraction:
                failures.add("upright fraction below target")
            if pose_rmse > criteria.max_pose_rmse_rad:
                failures.add("pose RMSE above target")
            if moving_legs.size < criteria.min_moving_leg_joints:
                failures.add("insufficient moving leg joints in reference")
            if dynamic_gain is None or dynamic_gain < criteria.min_dynamic_gain:
                failures.add("dynamic tracking gain below target")
            if (
                leg_dynamic_gain is None
                or leg_dynamic_gain <= criteria.min_leg_dynamic_gain
            ):
                failures.add("leg dynamic tracking did not improve on baseline")

        result["passed"] = not failures
        result["failures"] = sorted(failures)
        return result


class DanceEvaluation:
    """Accumulate raw dance state per vector lane without smoothing or lag search."""

    def __init__(
        self,
        num_envs: int,
        required_steps: int,
        criteria: DanceEvaluationCriteria | None = None,
    ) -> None:
        if (
            isinstance(num_envs, bool)
            or not isinstance(num_envs, (int, np.integer))
            or num_envs < 1
        ):
            raise ValueError("num_envs must be positive")
        if (
            isinstance(required_steps, bool)
            or not isinstance(required_steps, (int, np.integer))
            or required_steps < 1
        ):
            raise ValueError("required_steps must be positive")
        self.required_steps = int(required_steps)
        self.criteria = criteria or DanceEvaluationCriteria()
        self._current = [_Episode(index, 0) for index in range(num_envs)]
        self._episodes: list[dict[str, Any]] = []

    def observe(
        self,
        env_index: int,
        state: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not 0 <= env_index < len(self._current):
            raise IndexError(f"env_index out of range: {env_index}")
        episode = self._current[env_index]
        episode.observe(state, self.criteria)
        if terminated or truncated:
            self._episodes.append(
                episode.finish(
                    self.criteria,
                    self.required_steps,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            self._current[env_index] = _Episode(env_index, episode.index + 1)

    def report(self) -> dict[str, Any]:
        episodes = [
            *self._episodes,
            *(
                episode.finish(
                    self.criteria,
                    self.required_steps,
                    terminated=False,
                    truncated=False,
                )
                for episode in self._current
                if episode.steps
            ),
        ]
        covered_lanes = {episode["env_index"] for episode in episodes}
        missing_lanes = sorted(set(range(len(self._current))) - covered_lanes)
        completed = sum(bool(episode["complete"]) for episode in episodes)
        failures = sorted(
            {failure for episode in episodes for failure in episode["failures"]}
        )
        if missing_lanes or not episodes:
            failures.append("missing evaluation lane coverage")
        return {
            "criteria": {
                **self.criteria.to_dict(),
                "required_steps": self.required_steps,
            },
            "episodes": episodes,
            "completed_episodes": completed,
            "incomplete_episodes": len(episodes) - completed,
            "passed_episodes": sum(bool(episode["passed"]) for episode in episodes),
            "passed": bool(episodes) and not failures,
            "failures": failures,
            "measurement_scope": (
                "raw time-aligned control-step joint samples; no smoothing, "
                "phase lag, or temporal alignment adjustment"
            ),
            "dynamic_baseline": (
                "per-joint constant mean of the episode reference trajectory"
            ),
            "horizon_definition": (
                "required_steps is the caller-selected clip or excerpt horizon; "
                "400 steps represents 8 seconds at the 50 Hz control rate"
            ),
            "hardware_safe": False,
        }


__all__ = ["DanceEvaluation", "DanceEvaluationCriteria"]
