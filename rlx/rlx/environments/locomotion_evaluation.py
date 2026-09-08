"""Strict per-episode acceptance for commanded running and stilt locomotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np


REQUIRED_METRICS = (
    "command_forward_m_s",
    "command_lateral_m_s",
    "command_yaw_rad_s",
    "heading_forward_speed_m_s",
    "heading_lateral_speed_m_s",
    "world_x_m",
    "world_y_m",
    "heading_forward_x",
    "heading_forward_y",
    "upright",
    "left_foot_contact",
    "right_foot_contact",
)


@dataclass(frozen=True)
class LocomotionEvaluationCriteria:
    """Predeclared physical floors, independent of training reward."""

    version: int
    recipe: str
    required_steps: int
    control_dt_s: float
    min_upright: float
    min_upright_fraction: float
    min_command_speed_m_s: float
    min_commanded_fraction: float
    min_mean_command_directed_speed_m_s: float
    min_command_directed_displacement_m: float
    min_command_speed_tracking_ratio: float
    min_alternating_support_switches: int
    min_each_foot_air_fraction: float
    min_each_foot_contact_fraction: float
    min_aerial_fraction: float | None
    all_episodes_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _criteria(recipe: str, required_steps: int) -> LocomotionEvaluationCriteria:
    if recipe == "running":
        if required_steps < 600:
            raise ValueError("running evaluation requires at least 600 steps (12 seconds)")
        return LocomotionEvaluationCriteria(
            version=2,
            recipe=recipe,
            required_steps=required_steps,
            control_dt_s=0.02,
            min_upright=0.9,
            min_upright_fraction=0.95,
            min_command_speed_m_s=0.1,
            min_commanded_fraction=0.5,
            min_mean_command_directed_speed_m_s=0.35,
            min_command_directed_displacement_m=2.0,
            min_command_speed_tracking_ratio=0.35,
            min_alternating_support_switches=8,
            min_each_foot_air_fraction=0.08,
            min_each_foot_contact_fraction=0.10,
            min_aerial_fraction=0.03,
        )
    if recipe == "stilts":
        if required_steps < 500:
            raise ValueError("stilts evaluation requires at least 500 steps (10 seconds)")
        return LocomotionEvaluationCriteria(
            version=2,
            recipe=recipe,
            required_steps=required_steps,
            control_dt_s=0.02,
            min_upright=0.9,
            min_upright_fraction=0.95,
            min_command_speed_m_s=0.05,
            min_commanded_fraction=0.5,
            min_mean_command_directed_speed_m_s=0.06,
            min_command_directed_displacement_m=0.30,
            min_command_speed_tracking_ratio=0.30,
            min_alternating_support_switches=4,
            min_each_foot_air_fraction=0.05,
            min_each_foot_contact_fraction=0.10,
            min_aerial_fraction=None,
        )
    raise ValueError("locomotion evaluation recipe must be 'running' or 'stilts'")


@dataclass
class _Episode:
    env_index: int
    index: int
    steps: int = 0
    measured_steps: int = 0
    upright_steps: int = 0
    commanded_steps: int = 0
    command_speed_sum: float = 0.0
    directed_speed_sum: float = 0.0
    directed_displacement_m: float = 0.0
    left_contact_steps: int = 0
    right_contact_steps: int = 0
    aerial_steps: int = 0
    alternating_support_switches: int = 0
    left_liftoffs: int = 0
    right_liftoffs: int = 0
    left_touchdowns: int = 0
    right_touchdowns: int = 0
    previous_contacts: tuple[bool, bool] | None = None
    previous_exclusive_support: str | None = None
    previous_position: np.ndarray | None = None
    previous_command: np.ndarray | None = None
    previous_heading: np.ndarray | None = None
    previous_yaw_command: float | None = None
    intended_yaw: float | None = None
    failures: set[str] = field(default_factory=set)

    def observe(
        self,
        metrics: dict[str, Any],
        criteria: LocomotionEvaluationCriteria,
    ) -> None:
        self.steps += 1
        if not isinstance(metrics, dict):
            self.failures.add("missing or non-finite locomotion metrics")
            self._clear_motion_history()
            return
        try:
            values = {key: float(metrics[key]) for key in REQUIRED_METRICS}
        except (KeyError, TypeError, ValueError):
            self.failures.add("missing or non-finite locomotion metrics")
            self._clear_motion_history()
            return
        if not all(math.isfinite(value) for value in values.values()):
            self.failures.add("missing or non-finite locomotion metrics")
            self._clear_motion_history()
            return
        if values["left_foot_contact"] not in (0.0, 1.0) or values[
            "right_foot_contact"
        ] not in (0.0, 1.0):
            self.failures.add("invalid foot contact metrics")
            self._clear_motion_history()
            return

        heading = np.array(
            [values["heading_forward_x"], values["heading_forward_y"]],
            dtype=np.float64,
        )
        heading_norm = float(np.linalg.norm(heading))
        if not 0.99 <= heading_norm <= 1.01:
            self.failures.add("invalid heading basis")
            self._clear_motion_history()
            return
        heading /= heading_norm
        command = np.array(
            [values["command_forward_m_s"], values["command_lateral_m_s"]],
            dtype=np.float64,
        )
        velocity = np.array(
            [
                values["heading_forward_speed_m_s"],
                values["heading_lateral_speed_m_s"],
            ],
            dtype=np.float64,
        )
        position = np.array(
            [values["world_x_m"], values["world_y_m"]],
            dtype=np.float64,
        )
        command_speed = float(np.linalg.norm(command))
        yaw_command = values["command_yaw_rad_s"]
        commanded = command_speed >= criteria.min_command_speed_m_s

        self.measured_steps += 1
        self.upright_steps += int(values["upright"] >= criteria.min_upright)
        if commanded:
            command_direction = command / command_speed
            self.commanded_steps += 1
            self.command_speed_sum += command_speed
            self.directed_speed_sum += float(velocity @ command_direction)

        if self.intended_yaw is None or (
            not np.array_equal(command, self.previous_command)
            or yaw_command != self.previous_yaw_command
        ):
            anchor = self.previous_heading if self.previous_heading is not None else heading
            self.intended_yaw = math.atan2(float(anchor[1]), float(anchor[0]))
        if self.previous_position is not None:
            turn = yaw_command * criteria.control_dt_s
            if commanded:
                projection_yaw = self.intended_yaw + 0.5 * turn
                intended_forward = np.array(
                    [math.cos(projection_yaw), math.sin(projection_yaw)]
                )
                intended_side = np.array([-intended_forward[1], intended_forward[0]])
                world_direction = (
                    intended_forward * command[0] + intended_side * command[1]
                ) / command_speed
                self.directed_displacement_m += float(
                    (position - self.previous_position) @ world_direction
                )
            self.intended_yaw += turn

        left = values["left_foot_contact"] == 1.0
        right = values["right_foot_contact"] == 1.0
        self.left_contact_steps += int(left)
        self.right_contact_steps += int(right)
        self.aerial_steps += int(not left and not right)
        if self.previous_contacts is not None:
            previous_left, previous_right = self.previous_contacts
            self.left_liftoffs += int(previous_left and not left)
            self.right_liftoffs += int(previous_right and not right)
            self.left_touchdowns += int(not previous_left and left)
            self.right_touchdowns += int(not previous_right and right)
        exclusive_support = "left" if left and not right else (
            "right" if right and not left else None
        )
        if exclusive_support is not None:
            if (
                self.previous_exclusive_support is not None
                and exclusive_support != self.previous_exclusive_support
            ):
                self.alternating_support_switches += 1
            self.previous_exclusive_support = exclusive_support

        self.previous_contacts = (left, right)
        self.previous_position = position
        self.previous_command = command
        self.previous_heading = heading
        self.previous_yaw_command = yaw_command

    def _clear_motion_history(self) -> None:
        self.previous_contacts = None
        self.previous_exclusive_support = None
        self.previous_position = None
        self.previous_command = None
        self.previous_heading = None
        self.previous_yaw_command = None
        self.intended_yaw = None

    def finish(
        self,
        criteria: LocomotionEvaluationCriteria,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        failures = set(self.failures)
        complete = (
            truncated
            and not terminated
            and self.steps >= criteria.required_steps
        )
        if terminated:
            failures.add("episode terminated or fell")
        if not complete:
            failures.add(
                f"episode did not complete the required "
                f"{criteria.required_steps * criteria.control_dt_s:g}-second horizon"
            )
        if self.steps == 0 or self.measured_steps != self.steps:
            failures.add("incomplete locomotion metric coverage")

        denominator = self.steps or 1
        commanded_denominator = self.commanded_steps or 1
        upright_fraction = self.upright_steps / denominator
        commanded_fraction = self.commanded_steps / denominator
        mean_command_speed = self.command_speed_sum / commanded_denominator
        mean_directed_speed = self.directed_speed_sum / commanded_denominator
        tracking_ratio = (
            self.directed_speed_sum / self.command_speed_sum
            if self.command_speed_sum > 0.0
            else None
        )
        left_contact_fraction = self.left_contact_steps / denominator
        right_contact_fraction = self.right_contact_steps / denominator
        left_air_fraction = 1.0 - left_contact_fraction
        right_air_fraction = 1.0 - right_contact_fraction
        aerial_fraction = self.aerial_steps / denominator

        if upright_fraction < criteria.min_upright_fraction:
            failures.add("upright fraction below target")
        if commanded_fraction < criteria.min_commanded_fraction:
            failures.add("insufficient commanded-motion coverage")
        if mean_directed_speed < criteria.min_mean_command_directed_speed_m_s:
            failures.add("command-directed speed below target")
        if self.directed_displacement_m < criteria.min_command_directed_displacement_m:
            failures.add("command-directed displacement below target")
        if (
            tracking_ratio is None
            or tracking_ratio < criteria.min_command_speed_tracking_ratio
        ):
            failures.add("command speed tracking ratio below target")
        if (
            self.alternating_support_switches
            < criteria.min_alternating_support_switches
        ):
            failures.add("insufficient alternating foot support")
        if min(left_air_fraction, right_air_fraction) < criteria.min_each_foot_air_fraction:
            failures.add("one or both feet did not leave the ground enough")
        if min(left_contact_fraction, right_contact_fraction) < criteria.min_each_foot_contact_fraction:
            failures.add("one or both feet did not establish ground contact enough")
        if not all(
            count > 0
            for count in (
                self.left_liftoffs,
                self.right_liftoffs,
                self.left_touchdowns,
                self.right_touchdowns,
            )
        ):
            failures.add("both feet must lift off and touch down")
        if (
            criteria.min_aerial_fraction is not None
            and aerial_fraction < criteria.min_aerial_fraction
        ):
            failures.add("aerial fraction below running target")

        return {
            "env_index": self.env_index,
            "episode_index": self.index,
            "steps": self.steps,
            "measured_steps": self.measured_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "complete": complete,
            "upright_fraction": upright_fraction,
            "commanded_fraction": commanded_fraction,
            "mean_command_speed_m_s": mean_command_speed,
            "mean_command_directed_speed_m_s": mean_directed_speed,
            "command_speed_tracking_ratio": tracking_ratio,
            "command_directed_displacement_m": self.directed_displacement_m,
            "left_contact_fraction": left_contact_fraction,
            "right_contact_fraction": right_contact_fraction,
            "left_air_fraction": left_air_fraction,
            "right_air_fraction": right_air_fraction,
            "aerial_fraction": aerial_fraction,
            "alternating_support_switches": self.alternating_support_switches,
            "left_liftoffs": self.left_liftoffs,
            "right_liftoffs": self.right_liftoffs,
            "left_touchdowns": self.left_touchdowns,
            "right_touchdowns": self.right_touchdowns,
            "passed": not failures,
            "failures": sorted(failures),
        }


class LocomotionEvaluation:
    """Accumulate seeded command-tracking evidence independently per episode."""

    def __init__(self, recipe: str, num_envs: int, required_steps: int) -> None:
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
        self.criteria = _criteria(recipe, int(required_steps))
        self._current = [_Episode(index, 0) for index in range(num_envs)]
        self._episodes: list[dict[str, Any]] = []

    def observe(
        self,
        index: int,
        metrics: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not 0 <= index < len(self._current):
            raise IndexError(f"env index out of range: {index}")
        episode = self._current[index]
        episode.observe(metrics, self.criteria)
        if terminated or truncated:
            self._episodes.append(
                episode.finish(
                    self.criteria,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            self._current[index] = _Episode(index, episode.index + 1)

    def report(self) -> dict[str, Any]:
        episodes = [
            *self._episodes,
            *(
                episode.finish(
                    self.criteria,
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
        failures = {
            failure
            for episode in episodes
            for failure in episode["failures"]
        }
        if missing_lanes or not episodes:
            failures.add("missing evaluation lane coverage")
        return {
            "criteria": self.criteria.to_dict(),
            "episodes": episodes,
            "completed_episodes": completed,
            "incomplete_episodes": len(episodes) - completed,
            "passed_episodes": sum(bool(episode["passed"]) for episode in episodes),
            "passed": bool(episodes) and not failures,
            "failures": sorted(failures),
            "measurement_scope": (
                "50 Hz per-episode samples; heading-frame velocity is compared "
                "with the active local xy command. Signed world-position deltas "
                "use a heading anchored at each change of twist command, advanced "
                "only by commanded yaw at 0.02 s per sample. Zero-yaw segments "
                "measure net intended progress, not distance along a drifting heading. "
                "The first sample establishes position; its displacement is not credited."
            ),
            "command_policy": (
                "evaluates the command sequence supplied by the environment; "
                "translational commands in any direction count and standing/"
                "turn-only samples do not count toward commanded-motion coverage"
            ),
            "threshold_rationale": (
                "running uses the recipe's 0.4 m/s running-regime boundary as "
                "context but sets a 0.35 m/s physical floor plus true aerial "
                "samples; stilts uses lower speed/displacement floors appropriate "
                "to the 0.25 m/s command ceiling. Both require repeated bilateral "
                "support exchange and 95% upright control."
            ),
            "hardware_safe": False,
        }


__all__ = [
    "LocomotionEvaluation",
    "LocomotionEvaluationCriteria",
    "REQUIRED_METRICS",
]
