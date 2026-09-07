"""Per-episode Swing acceptance at control-rate samples, never pooled across resets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class SwingEvaluationPlan:
    """A still-start 24-second, tensioned, bidirectional simulated swing criterion."""

    version: int = 1
    min_bidirectional_span_deg: float = 150.0
    required_steps: int = 1200
    max_abs_lateral_m: float = 0.020
    max_alignment: float = 0.050
    min_string_length_m: float = 0.370
    max_string_length_m: float = 0.394
    min_string_tension_n: float = 0.0
    all_episodes_required: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_bidirectional_span_deg) or not 0 < self.min_bidirectional_span_deg <= 180:
            raise ValueError("Swing minimum bidirectional span must be finite and in (0, 180]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_METRICS = (
    "swing_angle_deg", "lateral_offset_m", "alignment_penalty", "valid_geometry",
    "string_left_m", "string_right_m", "string_left_tension_n", "string_right_tension_n",
)


@dataclass
class _Episode:
    env_index: int
    index: int
    steps: int = 0
    measured_steps: int = 0
    positive_peak: float = 0.0
    negative_peak: float = 0.0
    valid_steps: int = 0
    tensioned_steps: int = 0
    max_lateral: float = 0.0
    max_alignment: float = 0.0
    min_length: float | None = None
    max_length: float | None = None
    min_tension: float | None = None
    failures: set[str] = field(default_factory=set)

    def observe(self, metrics: dict[str, Any], plan: SwingEvaluationPlan) -> None:
        self.steps += 1
        angle = metrics.get("swing_angle_deg")
        if isinstance(angle, (int, float)) and math.isfinite(angle):
            self.positive_peak = max(self.positive_peak, float(angle))
            self.negative_peak = max(self.negative_peak, -float(angle))
        if any(not isinstance(metrics.get(key), (int, float))
               or not math.isfinite(metrics[key]) for key in REQUIRED_METRICS):
            self.failures.add("missing or non-finite physical metrics")
            return
        self.measured_steps += 1
        lateral = abs(float(metrics["lateral_offset_m"]))
        alignment = float(metrics["alignment_penalty"])
        lengths = (float(metrics["string_left_m"]), float(metrics["string_right_m"]))
        tensions = (float(metrics["string_left_tension_n"]), float(metrics["string_right_tension_n"]))
        self.max_lateral = max(self.max_lateral, lateral)
        self.max_alignment = max(self.max_alignment, alignment)
        self.min_length = min(lengths) if self.min_length is None else min(self.min_length, *lengths)
        self.max_length = max(lengths) if self.max_length is None else max(self.max_length, *lengths)
        self.min_tension = min(tensions) if self.min_tension is None else min(self.min_tension, *tensions)
        geometry_valid = (lateral <= plan.max_abs_lateral_m
                          and 0 <= alignment <= plan.max_alignment
                          and all(plan.min_string_length_m <= length <= plan.max_string_length_m for length in lengths)
                          and metrics["valid_geometry"] == 1)
        tensioned = all(tension > plan.min_string_tension_n for tension in tensions)
        self.valid_steps += int(geometry_valid)
        self.tensioned_steps += int(tensioned)
        if not geometry_valid:
            self.failures.add("invalid swing geometry")
        if not tensioned:
            self.failures.add("one or both strings lost positive spring tension")

    def finish(self, plan: SwingEvaluationPlan, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        failures = set(self.failures)
        complete = truncated and not terminated and self.steps >= plan.required_steps
        if terminated:
            failures.add("episode terminated")
        if not complete:
            failures.add("incomplete 24-second episode")
        if self.measured_steps != self.steps or self.steps == 0:
            failures.add("incomplete physical measurement coverage")
        span = 2 * min(self.positive_peak, self.negative_peak)
        if span < plan.min_bidirectional_span_deg:
            failures.add("bidirectional span below target")
        return {
            "env_index": self.env_index, "episode_index": self.index,
            "steps": self.steps, "measured_steps": self.measured_steps,
            "terminated": terminated, "truncated": truncated, "complete": complete,
            "positive_peak_deg": self.positive_peak, "negative_peak_deg": -self.negative_peak,
            "peak_to_peak_span_deg": self.positive_peak + self.negative_peak,
            "bidirectional_span_deg": span,
            "valid_geometry_fraction": self.valid_steps / self.steps if self.steps else 0.0,
            "both_strings_tensioned_fraction": self.tensioned_steps / self.steps if self.steps else 0.0,
            "max_abs_lateral_m": self.max_lateral if self.measured_steps else None,
            "max_alignment": self.max_alignment if self.measured_steps else None,
            "min_string_length_m": self.min_length, "max_string_length_m": self.max_length,
            "min_spring_tension_n": self.min_tension,
            "passed": not failures, "failures": sorted(failures),
        }


class SwingEvaluation:
    """Accumulate every vector lane independently; partial episodes cannot pass."""

    def __init__(self, num_envs: int, plan: SwingEvaluationPlan):
        self.plan = plan
        self._current = [_Episode(index, 0) for index in range(num_envs)]
        self._episodes: list[dict[str, Any]] = []

    def observe(self, env_index: int, metrics: dict[str, Any], *, terminated: bool, truncated: bool) -> None:
        episode = self._current[env_index]
        episode.observe(metrics, self.plan)
        if terminated or truncated:
            self._episodes.append(episode.finish(self.plan, terminated=terminated, truncated=truncated))
            self._current[env_index] = _Episode(env_index, episode.index + 1)

    def report(self) -> dict[str, Any]:
        episodes = [*self._episodes,
                    *(episode.finish(self.plan, terminated=False, truncated=False)
                      for episode in self._current if episode.steps)]
        missing_lanes = sorted(set(range(len(self._current))) - {episode["env_index"] for episode in episodes})
        completed = sum(episode["complete"] for episode in episodes)
        failures = sorted({failure for episode in episodes for failure in episode["failures"]})
        if missing_lanes or not episodes:
            failures.append("missing evaluation lane coverage")
        return {
            "criteria": self.plan.to_dict(), "episodes": episodes,
            "completed_episodes": completed, "incomplete_episodes": len(episodes) - completed,
            "passed_episodes": sum(episode["passed"] for episode in episodes),
            "passed": bool(episodes) and not failures,
            "failures": failures,
            "measurement_scope": "50 Hz control-step samples; spring tension excludes safety-limit constraint forces",
            "span_definition": "2 * min(positive peak, absolute negative peak) within each episode",
            "hardware_safe": False,
        }
