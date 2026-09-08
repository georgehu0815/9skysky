import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


@pytest.fixture
def audit_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("scenario_audit_test", scripts / "audit_scenarios.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_audit_swing_never_uses_assisted_start(audit_module):
    options = audit_module.environment_options({
        "maxEpisodeS": 24, "swingInitialAngleDeg": 50, "swingInitialRateRadS": 4,
        "swingPlanarActions": True,
    })
    assert options["swing_initial_angle_deg"] == 0
    assert options["swing_initial_rate_rad_s"] == 0
    assert options["swing_planar_actions"] is True
    assert not options["domain_rand"]


def test_audit_preserves_stilt_morphology(audit_module):
    options = audit_module.environment_options({"maxEpisodeS": 10, "stiltHeightCm": 2, "stiltBlend": .1, "stiltMassKg": .014})
    assert options["stilt_height_cm"] == 2
    assert options["stilt_blend"] == .1
    assert options["stilt_mass_kg"] == .014


@pytest.mark.parametrize("step_count,finite,expected", [(2048, True, True), (1024, True, False), (2048, False, False)])
def test_audit_requires_complete_finite_training(audit_module, tmp_path, step_count, finite, expected):
    update = dict.fromkeys(("mean_loss", "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction", "explained_variance"), .1)
    update.update(phase="update", env_steps=step_count, steps=step_count, optimizer_steps=16)
    if not finite:
        update["value_loss"] = float("nan")
    events = [{"phase": "collection", "steps": step_count, "env_steps": step_count}, update]
    (tmp_path / "training-metrics.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    _, result = audit_module.training_evidence(tmp_path, 2048)
    assert result["passed"] is expected
    assert result["observed_transitions"] == step_count


def test_audit_empty_updates_record_failure(audit_module, tmp_path):
    (tmp_path / "training-metrics.jsonl").write_text("")
    _, result = audit_module.training_evidence(tmp_path, 2048)
    assert result["passed"] is False
    assert result["max_approximate_kl"] is None
    json.dumps(result, allow_nan=False)


def test_failed_audit_invalidates_previous_success(audit_module, tmp_path):
    (tmp_path / "audit.json").write_text('{"passed": true}')
    args = SimpleNamespace(output=tmp_path, recipe_json=tmp_path / "missing.json")
    with pytest.raises((FileNotFoundError, ImportError)):
        audit_module.audit(args)
    assert json.loads((tmp_path / "audit.json").read_text())["passed"] is False
