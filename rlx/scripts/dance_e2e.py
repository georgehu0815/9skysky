"""Reproduce the dance experiment's reference and audit its saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(source: Path, output: Path, seconds: float) -> dict:
    from microduck_local.motion import load_clip

    source = source.resolve(strict=True)
    original = json.loads(source.read_text())
    if not 0 < seconds <= original["duration"]:
        raise ValueError("excerpt duration must lie within the source clip")
    clip = load_clip(source.stem, source.parent)
    steps = round(seconds * 50)
    keys = [key for key in original["keys"] if key["t"] < seconds]
    joints, pitch = clip.at(steps)
    keys.append({"t": seconds, "joints": joints.tolist(), "rootPitch": pitch})
    reference = {
        "version": original["version"],
        "name": f'{original["name"]} — first {seconds:g} seconds',
        "duration": seconds,
        "loop": True,
        "keys": keys,
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "reference.clip.json"
    target.write_text(json.dumps(reference, indent=2) + "\n")
    provenance = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_seconds": original["duration"],
        "reference": str(target.resolve()),
        "reference_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "excerpt_start_seconds": 0,
        "excerpt_seconds": seconds,
        "transformation": "First contiguous excerpt; no joint amplitude scaling or teacher actions. Endpoint interpolated at 50 Hz; looping flag retained.",
        "scope": "Joint angles and rootPitch only. Existing loader ignores rootYaw, rootRoll, rootPosition and contact annotations. Not full-source choreography.",
        "joint_std_rad": np.std(clip.joints[:steps], axis=0).tolist(),
    }
    (output / "reference-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--run-name", default="dance-e2e-reproduction")
    parser.add_argument("--train-steps", type=int, default=1_001_472)
    args = parser.parse_args()
    if args.train_steps < 2048 or args.train_steps % 2048:
        parser.error("--train-steps must be a positive multiple of the 2048-transition rollout")
    provenance = prepare(args.source, args.output, args.seconds)
    recipe = {
        "experimentId": "dance", "runName": args.run_name, "profile": "full",
        "danceClip": provenance["reference"], "dancePoseSigma": 0.2,
        "totalTimesteps": args.train_steps, "numEnvs": 16, "numSteps": 128,
        "numMinibatches": 4, "updateEpochs": 4, "seed": 7,
        "maxEpisodeS": args.seconds, "evalSteps": round(args.seconds * 50),
        "renderSeconds": args.seconds, "learningRate": 1e-4, "gamma": .99,
        "clipCoefficient": .2, "entropyCoefficient": 0, "maxGradNorm": .5,
        "initialStd": .1, "normalizeRewards": True, "checkpointInterval": 200000,
        "domainRand": False, "obsNoise": False, "actionDelay": False,
        "randomYaw": False, "resumeFromCheckpoint": False,
        "rewardWeights": {"pose_match": 40, "travel": 0, "save_energy": .0005},
    }
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
