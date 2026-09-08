"""Run deterministic PPO arithmetic and inspect bundled evidence; never train."""

import argparse
import json
import math
from pathlib import Path


BOOK = Path(__file__).resolve().parent


def rollout_budget(requested, environments, horizon, minibatches, epochs):
    values = (requested, environments, horizon, minibatches, epochs)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Budget inputs must be positive integers")
    batch = environments * horizon
    if batch % minibatches:
        raise ValueError("This exercise requires equally sized minibatches")
    iterations = (requested + batch - 1) // batch
    return {
        "requested_transitions": requested,
        "rollout_transitions": batch,
        "minibatch_transitions": batch // minibatches,
        "iterations": iterations,
        "actual_transitions": iterations * batch,
        "overshoot": iterations * batch - requested,
        "optimizer_steps": iterations * epochs * minibatches,
        "sample_presentations": iterations * batch * epochs,
        "aggregate_simulated_seconds": iterations * batch * 0.02,
    }


def clipped_surrogate(advantage, ratio, clip=0.2):
    if not all(math.isfinite(value) for value in (advantage, ratio, clip)):
        raise ValueError("Inputs must be finite")
    if ratio <= 0 or not 0 <= clip < 1:
        raise ValueError("Ratio must be positive and clip must be in [0, 1)")
    bounded_ratio = min(max(ratio, 1 - clip), 1 + clip)
    return min(ratio * advantage, bounded_ratio * advantage)


def scalar_gae(
    rewards,
    values,
    terminated,
    truncated,
    timeout_values,
    last_value,
    gamma=0.99,
    gae_lambda=0.95,
):
    sequences = (rewards, values, terminated, truncated, timeout_values)
    if not rewards or any(len(sequence) != len(rewards) for sequence in sequences):
        raise ValueError("GAE inputs must be nonempty and have matching lengths")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Discount and lambda must be in [0, 1]")
    advantages = [0.0] * len(rewards)
    next_value = last_value
    next_advantage = 0.0
    for index in reversed(range(len(rewards))):
        done = terminated[index] or truncated[index]
        if terminated[index]:
            bootstrap = 0.0
        elif truncated[index]:
            bootstrap = timeout_values[index]
        else:
            bootstrap = next_value
        delta = rewards[index] + gamma * bootstrap - values[index]
        next_advantage = delta + gamma * gae_lambda * (0.0 if done else next_advantage)
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages


def evidence_summary():
    summaries = {}
    for scenario in ("dance", "running", "stilts", "swing"):
        audit = json.loads((BOOK / f"assets/{scenario}-audit.json").read_text())
        recipe = audit["recipe"]
        training = audit["training"]
        budget = rollout_budget(
            training.get("requested_transitions", recipe["totalTimesteps"]),
            recipe["numEnvs"],
            recipe["numSteps"],
            recipe["numMinibatches"],
            recipe["updateEpochs"],
        )
        expected = (
            budget["actual_transitions"],
            budget["iterations"],
            budget["optimizer_steps"],
        )
        observed = (
            training["observed_transitions"],
            training["updates"],
            training["optimizer_steps"],
        )
        if expected != observed:
            raise ValueError(
                f"{scenario}: budget arithmetic disagrees with saved audit"
            )
        trials = audit["controls"]["trained"]
        summaries[scenario] = {
            "run_name": recipe["runName"],
            "budget": budget,
            "saved_passes": sum(trial["passed"] is True for trial in trials),
            "saved_trials": len(trials),
            "policy_sha256": audit["policy_sha256"],
            "budget_input_source": "training.requested_transitions"
            if "requested_transitions" in training
            else "recipe.totalTimesteps (effective)",
            "scope": "Selected phase only; saved evidence, not a fresh evaluation",
        }
    return summaries


def gae_walkthrough():
    rewards = [0.5, 0.2, 0.0]
    values = [1.0, 1.2, 1.1]
    final_value = 0.9
    gamma = 0.99
    gae_lambda = 0.95
    cases = {}
    for name, is_terminal in (("timeout", False), ("terminal", True)):
        terminated = [False, False, is_terminal]
        truncated = [False, False, not is_terminal]
        bootstraps = [1.2, 1.1, 0.0 if is_terminal else final_value]
        advantages = scalar_gae(
            rewards,
            values,
            terminated,
            truncated,
            [0.0, 0.0, final_value],
            last_value=99.0,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        cases[name] = {
            "td_residuals": [
                reward + gamma * bootstrap - value
                for reward, bootstrap, value in zip(rewards, bootstraps, values)
            ],
            "advantages": advantages,
            "value_targets": [
                advantage + value for advantage, value in zip(advantages, values)
            ],
        }
    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "rewards": rewards,
        "values_including_final_observation": [*values, final_value],
        **cases,
        "lambda_comparison": {
            str(trace_lambda): scalar_gae(
                rewards,
                values,
                [False] * 3,
                [False, False, True],
                [0.0, 0.0, final_value],
                last_value=99.0,
                gamma=gamma,
                gae_lambda=trace_lambda,
            )
            for trace_lambda in (0.0, 0.95, 1.0)
        },
        "scope": "Synthetic three-transition example; not measured robot rewards",
    }


def arithmetic_examples():
    return {
        "swing_budget": rollout_budget(524288, 16, 256, 4, 2),
        "hypothetical_four_million_request": rollout_budget(4000000, 16, 128, 4, 4),
        "surrogate": [
            {
                "advantage": advantage,
                "ratio": ratio,
                "maximize": clipped_surrogate(advantage, ratio),
            }
            for advantage in (2.0, -2.0)
            for ratio in (0.7, 1.0, 1.3)
        ],
        "terminal_advantage": scalar_gae([1.0], [2.0], [True], [False], [5.0], 99.0),
        "timeout_advantage": scalar_gae([1.0], [2.0], [False], [True], [5.0], 99.0),
        "scope": "Synthetic arithmetic examples, not robot performance measurements",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--evidence",
        action="store_true",
        help="Check all four bundled audits instead of synthetic examples",
    )
    selection.add_argument(
        "--gae",
        action="store_true",
        help="Compute the three-step timeout/terminal GAE lesson",
    )
    arguments = parser.parse_args()
    if arguments.evidence:
        result = evidence_summary()
    elif arguments.gae:
        result = gae_walkthrough()
    else:
        result = arithmetic_examples()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
