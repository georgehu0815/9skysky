import json

import mlx.core as mx
import numpy as np
import onnxruntime
from rlx.environments.microduck_recipes import make_single_recipe_env


mx.eval(mx.ones((2,)) + 1)
results = []
for scenario in ("dance", "swing", "running", "stilts"):
    environment = make_single_recipe_env(
        scenario,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
    )
    try:
        observation, _ = environment.reset(seed=0)
        if observation.shape != (61,) or not np.isfinite(observation).all():
            raise RuntimeError(f"{scenario}: invalid reset observation")
        if environment.action_space.shape != (14,):
            raise RuntimeError(f"{scenario}: invalid action contract")
        observation, reward, terminated, truncated, _ = environment.step(
            np.zeros(14, dtype=np.float32)
        )
        if observation.shape != (61,) or not np.isfinite(observation).all() or not np.isfinite(reward):
            raise RuntimeError(f"{scenario}: invalid transition")
        results.append({"scenario": scenario, "reset_step_passed": True})
    finally:
        environment.close()
print(json.dumps({"runtime_ready": True, "onnxruntime": onnxruntime.__version__, "scenarios": results}))
