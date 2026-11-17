#!/usr/bin/env python3
"""Evaluate a Kinetix `l/h2_car_ramp` actor against the level, on this machine.

Reads ``eval_config.json`` at the repository root (the file every experiment
node on this project may vary on its branch; the run command itself is fixed):

.. code-block:: json

    {
      "label": "H_base",
      "episodes": 1024,            // default episodes per eval (overridable per eval)
      "max_steps": 256,            // optional, default 256
      "weights": "weights/base_ppo_final_seed0.npz",   // default weights (overridable per eval)
      "format": "params",          // "params" (flax ppo checkpoint) | "actor" (14 actor keys)
      "evals": [                   // list of evaluations to run in one process
        {"seed": 0, "argmax": false, "tag": "s0-sampled"},
        {"weights": "weights/other.npz", "seed": 0, "argmax": true, "episodes": 256, "tag": "other-argmax"}
      ]
    }

Prints one ``RESULT {…}`` JSON line per evaluation (parseable by grep/jq) and a
final ``SUMMARY {…}`` line. No files are written except what stdout carries.

Environment: Kinetix ``kinetix-env==3.0.2`` on JAX CPU. Episodes are vmapped;
outcome indexing uses the FIRST done step (episodes that never finish count as
timeouts, per first-done-step indexing); the green/red "goal reached" flag is
read from the per-step ``info["GoalR"]`` dictionary, matching the protocol used
for the prior Experiment-4 measurements.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parent.parent

ENTRY_DIMS = (3, 3, 3, 3, 2, 2)
ENTRY_SLICES: list[slice] = []
_o = 0
for _d in ENTRY_DIMS:
    ENTRY_SLICES.append(slice(_o, _o + _d))
    _o += _d

NAME_MAP = {  # flax checkpoint key -> actor key
    "embed.W": "params/Dense_0/kernel",
    "embed.b": "params/Dense_0/bias",
    "actor_head.W": "params/GeneralActorCriticRNN_0/Dense_0/kernel",
    "actor_head.b": "params/GeneralActorCriticRNN_0/Dense_0/bias",
}
for _i in range(5):
    NAME_MAP[f"actor{_i}.W"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{_i}/Dense_0/kernel"
    NAME_MAP[f"actor{_i}.b"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{_i}/Dense_0/bias"


def load_actor_weights(path: Path, fmt: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    if fmt == "params":
        out = {}
        for k, fk in NAME_MAP.items():
            out[k] = d[fk].astype(np.float32)
        return out
    elif fmt == "actor":
        return {k: d[k].astype(np.float32) for k in d.files}
    raise ValueError(f"unknown format {fmt!r} (expected 'params' or 'actor')")


def actor_logits(W, obs):
    x = jax.nn.relu(obs @ W["embed.W"] + W["embed.b"])
    for i in range(5):
        x = jnp.tanh(x @ W[f"actor{i}.W"] + W[f"actor{i}.b"])
    return x @ W["actor_head.W"] + W["actor_head.b"]


def build_env():
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.environment.env_state import EnvParams
    from kinetix.environment.utils import static_env_params_from_size
    from kinetix.util.saving import load_evaluation_levels

    env_params = EnvParams().replace(dense_reward_scale=0.2)
    static_env_params = static_env_params_from_size("l")
    levels, static_env_params = load_evaluation_levels(["l/h2_car_ramp"])
    env = make_kinetix_env(
        ActionType.MULTI_DISCRETE,
        ObservationType.SYMBOLIC_FLAT,
        None,
        env_params,
        static_env_params,
        auto_reset=False,
    )
    return env, env_params, static_env_params, levels


def make_rollout(env, env_params, static_env_params, levels, max_steps: int):
    import functools

    level = jax.tree.map(lambda x: x[0], levels)  # drop leading level-batch dim

    @functools.partial(jax.jit, static_argnames=("episodes", "argmax", "max_steps"))
    def rollout(W, key, episodes: int, argmax: bool, max_steps: int = max_steps):
        level_tiled = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (episodes,) + x.shape) if x.ndim else jnp.full((episodes,), x),
            level,
        )
        rng, reset_rng = jax.random.split(key)
        obs, state = jax.vmap(env.reset, in_axes=(0, None, 0))(
            jax.random.split(reset_rng, episodes), env_params, level_tiled
        )

        def step_fn(carry, _):
            obs, state, done, rng = carry
            logits = actor_logits(W, obs.astype(jnp.float32))
            rng, *sub = jax.random.split(rng, 7)
            acts = []
            for i, s in enumerate(ENTRY_SLICES):
                lg = logits[:, s]
                if argmax:
                    acts.append(jnp.argmax(lg, axis=-1))
                else:
                    acts.append(jax.random.categorical(sub[i], lg))
            action = jnp.stack(acts, axis=-1).astype(jnp.int32)
            new_obs, new_state, reward, new_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None, None)
            )(jax.random.split(rng, episodes), state, action, env_params, None)
            done = done | new_done
            return (new_obs, new_state, done, rng), (reward, new_done, info["GoalR"])

        init = (obs, state, jnp.zeros((episodes,), dtype=bool), rng)
        _, (rewards, dones, goalrs) = jax.lax.scan(step_fn, init, None, length=max_steps)

        done_any = dones.any(axis=0)  # (N,)
        first_idx = jnp.argmax(dones.astype(jnp.int32), axis=0)
        success = goalrs[first_idx, jnp.arange(episodes)]
        first_reward = rewards[first_idx, jnp.arange(episodes)]
        # episode return = sum of rewards up to and including the first done step
        cum = jnp.cumsum(rewards, axis=0)
        ep_return = cum[first_idx, jnp.arange(episodes)]
        red = (~success) & (first_reward < -0.5)
        timeout = (~success) & (~red)
        ep_len = first_idx + 1
        return {
            "success": success.astype(jnp.float32).mean(),
            "red": red.astype(jnp.float32).mean(),
            "timeout": timeout.astype(jnp.float32).mean(),
            "mean_len": ep_len.astype(jnp.float32).mean(),
            "mean_return": ep_return.astype(jnp.float32).mean(),
            "success_count": success.astype(jnp.int32).sum(),
            "done_any": done_any.astype(jnp.float32).mean(),
        }

    return rollout


def main() -> int:
    cfg_path = REPO / "eval_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    default_episodes = int(cfg.get("episodes", 1024))
    default_max_steps = int(cfg.get("max_steps", 256))
    default_weights = cfg.get("weights")
    default_format = cfg.get("format", "actor")
    label = cfg.get("label", "unlabeled")

    env, env_params, static_env_params, levels = build_env()
    rollout = make_rollout(env, env_params, static_env_params, levels, default_max_steps)

    header = {
        "label": label,
        "orx_run_id": os.environ.get("ORX_RUN_ID"),
        "orx_commit": os.environ.get("ORX_COMMIT"),
        "orx_experiment": os.environ.get("ORX_EXPERIMENT"),
        "kinematics": "l/h2_car_ramp symbolic_flat multi_discrete frame_skip=2 dense_reward_scale=0.2 auto_reset=false",
        "max_steps": default_max_steps,
    }
    print("CONFIG " + json.dumps(header), flush=True)

    results = []
    weights_cache: dict[tuple[str, str], dict] = {}
    for ev in cfg["evals"]:
        weights_rel = ev.get("weights", default_weights)
        fmt = ev.get("format", default_format)
        episodes = int(ev.get("episodes", default_episodes))
        seed = int(ev["seed"])
        argmax = bool(ev.get("argmax", False))
        tag = ev.get("tag", f"{Path(weights_rel).stem}-s{seed}{'-argmax' if argmax else ''}")
        if weights_rel is None:
            raise ValueError("eval entry with no 'weights' and no top-level default")
        key = (weights_rel, fmt)
        if key not in weights_cache:
            weights_cache[key] = load_actor_weights(REPO / weights_rel, fmt)
        W = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in weights_cache[key].items()}
        prng = jax.random.PRNGKey(seed)
        t0 = time.time()
        out = rollout(W, prng, episodes=episodes, argmax=argmax, max_steps=default_max_steps)
        wall = time.time() - t0
        res = {
            "label": label,
            "tag": tag,
            "weights": weights_rel,
            "format": fmt,
            "episodes": episodes,
            "seed": seed,
            "argmax": argmax,
            "success": float(out["success"]),
            "success_count": int(out["success_count"]),
            "red": float(out["red"]),
            "timeout": float(out["timeout"]),
            "mean_len": float(out["mean_len"]),
            "mean_return": float(out["mean_return"]),
            "done_frac": float(out["done_any"]),
            "wall_s": wall,
        }
        results.append(res)
        print("RESULT " + json.dumps(res), flush=True)

    print("SUMMARY " + json.dumps({"label": label, "n_evals": len(results), "results": results}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
