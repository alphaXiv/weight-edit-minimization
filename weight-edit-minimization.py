# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "marimo",
#     "numpy==2.4.6",
#     "matplotlib",
#     "jax==0.9.0",
#     "kinetix-env==3.0.2",
#     "torch",
#     "transformers",
#     "huggingface_hub",
#     "math-verify",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Weight-edit minimization — replication notebook",
)


@app.cell
def _():
    import json
    import math
    import os as _os
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    REPO = "alphaXiv/weight-edit-minimization"


    def _find_root():
        here = mo.notebook_dir() or Path.cwd()
        if (here / "data").is_dir():
            return here
        # molab can run the notebook file without the rest of the repo; fetch the repo once
        import io
        import tarfile
        import tempfile
        import urllib.request

        dest = Path(tempfile.gettempdir()) / "weight-edit-minimization"
        if not (dest / "data").is_dir():
            with mo.status.spinner(f"downloading the notebook's data from github.com/{REPO}"):
                with urllib.request.urlopen(f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main", timeout=300) as _r:
                    _buf = io.BytesIO(_r.read())
            with tarfile.open(fileobj=_buf) as _t:
                _members = []
                for _m in _t.getmembers():
                    _m.name = _m.name.partition("/")[2]  # drop the "<repo>-main/" prefix
                    if _m.name:
                        _members.append(_m)
                _t.extractall(dest, members=_members, filter="data")
        return dest


    ROOT = _find_root()
    # matplotlib reads the viewer's ~/.matplotlib/matplotlibrc unless pointed elsewhere; use the bundled one
    _os.environ["MATPLOTLIBRC"] = str(ROOT / "figs" / "matplotlibrc")

    import matplotlib.pyplot as plt
    import numpy as np

    CR = ROOT / "data" / "car-ramp"
    EX = ROOT / "data" / "exp8"
    sys.path.insert(0, str(ROOT / "figs"))
    sys.path.insert(0, str(ROOT / "code"))
    import orx_figstyle as house

    house.use_style()
    # the house style is sized for print columns; notebook figures need larger text
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    PALETTE, GREY = house.PALETTE, house.BASELINE
    return CR, EX, GREY, PALETTE, ROOT, json, math, mo, np, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # Weight-edit minimization — replication notebook

    Two campaigns, one question: **what is the smallest set of weights whose edit produces the behavior?**
    Both start from an edit written by Astra (`RikiyaT/lm_understand-logs`) and strip it down.

    Every number below is **recomputed in this notebook** from the checkpoints, edit files and raw
    per-problem outputs in `data/` — and the expensive parts (Kinetix rollouts, Qwen generations) have
    **run buttons** so you can re-run the experiments, not just read about them. Until you press one, its section
    shows the recorded result from the original runs.

    | section | what you can run here | cost on a laptop |
    |:--|:--|:--|
    | §1.1 car-ramp networks | rebuild all 8 networks from their edit recipes, bit-check against the shipped checkpoints | instant |
    | §1.3 one evaluation | the exact Kinetix harness from the orx runs, any network / seed / episode count | 10–45 s |
    | §1.4 the ladder | 6,073 → 1,939 → 1,718 → 12 → 11 and 14 → 10, side by side with the recorded runs | ~2 min @ 256 ep |
    | §1.5 drop-one audit | remove each of the 12 entries in turn | ~2.5 min |
    | §1.6 embed-restore ablation | un-erase one repurposed embed unit at a time | ~40 s |
    | §1.7 mechanism trace | per-step car state, gate units and action probabilities for one episode | ~15 s |
    | §1.8 edit playground | write your own entries, evaluate them | 10–45 s |
    | §2.1–2.3 Exp 8 edit | apply Astra's bf16 edit, recover its rank-1 formula, regenerate & bit-count all 27 campaign candidates | ~20 s |
    | §2.4 screening re-analysis | accuracy, cut share, paired McNemar tests and bootstrap CIs from the 29 raw screens | instant |
    | §2.6 run Qwen on this machine | paired base-vs-candidate generations on screening problems (Apple GPU / CUDA) | minutes |
    | §2.7 GPU replication | write any candidate as a checkpoint + the exact vLLM screening command | — |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1 · Car-ramp: a PPO actor that never solves the level, fixed by 10–12 weights

    Kinetix `l/h2_car_ramp`: back the car up, drive into the red wall, fly the ramp, stop in the green box.
    The PPO actor (1.001 B steps) **never** does it. Astra's hand-written edit changes 6,073 entries and
    solves it every time. The campaign asked how many of those entries matter.

    **Rounding convention.** Astra saw each tensor rounded to 3 significant digits
    (`scale = 10**(floor(log10(max|W|)) - 2)`), so every edited network is *rounded base + entries*, and
    entry counts are against the rounded base.
    """)
    return


@app.cell
def _(CR, json, np):
    NAME_MAP = {
        "embed.W": "params/Dense_0/kernel",
        "embed.b": "params/Dense_0/bias",
        "actor_head.W": "params/GeneralActorCriticRNN_0/Dense_0/kernel",
        "actor_head.b": "params/GeneralActorCriticRNN_0/Dense_0/bias",
    }
    for _i in range(5):
        NAME_MAP[f"actor{_i}.W"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{_i}/Dense_0/kernel"
        NAME_MAP[f"actor{_i}.b"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{_i}/Dense_0/bias"


    def round3(x):
        scale = 10.0 ** (np.floor(np.log10(np.abs(x).max())) - 2)
        return np.round(x / scale) * scale


    _base = np.load(CR / "base_ppo_final_seed0.npz")
    base_raw = {k: _base[v].astype(np.float64) for k, v in NAME_MAP.items()}
    rounded = {k: round3(v) for k, v in base_raw.items()}
    _ed = np.load(CR / "edited_actor.npz")
    astra = {k: _ed[k].astype(np.float64) for k in _ed.files}


    def apply_entries(entries):
        """Rounded base + entries. Each entry is (tensor, index, value); value None = Astra's value there."""
        W = {k: v.copy() for k, v in rounded.items()}
        for t, idx, val in entries:
            idx = tuple(idx)
            W[t][idx] = astra[t][idx] if val is None else val
        return W


    def changed_entries(W):
        return [(k, tuple(int(i) for i in ix)) for k in rounded for ix in zip(*np.nonzero(W[k] != rounded[k]))]


    # --- the recipes ---------------------------------------------------------
    KEEP_1939 = [(t, tuple(i), None) for t, i in json.load(open(CR / "keep_joint_free.json"))]
    GREEDY12 = [(t, tuple(i), None) for t, i in json.load(open(CR.parent / "car-ramp-evidence" / "greedy_gf5_final.json"))]
    GREEDY11 = [e for e in GREEDY12 if (e[0], e[1]) != ("actor1.W", (122, 126))]
    C2_14 = [(t, tuple(i), float(v)) for t, i, v in json.load(open(CR / "c2_edits.json"))]
    _C2_DROPPED = {("embed.W", (52, 26)), ("actor0.W", (26, 114)), ("actor0.b", (23,)), ("actor0.b", (114,))}
    C2_MIN_10 = [e for e in C2_14 if (e[0], e[1]) not in _C2_DROPPED]
    # the 1,718 waypoint was found by a greedy keep/drop search whose keep-list was not saved; take its
    # entry set from the shipped checkpoint (every value equals Astra's, checked below)
    _wp = np.load(CR / "greedy_waypoint_1718.npz")
    WAYPOINT_1718 = [(t, i, None) for t, i in changed_entries({k: _wp[k].astype(np.float64) for k in _wp.files})]

    networks = {
        "base PPO": base_raw,
        "Astra edit": astra,
        "joint_free": apply_entries(KEEP_1939),
        "waypoint": apply_entries(WAYPOINT_1718),
        "greedy-12": apply_entries(GREEDY12),
        "greedy-11": apply_entries(GREEDY11),
        "C2, from scratch": apply_entries(C2_14),
        "C2-min, from scratch": apply_entries(C2_MIN_10),
    }
    SHIPPED = {
        "base PPO": "base_ppo_final_seed0.npz",
        "Astra edit": "edited_actor.npz",
        "joint_free": "joint_free.npz",
        "waypoint": "greedy_waypoint_1718.npz",
        "greedy-12": "greedy12_astra.npz",
        "greedy-11": "greedy11_drop02.npz",
        "C2, from scratch": "c2_14entry.npz",
        "C2-min, from scratch": "c2_min_10entry.npz",
    }
    return (
        C2_MIN_10,
        GREEDY12,
        KEEP_1939,
        NAME_MAP,
        SHIPPED,
        apply_entries,
        astra,
        changed_entries,
        networks,
        rounded,
    )


@app.cell
def _(
    C2_MIN_10,
    CR,
    GREEDY12,
    NAME_MAP,
    SHIPPED,
    astra,
    changed_entries,
    mo,
    networks,
    np,
    rounded,
):
    _rows = []
    for _name, _W in networks.items():
        _z = np.load(CR / SHIPPED[_name])
        _ship = {k: _z[NAME_MAP[k] if _name == "base PPO" else k].astype(np.float64) for k in rounded}
        _same = all(np.array_equal(_W[k], _ship[k]) for k in rounded)
        _n = 0 if _name == "base PPO" else len(changed_entries(_W))
        _rows.append({"network": _name, "entries vs rounded base": f"{_n:,}",
                      "rebuilt == shipped checkpoint": "✅ bit-identical" if _same else "❌ differs",
                      "file": SHIPPED[_name]})
    _wp_astra = all(networks["waypoint"][t][i] == astra[t][i] for t, i in changed_entries(networks["waypoint"]))

    def _fmt(entries):
        return "<br>".join(f"`{t}[{','.join(map(str, i))}]` = {astra[t][i] if v is None else v:g}" for t, i, v in entries)

    mo.vstack([
        mo.md("### 1.1 Build every network from its recipe\n\nRecipes live in the cell above (`KEEP_1939`, `GREEDY12`, `C2_14`, …). Each is rebuilt from the base checkpoint and compared bit-for-bit with the network the orx runs evaluated:"),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"{'✅' if _wp_astra else '❌'} Every one of the waypoint's 1,718 values is Astra's edited value."),
        mo.hstack([
            mo.md("**greedy-12** (a subset of Astra's own edit)<br>" + _fmt(GREEDY12)),
            mo.md("**C2-min-10** (written from scratch, zero erasures)<br>" + _fmt(C2_MIN_10)),
        ], widths="equal"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 The evaluation harness

    This is `tools/eval_actor.py` from the orx experiment branches (`code/car_ramp_eval_actor.py`), the
    fixed run command of every car-ramp node: official Kinetix (`kinetix-env==3.0.2`, JAX CPU),
    `symbolic_flat` observations (679-d), `multi_discrete` actions (entry 0 drives the motor), frame-skip 2,
    256 steps, outcome read at the **first** done step from `info["GoalR"]`. `trace_episode` is the same
    rollout for a single episode, also recording the car state and every hidden unit.
    """)
    return


@app.cell
def _(np, time):
    import functools
    import logging as _logging

    _logging.getLogger("absl").setLevel(_logging.ERROR)  # Kinetix's grain import warns about profiling support
    import jax
    import jax.numpy as jnp
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.env_state import EnvParams
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.util.saving import load_evaluation_levels

    ENTRY_SLICES = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 14), slice(14, 16)]
    OBS_X, OBS_VX, OBS_GX = 52, 54, 234  # car-body x, car-body vx, green-wheel x (see report.md)


    def actor_forward(W, obs):
        h = [jax.nn.relu(obs @ W["embed.W"] + W["embed.b"])]
        for i in range(5):
            h.append(jnp.tanh(h[-1] @ W[f"actor{i}.W"] + W[f"actor{i}.b"]))
        return h[-1] @ W["actor_head.W"] + W["actor_head.b"], h


    _env_params = EnvParams().replace(dense_reward_scale=0.2)
    _levels, _static = load_evaluation_levels(["l/h2_car_ramp"])
    _env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None, _env_params, _static, auto_reset=False)
    _level = jax.tree.map(lambda x: x[0], _levels)


    def _reset(key, episodes):
        tiled = jax.tree.map(lambda x: jnp.broadcast_to(x, (episodes,) + x.shape) if x.ndim else jnp.full((episodes,), x), _level)
        return jax.vmap(_env.reset, in_axes=(0, None, 0))(jax.random.split(key, episodes), _env_params, tiled)


    def _act(logits, rng, argmax):
        # same key schedule as eval_actor.py, so sampled runs reproduce the recorded ones exactly
        rng, *sub = jax.random.split(rng, 7)
        acts = [jnp.argmax(logits[:, s], -1) if argmax else jax.random.categorical(sub[i], logits[:, s]) for i, s in enumerate(ENTRY_SLICES)]
        return jnp.stack(acts, -1).astype(jnp.int32), rng


    @functools.partial(jax.jit, static_argnames=("episodes", "argmax"))
    def _rollout(W, key, episodes, argmax):
        rng, reset_rng = jax.random.split(key)
        obs, state = _reset(reset_rng, episodes)

        def step(carry, _):
            obs, state, done, rng = carry
            logits, _ = actor_forward(W, obs.astype(jnp.float32))
            action, rng = _act(logits, rng, argmax)
            obs, state, reward, new_done, info = jax.vmap(_env.step, in_axes=(0, 0, 0, None, None))(
                jax.random.split(rng, episodes), state, action, _env_params, None)
            return (obs, state, done | new_done, rng), (reward, new_done, info["GoalR"])

        _, (rewards, dones, goal) = jax.lax.scan(step, (obs, state, jnp.zeros(episodes, bool), rng), None, length=256)
        first = jnp.argmax(dones.astype(jnp.int32), 0)
        ar = jnp.arange(episodes)
        success = goal[first, ar]
        red = (~success) & (rewards[first, ar] < -0.5)
        return {"success_count": success.sum(), "red": red.mean(), "timeout": ((~success) & (~red)).mean(),
                "mean_len": (first + 1).mean(), "mean_return": jnp.cumsum(rewards, 0)[first, ar].mean()}


    def evaluate(W, episodes=256, seed=0, argmax=False):
        """One evaluation, as `eval_actor.py` runs it. W: tensor name -> array."""
        Wj = {k: jnp.asarray(v, jnp.float32) for k, v in W.items()}
        t0 = time.perf_counter()
        out = {k: float(v) for k, v in _rollout(Wj, jax.random.PRNGKey(seed), episodes, argmax).items()}
        out.update(episodes=episodes, seed=seed, argmax=argmax, success_count=int(out["success_count"]),
                   wall_s=round(time.perf_counter() - t0, 1))
        return out


    @jax.jit
    def _trace(W, key):
        rng, reset_rng = jax.random.split(key)
        obs, state = _reset(reset_rng, 1)

        def step(carry, _):
            obs, state, done, rng = carry
            logits, h = actor_forward(W, obs.astype(jnp.float32))
            action, rng = _act(logits, rng, True)
            rec = (obs[0, jnp.array([OBS_X, OBS_VX, OBS_GX])], jnp.stack(h)[:, 0], jax.nn.softmax(logits[0, :3]), action[0, 0], done[0])
            obs, state, _, new_done, info = jax.vmap(_env.step, in_axes=(0, 0, 0, None, None))(
                jax.random.split(rng, 1), state, action, _env_params, None)
            return (obs, state, done | new_done, rng), rec + (info["GoalR"][0],)

        _, rec = jax.lax.scan(step, (obs, state, jnp.zeros(1, bool), rng), None, length=256)
        return rec


    def trace_episode(W, seed=0):
        """Argmax episode: car state (x, vx, green-x), hidden activations [layer, unit], p(action 0..2), outcome."""
        obs, h, probs, action, done, goal = (np.asarray(a) for a in _trace({k: jnp.asarray(v, jnp.float32) for k, v in W.items()}, jax.random.PRNGKey(seed)))
        n = int(np.argmax(done)) if done.any() else 256
        return {"obs": obs[:n], "h": h[:n], "probs": probs[:n], "action": action[:n], "steps": n,
                "solved": bool(goal[max(n - 1, 0)]) if done.any() else False}

    return evaluate, trace_episode


@app.cell
def _(ROOT, json):
    import re

    recorded = []
    for _f in sorted((ROOT / "data" / "logs").glob("*.log")):
        for _m in re.finditer(r"RESULT (\{.*\})", _f.read_text()):
            recorded.append(json.loads(_m.group(1)))


    def recorded_result(weights_file, seed=0, argmax=False, episodes=None):
        for r in recorded:
            if r["weights"].endswith(weights_file) and r["seed"] == seed and r["argmax"] == argmax and (episodes is None or r["episodes"] == episodes):
                return r
        return None

    return recorded, recorded_result


@app.cell
def _(mo, networks):
    one_net = mo.ui.dropdown(list(networks), value="greedy-12", label="network")
    one_eps = mo.ui.dropdown(["64", "256", "1024"], value="256", label="episodes")
    one_seed = mo.ui.number(0, 99, value=0, label="seed")
    one_argmax = mo.ui.checkbox(label="argmax")
    one_run = mo.ui.run_button(label="▶ run")
    mo.vstack([mo.md("### 1.3 Run one evaluation"), mo.hstack([one_net, one_eps, one_seed, one_argmax], justify="start"), one_run])
    return one_argmax, one_eps, one_net, one_run, one_seed


@app.cell
def _(
    SHIPPED,
    evaluate,
    mo,
    networks,
    one_argmax,
    one_eps,
    one_net,
    one_run,
    one_seed,
    recorded_result,
):
    _rec = recorded_result(SHIPPED[one_net.value], int(one_seed.value), one_argmax.value)
    _rec_txt = (f"Recorded orx run (`{_rec['label']}`, {_rec['episodes']} episodes): **{_rec['success_count']}/{_rec['episodes']}** solved, mean length {_rec['mean_len']:.1f}."
                if _rec else "No recorded orx run with this seed and action mode.")
    mo.stop(not one_run.value, mo.md(f"{_rec_txt}<br>_Press **run** to run it here (the first run compiles for ~10 s)._"))
    _r = evaluate(networks[one_net.value], int(one_eps.value), int(one_seed.value), one_argmax.value)
    mo.md(f"**Re-run now:** solved **{_r['success_count']}/{_r['episodes']}** · red wall {100*_r['red']:.1f}% · timeout {100*_r['timeout']:.1f}% · "
          f"mean length {_r['mean_len']:.1f} steps · {_r['wall_s']} s<br>{_rec_txt}")
    return


@app.cell
def _(mo):
    ladder_eps = mo.ui.dropdown(["256", "1024"], value="256", label="episodes per network")
    ladder_run = mo.ui.run_button(label="▶ run the whole ladder")
    mo.vstack([
        mo.md("### 1.4 The ladder — every rung, re-run\n\nAll 8 networks at seed 0, sampled actions. 256 episodes ≈ 12 s per network; 1,024 ≈ 45 s."),
        mo.hstack([ladder_eps, ladder_run], justify="start"),
    ])
    return ladder_eps, ladder_run


@app.cell
def _(
    PALETTE,
    SHIPPED,
    changed_entries,
    evaluate,
    ladder_eps,
    ladder_run,
    mo,
    networks,
    np,
    plt,
    recorded_result,
):
    _live = ladder_run.value
    _eps = int(ladder_eps.value)
    _items = list(networks.items())
    _rows, _rec_rate, _live_rate = [], [], []
    for _name, _W in (mo.status.progress_bar(_items, title="evaluating") if _live else _items):
        _rec = recorded_result(SHIPPED[_name], 0, False, 1024)
        _row = {"network": _name, "entries": 0 if _name == "base PPO" else len(changed_entries(_W)),
                "recorded orx run (1,024 ep, seed 0)": f"{_rec['success_count']}/1024 · len {_rec['mean_len']:.0f}"}
        _rec_rate.append(_rec["success_count"] / 1024)
        if _live:
            _r = evaluate(_W, _eps, 0)
            _row[f"re-run now ({_eps} ep, seed 0)"] = f"{_r['success_count']}/{_eps} · len {_r['mean_len']:.0f}"
            _live_rate.append(_r["success_count"] / _eps)
        _rows.append(_row)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _y = np.arange(len(_rows))[::-1]
    _h = 0.38 if _live else 0.6
    _series = [("recorded orx run", _rec_rate, PALETTE["orange"], -_h / 2 if _live else 0)]
    if _live:
        _series.append(("re-run now", _live_rate, PALETTE["blue"], _h / 2))
    for _lab, _vals, _col, _off in _series:
        _ax.barh(_y + _off, _vals, _h, color=_col, label=_lab)
        for _yy, _v in zip(_y + _off, _vals):
            _ax.text(_v + 0.01, _yy, f"{100 * _v:.0f}%", va="center", fontsize=8)
    _ax.set_yticks(_y, [f"{r['network']} — {r['entries']:,} entries" for r in _rows])
    _ax.set_xlabel("episodes solved")
    _ax.set_xlim(0, 1.12)
    _ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, frameon=False)
    _fig.tight_layout()
    mo.vstack([
        mo.md("_Showing the recorded orx runs. Press **run the whole ladder** to re-run every network here._" if not _live
              else f"_Re-ran all {len(_rows)} networks just now. Sampled episodes use the same seed and key schedule as the orx runs, so at 1,024 episodes the counts match exactly._"),
        mo.ui.table(_rows, selection=None, pagination=False), _fig,
    ])
    return


@app.cell
def _(mo):
    drop_run = mo.ui.run_button(label="▶ run drop-one audit (12 × 256 episodes)")
    mo.vstack([
        mo.md("### 1.5 Drop-one audit of greedy-12\n\nRevert each of the 12 entries to its rounded-base value and re-evaluate (256 episodes, seed 0) — the `H_greedy12_min` experiment."),
        drop_run,
    ])
    return (drop_run,)


@app.cell
def _(
    GREEDY12,
    PALETTE,
    apply_entries,
    drop_run,
    evaluate,
    mo,
    np,
    plt,
    recorded,
):
    _live = drop_run.value
    _logged = {r["tag"]: r for r in recorded if r["label"] == "H_greedy12_min"}
    _rows = []
    for _i in (mo.status.progress_bar(range(12), title="drop-one") if _live else range(12)):
        _t, _idx, _ = GREEDY12[_i]
        _l = _logged[f"drop-{_i:02d}"]
        _row = {"dropped entry": f"{_t}[{','.join(map(str, _idx))}]",
                "recorded orx run (256 ep)": f"{_l['success_count']}/256 · len {_l['mean_len']:.0f}", "_rec": _l["success_count"]}
        if _live:
            _r = evaluate(apply_entries([e for j, e in enumerate(GREEDY12) if j != _i]), 256, 0)
            _row["re-run now"] = f"{_r['success_count']}/256 · len {_r['mean_len']:.0f}"
            _row["_now"] = _r["success_count"]
        _rows.append(_row)
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _y = np.arange(12)[::-1]
    _h = 0.38 if _live else 0.6
    _series = [("recorded orx run", [r["_rec"] for r in _rows], PALETTE["orange"], -_h / 2 if _live else 0)]
    if _live:
        _series.append(("re-run now", [r["_now"] for r in _rows], PALETTE["blue"], _h / 2))
    for _lab, _vals, _col, _off in _series:
        _ax.barh(_y + _off, [v / 256 for v in _vals], _h, color=_col, label=_lab)
        for _yy, _v in zip(_y + _off, _vals):
            _ax.text(_v / 256 + 0.01, _yy, f"{_v}/256", va="center", fontsize=8)
    _ax.set_yticks(_y, [r["dropped entry"] for r in _rows])
    _ax.set_xlabel("episodes solved with this entry removed")
    _ax.set_xlim(0, 1.15)
    _ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, frameon=False)
    _fig.tight_layout()
    mo.vstack([
        mo.md("_Showing the recorded orx run. Press the button to re-run it here._" if not _live else "_Re-ran all 12 just now._"),
        _fig,
        mo.md("Ten entries are each indispensable (0/256 without them). `actor1.W[122,126]` is redundant: dropping it gives the 11-entry edit. "
              "Dropping `actor_head.b[0]` costs 24 episodes."),
        mo.ui.table([{k: v for k, v in r.items() if not k.startswith("_")} for r in _rows], selection=None, pagination=False),
    ])
    return


@app.cell
def _(KEEP_1939, apply_entries, astra, mo):
    restore_run = mo.ui.run_button(label="▶ run embed-restore ablation (3 × 256 episodes)")
    # joint_free keeps 1,884 embed.W erases: Astra zeroed most of embed columns 6/7/8 so those dead units
    # read only one observation each (car x, car vx, green-wheel x). Restoring a column = undo its erases.
    restore_nets = {
        _c: apply_entries([e for e in KEEP_1939 if not (e[0] == "embed.W" and e[1][1] == _c and astra["embed.W"][e[1]] == 0)])
        for _c in (6, 7, 8)
    }
    mo.vstack([
        mo.md("### 1.6 Why the 1,884 embed erases matter (`H_embed_restore`)\n\n"
              "In `joint_free`, the only large group left is 1,884 zeroed entries in embed columns 6/7/8. "
              "They turn three dead ReLU units into clean single-input taps (x, vx, green-x). Undo one column's erases:"),
        restore_run,
    ])
    return restore_nets, restore_run


@app.cell
def _(evaluate, mo, recorded, restore_nets, restore_run):
    _logged = {r["tag"]: r for r in recorded if r["label"] == "H_embed_restore"}
    _rows = []
    for _c, _W in restore_nets.items():
        _l = _logged[f"restore-col{_c}-s0"]
        _row = {"restored column": _c, "unit reads": {6: "car x", 7: "car vx", 8: "green-wheel x"}[_c],
                "recorded orx run (256 ep)": f"{_l['success_count']}/256 · len {_l['mean_len']:.0f}"}
        if restore_run.value:
            _r = evaluate(_W, 256, 0)
            _row["re-run now"] = f"{_r['success_count']}/256 · len {_r['mean_len']:.0f} · {'red wall' if _r['red'] > 0.5 else 'timeout'}"
        _rows.append(_row)
    mo.vstack([
        mo.md("_Showing the recorded orx run. Press the button to re-run it here._" if not restore_run.value else "_Re-ran just now._"),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("Undoing any one column's erases breaks the task. `data/docs/car-ramp/report.md` traces each failure to the comparator that column feeds."),
    ])
    return


@app.cell
def _(mo, networks):
    trace_net = mo.ui.dropdown(list(networks), value="greedy-12", label="network")
    trace_units = mo.ui.text(value="0:122, 0:127, 1:126, 2:126", label="hidden units to plot (actor layer:unit)", full_width=True)
    mo.vstack([
        mo.md("### 1.7 Mechanism trace — what the patch does, step by step\n\n"
              "One argmax episode. Unit `L:u` is the tanh output of `actor{L}` unit `u`. For greedy-12 the chain is "
              "PPO's own speed detector `0:122` → gate `1:126` → relay `2:126`; the green comparator is `0:127`. "
              "For C2-min try `0:23, 1:29, 2:105, 4:73`. Base PPO for comparison."),
        mo.hstack([trace_net, trace_units], widths=[1, 2]),
    ])
    return trace_net, trace_units


@app.cell
def _(PALETTE, mo, networks, np, plt, trace_episode, trace_net, trace_units):
    _tr = trace_episode(networks[trace_net.value])
    _base = trace_episode(networks["base PPO"])
    _units = [tuple(int(x) for x in s.split(":")) for s in trace_units.value.replace(" ", "").split(",") if ":" in s]
    _fig, _axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    _t = np.arange(_tr["steps"])
    _axes[0].plot(_t, _tr["obs"][:, 0], color=PALETTE["blue"], label="car x")
    _axes[0].plot(_t, _tr["obs"][:, 1], color=PALETTE["orange"], label="car vx")
    _axes[0].plot(_t, _tr["obs"][:, 2], color=PALETTE["green"], label="green-wheel x")
    _axes[0].plot(np.arange(_base["steps"]), _base["obs"][:, 0], color=PALETTE["blue"], alpha=0.3, linestyle="--", label="car x (base PPO)")
    _axes[0].set_ylabel("observation")
    _axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    for _k, (_L, _u) in enumerate(_units):
        _axes[1].plot(_t, _tr["h"][:, _L + 1, _u], label=f"{_L}:{_u}", color=list(PALETTE.values())[_k % 7])
    _axes[1].set_ylabel("unit activation")
    _axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    for _a, _lab in enumerate(["motor action 0", "motor action 1", "motor action 2"]):
        _axes[2].plot(_t, _tr["probs"][:, _a], label=_lab)
    _axes[2].set_ylabel("action probability")
    _axes[2].set_xlabel("step")
    _axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _fig.tight_layout()
    mo.vstack([mo.md(f"**{trace_net.value}**: {'solved' if _tr['solved'] else 'not solved'} in {_tr['steps']} steps "
                     f"(base PPO: {'solved' if _base['solved'] else 'not solved'}, {_base['steps']} steps)."), _fig])
    return


@app.cell
def _(mo):
    play_text = mo.ui.text_area(
        value="\n".join([
            "# rounded base + these entries. Format: tensor[i,j] = value   (tensors: embed.W/b, actor0-4.W/b, actor_head.W/b)",
            "# C2-min-10, the from-scratch 10-entry fix. Delete a line, change a gain, or write your own circuit.",
            "embed.b[26] = 15.2", "actor0.W[116,23] = 8", "actor1.W[114,29] = 12", "actor1.W[23,29] = 12",
            "actor1.b[29] = -18", "actor2.W[29,105] = 10", "actor3.W[105,17] = 10", "actor4.W[17,73] = 10",
            "actor_head.W[73,1] = -40", "actor_head.W[73,2] = 40",
        ]),
        rows=14, full_width=True,
    )
    play_eps = mo.ui.dropdown(["64", "256", "1024"], value="256", label="episodes")
    play_run = mo.ui.run_button(label="▶ build & evaluate")
    mo.vstack([mo.md("### 1.8 Edit playground"), play_text, mo.hstack([play_eps, play_run], justify="start")])
    return play_eps, play_run, play_text


@app.cell
def _(
    apply_entries,
    evaluate,
    mo,
    play_eps,
    play_run,
    play_text,
    rounded,
    trace_episode,
):
    import re as _re

    _entries, _errors = [], []
    for _ln, _line in enumerate(play_text.value.splitlines(), 1):
        _line = _line.split("#")[0].strip()
        if not _line:
            continue
        _m = _re.fullmatch(r"([\w.]+)\[([\d,\s]+)\]\s*=\s*([-+.\deE]+)", _line)
        if not _m or _m.group(1) not in rounded:
            _errors.append(f"line {_ln}: can't parse `{_line}`")
            continue
        _idx = tuple(int(x) for x in _m.group(2).split(","))
        if len(_idx) != rounded[_m.group(1)].ndim or any(i >= s for i, s in zip(_idx, rounded[_m.group(1)].shape)):
            _errors.append(f"line {_ln}: index {_idx} out of range for {_m.group(1)} {rounded[_m.group(1)].shape}")
            continue
        _entries.append((_m.group(1), _idx, float(_m.group(3))))
    mo.stop(bool(_errors), mo.md("**Fix these first:**<br>" + "<br>".join(_errors)))
    mo.stop(not play_run.value, mo.md(f"_{len(_entries)} entries parsed. Press **build & evaluate**._"))
    _W = apply_entries(_entries)
    _r = evaluate(_W, int(play_eps.value), 0)
    _tr = trace_episode(_W)
    mo.md(f"**{len(_entries)} entries** → solved **{_r['success_count']}/{_r['episodes']}** · red {100*_r['red']:.0f}% · "
          f"timeout {100*_r['timeout']:.0f}% · mean length {_r['mean_len']:.1f} · argmax episode: "
          f"{'solved' if _tr['solved'] else 'not solved'} in {_tr['steps']} steps")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2 · Experiment 8: Qwen3.5-0.8B math edit, and how much of it is needed

    Astra's p50c edit makes Qwen3.5-0.8B stop looping on math (MATH-500 7.9% → 48.1% in the published
    evaluation). It touches `layers.{15,19}.mlp.down_proj.weight` (1024 × 3584, bf16) and changes
    **130,188** entries. The minimal-edit campaign built 27 subsets of it and screened each on the same
    256 MATH-train problems (vLLM, thinking on, temp 0.6, 32k tokens, n = 1).
    """)
    return


@app.cell
def _(EX, json, np):
    def bf16_bits(x):
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


    def bf16_to_f32(b):
        return (b.astype(np.uint32) << 16).view(np.float32)


    LAYERS = (15, 19)
    base_bits = {L: bf16_bits(np.load(EX / f"base_l{L}_down_proj_f32.npy")) for L in LAYERS}
    W0 = {L: bf16_to_f32(base_bits[L]).astype(np.float64) for L in LAYERS}
    _z = np.load(EX / "edit.npz")
    edit_idx = {L: _z[f"t{i}_idx"].astype(np.int64) for i, L in enumerate(LAYERS)}
    edit_val = {L: _z[f"t{i}_val"] for i, L in enumerate(LAYERS)}
    selected = {L: np.array(json.load(open(EX / "frozen_weights_manifest.json"))[str(L)]["selected_neurons"]) for L in LAYERS}


    def build_bits(spec):
        """Port of code/build_candidate.py: keep the subset of Astra's stored bf16 entries the spec selects
        (cols/rows/entries per layer; missing key = no filter), optionally scaling the delta."""
        out = {}
        for L in LAYERS:
            idx, val = edit_idx[L], edit_val[L]
            rows, cols = idx // 3584, idx % 3584
            keep = np.ones(len(idx), bool)
            if f"cols{L}" in spec:
                keep &= np.isin(cols, spec[f"cols{L}"])
            if f"rows{L}" in spec:
                keep &= np.isin(rows, spec[f"rows{L}"])
            if f"entries{L}" in spec:
                e = np.asarray(spec[f"entries{L}"], dtype=np.int64).reshape(-1, 2)
                keep &= np.isin(idx, e[:, 0] * 3584 + e[:, 1])
            flat = base_bits[L].reshape(-1).copy()
            s = spec.get("scale", 1.0)
            if s == 1.0:
                flat[idx[keep]] = val[keep]
            else:
                orig = bf16_to_f32(flat[idx[keep]]).astype(np.float64)
                flat[idx[keep]] = bf16_bits(orig + (bf16_to_f32(val[keep]) - orig) * s)
            out[L] = flat.reshape(base_bits[L].shape)
        return out


    def n_changed(bits):
        return int(sum((bits[L] != base_bits[L]).sum() for L in LAYERS))


    astra_bits = build_bits({})
    return (
        LAYERS,
        W0,
        astra_bits,
        base_bits,
        bf16_bits,
        bf16_to_f32,
        build_bits,
        n_changed,
        selected,
    )


@app.cell
def _(
    EX,
    LAYERS,
    W0,
    astra_bits,
    base_bits,
    bf16_bits,
    bf16_to_f32,
    json,
    mo,
    np,
    selected,
):
    rank1 = {}
    _metrics = {}
    for _L in LAYERS:
        _D = bf16_to_f32(astra_bits[_L]).astype(np.float64) - W0[_L]
        _U, _s, _Vt = np.linalg.svd(_D, full_matrices=False)
        _sign = np.sign(_U[:, 0] @ np.load(EX / f"factors_layer{_L}.npz")["u"])
        _u, _v = _U[:, 0] * _sign, _Vt[0] * _sign  # Δ = σ·u·vᵀ with u aligned to the campaign's factor
        _mask = np.zeros(3584)
        _mask[selected[_L]] = 1
        _Wf = W0[_L] - 6.0 * np.outer(_u, _mask * (_u @ W0[_L]))
        _stored = bf16_to_f32(astra_bits[_L])[:, selected[_L]].astype(np.float64)
        _diff = np.abs(bf16_to_f32(bf16_bits(_Wf.astype(np.float32)))[:, selected[_L]] - _stored)
        _step = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(_stored), 1e-30))) - 7)  # one bf16 rounding step at each value
        rank1[_L] = {"u": _u, "v": _v, "c": _u @ W0[_L]}
        _metrics[_L] = {
            "bf16 entries changed": f"{int((astra_bits[_L] != base_bits[_L]).sum()):,}",
            "columns (MLP neurons) touched": f"{int((_D != 0).any(0).sum())} of 3,584",
            "σ₂ / σ₁ of Δ (rank-1 ⇔ ≈ 0)": f"{_s[1] / _s[0]:.1e}",
            "‖Δ‖ / ‖W₀‖": f"{np.linalg.norm(_D) / np.linalg.norm(W0[_L]):.3f}",
            "closed form within one bf16 step (of 65,536 touched)": f"{int((_diff <= _step).sum()):,}",
            "largest closed-form difference": f"{_diff.max():.1e}",
            "median |Δ| on touched entries": f"{np.median(np.abs(_D[:, selected[_L]])):.1e}",
            "Astra's own audit, exact u": f"rel. error {json.load(open(EX / 'frozen_edit_audit.json'))[str(_L)]['formula_relative_error']:.1e}",
        }
    _rows = [{"": k, "layer 15": _metrics[15][k], "layer 19": _metrics[19][k]} for k in _metrics[15]]
    _cos_u = abs(rank1[15]["u"] @ rank1[19]["u"])
    mo.vstack([
        mo.md(r"""### 2.1 Apply the edit and recover what it is

    Astra's stored bf16 entries are applied to the base tensors. An SVD of the change Δ shows it is rank-1, and Astra's
    closed form

    $$W = W_0 - 6\, u \left(m_{64} \odot u^\top W_0\right)$$

    (u: unit repetition direction; m₆₄: mask of the 64 edited neurons) is checked against the stored values:"""),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"Both layers subtract almost the same direction, |cos(u₁₅, u₁₉)| = **{_cos_u:.3f}**. "
              "Astra's exact u is not in the logs, so u here is estimated from Δ after bf16 rounding. That estimate reproduces "
              "98–99% of the touched entries to within one rounding step; the rest are small weights (|w| < 0.002) where the "
              "estimation error shows. Astra's run checked the formula against its exact u (last row, `frozen_edit_audit.json`)."),
    ])
    return (rank1,)


@app.cell
def _(LAYERS, PALETTE, mo, np, plt, rank1, selected):
    _fig, _axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    _txt = []
    for _ax, _L in zip(_axes, LAYERS):
        _c = np.abs(rank1[_L]["c"])
        _order = np.argsort(-_c)
        _rank = np.empty(3584, int)
        _rank[_order] = np.arange(3584)
        _sel = selected[_L]
        _ax.scatter(np.arange(1, 3585), _c[_order], s=2, color=PALETTE["blue"], alpha=0.3, label="all 3,584 neurons")
        _ax.scatter(_rank[_sel] + 1, _c[_sel], s=12, color=PALETTE["orange"], label="the 64 Astra edited", zorder=3)
        _ax.axvline(64.5, color="#999", linewidth=0.8, linestyle=":")
        _ax.set_xscale("log")
        _ax.set_xlabel("neuron rank by |u·W₀|")
        _ax.set_ylabel("|u·W₀| (writing strength)")
        _ax.set_title(f"layer {_L}")
        _txt.append(f"layer {_L}: {int((_rank[_sel] < 64).sum())} of the 64 are among the 64 strongest writers; the weakest ranks {int(_rank[_sel].max()) + 1:,}")
    _axes[0].legend(loc="lower left", frameon=False)
    _fig.tight_layout()
    mo.vstack([
        mo.md("**Which neurons were changed, and why those.** Each MLP neuron j writes the repetition direction with strength |u·W₀[:, j]|. "
              "Astra did not just take the 64 strongest writers: it ranked neurons by writing strength × how much more the neuron fires "
              "on looping text than on productive text (its notes, `data/docs/exp8/notes_astra_method.md`). "
              + "; ".join(_txt) + ". The activation half of that score needs Astra's forward passes, which are not in this package."),
        _fig,
    ])
    return


@app.cell
def _(EX, LAYERS, json, np, selected):
    _fz = {L: np.load(EX / f"factors_layer{L}.npz") for L in LAYERS}
    # the campaign's emit_specs.py ordered by the factors it saved; use those so specs regenerate exactly
    _fu = {L: _fz[L]["u"].astype(np.float64) for L in LAYERS}
    _fv = {L: _fz[L]["v"].astype(np.float64) for L in LAYERS}
    order_v = {L: [int(selected[L][i]) for i in np.argsort(-np.abs(_fv[L][selected[L]]))] for L in LAYERS}
    order_u = {L: [int(i) for i in np.argsort(-np.abs(_fu[L]))] for L in LAYERS}


    def top_entries(L, B):
        d = np.abs(np.outer(_fu[L], _fv[L])).ravel()
        i = np.argpartition(d, -B)[-B:]
        i = i[np.argsort(-d[i])]
        return [[int(k // 3584), int(k % 3584)] for k in i]


    def make_spec(family, n):
        """The campaign's candidate families (emit_specs.py)."""
        if family == "A · top-k columns, both layers":
            return {"scale": 1.0, "cols15": order_v[15][:n], "cols19": order_v[19][:n]}
        if family == "D · top-B entries per layer":
            return {"scale": 1.0, "entries15": top_entries(15, n), "entries19": top_entries(19, n)}
        if family == "B · layer 15 only, top-k columns":
            return {"scale": 1.0, "cols15": order_v[15][:n], "cols19": []}
        if family == "B · layer 19 only, top-k columns":
            return {"scale": 1.0, "cols19": order_v[19][:n], "cols15": []}
        if family == "ER · layer 15 only, top-R rows":
            return {"scale": 1.0, "rows15": order_u[15][:n], "cols19": []}
        raise ValueError(family)


    FAMILY_OF = {"A": ("A · top-k columns, both layers", lambda c: int(c.split("_topk")[1])),
                 "D": ("D · top-B entries per layer", lambda c: int(c[1:].split("k_")[0]) * 1024),
                 "ER": ("ER · layer 15 only, top-R rows", lambda c: int(c[2:].split("_")[0]))}
    CAMPAIGN_SPECS = {p.stem: json.load(open(p)) for p in sorted((EX / "specs").glob("*.json"))}
    CAMPAIGN_SPECS["B15_full_s125"] = {"scale": 1.25, "cols19": []}
    CAMPAIGN_SPECS["G000_scale"] = {"scale": 0.0}
    return CAMPAIGN_SPECS, FAMILY_OF, make_spec


@app.cell
def _(
    CAMPAIGN_SPECS,
    EX,
    FAMILY_OF,
    build_bits,
    json,
    make_spec,
    mo,
    n_changed,
):
    _recorded_counts = json.load(open(EX / "candidate_entries.json"))
    _rows = []
    entry_counts = {}
    for _cid, _spec in mo.status.progress_bar(list(CAMPAIGN_SPECS.items()), title="building candidates"):
        entry_counts[_cid] = n_changed(build_bits(_spec))
        _fam = next((f for k, f in FAMILY_OF.items() if _cid.startswith(k)), None)
        _regen = "—" if _fam is None else ("✅" if make_spec(_fam[0], _fam[1](_cid)) == _spec else "❌")
        _rec = _recorded_counts.get(_cid)
        _rows.append({"candidate": _cid, "entries (built now)": entry_counts[_cid],
                      "campaign's count": "—" if _rec is None else f"{_rec:,}",
                      "same": "—" if _rec is None else ("✅" if _rec == entry_counts[_cid] else "❌"),
                      "spec regenerated": _regen})
    _rows.sort(key=lambda r: r["entries (built now)"])
    mo.vstack([mo.md("### 2.2 Rebuild all campaign candidates\n\nEach spec is applied with `build_bits` (a port of the campaign's `build_candidate.py`); "
                     "the A/D/ER families are also regenerated from scratch with `make_spec` and compared with the specs the campaign ran."),
               mo.ui.table(_rows, selection=None, pagination=False, page_size=30)])
    return (entry_counts,)


@app.cell
def _(mo):
    fam = mo.ui.dropdown(["A · top-k columns, both layers", "D · top-B entries per layer", "B · layer 15 only, top-k columns",
                          "B · layer 19 only, top-k columns", "ER · layer 15 only, top-R rows"],
                         value="D · top-B entries per layer", label="family")
    fam_n = mo.ui.number(1, 65536, value=4096, label="k / B / R")
    mo.vstack([mo.md("### 2.3 Design a candidate\n\nAny size, not just the ones the campaign screened. The spec it prints is what `code/build_candidate.py` and §2.6–2.7 take."),
               mo.hstack([fam, fam_n], justify="start")])
    return fam, fam_n


@app.cell
def _(
    LAYERS,
    W0,
    astra_bits,
    bf16_to_f32,
    build_bits,
    fam,
    fam_n,
    json,
    make_spec,
    mo,
    n_changed,
    np,
):
    my_spec = make_spec(fam.value, int(fam_n.value))
    my_bits = build_bits(my_spec)
    _full = sum(np.abs(bf16_to_f32(astra_bits[L]) - W0[L]).sum() for L in LAYERS)
    _mine = sum(np.abs(bf16_to_f32(my_bits[L]) - W0[L]).sum() for L in LAYERS)
    _txt = json.dumps(my_spec)
    mo.md(f"**{n_changed(my_bits):,}** changed bf16 entries ({100 * n_changed(my_bits) / n_changed(astra_bits):.1f}% of Astra's edit), "
          f"carrying **{100 * _mine / _full:.1f}%** of its |Δ| mass.<br>"
          f"<details><summary>spec ({len(_txt):,} chars)</summary>\n\n```json\n{_txt[:3000]}{' …' if len(_txt) > 3000 else ''}\n```\n</details>")
    return (my_spec,)


@app.cell
def _(ROOT, json, math, np):
    screens = {}
    for _p in sorted((ROOT / "data" / "screens").glob("screen_*.jsonl")):
        screens[_p.stem[len("screen_"):]] = {r["id"]: r for r in map(json.loads, open(_p))}
    screens = {k: v for k, v in screens.items() if k not in ("ONE_L15E32", "ENERGY_16")}  # invalidated recipes


    def acc(cid):
        return np.mean([r["correct"][0] for r in screens[cid].values()])


    def paired(a, b, n_boot=4000, seed=0):
        """a vs b on the same problems: diff, discordant counts, exact McNemar p, bootstrap 95% CI."""
        ids = sorted(set(screens[a]) & set(screens[b]))
        x = np.array([screens[a][i]["correct"][0] for i in ids])
        y = np.array([screens[b][i]["correct"][0] for i in ids])
        n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        k, n = min(n10, n01), n10 + n01
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
        d = x - y
        boot = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(1)
        return {"diff": d.mean(), "a_only": n10, "b_only": n01, "p": p, "ci": tuple(np.percentile(boot, [2.5, 97.5])), "n": len(ids)}

    return acc, paired, screens


@app.cell
def _(acc, entry_counts, mo, np, paired, screens):
    screen_rows = []
    for _cid in sorted(screens, key=lambda c: entry_counts.get(c, 0)):
        _r = list(screens[_cid].values())
        _vA, _vB = paired(_cid, "A64_topk64"), paired(_cid, "G000_scale")
        screen_rows.append({
            "candidate": _cid, "entries": entry_counts.get(_cid, "—"), "acc": f"{100 * acc(_cid):.1f}%",
            "cut at 32k": f"{100 * np.mean([x['finish'][0] == 'length' for x in _r]):.0f}%",
            "Δ vs full edit": f"{100 * _vA['diff']:+.1f}pp",
            "95% CI": f"{100 * _vA['ci'][0]:+.1f} to {100 * _vA['ci'][1]:+.1f}",
            "p": f"{_vA['p']:.2g}",
            "Δ vs base": f"{100 * _vB['diff']:+.1f}pp",
        })
    mo.vstack([mo.md("### 2.4 Screening results, recomputed from the raw per-problem outputs\n\n"
                     "`data/screens/*.jsonl` hold one row per problem per candidate (correct, finish reason, tokens). "
                     "Each candidate is paired with the full edit (`A64_topk64`) on the same 256 problems: difference in accuracy, "
                     "95% bootstrap CI, exact McNemar p. `G000_scale` is the unedited base."),
               mo.ui.table(screen_rows, selection=None, pagination=False, page_size=30)])
    return


@app.cell
def _(GREY, PALETTE, acc, entry_counts, mo, plt, screens):
    _fams = {"A": ("o", PALETTE["blue"], "A · top-k columns"), "D": ("s", PALETTE["orange"], "D · top-B entries"),
             "ER": ("^", PALETTE["green"], "ER · L15 rows"), "B": ("D", PALETTE["purple"], "B · one layer")}
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _k, (_m, _c, _lab) in _fams.items():
        _pts = sorted((entry_counts[c], 100 * acc(c)) for c in screens if c.startswith(_k) and c in entry_counts)
        _pts = [p for p in _pts if p[0] > 0]
        if _k == "A":
            _ax.plot(*zip(*_pts), "-", color=_c, linewidth=0.8, alpha=0.6)
        _ax.scatter(*zip(*_pts), marker=_m, color=_c, s=26, label=_lab, zorder=3)
    _ax.axhline(100 * acc("A64_topk64"), color=PALETTE["blue"], linestyle=":", linewidth=0.9)
    _ax.axhline(100 * acc("G000_scale"), color=GREY, linestyle=":", linewidth=0.9)
    _ax.text(1.0, 100 * acc("G000_scale") + 0.8, "unedited base", color=GREY, ha="right", transform=_ax.get_yaxis_transform())
    _ax.text(1.0, 100 * acc("A64_topk64") + 0.8, "Astra's full edit", color=PALETTE["blue"], ha="right", transform=_ax.get_yaxis_transform())
    _ax.set_xscale("log")
    _ax.set_xlabel("changed bf16 entries (built in §2.2)")
    _ax.set_ylabel("MATH-train screening accuracy (%)")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Choosing **entries** by |Δ| (D) beats choosing whole **columns** (A): D4k — 8,192 entries, 6% of the edit — ties the full edit, "
                           "while A needs ~48 columns/layer (~98k entries) to get within noise. Single-sample screening on 256 problems: differences under ~5pp are noise.")])
    return


@app.cell
def _(mo, screens):
    pa = mo.ui.dropdown(sorted(screens), value="D4k_entries", label="candidate")
    pb = mo.ui.dropdown(sorted(screens), value="A64_topk64", label="against")
    mo.hstack([pa, pb], justify="start")
    return pa, pb


@app.cell
def _(mo, pa, paired, pb):
    _r = paired(pa.value, pb.value)
    mo.md(f"**{pa.value} − {pb.value}** on {_r['n']} paired problems: **{100*_r['diff']:+.1f}pp** "
          f"(95% CI {100*_r['ci'][0]:+.1f} to {100*_r['ci'][1]:+.1f}); solved only by {pa.value}: {_r['a_only']}, only by {pb.value}: {_r['b_only']}; "
          f"exact McNemar p = {_r['p']:.3g}.")
    return


@app.cell
def _(EX, json, mo, np):
    _rows = [json.loads(l) for l in open(EX / "validate_D4k_math500.jsonl")]
    _c = np.array([r["correct"] for r in _rows], float)
    mo.md(f"### 2.5 Held-out check (partial)\n\nD4k on MATH-500 × 4 samples (full 32k protocol, GPU): **{int(_c.sum()):,}/{_c.size:,} = {100*_c.mean():.1f}%**, "
          f"vs 48.1% for the full edit in the published evaluation. The A64 arm of this validation and the GSM8K/IFEval arms did not finish, "
          "so this is not a paired comparison.")
    return


@app.cell
def _(mo):
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    load_model_btn = mo.ui.run_button(label="▶ load Qwen3.5-0.8B (downloads ~1.7 GB once)")
    mo.vstack([
        mo.md(f"""### 2.6 Run the model on this machine

    Paired generations, base vs a candidate, on the screening problems, with the campaign's chat template, thinking on,
    and sampling (temp 0.6, top-p 0.95, top-k 20). Weights are loaded once in fp32 (exact for bf16 values) and the
    two `down_proj` tensors are swapped per arm. Device found on this machine: **{DEVICE}**{' — CPU runs at ~0.5 tok/s, too slow; use a Mac with Apple GPU or CUDA' if DEVICE == 'cpu' else ''}.

    **What to expect.** On an M-series Mac, 8 problems × 1,024 tokens × 2 arms takes ~1.5 min, and 4 × 4,096 takes ~4.5 min.
    The edit's effect does **not** show at these budgets: both arms are still thinking when they are cut off. It shows
    at the campaign's 32k cap, where the base runs out of tokens on 91% of problems and the full edit on 30%.
    The table therefore also shows each problem's recorded 32k-token screening result. Use this section to read and
    compare what the edited model writes. For the full-budget replication, use a GPU server (§2.7)."""),
        load_model_btn,
    ])
    return DEVICE, load_model_btn, torch


@app.cell
def _(DEVICE, LAYERS, W0, load_model_btn, mo, np, torch):
    qwen = tok = qwen_path = None
    if load_model_btn.value:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as _hf_logging

        _hf_logging.set_verbosity_error()  # Qwen3.5 warns about optional CUDA kernels on every forward pass

        with mo.status.spinner("downloading / loading Qwen3.5-0.8B"):
            # public model; token=False so a stale token on the viewer's machine can't break the download
            qwen_path = snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17", token=False)
            tok = AutoTokenizer.from_pretrained(qwen_path, padding_side="left")
            qwen = AutoModelForCausalLM.from_pretrained(qwen_path, dtype=torch.float32).to(DEVICE).eval()
        for _L in LAYERS:
            assert np.array_equal(qwen.model.layers[_L].mlp.down_proj.weight.detach().cpu().numpy(), W0[_L].astype(np.float32)), "base tensors differ from data/exp8"
    mo.md(f"Loaded `{type(qwen).__name__}` on **{DEVICE}**; layers 15/19 `down_proj` match the bundled base tensors bit-for-bit."
          if qwen is not None else "_Model not loaded._")
    return qwen, qwen_path, tok


@app.cell
def _(CAMPAIGN_SPECS, EX, json, mo):
    gen_arm = mo.ui.dropdown(["my design (§2.3)"] + sorted(CAMPAIGN_SPECS), value="A64_topk64", label="candidate (vs unedited base)")
    gen_n = mo.ui.slider(2, 32, value=8, step=2, label="problems", show_value=True)
    gen_tokens = mo.ui.dropdown(["512", "1024", "2048", "4096"], value="1024", label="max new tokens")
    gen_run = mo.ui.run_button(label="▶ generate")
    # a run made with this cell on an Apple M-series GPU, shown until you make your own
    _ex = EX / "example_generations.json"
    EXAMPLE_GEN = json.load(open(_ex)) if _ex.exists() else None
    get_gen, set_gen = mo.state(None)
    mo.hstack([gen_arm, gen_n, gen_tokens, gen_run], justify="start")
    return EXAMPLE_GEN, gen_arm, gen_n, gen_run, gen_tokens, get_gen, set_gen


@app.cell
def _(
    CAMPAIGN_SPECS,
    EX,
    LAYERS,
    ROOT,
    bf16_to_f32,
    build_bits,
    gen_arm,
    gen_n,
    gen_run,
    gen_tokens,
    json,
    mo,
    my_spec,
    qwen,
    set_gen,
    time,
    tok,
    torch,
):
    mo.stop(not gen_run.value, mo.md("_Press **generate** to run it on this machine._"))
    mo.stop(qwen is None, mo.md("_Load the model first (button above)._"))
    from math_verify import parse, verify

    _INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
    _probs = [json.loads(l) for l in open(EX / "screening_256.jsonl")][: gen_n.value]
    _prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + _INSTR}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True) for p in _probs]
    _T = int(gen_tokens.value)
    _spec = my_spec if gen_arm.value.startswith("my design") else CAMPAIGN_SPECS[gen_arm.value]
    _pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


    def _loop_share(ids, w=16):
        seen, rep = set(), 0
        for i in range(len(ids) - w + 1):
            g = tuple(ids[i:i + w])
            rep += g in seen
            seen.add(g)
        return rep / max(len(ids) - w + 1, 1)


    def _run_arm(bits):
        with torch.no_grad():
            for L in LAYERS:
                qwen.model.layers[L].mlp.down_proj.weight.copy_(torch.from_numpy(bf16_to_f32(bits[L])))
            torch.manual_seed(0)
            enc = tok(_prompts, return_tensors="pt", padding=True).to(qwen.device)
            out = qwen.generate(**enc, max_new_tokens=_T, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=_pad)
        res = []
        for p, row in zip(_probs, out[:, enc["input_ids"].shape[1]:].cpu().tolist()):
            ids = [t for t in row if t != _pad]
            text = tok.decode(ids, skip_special_tokens=True)
            try:
                ok = bool(verify(parse(f"${p['answer']}$"), parse(text)))
            except Exception:  # math-verify raises on unparsable answers; that is a wrong answer
                ok = False
            res.append({"correct": ok, "cut": len(ids) >= _T, "tokens": len(ids), "loop": _loop_share(ids), "text": text})
        return res


    _t0 = time.time()
    _arms = {}
    for _name, _bits in mo.status.progress_bar([("base", build_bits({"scale": 0.0})), (gen_arm.value, build_bits(_spec))], title="generating"):
        _arms[_name] = _run_arm(_bits)
    _run = {"candidate": gen_arm.value, "max_new_tokens": _T, "problems": [p["id"] for p in _probs],
            "answers": [p["answer"] for p in _probs], "device": str(qwen.device), "wall_s": round(time.time() - _t0),
            "arms": _arms}
    _out = ROOT / "runs" / f"generations_{gen_arm.value.split(' ')[0]}_{len(_probs)}x{_T}.json"
    _out.parent.mkdir(exist_ok=True)
    _out.write_text(json.dumps(_run))
    set_gen(_run)
    mo.md(f"_Done in {_run['wall_s']} s; saved to `runs/{_out.name}`._")
    return


@app.cell
def _(EXAMPLE_GEN, get_gen, mo, np, screens):
    gen_view = get_gen() or EXAMPLE_GEN
    mo.stop(gen_view is None, mo.md("_No generations yet. Load the model and press **generate**._"))
    _T = gen_view["max_new_tokens"]
    _cand = gen_view["candidate"]
    _b, _c = gen_view["arms"]["base"], gen_view["arms"][_cand]


    def _rec(cid, pid):
        r = screens.get(cid, {}).get(pid)
        return "—" if r is None else f"{'✓' if r['correct'][0] else '✗'} {r['n_tokens'][0]:,} tok{' (cut)' if r['finish'][0] == 'length' else ''}"


    def _summ(r):
        return (f"{sum(x['correct'] for x in r)}/{len(r)} correct · {sum(x['cut'] for x in r)}/{len(r)} hit the {_T:,}-token cap · "
                f"mean repeated-16-gram share {100 * np.mean([x['loop'] for x in r]):.0f}%")


    _rows = [{"problem": pid.removeprefix("math_train_"),
              "base": ("✓" if b["correct"] else "✗") + (" cut" if b["cut"] else "") + f" · loop {100 * b['loop']:.0f}%",
              _cand: ("✓" if c["correct"] else "✗") + (" cut" if c["cut"] else "") + f" · loop {100 * c['loop']:.0f}%",
              "base @ 32k": _rec("G000_scale", pid), f"{_cand} @ 32k": _rec(_cand, pid)}
             for pid, ans, b, c in zip(gen_view["problems"], gen_view["answers"], _b, _c)]
    mo.vstack([
        mo.md(("_Your run on this machine._" if get_gen() else
               f"_Bundled example run (Apple-silicon Mac GPU, {gen_view['wall_s']} s). Press **generate** to make your own._")
              + f"<br>**base:** {_summ(_b)}<br>**{_cand}:** {_summ(_c)}<br>"
              f"At a {_T:,}-token cap most thinking traces are cut in both arms. The \"@ 32k\" columns are the campaign's recorded screening of the same problems at its 32k cap."),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])
    return (gen_view,)


@app.cell
def _(gen_view, mo):
    _names = list(gen_view["arms"])
    peek_arm = mo.ui.dropdown(_names, value=_names[-1], label="arm")
    peek_i = mo.ui.dropdown({f"{i}: {p}": i for i, p in enumerate(gen_view["problems"])}, value=f"0: {gen_view['problems'][0]}", label="problem")
    mo.vstack([mo.md("**Read a generation** (last 3,000 characters)"), mo.hstack([peek_arm, peek_i], justify="start")])
    return peek_arm, peek_i


@app.cell
def _(gen_view, mo, peek_arm, peek_i):
    import html
    _r = gen_view["arms"][peek_arm.value][peek_i.value]
    mo.vstack([
        mo.md(f"{_r['tokens']:,} tokens · repeated-16-gram share {100 * _r['loop']:.0f}% · {'correct' if _r['correct'] else 'not correct'}"
              f"{' · cut at the token cap' if _r['cut'] else ''}"),
        mo.Html("<pre style='white-space: pre-wrap; font-size: 12px; max-height: 420px; overflow: auto'>…"
                + html.escape(_r["text"][-3000:]) + "</pre>"),
    ])
    return


@app.cell
def _(CAMPAIGN_SPECS, mo):
    ck_arm = mo.ui.dropdown(["my design (§2.3)"] + sorted(CAMPAIGN_SPECS), value="D4k_entries", label="candidate")
    ck_btn = mo.ui.run_button(label="▶ write checkpoint")
    mo.vstack([mo.md("### 2.7 Full-protocol replication on a GPU\n\nWrites base Qwen3.5-0.8B with the candidate's two tensors swapped in "
                     "(`checkpoints/<name>/`, ~1.7 GB, needs the model from §2.6), then gives the exact screening command the campaign ran."),
               mo.hstack([ck_arm, ck_btn], justify="start")])
    return ck_arm, ck_btn


@app.cell
def _(
    CAMPAIGN_SPECS,
    LAYERS,
    ROOT,
    build_bits,
    ck_arm,
    ck_btn,
    json,
    mo,
    my_spec,
    qwen_path,
):
    mo.stop(not ck_btn.value, mo.md("_Press **write checkpoint**._"))
    mo.stop(qwen_path is None, mo.md("_Load the model in §2.6 first; the checkpoint is built from its files._"))
    import os

    import st_numpy

    _name = "my_design" if ck_arm.value.startswith("my design") else ck_arm.value
    _spec = my_spec if _name == "my_design" else CAMPAIGN_SPECS[_name]
    _out = ROOT / "checkpoints" / _name
    _out.mkdir(parents=True, exist_ok=True)
    (_out / "candidate_spec.json").write_text(json.dumps(_spec))
    _st = next(f for f in os.listdir(qwen_path) if f.endswith(".safetensors"))
    for _f in os.listdir(qwen_path):
        if _f != _st and not (_out / _f).exists():
            os.symlink(os.path.join(qwen_path, _f), _out / _f)
    with mo.status.spinner("writing safetensors"):
        _tensors = st_numpy.load(os.path.join(qwen_path, _st))
        _bits = build_bits(_spec)
        for _L in LAYERS:
            _tensors[f"model.language_model.layers.{_L}.mlp.down_proj.weight"] = ("BF16", _bits[_L])
        st_numpy.save(str(_out / _st), _tensors, {"format": "pt"})
    mo.md(f"Wrote `checkpoints/{_name}/`. Copy this folder to a CUDA machine with `vllm` and `math-verify`, and from it run:\n\n```sh\n"
          f"python code/eval_vllm.py --model checkpoints/{_name} --problems data/exp8/screening_256.jsonl \\\n"
          f"    --out data/screens/screen_{_name}.jsonl --n 1 --max-tokens 32768 --gpu-mem 0.85\n```\n"
          "The output lands next to the other screens, so §2.4 includes it in the paired analysis on the next run. "
          "The checkpoint's other files are symlinks into the Hugging Face cache; copy with `cp -L` to resolve them.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3 · What both campaigns found

    | | Car-ramp actor | Exp 8, Qwen3.5-0.8B |
    |:--|:--|:--|
    | Astra's edit | 6,073 entries | 130,188 bf16 entries |
    | What it is | ~55 functional writes + 6,018 erasures isolating repurposed units | one direction subtracted through 64 chosen MLP neurons in each of 2 layers |
    | Smallest subset of Astra's edit that keeps the behavior | **12 entries** (§1.4–1.5), 11 if one redundant entry goes | **8,192 entries** (D4k, §2.4) — ties the full edit in screening |
    | Written from scratch | **10 entries** (§1.1, §1.8) | — |
    | Evidence | exact: Kinetix, 1,024 episodes × several seeds, re-runnable here | screening-grade: 256 MATH-train problems, n = 1; one partial held-out check |

    **Caveats.** Exp 8 screening is single-sample on train problems (SE ≈ 3pp). §2.6's laptop runs use a short token
    budget, at which the edit's effect does not show; the effect is measured at 32k tokens (§2.4). Car-ramp counts
    are against the rounded base. The waypoint's entry set
    is read from its checkpoint (its keep-list was not saved). Original write-ups: `data/docs/`.
    """)
    return


if __name__ == "__main__":
    app.run()
