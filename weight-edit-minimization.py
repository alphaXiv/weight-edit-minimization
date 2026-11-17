# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "marimo",
#     "numpy==2.4.6",
#     "matplotlib",
#     "jax==0.9.0",
#     "kinetix-env==3.0.2",
#     "torch",
#     "transformers==5.23.0",
#     "huggingface_hub",
#     "math-verify",
#     "certifi",
#     "altair",
#     "pandas",
#     "imageio",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Weight-edit minimization — replication notebook",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weight-edit minimization — run the experiments yourself

    Two campaigns asked the same question: **what is the smallest set of weights whose edit produces a behavior?**
    Both start from an edit written by Astra (`RikiyaT/lm_understand-logs`) and strip it down.

    This notebook is the experiment, not a report about it. Run the cells top to bottom (▷ **Run all**, bottom
    right). Each experiment is a short code cell, and the charts under it fill in as runs finish, next to the
    numbers the original orx runs recorded. Change a variable and rerun to try your own version. Collapsed cells hold
    plumbing (setup, the Kinetix harness, chart helpers); click one to read it.

    **Time.** Section 1 runs about 20 evaluations of 256 episodes each on the CPU: ~5 minutes on a laptop. Section 2's
    analysis is instant; running Qwen yourself (§2.5) is off until you set `RUN_QWEN = True`.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import math
    import os as _os
    import sys
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    alt.data_transformers.disable_max_rows()  # the neuron chart plots all 7,168 neurons

    REPO = "alphaXiv/weight-edit-minimization"


    def _find_root():
        here = mo.notebook_dir() or Path.cwd()
        if (here / "data").is_dir():
            return here
        # molab can run the notebook file without the rest of the repo; fetch the repo once
        import io
        import ssl
        import tarfile
        import tempfile
        import urllib.request

        import certifi

        dest = Path(tempfile.gettempdir()) / "weight-edit-minimization"
        if not (dest / ".complete").exists() and (dest / "weight-edit-minimization.py").exists() and (dest / "data").is_dir():
            # a cache from before the marker existed; the notebook file is among the last items extracted
            (dest / ".complete").touch()
        if not (dest / ".complete").exists():  # written only once a download is fully extracted
            # python.org builds on macOS ship without CA certificates; use certifi's bundle
            _ctx = ssl.create_default_context(cafile=certifi.where())
            with mo.status.spinner(f"downloading the notebook's data from github.com/{REPO}"):
                with urllib.request.urlopen(f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main", timeout=300, context=_ctx) as _r:
                    _buf = io.BytesIO(_r.read())
            import shutil

            # extract into a fresh folder and switch to it only when complete; an incomplete earlier cache is set aside
            # whole (never merged or deleted), so nothing you saved in it is lost
            _staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix="weight-edit-minimization-new-"))
            try:
                with tarfile.open(fileobj=_buf) as _t:
                    _members = []
                    for _m in _t.getmembers():
                        _m.name = _m.name.partition("/")[2]  # drop the "<repo>-main/" prefix
                        if _m.name:
                            _members.append(_m)
                    _t.extractall(_staging, members=_members, filter="data")
                (_staging / ".complete").touch()
            except BaseException:
                shutil.rmtree(_staging, ignore_errors=True)  # only our own partial extraction
                raise
            if dest.exists():
                dest.rename(dest.with_name(f"weight-edit-minimization-incomplete-{int(time.time())}"))
            _staging.rename(dest)
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
    PALETTE = house.PALETTE
    return CR, EX, PALETTE, ROOT, alt, json, math, mo, np, pd, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1 · Car-ramp: a PPO actor that never solves its level, fixed by 10–12 weights

    Kinetix `l/h2_car_ramp`: get the car's green wheel over the ramp and into the blue goal without the green wheel
    touching the red strip. A PPO actor trained for 1.001 B steps never does it. Astra's hand-written edit changes
    6,073 of its weights and solves it every time. How many of those 6,073 matter?

    ### 1.1 Settings and the base network

    Edits are made on the base weights rounded to 3 significant digits per tensor (Astra worked from that view), so
    "entries changed" counts differences from the rounded base. Every evaluation below uses these settings. The orx runs used 256 episodes for the audits and 1,024 for the
    headline numbers; sampled episodes use the same key schedule as `eval_actor.py`, so with the same episode count
    and seed you get exactly the recorded counts.
    """)
    return


@app.cell
def _():
    EPISODES = 256  # episodes per evaluation; 1,024 reproduces the headline runs (about 4x slower)
    SEED = 0
    return EPISODES, SEED


@app.cell(hide_code=True)
def _(CR, mo, np):
    NAME_MAP = {"embed.W": "params/Dense_0/kernel", "embed.b": "params/Dense_0/bias",
                "actor_head.W": "params/GeneralActorCriticRNN_0/Dense_0/kernel",
                "actor_head.b": "params/GeneralActorCriticRNN_0/Dense_0/bias"}
    for _i in range(5):
        NAME_MAP[f"actor{_i}.W"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{_i}/Dense_0/kernel"
        NAME_MAP[f"actor{_i}.b"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{_i}/Dense_0/bias"

    _ckpt = np.load(CR / "base_ppo_final_seed0.npz")
    base_raw = {k: _ckpt[v].astype(np.float64) for k, v in NAME_MAP.items()}


    def round3(x):
        """Astra saw each tensor rounded to 3 significant digits; every edit is made on top of that."""
        scale = 10.0 ** (np.floor(np.log10(np.abs(x).max())) - 2)
        return np.round(x / scale) * scale


    rounded = {k: round3(v) for k, v in base_raw.items()}
    mo.md(f"{sum(v.size for v in base_raw.values()):,} actor weights in {len(base_raw)} tensors")
    return base_raw, rounded


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The actor is a ReLU embedding (679 observations → 128 units), five tanh layers of 128, and a head of 16 action
    logits (only the first 3 drive the motor). The cell below (collapsed) is the evaluation harness: the exact
    `tools/eval_actor.py` the orx runs used, on official Kinetix (`kinetix-env==3.0.2`, JAX on CPU). It gives you
    `evaluate(weights, episodes, seed)`, `trace_episode(weights)` for one recorded episode, and `animate(...)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, time):
    import functools
    import hashlib
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
            obs, state, rng = carry
            logits, _ = actor_forward(W, obs.astype(jnp.float32))
            action, rng = _act(logits, rng, argmax)
            obs, state, reward, new_done, info = jax.vmap(_env.step, in_axes=(0, 0, 0, None, None))(
                jax.random.split(rng, episodes), state, action, _env_params, None)
            return (obs, state, rng), (reward, new_done, info["GoalR"])

        _, (rewards, dones, goal) = jax.lax.scan(step, (obs, state, rng), None, length=256)
        first = jnp.argmax(dones.astype(jnp.int32), 0)
        ar = jnp.arange(episodes)
        success = goal[first, ar]
        red = (~success) & (rewards[first, ar] < -0.5)
        return {"success_count": success.sum(), "red": red.mean(), "timeout": ((~success) & (~red)).mean(),
                "mean_len": (first + 1).mean(), "mean_return": jnp.cumsum(rewards, 0)[first, ar].mean()}


    _cache = {}


    def evaluate(W, episodes=256, seed=0, argmax=False):
        """Run `episodes` episodes of the car-ramp level with actor weights W (tensor name -> array)."""
        key = (hashlib.sha1(b"".join(np.ascontiguousarray(W[k], np.float32).tobytes() for k in sorted(W))).hexdigest(), episodes, seed, argmax)
        if key not in _cache:  # the same network shows up in several experiments; run it once
            Wj = {k: jnp.asarray(v, jnp.float32) for k, v in W.items()}
            t0 = time.perf_counter()
            out = {k: float(v) for k, v in _rollout(Wj, jax.random.PRNGKey(seed), episodes, argmax).items()}
            out.update(episodes=episodes, seed=seed, argmax=argmax, success_count=int(out["success_count"]),
                       wall_s=round(time.perf_counter() - t0, 1))
            _cache[key] = out
            return out
        return _cache[key] | {"cached": True}


    @jax.jit
    def _trace(W, key):
        rng, reset_rng = jax.random.split(key)
        obs, state = _reset(reset_rng, 1)

        def step(carry, _):
            obs, state, rng = carry
            logits, h = actor_forward(W, obs.astype(jnp.float32))
            action, rng = _act(logits, rng, True)
            before = (obs[0, jnp.array([OBS_X, OBS_VX, OBS_GX])], jnp.stack(h)[:, 0], jax.nn.softmax(logits[0, :3]), jax.tree.map(lambda x: x[0], state))
            obs, state, _, done, info = jax.vmap(_env.step, in_axes=(0, 0, 0, None, None))(
                jax.random.split(rng, 1), state, action, _env_params, None)
            return (obs, state, rng), before + (done[0], info["GoalR"][0])

        (_, final_state, _), rec = jax.lax.scan(step, (obs, state, rng), None, length=256)
        return rec, jax.tree.map(lambda x: x[0], final_state)


    def trace_episode(W, seed=0):
        """One argmax episode: car state (x, vx, green-x), every hidden activation, p(motor action), env states, outcome."""
        (obs, h, probs, states, done, goal), final_state = _trace({k: jnp.asarray(v, jnp.float32) for k, v in W.items()}, jax.random.PRNGKey(seed))
        done, goal = np.asarray(done), np.asarray(goal)
        n = int(np.argmax(done)) + 1 if done.any() else 256  # steps taken, through the first done step
        # states[i] is the state after i steps; append the state after the last step
        states = jax.tree.map(lambda s, f: jnp.concatenate([s, f[None]]), states, final_state)
        return {"obs": np.asarray(obs)[:n], "h": np.asarray(h)[:n], "probs": np.asarray(probs)[:n], "steps": n, "states": states,
                "solved": bool(done.any() and goal[n - 1])}


    from kinetix.render.renderer_pixels import make_render_pixels  # after the env import: kinetix has a circular import

    _render = jax.jit(make_render_pixels(_env_params, _static.replace(downscale=1)))


    def _frames(tr, stride=2, top=250):
        idx = list(range(0, tr["steps"], stride)) + [tr["steps"]]
        frames = []
        for i in idx:
            f = np.asarray(_render(jax.tree.map(lambda x: x[i], tr["states"])))
            frames.append(np.flipud(np.transpose(f, (1, 0, 2)))[top:].astype(np.uint8))
        return frames, idx


    def animate(*episodes):
        """Animated GIF of traced (argmax) episodes side by side: animate((trace, "label"), (trace, "label"), ...)."""
        import base64
        import io

        import imageio.v2 as iio
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default(size=22)
        panels = []
        for tr, label in episodes:
            frames, idx = _frames(tr)
            out = []
            for f, i in zip(frames, idx):
                im = Image.fromarray(f)
                status = ("solved" if tr["solved"] else "not solved") if i == idx[-1] else ""
                ImageDraw.Draw(im).text((12, 10), f"{label}   step {i}   {status}", fill=(20, 20, 20), font=font)
                out.append(np.asarray(im))
            panels.append(out)
        gap = np.full((panels[0][0].shape[0], 10, 3), 255, np.uint8)
        frames = []
        for i in range(max(len(p) for p in panels)):
            row = []
            for p in panels:
                row += [p[min(i, len(p) - 1)], gap]
            frames.append(np.concatenate(row[:-1], axis=1))
        buf = io.BytesIO()
        iio.mimwrite(buf, frames, format="GIF", duration=60, loop=0)
        return mo.Html(f"<img alt='car-ramp episodes' style='width:{min(100, 50 * len(episodes))}%' src='data:image/gif;base64,{base64.b64encode(buf.getvalue()).decode()}'>")

    return animate, evaluate, trace_episode


@app.cell(hide_code=True)
def _(PALETTE, ROOT, alt, evaluate, json, mo, pd):
    import re

    # every RESULT line the orx runs printed, to set next to what you run here
    recorded = []
    for _f in sorted((ROOT / "data" / "logs").glob("*.log")):
        for _m in re.finditer(r"RESULT (\{.*\})", _f.read_text()):
            recorded.append(json.loads(_m.group(1)))


    def recorded_run(weights_file=None, episodes=None, seed=0, tag=None):
        """The recorded orx RESULT for a network file (or a tag), preferring the same episode count."""
        hits = [r for r in recorded if not r["argmax"] and r["seed"] == seed
                and (tag is None or r["tag"] == tag) and (weights_file is None or r["weights"].endswith(weights_file))]
        same = [r for r in hits if r["episodes"] == episodes]
        return (same or hits or [None])[0]


    def report(name, r, rec=None):
        """One evaluation as a line of text, with the recorded orx run when there is one."""
        txt = (f"**{name}**: solved **{r['success_count']}/{r['episodes']}** · hit red {100 * r['red']:.0f}% · "
               f"timed out {100 * r['timeout']:.0f}% · mean episode {r['mean_len']:.0f} steps · "
               + ("same network already run above, result reused" if r.get("cached") else f"{r['wall_s']} s"))
        if rec is not None:
            same = rec["episodes"] == r["episodes"] and rec["success_count"] == r["success_count"] and round(rec["mean_len"], 3) == round(r["mean_len"], 3)
            txt += (f"<br>recorded orx run ({rec['episodes']} episodes, seed {rec['seed']}): {rec['success_count']}/{rec['episodes']}, "
                    f"mean {rec['mean_len']:.0f} steps" + (" — **same success count and mean length**" if same else ""))
        return mo.md(txt)


    def progress_chart(order, done, recorded_runs, title, current=None):
        """Bars for runs finished so far (blue) next to the recorded orx result (orange tick); unfinished rows say so."""
        rows = []
        for name in order:
            r, rec = done.get(name), recorded_runs.get(name)
            rows.append({"run": name, "solved (%)": None if r is None else round(100 * r["success_count"] / r["episodes"], 1),
                         "recorded (%)": None if rec is None else round(100 * rec["success_count"] / rec["episodes"], 1),
                         "recorded": "—" if rec is None else f"{rec['success_count']}/{rec['episodes']}, mean {rec['mean_len']:.1f} steps",
                         "label": ("running…" if name == current else "pending") if r is None else f"{r['success_count']}/{r['episodes']} · {r['mean_len']:.0f} steps"})
        df = pd.DataFrame(rows)
        y = alt.Y("run:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        x_scale = alt.Scale(domain=[0, 140])  # room right of 100% for the labels
        x = alt.X("solved (%):Q", scale=x_scale, title="episodes solved (%)", axis=alt.Axis(values=[0, 25, 50, 75, 100]))
        bars = alt.Chart(df).mark_bar(color=PALETTE["blue"]).encode(x=x, y=y, tooltip=["run", "label", "recorded"])
        ticks = alt.Chart(df).mark_tick(color=PALETTE["orange"], thickness=3, size=18).encode(
            x=alt.X("recorded (%):Q", scale=x_scale, title="episodes solved (%)"), y=y, tooltip=["run", "label", "recorded"])
        pending = alt.Chart(df).transform_filter("datum['solved (%)'] === null").mark_text(align="left", dx=4, color="#888").encode(
            x=alt.XDatum(0, scale=x_scale), y=y, text="label:N")
        labels = alt.Chart(df).transform_filter("datum['solved (%)'] !== null").mark_text(align="left", dx=8).encode(x=x, y=y, text="label:N")
        return mo.vstack([
            mo.md(f"**{title}** — {len(done)}/{len(order)} done. Blue: this run (solved, mean episode length). "
                  "Orange tick: the recorded orx run; hover a row for its numbers."),
            (bars + ticks + labels + pending).properties(width=360, height=24 * len(order)),
        ])


    def run_with_progress(networks, recorded_runs, title, episodes, seed):
        """Evaluate each network in turn, redrawing the progress chart after each one."""
        results = {}
        for name, W in networks.items():
            mo.output.replace(progress_chart(list(networks), results, recorded_runs, title, current=name))
            results[name] = evaluate(W, episodes, seed)
        mo.output.replace(progress_chart(list(networks), results, recorded_runs, title))
        return results

    return recorded_run, report, run_with_progress


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Experiment: does the base network solve the level?
    """)
    return


@app.cell
def _(EPISODES, SEED, base_raw, evaluate, recorded_run, report):
    base_result = evaluate(base_raw, EPISODES, SEED)  # the first evaluation also compiles the rollout (~10 s)
    report("base PPO", base_result, recorded_run("base_ppo_final_seed0.npz", EPISODES, SEED))
    return (base_result,)


@app.cell
def _(animate, base_raw, trace_episode):
    base_trace = trace_episode(base_raw)
    animate((base_trace, "base PPO"))
    return (base_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It drives toward the goal, climbs onto the ramp and stalls there until the episode times out. It never backs up:
    at the start p(reverse) ≈ 10⁻¹². (Each animation in this notebook is one episode in which the network always takes
    its most likely action, from the same start, so it does not change with `SEED`; the evaluations sample actions.)

    ### 1.3 Experiment: Astra's 6,073-entry edit
    """)
    return


@app.cell
def _(CR, mo, np, rounded):
    _ed = np.load(CR / "edited_actor.npz")
    astra = {k: _ed[k].astype(np.float64) for k in _ed.files}


    def changed_entries(W):
        """(tensor, index) of every weight that differs from the rounded base."""
        return [(k, tuple(int(i) for i in ix)) for k in rounded for ix in zip(*np.nonzero(W[k] != rounded[k]))]


    _changed = changed_entries(astra)
    _erased = sum(astra[t][i] == 0 for t, i in _changed)
    mo.md(f"{len(_changed):,} entries changed: {_erased:,} set to zero, {len(_changed) - _erased} given new values")
    return astra, changed_entries


@app.cell
def _(EPISODES, SEED, animate, astra, base_trace, evaluate, mo, recorded_run, report, trace_episode):
    astra_result = evaluate(astra, EPISODES, SEED)
    astra_trace = trace_episode(astra)
    mo.vstack([
        report("Astra edit", astra_result, recorded_run("edited_actor.npz", EPISODES, SEED)),
        animate((base_trace, "base PPO"), (astra_trace, "Astra edit")),
    ])
    return (astra_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The edited car backs up to the right first, then drives left over the ramp into the goal.

    ### 1.4 Experiment: strip the edit down — the ladder

    Each rung keeps fewer of Astra's entries (or, for C2, writes new ones from scratch) on top of the rounded base.
    The recipes are plain lists of `(tensor, index, value)`; `None` means "Astra's value at that index".
    """)
    return


@app.cell(hide_code=True)
def _(CR, astra, changed_entries, json, np, rounded):
    def apply_entries(entries):
        """Rounded base + entries; each entry is (tensor, index, value), value None = Astra's value there."""
        W = {k: v.copy() for k, v in rounded.items()}
        for t, idx, val in entries:
            W[t][tuple(idx)] = astra[t][tuple(idx)] if val is None else val
        return W


    # the saved entry lists of the campaign
    KEEP_1939 = [(t, tuple(i), None) for t, i in json.load(open(CR / "keep_joint_free.json"))]
    GREEDY12 = [(t, tuple(i), None) for t, i in json.load(open(CR.parent / "car-ramp-evidence" / "greedy_gf5_final.json"))]
    C2_14 = [(t, tuple(i), float(v)) for t, i, v in json.load(open(CR / "c2_edits.json"))]
    # the 1,718 waypoint's keep-list was not saved; read its entry set from the shipped checkpoint
    with np.load(CR / "greedy_waypoint_1718.npz") as _wp:
        WAYPOINT_1718 = [(t, i, None) for t, i in changed_entries({k: _wp[k].astype(np.float64) for k in _wp.files})]
    return C2_14, GREEDY12, KEEP_1939, WAYPOINT_1718, apply_entries


@app.cell
def _(C2_14, GREEDY12, KEEP_1939, WAYPOINT_1718, apply_entries, astra, changed_entries, mo):
    GREEDY11 = [e for e in GREEDY12 if e[:2] != ("actor1.W", (122, 126))]  # greedy-12 without its redundant entry
    C2_MIN_10 = [e for e in C2_14 if e[:2] not in {("embed.W", (52, 26)), ("actor0.W", (26, 114)), ("actor0.b", (23,)), ("actor0.b", (114,))}]

    ladder_networks = {
        "Astra edit (6,073)": astra,
        "joint_free (1,939)": apply_entries(KEEP_1939),  # 55 functional entries + 1,884 embed erasures
        "waypoint (1,718)": apply_entries(WAYPOINT_1718),
        "greedy-12": apply_entries(GREEDY12),  # greedy drop search on Astra's entries, run to a fixed point
        "greedy-11": apply_entries(GREEDY11),
        "C2, from scratch (14)": apply_entries(C2_14),  # designed from scratch, no erasures
        "C2-min, from scratch (10)": apply_entries(C2_MIN_10),
    }
    mo.md(" · ".join(f"{name}: {len(changed_entries(W)):,} entries" for name, W in ladder_networks.items()))
    return (ladder_networks,)


@app.cell(hide_code=True)
def _(CR, ladder_networks, np, rounded):
    # check: every network rebuilt above is bit-identical to the one the orx runs evaluated
    SHIPPED = {"Astra edit (6,073)": "edited_actor.npz", "joint_free (1,939)": "joint_free.npz",
               "waypoint (1,718)": "greedy_waypoint_1718.npz", "greedy-12": "greedy12_astra.npz", "greedy-11": "greedy11_drop02.npz",
               "C2, from scratch (14)": "c2_14entry.npz", "C2-min, from scratch (10)": "c2_min_10entry.npz"}


    def _same(W, file):
        with np.load(CR / file) as z:
            return all(np.array_equal(W[k], z[k].astype(np.float64)) for k in rounded)


    {name: _same(W, SHIPPED[name]) for name, W in ladder_networks.items()}
    return (SHIPPED,)


@app.cell
def _(EPISODES, SEED, SHIPPED, ladder_networks, recorded_run, run_with_progress):
    _recorded = {n: recorded_run(SHIPPED[n], EPISODES, SEED) for n in ladder_networks}
    ladder_results = run_with_progress(ladder_networks, _recorded, "The ladder", EPISODES, SEED)
    return (ladder_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every rung solves the level. The behavior needs a handful of weights, not 6,073: the 12-entry fixed point is a
    subset of Astra's own edit, and C2-min reaches the same with 10 entries written from scratch.

    ### 1.5 Experiment: is every one of the 12 needed?

    Remove each entry of greedy-12 in turn (revert it to the rounded base) and re-evaluate.
    """)
    return


@app.cell
def _(EPISODES, GREEDY12, SEED, apply_entries, recorded_run, run_with_progress):
    drop_networks = {f"without {t}[{','.join(map(str, i))}]": apply_entries(GREEDY12[:k] + GREEDY12[k + 1:])
                     for k, (t, i, _) in enumerate(GREEDY12)}
    # the recorded audit ran 256 episodes at seed 0 (tags drop-00 … drop-11)
    _recorded = {n: recorded_run(tag=f"drop-{k:02d}", episodes=EPISODES, seed=SEED) for k, n in enumerate(drop_networks)}
    drop_results = run_with_progress(drop_networks, _recorded, "Drop one entry of greedy-12", EPISODES, SEED)
    return (drop_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten of the twelve are each indispensable. `actor1.W[122,126]` is redundant (dropping it gives greedy-11), and
    `actor_head.b[0]` only costs a few episodes.

    ### 1.6 Experiment: why keep the 1,884 erasures in `joint_free`?

    Astra zeroed most of embed columns 6, 7 and 8, turning three dead ReLU units into clean single-input taps for car x,
    car vx and green-wheel x. Undo one column's erasures at a time:
    """)
    return


@app.cell
def _(EPISODES, KEEP_1939, SEED, apply_entries, astra, recorded_run, run_with_progress):
    restore_networks = {
        f"restore column {c} (reads {what})": apply_entries(
            [e for e in KEEP_1939 if not (e[0] == "embed.W" and e[1][1] == c and astra["embed.W"][e[1]] == 0)])
        for c, what in ((6, "car x"), (7, "car vx"), (8, "green-wheel x"))
    }
    _recorded = {n: recorded_run(tag=f"restore-col{c}-s0", episodes=EPISODES, seed=SEED) for n, c in zip(restore_networks, (6, 7, 8))}
    restore_results = run_with_progress(restore_networks, _recorded, "Restore one embed column", EPISODES, SEED)
    return (restore_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every restore breaks the task: those erasures are what make the three taps clean.

    ### 1.7 Look inside: what the 12 entries do

    Trace one episode and watch the units the edit wires together. Unit `L:u` is the tanh output of `actor{L}` unit
    `u`. In greedy-12, PPO's own speed detector `0:122` feeds a gate `1:126`, which drives a relay `2:126`; `0:127` is
    a green-wheel comparator. For C2-min try `0:23, 1:29, 2:105, 4:73`.
    """)
    return


@app.cell
def _(ladder_networks):
    TRACE_NETWORK = "greedy-12"  # any key of ladder_networks
    TRACE_UNITS = [(0, 122), (0, 127), (1, 126), (2, 126)]  # (actor layer, unit)
    inside_W = ladder_networks[TRACE_NETWORK]
    return TRACE_NETWORK, TRACE_UNITS, inside_W


@app.cell
def _(PALETTE, TRACE_NETWORK, TRACE_UNITS, base_trace, inside_W, np, plt, trace_episode):
    inside_trace = trace_episode(inside_W)
    _t = np.arange(inside_trace["steps"])
    _fig, _axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    _axes[0].plot(_t, inside_trace["obs"][:, 0], color=PALETTE["blue"], label="car x")
    _axes[0].plot(_t, inside_trace["obs"][:, 1], color=PALETTE["orange"], label="car vx")
    _axes[0].plot(_t, inside_trace["obs"][:, 2], color=PALETTE["green"], label="green-wheel x")
    _axes[0].plot(np.arange(base_trace["steps"]), base_trace["obs"][:, 0], color=PALETTE["blue"], alpha=0.3, linestyle="--", label="car x, base PPO")
    _axes[0].set_ylabel("observation")
    for _k, (_L, _u) in enumerate(TRACE_UNITS):
        _axes[1].plot(_t, inside_trace["h"][:, _L + 1, _u], label=f"{_L}:{_u}", color=list(PALETTE.values())[_k % 7])
    _axes[1].set_ylabel("unit activation")
    for _a in range(3):
        _axes[2].plot(_t, inside_trace["probs"][:, _a], label=f"motor action {_a}")
    _axes[2].set_ylabel("action probability")
    _axes[2].set_xlabel("step")
    for _ax in _axes:
        _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _axes[0].set_title(f"{TRACE_NETWORK}: {'solved' if inside_trace['solved'] else 'not solved'} in {inside_trace['steps']} steps")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.8 Your turn: write an edit

    Edit the list and rerun the cell. Each entry sets one weight on top of the rounded base:
    `(tensor, index, value)`. Tensors are `embed.W`, `embed.b`, `actor0.W` … `actor4.b`, `actor_head.W`,
    `actor_head.b`. It starts as C2-min; try deleting a line or changing a gain.
    """)
    return


@app.cell
def _(
    EPISODES,
    SEED,
    animate,
    apply_entries,
    base_trace,
    evaluate,
    mo,
    report,
    trace_episode,
):
    my_edit = [
        ("embed.b", (26,), 15.2),
        ("actor0.W", (116, 23), 8.0),
        ("actor1.W", (114, 29), 12.0),
        ("actor1.W", (23, 29), 12.0),
        ("actor1.b", (29,), -18.0),
        ("actor2.W", (29, 105), 10.0),
        ("actor3.W", (105, 17), 10.0),
        ("actor4.W", (17, 73), 10.0),
        ("actor_head.W", (73, 1), -40.0),
        ("actor_head.W", (73, 2), 40.0),
    ]
    my_network = apply_entries(my_edit)
    my_result = evaluate(my_network, EPISODES, SEED)
    mo.vstack([
        report(f"your edit ({len(my_edit)} entries)", my_result),
        animate((base_trace, "base PPO"), (trace_episode(my_network), "your edit")),
    ])
    return (my_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · Experiment 8: Qwen3.5-0.8B math edit, and how much of it is needed

    Astra's p50c edit makes Qwen3.5-0.8B stop looping on math (MATH-500 7.9% → 48.1% in the published evaluation).
    It changes **130,188** bf16 entries of `layers.{15,19}.mlp.down_proj.weight` (1024 × 3584 each). The minimal-edit
    campaign built 27 subsets of it and screened each on the same 256 MATH-train problems (vLLM, thinking on, temp
    0.6, 32k tokens, one sample each). Those screens ran on a GPU and are bundled as raw per-problem outputs; the
    analysis below recomputes everything from them, and §2.5 lets you run the model yourself.

    ### 2.1 Apply Astra's edit and recover what it is
    """)
    return


@app.cell(hide_code=True)
def _(EX, json, np):
    from st_numpy import bf16_bits, bf16_to_f32  # the campaign's own bf16 conversions (code/st_numpy.py)

    LAYERS = (15, 19)
    base_bits = {L: bf16_bits(np.load(EX / f"base_l{L}_down_proj_f32.npy")) for L in LAYERS}
    W0 = {L: bf16_to_f32(base_bits[L]).astype(np.float64) for L in LAYERS}
    _z = np.load(EX / "edit.npz")
    edit_idx = {L: _z[f"t{i}_idx"].astype(np.int64) for i, L in enumerate(LAYERS)}
    edit_val = {L: _z[f"t{i}_val"] for i, L in enumerate(LAYERS)}
    selected = {L: np.array(json.load(open(EX / "frozen_weights_manifest.json"))[str(L)]["selected_neurons"]) for L in LAYERS}


    def build_bits(spec):
        """Keep the subset of Astra's stored bf16 entries a spec selects (cols/rows/entries per layer; a missing key
        means no filter), optionally scaling the change. A port of the campaign's code/build_candidate.py."""
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

    return LAYERS, W0, base_bits, bf16_bits, bf16_to_f32, build_bits, n_changed, selected


@app.cell
def _(LAYERS, base_bits, build_bits, mo):
    astra_bits = build_bits({})  # an empty spec keeps every entry of Astra's edit
    mo.md(" · ".join(f"layer {L}: {int((astra_bits[L] != base_bits[L]).sum()):,} entries changed" for L in LAYERS))
    return (astra_bits,)


@app.cell
def _(
    EX,
    LAYERS,
    W0,
    astra_bits,
    bf16_bits,
    bf16_to_f32,
    json,
    mo,
    np,
    selected,
):
    # SVD of the change, and Astra's closed form W = W0 - 6 u (m64 ⊙ uᵀW0) checked against the stored bits
    rank1 = {}
    _metrics = {}
    for _L in LAYERS:
        _D = bf16_to_f32(astra_bits[_L]).astype(np.float64) - W0[_L]
        _U, _s, _ = np.linalg.svd(_D[:, selected[_L]], full_matrices=False)  # Δ is zero outside the 64 edited columns
        _sign = np.sign(_U[:, 0] @ np.load(EX / f"factors_layer{_L}.npz")["u"])
        _u = _U[:, 0] * _sign  # aligned with the campaign's saved factor
        _mask = np.zeros(3584)
        _mask[selected[_L]] = 1
        _Wf = W0[_L] - 6.0 * np.outer(_u, _mask * (_u @ W0[_L]))
        _stored = bf16_to_f32(astra_bits[_L])[:, selected[_L]].astype(np.float64)
        _diff = np.abs(bf16_to_f32(bf16_bits(_Wf.astype(np.float32)))[:, selected[_L]] - _stored)
        _step = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(_stored), 1e-30))) - 7)  # one bf16 rounding step at each value
        rank1[_L] = {"u": _u, "c": _u @ W0[_L]}
        _metrics[_L] = {
            "columns (MLP neurons) touched": f"{int((_D != 0).any(0).sum())} of 3,584",
            "σ₂ / σ₁ of Δ (rank-1 ⇔ ≈ 0)": f"{_s[1] / _s[0]:.1e}",
            "‖Δ‖ / ‖W₀‖": f"{np.linalg.norm(_D) / np.linalg.norm(W0[_L]):.3f}",
            "closed form within one bf16 step (of 65,536)": f"{int((_diff <= _step).sum()):,}",
            "largest closed-form difference": f"{_diff.max():.1e}",
            "median |Δ| on touched entries": f"{np.median(np.abs(_D[:, selected[_L]])):.1e}",
            "Astra's own audit, exact u": f"rel. error {json.load(open(EX / 'frozen_edit_audit.json'))[str(_L)]['formula_relative_error']:.1e}",
        }
    mo.vstack([
        mo.ui.table([{"": k, "layer 15": _metrics[15][k], "layer 19": _metrics[19][k]} for k in _metrics[15]], selection=None, pagination=False),
        mo.md(f"The change is rank-1: one direction u subtracted through 64 MLP neurons per layer, the same direction in both layers "
              f"(|cos(u₁₅, u₁₉)| = **{abs(rank1[15]['u'] @ rank1[19]['u']):.3f}**). Astra's exact u is not in the logs, so u here is "
              "estimated from Δ after bf16 rounding; that reproduces 98–99% of the touched entries to within one rounding step, and "
              "Astra's run checked the formula against its exact u (last row)."),
    ])
    return (rank1,)


@app.cell
def _(LAYERS, PALETTE, alt, mo, np, pd, rank1, selected):
    _rows, _txt = [], []
    for _L in LAYERS:
        _c = np.abs(rank1[_L]["c"])
        _rank = np.empty(3584, int)
        _rank[np.argsort(-_c)] = np.arange(1, 3585)
        _edited = np.zeros(3584, bool)
        _edited[selected[_L]] = True
        _rows.append(pd.DataFrame({"layer": f"layer {_L}", "neuron": np.arange(3584), "rank": _rank,
                                   "writing strength |u·W₀|": _c, "edited by Astra": np.where(_edited, "yes", "no")}))
        _txt.append(f"layer {_L}: {int((_rank[_edited] <= 64).sum())} of the 64 are among the 64 strongest writers, the weakest ranks {int(_rank[_edited].max()):,}")
    mo.vstack([
        mo.md("**Which neurons did Astra edit?** Neuron j writes the direction with strength |u·W₀[:, j]|. Astra did not take the "
              "64 strongest writers: it ranked by strength × how much more the neuron fires on looping text (its notes, "
              "`data/docs/exp8/notes_astra_method.md`). " + "; ".join(_txt) + ". Hover a point for its neuron number."),
        alt.Chart(pd.concat(_rows)).mark_circle().encode(
            x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="neuron rank by writing strength (log)",
                    axis=alt.Axis(values=[1, 10, 64, 100, 1000, 3584], format=",")),
            y=alt.Y("writing strength |u·W₀|:Q"),
            color=alt.Color("edited by Astra:N", scale=alt.Scale(domain=["no", "yes"], range=["#9DB4C8", PALETTE["red"]]),
                            legend=alt.Legend(orient="bottom")),
            size=alt.condition(alt.datum["edited by Astra"] == "yes", alt.value(28), alt.value(10)),
            order=alt.Order("edited by Astra:N"),
            tooltip=["layer", "neuron", "rank", alt.Tooltip("writing strength |u·W₀|:Q", format=".4f"), "edited by Astra"],
        ).properties(width=230, height=220).facet(column=alt.Column("layer:N", title=None)).resolve_scale(y="independent"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Experiment: build the campaign's 27 candidates

    Each candidate keeps part of Astra's edit: the top-k neurons (A), the B largest entries per layer (D), the top-R
    output rows of layer 15 (ER), or one layer only (B). `make_spec` regenerates the specs from the saved factors;
    `build_bits` applies them. The check: every candidate has the campaign's entry count, and the A, D and ER specs
    regenerate exactly (the B specs, one layer kept whole or its top 32 neurons, are used as saved).
    """)
    return


@app.cell(hide_code=True)
def _(EX, LAYERS, json, np, selected):
    _fz = {L: np.load(EX / f"factors_layer{L}.npz") for L in LAYERS}
    # the campaign's emit_specs.py ordered by the factors it saved; use those so specs regenerate exactly
    _fu = {L: _fz[L]["u"].astype(np.float64) for L in LAYERS}
    _fv = {L: _fz[L]["v"].astype(np.float64) for L in LAYERS}
    order_v = {L: [int(selected[L][i]) for i in np.argsort(-np.abs(_fv[L][selected[L]]))] for L in LAYERS}
    order_u = {L: [int(i) for i in np.argsort(-np.abs(_fu[L]))] for L in LAYERS}


    def top_entries(L, B):
        if B <= 0:
            return []
        d = np.abs(np.outer(_fu[L], _fv[L])).ravel()
        i = np.argpartition(d, -B)[-B:]
        i = i[np.argsort(-d[i])]
        return [[int(k // 3584), int(k % 3584)] for k in i]


    def make_spec(family, n):
        """The campaign's candidate families (emit_specs.py). family: "A", "D", "B15", "B19" or "ER"."""
        return {
            "A": lambda: {"scale": 1.0, "cols15": order_v[15][:n], "cols19": order_v[19][:n]},
            "D": lambda: {"scale": 1.0, "entries15": top_entries(15, n), "entries19": top_entries(19, n)},
            "B15": lambda: {"scale": 1.0, "cols15": order_v[15][:n], "cols19": []},
            "B19": lambda: {"scale": 1.0, "cols19": order_v[19][:n], "cols15": []},
            "ER": lambda: {"scale": 1.0, "rows15": order_u[15][:n], "cols19": []},
        }[family]()


    CAMPAIGN_SPECS = {p.stem: json.load(open(p)) for p in sorted((EX / "specs").glob("*.json"))}
    CAMPAIGN_SPECS["B15_full_s125"] = {"scale": 1.25, "cols19": []}
    CAMPAIGN_SPECS["G000_scale"] = {"scale": 0.0}  # the unedited base
    return CAMPAIGN_SPECS, make_spec


@app.cell
def _(CAMPAIGN_SPECS, EX, build_bits, json, make_spec, mo, n_changed):
    _size = {"A": lambda c: int(c.split("_topk")[1]), "D": lambda c: int(c[1:].split("k_")[0]) * 1024,
             "ER": lambda c: int(c[2:].split("_")[0])}
    _recorded = json.load(open(EX / "candidate_entries.json"))
    entry_counts = {}
    _rows = []
    for _cid, _spec in mo.status.progress_bar(list(CAMPAIGN_SPECS.items()), title="building candidates"):
        entry_counts[_cid] = n_changed(build_bits(_spec))
        _fam = next((k for k in ("ER", "A", "D") if _cid.startswith(k)), None)
        _rows.append({"candidate": _cid, "entries (built now)": entry_counts[_cid],
                      "campaign's count": _recorded.get(_cid, "—"),
                      "same": "—" if _cid not in _recorded else ("✅" if _recorded[_cid] == entry_counts[_cid] else "❌"),
                      "spec regenerated": "—" if _fam is None else ("✅" if make_spec(_fam, _size[_fam](_cid)) == _spec else "❌")})
    mo.ui.table(sorted(_rows, key=lambda r: r["entries (built now)"]), selection=None, pagination=False, page_size=30)
    return (entry_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Design your own candidate

    Change the family and size, rerun, and the next sections use it (`my_spec`). D keeps the B largest entries per
    layer, A the top-k neurons, ER the top-R rows of layer 15, B15/B19 one layer's top-k neurons.
    """)
    return


@app.cell
def _(LAYERS, W0, astra_bits, bf16_to_f32, build_bits, make_spec, mo, n_changed, np):
    my_spec = make_spec("D", 4096)  # family, size
    _bits = build_bits(my_spec)
    _mass = sum(np.abs(bf16_to_f32(_bits[L]) - W0[L]).sum() for L in LAYERS) / sum(np.abs(bf16_to_f32(astra_bits[L]) - W0[L]).sum() for L in LAYERS)
    _n = n_changed(_bits)
    mo.md(f"{_n:,} changed entries ({100 * _n / n_changed(astra_bits):.1f}% of Astra's edit), carrying {100 * _mass:.1f}% of its |Δ| mass")
    return (my_spec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Analyze the screening runs

    `data/screens/` holds one row per problem per candidate: correct or not, why generation stopped, tokens used.
    Each candidate is paired with the full edit (`A64_topk64`) on the same 256 problems.
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, json, math, mo, np):
    screens = {}
    for _p in sorted((ROOT / "data" / "screens").glob("screen_*.jsonl")):
        screens[_p.stem[len("screen_"):]] = {r["id"]: r for r in map(json.loads, open(_p))}
    # specs of screens you add yourself (written by §2.6 next to the checkpoint's name)
    screen_specs = {p.stem[len("spec_"):]: json.load(open(p)) for p in (ROOT / "data" / "screens").glob("spec_*.json")}


    def acc(cid):
        return np.mean([r["correct"][0] for r in screens[cid].values()])


    def paired(a, b, n_boot=4000, seed=0):
        """a vs b on the same problems: accuracy difference, discordant counts, exact McNemar p, bootstrap 95% CI."""
        ids = sorted(set(screens[a]) & set(screens[b]))
        x = np.array([screens[a][i]["correct"][0] for i in ids])
        y = np.array([screens[b][i]["correct"][0] for i in ids])
        n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        k, n = min(n10, n01), n10 + n01
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
        d = x - y
        boot = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(1)
        return {"diff": d.mean(), "a_only": n10, "b_only": n01, "p": p, "ci": tuple(np.percentile(boot, [2.5, 97.5])), "n": len(ids)}


    mo.md(f"{len(screens)} screens loaded: {', '.join(sorted(screens))}")
    return acc, paired, screen_specs, screens


@app.cell
def _(acc, build_bits, entry_counts, mo, n_changed, np, paired, pd, screen_specs, screens):
    FAMILIES = {"A": "A · top-k neurons", "D": "D · top-B entries", "ER": "ER · layer-15 rows", "B": "B · one layer", "G": "unedited base", "": "your candidates"}
    _rows = []
    _full = len(screens["A64_topk64"])
    _incomplete = [c for c in screens if set(screens[c]) != set(screens["A64_topk64"])]
    for _c in screens:
        if _c in _incomplete:  # accuracies over fewer problems aren't comparable with the 256-problem screens
            continue
        _p = paired(_c, "A64_topk64")
        _count = entry_counts[_c] if _c in entry_counts else n_changed(build_bits(screen_specs[_c])) if _c in screen_specs else None
        _rows.append({"candidate": _c, "family": next(v for k, v in FAMILIES.items() if _c.startswith(k)),
                      "changed entries": _count, "accuracy (%)": round(100 * acc(_c), 1),
                      "vs full edit (pp)": round(100 * _p["diff"], 1),
                      "95% CI": f"{100 * _p['ci'][0]:+.1f} to {100 * _p['ci'][1]:+.1f}", "McNemar p": float(f"{_p['p']:.2g}"),
                      "cut at 32k (%)": round(100 * np.mean([x["finish"][0] == "length" for x in screens[_c].values()]))})
    screen_table = pd.DataFrame(_rows).sort_values("changed entries")
    mo.vstack([screen_table] + ([mo.md(f"Left out, because they don't cover the same {_full} problems: {', '.join(_incomplete)}. "
                                       "The paired comparison below still works for them, on the problems they share.")] if _incomplete else []))
    return FAMILIES, screen_table


@app.cell
def _(mo, screen_table, screens):
    PICK = mo.ui.dropdown(list(screen_table["candidate"]), value="D4k_entries", label="candidate")
    AGAINST = mo.ui.dropdown(sorted(screens), value="A64_topk64", label="against")
    mo.hstack([PICK, AGAINST], justify="start")
    return AGAINST, PICK


@app.cell
def _(FAMILIES, PALETTE, PICK, acc, alt, mo, screen_table):
    _fams = [v for k, v in FAMILIES.items() if k != "G"]  # the base has no changed entries, so no point on a log axis
    _hl = alt.datum.candidate == PICK.value
    mo.vstack([
        alt.Chart(screen_table[screen_table["changed entries"].fillna(0) > 0]).mark_point(filled=True).encode(
            x=alt.X("changed entries:Q", scale=alt.Scale(type="log", nice=False, padding=20), title="changed bf16 entries"),
            y=alt.Y("accuracy (%):Q", scale=alt.Scale(zero=False, padding=12), title="MATH-train screening accuracy (%)"),
            color=alt.Color("family:N", scale=alt.Scale(domain=_fams, range=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"], PALETTE["red"]]),
                            legend=alt.Legend(orient="bottom", title=None, columns=2)),
            shape=alt.Shape("family:N", scale=alt.Scale(domain=_fams, range=["circle", "square", "triangle-up", "diamond", "cross"]), legend=None),
            size=alt.condition(_hl, alt.value(260), alt.value(80)),
            stroke=alt.condition(_hl, alt.value("black"), alt.value(None)),
            tooltip=["candidate", "family", alt.Tooltip("changed entries:Q", format=","), "accuracy (%)", "vs full edit (pp)", "95% CI", "McNemar p", "cut at 32k (%)"],
        ).properties(width=500, height=320),
        mo.md(f"Choosing **entries** by |Δ| (D) beats choosing whole **neurons** (A): D4k — 8,192 entries, 6% of the edit — ties the full edit "
              f"({100 * acc('A64_topk64'):.1f}%, the right-most point), while A needs ~48 neurons per layer (~98k entries) to get within noise. "
              f"Unedited base: {100 * acc('G000_scale'):.1f}%. With one sample on 256 problems, differences under ~5pp are noise. "
              "Hover a point for its numbers; the outlined point is the candidate picked above."),
    ])
    return


@app.cell
def _(AGAINST, PALETTE, PICK, alt, mo, paired, pd, screens):
    _a, _b = PICK.value, AGAINST.value
    mo.stop(not set(screens[_a]) & set(screens[_b]), mo.md(f"**{_a}** and **{_b}** share no problems, so there is nothing to compare."))
    _r = paired(_a, _b)
    _cats = {(1, 1): "both solved", (1, 0): f"only {_a}", (0, 1): f"only {_b}", (0, 0): "neither"}
    _ids = sorted(set(screens[_a]) & set(screens[_b]))
    _df = pd.DataFrame([{"problem": i.removeprefix("math_train_"), "col": k % 32, "row": k // 32,
                         "outcome": _cats[(screens[_a][i]["correct"][0], screens[_b][i]["correct"][0])],
                         f"{_a} tokens": screens[_a][i]["n_tokens"][0], f"{_b} tokens": screens[_b][i]["n_tokens"][0]}
                        for k, i in enumerate(_ids)])
    mo.vstack([
        mo.md(f"**{_a} − {_b}** on {_r['n']} problems: **{100 * _r['diff']:+.1f}pp** (95% CI {100 * _r['ci'][0]:+.1f} to {100 * _r['ci'][1]:+.1f}); "
              f"solved only by {_a}: {_r['a_only']}, only by {_b}: {_r['b_only']}; exact McNemar p = {_r['p']:.3g}. One square per problem; hover for details."),
        alt.Chart(_df).mark_rect(stroke="white", strokeWidth=1).encode(
            x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None),
            color=alt.Color("outcome:N", title=None, scale=alt.Scale(domain=list(_cats.values()),
                            range=[PALETTE["green"], PALETTE["blue"], PALETTE["orange"], "#dddddd"]), legend=alt.Legend(orient="bottom", columns=4)),
            tooltip=["problem", "outcome", f"{_a} tokens", f"{_b} tokens"],
        ).properties(width=500, height=130),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 Run the model yourself

    Paired generations, unedited base vs a candidate, on the first screening problems, with the campaign's chat template,
    thinking on, and sampling (temp 0.6, top-p 0.95, top-k 20, presence penalty 1.5, as `code/eval_vllm.py`; the order
    in which vLLM and transformers apply these can differ slightly). Needs an Apple-silicon GPU or CUDA (CPU runs at ~0.5
    tokens/s) and downloads Qwen3.5-0.8B (~1.7 GB) once. On an M-series Mac, 8 problems × 1,024 tokens takes ~1.5 min.

    **What to expect.** At laptop budgets both versions are still thinking when they are cut off, so accuracy barely
    moves. The edit's effect shows at the campaign's 32k cap (base cut off on 91% of problems, full edit 30%), so the
    table also shows each problem's recorded 32k screening. Until you run it, the table shows a bundled example run.
    """)
    return


@app.cell(hide_code=True)
def _(EX, LAYERS, ROOT, W0, bf16_to_f32, build_bits, json, mo, np, time):
    from eval_vllm import INSTR
    import hashlib as _hashlib

    QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


    def candidate_name(label, spec):
        """Campaign ids name themselves; your own spec is named by its content, so different specs never overwrite."""
        if label != "my_spec":
            return label
        return "my_spec_" + _hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


    def generate_pair(label, spec, problems, max_new_tokens):
        """Generate on the first `problems` screening problems with the unedited base and with `spec`, using the campaign's
        chat template and sampling (code/eval_vllm.py: temp 0.6, top-p 0.95, top-k 20, presence penalty 1.5)."""
        import torch
        from huggingface_hub import snapshot_download
        from math_verify import parse, verify
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
        from transformers.utils import logging as hf_logging

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        mo.stop(device == "cpu", mo.md("**No Apple GPU or CUDA found.** On a CPU Qwen generates ~0.5 tokens/s, so this run would take hours. "
                                       "Run the notebook on a GPU machine (or molab with a GPU attached)."))
        hf_logging.set_verbosity_error()  # Qwen3.5 warns about optional CUDA kernels on every forward pass
        with mo.status.spinner("downloading / loading Qwen3.5-0.8B"):
            # public model; token=False so a stale token on the viewer's machine can't break the download
            path = snapshot_download("Qwen/Qwen3.5-0.8B", revision=QWEN_REVISION, token=False)
            tok = AutoTokenizer.from_pretrained(path, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
        for L in LAYERS:
            assert np.array_equal(model.model.layers[L].mlp.down_proj.weight.detach().cpu().numpy(), W0[L].astype(np.float32)), "base tensors differ from data/exp8"

        rows = [json.loads(l) for l in open(EX / "screening_256.jsonl")][:problems]
        prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTR}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in rows]
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        stop_ids = set(np.atleast_1d(model.generation_config.eos_token_id).tolist())

        class PresencePenalty(LogitsProcessor):
            # vLLM's presence_penalty: subtract a constant from every token already in this completion
            def __init__(self, prompt_len, penalty):
                self.prompt_len, self.penalty = prompt_len, penalty

            def __call__(self, input_ids, scores):
                generated = input_ids[:, self.prompt_len:]
                if generated.shape[1]:
                    scores = scores - self.penalty * torch.zeros_like(scores).scatter_(1, generated, 1.0)
                return scores

        def loop_share(ids, w=16):
            seen, rep = set(), 0
            for i in range(len(ids) - w + 1):
                g = tuple(ids[i:i + w])
                rep += g in seen
                seen.add(g)
            return rep / max(len(ids) - w + 1, 1)

        def run(bits):
            with torch.no_grad():
                for L in LAYERS:
                    model.model.layers[L].mlp.down_proj.weight.copy_(torch.from_numpy(bf16_to_f32(bits[L])))
                torch.manual_seed(0)
                enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                     logits_processor=LogitsProcessorList([PresencePenalty(enc["input_ids"].shape[1], 1.5)]), pad_token_id=pad)
            res = []
            for p, row in zip(rows, out[:, enc["input_ids"].shape[1]:].cpu().tolist()):
                stopped = any(t in stop_ids for t in row)  # ended on its own, even if on the last allowed token
                ids = [t for t in row if t != pad]
                text = tok.decode(ids, skip_special_tokens=True)
                try:
                    ok = bool(verify(parse(f"${p['answer']}$"), parse(text)))
                except Exception:  # math-verify raises on unparsable answers; that is a wrong answer
                    ok = False
                res.append({"correct": ok, "cut": not stopped, "tokens": len(ids), "loop": loop_share(ids), "text": text})
            return res

        name = candidate_name(label, spec)
        t0 = time.time()
        arms = {}
        for arm, arm_spec in mo.status.progress_bar([("base", {"scale": 0.0}), (name, spec)], title="generating"):
            arms[arm] = run(build_bits(arm_spec))
        result = {"candidate": name, "spec": spec, "max_new_tokens": max_new_tokens, "problems": [p["id"] for p in rows], "device": device,
                  "wall_s": round(time.time() - t0), "arms": arms}
        (ROOT / "runs").mkdir(exist_ok=True)
        (ROOT / "runs" / f"generations_{name}_{len(rows)}x{max_new_tokens}_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(result))
        return result | {"source": "this machine"}


    def write_checkpoint(name, spec):
        """Base Qwen3.5-0.8B with the candidate's two tensors swapped in, as checkpoints/<name>/ (~1.7 GB).
        Returns a new screen name for its vLLM result, so a run never overwrites a recorded or earlier screen."""
        import os

        import st_numpy
        from huggingface_hub import snapshot_download

        qwen_path = snapshot_download("Qwen/Qwen3.5-0.8B", revision=QWEN_REVISION, token=False)
        out = ROOT / "checkpoints" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "candidate_spec.json").write_text(json.dumps(spec))
        screen = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        # so §2.4 can count the entries of the screen you add under this name
        (ROOT / "data" / "screens" / f"spec_{screen}.json").write_text(json.dumps(spec))
        st = next(f for f in os.listdir(qwen_path) if f.endswith(".safetensors"))
        for f in os.listdir(qwen_path):
            if f != st and not (out / f).exists():
                os.symlink(os.path.join(qwen_path, f), out / f)
        with mo.status.spinner("writing safetensors"):
            tensors = st_numpy.load(os.path.join(qwen_path, st))
            bits = build_bits(spec)
            for L in LAYERS:
                tensors[f"model.language_model.layers.{L}.mlp.down_proj.weight"] = ("BF16", bits[L])
            # write beside the old file and swap it in, so an interrupted write never leaves a truncated checkpoint
            st_numpy.save(str(out / (st + ".partial")), tensors, {"format": "pt"})
            os.replace(out / (st + ".partial"), out / st)
        return screen

    return candidate_name, generate_pair, write_checkpoint


@app.cell
def _():
    RUN_QWEN = False  # set True and run this cell to generate on this machine
    GENERATE = {"candidate": "A64_topk64", "problems": 8, "max_new_tokens": 1024}  # candidate: a campaign id or "my_spec"
    return GENERATE, RUN_QWEN


@app.cell
def _(CAMPAIGN_SPECS, EX, GENERATE, RUN_QWEN, generate_pair, json, mo, my_spec):
    if RUN_QWEN:
        _spec = my_spec if GENERATE["candidate"] == "my_spec" else CAMPAIGN_SPECS[GENERATE["candidate"]]
        generations = generate_pair(GENERATE["candidate"], _spec, GENERATE["problems"], GENERATE["max_new_tokens"])
    else:
        _ex = EX / "example_generations.json"
        generations = json.load(open(_ex)) | {"source": "bundled example (Apple-silicon Mac)"} if _ex.exists() else None
    mo.md("_No bundled example run in `data/exp8/`; set `RUN_QWEN = True` to make one._" if generations is None else
          f"{generations['source']}: {generations['candidate']} vs base, {len(generations['problems'])} problems × "
          f"{generations['max_new_tokens']:,} tokens, {generations['wall_s']} s")
    return (generations,)


@app.cell
def _(generations, mo, np, screens):
    mo.stop(generations is None)
    _T, _cand = generations["max_new_tokens"], generations["candidate"]
    _b, _c = generations["arms"]["base"], generations["arms"][_cand]


    def _rec(cid, pid):
        r = screens.get(cid, {}).get(pid)
        return "—" if r is None else f"{'✓' if r['correct'][0] else '✗'} {r['n_tokens'][0]:,} tok{' (cut)' if r['finish'][0] == 'length' else ''}"


    def _arm(x):
        return ("✓" if x["correct"] else "✗") + (" cut" if x["cut"] else "") + f" · loop {100 * x['loop']:.0f}%"


    def _summary(r):
        return (f"{sum(x['correct'] for x in r)}/{len(r)} correct · {sum(x['cut'] for x in r)}/{len(r)} cut at {_T:,} tokens · "
                f"repeated-16-gram share {100 * np.mean([x['loop'] for x in r]):.0f}%")


    mo.vstack([
        mo.md(f"**base:** {_summary(_b)}<br>**{_cand}:** {_summary(_c)}"),
        mo.ui.table([{"problem": pid.removeprefix("math_train_"), "base": _arm(b), _cand: _arm(c),
                      "base @ 32k (screen)": _rec("G000_scale", pid), f"{_cand} @ 32k (screen)": _rec(_cand, pid)}
                     for pid, b, c in zip(generations["problems"], _b, _c)], selection=None, pagination=False),
    ])
    return


@app.cell
def _(generations, mo):
    mo.stop(generations is None)
    READ = mo.ui.dropdown({f"{arm}: {pid.removeprefix('math_train_')}": (arm, i) for arm in generations["arms"] for i, pid in enumerate(generations["problems"])},
                          value=f"{generations['candidate']}: {generations['problems'][0].removeprefix('math_train_')}", label="read a generation")
    READ
    return (READ,)


@app.cell
def _(READ, generations, mo):
    import html

    _arm, _i = READ.value
    _r = generations["arms"][_arm][_i]
    mo.vstack([
        mo.md(f"{_r['tokens']:,} tokens · repeated-16-gram share {100 * _r['loop']:.0f}% · {'correct' if _r['correct'] else 'not correct'}"
              f"{' · cut at the token cap' if _r['cut'] else ''} · last 3,000 characters:"),
        mo.Html("<pre style='white-space: pre-wrap; font-size: 12px; max-height: 420px; overflow: auto'>…" + html.escape(_r["text"][-3000:]) + "</pre>"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6 Replicate a screen at full budget (GPU)

    Set `CHECKPOINT` to a candidate id (or `"my_spec"`) and run the cell. It writes base Qwen3.5-0.8B with the
    candidate's two tensors swapped in to `checkpoints/<name>/` (~1.7 GB) and prints the campaign's screening command.
    """)
    return


@app.cell
def _(CAMPAIGN_SPECS, candidate_name, mo, my_spec, write_checkpoint):
    CHECKPOINT = None  # a campaign id such as "D4k_entries", or "my_spec"

    if CHECKPOINT is None:
        _out = mo.md("_Set `CHECKPOINT` to write one._")
    else:
        _spec = my_spec if CHECKPOINT == "my_spec" else CAMPAIGN_SPECS[CHECKPOINT]
        _name = candidate_name(CHECKPOINT, _spec)
        _screen = write_checkpoint(_name, _spec)
        _out = mo.md(f"Wrote `checkpoints/{_name}/`. Copy the folder (`cp -L` resolves its symlinks into the Hugging Face cache) to a CUDA machine "
                     f"with `vllm` and `math-verify`, and from the repo run:\n\n```sh\npython code/eval_vllm.py --model checkpoints/{_name} "
                     f"--problems data/exp8/screening_256.jsonl \\\n    --out data/screens/screen_{_screen}.jsonl --n 1 --max-tokens 32768 --gpu-mem 0.85\n"
                     f"cp checkpoints/{_name}/candidate_spec.json data/screens/spec_{_screen}.json\n```\n"
                     "The result and its spec land next to the other screens, so §2.4 includes it on the next run there "
                     f"(or here, after copying `screen_{_screen}.jsonl` back).")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · What both campaigns found

    | | Car-ramp actor | Exp 8, Qwen3.5-0.8B |
    |:--|:--|:--|
    | Astra's edit | 6,073 entries | 130,188 bf16 entries |
    | What it is | ~55 functional writes + 6,018 erasures isolating repurposed units | one direction subtracted through 64 chosen MLP neurons in each of 2 layers |
    | Smallest subset that keeps the behavior | **12 entries** (§1.4–1.5), 11 without the redundant one | unknown: **D4k (8,192 entries)** is the smallest tested subset whose observed score is within 1pp of the full edit (58.2% vs 58.6%); D1k and D2k (2,048 and 4,096) score 3–4pp lower, inside the screening noise, so they may also suffice (§2.4) |
    | Written from scratch | **10 entries** (§1.4, §1.8) | — |
    | Evidence | exact: Kinetix rollouts you just re-ran; at the default `EPISODES` and `SEED` the drop-one and restore audits match their recorded runs, and the base and ladder match the success rates of theirs (1,024 episodes) | screening-grade: 256 MATH-train problems, one sample each; one partial held-out check |

    **Caveats.** Exp 8 screening is single-sample on train problems (SE ≈ 3pp). §2.5's laptop runs use short token
    budgets, at which the edit's effect does not show. Car-ramp entry counts are against the rounded base; the
    waypoint's entry set is read from its checkpoint. A partial held-out check, D4k on MATH-500 × 4 samples, scored
    54.2% (1,084/2,000; `data/exp8/validate_D4k_math500.jsonl`) against 48.1% for the full edit in the published
    evaluation; its paired arms did not finish. Original write-ups: `data/docs/`.
    """)
    return


if __name__ == "__main__":
    app.run()
