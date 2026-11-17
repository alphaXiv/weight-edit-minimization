# Weight-edit minimization — replication notebook

A marimo notebook that re-runs the two minimal-weight-edit campaigns of the
weight-space-programming project:

1. **Car-ramp** (Kinetix `l/h2_car_ramp`): Astra's 6,073-entry edit of a PPO actor, cut down
   to 12 entries, plus a 10-entry fix written from scratch.
2. **Experiment 8** (Qwen3.5-0.8B, math): Astra's 130,188-entry edit of two MLP down-projections,
   and the 27-candidate screening campaign that tested which subsets of it are needed.

## Run it

**In your browser, nothing to install:** [open in molab](https://molab.marimo.io/github/alphaXiv/weight-edit-minimization/blob/main/weight-edit-minimization.py).
molab runs the notebook on a cloud server; fork it to your own molab workspace to edit and keep changes.

**On your machine:**

```sh
git clone https://github.com/alphaXiv/weight-edit-minimization
cd weight-edit-minimization
uvx marimo edit --sandbox weight-edit-minimization.py
```

`--sandbox` installs the dependencies listed at the top of the notebook (Python 3.11, Kinetix 3.0.2,
JAX 0.9.0 CPU, torch, transformers) into an isolated environment, so nothing on your machine is used
except `uv`. The first launch takes about a minute.

## How it works

The notebook is the experiment. marimo opens it without running anything; press **Run all** (▷, bottom right)
and it runs top to bottom, about 5 minutes on a laptop. Each experiment is a short code cell, and the chart under
it fills in as runs finish, next to the result the original orx run recorded. Change a variable and rerun a cell
to try your own version: `EPISODES`, `TRACE_NETWORK`, `my_edit`, `my_spec`, `RUN_QWEN`, `CHECKPOINT`.

| section | what you run | checked against |
|---|---|---|
| §1.2–1.3 | the base actor and Astra's 6,073-entry edit, with animations of each episode | recorded orx runs |
| §1.4 | the ladder: 6,073 → 1,939 → 1,718 → 12 → 11 entries, and 14 → 10 written from scratch | rebuilt networks are bit-identical to the shipped checkpoints |
| §1.5–1.6 | drop each of the 12 entries; undo each embed column's erasures | identical success counts and episode lengths to the recorded runs at 256 episodes |
| §1.7–1.8 | trace the units the edit wires together; write and evaluate your own edit | — |
| §2.1–2.2 | apply Astra's bf16 Qwen edit, recover its rank-1 form, rebuild the 27 campaign candidates | entry counts and specs match the campaign's |
| §2.3–2.4 | design a candidate; analyze the 29 raw screens (paired tests, interactive chart, per-problem grid) | the campaign's results table |
| §2.5 | Qwen generations on your machine, base vs a candidate (`RUN_QWEN = True`; Apple GPU or CUDA) | a bundled example run until you do |
| §2.6 | write a candidate as a checkpoint for `code/eval_vllm.py`, the campaign's 32k-token screen (CUDA) | — |

Everything the notebook reads is in this repo (when only the notebook file is present, as on molab, it downloads
the repo once), except Qwen3.5-0.8B, which §2.5 and §2.6 download from Hugging Face when you turn them on.

`weight-edit-minimization.html` is a static export of a full run, for reading without Python. It loads marimo's
page code from a CDN, so it needs an internet connection.

## Layout

```
weight-edit-minimization.py   the notebook
code/                         the campaigns' own scripts: car-ramp eval harness (reference copy; the notebook inlines it),
                              Exp 8 candidate builder and vLLM screening script, safetensors I/O
data/car-ramp/                base PPO checkpoint, Astra's edit, every ladder network, keep-lists, C2 edits
data/car-ramp-evidence/       recorded evaluation results
data/logs/                    raw RESULT lines from the orx runs
data/exp8/                    base down_proj tensors, Astra's edit (edit.npz), factors, specs, screening problem set
data/screens/                 per-problem screening outputs, one file per candidate
data/docs/                    the original write-ups
figs/                         figure style, and a matplotlibrc that keeps your own matplotlib config out
runs/, checkpoints/           created by §2.5 / §2.6
```

Exp 8 numbers are screening-grade: 256 MATH-train problems, one sample each. Differences under
about 5pp are noise.
