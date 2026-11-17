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
except `uv`. The first launch takes about a minute. marimo opens notebooks without running them:
press **Run all** (▷, bottom right) once, about 30 s.

Everything the notebook reads is in this repo (when only the notebook file is present, as on molab, it downloads the repo once), except Qwen3.5-0.8B, which §2.6 downloads from Hugging Face
when you press its button.

Without the expensive runs, every section shows the recorded results: the orx run logs for the car-ramp, the raw
screening outputs for Exp 8, and one bundled example of §2.6's generations. The run buttons re-run them on your
machine next to the recorded numbers.

`weight-edit-minimization.html` is a static export of the notebook after **Run all**, for reading without
Python. It loads marimo's page code from a CDN, so it needs an internet connection.

## What re-runs

| section | what it does | verified against the orx runs |
|---|---|---|
| §1.1 | rebuilds all 8 car-ramp networks from their edit recipes | bit-identical to every shipped checkpoint |
| §1.3–1.6 | the exact `eval_actor.py` harness: one evaluation, the whole ladder, the drop-one audit, the embed-restore ablation | same success counts, mean lengths and returns as the recorded runs (same seeds, same key schedule) |
| §1.7–1.8 | step-by-step mechanism traces; write and evaluate your own edit | — |
| §2.1–2.3 | applies Astra's bf16 edit, recovers its rank-1 closed form, rebuilds and regenerates all campaign candidates | entry counts and specs match the campaign's |
| §2.4 | paired statistics from the 29 raw per-problem screening outputs | reproduces the campaign's results table |
| §2.6 | paired base-vs-candidate Qwen generations on your machine (Apple GPU or CUDA) with a short token budget | — |
| §2.7 | writes any candidate as a checkpoint; `code/eval_vllm.py` runs the full 32k-token screening on a CUDA GPU | — |

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
runs/, checkpoints/           created when you run §2.6 / §2.7
```

Exp 8 numbers are screening-grade: 256 MATH-train problems, one sample each. Differences under
about 5pp are noise.
