# The Experiment-4 minimal edit, re-run under tree discipline

**Project:** weight-space-programming · **Task:** reproduce and properly track the Experiment-4 (`l/h2_car_ramp`) minimal-edit investigation through the OpenResearch experiment tree · **Date:** 2026-11-12
**Supersedes (in tracking only, not in conclusions):** the scratch run documented in `report.md` §1–5 (same folder) — that analysis is reused here by reference; every number below is re-measured on this machine through `orx exp run --backend local`.

---

## Headline

All previously hand-verified network claims reproduce exactly on this machine, now tracked as nodes in the experiment tree of project `086ca115-4a45-466f-ba94-6d23135135db`:

- base PPO checkpoint: **0 / 1024** (all timeout) — node H_base
- Astra's 6,073-entry edit: **1,024 / 1,024** (sampled, s0+s1) — node H_edited
- 1,939-entry joint_free subset: **1,024 / 1,024** (sampled, s0+s1) — node H_jointfree
- 1,718-entry greedy waypoint: **1,024 / 1,024** (sampled s0+s1, argmax s0) — node H_waypoint_1718
- 12-entry greedy fixed point (Astra's own entries): **1,024 / 1,024 × 4 sampled seeds + argmax** — node H_greedy12
- 10-entry from-scratch design (zero erasures): **1,024 / 1,024 × 4 sampled seeds + argmax** — node H_C2min
- 14-entry from-scratch design (zero erasures): **1,024 / 1,024 × 2 seeds + argmax** — node H_C2

Optional probes all confirm the minimality structure, with one refined claim:

- embed tap columns 6/7/8, restored one at a time, each break success exactly as before (timeout / 32-step red, 100 % red / 69-step red, 100 % red) — node H_embed_restore
- reverting any of three downstream isolation groups (G, J, O) keeps success (256/256 each) — node H_decomp_joint_revert
- drop-one audit of the greedy-12 fixed point (12 nodes × 256 ep, s0, sampled): **10 of 12 drops break success outright; dropping actor_head.b[0] drops success to 232/256 (90.6 %); dropping actor1.W[122,126] leaves 256/256 intact** — node H_greedy12_min. The follow-up probe at full resolution confirms the redundancy: **H_greedy11 (greedy-12 minus `actor1.W[122,126]`) scores 1024/1024 × 4 (s0/s1/s2 sampled + argmax s0, mean len 77.0)** — node H_greedy11. Inside Astra's edit, the minimal sufficient set measured in this pass is therefore at most **11 entries** (the 12-entry greedy fixed point minus one redundant entry); the 10-entry from-scratch floor stands below it.

Nothing here overturns the prior conclusions; the one new, decision-relevant finding is that the greedy-12 fixed point contains one individually redundant entry (`actor1.W[122,126]`), confirmed redundant at both 256- and 1,024-episode resolution (nodes H_greedy12_min and H_greedy11), and the 10-entry from-scratch floor stands.

---

## 1. What was rebuilt, and why the tree

The previous session (2026-11-11/12) produced correct measurements but entirely in `/tmp/opencode/exp4/` — no experiment nodes, no run tracking, no descriptions; nothing was reproducible through the project. This report re-derives the same results through the tree: one baseline vehicle node (`930688ba`, branch `orx/car-ramp-eval-harness-baseline-vehicle`) carrying the harness, reference networks, and the fixed run command; all hypothesis nodes are its **children**, each differing from it in `eval_config.json` only (plus, for the three builder nodes, a small committed build tool); one deeper probe (`H_greedy11`) descends off the greedy-12 winner.

**Harness** (`tools/eval_actor.py` at the baseline branch): loads a weights npz, builds the official Kinetix env — `kinetix-env==3.0.2`, JAX 0.9.0 CPU, level `l/h2_car_ramp`, `symbolic_flat` observation (679-dim), `multi_discrete` actions (16 logits; only entry 0, 3 values, wired to motors in this scene), `frame_skip=2`, `dense_reward_scale=0.2`, `auto_reset=False`, 256 steps — vmapped over N episodes. Outcome indexing uses the **first done step**; green/red success is read from the per-step `info["GoalR"]` dictionary, matching the protocol of the prior measurements. Sampled actions unless the eval entry sets `argmax: true`. Prints `CONFIG` / one `RESULT {…}` JSON line per evaluation / a final `SUMMARY {…}` line. `eval_config.json` selects the evaluations; the run command is fixed at the project level (`/tmp/opencode/exp4/venv/bin/python tools/eval_actor.py`) and inherited by every node. The environment is the pre-existing venv at `/tmp/opencode/exp4/venv` (Python 3.11.2, kinetix-env 3.0.2, jax 0.9.0, numpy 2.4.6); the README on the baseline branch documents re-creation (`pip install kinetix-env==3.0.2 jax==0.9.0 numpy==2.4.6`; network needed at install only). The Kinetix level ships inside the `kinetix-env` package; evaluation needs no data download and no network.

**Reference networks** (copied read-only from `/tmp/opencode/exp4/` into `weights/` at the baseline branch; sha256 in the baseline README): `base_ppo_final_seed0.npz` (flax `params/…` layout), `edited_actor.npz` (the 6,073-entry edit), `greedy_waypoint_1718.npz`, `greedy12_astra.npz`, `c2_14entry.npz`, `c2_min_10entry.npz`. `joint_free.npz` was **rebuilt** by `tools/build_joint_free.py` (rounded base + Astra's edit at the 1,939 keep entries in `config/keep_joint_free.json`) and verified entry-for-entry identical to the prior `/tmp/opencode/exp4/cand_joint_free.npz` (`--check OK`). The embed-restore candidates, the group-revert candidates, the 12 drop-one greedy-12 candidates, and the H_greedy11 candidate were likewise rebuilt from committed sources on their nodes; the rebuilt greedy-12 from rounded base + edit equals the committed `greedy12_astra.npz` exactly (verified by the H_greedy12_min builder).

---

## 2. Tree map and runs

```
root: weight-space-programming (main, empty)
└── 930688ba  Car-ramp eval harness (baseline vehicle)      [root experiment; branch orx/car-ramp-eval-harness-baseline-vehicle]
    ├── 078b2856  H_base        — base 0/1024 (reference)                       run b476233e-03cf-462f-bad7-7ecb14b408da
    ├── c070e60a  H_edited      — Astra 6,073-entry edit 1024/1024              run e548146d-0fe1-469a-aa45-44d85ae1535d
    ├── 2313ea0b  H_jointfree   — 1,939-entry subset 1024/1024                  run eb67cfba-b6d9-408b-afaa-83c718998a79
    ├── e6f2909a  H_waypoint_1718 — 1,718-entry greedy waypoint 1024/1024       run 371a4351-ce2b-4271-b7f1-8cebfc04b99e
    ├── ec34029b  H_greedy12    — 12-entry greedy fixed point 1024/1024         run 8da1240d-96c5-4bd6-aae1-f40329a68929
    │   └── a4478018  H_greedy11 — 11-entry (drop actor1.W[122,126])            run 06029ff1-27d0-4602-960f-f2ab828e8d8f
    ├── 1ca3b457  H_C2          — 14-entry from-scratch 1024/1024               run 31932c35-8a66-4205-8d17-d665c5bb6aba
    ├── a9999043  H_C2min       — 10-entry from-scratch 1024/1024               run 00bfc225-72d5-4495-9dbf-7ec25a23cf7e
    ├── 7e966912  H_embed_restore — restore tap col 6/7/8 breaks (256 ep)       run c3435537-1e6f-4a96-89ea-562f19876158
    ├── d017a0fd  H_decomp_joint_revert — revert G/J/O keeps success            run 3483db0c-d5aa-46b5-808e-dae2036426a6
    └── e56666da  H_greedy12_min — drop-one audit of the 12-entry fixed point   run 4125dd18-24ab-4dbc-a29a-8bf5047aba3f
```

**Shape rationale.** The H\_* nodes are the co-equal options of one decision — "what is the minimal sufficient edit" — so they fan as siblings under the single decision node (the baseline vehicle), not off the project root and not in a chain. Exactly one deeper probe was added (H_greedy11 off H_greedy12), because the audit produced a decision-relevant anomaly; nothing else was padded in. The baseline's own run (`db907741`, 128-ep base smoke, 0/128, mean return 0.1520 — matching the prior base measurement) validated the harness before any hypothesis node consumed CPU.

**Repair count.** Zero failed runs; every node answered on its first launch. All nodes are terminal. Every node carries a durable `orx exp desc` note with its verdict.

**Run environment.** All runs: `--backend local`, this machine (8 cores, CPU-only JAX), one process per node, 1,024-episode evals with 1 JIT compile per process (the argmax protocol recompiles the scan body once). Seven 1,024-episode nodes shared the CPU; wall times per node 16–35 min (each `RESULT` reports its own eval time; `wall_s` includes compile only for the first eval in a process — later evals in the same process report ~0 s dispatch and are materialized via float() before printing).

---

## 3. Measured results (this machine, Kinetix CPU, 1,024 episodes unless noted)

| node | network | protocol | result | prior |
|---|---|---|---|---|
| H_base | base PPO (`ppo_final_seed0.npz`) | 1024 ep, sampled, s0 | **0 / 1024** · all timeout · len 256.0 · return 0.1521 | 0 / 1024 · return 0.152 ✓ |
| H_edited | Astra 6,073-entry edit | 1024 ep, sampled, s0 + s1 | **1024 / 1024** both · len 73.0 · return 1.374 | 1024×2, len 73.0 ✓ |
| H_jointfree | 1,939 keep entries | 1024 ep, sampled, s0 + s1 | **1024 / 1024** both · len 72.0 | 1024×2 (+argmax) ✓ |
| H_waypoint_1718 | 1,718 greedy waypoint | 1024 ep, s0 + s1 + argmax s0 | **1024 / 1024** × 3 · len 70.0 | 1024×3 ✓ |
| H_greedy12 | 12 entries (Astra's own) | 1024 ep, s0–s3 + argmax s0 | **1024 / 1024** × 5 · len 71.0 | 1024×5 ✓ |
| H_greedy11 | 11 entries (drop `actor1.W[122,126]`) | 1024 ep, s0–s2 + argmax s0 | **1024 / 1024** × 4 · len 77.0 | new probe off H_greedy12 |
| H_C2 | 14 scratch entries | 1024 ep, s0 + s1 + argmax s0 | **1024 / 1024** × 3 · len 73.0 | 1024×3 ✓ |
| H_C2min | 10 scratch entries | 1024 ep, s0–s3 + argmax s0 | **1024 / 1024** × 5 · len 71.0 | 1024×5 ✓ |

### Ablation / necessity probes (256 episodes, s0, sampled)

| node | candidate (built from committed sources; verified equal to prior `/tmp` equivalents where they exist) | result | prior |
|---|---|---|---|
| H_embed_restore | joint_free with group-A erase of tap col 6 reverted (kept 1,307) | **0 / 256** · timeout · len 256 | 0 / 256 timeout ✓ |
| H_embed_restore | same for col 7 (kept 1,317) | **0 / 256** · 100 % red · len 32 | 0 / 256, red, 32 steps ✓ |
| H_embed_restore | same for col 8 (kept 1,309) | **0 / 256** · 100 % red · len 69 | 0 / 256, red, 69 steps ✓ |
| H_decomp_joint_revert | full edit minus group G (actor0.W cols 122:128, 749) | **256 / 256** · len 72 | 256 / 256 ✓ |
| H_decomp_joint_revert | full edit minus group J (actor1.W rows 122:128, 758) | **256 / 256** · len 73 | 256 / 256 ✓ |
| H_decomp_joint_revert | full edit minus group O (actor2.W cols 126:128, 248) | **256 / 256** · len 73 | 256 / 256 ✓ |
| H_greedy12_min | greedy-12 minus `embed.W[234,8]` (green tap) | **0 / 256** · red · len 94 | breaks ✓ |
| H_greedy12_min | minus `actor0.W[8,127]` (gain-80 green comparator) | **0 / 256** · red · len 44 | breaks ✓ |
| H_greedy12_min | minus `actor1.W[122,126]` (−10 into BACK gate) | **256 / 256** · len 77 | fixed-point claim refined — see §6 |
| H_greedy12_min | minus `actor1.W[127,124]` (release relay) | **0 / 256** · red · len 160 | breaks ✓ |
| H_greedy12_min | minus `actor1.b[126]` (−10 gate bias) | **0 / 256** · red · len 69 | breaks ✓ |
| H_greedy12_min | minus `actor2.W[124,126]` (+20 wiring) | **0 / 256** · red · len 32 | breaks ✓ |
| H_greedy12_min | minus `actor2.W[126,126]` (+10 wiring) | **0 / 256** · timeout · len 256 | breaks ✓ |
| H_greedy12_min | minus `actor2.b[126]` (−10 gate bias) | **0 / 256** · timeout · len 256 | breaks ✓ |
| H_greedy12_min | minus `actor3.W[126,126]` (relay) | **0 / 256** · red · len 33 | breaks ✓ |
| H_greedy12_min | minus `actor4.W[126,126]` (relay) | **0 / 256** · timeout · len 256 | breaks ✓ |
| H_greedy12_min | minus `actor_head.W[126,2]` (+200 BACK vote) | **0 / 256** · 9.4 % red, 90.6 % timeout · len 236 | breaks ✓ |
| H_greedy12_min | minus `actor_head.b[0]` (−200 bias) | **232 / 256** (90.6 %) · len 80 | breaks, but only partially — see §6 |

---

## 4. The decomposition, re-derived

Full analysis of *why* the edit works is in `report.md` (same folder) and is not reproduced here; the load-bearing facts re-derived and re-verified in this pass:

- **Rounding.** Astra was shown weights rounded to 3 significant digits per tensor (`scale = 10^(floor(log10(max|W|)) − 2)`); all candidate construction and all claims here are vs the rounded base. Replaying Astra's 51 edit statements on the rounded base reproduces `edited_actor.npz` entry-for-entry and reproduces the published change count **6,073 exactly** (embed.W 1887, embed.b 1, actor0.W 1132/.b 6, actor1.W 1255/.b 4, actor2.W 760/.b 2, actor3.W 507/.b 2, actor4.W 506/.b 2, head.W 6/.b 3) — this replay is embedded in the committed builder tools (`tools/build_joint_free.py`, `tools/build_restore_cols.py`, `tools/build_group_revert.py`), which rebuild the candidates from the base checkpoint and the keep-masks alone.
- **55 functional + 6,018 isolation erases.** The functional entries (comparator thresholds, ±10–80 wiring, saturating relays, ±100–200 head votes and biases) carry the mechanism; the erases wash out repurposed units. Downstream isolation (F, G, J, K, N, O, R, S, V, W — 4,134 entries) is behaviourally free on this task (H_jointfree = 1,024/1,024; spot-checks H_decomp_joint_revert = 256/256 × 3 groups). Embed-stage erasure (group A, the 1,884 inputs of the three repurposed embed units 6/7/8) is load-bearing (H_embed_restore: each restore breaks success with the same failure signature as the prior teardown — col 6 timeout, col 7 100 % red at step 32, col 8 100 % red at step 69).
- **The ladder.** 6,073 (Astra) → 1,939 (joint_free: 55 functional + 1,884 embed) → 1,718 (waypoint) → 12 (greedy fixed point, Astra's entries) → **11 (drop the redundant `actor1.W[122,126]`; H_greedy11)** → 14 (from-scratch) → **10 (from-scratch floor)** — each rung re-measured green at this machine's protocol; see table in §3.
- **Mechanism in one line per candidate class.** Astra's fix: hand-written comparators on (x, vx, gx) → gates → saturating relays → head override (|Δlogits| ≥ 200) with a self-cancelling idle state; the from-scratch designs instead exploit PPO's own velocity detector (embed unit 116) and dead ReLUs (embed unit 26 lifted by a +15.2 bias) implementing "back up to stall on the wall, then launch" with no x-comparator at all; greedy-12 reuses PPO's rest/motion staging signal (actor0 u122) wrapped in one gate. Details: `report.md` §3, §5, §10.
- **Why the base fails (unchanged).** p(reverse) ≈ 1e-10–1e-12 at the start state; the throttle/coast controller stalls at x ≈ 2.0 for the whole episode; 0/1024 across the training record and here (H_base).

---

## 5. How the minimal set was derived (short)

1. **Verify the published edit first** (H_edited) — the fix works, 1,024/1,024, as shipped.
2. **Attribute every changed entry to the edit statement that wrote it** (exact replay) and classify: functional (55) vs isolation erases (6,018).
3. **Revert, don't add**: restore the 4,134 downstream erase entries to base — free (H_jointfree, H_decomp_joint_revert). The embed-stage must keep its erasures (H_embed_restore).
4. **Greedy-eliminate** the 1,939-keep set with an environment gate (success in short rollouts) to its fixed point: the 12 entries listed in `report.md` §10 — re-verified exhaustively at 1,024 ep × 5 protocol points (H_greedy12).
5. **Go from scratch** (no erasures at all): 14 entries suffice; greedy trimming finds **10** — re-verified × 5 (H_C2, H_C2min).
6. **Audit the fixed point** for necessity one entry at a time (H_greedy12_min): 10 of 12 are individually load-bearing; `actor_head.b[0]` is necessary for completeness (232/256 without); `actor1.W[122,126]` is redundant at 256-episode resolution — tested at full resolution in H_greedy11.

---

## 6. H_greedy11 — the one deeper probe

`H_greedy11` (child of H_greedy12; run `06029ff1-27d0-4602-960f-f2ab828e8d8f`; `weights/greedy11_drop02.npz`, byte-identical to the drop-02 candidate audited on H_greedy12_min: sha256 `e8759efd47365dd9a644547207355b1b1f514cc96b49c1cd2cea70ec9bd7631b`): 1024 episodes, seeds 0/1/2 sampled + argmax s0 — all **1024 / 1024**, mean len 77.0.

**What this means.** The greedy-12 fixed point — the smallest set of Astra's own edit that the prior elimination kept — contains one individually redundant entry: `actor1.W[122,126]` (−10, the u122 → BACK-gate wiring) can be removed without effect at 256-episode resolution (audit) and at 1,024-episode resolution × 4 protocol points (this node). So the minimal sufficient set inside Astra's edit, measured here, is **11 entries**:

`embed.W[234,8]`, `actor0.W[8,127]`, `actor1.W[127,124]`, `actor1.b[126]`, `actor2.W[124,126]`, `actor2.W[126,126]`, `actor2.b[126]`, `actor3.W[126,126]`, `actor4.W[126,126]`, `actor_head.W[126,2]`, `actor_head.b[0]`

i.e. the 12-entry list of `report.md` §10 minus `actor1.W[122,126]`. The remaining 11 are: 10 individually load-bearing (every drop breaks success outright) plus `actor_head.b[0]`, without which success degrades to 232/256 (90.6 %) — necessary for reliability, not for a single episode. Whether any subset smaller than 11 exists inside the edit was not tested and would be new work (the greedy screen's own granularity is implicated); the from-scratch floor (10 entries, H_C2min) remains the absolute minimum validated in this investigation.

**Relation to the 12-entry claim.** Nothing here contradicts the prior verification of the 12-entry candidate (1024/1024 × 5, re-measured) — it refines the *necessity* side of the ladder and removes one entry from the Astra-mechanism minimum. The deletion is behaviorally neutral by construction (drop-02 = the audited network).

---

## 7. Provenance, caveats, and protocol notes

- **Provenance.** The Kinetix rollout here reuses the exact env configuration of the prior Experiment-4 protocol (env size `l`, `symbolic_flat`, `MULTI_DISCRETE`, `frame_skip=2`, `dense_reward_scale=0.2`, `auto_reset=False`, 256 steps, first-done-step indexing, `info["GoalR"]`), and JAX is CPU-only on this machine. The base checkpoint is the public `ppo_final_seed0.npz` (PPO, 1.001 B steps, seed 0).
- **`wall_s`** in each `RESULT` line measures the first eval in a process including JIT compile; subsequent evals in the same process report dispatch time (~0 s) — the printed metrics are always fully materialized.
- **Sampled-action ties across seeds.** For the fix networks, s0 and s1 metrics are bit-identical (logit margins ≥ 200 make sampling effectively deterministic); this is expected, not a bug — joint_free's `mean_return` differs in the 5th decimal because one or two episodes differ.
- **Resolution dependence.** Necessity statements are protocol-dependent: "individually required" is claimed at 256 episodes, s0, sampled (and drop-11 shows partial rather than total breakage — 232/256). The greedy elimination that produced the 12-entry fixed point used a coarser screen; the environment at this resolution shows its granularity. Nothing here weakens the 1,024 × 5 verification of the candidate itself.
- **No new hypotheses** were introduced beyond the minimality question asked; the tree contains no speculative nodes. Publication is disabled; nothing was pushed anywhere. `/root/OpenResearch/weight-space-programming` was never touched; all writes happened in this session's worktree and were committed to the `orx/*` branches. The prior `report.md` and `evidence/` were read, not modified.
- **Artifacts of this pass** are this file plus the tree itself; the reference networks, builders, and per-node descriptions live on the branches (`git log orx/car-ramp-eval-harness-baseline-vehicle..<branch>`).

---

## 8. Reproduce

```sh
# from a checkout of this project (the orx branches hold the harness + networks):
git fetch .   # local clone already has orx/* branches
git checkout orx/car-ramp-eval-harness-baseline-vehicle   # or any H_* node branch
/tmp/opencode/exp4/venv/bin/python tools/eval_actor.py
```

Per node: `orx project view 086ca115-4a45-466f-ba94-6d23135135db` or `orx exp status <expId>` shows the branch; `git diff orx/car-ramp-eval-harness-baseline-vehicle...orx/<node>` shows exactly what that node changed (all hypothesis nodes: only `eval_config.json`; builder nodes add their build tool + generated npz weights).

Per node: `orx project view 086ca115-4a45-466f-ba94-6d23135135db`, pick the node, check out its branch, run the fixed command (`orx` extracted snapshot is what actually ran). Every node's `eval_config.json` states its exact protocol; `orx logs <runId>` holds the raw `RESULT`/`SUMMARY` lines. `tools/build_joint_free.py --check <ref.npz>` re-verifies the joint_free construction against a reference; the restore/revert/drop builders re-derive their candidates from the base checkpoint + keep-masks.
