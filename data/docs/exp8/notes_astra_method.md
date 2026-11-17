# Astra's Qwen3.5-0.8B weight edit (Experiment 8, run p50c) — method & reconstruction notes

Compiled 2026-11-12 from the public logs repo `RikiyaT/lm_understand-logs` and from local tensors in
`/tmp/opencode/wsp_exp8/`. All heavy file reads were CPU-only; no GPU was touched.

Repo base for p50c: `https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/`
(raw: `https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/`)

---

## 0. TL;DR

- The edit is, to bfloat16 rounding, **exactly** `W = W0 − 6 · outer(u, mask64 · (u @ W0))` for
  `model.language_model.layers.{15,19}.mlp.down_proj.weight`, where `u` is a unit repetition
  direction and `mask64` zeroes all but 64 of the 3584 columns. I re-verified this bit-level
  against the public base model (see §2.2).
- The 64 columns were selected **on the original model's fwd_4** (8 hand-written
  productive/repetitive texts), by a *signed, activation-weighted* score
  `score_j = (u @ W0)_j · Δact_j`, `Δact_j = mean over 4 matched productive/repetitive pairs of
  (mean act[repetitive] − mean act[productive])` over the last 24 tokens of each text — i.e., an
  **activation-contrast-weighted directional-writer coefficient**, *not* the top-|c| coefficient,
  *not* a pure activation criterion, *not* dose-response, *not* correctness.
- Dose-response experiments only motivated the strength (`alpha = 6`), never the subset.

---

## 1. How Astra selected the 64 neurons per layer

### 1.1 The decisive code (pass 17 / explanations turn 68–69)

In `tool_calls.jsonl` (entry with the hypothesis "the repetition effect has a dose-response and may
be concentrated in a small, transferable set of MLP writers"; it ran right after the alpha1
transfer probe, pass 17):

https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl

```python
directions=torch.load("/work/repetition_directions.pt",weights_only=True)
neuron_subsets={}
for layer,u0 in directions.items():
    u=u0.clone()
    h=np.load(f"/work/fwd_24/hidden_{layer+1}.npy",mmap_mode="r")
    tails=np.stack([np.asarray(h[i,n-24:n],dtype=np.float32).mean(0)
                    for i,n in enumerate(json.loads(Path("/work/fwd_24/meta.json").read_text())["lengths"][:12])])
    if float(torch.tensor((tails[1::2]-tails[::2]).mean(0))@u)<0: u=-u          # sign-align u to the transfer contrast
    w=load_original(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
    proj=u@w                                                                   # per-column writer coefficient c_j
    scores=[]
    for folder,count in (("fwd_4",8),("fwd_24",12)):
        meta=json.loads(Path("/work",folder,"meta.json").read_text())
        a=np.load(Path("/work",folder,f"mlp_act_{layer}.npy"),mmap_mode="r")
        means=np.stack([np.asarray(a[i,n-24:n],dtype=np.float32).mean(0)
                        for i,n in enumerate(meta["lengths"][:count])])
        scores.append(proj*torch.tensor((means[1::2]-means[::2]).mean(0)))      # c_j * Δact_j
    score=scores[0]                              # <-- fwd_4 score is the one used for selection
    top=torch.argsort(score,descending=True)
    neuron_subsets[layer]=top[:64].clone()      # top-64 by SIGNED score, descending
```

Its printed output (same entry):

> Writer layer 15 top12 [294, 371, 0, 46, 1130, 266, 63, 601, 1155, 1060, 516, 25] positive mass
> top8/32/64 [0.193, 0.389, 0.498] transfer rank correlation 0.666 top64 overlap 55
>
> Writer layer 19 top12 [11, 1, 161, 343, 171, 246, 5, 17, 119, 13, 127, 744] positive mass
> top8/32/64 [0.397, 0.498, 0.555] transfer rank correlation 0.583 top64 overlap 36

Key properties of the criterion:

- **Score = c_j × Δact_j** (elementwise), where `c = u @ W0` (the coefficient of `u` in `down_proj`'s
  columns — i.e., how strongly column `j` writes along the repetition direction) and
  `Δact = mean over matched productive/repetitive pairs of (mean over the last 24 tokens of MLP
  activation of the repetitive member − the productive member)`.
  Note `means[1::2] − means[::2]`: even-indexed texts (0,2,4,6) are the productive ones, odd-indexed
  (1,3,5,7) the repetitive ones (see §1.3).
- **Signed, descending argsort, top 64** — it keeps the 64 columns with the largest *positive*
  product `c_j·Δact_j`, i.e., columns that both (a) write along `+u` and (b) are *more active in the
  repetitive branch than the productive branch*. Columns with negative or tiny products are excluded
  even if |c| is large.
- **fwd_4 only.** `scores[1]` (fwd_24, the 12-text held-out transfer probe) was used exclusively to
  check transfer (rank correlation 0.666/0.583, top-64 overlap 55/36). `repetition_writer_top64.pt`
  (saved in that same run) stores the fwd_4 subsets. Frozen notes: “Top64 writer subsets derived
  ONLY from fwd4 … Selected64 writers per layer chosen on original fwd4, not held-out correctness.”
  (`work_notes/notes.md`, “frozen repetition repair” section; same text in `work_notes/final_report.md`).
  URLs: [notes.md](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/notes.md),
  [final_report.md](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/final_report.md)
- **u sign:** `repetition_directions.pt` stored `u` up to sign (its sign was fixed by
  `u = d[:, argmax_j |col_j|]` of the edit delta, see below); in the selection code `u` was
  sign-flipped again so that `u · (transfer contrast at residual outputs 16/20) ≥ 0` on fwd_24.
  Sign of `u` does not matter for the final edit (see §2), but it *does* matter for ranking by
  signed score — Astra aligned it to the repetitive>productive contrast before scoring.
- It is **not** the top-64 by |c_j|. Direct verification (§2.2, item 7): in units of `6·|c|` (i.e.,
  `|−6·c_j|`, the per-neuron change scale), outside max = 0.405/0.450 > inside min = 0.046/0.022
  (layer 15/19) — matching the previously established fact. In raw units `|c|`:
  selected min/max = 0.0077/0.0883 (L15) and 0.0037/0.1068 (L19); outside max = 0.0674 / 0.0750.
- It is **not** a pure activation criterion either (activation means alone would ignore c_j's sign
  and magnitude along u), and **not** a dose-response criterion (dose-response was used only to pick
  `alpha`; see §2.3), and **not** related to correctness or benchmark performance (“chosen on
  original fwd4, not held-out correctness”).

### 1.2 Why these particular 64: attribution evidence

Very early in the run (pass 2; before Astra's explanation requirement kicked in — explanations.md
only covers the run from turn 28 on), Astra attributed growth of the repetitive contrast across
layers and ranked “MLP writers” by `avg = mean over the 4 pairs of (Δact_j · (d @ down_proj_j))`
with 36-token tail means. Top scorers included layer 19 neuron 11 (0.12001), layer 15 neurons
294 (0.02533), 371 (0.01421), 0 (0.01245) — all present in the final 64. That analysis localized
the writers and named the first single-neuron ablation probe; the final 64-set came only later
from the pass-17 score above.

Tool call (index 9, “Top MLP writers … Probe: layer 19, MLP neuron 11 output ablated”):
https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl

### 1.3 The eight fwd_4 texts (selection contexts) — verbatim structure

From the same file, the `fwd_4` request (k = 4, `type: teacher_forced`, `model: original`, raw text,
**no chat markup except `<|im_start|>user … <|im_start|>assistant\n<think>\n`**; i.e., NOT the chat
template the harness used for chats — Astra found the chat template injected an empty think block
and switched to the raw template):

| idx | role | problem (user) | assistant starts with |
|---|---|---|---|
| 0 | productive | Rafael works 10 hours Monday, 8 Tuesday, 20 left, $20/h | “The first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours …” |
| 1 | repetitive | same prompt | same opening, then “But is 20 the total? No, it is the remaining time.” ×3 |
| 2 | productive | Find the remainder when 6x^4−14x^3−4x^2+2x−26 is divided by 2x−6 | “Set 2x-6=0, giving x=3. The remainder is P(3)=486−378−36+6−26=52 …” |
| 3 | repetitive | same prompt | opening identical, then “But P(x) has -26. The constant term is -26.” ×3 |
| 4 | productive | The sequence is 88,85,82,… How many terms precede -17? | “The common difference is -3. Solve 88-3(n-1)=-17. Then …” |
| 5 | repetitive | same prompt | opening identical, then “Now 88-85=3 and 85-82=3.” ×3 |
| 6 | productive | A machine makes 45 necklaces, another 2.4× as many. Find the total. | “The second machine makes 45*2.4=108 necklaces. Add the first machine's 45 …” |
| 7 | repetitive | same prompt | opening identical, then “Is 108 the total? No, 45+108=153.” ×3 |

Four matched productive/repetitive pairs, each pair sharing the exact same correct opening. Lengths
124/123/131/136/107/146/98/135 tokens. Note: fwd_4 requested `hidden_<l>` and `mlp_act_<l>` for all
layers (`return` included hidden states and MLP activations), `model: original`.
This is the data behind “Directions from FOUR matched productive/repetitive contexts in original
fwd4 at residual outputs16,20” (`work_notes/final_report.md`).

### 1.4 Where the repetition direction u came from (same fwd_4)

Pass 2 (tool_calls index 9):
- `meansH[l]` = mean over last **36** tokens of `hidden_l`, per text.
- `diffH[l] = (meansH[1::2] − meansH[::2]).mean(0)` — the productive→repetitive
  residual-stream contrast at block output `l`.
- For `mlp_direction_edit(layers)`: `d = normalize(diffH[layer+1])` (so residual **outputs 16 and 20**
  for layers 15/19), `W' = W − alpha · outer(d, d @ W)`. The first edit ever applied by Astra was
  this rank-1 direction edit with `alpha = 1`.
- Later (`repetition_directions.pt`, saved in tool_calls index 36/37), u was re-derived as
  `u = (load(name) − load_original(name))`, `u ← u[:, argmax |col|]`, `u ← u / |u|`. Since the
  rank-1 delta columns are all ±parallel to `d`, this recovers `u = ±d` (unit, sign arbitrary). The
  two u's have norm 1 and `cos(u15, u19) = 0.849` (recomputed, §2.2 item 6).
- The direction is the same contrast the selection code used, just at 24- vs 36-token windows for
  the tail.

### 1.5 Decisive quotes with URLs

- [explanations.md, Turn 68 (python; passes used 17/50)](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/explanations.md):
  “I'll also check whether a small set of MLP neurons accounts for most of the repetition-directed
  output, which could suggest a more selective repair.”
- Turn 69: “The neuron analysis also transfers: 55 of layer 15's top 64 writers and 36 of layer 19's
  recur in the new contexts. I'll now test strength two…”
- Turn 98 (python; passes used 25/50): “…a more localized version that changes only the 64
  previously identified repetition-writing neurons in each MLP, rather than every column's
  contribution along the direction. Those neurons were selected from the original diagnostic
  contexts, not from these transfer results.”
- Turn 99 (forward; passes used 25/50): “The selected 64 neurons account for about 91% and 98% of
  the measured repetitive-versus-productive writing contrast in layers 15 and 19, respectively,
  despite retaining only 50% and 40% of the directional weight norm.”
- Turn 101 (forward; passes used 26/50): “The restricted edit … repeated-continuation likelihood
  falls by 0.065 nats/token, while productive reasoning changes by only −0.005.”
- Turn 105 (python; passes used 28/50): increasing only the selected columns to strength 6.
- `tool_calls.jsonl` indices 61 (selection), 90 (“TOP64 alpha4” masked edit, idx list
  `writer_subsets[l]`, `masked = zeros_like(writer); masked[ix] = writer[ix];
  delta = −4·outer(u, masked_writer)`), 97 (`set_subset_edit(6)` → final alpha = 6), 111 (audit),
  127 ( frozen notes): https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl
- `work_notes/frozen_weights_manifest.json` (identical to the local copy; sha256 9237d35f…/3091b77b…,
  `selected_neurons` 64 per layer): https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json
- `work_notes/frozen_edit_audit.json` (`formula_relative_error` 3.2e-08 / 4.3e-08,
  `off_direction_fraction` 1.2e-07 / 1.6e-07, `changed_columns` 64): https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/frozen_edit_audit.json
- Run README table (“subtracts the loop direction from the 64 output columns per layer of the MLPs
  of layers 15 and 19 that write it most”): https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md

---

## 2. The formula actually applied (and its history)

### 2.1 Final formula — confirmed

For each of the two tensors (`model.language_model.layers.15.mlp.down_proj.weight`,
`…19.mlp.down_proj.weight`, both bf16, shape [1024, 3584]):

```
c     = u @ W0                      # per-column writer coefficient, u unit (1024,), W0 (1024,3584)
mask  = zeros(3584); mask[selected_64] = 1
W     = W0 − 6 * outer(u, mask * c)
```

i.e. exactly `W = W0 − 6*outer(u, mask64*(u@W0))` as stated in the blog/README/final_answer.md.

- **Sign-free:** `outer(−u, mask·((−u)@W0)) = outer(u, mask·(u@W0))`, so u's sign convention is
  irrelevant.
- **Per-neuron alpha is not used.** Every selected column gets the same implicit `alpha = 6`;
  unselected columns are exactly unchanged. There is no per-neuron scaling, no clipping, no
  normalization of `c`, no added noise, no bias change.
- **Only those two tensors.** No embeddings, output head, final norm, biases, or any other tensor
  was touched (`frozen_edit_audit.json`, `final_choice.json`).
- **Rounding:** the submission was float32; the evaluation cast to bf16. After bf16 rounding,
  130,188 of 131,072 touched entries differ (65,260 / 64,928). Local `manifest.json`
  (identical to the released edit): `form: sparse`, `changed_columns: 64` per tensor,
  `changed_entries_total: 130188`.

### 2.2 Verification performed locally (this session)

Base model read from the local HF cache snapshot
`/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors`
(revision matches the one named in `code/apply_edit.py`). Steps (numpy only; `st_numpy.py` safetensors
reader):

1. Rebuilt the edited bf16 tensors from `edit.npz` (sparse form) and confirmed that the columns
   whose max-abs delta is nonzero are **exactly** `frozen_weights_manifest.json`'s
   `selected_neurons` (64 per layer, set equality).
2. `delta = edited − W0` is rank-1 to numerical noise: top singular values
   `1.7489e0, 9.87e-04, …` (L15) and `1.3988e0, 9.01e-04, …` (L19), and the left singular vector is
   the SVD u (|u| = 1.7489 / 1.3988 — i.e., factors.json stores `U·σ`), cos = 1.0000000 vs
   normalized factors u.
3. `recon = W0 − 6·outer(u_hat, masked(u_hat @ W0))` (u_hat unit, sign-free) vs rebuilt edited:
   max abs diff 2.7e-04 / 2.6e-04, relative Frobenius 2.3e-03 / 2.9e-03 (pure bf16 rounding:
   bf16 eps ≈ 7.8e-03). **Bit-exact bf16 entries: 3,664,092 / 3,670,016 (L15) and 3,663,854 /
   3,670,016 (L19)**; all differing entries by exactly 1 ulp (Astra's float32 submission cast to
   bf16 vs a float64 reconstruction).
4. Frozen audit (Astra's own, float32): formula_relative_error **3.2e-08 / 4.3e-08**,
   off_direction_fraction 1.2e-07 / 1.6e-07 — i.e., the delta is 99.99999% parallel to
   `outer(u, masked(u@W0))` and the non-selected columns are exactly zero.
5. The local `delta_0.npy` / `delta_1.npy` equal the bf16 reconstructed deltas
   (`allclose`, atol 1e-3) — consistent.
6. `cos(u15,u19) = 0.8490` (matches the established fact); `cos(v15,v19) = −0.0275` (near-orthogonal
   right factors — the two edits are nearly non-overlapping row spaces but share ~72% of their
   direction cosine).
7. `|c|` analysis confirming task's established fact: with `c = u_hat @ W0` (u unit), in units of
   `6|c|`: L15 inside [0.046, 0.530], outside max 0.405; L19 inside [0.022, 0.641], outside max
   0.450. (In raw units: L15 inside min 0.0077 vs outside max 0.0674; L19 inside min 0.0037 vs
   outside max 0.0750.) → selection ≠ top-|c|, decisively.
8. Local `frozen_weights_manifest.json` is byte-identical to the repo's
   `runs/p50c/work_notes/frozen_weights_manifest.json`.

Conclusion: the released edit is **exactly** `W = W0 − 6*outer(u, mask64*(u@W0))` with unit u
(up to sign), the manifest's 64 column indices, and α = 6 for both layers. Nothing else.

### 2.3 History of the formula in the run (why α=6, why mask64)

- Pass 2 (python 9): rank-1 `W' = W − alpha·outer(d, d@W)`, `d` = normalized fwd_4 layer-output
  contrast, layers 15/19, α = 1. First applied after a single-neuron probe (L19 neuron 11, ablation,
  +6.25pp SE4.35 on 32 problems, exploratory) and a L19-only rank-1 test.
- Pass 10–12: α = 1 → independent +24.22pp (pass10/gen13); α = 2 → +24.61pp (pass12/gen16, same
  problem set, weak pairing) and independent +34.77pp SE3.53 (pass19/gen27, validation_2). This
  made full-direction α = 2 the “BEST INDEPENDENTLY CONFIRMED” candidate.
- Pass 20–23: α = 3 exploratory (+39.84pp SE3.82, pass20/gen29, same-set only); α = 4 generation
  failed to initialize repeatedly; a norm-matched orthogonal random control (pass23/gen35) gained
  only +1.56pp SE3.52 → random control REMOVED; head-ablation alternatives (L15h7, L19h5 induction
  heads) were “less selective” → REMOVED.
- Pass 25–26 (python 90/92, fwd 37/38/39): L15-only α4 (−0.0813 rep, −0.0006 prod) and L19-only α4
  (−0.0345 rep, +0.0035 prod) both supported; constructed the **64-neuron masked edit**
  `W' = W − 4·outer(u, mask64(u@W0))`.
  Transfer probe (fwd39): repetitive branch −0.0645 nats/token, productive −0.0047 nats/token
  (vs full α4: −0.1037 / +0.0045; vs full α1: −0.0241 / +0.0023). Selected columns carried
  ~91% (L15) / ~98% (L19) of the repetitive−productive writing contrast while holding only
  ~50% / ~40% of the directional weight norm (`masked_writer.norm()/writer_weights.norm()`
  = 0.502 / 0.400).
- Pass 27 (gen41, 16k tokens): restricted α4 on 128 problems → 13→47 correct, +26.56pp SE4.45, vs
  full α4 (gen42, same cap/set) 14→48, +26.56pp SE4.59 — statistically indistinguishable, restricted
  edit changed only ~half the weights. Exploratory.
- Pass 27/gen42 (python 97): `set_subset_edit(6)` — same mask, α raised 4 → **6**. Saved
  `candidate_top64_a6`. Motivation: exploratory dose escalation of the restricted edit.
- Pass 29 (gen45, FULL 32,768 tokens): top64 α6 exploratory 7/64 → 35/64 (+43.75pp SE6.63).
- Pass 30+: frozen; `frozen_edit_audit.json`; confirmation on validation_3 (+50.78pp SE4.44),
  validation_4 (+48.05pp SE3.18), and five locked final batches of 256 each — pooled over 1,152
  disjoint held-out practice problems: 10.6% → 60.4% (+49.83pp SE1.50), 579 rescues, 5 regressions
  (`work_notes/final_report.md`, `final_answer.md`). These are practice problems, not benchmarks
  (benchmarks: MATH-500 7.9→48.1, GSM8K 13.8→67.1; see `eval/p50c/merged.json` in the repo).
- **The dose-response evidence** (full-direction α1/2/4: rep −0.0241/−0.0487/−0.1037 nats/token,
  productive +0.0023/+0.0034/+0.0045; top64 α6 transfer fwd44: rep −0.10311, productive −0.00903;
  broader productive controls fwd48/49: −0.00893; numeric tokens −0.00704 (429 tokens)) is what
  licenses α = 6 for the sparse edit. It played no role in *which* 64 columns were chosen.

So per neuron j: `ΔW[:, j] = −6 · u · c_j` for j in the 64, `0` otherwise; `c_j = (u @ W0)[:, j]`.
Row direction u is identical across all selected columns of a tensor.

### 2.4 The two u directions (what “repetition direction” is)

- Layer 15: u₁₅ = normalize(fwd_4 contrast at residual **output 16**) — i.e., blocks 15’s output.
  norm stored un-normalized in factors.json as U·σ (|U·σ| = 1.7489 = σ₁, the largest
  singular value of Δ).
- Layer 19: u₁₉ = normalize(fwd_4 contrast at residual **output 20**) — blocks 19’s output.
  σ₁ = 1.3988.
- `cos(u15, u19) = 0.849`. Both u's point (up to sign) along “the direction whose residual-stream
  magnitude grows specifically in the repetitive branch”. They are not principal components of the
  weights; they are empirical contrasts of hidden states on 8 hand-built texts.

---

## 3. Extra verifications requested/done here (local, CPU)

See §2.2. Raw artifacts: `/tmp/opencode/wsp_exp8/{delta_0.npy,delta_1.npy,factors.json,
factors_layer15.npz,factors_layer19.npz,frozen_weights_manifest.json,manifest.json,edit.npz}`.
The HF base snapshot is present at `/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/
snapshots/2fc06364715b967f1860aea9cf38778875588b17/`.

One honest caveat: the *selection* re-derivation (score_j = c_j·Δact_j on fwd_4 activations) cannot
be bit-reproduced here because Astra's fwd_4 `mlp_act_{15,19}.npy` files were not kept (“the large
generated texts and activations it worked on are not kept” — repo README). The selection code and
its printed output are, however, fully preserved in `tool_calls.jsonl`, and the resulting 64-set is
exactly the one in the manifest and in the released edit. Both local checks agree.

---

## 4. Cheap teacher-forced probe to rank candidate edits (for the GPU session)

Goal: given a handful of *candidate* edits (e.g., different column subsets, different α), rank
them by their expected repetition-suppressing benefit vs collateral damage **without any full
generation** — mirroring Astra's own fwd24/fwd25/…/fwd44 transfer methodology (which he used as a
pre-filter before every generation pass; each such probe cost one forward pass of ≤1024 texts).

### 4.1 The probe (14 fixed texts, raw thinking template)

Reuse Astra's 6 six-case transfer set verbatim (public:
[work_notes/transfer_probe_request.json](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/transfer_probe_request.json),
bounds [transfer_probe_bounds.json](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/transfer_probe_bounds.json),
labels [transfer_probe_labels.json](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/transfer_probe_labels.json)).

Template per text (raw, no chat wrapper — the chat template inserts an empty think block and
destroyed the alignment in Astra's hands):

```
<|im_start|>user
{problem}
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
<think>
{common_prefix}{continuation}
```

The 6 problems and their (common prefix, productive continuation, repetitive continuation) — the
repetitive continuation is the “But …?” self-check sentence ×4:

1. “Find the area of a right triangle with legs 6 and 8.”
   common: “The two perpendicular legs are 6 and 8. The hypotenuse is sqrt(36+64)=10. “
   productive: “The area uses the perpendicular legs, not the hypotenuse. Thus A=6*8/2=24. Checking with a rectangle of sides 6 and 8, its area is 48 and the diagonal divides it into two equal triangles.”
   repetitive: “But is the hypotenuse 10? Yes, sqrt(36+64)=10. “ ×4
2. “A bag has 3 red and 2 blue balls. Two balls are drawn without replacement. What is the probability both are red?”
   common: “There are 5 balls. The probability the first is red is 3/5. “
   productive: “After a red ball is removed, 2 red balls remain out of 4. The probability is (3/5)*(2/4)=3/10. Equivalently, there are 3 red pairs among 10 possible pairs, giving the same result.”
   repetitive: “But is there replacement? No, the problem says without replacement. “ ×4
3. “Find the sum of the first 20 positive odd integers.”
   common: “The first term is 1 and the twentieth term is 39. There are 20 terms. “
   productive: “Pair the first and last terms: 1+39=40. Each pair has sum 40, and there are 10 pairs. The total is 400. This also follows from the identity that the first n positive odd integers sum to n squared.”
   repetitive: “But are there 20 terms? Yes, the problem asks for the first 20 odd integers. “ ×4
4. “Find the remainder when 7^20 is divided by 5.”
   common: “Since 7 has remainder 2 modulo 5, we can instead compute 2^20 modulo 5. “
   productive: “Squaring gives 2^2=4, and squaring again gives 2^4=16, which is 1 modulo 5. Therefore 2^20=(2^4)^5 has remainder 1. The exponent 20 is divisible by the cycle length 4.”
   repetitive: “But is the remainder of 7 equal to 2? Yes, 7=5+2. “ ×4
5. “Solve x^2-5x+6=0 over the real numbers.”
   common: “The constant term is 6 and the coefficient of x is -5. “
   productive: “The numbers -2 and -3 have product 6 and sum -5. Thus the polynomial factors as (x-2)(x-3). A product is zero when either factor is zero, so x=2 or x=3. Substitution confirms both roots.”
   repetitive: “But is the constant 6? Yes, the constant term is 6. “ ×4
6. “For f(x)=x^3-4x, compute f'(2).”
   common: “The derivative of x^3 is 3x^2 and the derivative of -4x is -4. “
   productive: “Consequently f'(x)=3x^2-4. Substituting x=2 gives 3*4-4=8. We need the derivative at 2, not the function value f(2)=0; these are different quantities.”
   repetitive: “But do we need f(2)? No, the question asks for the derivative at 2. “ ×4

Plus 2 premature-close controls (cases 2 and 4): same prefix+common, continuation = `</think>`
immediately.

Total **14 texts**, ~120–160 tokens each ≈ **≈ 2,100 tokens per teacher-forced pass**, one pass per
candidate (plus a single shared original-model baseline pass, computed once and reused for all
candidates). This is ~200× cheaper than one generation pass (128 problems × 32k tokens ≈ 4M tokens).

### 4.2 Alignment and metrics

Per text i, find the token index b of the first token whose character offset ≥ `branch_char`
(`branch_char = len(prefix+prompt+suffix+common)`), exactly as Astra did
(`enc.offsets`; same-prefix tokens). With per-token log-probs `lp[i, t]`:

- `Δ_rep` = mean over the 6 repetitive texts of `mean_{t in [b-1, n-1)} (lp_edit[t] − lp_base[t])`
- `Δ_prod` = the same mean over the 6 productive texts
- `selectivity S = Δ_rep − Δ_prod`
- `premature_close` = `lp_edit` at the `</think>` position minus `lp_base` (want ≤ +0.5 nat;
  baselines ≈ −19.7; Astra's edits stayed ≤ −18.8, i.e., still essentially zero probability)
  (the controls are texts for problems 2 and 4, 1-based numbering — “cases `i in (1,3)`” 0-based in Astra's code)

Reference values for known edits (from the logs; original model = baseline):

| edit | Δ_prod | Δ_rep | S | premature-close logp (cases 2/4, 1-based) |
|---|---|---|---|---|
| full α1 | +0.0023 | −0.0241 | −0.026 | (≈ baseline) |
| full α2 | +0.0034 | −0.0487 | −0.052 | −19.27 / −19.29 |
| full α4 | +0.0045 | −0.1037 | −0.108 | −18.81 / −18.77 |
| top64 α4 | −0.0047 | −0.0645 | −0.060 | −19.46 / −19.55 |
| top64 α6 (fwd44) | −0.0090 | −0.1031 | −0.094 | (≈ top64α4) |
| random norm-matched control α4 | +0.0039 | +0.0096 | +0.006 | −19.67 / −19.43 |

### 4.3 Decision rule / ranking

Score candidates by S, but require:

1. `Δ_rep ≤ −0.05` nats/token (effectiveness),
2. `Δ_prod ≥ −0.01` nats/token (no collateral damage to productive reasoning),
3. premature-close logp increase ≤ +0.5 nat (no forced early termination),
4. (optional tie-breaker) repetitive tail-24-token Δ and productive tail-24 Δ.

If a candidate passes 1–3, it is a plausible replacement for the current edit; confirm its
generation accuracy on a *small* disjoint held-out split before committing (Astra's experience:
aggregate teacher-forced deltas tracked generation gains well — e.g. gen41/gen45 — but they do not
measure math correctness).

### 4.4 Cheaper entropy-only variant (secondary signal)

If you cannot afford 14 texts, a still-useful screen is: mean next-token entropy over the last
48 tokens on a single degenerate context — e.g., a reasoning trace that ends in an 8-way repeated
sentence (Astra found ~95–96% duplicate 8-grams in cutoff tails; `.958` mean duplicate8-share, and
`mean_nonwhite_cut_tail_duplicate8 = .493` after the edit) — versus a normal context. But this
correlates with, rather than measures, the mechanism; treat it as a smoke test, not a ranker. The
branch-logprob probe is the primary recommendation: it is the exact instrument Astra used to select
between α1/2/3/4, full vs L15-only vs L19-only vs top64, and targeted vs random control.

### 4.5 What this probe cannot tell you

- Accuracy/correctness on problems (needs generation; sample-budget decisions).
- Whitespace-degeneration failure mode (51/212 cut tails in the edited model were whitespace loops;
  needs generation inspection).
- Whether gains transfer to *originally finished* answers (Astra's stratum "originally finished",
  n=100: edited 93.0% vs original 94.0%, −1.0pp SE2.2 — the edit is neutral there, as expected).

### 4.6 Suggested protocol

1. One baseline TF pass (original weights) — cache log-probs once.
2. Per candidate: one TF pass on the same 14 texts (≈2.1k tokens, one forward pass, no sampling).
3. Rank by S subject to gates 1–3; keep the top 2 for a 32-problem disjoint exploratory generation
   (2 samples, 16k cap) if needed; final confirmation on the full protocol otherwise.

Cost for 10 candidates ≈ 11 passes ≈ 23k tokens ≈ 0.0006% of one full-budget generation pass.
