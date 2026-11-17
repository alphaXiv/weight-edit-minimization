# Campaign plan: minimal-edit candidates for the Experiment 8 weight edit (Qwen3.5-0.8B, layers 15 & 19 MLP down_proj)

**Date:** 2026-11-12 · **Status:** ready to run · **Compute requested:** ~0.25 GPU-day screening (round 1) + ~0.5-0.6 GPU-day full validation (round 2) · **GPU used to write this plan:** none (all analysis CPU-only)

Goal: find the **fewest edits** to the two `down_proj.weight` tensors (`layers.15.mlp.down_proj.weight`,
`layers.19.mlp.down_proj.weight`, both 1024×3584 bf16) that preserve the edited checkpoint's behavior
(~48.1% MATH-500, ~67.1% GSM8K, ~62.3% IFEval, ~33.1% token-limit cut-share), measured first on 256 fixed
MATH-train screening problems (never test problems), then on full validation (MATH-500 ×4, GSM8K ×2, IFEval ×2).

Candidate build: `build_candidate.py --out <dir> --spec specs/<id>.json` (CPU, numpy only). Spec format:
`{"scale": float, "cols15": [...], "cols19": [...], "rows15": [...], "rows19": [...], "entries15": [[r,c],...], "entries19": [[r,c],...]}`.
`cols*` keeps only those columns of `outer(u, v)`; `rows*` keeps only those rows; `entries*` keeps only those `[r, c]` pairs;
per layer the three masks are mutually exclusive (`elif` in `build_candidate.py`); `scale` multiplies the whole delta.
**Single-layer candidates are expressed as `cols<other-layer>: []`** (an empty column list masks the layer to a zero delta).
`{"scale": 0.0}` reproduces the base model bit-for-bit (spec `G000_scale.json`) — the unedited screening control.
Spec files for every candidate below are pre-generated in `specs/` (34 files); only A64 needs noting (see §6).

## 1. What "number of edits" means — proposed definition and ordering

The families below cannot be ranked by one naive metric, so separate **count**, **structure**, and **dose**:

- **Primary currency (headline "number of edits"): `changed bf16 entries` — the count of entries whose bf16 bits differ from base**, the same quantity the project manifest reports (`changed_entries_total = 130,188`; layers 65,260 / 64,928).
  *Why:* family-agnostic (columns, rows, entry-masking, and dose-scaling all reduce to it), exactly measurable on CPU before any GPU run, monotone in dose, and it is the storage/apply cost of a sparse checkpoint patch — what "fewer edits" would actually save. The rank-1 description (2×1024 + 2×3584 + scale ≈ 9.2k floats) is *not* used: it measures description length, not how many numbers in the checkpoint move, and is blind to masking families.
- **Secondary (structure): `columns touched`** (both layers: 2×k; entry/row families: 128/64) and **`rows touched`**. Report alongside; this is what a human regularizing an edit would sweep.
- **Tertiary (dose): relative Frobenius change** `||Δ||_F / ||W0||_F` per tensor (full edit: 0.106 / 0.081). Tie-breaks equal-entry candidates and detects "many tiny flips" (scale-family) vs "fewer meaningful flips".
- **Lexicographic order for "fewer edits" claims:** (1) layers dropped, (2) tensors dropped, (3) columns touched, (4) changed bf16 entries, (5) relative Frobenius change. Use the tuple for cross-family comparisons; quote the scalar entries count in the headline.
- Also always report **layers/tensors touched** (2/2 here, or 1/1 for the single-layer family) — a 1-layer candidate at 65k entries is a *different kind* of minimality than a 2-layer candidate at 33k.

All counts below were computed exactly on CPU with the repo's `st_numpy` bf16 roundel (`S.bf16_bits`) against the base snapshot, mirroring `build_candidate.py`'s pipeline (float64 delta → float32 add → bf16 round-to-nearest-even).

## 2. Structural facts measured on CPU (they shape the design; no GPU needed)

1. **The edit is exactly rank-1 on a 64-column support per layer.** `v` has only 64 numerically nonzero entries per layer (non-support |v| ≲ 2e-16); the manifest's "64 selected neurons" are exactly the top-64 columns by |v| (`sel == top64|v|` verified for both layers). `delta_*.npy` matches `outer(u, v)` up to ≤2.7e-4 (fp32 factor rounding; build_candidate output for `A64_topk64` differs from the evaluated edit by only 54 bf16 entries — see §6).
2. **v is (minus) the projection of the base weights onto u, restricted to the selected set:** per-column ratio `v[j] / (uᵀ W0[:,j]) ≈ −1.962` (L15) and `≈ −3.067` (L19), std < 1.1e-3, cosine ≈ −1.0. So on the 64-column support, the orderings by |v|, by |Δ column norm| (= |v_j|·||u||, u-norm 1.749 / 1.399), by v² (energy), and by |u·W0 col| **all coincide** (Spearman ≈ 0.9999–1.0). **Family (c) "alternative selection criteria" therefore collapses to family (a)** and earns no dedicated screening slots; only a genuinely u-agnostic control is kept (`Cw0col16`).
   - Corollary: the selected set is *not* top-64 by |u·W0 col| over all 3584 columns (checked: `top64|uW| ≠ sel`, cutoff |uW| at ranks 62-65 ≈ 0.0444/0.0439/0.0448/0.0456 — an near-tie, but the sets differ). The ordering *within* the selected set is what matters for pruning, and there the criteria agree.
3. **Importance (|v| mass) decays slowly** (cumulative over the selected 64):
   - L15: k=8 → 29%, k=16 → 49%, k=24 → 63%, k=32 → 74%, k=48 → 91%, k=64 → 100%.
   - L19: k=8 → 31%, k=16 → 47%, k=24 → 59%, k=32 → 70%, k=48 → 88%, k=64 → 100%.
   (Task brief quotes 30/53/70/86/98/100 and 34/54/71/86/96/100 — those are cumulative shares of the in-set *|coef|* including bf16-rounding effects; absolute numbers differ slightly, shapes agree.)
4. **Rows:** top-|u| rows carry mass slowly too — L15/L19 top-128 → 40%, top-256 → 58-59%, top-512 → 82-83%. A row mask keeps all 64 columns but only R rows.
5. **Entries:** |Δ| mass by top-|Δ| entries — top-4096 → 33%, top-8192 → 47-48%, top-16384 → 65-67%, top-32768 → 86-88% (per tensor, |Δ| sums over the 64×1024 support).
6. **Iso-edit triples (near-identical bit-diff counts):**
   A08 / D8k / E128 ≈ 16.3-16.4k · A16 / D16k / E256 ≈ 32.7k · A32 / D32k / E512 ≈ 65.5k.
   This gives the round-1 design its evidentiary spine: *at equal edit count, which structure — columns, scattered entries, or row-block — survives?*
7. **cos(u15, u19) = 0.849** (precomputed; similar but not identical directions) — both layers must be tested before dropping either.

## 3. Candidate families, cheapest-first (all specs pre-generated in `specs/`)

| # | Family | Spec pattern (keys) | Examples (bit-diff entries, both layers) |
|---|--------|--------------------|------------------------------------------|
| G0 | Base / sanity | `{"scale": 0.0}` | 0 — the unedited control (paired-diff denominator) |
| A | Both layers, top-k cols by \|v\| | `cols15/cols19` = top-k of §2 | k=2: 4,088 · k=4: 8,174 · k=8: 16,338 · k=16: 32,653 · k=24: 48,951 · k=32: 65,221 · k=48: 97,756 · k=64: 130,134 |
| B | Single-layer drops, full + reduced | `cols<other>: []` | B15_full 65,240 · B19_full 64,894 · B15_32 32,696 · B19_32 32,525 (opt: B15_48 / B19_48) |
| C | Alt. selection criteria | — | **Collapsed** (§2.2): same order as A; only u-agnostic control kept as `Cw0col16` |
| D | Entry-magnitude threshold | `entries15/entries19` = top-\|Δ\| `[r,c]` | D4k 8,192 · D8k 16,384 · D16k 32,768 · D32k 65,536 |
| E | Row sparsity (top-\|u\| rows, all 64 cols) | `rows15/rows19` | E128 16,384 · E256 32,768 · E512 65,536 |
| F | Per-layer different k | `cols*` top-k_L15 / top-k_L19 | F32-16 48,989 · F16-32 48,885 · F48-24 73,424 |
| G | Dose-scale sweep (full edit) | `{"scale": s}` | G050 129,309 · G075 129,859 · G125 130,307 · G150 130,419 · G200 130,559 (bit count barely moves; row count is flat — dose ≠ fewer entries, it is a *reference*, not a minimization axis) |
| Ctl | Controls | as A | Cbottom16 (bottom-16 \|v\|, 32,378 entries, 9-12% mass) · Cw0col16 (top-16 by base column norm, 32,560, 27-41% mass) |

`F48-24`, `B15_48`, `B19_48`, `G050/G125/G200`, `E128` are generated but reserved (§5).

## 4. Round 1 — the campaign (~28 screening evals, ≈ 5.5-7 h ≈ 0.23-0.29 A100-day)

Screening protocol (per candidate): 256 fixed MATH-train problems, 1 sample, 32768 max tokens, temp 0.6 / top_p 0.95 / top_k 20 / presence_penalty 1.5, seed 0 → `eval_vllm.py --model <built dir> --problems screening_256.jsonl --n 1 --max-tokens 32768`.
**Use paired per-problem comparison vs A64_topk64 (and vs G000) throughout — same problems, same seed.** With n=256, single-accuracy SE ≈ 3.1pp at p≈0.5; paired differences of < ~4-5pp are noise. All accuracies here are vs the 48.1%/7.9% reference only loosely (MATH-train ≠ MATH-500; A64 re-screens to calibrate the offset).

| # | id | family | spec keys | changed bf16 entries | |Δ| mass retained (L15/L19) | rel. Frobenius (L15/L19) | GPU | What we learn |
|---|----|--------|-----------|---------------------|--------------------------|--------------------------|-----|---------------|
| 1 | G000_scale | base | scale 0.0 | 0 | 0/0 | 0/0 | ~12 min | unedited screening anchor; noise floor & harness check |
| 2 | A02_topk2 | A | cols 2/2 | 4,088 | .09/.12 | .044/.044 | ~12 min | extreme-prune floor |
| 3 | A04_topk4 | A | cols 4/4 | 8,174 | .16/.20 | .058/.055 | ~12 min | dose-response bottom |
| 4 | A06_topk6 | A | cols 6/6 | 12,255 | .23/.26 | .067/.059 | ~12 min | knee detection |
| 5 | A08_topk8 | A | cols 8/8 | 16,338 | .29/.31 | .075/.062 | ~12 min | iso-edit point vs D8k/E128 |
| 6 | A12_topk12 | A | cols 12/12 | 24,502 | .40/.40 | .084/.066 | ~12 min | fills curve gap |
| 7 | A16_topk16 | A | cols 16/16 | 32,653 | .49/.47 | .089/.069 | ~12 min | **likely knee candidate** |
| 8 | A24_topk24 | A | cols 24/24 | 48,951 | .63/.59 | .097/.073 | ~12 min | mid-knee |
| 9 | A32_topk32 | A | cols 32/32 | 65,221 | .74/.70 | .101/.075 | ~12 min | iso-edit point vs D32k/E512 |
| 10 | A48_topk48 | A | cols 48/48 | 97,756 | .91/.88 | .105/.079 | ~12 min | near-saturation |
| 11 | A64_topk64 | A | cols 64/64 | 130,134 | 1.00/1.00 | .106/.081 | ~12 min | **calibration anchor ≈ full edit (expect ≈48% modulo set/seed offset)** |
| 12 | B15_full | B | cols19=[], 64/0 | 65,240 | 1.00/0 | .106/0 | ~12 min | layer-15 causal share |
| 13 | B19_full | B | cols15=[], 0/64 | 64,894 | 0/1.00 | 0/.081 | ~12 min | layer-19 causal share |
| 14 | B15_32 | B | cols15=32, cols19=[] | 32,696 | .74/0 | .101/0 | ~12 min | L15-only minimization |
| 15 | B19_32 | B | cols19=32, cols15=[] | 32,525 | 0/.70 | 0/.075 | ~12 min | L19-only minimization |
| 16 | D4k_entries | D | entries 4096/tensor | 8,192 | .33/.33 | .092/.071 | ~12 min | scatter vs columns at 8k |
| 17 | D8k_entries | D | entries 8192/tensor | 16,384 | .48/.47 | .098/.075 | ~12 min | iso-edit vs A08/E128 |
| 18 | D16k_entries | D | entries 16384/tensor | 32,768 | .68/.66 | .103/.078 | ~12 min | iso-edit vs A16/E256 |
| 19 | D32k_entries | D | entries 32768/tensor | 65,536 | .88/.86 | .106/.081 | ~12 min | iso-edit vs A32/E512 |
| 20 | E128_rows | E | rows 128/tensor | 16,384 | .40/.40 | .092/.070 | ~12 min | iso-edit vs A08/D8k |
| 21 | E256_rows | E | rows 256/tensor | 32,768 | .59/.58 | .099/.075 | ~12 min | iso-edit vs A16/D16k |
| 22 | E512_rows | E | rows 512/tensor | 65,536 | .83/.82 | .105/.080 | ~12 min | iso-edit vs A32/D32k |
| 23 | F32-16 | F | L15 32 / L19 16 | 48,989 | .74/.47 | .101/.069 | ~12 min | layer-asymmetric knee |
| 24 | F16-32 | F | L15 16 / L19 32 | 48,885 | .49/.70 | .089/.075 | ~12 min | layer-asymmetric knee (mirror) |
| 25 | G075_scale | G | scale 0.75 | 129,859 | .75/.75 | .080/.061 | ~12 min | dose-response left of 1.0 |
| 26 | G150_scale | G | scale 1.5 | 130,419 | 1.5/1.5 | .160/.122 | ~12 min | dose-response right of 1.0 (overshoot risk) |
| 27 | Cbottom16 | Ctl | bottom-16 \|v\| both | 32,378 | .09/.12 | .016/.017 | ~12 min | **falsification:** if ≈A16, the \|v\| ranking carries no causal signal |
| 28 | Cw0col16 | Ctl | top-16 base-col-norm | 32,560 | .41/.27 | .079/.040 | ~12 min | u-agnostic criterion vs A16 at same count |

**Total ≈ 28 × 12 min ≈ 5.6 h ≈ 0.23 A100-day.** (Reserve if slots free up: `F48-24`, `E128` already in; `B15_48`, `B19_48` to separate layer shares at 48 cols; `G050`, `G125`, `G200` for a 4-point dose curve.)

### Flagged / expected-to-regress
- **A02, A04, A06** (and likely A08 if the effect is super-linear in mass): ≤26% |v| mass — plausibly indistinguishable from base on MATH-500; flag as likely no-op rather than useful minimization. Run them anyway as the dose-response floor (they are the cheapest evals in the set and anchor the Pareto curve).
- **Cbottom16 should regress to ≈ base.** If it matches A16, the entire importance-ordering premise fails and the search reverts to unstructured entry thresholds (D family).
- **D4k_entries:** scatters 8,192 flips over the support; worst-case pairing against A08 is the key iso-count readout. Not predicted to regress below base, but likely *worse per edit than column-keeping* — retains 33% mass in entries whose row/col structure is destroyed (the §2.2 rank-1 form is broken: the candidate is no longer ≈ outer(û, v̂)-structured).
- **G150_scale:** not a regression in edit count (dose family is flat), but overshoot can push IFEval down and cut-share up; a regression here is dose, not minimization. G075 is the safer "smaller effective edit" probe.
- **G000/A64 bracket the harness:** if A64 screens far from expectation on MATH-train, recalibrate all paired deltas, don't chase single-run noise.

### Information per GPU-hour (round 1)
- A-curve (10 evals): the full dose-response → identifies the knee and the minimal k; 10/28 of budget, highest information density.
- Iso-count triples (A08/D8k/E128, A16/D16k/E256, A32/D32k/E512; 7 evals): structure-at-fixed-edit-count — the single most decision-relevant question after the knee.
- B + F (6 evals): layer causality and asymmetry — enables the "1-layer minimization" claim if one layer is expendable.
- G (2 evals + G000): dose check & harness calibration — cheap, prevents misreading A-curve.
- Controls (2 evals): criterion validity — guards the whole family logic.
Round-1 cost ≈ 0.23 GPU-day vs 1 validation ≈ 0.11 GPU-day; **screening outruns validation ~11:1 per GPU-hour**, so breadth now, depth later.

## 5. Round 2 — confirmation & full validation (only after round 1)

1. **Refine around the knee** (if between A16 and A24): screen `k ∈ {18, 20, 28, 40}`-style refinements and/or the winning iso-family's neighboring budgets (e.g., D24k) — ~4-6 screenings ≈ 1-1.2 h.
2. **Full validation of the winner + 1 runner-up + A64 (re-baseline for the validation grid):** MATH-500 ×4 + GSM8K ×2 + IFEval ×2 per candidate. At ~2.7 h each ≈ 8.1 h ≈ 0.34 GPU-day for three; if budget allows, add the strongest single-layer candidate (≈10.8 h, 0.45 GPU-day).
3. **Decision rule:** accept the minimal candidate if its full-validation MATH-500 is within ~2pp of A64 (and GSM8K/IFEval not materially worse, cut-share not materially worse). Otherwise take the next-larger candidate on the A-curve (structure families only if an iso-triple wins decisively).
4. **Stretch (only if round 2 passes):** per-layer asymmetric refinement around the winner (e.g., 20/16-style), then a final held-out sanity check. Never iterate on MATH-500 itself beyond the final pick; keep screening on MATH-train.

**Total campaign ≈ 5.6 h (R1) + ~1 h (R2 refine) + ~8-11 h (validation) ≈ 15-18 h ≈ 0.6-0.75 A100-day,** vs a single full arm at 2.7 h — the whole minimal-edit search costs less than ~7 full evaluations.

## 6. Caveats & integrity notes

- **A64 ≠ bit-identical to the evaluated edit.** `build_candidate.py` recomputes `outer(u,v)` in float64 from factors and rounds; the evaluated edit was built from the submitted float32 copy. A64 differs in only 54 of 130,188 bf16 entries — negligible, but quote A64 as "≈ full edit", not bit-equal.
- **Screening ≠ MATH-500/GSM8K/IFEval.** 1 sample, seed 0, MATH-train; expect a systematic offset vs the 48.1% figure. All round-1 decisions should use *paired per-problem deltas vs A64 and G000*, never absolute values vs the blog numbers.
- **n=1 screening can misrank close candidates.** Any "winner" within ~4-5pp of A64 must be re-checked on the validation grid before any claim is made.
- **Entry-mask candidates are rank-not-1.** Their mechanics (sparse scatter over the support) may behave discontinuously under bf16 rounding (already accounted in bit counts); interpret D-family results per iso-count triple, not in isolation.
- **`scale` family is flat in edit count** (~130k at any scale): it calibrates dose, not minimality. Do not present it as a size-reduction axis.
- **Snap-in checks before spending GPU:** for each built candidate, verify symlinks/`model.safetensors-00001-of-00001.safetensors` loads; optionally verify a couple of SHA comparisons via `st_numpy` vs the manifest for A64 parity; confirm `eval_vllm.py` uses `enable_thinking=True` and the exact think35 sampling params.
- **files:** specs live in `specs/` (this directory); emitters `emit_specs.py`, analyses `analyze.py`, `analyze2.py`, `analyze3.py`, `overlap.py` — all new files, no existing files modified, no git commits, no GPU touched.

## 7. Budget summary

| Stage | Evals | GPU time | A100-days |
|-------|-------|----------|-----------|
| Round 1 screening | 28 × 256 problems × 1 sample | ≈ 5.6 h | ≈ 0.23 |
| Round 2 refinement screening | 4-6 | ≈ 1 h | ≈ 0.04 |
| Round 2 full validation | 3-4 × (MATH-500 ×4 + GSM8K ×2 + IFEval ×2) | ≈ 8-11 h | ≈ 0.34-0.45 |
| **Total** | | **≈ 15-18 h** | **≈ 0.6-0.75** |
