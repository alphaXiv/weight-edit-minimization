# Exp 8 screening — discordance analysis (A64_topk64 vs G000_scale base control)

Protocol: 256 fixed MATH-train problems, n=1, seed 0, 32k cap, temp 0.6 / top_p 0.95 / top_k 20 / presence-penalty 1.5, math-verify grading.
Base = `screen_G000_scale.jsonl` (scale-only control, treated as base); edits = `screen_A64_topk64.jsonl`, `screen_A04_topk4.jsonl`, `screen_A02_topk2.jsonl`. Subject/level from `math_train.jsonl`.
Caveat: single sample per problem; ~4–5pp paired noise. Discordance counts, not stable per-problem truth.

## Headline
A64 58.6% vs base 23.8% → **+34.8pp (+89 net problems)** = **93 rescues − 4 regressions**.
**All 93 rescues are problems the base cuts at the 32k cap.** Base finished-and-wrong is an empty cell (22/22 base `stop` finishes were correct). The edit's entire gain on this screen comes from shortening/terminating over-long (looping) generations — not from fixing wrong-but-finished reasoning.

## 1. Rescue set (n=93) vs regression set (n=4)

Level distribution (base→A64 rescues):
| Level | rescue | % of 93 | in 256 | base cut&wrong pool (195) | rescue rate in pool |
|---|---|---|---|---|---|
| 1 | 7 | 7.5% | 22 | 8 | 0.875 |
| 2 | 24 | 25.8% | 49 | 32 | 0.750 |
| 3 | 19 | 20.4% | 55 | 39 | 0.487 |
| 4 | 29 | 31.2% | 61 | 54 | 0.537 |
| 5 | 14 | 15.1% | 69 | 62 | 0.226 |

- L4+L5 fraction of rescues: **43/93 = 46.2%** (vs 50.8% of the 256; vs 71.6% of neither-correct set). Rescues skew easier but are not only easy: 43% are L4/5.
- Rescue rate falls monotonically-ish with difficulty in the base-cut pool: L1 87.5%, L2 75.0%, L3 48.7%, L4 53.7%, L5 22.6%.
- Mean level of rescues: 3.20 (A02 2.93, A04 2.78 vs A64 3.20).

Subjects (rescues; pool rescue rate): algebra 26 (0.765), prealgebra 20 (0.625), intermediate_algebra 15 (0.333), precalculus 12 (0.480), number_theory 11 (0.458), geometry 6 (0.375), counting_and_probability 3 (0.158).
All256 subject mix: algebra 60, prealgebra 50, int_alg 48, NT 28, precalc 27, C&P 23, geometry 20. Algebra/prealgebra over-represented in rescues; C&P strongly under-rescued.

Mean tokens:
- A64 on rescues: **18,809** vs A64 on non-rescues: 22,806 (rescues are the *shorter* A64 generations).
- A64 on finished rescues (83): 17,127 mean tokens.
- Base on rescues: **32,768 (all 93 cut at cap)**; base on non-rescues: 29,094.
- A64 on the 4 regressions: 30,912 (3 of 4 still cut at cap).

Regressions (n=4; all 4 were base `length`-cut but graded correct — presence-penalty-lucky boxed answers truncated at 32k — and all 4 are L3–L5):
- math_train_intermediate_algebra_1249 (L5, A64 cut)
- math_train_number_theory_526 (L4, A64 stop, 25,346 tok)
- math_train_number_theory_51 (L5, A64 cut)
- math_train_counting_and_probability_275 (L3, A64 cut)
Levels: 1×L3, 1×L4, 2×L5 → 75% L4/5 (harder than the rescue set). Subjects: NT×2, int_alg×1, C&P×1.
Churn (either direction): 97 problems (93+4).

## 2. Additivity / subset structure

| pair | |R_small| | |R_small ∩ R64| | Jaccard vs R64 | subset? | A64-unique misses |
|---|---|---|---|---|---|---|
| A02 (top-2) | 46 | 39 | 0.390 | no | 7 (algebra_1199, algebra_875, int_alg_136, int_alg_238, NT_324, prealgebra_1076, prealgebra_709) |
| A04 (top-4) | 51 | 42 | 0.412 | no | 9 (algebra_1199, algebra_519, algebra_875, int_alg_1149, int_alg_344, int_alg_693, NT_99, prealgebra_295, prealgebra_709) |

- R02 ⊂ R04? No — |R02 ∩ R04| = 28, Jaccard(R02,R04) = 0.406. Small-k sets are *not* nested either; they are noisy subsets of the same underlying "fix the loop" mechanism with large run-to-run variance (consistent with n=1, temp 0.6).
- Small-k rescues skew **easier** than A64's: mean level A02 2.93 / A04 2.78 / A64 3.20; L1–2 fraction A02 41.3% / A04 43.1% / A64 33.3%. So yes, small-k rescues pick off the easiest cut-loop problems first; A64's extra ~40 rescues are disproportionately L3–L5.
- **Yes — A04 rescues problems A64 does not** (9 of them). Same for A02 (7; 5 shared with A04's list: algebra_1199, algebra_875, prealgebra_709 + int_alg_136/238 for A02, algebra_519, int_alg_1149/344/693, NT_99, prealgebra_295 for A04):
  - algebra_1199 (L3): A04 stop/23.4k, A64 cut
  - algebra_519 (L4): A04 cut-but-correct, A64 cut
  - algebra_875 (L1): A04 stop/12.0k, A64 stop/19.4k wrong
  - int_alg_1149 (L4): A04 stop/26.8k, A64 stop/32.7k wrong (n_tokens 32717, finished but wrong answer)
  - int_alg_344 (L4): A04 stop/29.9k, A64 stop/22.3k wrong
  - int_alg_693 (L3): A04 stop/26.7k, A64 cut
  - NT_99 (L3): A04 cut-but-correct, A64 cut
  - prealgebra_295 (L3): A04 stop/7.4k, A64 stop/14.0k wrong
  - prealgebra_709 (L2): A04 stop/20.6k, A64 stop/24.1k wrong
  None of these were base-correct (all base-cut), so they are pure "small-k fixed, large-k broke/unfixed" disagreements — evidence of non-monotonic, high-variance behavior of the edit across k, not a clean subset relationship. (Also note: A02's & A04's rescue sets contain zero A64-both-correct problems.)
- A02/A04 each also have many base regressions (18 / 13 vs A64's 4), so small-k edits are strictly worse net but slightly less destructive per-rescue.

## 3. Decomposition of the +34.8pp (89 net problems)

Base finish behavior on the 256: 234 `length` (91.4% cut_share), 22 `stop` — **all 22 stops were correct; 0 finished-but-wrong**. 39 of the cuts were graded correct anyway (truncated-but-lucky), 195 cut-and-wrong.

| component | count | pp |
|---|---|---|
| (a) base cut → A64 finished (`stop`) + correct | **83** | +32.4 |
| (a′) base cut → A64 still `length` but graded correct | **10** | +3.9 |
| (b) base finished-wrong → A64 correct | **0** | 0.0 |
| (c) churn: base correct → A64 wrong (regressions) | **−4** | −1.6 |
| **net** | **89** | **+34.8** |

(a)+(a′) = 93 = all rescues. Churn total (both directions) = 97 problems. A64's own cut_share fell 91.4% → 30.1% (77 cuts, of which 13 correct; 42 of 179 stops wrong).

Implications for "which columns matter beyond the top-k":
- The screen isolates a single causal pathway: breaking the 32k-loop failure mode. Rescue ⊂ base-cut set is exact (195/195 base-cut-wrong pool contains all 93; 0 rescues from the 22 base-stop finishes). Any column-selection criterion aimed at reproducing the gain should target the loop/repetition direction's effect on *termination*, not on answer correctness per se.
- Rescue rate in the cut pool drops with level (L5 22.6%) and is near-zero for C&P (0.158) vs algebra 0.765 — the edit terminates loops but doesn't add missing capability; the L5/C&P residue is likely genuine capability failure, not loop.
- The 10 cut-but-correct rescues show some "rescues" are luck, not termination: the boxed answer appears before the cap but generation never stops. Distinguishing these matters if one wants columns that improve *stopping* vs columns that merely decorrelate the loop direction.
- 0 finished-wrong base cases means this screen cannot measure (b) at all — finished-wrong→correct would be the signature of an edit that fixes reasoning rather than looping; it needs a different (non-cut-dominated) eval slice to detect.
- Note the asymmetry vs the published MATH-500 cut_share (93% base): G000_scale reproduces that base pathology almost exactly (91.4%), so these conclusions transfer to the real base model modulo the seed-0 n=1 caveat.

## Files
- Inputs: screen_G000_scale.jsonl (base), screen_A64_topk64.jsonl, screen_A04_topk4.jsonl, screen_A02_topk2.jsonl, math_train.jsonl — all local copies byte-identical to the box (sizes match; box mtimes Nov 12 22:34–Nov 13 00:49, local copies Nov 13 01:27). No box modifications.
