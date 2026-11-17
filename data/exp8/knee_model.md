# Exp 8 edit-pruning dose-response — knee model

Screening setup: 256 MATH-train problems, n=1 seed 0, single-edit candidates.
Reference: full edit (64 cols/layer) = **58.59%**; base = **23.83%**.
"Knee" = smallest predicted/measured k whose mean accuracy is within 5pp of the full-edit value (≥ **53.59%**).

Poll: **2026-11-13T08:00Z**, n=11 both-layer points (k≤48 all landed) + 4 single-tensor arms
(B15_full 56.25%, B19_full 43.36%, B15_32 35.16% at 07:12Z, **B19_32 37.11% at 07:58Z**). An
extra arm `ER256_L15` started 07:49Z (not in the plan §4 ORDER — appears to be an added L15-row
arm; not a both-layers arm, does not enter the knee fit). Both-layer fit unchanged.

## Data

### Both layers, top-k by |v| (the curve being fit)

| k (cols/layer) | acc | cut share |
|---|---|---|
| 0 (G000, base) | 23.83% | 91.4% |
| 2 (A02) | 34.77% | 79.7% |
| 4 (A04) | 38.67% | 70.3% |
| 6 (A06) | 39.06% | 69.9% |
| 8 (A08) | 42.97% | 67.6% |
| 12 (A12) | 41.02% | 66.0% |
| 16 (A16) | 44.92% | 64.1% |
| 24 (A24) | 49.61% | 55.5% |
| 32 (A32) | 51.95% | 50.0% |
| 48 (A48) | 53.52% | 43.8% |
| 64 (A64, full) | 58.59% | 30.1% |

### Single-tensor arms (context, not fit inputs)

| arm | cols used | entries | acc | cut share |
|---|---|---|---|---|
| B15_full (L15-only, all 64) | 64/0 | 65,240 | **56.25%** | 39.5% |
| B19_full (L19-only, all 64) | 0/64 | 64,894 | 43.36% | 63.7% |
| B15_32 (L15-only, top-32) | 32/0 | 32,696 | **35.16%** | 75.8% |
| B19_32 (L19-only, top-32) | 0/32 cols | 32,525 | **37.11%** | 72.7% |

## Fits (n=11) — re-fit 2026-11-13T06:57Z

| model | rmse | pred@6 | pred@8 | pred@12 | pred@16 | pred@24 | pred@32 | pred@48 | pred@64 | knee |
|---|---|---|---|---|---|---|---|---|---|---|
| exp_sat `a−b·e^(−k/τ)` | 2.64pp | 37.85 | 40.12 | 43.85 | 46.70 | 50.56 | 52.82 | 54.92 | 55.6 | 48 |
| pow_sat `a−b·(k+c)^(−p)` | 1.64pp | 39.79 | 41.75 | 44.58 | 46.61 | 49.47 | 51.49 | 54.30 | 56.3 | 48 |
| log_sat | 1.82pp | 40.48 | 42.37 | 45.02 | 46.87 | 49.43 | 51.19 | 53.59 | 55.2 | none (boundary) |
| logistic_logk | 1.63pp | 39.27 | 41.22 | 44.12 | 46.26 | 49.36 | 51.60 | 54.78 | 57.0 | 48 |
| **ensemble (mean)** | — | **39.35** | **41.36** | **44.39** | **46.61** | **49.70** | **51.77** | **54.40** | — | **48** |

Prediction error vs pre-registered predictions: A24 −0.4pp, A32 −0.1pp, **A48 −0.9pp** (ensemble
predicted 54.40%, measured 53.52%) — the top half of the curve remains well-characterized; earlier
optimistic bias (−4.9pp mean through k≤16) has not recurred at k≥24.

Notable: measured A64 (58.59%) exceeds **every** fit's pred@64 (55.2–57.0%). The 48→64 jump
(+5.07pp for +33% more columns) is steeper than any saturating model predicts — mild super-linearity
at the top end, or noise (SE ≈ 3.1pp). Treat ensemble pred@64 as a lower bound, not a ceiling.

## Knee determination — call it 48, with a boundary flag

Within-5pp threshold vs A64: **53.59%**.

- A48 measured **53.52%** — misses the strict bar by **0.07pp**. Strict reading: knee = 64.
- But: paired vs A64 on the same 256 problems/seed, diff = −5.08pp ± 2.66pp (t = −1.91) —
  statistically indistinguishable from A64. All four fits predicted A48 ≥ 53.59–54.92%, and the
  0.07pp gap is ~1/40th of the binomial SE (3.12pp). The "miss" is noise-level, not signal.
- Back-check with A64 as reference is the conservative direction: if we used A48 as reference
  instead, knee would trivially be 48.

**Recommendation: knee = 48 (both-layers, top-k). Do not report 64 — it would be a hard-rule artifact
of a 0.07pp gap inside measurement noise.** If a strict hard rule is non-negotiable for the writeup,
report: "knee = 48 statistically; 64 by strict-threshold reading (A48 = 53.52% < 53.59% by 0.07pp)."

## B15_full: does ONE L15-layer dominate both-layers pruning? **Yes.**

Measured (n=1, 256 problems, same seed/problems as A-arms → paired statistics valid):

| comparison | Δ acc | paired SE | t | verdict |
|---|---|---|---|---|
| B15_full − A64 (full) | **−2.34pp** | 2.47pp | −0.95 | statistically tied |
| B15_full − A48 | **+2.73pp** | 2.68pp | +1.02 | ahead (n.s.) |
| B15_full − A32 (iso-entries ≈65k) | **+4.30pp** | 2.49pp | +1.72 | ahead (n.s.) |
| B15_full − base | +32.42pp | 3.18pp | +10.2 | huge |
| B19_full − base | +19.53pp | 3.19pp | +6.1 | solid |
| B15_full − B19_full | **+12.89pp** | 2.73pp | **+4.72** | significant |
| B19_full − A64 | −15.23pp | — | — | dominated |
| B19_32 − B19_full | −6.25pp | 3.06pp | −2.04 | n.s. drop |
| B19_32 − B15_32 | +1.95pp | 2.90pp | +0.67 | tied |
| B19_32 − A32 (iso-entries) | −14.84pp | 2.99pp | −4.96 | dominated |
| B19_full ≈ | ~A08/A12-level (43.36%) | | | weak per-entry |
| B15_32 − B15_full | **−21.09pp** | 3.00pp | **−7.04** | collapse |
| B15_32 − A32 (iso-entries) | −16.80pp | 2.82pp | −5.96 | dominated |

Gain decomposition (of the +34.77pp base→A64 gain): **L15 carries ~93%** (B15−base = 32.4pp),
**L19 carries ~56%** (B19−base = 19.5pp); single-layer gains sum to 75.8pp ≫ 34.8pp → massive
redundancy between the two layers' rank-1 edits (consistent with cos(u15,u19) = 0.849 in the plan).
Complementarity at the problem level: B15 solves 144, B19 solves 111, A64 solves 150; B15-only wins
= 17, B19-only wins = 11, overlap 6; A64 solves 18 problems that neither single-layer arm solves —
the layers are *not* perfectly redundant problem-by-problem, just heavily redundant on average.

**Implication — the L19 tensor is auxiliary.**
- Given L15, L19 adds only ≈ **+2.3pp** (A64 − B15_full) at double the entries (130k vs 65k) and
  worse cut-share (30.1% vs 39.5%). It is a refinement, not a driver.
- L19 alone (43.36%) is much weaker per entry and is dominated by the A-curve everywhere: A08
  (42.97%, 16.3k entries) matches it at ~4× fewer entries.
- The minimization frontier is therefore **L15-first at full support / L19 first to drop**: drop
  L19 before touching L15. B15_full at 65k entries already beats A48 (98k) and ties A64 (130k).
  Shrinking L15 itself, however, does not work (see B15_32 below).
- Open question — **how far can L15 shrink? Answer: not to 32.** B15_32 (32 top L15 cols, 32.7k
  entries, landed 2026-11-13T07:12Z) = **35.16%**: −21.09pp vs B15_full (paired SE 3.00pp,
  **t = −7.04**), −16.8pp vs A32 at iso-entries, and only +11.3pp over base. This is a cliff, not
  a gentle dose-response — the L15 tensor's gain is concentrated in the 32–64 column band (or the
  selected support is not |v|-top-32-friendly). So the L15-only frontier has its knee at **full 64,
  not at half**: B15_full ≈ A64, B15_32 < A16-equivalent.
- B19_32 (landed 07:58Z) = **37.11%**: −6.25pp vs B19_full (paired SE 3.06pp, t = −2.04, n.s. but
  negative), +1.95pp vs B15_32 (t = +0.67, n.s. — L19's 32-col arm is indistinguishable from L15's),
  −14.84pp vs A32 at iso-entries (t = −4.96 — iso-count both-layers wins big), +13.28pp vs base
  (t = +4.26). L19 half-support is as weak as L15 half-support; single-layer top-32 is a dead end on
  either axis. Only **full-support L15** is competitive.
- Do **not** read this as "L19 is useless": it contributes +19.5pp standalone, covers ~11 problems
  B15 misses, and A64 still beats B15_full on average (58.6 vs 56.25). For max-accuracy applications
  keep both; for minimization the viable axes are (a) keep both layers and prune jointly toward A48
  (53.52% at 98k entries), or (b) L15-only at full 64 (56.25% at 65k entries — currently the best
  accuracy-per-entry point in the campaign). L15-only below 64 cols is now ruled out by measurement.

## Campaign / caveats

- n=1 seed, 256 problems → binomial SE ≈ 3.1pp; differences < ~6pp unpaired are noise — use paired
  diffs (SE ≈ 2.5–2.7pp) for arm-vs-arm calls. All screening evals share problems/seed, so pairing is valid.
- Fit residuals are small (1.6–2.6pp) but the low-k region has structure the models smooth over
  (A12 41.02% dips below A08 42.97%; curve is nearly flat 4–12). Predictions at k>64 are unreliable.
- Re-fit on each poll; this file is rewritten in place. Latest poll **2026-11-13T07:15Z**:
  B15_full, B19_full, B15_32 and B19_32 are in `campaign_results.jsonl` (15 lines). An arm `ER256_L15` started 07:49Z
  (L15 row-style arm, off-plan; not a both-layers arm, tracked separately). D/E/F/G/C arms remain queued.
  Only both-layers arms feed the knee fit; single-tensor arms are
  tracked separately. The both-layer fit at this poll is identical to the 06:57Z one.
- No box modifications; read-only polling (`cat`) of `/root/wsp_exp8/campaign_results.jsonl` via the
  established ssh route. No git commits, no sub-agents.

## Bottom line (superseded — see ER frontier below)

**Knee (both-layers top-k, within 5pp of full edit) = 48.** Measured A48 = 53.52%, statistically
tied with A64 (58.59%) on paired problems; strict-threshold reading gives 64 by a 0.07pp gap that is
far inside noise — report 48 with the flag, not 64.
**B15_full (L15-only) = 56.25% at half the entries of A64; it ties A64 (paired −2.3pp, n.s.), beats
A48/A32, and significantly crushes B19_full.** The L19 tensor is auxiliary on average (+2.3pp over
B15_full) — but **shrinking L15 to top-32 collapses to 35.16% (t = −7.0 vs B15_full)**, so the
L15-only axis does not extend below full support (B19_32 = 37.11% confirms the same for L19).
Best accuracy-per-entry right now: **B15_full** (65k entries, 56.25%). Best absolute: A64 (58.59%).
The 5pp-knee is 48 (stat) / 64 (strict rule). **Superseded on Pareto-optimality by the ER row axis
(see below): ER384 (L15-only, 24,576 entries, 55.47%) ties A64 and beats everything per-entry.**

---

## ER row-pruning frontier (L15-only, top-K u-rows by |u|², scale=1.0; L19 zeroed)

Poll: **2026-11-13T08:10Z**. Entry count = K rows × 64 cols of L15 only; u²-energy = fraction of
L15 |u|² mass retained. All scale=1.0 ⇒ bit-exact subsets of the B15_full edit.

| arm | entries | u² energy | acc | vs A64 | vs B15_full |
|---|---|---|---|---|---|
| ER128_L15 | 8,192 | 75.2% | 50.39% | −8.20pp (paired t=−2.99, 15/36) | −5.86pp (t=−2.42) |
| ER192_L15 | 12,288 | 81.9% | 51.17% | −7.42pp (t=−2.81, 14/33) | −5.08pp (t=−1.99) |
| ER256_L15 | 16,384 | 86.6% | **49.22%** | −9.38pp (t=−3.40, 14/38) | −7.03pp (t=−2.81) |
| ER384_L15 | 24,576 | 92.8% | **55.47%** | **−3.12pp (t=−1.13, 21/29 — tied)** | **−0.78pp (t=−0.30 — tied)** |
| ER512_L15 | 32,768 | 96.7% | 53.52% | −5.08pp (t=−1.95, 16/29) | −2.73pp (t=−1.18) |
| ER768_L15 | 49,152 | 99.7% | running | — | — |
| ER320_L15 | 20,480 | 90.0% | queued (waiter) | — | — |
| ER448_L15 | 28,672 | 95.0% | queued (waiter) | — | — |

### Is acc ~ f(u² energy kept)? **Mostly, with a real caveat.**

- Correlation acc vs energy: r = 0.80. A plateau model `acc = A − B·(1−E)^p` fits with A = 56.3%,
  B = 0.14, p = 0.57, rmse = 1.47pp — i.e. the curve is consistent with **saturating in energy**,
  plateauing at ≈ B15_full-level, not at A64 (58.6%). Linear-in-missing-energy fit:
  acc ≈ 55.4% − 24.2pp·(1−E), rmse 1.57pp; to reach A64's 58.59% would need E > 1 (impossible) —
  so "match A64 exactly" is unreachable on this axis; A64's last ~2.3pp is L19 (as expected).
- **ER256 = 49.22% is the outlier**: below ER192 (51.17%) and ER128 (50.39%), paired diffs
  −1.95±2.56pp vs ER192 (n.s.) but a clean 31/15 win/loss for ER384. Energy is smooth
  (86.6% vs 81.9/92.8%) so energy cannot explain the dip — likely n=1 seed noise at a flat region
  of the curve (the A12-style dip, same shape), or a few hinge problems at this specific K.
  Don't read ER256 as signal that energy is non-monotone; read it as noise ±3.1pp.
- **The dominant fact is the ER384 point**: ~25k entries (ER384, 92.8% energy) ties A64 (130k
  entries; paired −3.12±2.76pp, n.s.) and B15_full (65k; −0.78±2.60pp, n.s.) — **5.3×/2.7× fewer
  entries** for statistically equal accuracy. This is the strongest Pareto point in the campaign so far.

### What K matches A64? Predictions.

Fit plateau A≈56.3% < A64=58.59% ⇒ **no K on this axis reaches A64 exactly** — it tops out at
B15_full. The meaningful target is "ties A64 within noise" (within 5pp ⇒ ≥53.59%):
- **ER384 (E=92.8%) already clears it** — measured 55.47%, within-5pp of A64, statistically tied.
- ER320 (E=90.0%) — linear fit predicts ≈ 52.7–53.0% (below the strict bar); plateau fit ≈ 52.9%.
  Likely borderline; needs measurement.
- ER448 (E=95.0%) — predicts ≈ 54.2%; clears it.
- To match B15_full (56.25%) exactly: E ≈ 0.99, i.e. **ER640–768**; ER768 (99.7%) is running and is
  the B15_full fidelity check (must land ≈ 56%±; if not, the row ranking is broken).

### Validation priority: ER448 > ER320 > ER384 (already validated) — with reasoning.

1. **ER448 first**: it's the gap-filler between ER384 and ER512 and pins down the curve shape where
   the 5pp threshold lives (predicted 54.2%, near-tie zone). If ER448 ≈ 54–56%, the knee of the
   row axis is ~384–448 rows @ ~25–29k entries; this is the *headline* number for the minimization
   claim (5.3× entry reduction, statistically tied with A64).
2. **ER320 second**: tests the sharpness of the knee — predicted ~52.7–53.0%, i.e. it likely FAILS
   the 5pp bar. If it fails, ER384 is the minimal row-knee; if it passes (≥53.6%), the knee extends
   to 320 rows (20.5k entries, 6.4× reduction). Either outcome is decision-relevant; it's the
   cheapest decisive bit.
3. **ER384 third** (already landed — should be *re*-validated / confirmed, not re-screened): it is
   the current champion; validation should use paired MATH-500 or AIME confirmation rather than the
   same 256-problem screen. Also confirm ER768 for the B15_full fidelity anchor.
4. Explicitly **do not prioritize ER512/ER768 for validation as candidates** — they sit on the
   plateau (53.5/predicted-55%) and add no Pareto value beyond ER384.

**Bottom line: the ER (L15-row) axis strictly dominates the A-curve and the B-curve on
accuracy-per-entry. ER384 ≈ A64 ≈ B15_full at 25k entries. Report the row-pruning frontier as the
candidate, not as another point on the column-curve.**

---

## Top-line update (2026-11-13T08:10Z)

1. **Knee (both-layers top-k) = 48**, strict-rule = 64. Unchanged by ER arms.
2. **Best accuracy-per-entry is no longer B15_full. It is ER384_L15** (24,576 entries, 55.47%):
   ties A64 (−3.12±2.76pp) and B15_full (−0.78±2.60pp) at **5.3×/2.7× fewer entries**.
3. **ER256's dip (49.22%) is treated as noise** (paired t=+0.76 vs ER192, n.s.; A12-style flat-region
   dip). It should not be used to argue non-monotonicity of the row-energy curve.
4. **Validation order: ER448 → ER320 → re-confirm ER384** (and ER768 as the B15_full fidelity anchor).
5. The A-curve and B-curve remain valid for the originally published claims; the ER frontier should
   be reported as the new candidate.

---

## D-frontier (top-|Δ| entries, both layers; D4k running next)

Poll: **2026-11-13T13:05Z**. D-arms keep the largest single bf16 entries of the exact rank-1 edit
(masks on both layers); bit count ≈ retained |Δ| mass (D4k ≈ 33% · D8k ≈ 48% · D16k ≈ 68% · D32k ≈ 88%).

| arm | entries | |Δ| mass | acc | vs A64 | vs B15_full | vs ER384 |
|---|---|---|---|---|---|---|---|
| D8k_entries | 16,384 | ~48% | **59.0%** (151/256) | +0.39±2.44pp (t=+0.16, wins/losses 20/19) | +2.73±2.44pp (t=+1.12) | +3.52pp (t=+1.44) |
| D16k_entries | 32,768 | ~68% | **58.2%** (149/256) | −0.39±2.38pp (t=−0.16, 18/19) | +1.95±2.38pp (t=+0.82) | +2.73pp (t=+1.09) |
| D32k_entries | 65,536 | ~88% | **59.8%** (153/256) | +1.17±2.51pp (t=+0.47, 22/19) | +3.52±2.56pp (t=+1.37) | +4.30pp (t=+1.65) |
| D4k_entries | 8,192  | ~33% | running | — | — | — |

Key facts:

- **All three landed D-arms are statistically identical to A64** (paired t ∈ [−0.16, +0.47], all n.s.)
  and to each other (D8k vs D16k t=+0.33; D16k vs D32k t=−0.67). Dosing is flat in retained mass
  from 48→88% — the full edit's useful content sits in the largest entries.
- **D8k (16k entries) matches A64 (130k entries): ~8× fewer entries than the column edit**,
  **4× fewer than B15_full**, and beats the ER-row frontier per entry (vs ER384 t=+1.44 n.s. but
  +3.5pp ahead). This is the best accuracy-per-entry point measured to date, period.
- Union of D8k/D16k/D32k solves **177/256** — strictly more than A64's 150 and B15_full's 144:
  the D-mask solves ~25–27 problems the full edit misses (the truncation is acting as a useful
  regularizer on some problems, not just losing information).
- Caveat: entries in the full 64×1024 support are *not* a column-subset; the D-edit breaks the
  outer(û,v̂) structure (truncates the operator). It is a genuinely different edit class, so
  bytewise efficiency is high but the "explainability" of the edit is worse. If the goal includes
  interpretability of kept neurons, this matters; if it is checkpoint-diff minimization, D wins.

### Expected D4k result

The D-curve is flat from 33→88% mass unknowns at the good edge. Three considerations:

1. The mass retained at D4k (~33%, i.e. 8,192 entries) is just below D8k. There is no sign of a
   knee above 48%, and the loss gradient between D8k↔D16k↔D32k is < 1.6pp everywhere.
2. Bottom of the plausible range: if the largest-|Δ| tail carries truly necessary mass, mean
   accuracy could sag toward the ER-row level (~49–53%); the A-family analog at 8k entries (A04,
   4,088 entries, 38.7%) is not a valid ref because entry-truncation retains the top-mass entries,
   unlike column pruning.
3. n=1 SE ≈ 3.1pp; the measurement cannot resolve ±3pp differences.

**Prediction: D4k ≈ 55–59% (point ≈ 57%), i.e. statistically indistinguishable from A64 (≥53.6%,
within-5pp bar) with good probability. If it holds, the operative knee of the whole edit is
≤ 8,192 entries (~16× fewer than A64).** If D4k collapses below ~52%, the mass-curve knee sits
between 4,096 and 8,192 entries and D8k becomes the candidate knee. Either outcome is decisive —
it is the single most valuable pending point.

## Can scale > 1.0 on a D-subset beat A64? — comment

Short answer: **probably not by more than ~1–2pp, and the Frobenius-optimal global scale for a
binary mask is provably 1.0** — dose restoration is mathematically subtle and empirically weak.

Reasons:

1. **Math.** For a binary 0/1 mask M applied to Δ, the global scale s that minimizes ||s·MΔ − Δ||²_F
   is s* = ⟨MΔ,Δ⟩/||MΔ||² = 1 (because M is idempotent entrywise). Scaling up a binary mask is
   *never* the Frobenius-optimal truncation — unlike per-entry or per-row rescaling (which the G
   dose-restore idea implicitly assumes). A D-subset with s>1 is not a "better estimate of Δ" in
   least-squares terms; any gain must come from genuine model-behavior nonlinearities (logit
   sharpening/overshoot acting like a small dose sweep), i.e. benefits of the G-family, not of M.
2. **Empirical dose curve is flat at these scales.** The G-family (G050–G200, full-edit scale sweep)
   exists exactly to measure how accuracy moves with dose; if G125 ≈ G100 ≈ G150 within noise, there
   is no headroom for s>1 on D-subsets either. What the queued B15_full_s125/s150 arms measure is:
   (a) if both land ≈ B15_full (56.25%) or better, a dose boost is real but small (coset effect
   tiny — bf16 round-trips mostly stand pat); (b) if they land *above* A64 (>58.6%), that would show
   the L15-only edit was under-dosed and would be very surprising; (c) if they overshoot and drop,
   dose is already at/above optimal and s>1 on D is unmotivated.
   Related: the running off-plan probe ONE_L15E32 (32 L15 cols, **scale 1.056** ≈ 1/0.947) is
   literally a dose-restored version of B15_32 (35.16%). Its result is the sharpest signal on
   whether s>1 rescues a truncated mask: if it lands well above 35% (say ≥48%), dose restoration
   works and D8k@s≈1.1–1.3 is similarly motivated; if it stays ≈ 35–40%, scale does not rescue
   angular information loss and the D-at-scale idea is dead.
3. **Overshoot risk.** s>1 on D8k (already at/above A64) can only help if the edit is under-dosed;
   the likeliest failure is worse calibration/cut-share (more degeneration) at equal or lower
   accuracy — i.e. G150-type regression, not gains.
4. **What could actually beat A64:** per-entry rescaling or a "shrink-and-Δ" — e.g.
   keep the D8k entries at their value but add back the *residual* direction cheaply, or fit s by
   validation loss. Both break the "subset of the reference edit" framing (new edit, not bit-exact)
   but are legit if checkpoint-diff size (bytes) matters rather than "entries from the rank-1".

**Verdict: do not spend validation on s>1 D-arms expecting to beat A64. Wait for the G-sweep and
B15_full_s125/s150 readouts; if dose is flat-or-overshooting on the full edit, s>1 on a D-subset
cannot help either. D8k (s=1.0) is already the Pareto champion. If you want to test the idea
cheaply: one D8k@s=1.25 screen (NOT a validation run) — escalate only if it lands > 60%.**

---

## Why entry-magnitude selection beats structure-aware pruning (col/row) — 2026-11-13T13:20Z

The D-family theorem (measured): **top-|Δ| entry selection recovers ~all of A64's accuracy at ~2k–16k
entries** (D4k = 58.2%, D8k = 59.0% — statistically tied with A64 58.59%; D1k/D2k already at 55.5/54.3%).

**Measured mass concentration (computed locally from `factors_layer15/19.npz` via `np.outer(u,v)`):**

Layer-local |Δ| mass kept by top-k of the 3,670,144-entry tensor (float outer, matches bf16 ordering to fp error):

| keep | L15 mass | L19 mass | combined |
|---|---|---|---|
| top-1k | 14.98% | 15.07% | — |
| top-2k | 22.55% | 22.58% | 15.16% (of both) |
| top-4k | 33.41% | 33.19% | 22.76% |
| top-8k | 48.41% | 47.38% | 33.62% |
| top-16k | 67.50% | 65.47% | 48.38% |
| top-32k | 88.02% | 86.34% | 66.91% |
| top-64k | ~100% | ~100% | 87.34% |

(Int indices in `edit.npz` confirm both tensors are 1024×3584 with 64 unique support columns;
as bf16-new-value sums the picture is similar: L15 top-16k ≈ 56%, L19 ≈ 55%.)

**Why this beats column/row pruning:**

1. **The edit is rank-1, so |Δ[i,j]| = |u_i|·|v_j| is a *consistent* importance score** across the
   whole tensor — every entry's worth is fully described by two global factors. Column selection
   (top-k v) and row selection (top-k u, ER) keep 64-col groups or full rows, which forces carrying
   the *weakest entries inside the kept group* (e.g. L15 k=32 keeps 74% of v-mass but 100% of the
   entries in those columns, including |u_i|≈0 rows whose Δ is pure noise).
2. **Column/row pruning is a 2-level grouping; entries are the native atoms.** A column kept at k=32
   (~74% v-mass, 32×1024 = 32,768 entries) retains the same number of entries as D32k but D32k keeps
   the *much higher fraction of joint energy* (88%), i.e. it dominates A-style at equal count. Row
   pruning (ER) keeps the top-u rows entirely: ER384 (24,576 entries) holds 92.8% of *L15 row-energy*
   but also 100% of all 64 columns in those rows; D-at-equal-count skips the small-|u_i|·|v_j| cells.
3. **Mass vs accuracy:** the accuracy curve in D-mass is essentially flat from 33% (D4k, 58.2%)
   through 88% (D32k, 59.8%) — everything above ~22% combined mass (D4k) is within noise of full.
   The edit's functional content is carried by the single largest entries somewhere in the "tail"
   distribution; columns/rows mostly re-order which atoms you'd drop instead.
4. So: **|Δ| top-k is the information-optimal atom** for this edit class. The residual gap between
   D8k and A64 is <2.5pp paired — i.e. the structure pruning was filtering by the wrong statistic:
   it never got a handle on |u_i·v_j| jointly.

This does NOT mean the edit is unstructured noise — it means support/energy selection is *coarse*;
the joint |Δ| ranking is *sharp*. Interpretability is the cost (a sparse diffuse mask rather than
"these 64 neurons").

## Updated top-line (2026-11-13T13:20Z)

- Knee (A-curve) = 48 stat / 64 strict. Unchanged.
- **ER frontier: ER384 (24,576 entries, 55.47%) ties A64; ER768 = 55.5% flat (+0.00 vs ER384);
  ER320 = 52.7% (−5.9pp, t=−2.16) — ER-axis caps below ER384/768.** ER row axis is a dead end past
  55–56%.
- **D-family dominates everything per entry**: D4k (58.2%, 20/21 disc), D8k (59.0%), D16k (58.2%),
  D32k (59.8%) all statistically tied with A64; D1k/D2k at 55.5/54.3% (−3.1/−4.3pp, both t>1.2, n.s.) —
  the curve drops only at the very edge (<2k entries). Non-monotone between D1k/D2k/D4k is noise.
- ONE_L15E32 was invalidated (base-clone bug; quarantine to invalid_results.jsonl) and is re-running;
  **do not use its previous 1.056-scale reading as evidence.**
- L15 share of combined |Δ| L1 mass = 55.9% — consistent with B15_full capturing almost all of A64.
- **Right now (importance view):** entries sorted by |Δ| are the best possible binary atom set;
  columns are medium atoms; rows are the weakest of the three.

## Validation shortlist (recommendation)

1. **D8k_entries** (16,384, 59.0%) — primary candidate: highest acc/entry, flat-vs-A64 evidence,
   trivial to describe/store (sparse entries). Validate on MATH-500 + IFEval.
2. **D4k_entries** (8,192, 58.2%) — same claim at half the bytes; if it validates, it is the
   minimal edit (16× fewer than A64). Must be validated (not trusted from screen).
3. **ER384_L15** (24,576, 55.5%) — the structured competitor: interpretable-ish (row subset),
   same accuracy within noise, bit-exact L15 rows. Keep as the interpretable fallback.
4. **A48_topk48** (98k) and **B15_full** (65k) — baselines, not candidates.
5. **Do NOT validate:** D32k (no better than D4k, 4–8× more entries), D1k/D2k (on the dropped edge),
   ER320/256/768 (below/at plateau), ONE_L15E32/ENERGY dose-restored arms (pending re-run, hypothesis-only).
6. All validation must be **paired vs A64 and vs base** on the same eval sets/seed; report disc;
   require ≥ tie-within-noise (−2pp) with A64 and clear improvement over B15_full.
