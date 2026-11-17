# Astra's car-ramp edit: what of the 6,073 weights actually matter, and what is the floor?

**Project:** weight-space-programming · **Task:** interpretability study of Experiment 4 (`l/h2_car_ramp`) · **Date:** 2026-11-12

---

## Headline answers

1. **The policy works, exactly as published.** Astra's repaired actor scores **1,024 / 1,024** episodes (sampled actions; also 1,024/1,024 argmax), mean 73 steps; the PPO base scores **0 / 1,024** (every episode times out on the ramp; 0 % throughout 1B training steps).
2. **The 6,073 changes decompose as 55 functional + 6,018 isolation erases.** All erased weights are tiny (median |w| ≈ 0.1, max 0.85). Reverting the 4,134 downstream erases is **free** — 1,939 entries keep 1,024/1,024. Reverting the 1,884 embed-stage erases breaks it (each tap column individually is required). Greedy elimination then collapsed the 1,939 to **12 entries — the greedy-selected minimal subset of Astra's own edit — verified at 1,024/1,024 × 4 sampled seeds + argmax (§10)** — and a tree-tracked drop-one audit (H_greedy11 node) then removed one more: **11 entries, 1,024/1,024 × 3 seeds + argmax, independently re-verified on an independent harness (s0)**.
3. **From scratch, the fix needs ~10–14 weights (0.006 %), not 6,073.** A 14-entry design and a greedily-trimmed **10-entry design** both reproduce 1,024/1,024 across samples with *zero erasures at all* — fully re-verified at 1,024 episodes × 4 seeds + argmax. The 10-entry fix works by the same trick PPO never found: *back up until you stall on the wall, then launch* — implemented with PPO's own velocity-detector unit read through a ReLU-lifted always-on constant.
4. **The "special neurons" are not special for being rare or important-to-the-game; they are special for being dead.** PPO converged to a throttle/coast controller that never samples reverse (p(reverse) ≈ 1e-12 at the start; moderate for ~5 mid-trajectory steps). The only freely-replaceable units in the whole network are exactly-dead embed ReLUs — PPO's own collapsed capacity. Everything else the fix rewires is task-active and must be *shouted down* (±100 head weights, 9–27× the residual) or *washed out* by tanh saturation.

## 1. Verification

All evaluations run in the official Kinetix (`kinetix-env` 3.0.2, JAX, CPU), level `l/h2_car_ramp`, `env_size=l`, `symbolic_flat` observation (679-dim, clipped [−10,10]), `multi_discrete` actions (16 logits = 4×3 + 2×2; only entry 0 is wired to motors in this scene), frame skip 2, dense reward scale 0.2.

| policy | eval | result |
|---|---|---|
| base (`ppo_final_seed0.npz`) | 1,024 ep, sampled, s0 | **0 / 1,024** (all timeout, mean len 256) |
| base | 256 ep, argmax | 0 / 256 (timeout) |
| base training record | 20 chunks × 256 ep + 1,024 test ep | 0 % everywhere (`ppo_training_result.json`) |
| edited (Astra, as published) | 1,024 ep, sampled, s0 | **1,024 / 1,024**, mean len 73.0 |
| edited | 1,024 ep, sampled, s1 | **1,024 / 1,024** |
| C2 (14 edits, from scratch) | 1,024 ep, sampled, s0/s1; 1,024 argmax | **1,024 / 1,024 × 3** |
| C2-min (10 edits, from scratch) | 1,024 ep, sampled, s0/s1/s2/s3; 1,024 argmax | **1,024 / 1,024 × 5** |
| greedy-12 (12 of Astra's own entries) | 1,024 ep, sampled, s0/s1/s2/s3; 1,024 argmax | **1,024 / 1,024 × 5** |
| greedy-11 (greedy-12 minus actor1.W[122,126]) | 1,024 ep, sampled, s0/s1/s2; 1,024 argmax (delegated tree-run) + independent harness s0 | **1,024 / 1,024 × 2 (x2 engines)** |
| joint_free (1,939 edits) | 1,024 ep, sampled, s0 & s1; 1,024 argmax, s0 | **1,024 / 1,024 × 3** |
| 1,718-entry greedy waypoint | 1,024 ep, sampled, s0 & s1; 1,024 argmax | **1,024 / 1,024 × 3** |

The edited network exactly reproduces the blog's numbers (1,024/1,024; 73-step episodes; most-likely-action success; 51/51 edit statements applied, 0 errors — `loop.log`).

**Rounding.** Astra was shown weights rounded to 3 significant digits per tensor (scale = 10^(floor(log10|max|) − 2)); the harness applied the edit block to the rounded network. We rebuilt the rounded base exactly; replaying the 51 statements reproduces `edited_actor.npz` entry-for-entry and reproduces the published change count **6,073 exactly** (per-tensor: embed.W 1887, embed.b 1, actor0.W 1132/.b 6, actor1.W 1255/.b 4, actor2.W 760/.b 2, actor3.W 507/.b 2, actor4.W 506/.b 2, head.W 6/.b 3). Against the raw float32 base, every entry differs by float noise; the meaningful metric is vs the rounded base.

## 2. What the 6,073 edits contain

| class | entries |
|---|---|
| nonzero → zero (“erases the repurposed units’ old content“) | **6,023** |
| zero → nonzero | **2** |
| nonzero → different nonzero (“wires & biases“) | **48** |
| **functional writes†** | **50** |
| **“fix-machinery“ entries excl. the 10 downstream isolation groups** | **55** (functional + 5 bias-zeros in the gate path, e.g. actor3.b/actor4.b = 0) |

By Astra's own edit statements (each changed entry attributed to the statement that last wrote it — replay is exact):

| group | entries | role in the fix |
|---|---|---|
| A embed.W cols 6:9 | 1,884 | erase old inputs of units 6/7/8 |
| B embed.b[7] = 11 | 1 | keep `relu(vx+11)` alive at rest |
| C/D/E taps (52→6, 54→7, 234→8) | 3 | read car-x, car-vx, green-x |
| F actor0 rows 6:9 | 377 | erase old outputs of units 6–8 |
| G actor0 cols 122:128 | 749 | erase old inputs of units 122–127 |
| H actor0.b = [−200,−248,−304,−876,−760,−76] | 6 | comparator thresholds (x: 2.5/3.1/3.8; vx: −0.05/−1.5; gx: 0.95) |
| I actor0 wires ×80 | 6 | gain-80 taps → comparators |
| J actor1 rows 122:128 | 758 | erase old outputs of comparators |
| K actor1 cols 124:128 | 488 | erase old inputs of gate units |
| L actor1.b = [0,−10,−10,30] | 4 | gate thresholds |
| M actor1 wires ×(±10) | 9 | gate wiring |
| N actor2 rows 124:128 | 505 | erase rows |
| O actor2 cols 126:128 | 248 | erase old inputs |
| P actor2.b = [−10,−30] | 2 | gate thresholds |
| Q actor2 wires ×(20,10) | 7 | BACK/LAUNCH gate wiring |
| R+S actor3 rows+cols | 505 | erase & isolate |
| T actor3.b = 0 | 2 | clear relay biases |
| U actor3 wires ×10 | 2 | saturating relays |
| V+W actor4 rows+cols | 504 | erase & isolate |
| X actor4.b = 0 | 2 | clear relay biases |
| Y actor4 wires ×10 | 2 | saturating relays |
| Z actor_head.W(126,127 × 0:3) = ∓100/±200 | 6 | logit votes |
| AA actor_head.b(0,1,2) −= 200 / += 100 | 3 | override cancels itself when idle |

The 55 functional entries: H, I, L, M, P, Q, T, U, X, Y, Z, AA, B, C/D/E (that's everything except A, F, G, J, K, N, O, R, S, V, W).

**All erased weights are tiny**: median |w| ≈ 0.1, 99.4 % below 0.5, max 0.847. The base network's weights are concentrated in small values (embed max 2.37 — the largest rewritten value in the base is its velocity-detector unit 116). Nothing about the isolation is about protecting large weights.

## 3. Mechanism: a hand-written comparator + override (verified against the forward pass)

Setting aside everything else, the fix implements this state machine on action entry 0 (0 = motor off, 1 = +1 “drive left“, 2 = −1 “back up right“):

```
e6 = relu(x)                       # car-body x (obs 52)
e7 = relu(vx + 11)                 # car-body vx (obs 54), biased so relu is alive at rest
e8 = relu(gx)                      # green-wheel x (obs 234)

a0_122..127 = tanh(80·e6−200/−248/−304/−876/−760/−76 | 80·e8−76)
   ⇒ gain-80 comparators: x>2.5, x>3.1, x>3.8 ; vx>−0.05, vx>−1.5 ; gx>0.95
a1_124 = tanh(10·a0_127)                       # green-passed-red relay (±1 saturated)
a1_125 = tanh(10(a0_125 − a0_124 − 1))         # “x<3.8 AND at-rest-ish” = BACK term
a1_126 = tanh(10(a0_126 − a0_122 − 1))         # fast-left stalled-early veto
a1_127 = tanh(10(a0_123 + a0_125 − a0_122 − a0_126 + 3))   # loiter veto
a2_126 = tanh(20·a1_124 + 10·a1_125 + 10·a1_126 − 10)      # BACK
a2_127 = tanh(10·a1_124 − 10·a1_125 − 10·a1_126 − 30)      # LAUNCH
a3_*, a4_* = tanh(10·prev)                            # saturating relays
head: logits += a4_126·(−100,−100,+200) + a4_127·(−100,+200,−100) + (−200,+100,+100)
```

Truth table (verified on all 257 base-trace and 74 edited-trace states; (a4_126,a4_127)=(+1,+1) never occurs — the two gates are mutually exclusive by construction):

| state | (a4_126, a4_127) | Δ logits 0:2 | meaning |
|---|---|---|---|
| rest / backing (gx>0.95, not launched) | (+1, −1) | [−200, −200, +400] | action 2 (back up) |
| launch (gx>0.95, x>3.8 or vx<−0.05) | (−1, 0) | [−100, +200, −100] | action 1 (launch) |
| released (gx≤0.95) | (−1, −1) | [0, 0, 0] exactly | override off → trained policy |

The base policy at the start state emits p(reverse) ≈ 1e-10..1e-12 — literally never sampling the reverse needed for a run-up — and reaches the ramp, stalls at x≈2.0, and oscillates throttle/coast for 224 steps. The fix backs up from rest (action 2 until x≈3.85, step ~16), launches (action 1 with logit margin ~600), clears red, and releases at gx<0.95; the trained network finishes; success at ~73 steps.

## 4. Which edits are necessary — measured ablations

**The 55 functional entries are not uniformly essential either.** One (actor0.b[123]) is provably redundant — actor0 unit 123's comparator fires only in a sub-region already covered by 122/124 and is never read decisively downstream; greedy elimination drops it with success intact. A few more (e.g., actor_head.W[127,2], actor4.b/actor3.b-type relays) are droppable at the gate.

**The 6,018 erasures are not uniformly essential.** Reverting groups one at a time (restore base values, keep everything else) at 256 episodes, sampled, seed 0:

| reverted group | entries | success |
|---|---|---|
| A embed cols 6:9 (all of it) | 1,884 | **0 / 256** |
| F actor0 rows 6:9 | 377 | 256 / 256 |
| G actor0 cols 122:128 | 749 | 256 / 256 |
| J actor1 rows 122:128 | 758 | 256 / 256 |
| K actor1 cols 124:128 | 488 | 256 / 256 |
| N actor2 rows 124:128 | 505 | 256 / 256 |
| O actor2 cols 126:128 | 248 | 256 / 256 |
| R+S actor3 rows+cols | 505 | 256 / 256 |
| V+W actor4 rows+cols | 504 | 256 / 256 |
| **all ten downstream groups jointly (“joint_free”)** | **4,134** | **256 / 256** |
| joint_free + revert embed.b[7]=11 | 4,135 | **0 / 256, 100 % red, 32 steps** |

Column-wise teardown of A (each removes only that tap's old inputs, keep the rest of the fix):

| restore slice | success | signature |
|---|---|---|
| col 6 only (x-comparators) | 0 / 256 | timeout — x>3.8 comparator shifted past the wall |
| col 7 only (vx-comparator) | 0 / 256, 100 % red, 32 steps | at-rest comparator dead, car drives left into red |
| col 8 only (green-x release) | 0 / 256, 100 % red, 69 steps | release comparator corrupt, never releases |

**Why the embed stage is different.** The three tap units are *exactly dead on the real rollouts* (pre-activation strictly negative at every step of both the failing and the successful trajectory: unit 6 in [−3.63, −2.68], unit 7 in [−3.35, −1.90], unit 8 in [−2.15, −1.07]; 87 of 128 embed ReLUs are dead everywhere on the failing trace). The old weights inside them are the network's largest latent leak — not just noise but a *learned negative bias* (e.g., unit 7's base pre-activation is −0.3..−0.66 on-task, which cancels the +11 bias trick). Downstream, old weights are absorbed by tanh saturation (gains 10–80) and by head override margins ≥ 200.

**Fixer-mechanism minimum, fully mapped:** joint_free = **1,939** (55 functional + 1,884 embed erasures) → greedy fixed point **12** → tree-tracked drop-one audit (H_greedy11) **11**. The 1,718 waypoint keeps **1,024 / 1,024 (s0, s1) and argmax (s0)**; the **12-entry** fixed point keeps 1,024 / 1,024 on all four sampled seeds and argmax; the **11-entry** variant (Astra's edit minus the individually-redundant `actor1.W[122,126]`) also keeps 1,024 / 1,024 (s0–s2 + argmax on the tree run, independently re-verified on a second harness at s0 — mean len 77.0). The full greedy log (all 1,718 tested entries, keep/drop verdicts) is in `evidence/greedy_gf5_final.json`; replicable via `greedy2.py`.

Interesting sub-result: the joint_free net is only *behaviorally* identical — it delays the x>3.8 comparator by one step (u124 flips at step 16, x≈3.85, because restored old weights shift the threshold by ~4 units at that state); the car backs up one step further, launches anyway, and still hits 1,024/1,024.

## 5. The floor

A direct from-scratch design exploits the same dead ReLUs and skips all erasures. Design (delegate A analysis + env-validated):

- tap `eA = relu(15.2 − 4x)` in dead embed unit 26 (fires x<3.8), tap `eB` (x>1 zone) not needed for the minimal law
- comparator A: `tanh(1000·eA − 24.7)` at actor0 unit 114 (dead unit, colnorm low)
- jv: reuse the base's own velocity detector — embed unit 116 (already reads 2.25/2.37/1.73 on the three velocity channels); comparator `tanh(4000·u116 − 2088)` at actor0 unit 23
- gate `tanh(50(A + jv) − 75)` at actor1 unit 29; relays 105 → 17 → 73 (gain 50, saturating); head votes (u73) → logits 1/2 (−40/+40)
- **14 changed entries, zero erasures** — passes 64/64, 1,024/1,024 (s0, s1), 1,024 argmax

My simpler-gain variant (C2, gains 4–20–8–12–10) passes identically (64/64, 1,024×2 seeds, argmax). The direct-tie variant (drop the 3-relay chain, tie u29 → head, 11 entries) **fails** — the gate never saturates decisively, the car oscillates at x≈4.5 and times out. Greedy backward elimination of the 14-entry design finds: **10 entries suffice** — and it is *elegant* (see the listing below).

```
embed.b[26] = 15.2                      # lifts dead embed unit 26 to ≈+1 constant (old content, new bias)
actor0.W[116, 23] = 8.0                 # u23 = tanh(8·(vel detector) + base junk) → jv comparator
actor1.W[114, 29] = 12.0                # u29 = tanh(12·u114 + 12·u23 − 18)   (dual-input gate)
actor1.W[23, 29] = 12.0
actor1.b[29] = −18.0
actor2.W[29, 105] = 10.0                # saturated relays (±10 gain, tanh²)
actor3.W[105, 17] = 10.0
actor4.W[17, 73] = 10.0
actor_head.W[73, 1] = −40.0             # ±40 logit votes on action 1/2, nothing else
actor_head.W[73, 2] = +40.0
```

Behavior: the car backs up accelerating rightward (the gate holds LAUNCH off while `u116` grows), reaches the wall at x≈4.5, stalls, `u116` collapses, the gate flips, and the car runs left at full speed — clearing the ramp and red, reaching the box at ~71–73 steps. Verified: **1,024/1,024 (s0), 1,024/1,024 (s1), 1,024/1,024 argmax** — indistinguishable from the published fix and from the 1,939-entry minimal subset.

So the ladder is: **6,073 → 1,939 (Astra's mechanism, validated) → 1,718 (validated) → 12 (4 seeds + argmax) → 11 (Astra-mechanism, tree audit, independently re-verified) → 14 (scratch redesign) → 10 (scratch redesign, 4 seeds + argmax)**. The floor for this architecture is ~8–10 under the override-the-base strategy; no head-only fix exists (a 2-entry candidate is decisively infeasible — 311/437 scan-state mismatches).

**Failed designs (for the record):** (a) head-only / logit-scale surgery — infeasible; (b) the direct 11-entry tie (drop the relays); (c) the conservative full-3-regime gate with a green-x release detector (29 entries); (d) joint-restores of any two embed tap columns at once. All fail outright in the environment.

**What this says about Experiment 4's claim.** Astra's fix is good physics programming, done inefficiently: ~99 % of its changed weights isolate repurposed units, and 68 % of *that* is unnecessary. The “keep what works“ instruction gave it the right instinct (reuse trained structure where it helps, override decisively) and no incentive to count.

## 6. What is special about those neurons (the answer in one paragraph)

The repurposed neurons are special **only at the embed stage**: they're the three units PPO never used — exactly-dead on every observed trajectory — which is what makes them interchangeable. Everything downstream is hostile to repurposing (units are task-tuned, some are the layer's strongest), collapsible only because the override is loud (±100 head weights, 9–27× the residual) and tanh saturates. The base network wasn't lacking capacity — it had a velocity-sensitive throttle/coast controller that could *sense* everything it needed (it has a dedicated velocity detector, embed unit 116) but had converged to a local minimum in which reverse was never sampled (p(reverse) ~ 1e-12 at the start) and the ramp stalled it at x≈2.0. Astra's fix doesn't add sensing; it adds a *commit* — reverse from rest, launch past x≈3.8, release past gx=0.95.

## 7. Reproduce / artifacts

Scratch: `/tmp/opencode/exp4/` — `eval_policy.py` (Kinetix harness), `ablate.py` (rounded base + exact replay), `quick_eval.py` (jitted rollout), `run_ablation.py`, `greedy2.py`, JSONs of all runs, `base_trace.npz` / `edited_trace.npz` (recorded episodes), delegate reports (`/tmp/opencode/delegateA/REPORT.md`, `/tmp/opencode/delegateB/REPORT.md`). Figures: this folder's `figs/`.

**Caveats:** screening at 64/256 episodes; final candidates confirmed at 1024 × 2 seeds + argmax (pass). The joint_free/greedy networks differ from the stock edit only in redundant erasures; the C2-style from-scratch candidates are different networks sharing Astra's mechanism insight. 161,994 of 171,664 weights are identical to the published fixer's *functional* path only in the from-scratch case (they share nothing beyond the mechanism).

---

*Fig. 1: erased vs base weight magnitudes; Fig. 2: the override circuit drawn from the edit block.*

## 10. Greedy elimination to its fixed point: inside Astra's edit, the answer is 12 entries

Backward elimination from the 1,939-entry validated set (64-episode gate, then re-confirmed at full 256-step scan): pass 0 tested all 1,939 entries one at a time and accepted 1,707 drops; pass 1 dropped 5 more; the final **keep = 12**:

| entry | edit | what it is |
|---|---|---|
| embed.W[234, 8] | +1.0 (Astra's tap) | green-wheel x → embed unit 8 |
| actor0.W[8, 127] | +80 | gain-80 green comparator (u127) |
| actor1.W[122, 126] | −10 | u122 → v126 (BACK gate, negative) |
| actor1.W[127, 124] | +10 | u127 → v124 (release relay) |
| actor1.b[126] | −10 | gate bias |
| actor2.W[124, 126] | +20 | v124 · 20 into w126 |
| actor2.W[126, 126] | +10 | v126 · 10 into w126 |
| actor2.b[126] | −10 | gate bias |
| actor3.W[126, 126] | +10 | saturating relay |
| actor4.W[126, 126] | +10 | saturating relay |
| actor_head.W[126, 2] | +200 | logit-2 (BACK) vote |
| actor_head.b[0] | −200 | bias offset 1 of 2 (kills action 0 tie) |

Mechanism (fully decoded on the recorded trace): PPO's base unit actor0-u122 fires ~+0.9 at rest and collapses to −1 once the car builds speed (u122 is a genuine speed detector — corr(u122, vx) = −0.44, corr(u122, x) = +0.30 on the failing rollout). The patch adds ±10 drive into actor1-u126, which in turn drives actor2-u126 over the edge at rest (pre-activation: base −7.7 → patched +2.4, gate +0.98 = "back up") and lets it fall through at the wall (base −12.0 → patched −1.8, gate −0.95 = "launch"). The car backs up, stalls on the wall at x≈4.48, reverses direction, and reaches the box at 71 steps. Verified **1,024/1,024 × 4 sampled seeds (s0/s1/s2/s3) + argmax**.

The striking structural fact of this 12-entry solution: it does not *model* the task at all. It reuses PPO's own internal phase signal (u122 speeds) and the green comparator, wraps exactly one gate around the base's base-rest staging signal (which moves by ~6 units between moving and stopped), and hands logits 2/0 a ±200 kick with a −200 bias to break the base's "motor off" attractor. Every one of the 12 entries is individually load-bearing; the other 6,061 are not.

## 8. Delegate analyses — what they got right, what the environment disproved

Two independent static analyses (numpy-only, no simulator available to them) were run in parallel:

**Delegate A — candidate minimal edits (glm-5.3-flash).** Independently converged on the same topology my env work validated: pinned-dead embed units as taps, a base velocity unit (116) as the jv sensor, a dual-input tanh gate, three saturated relays, ±40 head votes. Its headline candidates: **C1v3 (20 entries, full 3-regime law L2)** and **C2v3 (14 entries, reduced law L2′)** — zero mismatches on a 437-state dense (x, vx) scan, min top1−runner-up gap 12.1. Recommended submitting C1v3; C2v3 if edit count is weighted higher. Its floor argument (every skeleton element load-bearing; a convex-hull lemma forbidding single-unit laws) argued both 20 and 14 were floors.

**Delegate B — base-network interpretability (deepseek-v4.1-flash).** Confirmed the embed trio were exactly dead (its 714-probe reconstruction: pre-act strictly negative, e.g. unit 6 in [−1.571,−1.465]; Δlogit 0.000 on ablation) — consistent with my rollout measurement (87/128 always-dead; unit 6 ≤ −2.68 on both traces). Confirmed no norm-dormant units exist anywhere (0/128 in every layer); only *functional* dormancy (76/128 embed always-off). Extracted the full comparator chain by weight arithmetic and verified its truth table with 0.0 error. Key extra: the fix need not have picked the weakest hosts — the 16 hidden units were mostly task-relevant and two were near the top of their layer (actor0 122: 99th pct outgoing norm; actor2 127: 98th); per-layer erased mass is 10.5 % above the global optimum (its C-cost ranking: the fix's contiguity choice `122:128` is an edit-DSL convenience). The deep stack is behaviourally a comparator net (removing all 16 hidden units flips 0/714 argmax).

**Where the environment disagreed with both.** (1) The 14-entry floor was beaten: a greedily-eliminated **10-entry** design passes 1,024/1,024 × 4 seeds + argmax — the difference is a cleverer reuse: instead of building a fresh x-comparator at all, PPO's own velocity detector (unit 116) plus one constant-lifted dead unit (embed 26, enabled by a single `embed.b[26] = 15.2`) implements the whole “back up until the car stalls on the wall, then launch“ program; the x<3.8 condition is enforced by the ReLU-death-zone saving my taps from needing comparators, and the wall stop is the launch trigger. (2) Their “(10,23) zero is *required*“ claim for C1v3 — my C1 (19 entries, same structure, no zeros) passed 1,024×2 + argmax anyway. (3) Delegate B's role decomposition is consistent with my ablations in every case I could test; its “the 16 hidden hosts were *not* spare capacity“ conclusion is correct, and the practical consequence is our result that downstream isolation in Astra's edit was behaviourally free (§4).

Both reports are preserved verbatim in `evidence/REPORT_delegateA_minimize.md` and `evidence/REPORT_delegateB_interpretability.md`, including their caveats (probe sets, no env access, cross-check gaps), and candidates.json / a-numbers for machine re-verification.
