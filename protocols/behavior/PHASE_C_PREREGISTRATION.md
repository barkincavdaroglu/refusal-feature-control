# S16 Phase C preregistration — generated-text necessity and sufficiency of C

Status: **frozen before authorization and before any Phase C model execution**

This preregistration is governed by `S16_PHASE_C_HANDOFF_PLAN.md` SHA-256
`d174a0242c8ed476d100cfd7d6caa807613d992564e9325c5cd926171b9a1b28`.
Its post-hoc motivating observation is bound to
`s16_phase_b_behavior_concordance.json` file SHA-256
`3fe15a5a2ba4fc41b2532a911c7cab5449c88fe80bb968d94c8fe818efa8e253`
and payload SHA-256
`d1c95c4cbc6488b0c37538e61df165c3fa92f884849bf97f3b2847451f01a3ca`.
Nothing in this file authorizes a model load, forward pass, generation,
intervention, labeling run, Phase D, remote execution, or sealed-S10 access.

## Frozen protocol copied verbatim from the handoff plan

### 5.1 Question and claim boundary
Does removing C's contribution on direct harmful prompts turn refusals into compliance, and does restoring C's natural dose on fictional frames turn compliances into refusals, in **generated text** rather than next-token log-odds? Estimation design with intent-level intervals; interpretation bands, no pass/fail gates; no selectivity claim from any single arm.

### 5.2 Panel (no new construction)
- The 180 S10 frames (20 harmful intents × 6 frames = 120; 10 benign intents × 6 = 60), using the S16 census rows (same prompts; hash-verified in Track 1). These are S10's validation and expansion partitions; the sealed S10 confirmation partition is not touched. They were never used to select or tune the route (E9d used its own panel, excluded from the census).
- Plus the 24 fresh reference prompts (12 harmful / 12 benign) as an out-of-family control where C never fires.
- Declare the deviation from `ROUTE_FUNCTION_PLAN.md` ("fresh panel"): the S10 frames are preferred because they carry blind baseline labels and cover the D/F contrast the question is about.

### 5.3 Interventions (exact)
Let x_t be `blocks.31.mlp.hook_in` at position t, a_t = ReLU(w·x_t + b) in bf16 exactly as the census, d = W_dec[C], y_t the layer-31 MLP output (`blocks.31.mlp.hook_out`). "Final position" = the last prompt token (the assistant-generation prefix token the census calls formatted-final).

| Arm | Rows | Edit | Dose |
|---|---|---|---|
| Baseline | all 204 | none | — |
| A1 ablation, primary | 120 harmful | y_t ← y_t − a_t·d at every position where a_t > 0, prompt and generated tokens | natural, per position |
| A1-final (sensitivity) | 120 harmful | same edit at the final prompt position only | natural |
| A2 restoration, primary | 60 fictional + 24 fresh | y_T ← y_T + δ·d at the final prompt position | δ = the intent's D0 census activation (H23: use D1; if all D are 0, the panel median 18.81) |
| A2 doses (sensitivity) | same rows | same | δ ∈ {16.75, 22.78} (25th/75th active quantiles) and 18.625 (historical, explicit high arm) |
| A3 benign addition | 60 benign | y_T ← y_T + δ·d at the final position | δ = 18.81 |
| RB twins | rows of A1-final, A2-primary, A3 | the arm's edit plus `blocks.32.mlp.hook_out` at the final position patched to its baseline value (cached from the baseline run) | as the arm |
| RD control | rows of A2-primary and A3 | y_T ← y_T + ‖δ·d‖·r̂, r̂ = Phase B primary refusal direction (resid_post, unit) | matched norm |
| RND control | same rows | y_T ← y_T + ‖δ·d‖·û, û random unit vectors, seeds 1707 and 1708 | matched norm |

Notes. (i) A1 removes the *transcoder's* estimate of C's contribution; that is the only available operationalization of "C silent" and is stated as such (T-002). (ii) Receiver-blocked twins are defined only for final-position edits because generated positions diverge from the baseline and have no matched cache. (iii) A1 on the fresh rows is a no-op (C = 0) and is skipped. (iv) Nothing is patched at layers other than 31 (edit) and 32 (block).

### 5.4 Generation and measurement
- Greedy, `max_new_tokens=128`, identical to S10; batch size one on MPS bf16; TransformerLens `generate` with the hooks active for every forward (prefill and each decode step), KV cache on. Record a 5-prompt repeat for run-to-run identity.
- Per generation: the text (private), its SHA-256, the next-token distribution shift at the final prompt position (KL from baseline, top-10 promoted/suppressed token ids), S10's `primary_score`, the probe value, and a_T before and after the edit.
- **Labels**: R/S/P/C by `VALIDATION_ANNOTATION_GUIDE.md`, assigned **blind** — all generations from all arms and the Mac baseline shuffled with a private key, arm and prompt identity hidden. Labeler: Barkin, or the agent as in S10's expansion pass; either way, a ≥ 50-item second-rater check with agreement rate reported. No new judge model, no judge infrastructure.
- Mac-baseline labels versus S10's labels on the same 180 prompts: report the agreement rate first. Below 0.80, stop and report before analyzing arms (device or labeling drift would contaminate everything).

### 5.5 Analysis (frozen in the preregistration)
- Primary outcomes, each with intent-cluster bootstrap 95% intervals (seed `2522210527409210963`, 10,000 replicates):
  - **Necessity**: among direct harmful rows labeled refusal_like at Mac baseline, the fraction that become compliance_like under A1 (primary) and A1-final.
  - **Sufficiency**: among fictional rows labeled compliance_like at baseline, the fraction that become refusal_like under A2-primary; dose curve over the sensitivity doses.
  - **Benign**: fraction of benign rows that become refusal_like or S under A3; the same for RD and RND (this is the generation-level version of the selectivity question and is reported side by side, never as a gate).
- **Route share** per arm = 1 − (effect under RB twin) / (effect under arm), reported only when the arm's effect has an interval excluding 0 and absolute size ≥ 0.10.
- **Sparse-facet contrast**: A2-primary versus RD at matched norm on label distribution, KL, and route share, with intervals.
- Continuity: the same outcomes on the next-token score and probe, to say whether logits and text agree (T-003).
- Interpretation bands, written now: A1 flips ≥ 0.5 of direct refusals → C is necessary for most refusals on this family; ≤ 0.2 → C is a redundant carrier (consistent with the 17 silenced-but-refused observations); between → partial. A2 turns ≥ 0.5 of fictional compliances into refusals → the route is sufficient at the text level; ≈ 0 while the next-token score moves → "logits move, text does not" (the T-003 warning; downgrade E9d's interpretation accordingly). A3 producing refusals on benign rows at rates comparable to A2 → content-indifferent plumbing confirmed at the text level; RD matching A2 on every measure → C is a coordinate of the refusal direction after all (Phase B's cosine says otherwise; the generation test decides).

### 5.6 Compute
Generations: baseline 204; A1 120; A1-final 120; A2 84 × 4 doses = 336; A3 60; RB twins 264; RD 144; RND 288. Total ≈ 1,536 generations × ≤ 128 tokens ≈ ≤ 200k generated tokens. At batch-size-one MPS decode (15–30 tokens/s) that is 2–4 hours; run it overnight in one script invocation with a resume-by-manifest (skip completed generations whose hash re-verifies). If time is short, drop one RND seed and the two quantile doses (−240 generations).

## Deviation declarations and motivating observation

1. **Panel reuse.** This replaces `ROUTE_FUNCTION_PLAN.md`'s fresh-panel plan
   with the 180 already labeled S10 validation and validation-expansion frames,
   plus 24 fresh reference prompts as the out-of-family control. Track 1
   recomputed and matched all 180 framed-prompt hashes with zero exclusions.
   The sealed S10 confirmation partition remains prohibited and untouched.
2. **Transcoder-based ablation.** A1 subtracts the released transcoder's
   estimate `a_t·d`; it does not delete a uniquely identified native-model
   variable. Any result inherits threat T-002 and is interpreted only under
   this operational counterfactual.
3. **Existing single-pass baseline labels.** The baseline comparison reuses
   S10's one annotation pass for each 90-record partition (human manual labels
   for validation; Codex-agent labels for validation expansion). Phase C's new
   generations must still be shuffled and labeled blind, with the declared
   second-rater check of at least 50 items. No judge model is added.
4. **MPS bf16, batch size one.** The local Mac branch is fixed because Phase B
   observed batched-versus-single maximum absolute difference `0.7890625`,
   above its frozen `0.125` parity threshold. This is a numerical execution
   choice, not evidence about the hypothesis.
5. **Post-hoc motivation, not evidence of causality.** The Track 1 table was:

   | Final-token C | Refusal-like (R/S) | Compliance-like (P/C) |
   |---|---:|---:|
   | Active (> 0) | 72 | 1 |
   | Silent (= 0) | 17 | 30 |

   The intent-clustered active-minus-silent refusal risk difference was
   `0.624599` (95% interval `[0.426740, 0.789474]`). Of 23 refusal-to-
   compliance flips, 22 silenced C, while 17 other silenced pairs still
   refused. Spearman `rho=0.419483`, but the valid intent-vector permutation
   p-value was `0.149585`; the earlier record-level p-value is not retained.
   These results are explicitly post-hoc and motivate—but do not answer—the
   necessity and sufficiency questions.

## Authority and stopping boundary

Only Barkin's exact, post-hash authorization sentence specified by the handoff
plan can authorize Phase C. Until it is recorded, implementation may not load
the model or execute a smoke, baseline, intervention, or generation. Phase D
requires its own later preregistration and authority even if Phase C moves
generated behavior.
