# S16 Phase C amendment 01 — binary agreement, audited labels, and Tier-1 continuation

Status: **frozen after baseline labeling and its independent audit, before every
intervention-arm outcome; awaiting exact post-hash authorization**

This amendment modifies the baseline-agreement estimand, baseline reuse
boundary, first intervention tier, annotation plan, and execution order in
`PHASE_C_PREREGISTRATION.md` SHA-256
`25928a120f53a277553a15a6f271463b990f52f186653773f117507ff0a6f13d`.
Every unmodified privacy, model, device, generation, analysis, and Phase-D
boundary in the parent preregistration remains in force.

## 1. Timing and evidence available at amendment freeze

This is a transparent post-baseline, post-label-audit, pre-arm amendment. The
204 Mac baseline generations and their first blind R/S/P/C labels exist. No
intervention or control generation exists, so every Tier-1 arm outcome remains
unseen.

The amendment is bound to `PHASE_C_BASELINE_LABEL_AUDIT.md`:

| Audit artifact | SHA-256 |
|---|---|
| `PHASE_C_BASELINE_LABEL_AUDIT.md` file | `c5c4b65c45d53f471d2ec3ef94e989afda9b6bf99f24adeedecb57135cd9d0e1` |
| `outputs/phase_c_local/s16_phase_c_baseline_label_audit.json` file | `1644fe83e2c42ac39741d1873ff43b1fdadf388be7dfa9d500ed5587dab4d56f` |
| JSON semantic payload | `21f7049be76295fc5105efc5c3a44b03a5c67866eca55320451e4fd8c5a6df30` |

The JSON is the machine-readable authority for the audit numbers; the Markdown
is its public-safe report. Neither contains prompt or completion text. The
user-supplied transcription ended `...fd8a5a6df30`; it differs by one nibble
from both the stored and independently recomputed payload and is therefore not
used as an authentication value.

## 2. Amended binary agreement gate; threshold unchanged

Phase C's primary generated-text estimands in parent §5.5 are binary:

- `refusal_like = {R, S}`;
- `compliance_like = {P, C}`;
- denominator: the same 180 prompts shared with S10;
- threshold: **unchanged at 0.80**;
- observed agreement: `169/180 = 0.9388889`, therefore passed.

The separate binary-gate record has file SHA-256
`f8e44f084ad3ea9dced07e652de65d0427e87b29f7308236f0660e5192eda1eb`
and payload SHA-256
`96f48a31102e189113f204a4b334c4a02830026444f7fe1278452aa13e8fb0df`.

Exact four-class agreement remains visible as `120/180 = 0.6666667`; it is not
deleted, relabeled, or represented as passing. The audit's blind second-rater
check found binary Cohen's kappa `0.8361096` on all 61 rated baselines and
`0.8797673` on the random 50, versus four-class kappa `0.4868069` and
`0.4687131`. Mac/S10 within-class disagreements were dominated by Mac `S`
versus S10 `R` (`42` cases). Therefore every **R-versus-S contrast is
labeler-dependent**. R/S arm counts may be reported only as labeler-attributed
descriptions; no necessity, sufficiency, or interpretation-band conclusion may
depend on an R/S split. The binding outcomes remain refusal-like versus
compliance-like.

No post-unblinding change to a baseline label is permitted.

## 3. Immutable reuse of the 204 baselines

Continuation reuses the completed baselines. It does not repeat the smoke,
regenerate any baseline, or repeat baseline annotation.

| Reused artifact | SHA-256 |
|---|---|
| Run manifest at the continuation boundary | `d3d61392e4bd94281b8a2d3177d3f110e6ac38f0f7f8e878661015a1596dac84` |
| Smoke receipt | `cf889d835bbb94a7f772f3ebd9403b78973396ac301f0fc1067b33d3cafa134f` |
| Five-prompt repeat receipt | `80c3eb8679ed3a6cf629633a9c68a8e7d73b16698d983c3ba7e30ad1bf593ce0` |
| Baseline blind-packet receipt | `7ba405a7e8cce3b7fd4534ea2fa077fe88f8fc46ad44ff7be6152d588bffee41` |
| Frozen baseline annotations | `3d0a9daec25d21b938b06587931ee4a0131d932a6d07ed7a6e70cc9b46767c43` |
| Original exact-label gate | `b5745b0914f31ba2592b1907f1f483f4697ca0928494dc74a323bb45585868ec` |

At amendment freeze the manifest status is `halted_baseline_gate_failed`, it
binds exactly 204 baseline records, and the public records directory has zero
non-baseline records. Before each intervention, the runner must authenticate
that row's baseline public record and private layer-32 cache.

## 4. Tier 1 — exactly 696 preregistered arms plus 120 random-ablation controls

The 696-generation core is the three primary arms, their available
receiver-blocked and refusal-direction controls, and one of the two already
preregistered random-direction seeds for A2/A3:

| Arm ID | Rows | n |
|---|---|---:|
| `a1_persistent` | all harmful S10 frames; natural C contribution removed wherever active during prompt and decoding | 120 |
| `a2_natural` | harmful fictional frames plus 24 fresh references; intent-natural dose at final prompt position | 84 |
| `rb_a2_natural` | receiver-blocked twin of `a2_natural` | 84 |
| `rd_a2_natural` | matched-norm refusal-direction control for `a2_natural` | 84 |
| `rnd1707_a2_natural` | matched-norm seed-1707 random-direction control for `a2_natural` | 84 |
| `a3_benign` | benign S10 frames; dose 18.81 at final prompt position | 60 |
| `rb_a3_benign` | receiver-blocked twin of `a3_benign` | 60 |
| `rd_a3_benign` | matched-norm refusal-direction control for `a3_benign` | 60 |
| `rnd1707_a3_benign` | matched-norm seed-1707 random-direction control for `a3_benign` | 60 |
| **Tier-1 core** | | **696** |

The additional 120-generation random-ablation control is:

| Arm ID | Rows | Frozen edit | n |
|---|---|---|---:|
| `rnd1707_a1_persistent` | all 120 harmful S10 frames | At every prompt or generated position where A1 computes `a_t`, subtract `a_t ||d_C|| r_1707`, where `r_1707` is the preregistered seed-1707 unit random direction and `d_C` is C's decoder. | 120 |

This control matches A1's per-position edit norm exactly but not its direction.
It is a nonspecific persistent-edit control, not an ablation of C and not a
second claim that C was silenced.

The authorized continuation therefore contains exactly **816 new generations**.
It defers 636 parent-preregistered generations: A1-final and its RB twin (240),
the three A2 sensitivity doses (252), and seed 1708 for A2/A3 (144). Including
the new random-ablation control, the amended complete arm universe is 1,452;
only the 816 records above are in Tier 1.

Tier 1 is an information-first subset fixed before any arm outcome. It adds no
outcome-dependent pass/fail gate.

## 5. Execution and blind annotation

1. Before model loading, authenticate this amendment by its exact command-line
   SHA-256, both audit files and the audit payload in §1, the binary gate, the
   original input closure, and every reused baseline/cache binding.
2. Continue with the same local Mac MPS-bf16, batch-size-one, greedy,
   `max_new_tokens=128` protocol. Existing baseline records are skipped and
   never overwritten. The Tier-1 run is resumable by authenticated manifest if
   interrupted.
3. Execute only the 696 core records and 120 random-ablation-control records in
   §4. The expected runtime is roughly 8.5 hours; this is an estimate, not a
   wait, deadline, lease, clock gate, or teardown trigger.
4. Once all 816 records authenticate, shuffle them behind private blind IDs.
   The primary labeler is the agent in a fresh blind pass: it sees prompt and
   completion text but no record, row, prompt-family, risk, or arm identity and
   assigns R/S/P/C using `VALIDATION_ANNOTATION_GUIDE.md`.
5. Preserve the 204 baseline labels byte-for-byte. Primary effect estimates use
   only the frozen binary classes in §2; exact R/S/P/C distributions remain
   labeler-attributed descriptive detail.
6. Barkin independently second-rates exactly 100 Tier-1 items blind, ten from
   each of the ten arm IDs in §4. Within each arm, choose the ten smallest
   SHA-256 ranks of `s16-phase-c-tier1-second-rater-v1|record_id`; reshuffle the
   selected items by the analogous `...-order-v1|record_id` rank before display.
   The packet and resume-safe CSV follow the same private interactive-labeler
   pattern as the baseline audit and reveal no arm or record identity.
7. Report Tier-1 binary kappa with a 95% bootstrap CI over the 100 items. If the
   CI excludes 0.8361096 below, report before any effect estimate that the arm
   texts were more labeler-dependent than the audited baseline benchmark. A
   point estimate below the benchmark whose CI includes it is reported as
   indistinguishable from the benchmark.

Private generations, blind keys, packets, and annotations remain mode 0600 in
the existing private output directory. No raw prompt or completion is made
public.

## 6. Implementation and budget accounting

The runner authenticates the audit before model loading, implements the
matched-norm persistent random-ablation control, and rejects any Tier-1
schedule other than `696 + 120 = 816`. The finalizer accepts the Tier-1 blind
packet without rewriting the original exact gate.

Frozen source hashes at amendment freeze are:

- `src/run_s16_phase_c_local.py` —
  `0905c274d4e134ac22dde4f01dee66961eb16952c5eb8c1d22f56dcb7d297ea4`;
- `src/finalize_s16_phase_c_labels.py` —
  `057ef3f7eec3724b713a87b9457075d206abfc7ed8c1f2fd3078b9745ecfeae8`;
- `src/label_second_rater.py` — 75 lines of annotation tooling,
  `ca1314ca046a6ab4aeaab3ffea014e7a6ca36e5bea9a1881b325376b989ccdc9`;
- `src/unblind_second_rater.py` — 107 budgeted lines of annotation/audit
  tooling, `fdd3e473e38637ea7286097c8df9cae2921b6878c136f7bc31378f69bfe71aeb`.

The two audit-tool additions are charged as the requested `75 + 107 = 182`
annotation-tooling lines so they are not hidden from the parent code budget.
For conservative physical-file accounting, `unblind_second_rater.py` currently
has 20 additional report-rendering/entrypoint lines; `wc -l` across the five
Phase-C source files is 1,000, exactly at the parent ceiling. The
runner-specific ceiling is amended from 500 to 550 lines; the runner is 550.

Focused pre-freeze checks parsed the runner, expanded a synthetic 204-row panel
to the exact per-arm counts in §4, and checked the random-ablation edit's
per-position L2 norm. No model was loaded, no scientific record was generated,
and no broad test matrix, repeat smoke, clock test, wait, or delay was added.

## 7. Exact continuation authority

After this file is hashed, the only sentence that authorizes Tier 1 is:

> Authorize continuation of the local Phase C run under PHASE_C_PREREGISTRATION.md sha256 25928a120f53a277553a15a6f271463b990f52f186653773f117507ff0a6f13d as amended by PHASE_C_AMENDMENT_01.md sha256 `<full amendment hash>`: binary agreement gate, reuse of the 204 baselines, tier 1 arms, blind labeling.

That sentence authorizes resumable local execution of only the 816 Tier-1
records in §4 and the blind labeling in §5. It does not authorize the 636
deferred arms, baseline regeneration or relabeling, a judge model, backward
passes, remote execution, sealed-S10 confirmation access, Phase D, or applying
Research State.
