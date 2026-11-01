# E8b pre-outcome semantic erratum and numerical extension

Status: **frozen before any E8b model load or route outcome**  
Date: **2026-08-23 (America/Los_Angeles)**

This document does not edit or replace E8a. It records how E8b resolves one
internal E8a analysis inconsistency and one previously unstated numerical
detail before execution. The immutable E8a plan and preregistration remain
preserved at their original hashes.

## Basic version

E8a says in two especially explicit places that a scientific win means a value
is literally greater than zero: the displayed equations in the preregistration
and the machine-readable `decision_contract`. A different, generic analysis
block says values at or below `1e-7` are nonwins. That tolerance was also
defined as a bookkeeping check for `T_S = D_R + B_R`, not as a practical-effect
threshold.

E8b therefore uses:

- literal `> 0` for the compound dependence and rescue rules;
- literal `> 0` for the five topic-pair difference counts;
- `> 1e-7` only for the secondary exact sign-test summaries, matching the
  otherwise-conflicting `analysis_contract` convention;
- positive raw nominated `D_R` and `Q_R` medians as a reported consistency
  check. A full 9/10 compound pass already implies these medians are positive,
  so this does not add or remove a primary pass route.

The feature calculations also need a declared numeric convention. E8b matches
the developed E7c operational path: C and D feature preactivations use the
stored BF16 encoder, state, and bias tensors with PyTorch BF16 output semantics.
Decoded edit vectors and equal-L2 norms are then constructed and audited in
nominal float32; the actually applied BF16 edit norms are reported separately.

## Academic version

### Frozen-source inconsistency

The E8a preregistration (file SHA-256
`5875b5cd5b18984a4cd2eb8ea2e1f99cb16f51116ea6eeb2d1cb127a7957a728`)
defines

```text
A_i = 1[T_S(i,C) > 0 and D_R(i,C,final) > 0
          and D_R(i,C,final) - D_R(i,M,final) > 0]

R_i = A_i * 1[Q_R(i,C,final) > 0
                and Q_R(i,C,final) - Q_R(i,M,final) > 0].
```

It likewise requires `Delta_D(j) > 0` in all five topic pairs. The immutable
E8a plan (file SHA-256
`76a6879af6ac9fd2f2595ab41b275318720adf0d47320455c0b16c112e3503ed`,
payload SHA-256
`270167e9e8bb9c35f997e5795529c265e88b57bf138164b2811c6346355b794b`)
repeats those literal inequalities in `decision_contract`. Its separate
`analysis_contract`, however, states
`ties_at_or_below_invariant_atol_are_nonwins` and requires positive raw
nominated `D_R` and `Q_R` medians.

### Resolution hierarchy

E8b freezes the following interpretation before observing outcomes:

1. The displayed estimand equations and the dedicated `decision_contract`
   govern the primary compound count decisions and topic-pair decision.
   Their win threshold is exactly zero.
2. The invariant tolerance `1e-7` remains a technical evaluability threshold
   for `|T_S - D_R - B_R|`. It is not silently promoted to a minimum practical
   effect for the primary decision.
3. To retain rather than discard the contradictory analysis field, the two
   scalar exact sign-test summaries classify values `<= 1e-7` as nonwins.
   These tests are secondary summaries and cannot override the explicit
   compound 9/10 hierarchy. Their p-values are suppressed if technical runtime
   semantics are not evaluable.
4. Raw nominated median positivity is reported as a consistency condition.
   It is logically implied by a full 9/10 dependence/rescue pass for the
   corresponding nominated estimand and creates no alternate success path.

### Numerical extension

- Stored transcoder row dtype: BF16.
- C natural-activation and D gate-preactivation path: BF16 encoder/state/bias
  operands and PyTorch BF16 output semantics, matching E7c. The pinned runtime
  controls the underlying CUDA kernel implementation.
- Nominal decoded C and comparator edit vectors and their L2 norms: float32.
- Applied model residual edit dtype: BF16.
- Nominal equal-L2 relative error is gated at `1e-6`.
- Realized BF16 equal-L2 error is reported but is not a gate.
- A zero natural C dose produces exact zero C and comparator edits, with all
  four route cells retained.

### Claim boundary

This is an outcome-independent semantic repair, not a threshold refit. It does
not change the 15 intents, 60 route cells, 180 planned forwards, 198-forward
ceiling, intervention sites, receiver tensor, endpoint, comparator feature,
sealed confirmation boundary, or excluded arms. If a scientific result depends
materially on the distinction between literal `> 0` and `> 1e-7`, both
classifications must be shown and the E8a inconsistency must remain disclosed.
