# S9 E9a — project-untouched fixed-dose route-selectivity design

Status: **CPU-only design decision and semantic protocol frozen; private prompt
panel, execution bundle, model work, and GPU allocation are not authorized**

Frozen: **2026-08-23 (America/Los_Angeles)**

## Basic version

### Decision

Proceed to a stronger preregistered study design, but do not execute it yet.
E8b already answered the narrow route question well: changing C repeatedly
affected the next-token refusal score partly through the complete layer-32 MLP
output in all ten harmful intents. The most valuable unresolved question is now
whether that effect is meaningfully larger on harmful prompts than on benign
prompts that receive the same nonzero edit.

This follow-up has high information value because it targets the two exact
limitations that E8b could not resolve:

1. E8b reused prompts that already had prior target-model behavior and score
   outcomes. The new panel must be untouched by the target model within this
   project before the frozen execution.
2. Every E8b benign C edit had zero natural dose. A zero intervention cannot
   show that benign prompts are spared by a nonzero intervention.

The decision advances only to private panel construction and review. It does
not authorize prompt upload, Qwen loading, model forwards, generation, Colab,
GPU allocation, E7d, or runner implementation.

### The key design correction

The phrase **genuinely model-unseen** is too broad. We cannot establish that a
pretrained model never encountered an intent or similar language during
training. The protocol therefore uses the narrower and auditable term
**project-untouched target-model panel**:

- the exact prompt bytes have received zero prior Qwen3-4B calls in this
  project;
- no generation, score, activation, graph, dose screen, route outcome, or
  endpoint has been computed on those bytes;
- prompt authoring and metadata-only hashing may occur locally before the run;
  and
- no claim is made about pretraining exposure.

Natural source-to-target C patches cannot guarantee a nonzero benign dose
without first running the target model on the new prompts. That would break the
strict untouched boundary. E9 therefore freezes one nonzero numeric edit before
the new prompts exist.

### Fixed nonzero dose

The ten E8b harmful natural C activation differences were:

```text
19.875, 14.375, 19.625, 17.625, 17.625,
20.500, 21.750, 16.750, 15.125, 21.500
```

Their median is exactly `18.625`. E9 applies

```text
+18.625 * d_C
```

at the final-position layer-31 MLP output of every harmful and benign prompt,
where `d_C` is the decoder direction for `L31:F49541`. Every prompt therefore
receives the same positive, nonzero C decoder write. The comparator remains
`L31:F139261`, scaled to the same nominal residual-write L2 norm.

This solves the dose problem without inspecting the new prompts. It also
changes the sender estimand: E9 is a **fixed-dose confirmatory extension**, not
a literal replication of E8b's natural D0-to-F0 activation patch. That
difference must remain visible in every later report.

### Planned panel and route test

The next private panel must contain `12` new topic-matched pairs:

- `12` harmful intents;
- `12` benign intents;
- one target prompt per intent;
- new intent categories, reviewed for semantic near-duplicates;
- a prompt template family disjoint from E8b's D0/F0 templates; and
- no outcome-based replacement or sequential top-up.

The exact prompt bytes do not yet exist and are not authorized for
materialization in this stage. When separately authorized, they remain private
and only public IDs, metadata, and hashes may appear in public artifacts.

For both C and its equal-L2 comparator, the empirical receiver is the complete
final-token output of the layer-32 MLP. The same complete module output at the
penultimate token is retained only as a wrong-time causal-order and runtime
audit. It is not an empirical off-path receiver. Graph-low-coverage MLPs are
not used as physical controls because E7c showed that low graph coverage did
not make their whole-module effects weak.

### What would count as success

The independent units are harmful intents for route replication and matched
topic pairs for selectivity. Route cells are repeated measurements, not extra
samples.

The next-token refusal log odds remains primary; the probe is secondary. The
hierarchy requires:

1. at least `10/12` harmful intents pass the signed dependence rule;
2. at least `10/12` harmful intents then pass the signed rescue rule;
3. at least `10/12` topic pairs have larger comparator-adjusted dependence and
   rescue on the harmful member; and
4. in at least `10/12` pairs, both benign comparator-adjusted route effects are
   at most one quarter of the corresponding harmful magnitude.

Ten or more positive signs out of twelve has an exact one-sided sign-test tail
probability of `79/4096 = 0.019287109375` under a fair-sign null. The fixed-count
rules remain primary; route-cell pooling is forbidden.

Even a complete pass supports only fixed-dose endpoint-level selectivity on
this panel. It does not establish generated-answer behavior, benign behavioral
sparing, practical equivalence, a refusal switch, unique mediation through D,
a dominant pathway, or graph-topology value.

## Academic version

### Research question and information-rate decision

Let `C = L31:F49541`, `M = L31:F139261`, and let the receiver be the complete
vector at `blocks.32.mlp.hook_out` for the final prompt position. E8b supplied
strong prospective evidence for harmful-intent receiver dependence and rescue,
but its reused cohort and degenerate benign doses left the selectivity component
of `H-005` unresolved.

The expected information gain of a fixed-dose matched-risk design is high:

- a positive result would extend the bounded complete-module phenotype to a
  project-untouched, template-disjoint panel and establish a preregistered
  internal-endpoint risk differential;
- a negative result would preserve route dependence while rejecting the
  selective component under a controlled nonzero dose; and
- either result would be more discriminating than another D-gate-centered
  replication, topology regression, or generated-answer branch at the current
  evidence boundary.

The alternative of stopping and distilling E7/E8 remains scientifically valid,
but it leaves `Q-002` unresolved despite a tractable design that directly fixes
the observed dose failure. The alternative of screening new prompts for a
nonzero natural C delta was rejected because it would consume target-model
activations before the confirmatory execution and make the strict freshness
label false.

### Sender estimand

The development-calibrated fixed activation delta is

```text
delta* = median({delta_C(i): i in E8b harmful intents})
       = (17.625 + 19.625) / 2
       = 18.625.
```

For every future prompt `i`, the nominated sender world adds

```text
u_C = delta* d_C
```

to the final-position layer-31 MLP output. The control world adds

```text
u_M = delta_M d_M,
delta_M = ||delta* d_C||_2 / ||d_M||_2,
```

with a positive sign. Thus the nominal decoded residual-write norms match,
while semantic direction, baseline activity, local Jacobian, and downstream
reach remain intentionally unmatched. The equal-L2 comparator is a magnitude
control, not a sender-total or route-specificity control.

Because `delta*` is independent of each new prompt's natural C activation,
E9 estimates the response to a standardized intervention. It does not estimate
the effect of transporting the prompt's own natural counterfactual activation.

### Receiver worlds and estimands

For risk member `r` of topic pair `j`, sender arm `e in {C, M}`, and receiver
position `p`, let:

```text
Y0(j,r)          target baseline
YS(j,r,e)        fixed sender-edited target
YS,R0(j,r,e,p)   sender world with complete receiver restored to baseline
YR(S)(j,r,e,p)   baseline world with complete sender-world receiver transplanted
```

The signed estimands remain:

```text
T_S = YS - Y0
D_R = YS - YS,R0
B_R = YS,R0 - Y0
Q_R = YR(S) - Y0
T_S = D_R + B_R.
```

For `E in {D_R, Q_R}`, define comparator-adjusted route effects

```text
S_E(j,r) = E(j,r,C) - E(j,r,M)
```

and matched risk differences

```text
Delta_E(j) = S_E(j,harmful) - S_E(j,benign).
```

Restoration and rescue remain different hybrid counterfactuals; neither is
called a mediated percentage. The penultimate receiver is the same module and
tensor boundary at a causally earlier token. A nonzero penultimate result is a
semantic or runtime failure, not negative mechanism evidence.

### Frozen hierarchical decision rule

For each harmful intent, dependence wins when

```text
T_S(C) > 0 and D_R(C) > 0 and S_D(harmful) > 0.
```

The dependence gate requires at least `10/12` wins. Only after it passes,
rescue wins when the dependence rule holds and

```text
Q_R(C) > 0 and S_Q(harmful) > 0,
```

again in at least `10/12` harmful intents.

Only after both harmful route gates pass is risk selectivity evaluated. It
requires `Delta_D > 0` and `Delta_Q > 0` separately in at least `10/12` topic
pairs. The relative-smallness diagnostic additionally requires, for both
`E = D_R` and `E = Q_R`,

```text
|S_E(benign)| / |S_E(harmful)| <= 0.25
```

in at least `10/12` pairs. A nonpositive harmful denominator is a non-win, not
a removable or undefined pair. All 24 intents remain in their denominators.

The `0.25` rule is a preregistered relative practical bound, not a formal
equivalence test. It cannot license benign behavioral sparing because the
experiment has no generated-answer endpoint.

### Planned forward accounting

The panel protocol implies:

| Component | Count |
|---|---:|
| Target baselines | `24` |
| C and equal-L2 sender worlds | `24 x 2 = 48` |
| Two receiver positions x restoration/rescue | `24 x 2 x 2 x 2 = 192` |
| **Scientific total** | **`264`** |
| **Fail-closed audit ceiling** | **`291`** |

These counts are a future execution budget, not authorization. The eventual
panel manifest must materialize every planned intent and cell before any model
outcome. Missing rows, replacements, dose searches, new arms, or sequential
top-ups invalidate rather than enlarge the frozen design.

### Stop rules and claim boundary

| Outcome | Licensed disposition |
|---|---|
| Harmful dependence fails | The fixed-dose complete-receiver phenotype does not replicate stringently; stop the selectivity hierarchy. |
| Dependence passes but rescue fails | Dependence repeats, but transplant sufficiency does not; do not claim the full phenotype. |
| Harmful route gates pass but risk differential fails | Preserve nonselective route dependence and reject endpoint selectivity under this dose. |
| Differential passes but relative-smallness fails | Report a directional risk difference without calling benign effects small. |
| All gates pass | Support bounded fixed-dose endpoint-level complete-receiver selectivity on this frozen panel only. |
| Structural audit or integrity fails | Invalidate and debug; do not interpret scientific outcomes. |

D crossing is omitted from selection and advancement. E7d, graph-topology
regression, receiver-input intervention, generation, and backward computation
remain closed.

## Frozen artifact

The machine-readable design is
[`outputs/s9_e9a_design.json`](outputs/s9_e9a_design.json):

- canonical payload SHA-256:
  `0f3c01580c8a8837fe5a1819ea63c041ec5829f5acda24571f5ff6132aaf70f8`;
- file SHA-256:
  `dfb1f1ad5de297997301a74022d38d27421acc772426d4314eebc9996a777788`.

The CPU-only implementation files are:

- contract SHA-256:
  `0b66ad0715d81f95f5ae18e3536183e7e58de6a6e86c1b66ee8654b6403d4b67`;
- builder SHA-256:
  `f22ab4b39e4d6871c422fbc6c10d5fdeddbac9b235abf3eadef95ca0d1766a3c`;
- test SHA-256:
  `b8a90044ca7c490239507894184043436f88073289e88880ac98fb7b06191681`.

## Authorization boundary and next stages

| Action | Authorized? |
|---|---|
| CPU-only semantic contract and design artifact | **yes; completed** |
| Private prompt-panel materialization | **no; separate next stage** |
| Target-model screening of candidate prompts | **no; prohibited by this design** |
| Runner/runtime-bundle implementation | **no** |
| Prompt upload or external transfer | **no** |
| Qwen load or any model forward | **no** |
| Colab connection or GPU allocation | **no** |
| Generation, E7d, topology regression, or backward pass | **no** |

If the user authorizes the next local stage, E9b may construct the private
24-intent panel and a public hash-only manifest under this protocol. It may not
query the target model. E9c runner/runtime work and any later exact GPU run each
require a fresh, separately reviewable authorization boundary.
