# S9 E8a — C→D intent-disjoint route-replication preregistration

Status: **CPU-only scientific manifest and plan frozen; E8b runner, model
execution, and GPU allocation are not authorized**

Preregistered: **2026-08-23 (America/Los_Angeles)**

## Basic version

E7c found three developed cases where changing feature C at layer 31 affected
the next-token refusal score partly through the complete layer-32 MLP output.
Those cases were useful, but they came from only six intents that had already
been used for method development. E8 asks whether the same fixed C→D phenotype
reappears in different intents.

E8a uses the 15 S10 **validation-expansion** intents, `H31–H40` and `B15–B19`.
These intents are disjoint from the six E7c intents and have never received an
E7 route intervention or route outcome. They are therefore
**route-outcome-unseen and intent-disjoint from E7c**. They are **not pristine
or model-unseen**: S10 already generated answers, computed refusal-target
scores, and manually annotated those answers. This is a prospective test of a
fixed route phenotype on reused contexts, not a pristine model-evaluation
sample.

The separate S10 confirmation partition stays sealed. E8a does not inspect its
prompt text or run any confirmation graph, activation, score, or intervention.

For every expansion intent, E8a fixes the direct `D0` prompt as the activation
source and the fictional `F0` prompt as the target. It compares two layer-31
sender edits:

1. the developed C feature, `L31:F49541`; and
2. the sender comparator `L31:F139261`, scaled to have exactly the same decoded
   residual-write L2 norm as the C edit in that intent.

Each sender edit is tested against the complete layer-32 MLP output at two
target positions. The final-token receiver is the empirical route test. The
penultimate-token receiver is only a causal-order and runtime audit: a
final-token layer-31 edit cannot travel backward to an earlier token at layer
32 in a causal transformer. Feature `L32:F50401`, called D in the developed
C→D relation, is recorded only as a gate-crossing diagnostic. It is never the
restored or transplanted tensor.

There are exactly `15 × 2 × 2 = 60` route cells. Every cell runs even when D
does not cross its activation gate. The plan requires `180` forwards and has a
fail-closed ceiling of `198`. It generates no text.

The decision endpoint is the same signed next-token refusal log odds used in
E7c. Receiver dependence, `D_R`, is evaluated first. Receiver rescue, `Q_R`,
is evaluated second and cannot substitute for failed dependence. A stringent
replication requires at least 9 of the 10 harmful intents to show a positive C
sender effect, positive receiver dependence larger than the signed equal-L2
comparator dependence, and positive rescue larger than the signed comparator
rescue. The five benign intents and five topic-matched harmful/benign pairs are
specificity diagnostics, not alternate ways to pass.

## Data boundary and adaptivity disclosure

| Cohort | Intent IDs | Status before this preregistration | E8a use |
|---|---|---|---|
| E7c development | `B01`, `B04`, `H01`, `H03`, `H07`, `H09` | Exact route outcomes known | Excluded |
| S10 validation expansion | `B15–B19`, `H31–H40` | Generation, target scores, and annotations known; no route outcomes | Complete E8a panel |
| S10 confirmation | `B06`, `B08`, `B10`, `B12`, `B14`, `H12`, `H14`, `H16`, `H18`, `H20`, `H22`, `H24`, `H26`, `H28`, `H30` | Model outcomes remain zero under the S10 boundary | Sealed and excluded |

The fixed C feature, D marker, equal-L2 comparator, receiver module, direction,
endpoint, and replication rule were chosen after E7c. Cohort-level S10 outcomes
were also known to the designers, although no record-level S10 outcome was used
to include, exclude, or tune an E8a row. E8a can therefore test whether that
developed phenotype travels to new intents; it cannot estimate an unbiased
discovery rate for arbitrary graph-selected routes.

The expansion prompts reuse the S10 `D0/F0` frame templates. Intent and prompt
content are disjoint from E7c, but the lexical frame families are not. Any
positive result is generalization across intent content conditional on these
two frames, not generalization across arbitrary prompt transformations.

## Frozen 15-intent panel

No prompt text belongs in the public plan or result. The following IDs and
hashes pin the only prompts eligible for E8a.

| Intent | Risk/category | D0 source key | D0 SHA-256 | F0 target key | F0 SHA-256 |
|---|---|---|---|---|---|
| `B15` | benign / `food_allergen_prevention` | `S10BX-B15-D0` | `33c5db9ca5bccf85cf14a8fb9cd255a0965829a2f78d2c332a3c20423ebea355` | `S10BX-B15-F0` | `c5bccc21342c5dc5ea4a894e1d2920c1ccda9fa50a2ecf733665e4e5fab6ae19` |
| `B16` | benign / `evidence_preservation` | `S10BX-B16-D0` | `1a9a61a62573273b29ae7e9cac77ed733a5998ec936eb768a757e1f4a1a3cefb` | `S10BX-B16-F0` | `407f6040ad6cc1f0017c41f3e68ea5b3806de7c6f57a54c7a8a92fbad17c22d6` |
| `B17` | benign / `identity_theft_prevention` | `S10BX-B17-D0` | `f11b1375319e3f17b638fce4dcb282ec5ebd376c21e5ddec782293c6bd09ba81` | `S10BX-B17-F0` | `65dbd5884725e4dd5631317d246476c33887a7c4206de4ed60c5cbd165424acc` |
| `B18` | benign / `industrial_interlock_audit` | `S10BX-B18-D0` | `a291bd71d2dfc59f2359cb0c73f0241eb750b5772ff266625015816327cb38c1` | `S10BX-B18-F0` | `ae6804832d230fe463a38b5be905abece3f3f01a8f84eceadffe049c020ed1aa` |
| `B19` | benign / `legal_document_verification` | `S10BX-B19-D0` | `9b01f162c9b95d2189416b861f1f25d6c49252f4e90920827188a80c488149c0` | `S10BX-B19-F0` | `7b03b749eaefd37c51cde390a967b9708e66e170eaa79c8e4dcf2b1c05517962` |
| `H31` | harmful / `food_supply_contamination` | `S10BX-H31-D0` | `db62f480a35f511cc3ce4735be3cc960991d3198c3c7889404ad28f5d7191995` | `S10BX-H31-F0` | `af35617ef8806b5698811d8b0cd0e342896eec84ffba56f2a6d35389ba1f3241` |
| `H32` | harmful / `criminal_evidence_destruction` | `S10BX-H32-D0` | `5022d174a6cb6f0824627b54b9ff1037148f073a4c04e2c94a503d4e9b5b1a8e` | `S10BX-H32-F0` | `f3addbc4637bd0adde41b62ef6788e4c1299c595033b5e3bdc90ff1461507e85` |
| `H33` | harmful / `identity_based_credit_fraud` | `S10BX-H33-D0` | `07c99c028fe0d5a9e06c91a82abfc688930cd808bb99d88b5723b48c952bc1dd` | `S10BX-H33-F0` | `e66ccdd4f6e2be3cc7c8b9863e065d2110d1bbf35910ad7ca9dc4074523e8998` |
| `H34` | harmful / `industrial_safety_sabotage` | `S10BX-H34-D0` | `ca7ee6dd1db22a3cbb5f34935a02c88312ba1dd92269bf5eb52fcce19af99da8` | `S10BX-H34-F0` | `f9e30025d99259463d2385b3101922bab5132591d873fbe19595c59ad78685f1` |
| `H35` | harmful / `forged_court_orders` | `S10BX-H35-D0` | `a9cd5870dfcbfa695efc2a3247188184c11564e8e797d8cda927c9fb0f2e9d1f` | `S10BX-H35-F0` | `6bee2f08663de774f369d7ca237710453682d123a4bcdc7251d818138d607fbf` |
| `H36` | harmful / `companion_animal_poisoning` | `S10BX-H36-D0` | `45ad41d8d19c89a86c1af78b0b3d04020febe69071af4a129b243d9ff53cb73f` | `S10BX-H36-F0` | `045e162f9a67b77d597bbd3ef4fc3061ae5c155829fabc433eddd66dba5fb3fc` |
| `H37` | harmful / `emergency_dispatch_disruption` | `S10BX-H37-D0` | `5e52cad3e19a5e2781c8e6c5ae75a1c9b7c36a894ed8268e6fe4b58e36f50ca5` | `S10BX-H37-F0` | `be6d7138a6f4b0e178735a6583753c939d690d57c2344050a3482d754cac12bd` |
| `H38` | harmful / `disaster_relief_charity_fraud` | `S10BX-H38-D0` | `ba432e324c68c8f114b157f402bd7ecc6fb2a8f02076a72e8195eaafb2dab8e2` | `S10BX-H38-F0` | `d70973f7797d99feb5d42a0379a941d4aba8c7d7495208137a5571aadba55f6d` |
| `H39` | harmful / `aviation_maintenance_sabotage` | `S10BX-H39-D0` | `b4bd06e11cbafe60b9c4477149abd3da9338cdeed3004a550e7efbad7a481b89` | `S10BX-H39-F0` | `63bfe579b6ea94dc976939b2db7d384957f336897fb10aa6cd5aa8c02b4553ae` |
| `H40` | harmful / `witness_intimidation` | `S10BX-H40-D0` | `c1d0640b13dc19079cffbcc1a2fee2e1c579bb02089db0f76d137acb471ace88` | `S10BX-H40-F0` | `7c4dee9ab4de83e015ecbdc3e2a3dc46e90b7fa9027c521d0ad7ed998d6f7c41` |

All 15 intents and all 30 prompt rows remain in the panel. There is no
replacement, score-based filtering, sequential stopping, or post-runtime
selection.

## Academic version

### Research question

For a fixed final-token sender intervention on transcoder feature C,
`L31:F49541`, does the signed next-token refusal-log-odds effect depend on and
become partly reproducible through the complete final-token output of the
layer-32 MLP across route-outcome-unseen harmful intents? Does this dependence
exceed that produced by an exact equal-L2 sender edit on `L31:F139261`?

This is a test of a fixed operational receiver-dependence phenotype. It is not
a new graph-selector benchmark, a test of unique feature mediation, or a claim
that C or D is a refusal switch.

### Frozen sender operation

For intent `i`, let the cached final-token C activations on the D0 source and F0
target be `a_C(D0,i)` and `a_C(F0,i)`. The primary sender dose is the full
natural counterfactual difference

```text
delta_C(i) = a_C(D0,i) - a_C(F0,i).
```

The target F0 run receives the decoded residual-write edit
`delta_C(i) * d_C` at the final-position layer-31 MLP output, where `d_C` is
the decoder direction for `L31:F49541`. No clipping, thresholding, or
gate-conditioned rescaling is allowed.

For the comparator feature M = `L31:F139261`, freeze

```text
delta_M(i) = sign(delta_C(i))
             * ||delta_C(i) d_C||_2 / ||d_M||_2.
```

Thus `||delta_M(i)d_M||_2 = ||delta_C(i)d_C||_2` within numerical tolerance.
If `delta_C(i) = 0`, both edits have zero dose and the intent still runs and
remains in the denominator. The comparator controls decoded edit magnitude; it
does not match semantics, baseline activity, local Jacobian, downstream reach,
or dose response.

### Frozen receiver worlds and estimands

For each intent `i`, sender edit `e ∈ {C, M}`, and receiver position
`p ∈ {final, penultimate}`, let `Y` be the unchanged E7c next-token refusal log
odds on the F0 target:

```text
Y0(i)          F0 target baseline
YS(i,e)        sender-edited F0 target
YS,R0(i,e,p)   same sender edit with complete L32 MLP hook_out at p
                restored to its F0-baseline value
YR(S)(i,e,p)   F0 baseline with complete L32 MLP hook_out at p
                replaced by its sender-world value
```

The complete vector at `blocks.32.mlp.hook_out` is the receiver tensor. The raw
estimands are

```text
T_S(i,e)     = YS(i,e) - Y0(i)
D_R(i,e,p)   = YS(i,e) - YS,R0(i,e,p)
B_R(i,e,p)   = YS,R0(i,e,p) - Y0(i)
Q_R(i,e,p)   = YR(S)(i,e,p) - Y0(i).
```

Every route record must report signed `T_S`, `D_R`, `B_R`, and `Q_R`, plus the
absolute invariant error for `T_S = D_R + B_R`. Ratios such as `D_R/T_S` are
secondary diagnostics only and cannot enter a gate.

### Receiver positions and D diagnostic

The two receiver positions have different roles:

1. **Final token — empirical receiver.** This is the only position entering
   the replication rule or topic analysis.
2. **Penultimate token — structural audit only.** Under causal attention, a
   final-token layer-31 edit cannot change an earlier token's layer-32 MLP
   output. The cached baseline and sender-world penultimate receiver vectors
   must therefore agree in the authenticated deterministic runtime. Its
   restoration and rescue effects must be no-ops. A violation is a hook,
   indexing, caching, or runtime failure; a successful no-op is not evidence
   for an empirical off-path control.

At both positions, save the preactivation and post-ReLU activation of the D
marker `L32:F50401` in the baseline and sender worlds. Classify the final-token
diagnostic as:

```text
inactive_to_active: pre0 <= 0 < preS
active_to_inactive: preS <= 0 < pre0
active_stable:      pre0 > 0 and preS > 0
inactive_stable:    pre0 <= 0 and preS <= 0.
```

D is diagnostic metadata only. It is never patched, restored, rescued, used to
select an intent, or used to decide whether a cell executes. All 60 cells run
regardless of crossing class. Crossing-stratified summaries are post hoc
descriptions of the frozen complete panel and cannot replace the all-intent
gate.

### Frozen cell and forward accounting

| Component | Count | Purpose |
|---|---:|---|
| D0 and F0 prompt baselines | `30` | Cache source activations, target endpoint, and receiver states |
| Unique F0 sender worlds | `15 × 2 = 30` | C and exact equal-L2 comparator edits |
| Route restoration/rescue worlds | `60 × 2 = 120` | One restore and one rescue for every edit/position cell |
| **Planned forwards** | **`180`** | Fixed complete execution |
| **Audit ceiling** | **`198`** | Ten-percent ceiling for declared runtime/hash diagnostics only |

The 60 cells are `15 intents × 2 sender edits × 2 receiver positions`. The
ceiling does not authorize new prompts, replacement cells, dose searches,
layer searches, feature searches, crossing-conditioned reruns, or generation.
Exceeding `198` model forwards invalidates the run rather than enlarging the
panel.

### Hierarchical primary decision rule

Only final-token refusal-log-odds records enter the decision. For each harmful
intent `i ∈ {H31,…,H40}`, define the dependence indicator

```text
A_i = 1[
  T_S(i,C) > 0
  and D_R(i,C,final) > 0
  and D_R(i,C,final) - D_R(i,M,final) > 0
].
```

The **receiver-dependence gate** passes only if

```text
sum_i A_i >= 9 of 10 harmful intents.
```

`D_R` is evaluated before rescue. If this gate fails, a positive `Q_R` pattern
cannot rescue the route-dependence claim. If it passes, define

```text
R_i = A_i * 1[
  Q_R(i,C,final) > 0
  and Q_R(i,C,final) - Q_R(i,M,final) > 0
].
```

The **full C→D phenotype replicates** only if

```text
sum_i R_i >= 9 of 10 harmful intents.
```

This joint rule allows at most one heterogeneous harmful intent and requires
signed direction plus a positive signed paired advantage over the equal-L2
comparator. Absolute effects and ratios are not primary. The rule is
deliberately more stringent than a positive mean, pooled cell-level test, or
crossing-selected subset. It is a fixed replication rule, not a power guarantee
or a claim that the ten intents sample a universal harmful-prompt population.

Gate evaluation is valid only if all 60 cells and 180 planned forwards are
present, prompt and input hashes authenticate, every route identity satisfies
the frozen numerical tolerance, and the penultimate causal-order audit passes.
Missing or invalid intents are not removed from the denominator.

The frozen numerical gates are: one hook fire per declared capture,
`|T_S-D_R-B_R| <= 1e-7`, pre-replacement receiver mismatch `<= 1e-4`, and
within-intent C/comparator edit-L2 relative error `<= 1e-6`. At the penultimate
audit position, receiver-state maximum absolute change and `|D_R|, |Q_R|` must
each be `<= 1e-4`. Any audit violation invalidates runtime semantics; it is not
negative mechanism evidence.

The separate gate-crossing advancement rule requires at least `8/10` harmful
C edits to produce the nominated inactive-to-active D diagnostic and at least
seven of those crossings to satisfy both dependence and rescue rules. This is
a predeclared decision threshold, not a natural statistical boundary. It does
not decide whether the complete 15-intent panel is technically evaluable, and
it cannot filter which cells run.

### Topic-matched differential analysis

The first five harmful expansion intents have a preregistered benign topic
counterpart:

| Harmful intent | Benign counterpart | Shared topic axis |
|---|---|---|
| `H31` | `B15` | food safety |
| `H32` | `B16` | evidence handling |
| `H33` | `B17` | identity and credit security |
| `H34` | `B18` | industrial safety systems |
| `H35` | `B19` | legal-document integrity |

For estimand `E ∈ {D_R, Q_R}`, first compute the within-intent,
comparator-adjusted C advantage at the final receiver:

```text
S_E(i) = E(i,C,final) - E(i,M,final).
```

Then report all five topic differences

```text
Delta_E(j) = S_E(H_j) - S_E(B_j),
```

their median, and the number with `Delta_E(j) > 0`. A bounded harmful–benign
differential requires `Delta_D(j) > 0` in all `5/5` pairs, with `Delta_Q` only
corroborating. That wording is permitted only if every harmful/benign pair's
nominated edit-L2 ratio is within `[0.8, 1.25]`; otherwise the semantic
differential is `not_evaluable` and all raw results remain. These diagnostics
do not alter the 9/10 harmful gate, and no p-value over five pairs licenses a
confirmatory population claim. There is no preregistered practical-equivalence
margin, so E8a cannot claim benign sparing.

### Dose-confounding caveat

The equal-L2 comparator matches C and M **within an intent**, but the natural C
dose `||delta_C(i)d_C||_2` may differ substantially across intents and between
the harmful and benign members of a topic pair. Raw `D_R`, `Q_R`, and
`Delta_E` can therefore vary because of edit dose, baseline endpoint scale,
receiver-state scale, or local nonlinearity rather than risk class. E8a must
publish each intent's signed activation delta and decoded edit L2 norm and show
outcomes against dose. Dose-normalized values may be descriptive, but they are
not primary, do not replace the raw estimands, and cannot remove this
confounding without a separately randomized or dose-matched design.

## Interpretation and stop rules

| Outcome | Licensed interpretation | Next action |
|---|---|---|
| Dependence and full 9/10 gates pass | The fixed C edit shows stringent complete-L32 receiver dependence and rescue across these route-outcome-unseen harmful intents | Preserve as prospective replication evidence; separately decide whether a larger truly model-unseen study is warranted |
| Dependence passes, rescue fails | The complete receiver matters in the sender context, but transplant sufficiency does not replicate stringently | Do not call the full C→D phenotype replicated |
| Dependence fails | The developed E7c route-dependence phenotype does not meet the preregistered replication rule | Stop this branch and distill E7/E8 as bounded methodology evidence |
| Penultimate audit fails or integrity is incomplete | Runtime semantics are unauthenticated | Invalidate and debug locally; do not interpret route outcomes |

Even a full pass does not establish:

- unique mediation through D or any single sparse feature;
- a dominant, necessary, or behavior-controlling refusal pathway;
- graph topology value beyond AtP or node variables;
- generated-answer refusal or safety behavior;
- template-disjoint or model-unseen generalization;
- a population effect outside the 15 reused S10 expansion intents; or
- permission to run E7d, generation, or a fragility campaign.

E7d remains closed under this preregistration. It requires a separate design,
sufficient independent intents, an incremental baseline comparison, and new
explicit authorization even if E8 later passes.

## Provenance and privacy

| Input/evidence artifact | Relevant frozen fields | File SHA-256 |
|---|---|---|
| `outputs/s10_phase_b_expansion_plan.json` | `.expansion_panel.records[].{prompt_key,intent_id,risk_class,category,partition,frame_id,frame_class,core_sha256,prompt_sha256}` | `9341c387c9a1f409cace2641f3bad7abcd1f6585305b73c1f1fd2b101e842016` |
| `private/s10_phase_b_validation_expansion_intents.json` | `.intents[].{intent_id,risk_class,category,partition}` | `1ccd9a5bf20cecf794eaa8fa0c3d22ef789281ab36a32b4dd8ef21232acd9108` |
| `private/s10_phase_b_validation_expansion_prompts.csv` | IDs, frames, core hashes, and prompt hashes; raw text remains private | `f5f65a41a2ca6cf479a58f9d21e4cfa50becadd28fc033ceeaa3b88eb4bd41e8` |
| `outputs/s10_phase_b_expansion_generation.json` | proves prior model generation on this cohort | `bced1b358bd3349a4edea4218a691762e242c81884e4ff5754673d15f258baae` |
| `outputs/s10_phase_b_expansion_analysis.json` | `expanded_validation_gate_stop`; graphs/interventions remained false | `43d2439192e5c065461c2318252dc9598c97522205e87b65fbda2cbaea7907ea` |
| `outputs/s9_e7b_plan.json` | fixed E7c sender/receiver semantics and endpoint contract | `9ee3ca739caa97937a27d4f23571bd6c3ee6a7757d37669bb03d95efa0447666` |
| `outputs/s9_e7c_result.json` | developed C→D phenotype and disclosed adaptivity | `b5a4a7fd0a2cbe9b904ebb08adcaa56af857a3cb7a4a329abd1deb228bba937b` |
| `src/s9_e8_contract.py` | public privacy, cell, decision, technical, and authorization validators | `6fc5c7ea89b9549e6d558ab7fb5f06f1be5315da07bbceace8c60fc1e38d2253` |
| `src/build_s9_e8a_manifest.py` | metadata-only panel materialization | `93b06c5fc16aad0ce7368964f96edcfec04cafc82df57ac25d86e5c30885e896` |
| `src/build_s9_e8a_plan.py` | formula-defined 60-cell plan materialization | `2f47f0bf63f2f3df699b1a5554417b36005a0387e7a46528f3d73909a5fcd8c9` |
| `outputs/s9_e8a_replication_manifest.json` | 15 intent pairs; sealed confirmation boundary | `110932011bbb087a16bff0b2128b923e545534c9227e12648bfdb9bbf250744a` |
| `outputs/s9_e8a_plan.json` | 60 route cells; 180-forward/198-ceiling future contract | `76a6879af6ac9fd2f2595ab41b275318720adf0d47320455c0b16c112e3503ed` |

Public E8 artifacts must contain IDs, metadata, scalar outcomes, tensor hashes,
and runtime proof only. Prompt text and raw tensors remain private by default.
The exact model revision, tokenizer/chat bytes, endpoint implementation,
transcoder bytes, plan bytes, runner bytes, and runtime package versions must
be hash-pinned in a separately reviewable machine plan before any execution.

## Authorization boundary

At this preregistration boundary:

| Action | Authorized? |
|---|---|
| Local document and CPU-only contract creation | **yes; completed** |
| E8a public manifest and formula-defined plan materialization | **yes; completed** |
| E8b runner/runtime-bundle implementation | **no; fresh confirmation required** |
| Loading Qwen or executing any model forward | **no** |
| Allocating or connecting to a Colab GPU | **no** |
| Sending any E8 prompt text to a model | **no** |
| Generated-answer decoding | **no** |
| E7d or any incremental-topology regression | **no** |
| Inspecting or reusing the S10 confirmation partition | **no** |

No GPU, model load, model forward, generation, E7d analysis, or confirmation
reuse may begin from this document alone. The E8a scientific fields are frozen,
but the E8b runner/runtime bundle is intentionally absent. Execution remains
false until the user explicitly authorizes that separately reviewable stage.
