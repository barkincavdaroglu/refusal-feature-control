# S9 E9b — private project-untouched panel freeze

Status: **E9b complete; 12 private harmful/benign topic pairs frozen; public
hash-only manifest complete; E9c runner/model/runtime work not authorized**

Frozen: **2026-08-23 (America/Los_Angeles)**

## Basic version

### What step this is

E9a was the design step. It decided to use a project-untouched panel, one fixed
nonzero `+18.625 * d_C` edit on every prompt, and the complete layer-32 MLP
output as receiver. E9b is the next step: create and review the exact prompt
panel locally before any model sees it.

E9b materialized `12` topic-matched pairs:

- `12` harmful intents;
- `12` benign intents;
- `24` exact target prompts; and
- one common structured-planning template, `E9P0`, that is distinct from
  E8b's bare-request `D0` and fictional-dialogue `F0` frames.

The private prompt text remains in the private panel file. The public
[manifest](outputs/s9_e9b_panel_manifest.json) contains only pair and intent
identifiers, topic/category metadata, hashes, and review receipts.

### Frozen topic pairs

| Pair | Shared topic | Harmful intent category | Benign intent category |
|---|---|---|---|
| E9P01 | public procurement | procurement bid rotation | procurement bid evaluation |
| E9P02 | product reviews | coordinated review manipulation | verified customer feedback analysis |
| E9P03 | medical billing | medical billing upcoding | medical billing accuracy audit |
| E9P04 | sports tournament | amateur match fixing | amateur tournament integrity |
| E9P05 | event ticketing | ticket-queue automation abuse | equitable ticket-queue design |
| E9P06 | payroll | systematic wage underpayment | payroll overtime reconciliation |
| E9P07 | rental housing | discriminatory tenant screening | fair tenant screening |
| E9P08 | insurance claims | insurance claim fabrication | insurance claim documentation |
| E9P09 | subscription cancellation | cancellation obstruction | transparent cancellation |
| E9P10 | agricultural subsidies | subsidy misrepresentation | subsidy compliance |
| E9P11 | restaurant reservations | reservation hoarding and resale | fair reservation allocation |
| E9P12 | retail returns | retail return fraud | legitimate return processing |

These labels are public metadata, not prompt text.

### What the review checked

The panel was compared locally against the prior E7/E8/S10 registry:

- `59` prior intent IDs;
- `59` prior intent categories;
- `101` prior core hashes; and
- `366` prior prompt hashes.

The E9b panel has zero collisions in all four sets. All 12 pairs passed manual
topic-match, risk-contrast, prior-near-duplicate, and cross-pair review without
using the target model. The maximum lexical Jaccard overlap between prompts
from different E9b topics is `0.151515151515`, below the frozen `0.35` guard.
Within each harmful/benign pair, the local regex word-count difference is at
most `3`, below the frozen maximum of `4`. This word-count check is metadata
only; it does not invoke the Qwen tokenizer or model.

### What did not happen

No prompt was uploaded or sent to Qwen. No target-model screen, tokenizer
load, runner implementation, model forward, generation, Colab connection, GPU
allocation, backward pass, E7d analysis, or outcome-based replacement
occurred. E9c remains a separate authorization boundary.

## Academic version

### Sampling and freshness estimand

The independent selectivity units remain the `12` topic pairs. Within pair
`j`, one harmful intent and one benign intent share a topical domain and the
same `E9P0` response-format template. The exact UTF-8 prompt bytes are now
fixed before any target-model contact. Consequently, the auditable freshness
claim is:

> zero prior Qwen3-4B calls on these exact bytes within this project.

It is not a claim that the pretrained model never encountered related language
or concepts during training.

The panel is fixed-count and outcome-blind. All 24 intents must remain in later
denominators. No prompt may be screened, replaced, or topped up after a score,
activation, route outcome, or generated answer becomes available.

### Template and matched-risk construction

Every prompt uses one structured-planning template with canonical template
SHA-256
`0899b76a6b8a0f1fd774be41088cb720178d478d0ff830b215dd12d5fcf7c9b5`.
This is neither the exact `D0` bare core nor the `F0` fictional-dialogue frame.
The harmful and benign records have unique intent IDs, unique risk-specific
categories, and one shared pair-level topic category.

The word-count balance guard reduces a simple length discrepancy but does not
make the members semantically identical. Harmfulness necessarily introduces
different action and deception content. Topic matching therefore controls the
domain and response format, not every lexical feature. Later inference must
remain pair-level and cannot attribute a difference uniquely to abstract
"harmfulness."

### Duplicate and privacy audit

The deterministic builder authenticates the current E9a design, canonical
`D-006` and `Q-002`, the E7b/E8a manifests, S1/S6 development metadata, and the
full S10 validation/confirmation plus expansion metadata. It rejects any new
intent ID, category, core hash, or prompt hash found in that prior registry.

Manual non-target-model review supplies the semantic component that hashes and
lexical overlap cannot prove. The `0.35` cross-topic Jaccard guard catches
obvious wording reuse across different topics; the observed maximum is
`0.151515151515`. Neither check demonstrates population representativeness or
semantic equivalence.

The public manifest rejects prompt, core-request, completion, token, chat, and
tensor fields. It records the private source's file and payload hashes without
copying its text.

### Frozen artifacts and hashes

| Artifact | Payload SHA-256 | File SHA-256 |
|---|---|---|
| Private 24-prompt panel | `da3657aba93bafe307d960a5c6c8415185c3925cd8a744dcf281fb082e298356` | `d003b98d170721fedbc58f0dd5f225f7e00261a4d7daeb2ab6ae9803572202fa` |
| [Public hash-only manifest](outputs/s9_e9b_panel_manifest.json) | `a0aad6c7904ff08e05326ffb808c8d2fd8c3d468e927de8285df4dc5596e4f97` | `aeace06bdfdf637fce6e9d6190eb40fffeec28a16d964dd57ad6e0f8b1f9943b` |

Implementation hashes:

- [E9b contract](src/s9_e9b_contract.py):
  `90a7f4f497adf45e6a7bfc2d9e6b8196e534d17311885654632c6a6f1ac3ec7d`;
- [deterministic builder](src/build_s9_e9b_panel.py):
  `696afe54b7da89516cf80729533ecf28baddb3a96ad6c4770d7be49408cddd14`;
- [failing-first tests](tests/test_s9_e9b_panel.py):
  `0ee873a24502366b061f02036ea6884c4a8b8f03ce8cab89747ad01572b51ab1`.

The focused E9b suite passes `11/11`; the complete local S9 suite passes
`336/336`.

## Next authorization boundary

E9c may be considered only after reviewing this hash-frozen panel. A separate
authorization would be required even for CPU-only runner and execution-plan
implementation. Prompt upload, target-model load, model forwards, Colab, and
GPU allocation remain later and separately gated; this report licenses none of
them.
