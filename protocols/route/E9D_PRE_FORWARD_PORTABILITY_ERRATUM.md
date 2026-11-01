# E9d pre-forward portability erratum

Date: 2026-08-23 (America/Los_Angeles)  
Status: A100 attempt stopped before target-model load or any model forward;
science-preserving local repair frozen; prior receipt invalidated; fresh exact
authorization required

## Basic explanation

The authorized E9d run did **not** produce a scientific result. The exact 26
files were uploaded to an authenticated `NVIDIA A100-SXM4-40GB`, and the
no-forward gate verified every uploaded hash, all pinned package versions, and
the authorization receipt. The runner then stopped before loading Qwen because
several diagnostic length calculations were not bit-for-bit identical between
the local Mac and Colab.

The important intervention bytes were identical. Both applied BF16 edit hashes
matched exactly; only derived `torch.norm()` summaries differed, with maximum
relative drift `5.12102885976298e-07`. That is below the already frozen
equal-L2 tolerance `1e-6`. The most direct explanation is platform/version-
dependent floating-point reduction order: the plan was built with PyTorch
`2.7.1` on arm64 macOS, while the frozen runtime uses PyTorch `2.11.0+cu128` on
x86_64 Colab.

The repaired guard therefore treats the exact BF16 tensor hashes—the bytes
actually cast into the BF16 model activation—as authoritative. Derived float32
norm summaries must still be finite, individually satisfy the original dose
contract, and remain within relative tolerance `1e-6` of the frozen summaries.
No prompt, arm, dose record, receiver, endpoint, analysis rule, or forward count
changed.

The first receipt cannot be reused because the repaired runner and plan have
new hashes. The A100 was terminated, and `colab sessions` reported no active
runtime.

## Academic explanation

### Observed failure boundary

The proposal-bound receipt authorized proposal payload
`25ce065a1fde2a784a503675031fff305268cb929f1fa8c1ab5e9d69bdc4ac44`.
The runtime probe authenticated all 26 uploaded files and reported:

- device `NVIDIA A100-SXM4-40GB`, `39.4935302734375` GiB;
- `torch==2.11.0+cu128`, `transformers==4.57.3`,
  `transformer-lens==3.2.1`, `circuit-tracer==0.5.2`, and
  `safetensors==0.8.0`;
- private prompt JSON not parsed by the probe;
- tokenizer/model not loaded and no forward, generation, or backward operation
  executed by the probe.

The runner subsequently failed at its exact dictionary-equality check between
the frozen and runtime `fixed_dose` records. This check occurs after local input
and receipt validation but before `HookedTransformer.from_pretrained`, prompt
tokenization, result creation, or any model forward. Consequently:

- target tokenizer/model loads: `0`;
- scientific forwards: `0`;
- public result: absent;
- preregistered analysis: absent;
- generation, backward computation, E7d, new cells, confirmation access, and
  outcome-driven replacement: absent.

### Exact diagnostic

| Quantity | Frozen Mac summary | Colab summary | Relative drift |
|---|---:|---:|---:|
| Control activation delta | `18.623486016788174` | `18.623481889481816` | `2.2161835621399672e-07` |
| Nominal nominated edit L2 | `18.62409210205078` | `18.62409019470215` | `1.024129725283346e-07` |
| Nominal control edit L2 | `18.624099731445312` | `18.624094009399414` | `3.07238791724052e-07` |
| Realized BF16 nominated edit L2 | `18.622699737548828` | `18.622709274291992` | `5.12102885976298e-07` |
| Realized BF16 control edit L2 | `18.623273849487305` | `18.623281478881836` | `4.096697211982471e-07` |

The nominal equal-L2 diagnostic was `4.096518901133384e-07` locally and
`2.0482596603350523e-07` on Colab; both pass `<= 1e-6`. More importantly, the
actual BF16 edit identities were identical:

- nominated C edit:
  `07c385fbb75102c83e4a7d28e6a49fbad87e0e0413712e4234fa3e5072588b1c`;
- equal-L2 comparator edit:
  `224bbd06d429195f4dbb8792463a13c686578a4ba60cdab4f07582a4acd1bb0a`.

Changing Colab CPU thread counts over `1, 2, 4, 8, 10, 16, 32, 64, 96` did
not change its summaries. This rules out thread count as the immediate remedy,
but it does not uniquely identify which lower-level kernel or version change
caused the reduction difference.

### Repair semantics

The revised runner:

1. validates both fixed-dose records under the unchanged `<= 1e-6` equal-L2
   contract;
2. requires exact equality for feature IDs, the `+18.625` nominated dose,
   freshness flags, and both applied BF16 tensor hashes;
3. bounds the five platform-sensitive scalar summaries at relative drift
   `<= 1e-6`;
4. records the complete portability audit in the public result;
5. retains the frozen plan's dose record as the authoritative description;
6. rejects resume if the runtime portability audit changes.

The revised execution plan has payload
`392f22254bbc07ec69c952e4a2529d71ae7d2308ed2353e61953769600110741`
and file SHA-256
`a0d00b765984d259d93e80eb530aa8a36a799cdfe9ac05db0e7ff03565eda929`.
It still contains 24 baselines, 48 sender worlds, 96 route cells, 264 planned
scientific forwards, and a 291-forward ceiling. All prompt records, sender
worlds, route cells, dose bytes, analysis gates, exclusions, and privacy rules
are unchanged.

## Evidence and artifacts

| Artifact | SHA-256 |
|---|---|
| [Machine-readable erratum](outputs/s9_e9d_pre_forward_portability_erratum.json) | `1d998ffbc2d8641c2a2c30d60880ead10df29e81e7c7fb16a7993c57d0d8ddc3` |
| [Failed-at-preflight Colab log](outputs/s9_e9d_preflight_failed_colab_session.jsonl) | `216ebb72918ab1522dccea1fdffb0f2d3dad872f6753777218424cc136291df4` |
| [Authenticated no-forward runtime proof](outputs/s9_e9d_runtime_probe.json) | `5335cfea5688a5e11594baf2927523fe0e915ea402a27d8cc8d5a030e6a9c9d6` |
| [Revised execution plan](outputs/s9_e9c_execution_plan.json) | `a0d00b765984d259d93e80eb530aa8a36a799cdfe9ac05db0e7ff03565eda929` |
| [Revised runner](src/run_s9_e9c.py) | `d443f4c9a206786e6b6e1a567d8d7b34acfffe81934f6ac5a7bc600545da9cd6` |
| [Revised E9c contract](src/s9_e9c_contract.py) | `d4cd8251301634647cd6e02f2fa08afe19f2d22367f215cb7c67935e3c7ed8ee` |
| [Runner regression tests](tests/test_s9_e9c_runner.py) | `870e871f709b58f02684ffb0facde0e3ae2c856143712cf9261f86bebf095150` |
| [Erratum builder](src/build_s9_e9d_portability_erratum.py) | `c0d83626252d91debd10a4d74fdf84ab2a8616ca09a49a5c069f946b42182ca9` |
| [Erratum tests](tests/test_s9_e9d_portability_erratum.py) | `01758bad2eb1f43776a94594d2915ec5f6a0a824dc2b19fe8874f7c41531c1ec` |

The invalidated v1 plan, proposal, receipt, runner, contract, and runtime probe
are retained under [the invalidated artifact directory](outputs/invalidated/).
The prior executable receipt was removed from the active output path after its
exact bytes were archived.

## Validation and next authority boundary

The regression began with a failing import for the missing portability guard.
The repaired focused E9c plan/runner/erratum suite passes `32/32`. The existing
E9d proposal tests now intentionally fail because they remain pinned to the
superseded v1 plan; they must be rebuilt only after the research-state priority
is reconciled to the revised plan.

No scientific interpretation is possible from this attempt. The next action is
to stage and review a priority-only research-state correction for the revised
plan. After that exact state proposal is applied, a new E9d execution proposal
and receipt boundary must be frozen and separately confirmed. The prior
confirmation does not authorize changed source or plan hashes.
