# E9d first-forward hook portability erratum

Status: **stopped, repaired locally, and awaiting research-state reconciliation plus fresh exact authorization**

Date: 2026-08-24

## Basic version

The authorized E9d v2 run did not produce a scientific result.

The exact 26 approved files were uploaded to one ephemeral
`NVIDIA A100-SXM4-40GB`. The no-forward probe passed: it authenticated every
uploaded hash, the exact proposal and receipt, the revised E9c plan, the A100,
and all pinned package versions. The tokenizer and `Qwen/Qwen3-4B` model then
loaded successfully.

The first baseline attempt failed because two pieces of Python code used
different names for the same callback argument. TransformerLens `3.2.1` calls
a hook like this:

```python
callback(value, hook=self)
```

The frozen E9c callback expected the second argument to be named
`hook_point`. Python therefore raised:

```text
TypeError: _capture_positions.<locals>.hook() got an unexpected keyword argument 'hook'
```

This happened before the first baseline row was completed. The preserved
checkpoint records:

- `0` completed scientific forwards;
- `2` prior audit attempts plus `1` in-flight attempt, for `3`
  conservatively charged attempts under ceiling `291`;
- `0` prompt rows, `0` sender rows, and `0` route rows;
- no private receiver-tensor bundle;
- no preregistered analysis and no scientific result.

Generation, backward computation, E7d, confirmation access, new scientific
cells, and outcome-based replacement did not occur. The A100 was terminated,
and Colab reported no active sessions.

The local repair only renames the second parameter in four already-registered
hook callbacks from `hook_point` to `hook`. It does not change what any hook
does, where it fires, which tensor it reads or writes, the prompts, model,
dose, receivers, endpoints, route cells, analysis, or forward counts. A new
regression reproduces TransformerLens's keyword invocation and passes after
the repair.

Because the runner and execution-plan hashes changed, the v2 proposal and
receipt cannot be reused. They are archived exactly under
[`outputs/invalidated/`](outputs/invalidated/README.md). No further upload,
Colab allocation, model load, or forward is authorized.

## Academic version

### Authenticated execution boundary

The user authorized E9d proposal payload
`c4ecf588af10f9534d66f5da2e7b847917e491d549b7b2f14699bfd82df7db62`.
The materialized single-run receipt had payload/file SHA-256 values
`9c193738e1726e9276a42391b6a8cc5c96ae78de01c281328ba510a934b4d96c` /
`cd81531b272b750b669ef0d643f20743fb13431dc508bc52e3233429d900e226`.

The [v2 runtime probe](outputs/s9_e9d_runtime_probe_v2_hook_keyword_failure.json)
authenticated all 26 transferred files—including exactly the
two approved private inputs—on `NVIDIA A100-SXM4-40GB`. Its payload/file
SHA-256 values are
`db75f18f39482d470102120bbf4037c7f42b73ec6a4ebc42f3e239995a4b9d7f` /
`17a7777f296a3fef81394ef46e6084e2e07f52094ea3b8b5a179303aab538c18`.
The probe itself parsed no private prompt JSON, loaded no tokenizer/model, and
executed no forward, generation, or backward computation.

### Failure mechanism

The execution runner replaces each native TransformerLens MLP with
`ExactHookedMLP`, preserving the native forward mapping while adding hook
points. The first baseline calls `model.run_with_hooks` and registers
`_capture_positions` at `blocks.32.mlp.hook_out`.

In pinned TransformerLens `3.2.1`, `HookPoint` dispatches registered callbacks
using the keyword call `hook(module_output, hook=self)`. The frozen E9c closure
had signature
`hook(activations: torch.Tensor, hook_point)`. Keyword binding therefore fails
independently of tensor values or model outcomes. The traceback is preserved in
[`outputs/s9_e9d_v2_first_forward_hook_failure_colab_session.jsonl`](outputs/s9_e9d_v2_first_forward_hook_failure_colab_session.jsonl),
SHA-256
`b7119c4537db863d9f41e609deb648f8f1442b1adf2e7db4169c6fd69bb84450`.

The authenticated checkpoint
[`outputs/s9_e9d_v2_first_forward_hook_failure_checkpoint.json`](outputs/s9_e9d_v2_first_forward_hook_failure_checkpoint.json),
SHA-256
`43b124b5a87199dbc95099bb2559500d800817c9cd84f23d17d6b3f57a49dcf9`,
has `completed_scientific = 0`, `audit_forwards = 2`, one in-flight operation,
and `cumulative_total_across_resumes = 3`. The accounting is deliberately
conservative: it charges an interrupted attempt even though no row was
committed. There are no outcome metrics from which to interpret route
reliability.

### Minimal repair and invariance audit

The repair renames the callback parameter in exactly four closures:

1. `_source_edit_hook`;
2. `_capture_positions`;
3. `_capture_final`;
4. `_replacement_hook`.

Replacing every `hook_point` token in the archived v2 runner with `hook`
produces the repaired runner byte-for-byte. No other source difference exists.
The repaired runner SHA-256 is
`a547b31b7302f30adb81efa2c088cafb2e220ad078d4fde44e6aceee8ef883e8`.

The plan contract was versioned to
`s9-e9c-cpu-runner-execution-plan-v3`, yielding contract SHA-256
`a356fca9c1ab0c82d2ed778b259de38113bbbaf17415bfad4ddc1ac09ec812ef`.
The revised plan has payload/file SHA-256 values
`8c8979f6d75c36ec91447185d5dcc898f3c2de8b2c6842d1993d652bd2552c10` /
`affac25e379adaf1b89e7860bbdc3ebde8264d53512b47d8cfc28139f5f8564f`.

Relative to the v2 plan, the only input hashes that changed are the runner and
E9c contract. After normalizing those two hashes plus plan version/timestamp
fields, the plans are byte-identical. The following remain exactly unchanged:

- the 24 prompt baselines, 48 sender worlds, and 96 route cells;
- both intervention arms and the fixed BF16 edit identities;
- receiver layers, token positions, restoration/rescue semantics, and
  endpoints;
- runtime/model snapshot and package versions;
- preregistered analysis hierarchy and exclusions;
- 264 planned scientific forwards and ceiling 291;
- all privacy and forbidden-operation boundaries.

The machine-readable erratum is
[`outputs/s9_e9d_first_forward_hook_portability_erratum.json`](outputs/s9_e9d_first_forward_hook_portability_erratum.json),
with payload/file SHA-256 values
`b8ea9f8cdc17f9f29b77b82b3c51b6ffdd21858c784cd420c87a48bd78fffb4f` /
`b25a97b2afeb730fa16a62f10f1db28865cd1ced54a4a426abae0e44aa443345`.

### Evidence boundary and next authority gate

This attempt provides runtime-compatibility evidence only. It provides no
route-effect estimate, no harmful-versus-benign comparison, and no evidence
for or against Q-002.

Before another E9d execution, the canonical research-state priority must be
reconciled to the v3 plan. A fresh execution proposal must then bind the new
plan and source hashes, and the user must explicitly authorize its exact
payload, private transfer, Colab connection, A100 allocation, model loading,
and forward budget. The consumed v2 receipt confers no authority on the
repaired bytes.
