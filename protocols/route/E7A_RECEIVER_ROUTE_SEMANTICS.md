# S9 E7a — receiver-route semantics contract

Status: **complete locally; no Qwen forward, generation, or GPU allocation**  
Updated: **2026-08-23**

## Basic version

E7a defines exactly what it means to ask whether a sender's effect depends on a
complete receiver module.

The public-safe toy system has the same high-level shape as a transformer
residual block:

```text
sender ──> residual before module ───────────────────────┐
                         │                               │ residual bypass
                         v                               │
                  receiver input                        │
                         │                               │
                         v                               │
                  receiver output ──────────────────────┤
                                                         v
                                                residual after module
                                                         │
                                                         v
                                                      endpoint
```

The receiver module contributes a write, while the residual state also goes
around the module. Restoring the receiver output therefore does **not** erase
the whole sender change: the sender-modified residual bypass remains.

Four runs define the measurements:

1. baseline `Y0`;
2. sender patch `YS`;
3. sender patch with the complete receiver output restored to baseline
   `YS,R0`; and
4. baseline run with the sender-induced receiver output inserted `YR(S)`.

The linear toy produces:

| World/contrast | Value |
|---|---:|
| `Y0` | `3` |
| `YS` | `9` |
| `YS,R0` | `5` |
| `YR(S)` | `7` |
| sender total `T_S` | `+6` |
| receiver dependence `D_R` | `+4` |
| sender remainder `B_R` | `+2` |
| receiver rescue `Q_R` | `+4` |

The remainder is nonzero because the residual bypass still carries part of the
sender change. In a nonlinear version of the same toy, `D_R = 56` while
`Q_R = 40`: restoration and rescue create different hybrid states, so they are
not interchangeable percentages of mediation.

## Academic version

For a scalar endpoint `Y`, E7a implements

```text
T_S = YS - Y0
D_R = YS - YS,R0
B_R = YS,R0 - Y0
Q_R = YR(S) - Y0
```

and fails if the algebraic identity `T_S = D_R + B_R` is violated beyond
floating-point tolerance. This identity is bookkeeping, not evidence that the
system has a unique receiver route plus bypass route.

The sender patch copies only declared sender hooks from a complete
counterfactual cache into baseline roots. Receiver restoration then adds
baseline receiver overrides to that sender-patched run. Receiver rescue instead
copies the sender-induced receiver state into an otherwise baseline run. Every
node downstream of an override recomputes in topological order.

Two receiver boundaries are explicit:

- **module output, primary:** replace the complete downstream module write;
  the module input and residual bypass stay sender-modified during restoration;
- **module input, calibration:** replace the complete input and recompute the
  module output; the residual bypass still stays sender-modified.

For a deterministic one-input module the two operations can yield the same
endpoint, but the hook states are different. The tests verify this state-level
asymmetry rather than treating endpoint equality as semantic interchangeability.

Passing multiple receiver declarations produces one simultaneous receiver-set
intervention. The contract does not call that redundancy: causally ordered
receivers require order-aware interpretation in the original model.

## Required case coverage

| Handoff case | Executable result |
|---|---|
| sender patch plus module-output restoration | `YS,R0` restores the full write while retaining sender input/bypass |
| module-output rescue | sender-induced write is transplanted into baseline |
| input/output asymmetry | boundary hook states differ and remain separately labelled |
| hybrid-state interaction | nonlinear `D_R` and `Q_R` disagree |
| simultaneous receiver set | two outputs restore in one joint world |
| bookkeeping invariant | checked in linear and nonlinear systems |
| inactive receiver | `D_R = Q_R = 0` despite a nonzero sender total |
| off-path receiver | receiver changes but restoration/rescue are endpoint-null |
| changed receiver with no endpoint dependence | an endpoint-connected zero-sensitivity receiver yields `D_R = Q_R = 0` |
| residual bypass | receiver restoration leaves a measurable sender remainder |

Malformed receiver declarations fail closed: receiver names and hook nodes must
be distinct, every hook must exist, the module output must directly consume its
declared input, and senders must precede receivers.

## Implementation and validation

- semantic implementation: [`src/receiver_route_semantics.py`](src/receiver_route_semantics.py)
- focused tests: [`tests/test_e7a_receiver_route_semantics.py`](tests/test_e7a_receiver_route_semantics.py)
- reused E5a hook-DAG: [`src/path_patching_semantics.py`](src/path_patching_semantics.py)

The tests were written before the module existed and first failed at import.
After implementation, the focused E7a/E5a contract passed, and the complete S9
suite passed `207/207`.

The first full-suite run caught that editing the old E5a DAG source would break
an authenticated plan hash. The generic E5a source was restored byte-for-byte
and the E7 override evaluator was isolated in the new module. Final SHA-256:

```text
receiver_route_semantics.py       d0e72ff43dac353890d7b26936dddbf4ee537c27d40ab7136db62fd7018c6c8f
test_e7a_receiver_route_semantics  2b2cc2e320b6dfd2a66fdaad62a7ce7383cc22abc76411008b5c254bf8b8bda0
path_patching_semantics.py         516d7b43cddebaef7118a65393fe43914946888a0ec151546d07efed8f9dad91
```

The last value exactly matches the completed E5a plans.

## Evidence boundary and next action

E7a is an executable semantic contract, not empirical evidence about Qwen3-4B.
It contains no model import, prompt, token, generation, receiver outcome, or
topology result. It licenses drafting E7b's pre-outcome panel and estimating its
compute. E7c remains blocked on explicit user confirmation after E7b is
reviewable.
