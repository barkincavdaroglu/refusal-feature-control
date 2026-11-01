#!/usr/bin/env python3
"""Run the authorized S16 Phase-B forward-only activation census.

Importing this module never constructs a model. ``main`` authenticates the
complete local closure and exact execution request before creating outputs or
loading Qwen.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

import s16_phase_b_contract as contract


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def tensor_bytes_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array.astype("<i4", copy=False))
    import hashlib

    return hashlib.sha256(value.tobytes()).hexdigest()


def load_token_bundle(
    path: pathlib.Path, manifest: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"input_ids_flat", "row_offsets"}:
            raise contract.ContractError("Phase-B token-bundle key closure differs")
        tokens = np.asarray(archive["input_ids_flat"])
        offsets = np.asarray(archive["row_offsets"])
    if tokens.dtype != np.dtype("int32") or offsets.dtype != np.dtype("int64"):
        raise contract.ContractError("Phase-B token-bundle dtypes differ")
    if (
        tokens.ndim != 1
        or offsets.shape != (contract.EXPECTED_ROW_COUNT + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != tokens.size
        or np.any(offsets[1:] <= offsets[:-1])
        or np.any(tokens < 0)
        or np.any(tokens >= 151669)
    ):
        raise contract.ContractError("Phase-B token-bundle structure differs")
    for row in manifest["rows"]:
        index = row["row_index"]
        start, end = int(offsets[index]), int(offsets[index + 1])
        if (
            end - start != row["token_count"]
            or tensor_bytes_sha256(tokens[start:end]) != row["input_ids_sha256"]
        ):
            raise contract.ContractError(f"Phase-B token row differs: {index}")
    return tokens, offsets


def batch_tokens(
    tokens: np.ndarray,
    offsets: np.ndarray,
    row_indices: Sequence[int],
) -> np.ndarray:
    rows = [tokens[int(offsets[i]) : int(offsets[i + 1])] for i in row_indices]
    lengths = {row.size for row in rows}
    if len(lengths) != 1:
        raise contract.ContractError("Phase-B runtime batch contains padding")
    return np.stack(rows).astype(np.int64, copy=False)


def load_feature_row(path: pathlib.Path):
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required = {
            "layer_31.W_dec",
            "layer_31.W_enc",
            "layer_31.b_enc",
            "layer_31.feature_indices",
            "layer_32.W_dec",
            "layer_32.W_enc",
            "layer_32.b_enc",
            "layer_32.feature_indices",
        }
        if keys != required:
            raise contract.ContractError("Phase-B feature-bundle keys differ")
        indices = handle.get_tensor("layer_31.feature_indices")
        encoder = handle.get_tensor("layer_31.W_enc")
        decoder = handle.get_tensor("layer_31.W_dec")
        bias = handle.get_tensor("layer_31.b_enc")
    values = [int(value) for value in indices.tolist()]
    if values != [49541, 139261] or encoder.shape != (2, 2560) or decoder.shape != (2, 2560):
        raise contract.ContractError("Phase-B feature rows differ")
    if encoder.dtype != torch.bfloat16 or decoder.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        raise contract.ContractError("Phase-B feature storage dtype differs")
    return encoder[0].contiguous(), decoder[0].contiguous(), bias[0].reshape(()).contiguous()


def feature_values(states, encoder, bias):
    """Vectorized all-token BF16 path plus E8-compatible rowwise final path."""

    import torch

    if states.ndim != 3 or states.shape[-1] != encoder.numel():
        raise contract.ContractError("Phase-B feature state shape differs")
    if states.dtype != torch.bfloat16 or encoder.dtype != torch.bfloat16:
        raise contract.ContractError("Phase-B feature operands must be BF16")
    pre_all = torch.matmul(states, encoder) + bias
    act_all = torch.relu(pre_all)
    final_pre = torch.stack(
        [torch.dot(encoder, states[index, -1]) + bias for index in range(states.shape[0])]
    )
    final_act = torch.relu(final_pre)
    if any(value.dtype != torch.bfloat16 for value in (pre_all, act_all, final_pre, final_act)):
        raise contract.ContractError("Phase-B feature output dtype differs")
    return pre_all, act_all, final_pre, final_act


def feature_values_rowwise_all(states, encoder, bias):
    """Prespecified numerical fallback with one BF16 dot per real token."""

    import torch

    if states.ndim != 3 or states.shape[-1] != encoder.numel():
        raise contract.ContractError("Phase-B rowwise feature state shape differs")
    pre = torch.stack(
        [
            torch.stack([torch.dot(encoder, token) + bias for token in row])
            for row in states
        ]
    )
    activation = torch.relu(pre)
    if pre.dtype != torch.bfloat16 or activation.dtype != torch.bfloat16:
        raise contract.ContractError("Phase-B rowwise feature output dtype differs")
    return pre, activation


def reference_rows(manifest: Mapping[str, Any], indices: Sequence[int]) -> list[int]:
    return [
        offset
        for offset, index in enumerate(indices)
        if manifest["rows"][index].get("reference_pair_id") is not None
    ]


def _load_running(
    path: pathlib.Path, *, plan: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": contract.RUNNING_SCHEMA,
            "status": "running",
            "created_at_utc": _utc(),
            "execution_plan_payload_sha256": plan["payload_sha256"],
            "execution_request_payload_sha256": request["payload_sha256"],
            "completed_batches": [],
            "inflight": None,
            "forward_accounting": {
                "attempted_invocations": 0,
                "attempted_scientific_invocations": 0,
                "attempted_prompt_equivalents": 0,
                "attempted_scientific_prompt_equivalents": 0,
                "attempted_token_equivalents": 0,
                "attempted_scientific_token_equivalents": 0,
                "completed_unique_prompt_equivalents": 0,
                "completed_unique_token_equivalents": 0,
            },
            "audit": None,
            "audit_inflight": False,
            "execution_branch": None,
            "model_load_count": 0,
        }
    payload = contract.load_authenticated_json(path, schema=contract.RUNNING_SCHEMA)
    if (
        payload.get("execution_plan_payload_sha256") != plan["payload_sha256"]
        or payload.get("execution_request_payload_sha256")
        != request["payload_sha256"]
    ):
        raise contract.ContractError("Phase-B running authority binding differs")
    return payload


def _bind_running_allocation(
    state: dict[str, Any],
    *,
    allocation: Mapping[str, Any],
    running_existed: bool,
) -> None:
    """Initialize fresh lineage once; require exact lineage on every resume."""

    expected = contract.allocation_evidence_bindings(allocation)
    for key, value in expected.items():
        if running_existed and state.get(key) != value:
            raise contract.ContractError("Phase-B running allocation binding differs")
        if not running_existed:
            state[key] = value
    deadline = allocation["lease_expires_at_utc"]
    if running_existed and state.get("lease_expires_at_utc") != deadline:
        raise contract.ContractError("Phase-B running allocation lease binding differs")
    if not running_existed:
        state["lease_expires_at_utc"] = deadline


def _validate_premodel_proofs(
    output_dir: pathlib.Path,
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    allocation: Mapping[str, Any],
    binding: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name in (
        "s16_phase_b_runtime_preflight.json",
        "s16_phase_b_dependency_runtime_proof.json",
        "s16_phase_b_private_runtime_preflight.json",
    ):
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise contract.ContractError("Phase-B premodel proof path is unsafe")
    preflight = contract.load_authenticated_json(
        output_dir / "s16_phase_b_runtime_preflight.json",
        schema=contract.RUNTIME_PROOF_SCHEMA,
    )
    dependency = contract.load_authenticated_json(
        output_dir / "s16_phase_b_dependency_runtime_proof.json",
        schema=contract.DEPENDENCY_RUNTIME_PROOF_SCHEMA,
    )
    private = contract.load_authenticated_json(
        output_dir / "s16_phase_b_private_runtime_preflight.json",
        schema=contract.PRIVATE_RUNTIME_PROOF_SCHEMA,
    )
    public_input_hashes = {
        role: digest
        for role, digest in input_hashes.items()
        if plan["inputs"][role]["sensitivity"] == "public"
    }
    contract.validate_public_runtime_preflight(
        preflight,
        plan=plan,
        request=request,
        public_input_hashes=public_input_hashes,
        allocation=allocation,
        binding=binding,
    )
    contract.validate_dependency_runtime_proof(
        dependency,
        plan=plan,
        request=request,
        public_preflight=preflight,
    )
    private_input_hashes = {
        role: digest
        for role, digest in input_hashes.items()
        if plan["inputs"][role]["sensitivity"] != "public"
    }
    contract.validate_private_runtime_proof(
        private,
        plan=plan,
        request=request,
        public_preflight=preflight,
        dependency_proof=dependency,
        private_input_hashes=private_input_hashes,
        allocation=allocation,
        binding=binding,
    )
    for proof in (preflight, dependency, private):
        if (
            proof.get("execution_plan_payload_sha256") != plan["payload_sha256"]
            or proof.get("execution_request_payload_sha256")
            != request["payload_sha256"]
            or proof.get("proposal_payload_sha256")
            != request["proposal_payload_sha256"]
            or proof.get("model_loaded") is not False
            or proof.get("model_forwards") != 0
        ):
            raise contract.ContractError("Phase-B premodel proof binding differs")
    if (
        preflight.get("status")
        != "public_authority_and_inputs_authenticated_by_stdlib_before_dependency_install"
        or preflight.get("scientific_modules_imported") is not False
        or preflight.get("uploaded_public_input_hashes")
        != public_input_hashes
        or preflight.get("private_input_roles_confirmed_absent")
        != ["batch_schedule", "feature_bundle", "token_bundle"]
        or preflight.get("session_binding_payload_sha256")
        != binding["payload_sha256"]
    ):
        raise contract.ContractError("Phase-B premodel lifecycle proof differs")
    contract.validate_allocation_evidence_bindings(
        preflight, allocation, label="premodel public proof"
    )
    return preflight, dependency, private


def _validate_completed_shards(state: Mapping[str, Any], output_dir: pathlib.Path) -> set[int]:
    if state.get("status") not in {"running", "forward_complete_analysis_pending"}:
        raise contract.ContractError("Phase-B running-state transition differs")
    loads = state.get("model_load_count")
    if isinstance(loads, bool) or not isinstance(loads, int) or not 0 <= loads <= 2:
        raise contract.ContractError("Phase-B model-load accounting differs")
    complete: set[int] = set()
    completed_prompts = 0
    completed_tokens = 0
    for record in state["completed_batches"]:
        batch_id = record.get("batch_id")
        relative = pathlib.PurePosixPath(str(record.get("relative_path")))
        if relative.is_absolute() or ".." in relative.parts or batch_id in complete:
            raise contract.ContractError("Phase-B completed shard record differs")
        path = output_dir / pathlib.Path(*relative.parts)
        if path.is_symlink() or not path.is_file() or contract.file_sha256(path) != record.get("sha256"):
            raise contract.ContractError("Phase-B completed shard differs")
        complete.add(batch_id)
        completed_prompts += record["prompt_equivalents"]
        completed_tokens += record["token_equivalents"]
    account = state.get("forward_accounting")
    if not isinstance(account, Mapping) or (
        account.get("completed_unique_prompt_equivalents") != completed_prompts
        or account.get("completed_unique_token_equivalents") != completed_tokens
        or account.get("attempted_scientific_prompt_equivalents", 0) < completed_prompts
        or account.get("attempted_scientific_token_equivalents", 0) < completed_tokens
    ):
        raise contract.ContractError("Phase-B checkpoint accounting differs")
    return complete


def _validate_output_tree_before_forward(
    state: Mapping[str, Any], output_dir: pathlib.Path, *, state_path: pathlib.Path
) -> None:
    expected = {
        state_path.resolve(),
        (output_dir / "s16_phase_b_runtime_preflight.json").resolve(),
        (output_dir / "s16_phase_b_dependency_runtime_proof.json").resolve(),
        (output_dir / "s16_phase_b_private_runtime_preflight.json").resolve(),
    }
    for record in state["completed_batches"]:
        expected.add((output_dir / record["relative_path"]).resolve())
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise contract.ContractError("Phase-B output tree contains an unsafe symlink")
        if path.is_dir():
            continue
        if path.resolve() not in expected:
            raise contract.ContractError("Phase-B output tree contains an unexpected file")


def _write_shard(output_dir: pathlib.Path, batch_id: int, tensors: Mapping[str, Any]) -> dict[str, Any]:
    from safetensors.torch import save_file

    shard_dir = output_dir / "private" / "shards"
    contract.ensure_private_directory(output_dir / "private")
    contract.ensure_private_directory(shard_dir)
    temporary = shard_dir / f".batch-{batch_id:05d}.tmp.safetensors"
    save_file(dict(tensors), str(temporary))
    os.chmod(temporary, contract.PRIVATE_FILE_MODE)
    digest = contract.file_sha256(temporary)
    target = shard_dir / f"batch-{batch_id:05d}.{digest}.safetensors"
    if target.exists() or target.is_symlink():
        if not target.is_file() or contract.file_sha256(target) != digest:
            raise contract.ContractError("Phase-B immutable shard collision")
        temporary.unlink()
    else:
        os.replace(temporary, target)
    os.chmod(target, contract.PRIVATE_FILE_MODE)
    contract.require_private_file(target)
    return {
        "relative_path": target.relative_to(output_dir).as_posix(),
        "sha256": digest,
        "byte_count": target.stat().st_size,
    }


def _validate_recovery_shard(
    path: pathlib.Path,
    *,
    row_indices: Sequence[int],
    manifest: Mapping[str, Any],
) -> None:
    import torch
    from safetensors import safe_open

    contract.require_private_file(path)
    indices = list(row_indices)
    if not indices:
        raise contract.ContractError("Phase-B recovery shard has no rows")
    length = manifest["rows"][indices[0]]["token_count"]
    if any(manifest["rows"][index]["token_count"] != length for index in indices):
        raise contract.ContractError("Phase-B recovery shard row lengths differ")
    expected_reference = [
        index
        for index in indices
        if manifest["rows"][index].get("reference_pair_id") is not None
    ]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        expected_keys = {
            "row_indices",
            "c_preactivation_all",
            "c_activation_all",
            "c_preactivation_final",
            "c_activation_final",
            "c_preactivation_last_user",
            "c_activation_last_user",
            "reference_row_indices",
            "reference_resid_post",
            "reference_resid_pre",
            "reference_mlp_hook_in",
        }
        if set(handle.keys()) != expected_keys:
            raise contract.ContractError("Phase-B recovery shard key closure differs")
        tensors = {name: handle.get_tensor(name) for name in expected_keys}
    n, k = len(indices), len(expected_reference)
    if (
        tensors["row_indices"].dtype != torch.int64
        or tensors["row_indices"].shape != (n,)
        or tensors["row_indices"].tolist() != indices
        or tensors["reference_row_indices"].dtype != torch.int64
        or tensors["reference_row_indices"].shape != (k,)
        or tensors["reference_row_indices"].tolist() != expected_reference
    ):
        raise contract.ContractError("Phase-B recovery shard row identity differs")
    for name in ("c_preactivation_all", "c_activation_all"):
        if tensors[name].dtype != torch.bfloat16 or tensors[name].shape != (n, length):
            raise contract.ContractError("Phase-B recovery all-token tensor differs")
    for name in (
        "c_preactivation_final",
        "c_activation_final",
        "c_preactivation_last_user",
        "c_activation_last_user",
    ):
        if tensors[name].dtype != torch.bfloat16 or tensors[name].shape != (n,):
            raise contract.ContractError("Phase-B recovery endpoint tensor differs")
    for name in ("reference_resid_post", "reference_resid_pre", "reference_mlp_hook_in"):
        if tensors[name].dtype != torch.bfloat16 or tensors[name].shape != (k, 2560):
            raise contract.ContractError("Phase-B recovery reference tensor differs")


def _recover_inflight_shard(
    state: dict[str, Any],
    *,
    output_dir: pathlib.Path,
    scheduled: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    state_path: pathlib.Path,
) -> bool:
    """Adopt or replay only the shard named by the authenticated inflight ledger."""

    inflight = state.get("inflight")
    if inflight is None:
        return False
    if not isinstance(inflight, Mapping):
        raise contract.ContractError("Phase-B in-flight checkpoint is malformed")
    by_id = {batch["batch_index"]: batch for batch in scheduled}
    batch_id = inflight.get("batch_id")
    expected = by_id.get(batch_id)
    if (
        expected is None
        or inflight.get("row_indices") != expected["row_indices"]
        or any(record.get("batch_id") == batch_id for record in state["completed_batches"])
    ):
        raise contract.ContractError("Phase-B in-flight checkpoint/schedule differs")
    shard_dir = output_dir / "private" / "shards"
    contract.ensure_private_directory(output_dir / "private")
    contract.ensure_private_directory(shard_dir)
    temporary = shard_dir / f".batch-{batch_id:05d}.tmp.safetensors"
    targets = list(shard_dir.glob(f"batch-{batch_id:05d}.*.safetensors"))
    if temporary.is_symlink() or len(targets) > 1 or any(path.is_symlink() for path in targets):
        raise contract.ContractError("Phase-B inflight shard candidates differ")
    candidate = targets[0] if targets else temporary if temporary.exists() else None
    recovery_status = "replay_missing_inflight_shard"
    if candidate is not None:
        try:
            _validate_recovery_shard(
                candidate,
                row_indices=expected["row_indices"],
                manifest=manifest,
            )
        except contract.ContractError:
            if candidate != temporary:
                raise
            temporary.unlink()
            candidate = None
            recovery_status = "replay_invalid_inflight_temporary"
    if candidate is not None:
        digest = contract.file_sha256(candidate)
        target = shard_dir / f"batch-{batch_id:05d}.{digest}.safetensors"
        if candidate == temporary:
            os.replace(temporary, target)
        elif candidate != target:
            raise contract.ContractError("Phase-B inflight shard filename hash differs")
        os.chmod(target, contract.PRIVATE_FILE_MODE)
        prompt_equiv = len(expected["row_indices"])
        token_equiv = prompt_equiv * expected["token_count"]
        state["completed_batches"].append(
            {
                "batch_id": batch_id,
                "row_indices": expected["row_indices"],
                "prompt_equivalents": prompt_equiv,
                "token_equivalents": token_equiv,
                "relative_path": target.relative_to(output_dir).as_posix(),
                "sha256": digest,
                "byte_count": target.stat().st_size,
            }
        )
        state["completed_batches"].sort(key=lambda row: row["batch_id"])
        state["forward_accounting"]["completed_unique_prompt_equivalents"] += prompt_equiv
        state["forward_accounting"]["completed_unique_token_equivalents"] += token_equiv
        recovery_status = "adopted_authenticated_inflight_shard_without_forward"
    state.setdefault("recovery_events", []).append(
        {
            "batch_id": batch_id,
            "status": recovery_status,
            "recorded_at_utc": _utc(),
            "additional_forward_invocations": 0,
        }
    )
    state["inflight"] = None
    contract.atomic_json(state_path, state)
    return candidate is not None


def _authenticate_model_snapshot(snapshot: pathlib.Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    if (
        snapshot.is_symlink()
        or not snapshot.is_dir()
        or snapshot.name != plan["model"]["snapshot_directory_name"]
    ):
        raise contract.ContractError("Phase-B model snapshot directory differs")
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(plan["model"]["files"]):
        raise contract.ContractError("Phase-B model snapshot has missing or extra files")
    if any(path.is_dir() for path in snapshot.rglob("*") if not path.is_symlink()):
        raise contract.ContractError("Phase-B model snapshot contains an extra directory")
    files = []
    for name, expected in sorted(plan["model"]["files"].items()):
        path = snapshot / name
        if path.is_symlink():
            path = path.resolve(strict=True)
        size = path.stat().st_size if path.is_file() else None
        if (
            not path.is_file()
            or contract.file_sha256(path) != expected
            or size != contract.MODEL_FILE_SIZES[name]
        ):
            raise contract.ContractError(f"Phase-B model snapshot differs: {name}")
        files.append({"path": name, "size": size, "sha256": expected})
    proof = {
        "schema_version": contract.MODEL_SNAPSHOT_PROOF_SCHEMA,
        "repo_id": contract.MODEL_ID,
        "requested_revision": contract.MODEL_REVISION,
        "resolved_revision": snapshot.name,
        "file_count": len(files),
        "file_closure_sha256": contract.canonical_sha256({"files": files}),
        "files": files,
        "verified_before_first_forward": True,
    }
    if proof["file_closure_sha256"] != contract.MODEL_FILE_CLOSURE_SHA256:
        raise contract.ContractError("Phase-B model snapshot closure hash differs")
    return proof


def _branch_scientific_invocation_ceiling(
    scheduled: Sequence[Mapping[str, Any]],
) -> int:
    """Return the frozen branch's invocation count plus its exact 5% retry allowance."""

    if not scheduled:
        raise contract.ContractError("Phase-B execution branch is empty")
    return math.ceil(1.05 * len(scheduled))


def _install_wrapper(model):
    import torch.nn as nn
    from transformer_lens.hook_points import HookPoint

    class ExactHookedMLP(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original_mlp = original
            self.hook_in = HookPoint()
            self.hook_out = HookPoint()

        def forward(self, activations):
            return self.hook_out(self.original_mlp(self.hook_in(activations)))

    model.blocks[31].mlp = ExactHookedMLP(model.blocks[31].mlp)
    model.setup()


def _audit(model, token_rows, encoder, bias, tolerance: float) -> dict[str, Any]:
    import torch

    first = token_rows[0].unsqueeze(0)
    with torch.inference_mode():
        before = model(first, return_type="logits")
    _install_wrapper(model)
    with torch.inference_mode():
        after = model(first, return_type="logits")
    wrapper_max = float((after.float() - before.float()).abs().max().cpu())
    captures: list[torch.Tensor] = []

    def hook(value, _hook):
        captures.append(value.detach())
        return value

    singles = []
    final_parity_differences = []
    for row in token_rows:
        captures.clear()
        with torch.inference_mode():
            _ = model.run_with_hooks(row.unsqueeze(0), fwd_hooks=[(contract.FEATURE_HOOK, hook)])
        _, act, _, final_act = feature_values(captures[0], encoder, bias)
        singles.append(act[0].detach().float().cpu())
        final_parity_differences.append(
            float((act[0, -1].float() - final_act[0].float()).abs().cpu())
        )
    captures.clear()
    stacked = torch.stack(token_rows)
    with torch.inference_mode():
        _ = model.run_with_hooks(stacked, fwd_hooks=[(contract.FEATURE_HOOK, hook)])
    _, batched, _, batched_final = feature_values(captures[0], encoder, bias)
    batch_max = float(
        max((batched[i].float().cpu() - singles[i]).abs().max().item() for i in range(4))
    )
    final_parity_differences.extend(
        (batched[:, -1].float().cpu() - batched_final.float().cpu()).abs().tolist()
    )
    final_parity_max = float(max(final_parity_differences))
    return {
        "wrapper_identity_max_abs_logit_difference": wrapper_max,
        "batch_activation_max_abs_difference": batch_max,
        "vectorized_vs_rowwise_final_max_abs_difference": final_parity_max,
        "tolerance": tolerance,
        "normal_batch_schedule_allowed": (
            wrapper_max == 0.0
            and batch_max <= tolerance
            and final_parity_max <= tolerance
        ),
        "audit_invocations": 7,
        "audit_prompt_equivalents": 10,
    }


def _terminalize_if_schedule_complete(
    state: dict[str, Any],
    *,
    scheduled: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    state_path: pathlib.Path,
    decoder_cpu,
    probe_path: pathlib.Path,
    preflight_proof: Mapping[str, Any],
    dependency_proof: Mapping[str, Any],
    private_preflight_proof: Mapping[str, Any],
) -> bool:
    """Commit terminal forward state without another model load when complete."""

    expected_by_id = {batch["batch_index"]: batch for batch in scheduled}
    completed = state.get("completed_batches")
    if not isinstance(completed, list):
        raise contract.ContractError("Phase-B completed checkpoint closure differs")
    completed_by_id = {record.get("batch_id"): record for record in completed}
    if len(completed_by_id) != len(completed):
        raise contract.ContractError("Phase-B completed checkpoint closure differs")
    if set(completed_by_id) != set(expected_by_id):
        return False
    for batch_id, expected in expected_by_id.items():
        record = completed_by_id[batch_id]
        if (
            record.get("row_indices") != expected["row_indices"]
            or record.get("prompt_equivalents") != len(expected["row_indices"])
            or record.get("token_equivalents")
            != len(expected["row_indices"]) * expected["token_count"]
        ):
            raise contract.ContractError("Phase-B completed checkpoint/schedule differs")
    account = state.get("forward_accounting", {})
    if (
        state.get("inflight") is not None
        or account.get("completed_unique_prompt_equivalents")
        != contract.EXPECTED_ROW_COUNT
        or account.get("completed_unique_token_equivalents")
        != plan["corpus"]["total_token_count"]
        or state.get("branch_scientific_invocation_ceiling")
        != _branch_scientific_invocation_ceiling(scheduled)
    ):
        raise contract.ContractError("Phase-B terminal schedule/accounting closure differs")
    state["status"] = "forward_complete_analysis_pending"
    state.setdefault("completed_at_utc", _utc())
    state["c_decoder_sha256"] = contract.canonical_sha256(
        {
            "dtype": "bfloat16",
            "shape": [2560],
            "values_sha256": _torch_bytes_sha256(decoder_cpu),
        }
    )
    state["probe_input_path"] = probe_path.name
    state["premodel_proofs"] = {
        "preinstall_payload_sha256": preflight_proof["payload_sha256"],
        "dependency_payload_sha256": dependency_proof["payload_sha256"],
        "private_preflight_payload_sha256": private_preflight_proof["payload_sha256"],
    }
    contract.atomic_json(state_path, state)
    return True


def execute(
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    allocation: Mapping[str, Any],
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    tokens: np.ndarray,
    offsets: np.ndarray,
    feature_bundle: pathlib.Path,
    probe_path: pathlib.Path,
    output_dir: pathlib.Path,
    input_hashes: Mapping[str, str],
) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformer_lens import HookedTransformer

    contract.validate_execution_request(request, plan=plan)
    remaining_at_start = contract.remaining_allocation_lease_seconds(
        allocation, plan=plan, request=request
    )
    if remaining_at_start <= 0.0:
        raise contract.ContractError("Phase-B absolute allocation lease has expired")
    if (
        binding.get("schema_version") != contract.SESSION_BINDING_SCHEMA
        or contract.canonical_sha256(binding) != binding.get("payload_sha256")
        or binding.get("endpoint_sha256") != allocation.get("endpoint_sha256")
        or binding.get("execution_plan_payload_sha256") != plan["payload_sha256"]
        or binding.get("execution_request_payload_sha256")
        != request["payload_sha256"]
    ):
        raise contract.ContractError("Phase-B remote session binding differs")
    contract.validate_allocation_evidence_bindings(
        binding, allocation, label="remote session binding"
    )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise contract.ContractError("CUBLAS_WORKSPACE_CONFIG must be set before interpreter start")
    if not torch.cuda.is_available():
        raise contract.ContractError("Phase B requires CUDA")
    if "A100" not in torch.cuda.get_device_name(0):
        raise contract.ContractError("Phase B requires the frozen A100 accelerator")
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory < plan["resource_contract"]["minimum_cuda_memory_gib"]:
        raise contract.ContractError("Phase-B CUDA memory is below the contract")
    torch.manual_seed(plan["runtime"]["seed"])
    torch.cuda.manual_seed_all(plan["runtime"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    preflight_proof, dependency_proof, private_preflight_proof = _validate_premodel_proofs(
        output_dir,
        plan=plan,
        request=request,
        allocation=allocation,
        binding=binding,
        input_hashes=input_hashes,
    )

    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise contract.ContractError("Phase-B output root is unsafe")
        state_candidate = output_dir / "s16_phase_b_running.json"
        allowed_initial = {
            "s16_phase_b_runtime_preflight.json",
            "s16_phase_b_dependency_runtime_proof.json",
            "s16_phase_b_private_runtime_preflight.json",
        }
        initial_files = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        if not state_candidate.exists() and initial_files - allowed_initial:
            raise contract.ContractError("Phase-B preexisting output tree is unexpected")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "s16_phase_b_running.json"
    running_existed = state_path.exists()
    state = _load_running(state_path, plan=plan, request=request)
    _bind_running_allocation(
        state, allocation=allocation, running_existed=running_existed
    )
    state.setdefault("remaining_lease_seconds_at_runner_start", remaining_at_start)
    contract.atomic_json(state_path, state)
    if state.get("inflight") is not None:
        if not isinstance(state.get("audit"), Mapping) or state.get(
            "execution_branch"
        ) not in {
            "frozen_equal_length_batches",
            "prespecified_batch_size_one_fallback",
        }:
            raise contract.ContractError(
                "Phase-B inflight recovery lacks a frozen execution branch"
            )
        recovery_schedule = (
            schedule["batches"]
            if state["execution_branch"] == "frozen_equal_length_batches"
            else [
                {
                    "batch_index": index,
                    "row_indices": [index],
                    "token_count": manifest["rows"][index]["token_count"],
                }
                for index in range(contract.EXPECTED_ROW_COUNT)
            ]
        )
        _recover_inflight_shard(
            state,
            output_dir=output_dir,
            scheduled=recovery_schedule,
            manifest=manifest,
            state_path=state_path,
        )
    complete = _validate_completed_shards(state, output_dir)
    _validate_output_tree_before_forward(state, output_dir, state_path=state_path)
    if state.get("status") == "forward_complete_analysis_pending":
        return
    if (
        isinstance(state.get("audit"), Mapping)
        and state.get("execution_branch")
        in {"frozen_equal_length_batches", "prespecified_batch_size_one_fallback"}
    ):
        recovered_schedule = (
            schedule["batches"]
            if state["execution_branch"] == "frozen_equal_length_batches"
            else [
                {
                    "batch_index": index,
                    "row_indices": [index],
                    "token_count": manifest["rows"][index]["token_count"],
                }
                for index in range(contract.EXPECTED_ROW_COUNT)
            ]
        )
        if complete == {batch["batch_index"] for batch in recovered_schedule}:
            _encoder_cpu, recovered_decoder_cpu, _bias_cpu = load_feature_row(
                feature_bundle
            )
            if _terminalize_if_schedule_complete(
                state,
                scheduled=recovered_schedule,
                plan=plan,
                state_path=state_path,
                decoder_cpu=recovered_decoder_cpu,
                probe_path=probe_path,
                preflight_proof=preflight_proof,
                dependency_proof=dependency_proof,
                private_preflight_proof=private_preflight_proof,
            ):
                return
    if state.get("audit") is None and state.get("audit_inflight") is True:
        raise contract.ContractError(
            "Phase-B identity/parity audit was interrupted and cannot be retried "
            "under the frozen seven-invocation audit ceiling"
        )
    if (
        isinstance(state.get("audit"), Mapping)
        and state["audit"].get("wrapper_identity_max_abs_logit_difference") != 0.0
    ):
        raise contract.ContractError("Phase-B wrapper identity audit failed")

    encoder_cpu, decoder_cpu, bias_cpu = load_feature_row(feature_bundle)
    actual_versions = {
        name: importlib.metadata.version(name) for name in contract.PACKAGE_VERSIONS
    }
    if actual_versions != contract.PACKAGE_VERSIONS:
        raise contract.ContractError("Phase-B runtime package versions differ")
    if state.get("model_load_count", 0) + 1 > plan["resource_contract"]["maximum_authenticated_model_loads"]:
        raise contract.ContractError("Phase-B model-load ceiling exhausted")
    state["model_load_count"] = state.get("model_load_count", 0) + 1
    contract.atomic_json(state_path, state)
    snapshot = pathlib.Path(
        snapshot_download(
            repo_id=contract.MODEL_ID,
            revision=contract.MODEL_REVISION,
            allow_patterns=list(contract.MODEL_FILES),
        )
    )
    model_proof = _authenticate_model_snapshot(snapshot, plan)
    device = torch.device("cuda")
    model = HookedTransformer.from_pretrained(
        contract.MODEL_ID,
        revision=contract.MODEL_REVISION,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder, decoder, bias = (
        encoder_cpu.to(device=device),
        decoder_cpu.to(device=device),
        bias_cpu.to(device=device),
    )

    if state["audit"] is None:
        if contract.remaining_allocation_lease_seconds(
            allocation, plan=plan, request=request
        ) <= 0.0:
            raise contract.ContractError("Phase-B absolute allocation lease has expired")
        audit_indices = plan["runtime"]["audit_row_indices"]
        rows = [
            torch.as_tensor(
                tokens[int(offsets[i]) : int(offsets[i + 1])],
                dtype=torch.long,
                device=device,
            )
            for i in audit_indices
        ]
        if len(rows) != 4 or len({int(row.numel()) for row in rows}) != 1:
            raise contract.ContractError("Phase-B audit rows must be four equal-length sequences")
        audit_tokens = 2 * int(rows[0].numel()) + 2 * sum(
            int(row.numel()) for row in rows
        )
        if audit_tokens != plan["forward_accounting"]["audit_token_equivalents"]:
            raise contract.ContractError("Phase-B audit token accounting differs")
        account = state["forward_accounting"]
        if (
            account["attempted_invocations"] + 7
            > plan["forward_accounting"]["attempted_total_invocation_ceiling"]
            or account["attempted_prompt_equivalents"] + 10
            > plan["forward_accounting"]["attempted_prompt_equivalent_ceiling"]
            or account["attempted_token_equivalents"] + audit_tokens
            > plan["forward_accounting"]["attempted_total_token_equivalent_ceiling"]
        ):
            raise contract.ContractError("Phase-B audit attempt ceiling exhausted")
        account["attempted_invocations"] += 7
        account["attempted_prompt_equivalents"] += 10
        account["attempted_token_equivalents"] += audit_tokens
        state["audit_inflight"] = True
        contract.atomic_json(state_path, state)
        state["audit"] = _audit(
            model, rows, encoder, bias, plan["runtime"]["batch_parity_abs_tolerance"]
        )
        state["model_snapshot_proof"] = model_proof
        state["runtime"] = {
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_total_memory_gib": memory,
            "torch_cuda": str(torch.version.cuda),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in contract.PACKAGE_VERSIONS
            },
        }
        state["audit_inflight"] = False
        contract.atomic_json(state_path, state)
    else:
        if (
            state.get("model_snapshot_proof") != model_proof
            or state.get("runtime", {}).get("packages") != contract.PACKAGE_VERSIONS
        ):
            raise contract.ContractError("Phase-B resume model/runtime proof differs")
        _install_wrapper(model)

    normal = bool(state["audit"]["normal_batch_schedule_allowed"])
    if state["audit"]["wrapper_identity_max_abs_logit_difference"] != 0.0:
        raise contract.ContractError("Phase-B wrapper identity audit failed")
    branch = "frozen_equal_length_batches" if normal else "prespecified_batch_size_one_fallback"
    if state.get("execution_branch") not in {None, branch}:
        raise contract.ContractError("Phase-B resume execution branch differs")
    state["execution_branch"] = branch
    contract.atomic_json(state_path, state)
    scheduled = schedule["batches"] if normal else [
        {"batch_index": index, "row_indices": [index], "token_count": manifest["rows"][index]["token_count"]}
        for index in range(contract.EXPECTED_ROW_COUNT)
    ]
    branch_invocation_ceiling = _branch_scientific_invocation_ceiling(scheduled)
    if state.get("branch_scientific_invocation_ceiling") not in {
        None,
        branch_invocation_ceiling,
    }:
        raise contract.ContractError("Phase-B branch invocation ceiling differs")
    state["branch_scientific_invocation_ceiling"] = branch_invocation_ceiling
    contract.atomic_json(state_path, state)
    if complete and any(batch_id >= len(scheduled) for batch_id in complete):
        raise contract.ContractError("Phase-B resumed branch differs from prior schedule")
    scheduled_by_id = {batch["batch_index"]: batch for batch in scheduled}
    for record in state["completed_batches"]:
        expected = scheduled_by_id.get(record["batch_id"])
        if (
            expected is None
            or record.get("row_indices") != expected["row_indices"]
            or record.get("prompt_equivalents") != len(expected["row_indices"])
            or record.get("token_equivalents")
            != len(expected["row_indices"]) * expected["token_count"]
        ):
            raise contract.ContractError("Phase-B completed checkpoint/schedule differs")
    inflight = state.get("inflight")
    if inflight is not None:
        if not isinstance(inflight, Mapping):
            raise contract.ContractError("Phase-B in-flight checkpoint is malformed")
        expected = scheduled_by_id.get(inflight.get("batch_id"))
        if (
            expected is None
            or inflight.get("batch_id") in complete
            or inflight.get("row_indices") != expected["row_indices"]
        ):
            raise contract.ContractError("Phase-B in-flight checkpoint/schedule differs")

    for batch in scheduled:
        batch_id = batch["batch_index"]
        if batch_id in complete:
            continue
        if contract.remaining_allocation_lease_seconds(
            allocation, plan=plan, request=request
        ) <= 0.0:
            raise contract.ContractError("Phase-B absolute allocation lease has expired")
        indices = batch["row_indices"]
        prompt_equiv = len(indices)
        token_equiv = prompt_equiv * batch["token_count"]
        account = state["forward_accounting"]
        if (
            account["attempted_prompt_equivalents"] + prompt_equiv
            > plan["forward_accounting"]["attempted_prompt_equivalent_ceiling"]
            or account["attempted_scientific_invocations"] + 1
            > branch_invocation_ceiling
            or account["attempted_scientific_token_equivalents"] + token_equiv
            > plan["forward_accounting"]["attempted_scientific_token_equivalent_ceiling"]
            or account["attempted_token_equivalents"] + token_equiv
            > plan["forward_accounting"]["attempted_total_token_equivalent_ceiling"]
            or account["attempted_invocations"] + 1
            > plan["forward_accounting"]["attempted_total_invocation_ceiling"]
        ):
            raise contract.ContractError("Phase-B prompt-equivalent ceiling exhausted")
        state["inflight"] = {
            "batch_id": batch_id,
            "row_indices": indices,
            "started_at_utc": _utc(),
        }
        account["attempted_invocations"] += 1
        account["attempted_scientific_invocations"] += 1
        account["attempted_prompt_equivalents"] += prompt_equiv
        account["attempted_scientific_prompt_equivalents"] += prompt_equiv
        account["attempted_token_equivalents"] += token_equiv
        account["attempted_scientific_token_equivalents"] += token_equiv
        contract.atomic_json(state_path, state)

        batch_array = batch_tokens(tokens, offsets, indices)
        batch_tensor = torch.as_tensor(batch_array, dtype=torch.long, device=device)
        captures: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}

        def capture(name):
            def hook(value, _hook):
                counts[name] = counts.get(name, 0) + 1
                captures[name] = value.detach()
                return value

            return hook

        hooks = [
            (contract.FEATURE_HOOK, capture(contract.FEATURE_HOOK)),
            (contract.PRIMARY_REFERENCE_HOOK, capture(contract.PRIMARY_REFERENCE_HOOK)),
            *[(name, capture(name)) for name in contract.SECONDARY_REFERENCE_HOOKS if name != contract.FEATURE_HOOK],
        ]
        with torch.inference_mode():
            _ = model.run_with_hooks(batch_tensor, fwd_hooks=hooks)
        if set(counts) != {contract.FEATURE_HOOK, contract.PRIMARY_REFERENCE_HOOK, "blocks.31.hook_resid_pre"} or any(value != 1 for value in counts.values()):
            raise contract.ContractError("Phase-B hook fire closure differs")
        pre_vectorized, act_vectorized, final_pre, final_act = feature_values(
            captures[contract.FEATURE_HOOK], encoder, bias
        )
        if normal:
            pre_all, act_all = pre_vectorized, act_vectorized
        else:
            pre_all, act_all = feature_values_rowwise_all(
                captures[contract.FEATURE_HOOK], encoder, bias
            )
        last_user_positions = [
            manifest["rows"][index]["user_token_end_exclusive"] - 1
            for index in indices
        ]
        last_user_pre = torch.stack(
            [
                torch.dot(encoder, captures[contract.FEATURE_HOOK][offset, position]) + bias
                for offset, position in enumerate(last_user_positions)
            ]
        )
        ref_offsets = reference_rows(manifest, indices)
        ref_indices = torch.as_tensor([indices[offset] for offset in ref_offsets], dtype=torch.int64)
        empty = torch.empty((0, 2560), dtype=torch.bfloat16)
        shard_tensors = {
            "row_indices": torch.as_tensor(indices, dtype=torch.int64),
            "c_preactivation_all": pre_all.detach().cpu().contiguous(),
            "c_activation_all": act_all.detach().cpu().contiguous(),
            "c_preactivation_final": final_pre.detach().cpu().contiguous(),
            "c_activation_final": final_act.detach().cpu().contiguous(),
            "c_preactivation_last_user": last_user_pre.detach().cpu().contiguous(),
            "c_activation_last_user": torch.relu(last_user_pre).detach().cpu().contiguous(),
            "reference_row_indices": ref_indices,
            "reference_resid_post": (
                captures[contract.PRIMARY_REFERENCE_HOOK][ref_offsets, -1].cpu().contiguous()
                if ref_offsets
                else empty
            ),
            "reference_resid_pre": (
                captures["blocks.31.hook_resid_pre"][ref_offsets, -1].cpu().contiguous()
                if ref_offsets
                else empty
            ),
            "reference_mlp_hook_in": (
                captures[contract.FEATURE_HOOK][ref_offsets, -1].cpu().contiguous()
                if ref_offsets
                else empty
            ),
        }
        record = {
            "batch_id": batch_id,
            "row_indices": indices,
            "prompt_equivalents": prompt_equiv,
            "token_equivalents": token_equiv,
            **_write_shard(output_dir, batch_id, shard_tensors),
        }
        state["completed_batches"].append(record)
        state["completed_batches"].sort(key=lambda row: row["batch_id"])
        account["completed_unique_prompt_equivalents"] += prompt_equiv
        account["completed_unique_token_equivalents"] += token_equiv
        state["inflight"] = None
        contract.atomic_json(state_path, state)
        print(json.dumps({"batch_id": batch_id, "completed": len(state["completed_batches"])}), flush=True)
        del captures, pre_all, act_all, pre_vectorized, act_vectorized
        gc.collect()

    if not _terminalize_if_schedule_complete(
        state,
        scheduled=scheduled,
        plan=plan,
        state_path=state_path,
        decoder_cpu=decoder_cpu,
        probe_path=probe_path,
        preflight_proof=preflight_proof,
        dependency_proof=dependency_proof,
        private_preflight_proof=private_preflight_proof,
    ):
        raise contract.ContractError("Phase-B unique prompt closure differs")


def _torch_bytes_sha256(tensor) -> str:
    import hashlib

    raw = tensor.detach().contiguous().view(__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=pathlib.Path, required=True)
    parser.add_argument("--execution-plan", type=pathlib.Path, required=True)
    parser.add_argument("--execution-request", type=pathlib.Path, required=True)
    parser.add_argument("--allocation-provisional", type=pathlib.Path, required=True)
    parser.add_argument("--watchdog-arm", type=pathlib.Path, required=True)
    parser.add_argument("--allocation-ready", type=pathlib.Path, required=True)
    parser.add_argument("--session-binding", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = contract.load_authenticated_json(args.execution_plan, schema=contract.EXECUTION_PLAN_SCHEMA)
    contract.validate_execution_plan(plan)
    request = contract.load_authenticated_json(args.execution_request, schema=contract.EXECUTION_REQUEST_SCHEMA)
    contract.validate_execution_request(request, plan=plan)
    allocation_provisional = contract.load_authenticated_json(
        args.allocation_provisional,
        schema=contract.ALLOCATION_PROVISIONAL_RECEIPT_SCHEMA,
    )
    contract.validate_allocation_provisional_receipt(
        allocation_provisional, plan=plan, request=request
    )
    watchdog_arm = contract.load_authenticated_json(
        args.watchdog_arm, schema=contract.WATCHDOG_ARM_RECEIPT_SCHEMA
    )
    contract.validate_watchdog_arm_receipt(
        watchdog_arm,
        provisional=allocation_provisional,
        plan=plan,
        request=request,
    )
    allocation = contract.load_authenticated_json(
        args.allocation_ready, schema=contract.ALLOCATION_READY_GATE_SCHEMA
    )
    contract.validate_allocation_ready_gate(
        allocation,
        provisional=allocation_provisional,
        arm=watchdog_arm,
        plan=plan,
        request=request,
    )
    binding = contract.load_authenticated_json(
        args.session_binding, schema=contract.SESSION_BINDING_SCHEMA
    )
    hashes = contract.authenticate_plan_inputs(plan, project_root=args.project_root)
    role_paths = {
        role: args.project_root / record["path"] for role, record in plan["inputs"].items()
    }
    manifest = contract.load_authenticated_json(role_paths["corpus_manifest"], schema=contract.CORPUS_SCHEMA)
    schedule = contract.load_authenticated_json(role_paths["batch_schedule"], schema=contract.BATCH_SCHEDULE_SCHEMA)
    contract.validate_batch_schedule(schedule, corpus=manifest)
    tokens, offsets = load_token_bundle(role_paths["token_bundle"], manifest)
    if hashes["feature_bundle"] != contract.FEATURE_BUNDLE_SHA256 or hashes["feature_manifest"] != contract.FEATURE_MANIFEST_SHA256:
        raise contract.ContractError("Phase-B compact feature closure differs")
    execute(
        plan=plan,
        request=request,
        allocation=allocation,
        binding=binding,
        manifest=manifest,
        schedule=schedule,
        tokens=tokens,
        offsets=offsets,
        feature_bundle=role_paths["feature_bundle"],
        probe_path=role_paths["probe_direction"],
        output_dir=args.output_dir,
        input_hashes=hashes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
