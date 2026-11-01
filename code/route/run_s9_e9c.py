#!/usr/bin/env python3
"""Preflight or execute E9c's fixed-dose matched-risk route runner.

The frozen E9c plan is deliberately non-executable.  Its default preflight
path performs only local file, schema, hash, privacy, and accounting checks.
A later exact authorization receipt is required before this module may touch a
tokenizer, target model, accelerator, or prompt-transfer boundary.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import run_s9_calibration as base
from s9_contract import (
    METRICS,
    REFUSAL_TOKEN_IDS,
    canonical_sha256,
    sha256_file,
    sha256_text,
    validate_stamped_payload,
)
from s9_e9b_contract import validate_e9b_private_panel, validate_e9b_public_manifest
from s9_e9c_contract import (
    E9C_AUDIT_FORWARD_CEILING,
    E9C_INPUT_NAMES,
    E9C_PLAN_STATUS,
    E9C_PLANNED_SCIENTIFIC_FORWARDS,
    E9C_POSITIONS,
    E9C_REQUIRED_PROMPTS,
    E9C_REQUIRED_ROUTE_CELLS,
    E9C_REQUIRED_SENDER_WORLDS,
    E9C_RESULT_SEMANTICS_VERSION,
    E9C_RESULT_STATUS,
    E9C_RUNNING_STATUS,
    tensor_identity_sha256,
    validate_e9c_execution_plan,
    validate_e9c_source_bundle_manifest,
    validate_fixed_dose,
)


FINAL_POSITION = "final_prompt_position"
PENULTIMATE_POSITION = "penultimate_prompt_position"
AUTHORIZATION_RECEIPT_STATUS = "authorized_e9d_external_execution_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _position_index(position: str) -> int:
    if position == FINAL_POSITION:
        return -1
    if position == PENULTIMATE_POSITION:
        return -2
    raise ValueError(f"Unknown E9c token position: {position!r}")


def _sequence_view(activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim == 3:
        if activations.shape[0] != 1:
            raise ValueError("E9c position helpers require batch size one")
        return activations[0]
    if activations.ndim == 2:
        return activations
    raise ValueError(
        f"E9c position helpers require rank-two or rank-three state, got {activations.shape}"
    )


def position_vector(activations: torch.Tensor, position: str) -> torch.Tensor:
    sequence = _sequence_view(activations)
    if sequence.shape[0] < 1:
        raise ValueError("E9c final prompt position is unavailable")
    if position == PENULTIMATE_POSITION and sequence.shape[0] < 2:
        raise ValueError("E9c penultimate prompt position is unavailable")
    return sequence[_position_index(position)]


def replace_position_vector(
    activations: torch.Tensor,
    replacement: torch.Tensor,
    position: str,
) -> torch.Tensor:
    _ = position_vector(activations, position)
    output = activations.clone()
    value = replacement.to(device=output.device, dtype=output.dtype)
    index = _position_index(position)
    if output.ndim == 3:
        output[0, index] = value
    else:
        output[index] = value
    return output


def add_source_edit_final(
    activations: torch.Tensor,
    edit_vector: torch.Tensor,
) -> torch.Tensor:
    current = position_vector(activations, FINAL_POSITION)
    replacement = current + edit_vector.to(device=current.device, dtype=current.dtype)
    return replace_position_vector(activations, replacement, FINAL_POSITION)


def _source_edit_hook(edit_vector: torch.Tensor):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        return add_source_edit_final(activations, edit_vector)

    return hook


def _capture_positions(
    storage: dict[str, torch.Tensor],
    counts: list[int],
):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        counts.append(1)
        for position in E9C_POSITIONS:
            storage[position] = (
                position_vector(activations, position).detach().cpu().contiguous()
            )
        return activations

    return hook


def _capture_final(
    storage: dict[str, torch.Tensor],
    counts: list[int],
):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        counts.append(1)
        storage["final"] = (
            position_vector(activations, FINAL_POSITION).detach().cpu().contiguous()
        )
        return activations

    return hook


def _replacement_hook(
    replacement: torch.Tensor,
    *,
    expected_before: torch.Tensor,
    position: str,
    mismatch: list[float],
):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        current = position_vector(activations, position)
        expected = expected_before.to(device=current.device, dtype=current.dtype)
        mismatch.append(float((current.float() - expected.float()).abs().max().cpu()))
        return replace_position_vector(activations, replacement, position)

    return hook


def _diagnostics(logits: torch.Tensor) -> dict[str, float]:
    values = position_vector(logits, FINAL_POSITION).detach().float()
    top_values = torch.topk(values, k=2).values
    mask = torch.ones(values.shape[-1], dtype=torch.bool, device=values.device)
    mask[list(REFUSAL_TOKEN_IDS)] = False
    refusal_top = values[list(REFUSAL_TOKEN_IDS)].max()
    nonrefusal_top = values[mask].max()
    return {
        "final_logit_l2": float(values.norm().cpu()),
        "top1_minus_top2_logit_margin": float((top_values[0] - top_values[1]).cpu()),
        "refusal_top_minus_nonrefusal_top_logit_margin": float(
            (refusal_top - nonrefusal_top).cpu()
        ),
    }


@torch.inference_mode()
def _run_world(
    model,
    tokens: torch.Tensor,
    probe_direction: torch.Tensor,
    probe_bias: float,
    intervention_hooks: list[tuple[str, Any]],
) -> dict[str, Any]:
    receiver: dict[str, torch.Tensor] = {}
    final: dict[str, torch.Tensor] = {}
    receiver_counts: list[int] = []
    final_counts: list[int] = []
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[
            *intervention_hooks,
            (
                "blocks.32.mlp.hook_out",
                _capture_positions(receiver, receiver_counts),
            ),
            ("unembed.hook_pre", _capture_final(final, final_counts)),
        ],
    )
    if len(receiver_counts) != 1 or len(final_counts) != 1:
        raise RuntimeError("E9c declared cache hook did not fire exactly once")
    final_normalized = final["final"].to(device=logits.device)
    metrics = base.detached_metric_values(
        base.differentiable_metrics(
            logits, final_normalized, probe_direction, probe_bias
        )
    )
    return {
        "metrics": metrics,
        "diagnostics": _diagnostics(logits),
        "receiver_output": receiver,
    }


@torch.inference_mode()
def _run_endpoints(
    model,
    tokens: torch.Tensor,
    probe_direction: torch.Tensor,
    probe_bias: float,
    hooks: list[tuple[str, Any]],
) -> dict[str, Any]:
    final: dict[str, torch.Tensor] = {}
    counts: list[int] = []
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[*hooks, ("unembed.hook_pre", _capture_final(final, counts))],
    )
    if len(counts) != 1:
        raise RuntimeError("E9c endpoint hook did not fire exactly once")
    final_normalized = final["final"].to(device=logits.device)
    return {
        "metrics": base.detached_metric_values(
            base.differentiable_metrics(
                logits, final_normalized, probe_direction, probe_bias
            )
        ),
        "diagnostics": _diagnostics(logits),
    }


def _tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    materialized = tensor.detach().contiguous().cpu()
    return hashlib.sha256(materialized.view(torch.uint8).numpy().tobytes()).hexdigest()


def _fixed_edit_vectors(
    *,
    nominated_decoder: torch.Tensor,
    control_decoder: torch.Tensor,
    nominated_activation_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if nominated_decoder.dtype != torch.bfloat16 or control_decoder.dtype != torch.bfloat16:
        raise ValueError("E9c decoder rows must retain BF16 storage semantics")
    if nominated_decoder.ndim != 1 or control_decoder.shape != nominated_decoder.shape:
        raise ValueError("E9c source decoder rows have incompatible shapes")
    delta = float(nominated_activation_delta)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("E9c nominated fixed dose must be finite and positive")
    nominated = nominated_decoder.float() * delta
    control_norm = float(control_decoder.float().norm())
    nominated_l2 = float(nominated.norm())
    if control_norm <= 0 or nominated_l2 <= 0:
        raise ValueError("E9c source decoder norm is zero")
    control_delta = nominated_l2 / control_norm
    control = control_decoder.float() * control_delta
    return nominated, control, control_delta


def compute_fixed_dose(
    *,
    nominated_decoder: torch.Tensor,
    control_decoder: torch.Tensor,
    nominated_activation_delta: float,
) -> dict[str, Any]:
    nominated, control, control_delta = _fixed_edit_vectors(
        nominated_decoder=nominated_decoder,
        control_decoder=control_decoder,
        nominated_activation_delta=nominated_activation_delta,
    )
    nominated_l2 = float(nominated.norm())
    control_l2 = float(control.norm())
    relative = abs(control_l2 - nominated_l2) / nominated_l2
    nominated_bf16 = nominated.to(torch.bfloat16)
    control_bf16 = control.to(torch.bfloat16)
    payload = {
        "fixed_dose_id": "e9c:fixed-dose:v1",
        "nominated_source_feature": "L31:F49541",
        "control_source_feature": "L31:F139261",
        "nominated_activation_delta": float(nominated_activation_delta),
        "control_activation_delta": control_delta,
        "nominal_nominated_edit_l2": nominated_l2,
        "nominal_control_edit_l2": control_l2,
        "nominal_equal_l2_relative_error": relative,
        "realized_bfloat16_nominated_edit_l2": float(nominated_bf16.float().norm()),
        "realized_bfloat16_control_edit_l2": float(control_bf16.float().norm()),
        "nominated_edit_tensor_sha256": tensor_identity_sha256(nominated_bf16),
        "control_edit_tensor_sha256": tensor_identity_sha256(control_bf16),
        "prompt_activation_screen_used": False,
        "same_fixed_dose_for_all_24_prompts": True,
    }
    validate_fixed_dose(payload)
    return payload


def assert_runtime_fixed_dose_matches_plan(
    runtime: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    summary_rtol: float = 1e-6,
) -> dict[str, Any]:
    """Authenticate identical applied BF16 edits despite CPU norm drift.

    PyTorch CPU reductions are not bitwise portable across architectures and
    versions.  The intervention hook casts the edit to the model's BF16 dtype,
    so the exact BF16 tensor identities are the authoritative execution
    invariant.  Scalar norm summaries remain bounded diagnostics.
    """

    validate_fixed_dose(runtime)
    validate_fixed_dose(planned)
    exact_fields = (
        "fixed_dose_id",
        "nominated_source_feature",
        "control_source_feature",
        "nominated_activation_delta",
        "nominated_edit_tensor_sha256",
        "control_edit_tensor_sha256",
        "prompt_activation_screen_used",
        "same_fixed_dose_for_all_24_prompts",
    )
    for field in exact_fields:
        if runtime.get(field) != planned.get(field):
            label = "tensor hash" if "tensor_sha256" in field else "identity"
            raise ValueError(f"E9c runtime fixed-dose {label} differs: {field}")

    if not math.isfinite(summary_rtol) or summary_rtol <= 0:
        raise ValueError("E9c runtime fixed-dose summary tolerance is invalid")
    summary_fields = (
        "control_activation_delta",
        "nominal_nominated_edit_l2",
        "nominal_control_edit_l2",
        "realized_bfloat16_nominated_edit_l2",
        "realized_bfloat16_control_edit_l2",
    )
    relative_drifts: dict[str, float] = {}
    for field in summary_fields:
        observed = float(runtime[field])
        expected = float(planned[field])
        scale = max(abs(expected), abs(observed), 1e-30)
        relative_drifts[field] = abs(observed - expected) / scale
    maximum = max(relative_drifts.values(), default=0.0)
    if maximum > summary_rtol:
        raise ValueError("E9c runtime fixed-dose summary drift exceeds tolerance")

    runtime_equal_l2_error = float(runtime["nominal_equal_l2_relative_error"])
    planned_equal_l2_error = float(planned["nominal_equal_l2_relative_error"])
    equal_l2_error_drift = abs(runtime_equal_l2_error - planned_equal_l2_error)
    if equal_l2_error_drift > summary_rtol:
        raise ValueError("E9c runtime equal-L2 diagnostic drift exceeds tolerance")
    return {
        "validation": (
            "exact_bfloat16_edit_hashes_with_cross_platform_summary_tolerance_v1"
        ),
        "exact_bfloat16_edit_hashes_match": True,
        "relative_summary_drift_limit": float(summary_rtol),
        "relative_summary_drifts": relative_drifts,
        "maximum_relative_summary_drift": maximum,
        "planned_equal_l2_relative_error": planned_equal_l2_error,
        "runtime_equal_l2_relative_error": runtime_equal_l2_error,
        "equal_l2_error_absolute_drift": equal_l2_error_drift,
    }


def route_effects(
    world_metrics: Mapping[str, Mapping[str, float]],
    *,
    invariant_atol: float,
) -> dict[str, dict[str, float]]:
    if set(world_metrics) != {"Y0", "YS", "YS_R0", "YR_S"}:
        raise ValueError("E9c route worlds differ")
    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        y0 = float(world_metrics["Y0"][metric])
        ys = float(world_metrics["YS"][metric])
        ys_r0 = float(world_metrics["YS_R0"][metric])
        yr_s = float(world_metrics["YR_S"][metric])
        values = (y0, ys, ys_r0, yr_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("E9c route world contains a non-finite endpoint")
        ts = ys - y0
        dr = ys - ys_r0
        br = ys_r0 - y0
        qr = yr_s - y0
        error = abs(ts - dr - br)
        if error > invariant_atol:
            raise ValueError("E9c signed route accounting invariant failed")
        result[metric] = {
            "Y0": y0,
            "YS": ys,
            "YS_R0": ys_r0,
            "YR_S": yr_s,
            "T_S": ts,
            "D_R": dr,
            "B_R": br,
            "Q_R": qr,
            "invariant_error": error,
        }
    return result


def completed_scientific_forwards(payload: Mapping[str, Any]) -> int:
    prompts = payload.get("prompt_records", [])
    senders = payload.get("sender_worlds", [])
    cells = payload.get("route_cells", [])
    if not all(isinstance(value, list) for value in (prompts, senders, cells)):
        raise ValueError("E9c forward-accounting collections are malformed")
    inflight = payload.get("resume_state", {}).get("inflight_operation_id")
    return len(prompts) + len(senders) + 2 * len(cells) + int(inflight is not None)


def _refresh_accounting(payload: dict[str, Any]) -> None:
    completed = len(payload["prompt_records"]) + len(payload["sender_worlds"]) + 2 * len(
        payload["route_cells"]
    )
    audit = int(payload["forward_accounting"].get("audit_forwards", 0))
    inflight = int(payload["resume_state"].get("inflight_operation_id") is not None)
    payload["forward_accounting"].update(
        {
            "planned_scientific": E9C_PLANNED_SCIENTIFIC_FORWARDS,
            "completed_scientific": completed,
            "audit_forwards": audit,
            "cumulative_total_across_resumes": completed + audit + inflight,
            "audit_ceiling": E9C_AUDIT_FORWARD_CEILING,
            "audit_ceiling_exceeded": completed + audit + inflight
            > E9C_AUDIT_FORWARD_CEILING,
            "counter_reset_on_resume": False,
        }
    )


def _begin_forward(payload: dict[str, Any], output_path: Path, operation_id: str) -> None:
    if payload["resume_state"].get("inflight_operation_id") is not None:
        raise ValueError("E9c cannot begin a second in-flight forward")
    payload["resume_state"]["inflight_operation_id"] = operation_id
    _refresh_accounting(payload)
    _atomic_json(output_path, payload)


def _complete_forward(payload: dict[str, Any]) -> None:
    payload["resume_state"]["inflight_operation_id"] = None
    _refresh_accounting(payload)


def _reconcile_interrupted_forward(payload: dict[str, Any]) -> None:
    operation_id = payload["resume_state"].get("inflight_operation_id")
    if operation_id is None:
        return
    # A rescue checkpoint is written only after its paired restoration forward
    # has completed.  Conservatively charge both attempts if that checkpoint is
    # interrupted; this can overcount by one, but can never hide a forward.
    interrupted_attempts = 2 if str(operation_id).startswith("route_rescue:") else 1
    payload["forward_accounting"]["audit_forwards"] = int(
        payload["forward_accounting"].get("audit_forwards", 0)
    ) + interrupted_attempts
    payload["resume_state"]["inflight_operation_id"] = None
    _refresh_accounting(payload)
    if payload["forward_accounting"]["audit_ceiling_exceeded"]:
        raise ValueError("E9c interrupted-forward audit ceiling exceeded")


def assert_public_safe(value: Any) -> None:
    forbidden_keys = {
        "prompt",
        "core_request",
        "completion",
        "completion_text",
        "chat_text",
        "input_token_ids",
        "token_text",
        "receiver_tensor",
        "raw_tensor",
    }
    if isinstance(value, Mapping):
        if forbidden_keys.intersection(value):
            raise ValueError("E9c public output contains private text/token/tensor fields")
        for child in value.values():
            assert_public_safe(child)
    elif isinstance(value, list):
        for child in value:
            assert_public_safe(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("E9c public output contains a non-finite value")


def load_private_prompts(
    private_path: Path,
    manifest_path: Path,
    execution_plan: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    private_path = Path(private_path)
    manifest_path = Path(manifest_path)
    private = json.loads(private_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_e9b_private_panel(private)
    validate_e9b_public_manifest(manifest)
    if sha256_file(private_path) != execution_plan["inputs"]["e9b_private_panel"][
        "sha256"
    ]:
        raise ValueError("E9c private prompt panel hash differs")
    if sha256_file(manifest_path) != execution_plan["inputs"]["e9b_public_manifest"][
        "sha256"
    ]:
        raise ValueError("E9c public panel manifest hash differs")
    planned = {row["prompt_key"]: row for row in execution_plan["prompt_records"]}
    public = {
        pair[risk]["prompt_key"]: {
            **pair[risk],
            "pair_id": pair["pair_id"],
            "topic_category": pair["topic_category"],
        }
        for pair in manifest["topic_pairs"]
        for risk in ("harmful", "benign")
    }
    selected: dict[str, dict[str, str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for pair in private["topic_pairs"]:
        for risk in ("harmful", "benign"):
            record = pair[risk]
            key = record["prompt_key"]
            if key not in planned or key not in public:
                raise ValueError("E9c private prompt identity is outside the frozen panel")
            if sha256_text(record["core_request"]) != public[key]["core_sha256"]:
                raise ValueError("E9c private core bytes differ from their public hash")
            if sha256_text(record["prompt"]) != public[key]["prompt_sha256"]:
                raise ValueError("E9c private prompt bytes differ from their public hash")
            expected = planned[key]
            for field in (
                "pair_id",
                "topic_category",
                "intent_id",
                "intent_category",
                "risk_class",
                "prompt_key",
                "frame_id",
                "core_sha256",
                "prompt_sha256",
            ):
                observed = (
                    pair["pair_id"]
                    if field == "pair_id"
                    else pair["topic_category"]
                    if field == "topic_category"
                    else public[key].get(field)
                )
                if expected[field] != observed:
                    raise ValueError(f"E9c private/public plan metadata differs: {field}")
            selected[key] = {"prompt": record["prompt"]}
            metadata[key] = dict(expected)
    if set(selected) != set(planned) or len(selected) != E9C_REQUIRED_PROMPTS:
        raise ValueError("E9c private prompt closure differs")
    return selected, metadata


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def assert_execution_authorized(
    execution_plan: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any] | None = None,
) -> None:
    authorization = execution_plan.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PermissionError("E9c execution authorization is missing")
    if authorization_receipt is None:
        if authorization.get("execution_allowed") is not True:
            raise PermissionError(
                "E9c execution remains closed; an exact E9d authorization receipt is required"
            )
        raise PermissionError("E9c execution authorization receipt is missing")
    validate_stamped_payload(authorization_receipt)
    expected = {
        "status": AUTHORIZATION_RECEIPT_STATUS,
        "execution_plan_payload_sha256": execution_plan.get("payload_sha256"),
        "prompt_transfer_allowed": True,
        "tokenizer_load_allowed": True,
        "target_model_load_allowed": True,
        "model_forward_allowed": True,
        "gpu_allocation_allowed": True,
        "generation_allowed": False,
        "backward_pass_allowed": False,
        "e7d_allowed": False,
        "new_scientific_cells_allowed": False,
        "outcome_based_panel_replacement_allowed": False,
    }
    if any(authorization_receipt.get(key) != value for key, value in expected.items()):
        raise PermissionError("E9d execution authorization receipt differs")
    reference = authorization_receipt.get("user_confirmation_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise PermissionError("E9d execution authorization receipt lacks user confirmation")


def _load_source_decoders(
    bundle_path: Path,
    manifest_path: Path,
    e9a_design: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_e9c_source_bundle_manifest(
        manifest, e9a_design=e9a_design, bundle_path=bundle_path
    )
    with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
        indices = handle.get_tensor("feature_indices")
        decoders = handle.get_tensor("decoder_rows")
    return {
        f"L31:F{int(feature_index)}": decoders[row_index].detach().contiguous()
        for row_index, feature_index in enumerate(indices.tolist())
    }


def _resolve_execution_inputs(
    execution_plan: Mapping[str, Any], input_dir: Path | None
) -> dict[str, Path]:
    inputs = execution_plan.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(E9C_INPUT_NAMES):
        raise ValueError("E9c execution input closure differs")
    paths: dict[str, Path] = {}
    for name in E9C_INPUT_NAMES:
        record = inputs[name]
        local = Path(record["path"])
        if not local.is_file() and input_dir is not None:
            local = Path(input_dir) / record["upload_name"]
        if not local.is_file():
            raise FileNotFoundError(f"E9c execution input is missing: {name}")
        if sha256_file(local) != record["sha256"]:
            raise ValueError(f"E9c execution input hash differs: {name}")
        paths[name] = local.resolve()
    return paths


def _apply_runtime_determinism(runtime: Mapping[str, Any]) -> dict[str, Any]:
    contract = runtime["determinism"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract[
        "cublas_workspace_config"
    ]:
        raise RuntimeError("E9c deterministic workspace configuration differs")
    torch.manual_seed(int(contract["torch_manual_seed"]))
    torch.cuda.manual_seed_all(int(contract["cuda_manual_seed_all"]))
    torch.backends.cudnn.benchmark = bool(contract["cudnn_benchmark"])
    torch.backends.cudnn.allow_tf32 = bool(contract["allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(contract["allow_tf32"])
    torch.use_deterministic_algorithms(bool(contract["deterministic_algorithms"]))
    torch.set_grad_enabled(False)
    return dict(contract)


def _model_snapshot_proof(
    snapshot_path: Path,
    *,
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    snapshot_path = Path(snapshot_path)
    if snapshot_path.name != revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("E9c model snapshot revision differs")
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(snapshot_path).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("E9c model snapshot file closure is empty")
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": revision,
        "file_count": len(files),
        "file_closure_sha256": canonical_sha256({"files": files}),
        "files": files,
        "verified_before_first_forward": True,
    }


def _runtime_fingerprint(
    runtime: Mapping[str, Any], packages: Mapping[str, str]
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_gib": properties.total_memory / 1024**3,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": driver,
        "packages": dict(packages),
        "dtype": runtime["dtype"],
        "random_seed": runtime["random_seed"],
        "determinism": dict(runtime["determinism"]),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _state_record(tensor: torch.Tensor, raw_key: str) -> dict[str, Any]:
    materialized = tensor.detach().contiguous().cpu()
    return {
        "raw_key": raw_key,
        "dtype": str(materialized.dtype).removeprefix("torch."),
        "shape": list(materialized.shape),
        "tensor_identity_sha256": tensor_identity_sha256(materialized),
        "tensor_bytes_sha256": _tensor_bytes_sha256(materialized),
    }


def _store_receiver_states(
    world: Mapping[str, Any],
    *,
    prefix: str,
    raw_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for position in E9C_POSITIONS:
        key = f"{prefix}.receiver_output.{position}"
        tensor = world["receiver_output"][position].detach().contiguous().cpu()
        raw_tensors[key] = tensor
        records[position] = _state_record(tensor, key)
    return records


def _restore_receiver_states(
    record: Mapping[str, Any], raw_tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for position in E9C_POSITIONS:
        state = record[position]
        key = state["raw_key"]
        if key not in raw_tensors:
            raise ValueError("E9c private receiver state is missing")
        tensor = raw_tensors[key]
        if tensor_identity_sha256(tensor) != state["tensor_identity_sha256"]:
            raise ValueError("E9c private receiver tensor identity differs")
        result[position] = tensor
    return result


def _save_raw_checkpoint(
    base_path: Path,
    raw_tensors: Mapping[str, torch.Tensor],
    *,
    execution_plan_sha256: str,
) -> dict[str, Any]:
    if not raw_tensors:
        raise ValueError("E9c private raw checkpoint cannot be empty")
    from safetensors.torch import save_file

    base_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = base_path.with_suffix(base_path.suffix + ".tmp")
    save_file(
        {key: raw_tensors[key].detach().contiguous().cpu() for key in sorted(raw_tensors)},
        str(temporary),
        metadata={"execution_plan_payload_sha256": execution_plan_sha256},
    )
    digest = sha256_file(temporary)
    content_path = base_path.with_name(f"{base_path.stem}.{digest}{base_path.suffix}")
    if content_path.exists():
        if sha256_file(content_path) != digest:
            raise ValueError("E9c content-addressed private checkpoint differs")
        temporary.unlink()
    else:
        temporary.replace(content_path)
    return {
        "path": content_path.name,
        "sha256": digest,
        "tensor_count": len(raw_tensors),
        "public": False,
        "execution_plan_payload_sha256": execution_plan_sha256,
    }


def _raw_checkpoint_path(base_path: Path, record: Mapping[str, Any]) -> Path:
    name = record.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("E9c private checkpoint path is malformed")
    return base_path.parent / name


def _checkpoint(
    *,
    output: dict[str, Any],
    output_path: Path,
    raw_tensors: Mapping[str, torch.Tensor],
    raw_base_path: Path,
    execution_plan_sha256: str,
) -> None:
    _complete_forward(output)
    output["private_raw_tensor_bundle"] = _save_raw_checkpoint(
        raw_base_path,
        raw_tensors,
        execution_plan_sha256=execution_plan_sha256,
    )
    assert_public_safe(output)
    _atomic_json(output_path, output)


def _receiver_state_summary(
    baseline: torch.Tensor,
    sender: torch.Tensor,
    baseline_record: Mapping[str, Any],
    sender_record: Mapping[str, Any],
) -> dict[str, Any]:
    delta = sender.float() - baseline.float()
    return {
        "baseline_tensor_identity_sha256": baseline_record[
            "tensor_identity_sha256"
        ],
        "sender_tensor_identity_sha256": sender_record["tensor_identity_sha256"],
        "state_delta_l2": float(delta.norm()),
        "state_delta_max_abs": float(delta.abs().max()),
        "state_exactly_equal": bool(torch.equal(baseline, sender)),
    }


def _validate_complete_result(
    payload: Mapping[str, Any], execution_plan: Mapping[str, Any]
) -> None:
    validate_stamped_payload(payload)
    if payload.get("status") != E9C_RESULT_STATUS:
        raise ValueError("E9c result status differs")
    if payload.get("analysis_semantics_version") != E9C_RESULT_SEMANTICS_VERSION:
        raise ValueError("E9c result semantics differ")
    dose_audit = payload.get("runtime_fixed_dose_portability_audit")
    portability_limit = float(
        execution_plan["numerical_contract"][
            "cross_platform_float_summary_relative_tolerance"
        ]
    )
    if (
        not isinstance(dose_audit, Mapping)
        or dose_audit.get("exact_bfloat16_edit_hashes_match") is not True
        or float(dose_audit.get("maximum_relative_summary_drift", math.inf))
        > portability_limit
    ):
        raise ValueError("E9c result fixed-dose portability audit differs")
    if payload.get("execution_plan_payload_sha256") != execution_plan.get(
        "payload_sha256"
    ):
        raise ValueError("E9c result execution-plan identity differs")
    if len(payload.get("prompt_records", [])) != E9C_REQUIRED_PROMPTS:
        raise ValueError("E9c result prompt closure differs")
    if len(payload.get("sender_worlds", [])) != E9C_REQUIRED_SENDER_WORLDS:
        raise ValueError("E9c result sender closure differs")
    if len(payload.get("route_cells", [])) != E9C_REQUIRED_ROUTE_CELLS:
        raise ValueError("E9c result route-cell closure differs")
    accounting = payload.get("forward_accounting")
    if not isinstance(accounting, Mapping) or accounting.get(
        "completed_scientific"
    ) != E9C_PLANNED_SCIENTIFIC_FORWARDS or accounting.get(
        "audit_ceiling_exceeded"
    ) is not False:
        raise ValueError("E9c final forward accounting differs")
    planned_cells = {row["cell_id"]: row for row in execution_plan["route_cells"]}
    observed_cells = {row.get("cell_id"): row for row in payload["route_cells"]}
    if set(observed_cells) != set(planned_cells):
        raise ValueError("E9c result cell identities differ")
    for cell_id, planned in planned_cells.items():
        observed = observed_cells[cell_id]
        for field in planned:
            if observed.get(field) != planned[field]:
                raise ValueError(f"E9c result cell metadata differs: {cell_id}:{field}")
        for metric in METRICS:
            endpoints = observed.get("endpoints", {}).get(metric)
            if not isinstance(endpoints, Mapping):
                raise ValueError("E9c result endpoint is missing")
            if abs(float(endpoints["T_S"]) - float(endpoints["D_R"]) - float(endpoints["B_R"])) > 1e-7:
                raise ValueError("E9c result signed accounting identity failed")
    planned_prompts = {
        row["prompt_key"]: row for row in execution_plan["prompt_records"]
    }
    observed_prompts = {
        row.get("prompt_key"): row for row in payload["prompt_records"]
    }
    if set(observed_prompts) != set(planned_prompts):
        raise ValueError("E9c result prompt identities differ")
    for prompt_key, planned in planned_prompts.items():
        if any(observed_prompts[prompt_key].get(field) != value for field, value in planned.items()):
            raise ValueError(f"E9c result prompt metadata differs: {prompt_key}")
    planned_senders = {
        row["sender_world_id"]: row for row in execution_plan["sender_worlds"]
    }
    observed_senders = {
        row.get("sender_world_id"): row for row in payload["sender_worlds"]
    }
    if set(observed_senders) != set(planned_senders):
        raise ValueError("E9c result sender identities differ")
    for sender_id, planned in planned_senders.items():
        if any(observed_senders[sender_id].get(field) != value for field, value in planned.items()):
            raise ValueError(f"E9c result sender metadata differs: {sender_id}")
    assert_public_safe(payload)


def _preflight(
    execution_plan_path: Path,
    *,
    input_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    execution_plan = json.loads(execution_plan_path.read_text(encoding="utf-8"))
    paths = _resolve_execution_inputs(execution_plan, input_dir)
    design = json.loads(paths["e9a_design"].read_text(encoding="utf-8"))
    private = json.loads(paths["e9b_private_panel"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["e9b_public_manifest"].read_text(encoding="utf-8"))
    validate_e9c_execution_plan(execution_plan, design, private, manifest)
    source_manifest = json.loads(
        paths["source_decoder_manifest"].read_text(encoding="utf-8")
    )
    validate_e9c_source_bundle_manifest(
        source_manifest,
        e9a_design=design,
        bundle_path=paths["source_decoder_bundle"],
    )
    selected, metadata = load_private_prompts(
        paths["e9b_private_panel"], paths["e9b_public_manifest"], execution_plan
    )
    if len(selected) != 24 or len(metadata) != 24:
        raise ValueError("E9c preflight private prompt closure differs")
    return execution_plan, design, private, manifest, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-tensors", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    execution_plan, design, _, manifest, paths = _preflight(
        args.execution_plan, input_dir=args.input_dir
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "e9c_cpu_preflight_passed_without_target_model_contact",
                    "execution_plan_payload_sha256": execution_plan["payload_sha256"],
                    "prompt_records": len(execution_plan["prompt_records"]),
                    "sender_worlds": len(execution_plan["sender_worlds"]),
                    "route_cells": len(execution_plan["route_cells"]),
                    "planned_scientific_forwards": execution_plan["forward_contract"][
                        "planned_scientific"
                    ],
                    "execution_authorized": False,
                },
                indent=2,
            )
        )
        return
    if args.execution_authorization is None:
        assert_execution_authorized(execution_plan)
    receipt = json.loads(args.execution_authorization.read_text(encoding="utf-8"))
    assert_execution_authorized(execution_plan, receipt)
    if args.output is None or args.raw_tensors is None:
        raise ValueError("E9c authorized execution requires output and private raw paths")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an authorized E9c execution")

    packages = {
        package: importlib.metadata.version(package)
        for package in execution_plan["runtime_contract"]["package_versions"]
    }
    if packages != execution_plan["runtime_contract"]["package_versions"]:
        raise ValueError("E9c runtime package contract differs")
    device_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if not any(
        name in device_name
        for name in execution_plan["runtime_contract"]["preferred_accelerators"]
    ):
        raise ValueError(f"Unsupported E9c accelerator: {device_name}")
    if memory < float(execution_plan["runtime_contract"]["minimum_cuda_memory_gib"]):
        raise ValueError("E9c accelerator memory is below contract")
    _apply_runtime_determinism(execution_plan["runtime_contract"])
    runtime = _runtime_fingerprint(execution_plan["runtime_contract"], packages)

    decoders = _load_source_decoders(
        paths["source_decoder_bundle"], paths["source_decoder_manifest"], design
    )
    runtime_dose = compute_fixed_dose(
        nominated_decoder=decoders["L31:F49541"],
        control_decoder=decoders["L31:F139261"],
        nominated_activation_delta=18.625,
    )
    dose_portability_audit = assert_runtime_fixed_dose_matches_plan(
        runtime_dose,
        execution_plan["fixed_dose"],
        summary_rtol=float(
            execution_plan["numerical_contract"][
                "cross_platform_float_summary_relative_tolerance"
            ]
        ),
    )
    dose = copy.deepcopy(execution_plan["fixed_dose"])
    nominated_edit, control_edit, _ = _fixed_edit_vectors(
        nominated_decoder=decoders["L31:F49541"],
        control_decoder=decoders["L31:F139261"],
        nominated_activation_delta=18.625,
    )
    edit_by_arm = {
        "nominated_c": nominated_edit,
        "equal_l2_control": control_edit,
    }

    from transformer_lens import HookedTransformer

    device = torch.device("cuda")
    model = HookedTransformer.from_pretrained(
        execution_plan["model"]["id"],
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        dtype=torch.bfloat16,
        revision=execution_plan["model"]["revision"],
    )
    for block in model.blocks:
        block.mlp = base.ExactHookedMLP(block.mlp)
    model.unembed = base.ExactHookedUnembed(model.unembed)
    model.setup()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=execution_plan["model"]["id"],
            revision=execution_plan["model"]["revision"],
            local_files_only=True,
        )
    )
    model_snapshot = _model_snapshot_proof(
        snapshot,
        repo_id=execution_plan["model"]["id"],
        revision=execution_plan["model"]["revision"],
    )
    for key in decoders:
        decoders[key] = decoders[key].to(device=device, dtype=torch.bfloat16)
    edit_by_arm = {
        arm: vector.to(device=device, dtype=torch.float32)
        for arm, vector in edit_by_arm.items()
    }
    if any(vector.numel() != int(model.cfg.d_model) for vector in edit_by_arm.values()):
        raise ValueError("E9c source decoder width differs from model d_model")
    probe = np.load(paths["probe_direction"])
    probe_direction = torch.tensor(
        probe["direction_unit"].astype(np.float32), device=device, dtype=torch.float32
    )
    probe_bias = float(probe["bias_unit"])

    selected, prompt_metadata = load_private_prompts(
        paths["e9b_private_panel"], paths["e9b_public_manifest"], execution_plan
    )
    tokens_by_prompt: dict[str, torch.Tensor] = {}
    runtime_prompt_metadata = copy.deepcopy(prompt_metadata)
    for key in sorted(selected):
        chat = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": selected[key]["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = base.tokenize_chat(model, chat)
        if tokens.shape[-1] < 2:
            raise ValueError("E9c prompt has no penultimate token position")
        tokens_by_prompt[key] = tokens
        runtime_prompt_metadata[key].update(
            {
                "chat_text_sha256": sha256_text(chat),
                "input_token_count": int(tokens.shape[-1]),
                "input_token_tensor_sha256": tensor_identity_sha256(tokens),
            }
        )

    authorization_receipt_sha256 = sha256_file(args.execution_authorization)
    input_hashes = {name: execution_plan["inputs"][name]["sha256"] for name in E9C_INPUT_NAMES}
    prior: dict[str, Any] | None = None
    if args.output.exists():
        if not args.resume:
            raise ValueError("E9c output exists; use --resume")
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("status") == E9C_RESULT_STATUS:
            _validate_complete_result(prior, execution_plan)
            raw_record = prior.get("private_raw_tensor_bundle")
            if not isinstance(raw_record, Mapping):
                raise ValueError("Completed E9c result lacks private tensor metadata")
            raw_path = _raw_checkpoint_path(args.raw_tensors, raw_record)
            if not raw_path.is_file() or sha256_file(raw_path) != raw_record.get(
                "sha256"
            ):
                raise ValueError("Completed E9c private tensor bundle differs")
            print(json.dumps({"status": "already_complete"}, indent=2))
            return
        if prior.get("status") != E9C_RUNNING_STATUS:
            raise ValueError("E9c partial result status differs")
        if prior.get("execution_plan_payload_sha256") != execution_plan["payload_sha256"]:
            raise ValueError("E9c resumed execution-plan identity differs")
        if prior.get("runtime") != runtime or prior.get("model_snapshot_proof") != model_snapshot:
            raise ValueError("E9c resumed runtime/model snapshot differs")
        if prior.get("authorization_receipt_sha256") != authorization_receipt_sha256:
            raise ValueError("E9c resumed authorization receipt differs")

    if prior is None:
        output: dict[str, Any] = {
            "schema_version": 1,
            "analysis_semantics_version": E9C_RESULT_SEMANTICS_VERSION,
            "status": E9C_RUNNING_STATUS,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "completed_at_utc": None,
            "execution_plan_payload_sha256": execution_plan["payload_sha256"],
            "execution_plan_file_sha256": sha256_file(args.execution_plan),
            "authorization_receipt_sha256": authorization_receipt_sha256,
            "runtime": runtime,
            "model_snapshot_proof": model_snapshot,
            "execution_input_hashes": input_hashes,
            "fixed_dose": dose,
            "runtime_fixed_dose_portability_audit": dose_portability_audit,
            "prompt_records": [],
            "sender_worlds": [],
            "route_cells": [],
            "forward_accounting": {
                "planned_scientific": E9C_PLANNED_SCIENTIFIC_FORWARDS,
                "completed_scientific": 0,
                "audit_forwards": 0,
                "cumulative_total_across_resumes": 0,
                "audit_ceiling": E9C_AUDIT_FORWARD_CEILING,
                "audit_ceiling_exceeded": False,
                "counter_reset_on_resume": False,
            },
            "resume_state": {"inflight_operation_id": None},
            "private_raw_tensor_bundle": None,
        }
        raw_tensors: dict[str, torch.Tensor] = {}
        _atomic_json(args.output, output)
    else:
        output = prior
        if output.get("runtime_fixed_dose_portability_audit") != dose_portability_audit:
            raise ValueError("E9c resumed fixed-dose portability audit differs")
        _reconcile_interrupted_forward(output)
        raw_record = output.get("private_raw_tensor_bundle")
        if raw_record is None:
            if completed_scientific_forwards(output) != 0:
                raise ValueError("E9c partial output lacks private receiver states")
            raw_tensors = {}
        else:
            raw_path = _raw_checkpoint_path(args.raw_tensors, raw_record)
            if not raw_path.is_file() or sha256_file(raw_path) != raw_record["sha256"]:
                raise ValueError("E9c private checkpoint differs")
            raw_tensors = dict(load_file(str(raw_path), device="cpu"))
        _atomic_json(args.output, output)

    baselines: dict[str, dict[str, Any]] = {}
    for row in output["prompt_records"]:
        key = row["prompt_key"]
        expected = runtime_prompt_metadata[key]
        for field in ("chat_text_sha256", "input_token_count", "input_token_tensor_sha256"):
            if row[field] != expected[field]:
                raise ValueError("E9c resumed prompt tokenization differs")
        baselines[key] = {
            "metrics": row["metrics"],
            "diagnostics": row["diagnostics"],
            "receiver_output": _restore_receiver_states(row["state_records"], raw_tensors),
        }

    for index, key in enumerate(sorted(selected), 1):
        if key in baselines:
            continue
        _begin_forward(output, args.output, f"prompt_cache:{key}")
        world = _run_world(model, tokens_by_prompt[key], probe_direction, probe_bias, [])
        state_records = _store_receiver_states(
            world, prefix=f"prompt.{key}", raw_tensors=raw_tensors
        )
        row = {
            **runtime_prompt_metadata[key],
            "metrics": world["metrics"],
            "diagnostics": world["diagnostics"],
            "state_records": state_records,
        }
        output["prompt_records"].append(row)
        output["prompt_records"].sort(key=lambda value: value["prompt_key"])
        baselines[key] = world
        _checkpoint(
            output=output,
            output_path=args.output,
            raw_tensors=raw_tensors,
            raw_base_path=args.raw_tensors,
            execution_plan_sha256=execution_plan["payload_sha256"],
        )
        print(f"prompt cache {index}/{E9C_REQUIRED_PROMPTS}: {key}", flush=True)

    senders: dict[tuple[str, str], dict[str, Any]] = {}
    for row in output["sender_worlds"]:
        senders[(row["intent_id"], row["arm"])] = {
            "metrics": row["metrics"],
            "diagnostics": row["diagnostics"],
            "receiver_output": _restore_receiver_states(row["state_records"], raw_tensors),
        }
    for index, planned in enumerate(execution_plan["sender_worlds"], 1):
        identity = (planned["intent_id"], planned["arm"])
        if identity in senders:
            continue
        key = planned["target_prompt_key"]
        vector = edit_by_arm[planned["arm"]]
        _begin_forward(output, args.output, f"sender_world:{planned['sender_world_id']}")
        world = _run_world(
            model,
            tokens_by_prompt[key],
            probe_direction,
            probe_bias,
            [("blocks.31.mlp.hook_out", _source_edit_hook(vector))],
        )
        state_records = _store_receiver_states(
            world,
            prefix=f"sender.{planned['intent_id']}.{planned['arm']}",
            raw_tensors=raw_tensors,
        )
        row = {
            **planned,
            "source_activation_delta": (
                dose["nominated_activation_delta"]
                if planned["arm"] == "nominated_c"
                else dose["control_activation_delta"]
            ),
            "endpoints": {
                metric: {
                    "Y0": float(baselines[key]["metrics"][metric]),
                    "YS": float(world["metrics"][metric]),
                    "T_S": float(world["metrics"][metric])
                    - float(baselines[key]["metrics"][metric]),
                }
                for metric in METRICS
            },
            "metrics": world["metrics"],
            "diagnostics": world["diagnostics"],
            "state_records": state_records,
        }
        output["sender_worlds"].append(row)
        output["sender_worlds"].sort(key=lambda value: value["sender_world_id"])
        senders[identity] = world
        _checkpoint(
            output=output,
            output_path=args.output,
            raw_tensors=raw_tensors,
            raw_base_path=args.raw_tensors,
            execution_plan_sha256=execution_plan["payload_sha256"],
        )
        print(f"sender world {index}/{E9C_REQUIRED_SENDER_WORLDS}: {planned['sender_world_id']}", flush=True)

    prompt_state_records = {
        row["prompt_key"]: row["state_records"] for row in output["prompt_records"]
    }
    sender_state_records = {
        (row["intent_id"], row["arm"]): row["state_records"]
        for row in output["sender_worlds"]
    }
    completed_cells = {row["cell_id"] for row in output["route_cells"]}
    numerical = execution_plan["numerical_contract"]
    for index, planned in enumerate(execution_plan["route_cells"], 1):
        cell_id = planned["cell_id"]
        if cell_id in completed_cells:
            continue
        identity = (planned["intent_id"], planned["arm"])
        key = planned["target_prompt_key"]
        position = planned["receiver_token_position"]
        baseline = baselines[key]
        sender = senders[identity]
        baseline_receiver = baseline["receiver_output"][position]
        sender_receiver = sender["receiver_output"][position]
        vector = edit_by_arm[planned["arm"]]
        restoration_mismatch: list[float] = []
        rescue_mismatch: list[float] = []
        _begin_forward(output, args.output, f"route_restore:{cell_id}")
        restored = _run_endpoints(
            model,
            tokens_by_prompt[key],
            probe_direction,
            probe_bias,
            [
                ("blocks.31.mlp.hook_out", _source_edit_hook(vector)),
                (
                    "blocks.32.mlp.hook_out",
                    _replacement_hook(
                        baseline_receiver,
                        expected_before=sender_receiver,
                        position=position,
                        mismatch=restoration_mismatch,
                    ),
                ),
            ],
        )
        _complete_forward(output)
        _begin_forward(output, args.output, f"route_rescue:{cell_id}")
        rescued = _run_endpoints(
            model,
            tokens_by_prompt[key],
            probe_direction,
            probe_bias,
            [
                (
                    "blocks.32.mlp.hook_out",
                    _replacement_hook(
                        sender_receiver,
                        expected_before=baseline_receiver,
                        position=position,
                        mismatch=rescue_mismatch,
                    ),
                )
            ],
        )
        endpoints = route_effects(
            {
                "Y0": baseline["metrics"],
                "YS": sender["metrics"],
                "YS_R0": restored["metrics"],
                "YR_S": rescued["metrics"],
            },
            invariant_atol=float(numerical["invariant_absolute_tolerance"]),
        )
        baseline_record = prompt_state_records[key][position]
        sender_record = sender_state_records[identity][position]
        receiver_state = _receiver_state_summary(
            baseline_receiver, sender_receiver, baseline_record, sender_record
        )
        penultimate_passed = True
        if position == PENULTIMATE_POSITION:
            penultimate_passed = (
                receiver_state["state_delta_max_abs"]
                <= float(numerical["penultimate_receiver_state_max_abs_change"])
                and all(
                    abs(record["D_R"])
                    <= float(numerical["penultimate_endpoint_abs_tolerance"])
                    and abs(record["Q_R"])
                    <= float(numerical["penultimate_endpoint_abs_tolerance"])
                    for record in endpoints.values()
                )
            )
        row = {
            **planned,
            "worlds": {
                "Y0": {
                    "metrics": baseline["metrics"],
                    "diagnostics": baseline["diagnostics"],
                },
                "YS": {
                    "metrics": sender["metrics"],
                    "diagnostics": sender["diagnostics"],
                },
                "YS_R0": restored,
                "YR_S": rescued,
            },
            "endpoints": endpoints,
            "receiver_audit": {
                "restoration_hook_fire_count": len(restoration_mismatch),
                "rescue_hook_fire_count": len(rescue_mismatch),
                "restoration_pre_replacement_max_abs_mismatch": max(
                    restoration_mismatch, default=0.0
                ),
                "rescue_pre_replacement_max_abs_mismatch": max(
                    rescue_mismatch, default=0.0
                ),
                "mismatch_atol": float(
                    numerical["pre_replacement_max_abs_mismatch"]
                ),
            },
            "receiver_state": receiver_state,
            "penultimate_thresholds_passed": penultimate_passed,
        }
        output["route_cells"].append(row)
        output["route_cells"].sort(key=lambda value: value["cell_id"])
        _checkpoint(
            output=output,
            output_path=args.output,
            raw_tensors=raw_tensors,
            raw_base_path=args.raw_tensors,
            execution_plan_sha256=execution_plan["payload_sha256"],
        )
        print(f"route {index}/{E9C_REQUIRED_ROUTE_CELLS}: {cell_id}", flush=True)

    if output["resume_state"]["inflight_operation_id"] is not None:
        raise ValueError("E9c final result retained an in-flight forward")
    _refresh_accounting(output)
    if output["forward_accounting"]["completed_scientific"] != E9C_PLANNED_SCIENTIFIC_FORWARDS:
        raise ValueError("E9c scientific forward closure differs")
    if output["forward_accounting"]["audit_ceiling_exceeded"]:
        raise ValueError("E9c cumulative audit ceiling was exceeded")
    candidate = copy.deepcopy(output)
    candidate["status"] = E9C_RESULT_STATUS
    candidate["completed_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    candidate["payload_sha256"] = canonical_sha256(candidate)
    _validate_complete_result(candidate, execution_plan)
    _atomic_json(args.output, candidate)
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": candidate["status"],
                "forward_accounting": candidate["forward_accounting"],
                "payload_sha256": candidate["payload_sha256"],
                "private_raw_tensor_sha256": candidate[
                    "private_raw_tensor_bundle"
                ]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
