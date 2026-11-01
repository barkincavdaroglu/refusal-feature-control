#!/usr/bin/env python3
"""Run or cumulatively resume E8b's frozen C-to-D route replication."""

from __future__ import annotations

import argparse
import copy
import csv
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import run_s9_calibration as base
from build_s9_e8b_transcoder_bundle import (
    EXPECTED_BUNDLE_KEYS,
    REQUIRED_FEATURE_ROWS,
    save_deterministic_safetensors,
    validate_transcoder_bundle_manifest,
)
from s9_contract import (
    METRICS,
    REFUSAL_TOKEN_IDS,
    canonical_sha256,
    sha256_file,
    sha256_text,
    validate_stamped_payload,
)
from s9_e8_contract import validate_e8a_manifest, validate_e8a_plan
from s9_e8b_contract import (
    E8B_INPUT_NAMES,
    E8B_RESULT_SEMANTICS_VERSION,
    E8B_RESULT_STATUS,
    E8B_RUNNING_STATUS,
    E8B_SCHEMA_VERSION,
    validate_e8b_execution_plan,
    validate_e8b_result,
)


FINAL_POSITION = "final_prompt_position"
PENULTIMATE_POSITION = "penultimate_prompt_position"
POSITIONS = (FINAL_POSITION, PENULTIMATE_POSITION)
EXPECTED_PROMPT_RECORDS = 30
EXPECTED_DOSE_AUDITS = 15
EXPECTED_SENDER_WORLDS = 30
EXPECTED_ROUTE_CELLS = 60
EXPECTED_FORWARD_TOTAL = 180
AUDIT_FORWARD_CEILING = 198
RUNTIME_PROBE_STATUS = "runtime_and_uploaded_bytes_valid_before_e8b_model_load"
EXCLUDED_STAGE_EXECUTED_KEY = "e" + "7d_executed"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_identity_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact bytes without exposing its values."""

    materialized = tensor.detach().contiguous().cpu()
    identity = json.dumps(
        {
            "dtype": str(materialized.dtype).removeprefix("torch."),
            "shape": list(materialized.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = materialized.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(identity + b"\0" + raw).hexdigest()


def _position_index(position: str) -> int:
    if position == FINAL_POSITION:
        return -1
    if position == PENULTIMATE_POSITION:
        return -2
    raise ValueError(f"Unknown E8b token position: {position!r}")


def _sequence_view(activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim == 3:
        if activations.shape[0] != 1:
            raise ValueError("E8b position helpers require batch size one")
        return activations[0]
    if activations.ndim == 2:
        return activations
    raise ValueError(
        f"E8b position helpers require rank-two or rank-three state, got {activations.shape}"
    )


def position_vector(activations: torch.Tensor, position: str) -> torch.Tensor:
    """Return one declared prompt-position vector without changing its dtype."""

    sequence = _sequence_view(activations)
    index = _position_index(position)
    if position == PENULTIMATE_POSITION and sequence.shape[0] < 2:
        raise ValueError("E8b penultimate prompt position is unavailable")
    if sequence.shape[0] < 1:
        raise ValueError("E8b final prompt position is unavailable")
    return sequence[index]


def replace_position_vector(
    activations: torch.Tensor, replacement: torch.Tensor, position: str
) -> torch.Tensor:
    """Clone a state and replace exactly one declared prompt position."""

    _ = position_vector(activations, position)
    output = activations.clone()
    index = _position_index(position)
    value = replacement.to(device=output.device, dtype=output.dtype)
    if output.ndim == 3:
        output[0, index] = value
    else:
        output[index] = value
    return output


def add_source_edit_final(
    activations: torch.Tensor, edit_vector: torch.Tensor
) -> torch.Tensor:
    """Apply an E8b source edit to the final prompt position only."""

    current = position_vector(activations, FINAL_POSITION)
    replacement = current + edit_vector.to(device=current.device, dtype=current.dtype)
    return replace_position_vector(activations, replacement, FINAL_POSITION)


def _capture_positions(storage: dict[str, torch.Tensor], prefix: str, counts: list[int]):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        counts.append(1)
        for position in POSITIONS:
            storage[f"{prefix}:{position}"] = (
                position_vector(activations, position).detach().cpu().contiguous()
            )
        return activations

    return hook


def _capture_final(storage: dict[str, torch.Tensor], name: str, counts: list[int]):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        counts.append(1)
        storage[name] = position_vector(
            activations, FINAL_POSITION
        ).detach().cpu().contiguous()
        return activations

    return hook


def _source_edit_hook(edit_vector: torch.Tensor):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        return add_source_edit_final(activations, edit_vector)

    return hook


def _replace_hook(
    replacement: torch.Tensor,
    *,
    expected_before: torch.Tensor,
    position: str,
    mismatch_atol: float,
    mismatch: list[float],
):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        current = position_vector(activations, position)
        expected = expected_before.to(device=current.device, dtype=current.dtype)
        value = float((current.float() - expected.float()).abs().max().cpu())
        mismatch.append(value)
        # A mismatch is a recorded technical-evaluability observation.  It is
        # not grounds for discarding an otherwise completed scientific cell.
        _ = mismatch_atol
        return replace_position_vector(activations, replacement, position)

    return hook


def _diagnostics(logits: torch.Tensor) -> dict[str, Any]:
    values = position_vector(logits, FINAL_POSITION).detach().float()
    top_values, top_ids = torch.topk(values, k=2)
    mask = torch.ones(values.shape[-1], dtype=torch.bool, device=values.device)
    mask[list(REFUSAL_TOKEN_IDS)] = False
    refusal_top = values[list(REFUSAL_TOKEN_IDS)].max()
    nonrefusal_top = values[mask].max()
    return {
        "top_token_id": int(top_ids[0].cpu()),
        "runner_up_token_id": int(top_ids[1].cpu()),
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
    states: dict[str, torch.Tensor] = {}
    final: dict[str, torch.Tensor] = {}
    source_counts: list[int] = []
    receiver_input_counts: list[int] = []
    receiver_output_counts: list[int] = []
    final_counts: list[int] = []
    hooks = [
        *intervention_hooks,
        (
            "blocks.31.mlp.hook_in",
            _capture_final(states, "source_input_final", source_counts),
        ),
        (
            "blocks.32.mlp.hook_in",
            _capture_positions(states, "receiver_input", receiver_input_counts),
        ),
        (
            "blocks.32.mlp.hook_out",
            _capture_positions(states, "receiver_output", receiver_output_counts),
        ),
        ("unembed.hook_pre", _capture_final(final, "final", final_counts)),
    ]
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    if any(
        len(counts) != 1
        for counts in (
            source_counts,
            receiver_input_counts,
            receiver_output_counts,
            final_counts,
        )
    ):
        raise RuntimeError("E8b declared cache hook did not fire exactly once")
    final_normalized = final["final"].to(device=logits.device)
    metrics = base.detached_metric_values(
        base.differentiable_metrics(
            logits, final_normalized, probe_direction, probe_bias
        )
    )
    return {
        "metrics": metrics,
        "diagnostics": _diagnostics(logits),
        "states": {
            "source_input_final": states["source_input_final"],
            "receiver_input": {
                position: states[f"receiver_input:{position}"] for position in POSITIONS
            },
            "receiver_output": {
                position: states[f"receiver_output:{position}"] for position in POSITIONS
            },
        },
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
        fwd_hooks=[
            *hooks,
            ("unembed.hook_pre", _capture_final(final, "final", counts)),
        ],
    )
    if len(counts) != 1:
        raise RuntimeError("E8b endpoint hook did not fire exactly once")
    final_normalized = final["final"].to(device=logits.device)
    return {
        "metrics": base.detached_metric_values(
            base.differentiable_metrics(
                logits, final_normalized, probe_direction, probe_bias
            )
        ),
        "diagnostics": _diagnostics(logits),
    }


def compute_natural_activation_delta(
    *,
    encoder: torch.Tensor,
    bias: torch.Tensor | float,
    source_input: torch.Tensor,
    target_input: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate C activations on CUDA with E7c's BF16 operand/output path."""

    if encoder.dtype != torch.bfloat16:
        raise ValueError("E8b C encoder must use the frozen BF16 semantics")
    encoder_bf16 = encoder.detach()
    source_bf16 = source_input.detach().to(
        device=encoder.device, dtype=encoder.dtype
    )
    target_bf16 = target_input.detach().to(
        device=encoder.device, dtype=encoder.dtype
    )
    if (
        encoder_bf16.ndim != 1
        or source_bf16.shape != encoder_bf16.shape
        or target_bf16.shape != encoder_bf16.shape
    ):
        raise ValueError("E8b natural-delta vectors have incompatible shapes")
    bias_bf16 = torch.as_tensor(bias).detach().to(
        device=encoder.device, dtype=encoder.dtype
    ).reshape(())
    source_pre_tensor = torch.dot(encoder_bf16, source_bf16) + bias_bf16
    target_pre_tensor = torch.dot(encoder_bf16, target_bf16) + bias_bf16
    source_activation_tensor = torch.relu(source_pre_tensor)
    target_activation_tensor = torch.relu(target_pre_tensor)
    if source_pre_tensor.dtype != torch.bfloat16 or target_pre_tensor.dtype != torch.bfloat16:
        raise ValueError("E8b C activation output dtype differs from BF16")
    source_pre = float(source_pre_tensor.detach().float().cpu())
    target_pre = float(target_pre_tensor.detach().float().cpu())
    source_activation = float(source_activation_tensor.detach().float().cpu())
    target_activation = float(target_activation_tensor.detach().float().cpu())
    return {
        "source_preactivation": source_pre,
        "target_preactivation": target_pre,
        "source_activation": source_activation,
        "target_activation": target_activation,
        "activation_delta": source_activation - target_activation,
        "numeric_dtype": "bfloat16",
        "numeric_device": encoder.device.type,
    }


def _relative_norm_error(nominated: float, control: float) -> float:
    if nominated == 0.0 and control == 0.0:
        return 0.0
    if nominated == 0.0:
        raise ValueError("E8b zero nominated norm has a nonzero control norm")
    return abs(control - nominated) / nominated


def compute_dose_audit(
    *,
    nominated_activation_delta: float,
    nominated_decoder: torch.Tensor,
    control_decoder: torch.Tensor,
) -> dict[str, Any]:
    """Construct nominal float32 equal-L2 C/M edits and audit BF16 realization."""

    delta_c = float(nominated_activation_delta)
    if not math.isfinite(delta_c):
        raise ValueError("E8b nominated activation delta must be finite")
    decoder_c = nominated_decoder.detach().float().cpu()
    decoder_m = control_decoder.detach().float().cpu()
    if decoder_c.ndim != 1 or decoder_m.ndim != 1 or decoder_c.shape != decoder_m.shape:
        raise ValueError("E8b C/M decoder vectors have incompatible shapes")
    nominal_c = decoder_c * delta_c
    nominal_c_l2 = float(nominal_c.norm())
    control_decoder_l2 = float(decoder_m.norm())
    if delta_c == 0.0:
        delta_m = 0.0
    else:
        if control_decoder_l2 == 0.0:
            raise ValueError("E8b control decoder norm is zero for a nonzero C dose")
        delta_m = math.copysign(nominal_c_l2 / control_decoder_l2, delta_c)
    nominal_m = decoder_m * delta_m
    nominal_m_l2 = float(nominal_m.norm())
    nominal_error = _relative_norm_error(nominal_c_l2, nominal_m_l2)

    realized_c = nominal_c.to(torch.bfloat16).float()
    realized_m = nominal_m.to(torch.bfloat16).float()
    realized_c_l2 = float(realized_c.norm())
    realized_m_l2 = float(realized_m.norm())
    return {
        "nominated_activation_delta": delta_c,
        "control_activation_delta": delta_m,
        "nominated_decoder_l2": float(decoder_c.norm()),
        "control_decoder_l2": control_decoder_l2,
        "nominal_nominated_edit_l2": nominal_c_l2,
        "nominal_control_edit_l2": nominal_m_l2,
        "nominal_equal_l2_relative_error": nominal_error,
        "nominal_numeric_dtype": "float32",
        "realized_bfloat16_nominated_edit_l2": realized_c_l2,
        "realized_bfloat16_control_edit_l2": realized_m_l2,
        "realized_bfloat16_equal_l2_relative_error": _relative_norm_error(
            realized_c_l2, realized_m_l2
        ),
        "realized_numeric_dtype": "bfloat16",
        "zero_dose": delta_c == 0.0 and nominal_c_l2 == 0.0,
    }


def one_way_gate_crossing(
    baseline_preactivation: float, sender_preactivation: float
) -> bool:
    return float(baseline_preactivation) <= 0.0 < float(sender_preactivation)


def _crossing_class(baseline: float, sender: float) -> str:
    if baseline <= 0.0 < sender:
        return "inactive_to_active"
    if sender <= 0.0 < baseline:
        return "active_to_inactive"
    if baseline > 0.0 and sender > 0.0:
        return "active_stable"
    return "inactive_stable"


def _feature_value(
    *, encoder: torch.Tensor, bias: torch.Tensor | float, state: torch.Tensor
) -> tuple[float, float]:
    if encoder.dtype != torch.bfloat16:
        raise ValueError("E8b D encoder must use the frozen BF16 semantics")
    vector = state.detach().to(device=encoder.device, dtype=encoder.dtype)
    bias_value = torch.as_tensor(bias).detach().to(
        device=encoder.device, dtype=encoder.dtype
    ).reshape(())
    pre_tensor = torch.dot(encoder, vector) + bias_value
    activation_tensor = torch.relu(pre_tensor)
    if pre_tensor.dtype != torch.bfloat16:
        raise ValueError("E8b D preactivation output dtype differs from BF16")
    return (
        float(pre_tensor.detach().float().cpu()),
        float(activation_tensor.detach().float().cpu()),
    )


def _gate_diagnostic(
    feature_record: Mapping[str, Any], baseline: torch.Tensor, sender: torch.Tensor
) -> dict[str, Any]:
    baseline_pre, baseline_activation = _feature_value(
        encoder=feature_record["encoder"],
        bias=feature_record["bias"],
        state=baseline,
    )
    sender_pre, sender_activation = _feature_value(
        encoder=feature_record["encoder"],
        bias=feature_record["bias"],
        state=sender,
    )
    return {
        "feature": "L32:F50401",
        "numeric_dtype": "bfloat16",
        "numeric_device": feature_record["encoder"].device.type,
        "baseline_preactivation": baseline_pre,
        "sender_preactivation": sender_pre,
        "preactivation_delta": sender_pre - baseline_pre,
        "baseline_activation": baseline_activation,
        "sender_activation": sender_activation,
    }


def classify_gate_diagnostic_after_cell_closure(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one frozen D measurement only after all 60 cells exist."""

    if "crossing_class" in diagnostic or "gate_crossing" in diagnostic:
        raise ValueError("E8b D gate was classified before route-cell closure")
    baseline = float(diagnostic["baseline_preactivation"])
    sender = float(diagnostic["sender_preactivation"])
    return {
        **diagnostic,
        "crossing_class": _crossing_class(baseline, sender),
        "gate_crossing": one_way_gate_crossing(baseline, sender),
    }


def load_compact_feature_table(
    bundle_path: Path, *, expected_file_sha256: str
) -> dict[str, dict[str, Any]]:
    """Load and map exactly the authenticated C/M/D bundle rows."""

    if not bundle_path.is_file():
        raise FileNotFoundError(f"E8b compact transcoder bundle is missing: {bundle_path}")
    if sha256_file(bundle_path) != expected_file_sha256:
        raise ValueError("E8b compact transcoder bundle hash differs")
    layer_tensors: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(EXPECTED_BUNDLE_KEYS):
            raise ValueError("E8b compact transcoder bundle key closure differs")
        for layer in (31, 32):
            layer_tensors[layer] = {
                name: handle.get_tensor(f"layer_{layer}.{name}")
                for name in (*("W_dec", "W_enc", "b_enc"), "feature_indices")
            }

    table: dict[str, dict[str, Any]] = {}
    for layer in (31, 32):
        values = layer_tensors[layer]
        expected_indices = [
            index
            for _, (feature_layer, index, _) in REQUIRED_FEATURE_ROWS.items()
            if feature_layer == layer
        ]
        indices = [int(value) for value in values["feature_indices"].tolist()]
        if indices != expected_indices:
            raise ValueError(f"E8b compact transcoder indices differ at layer {layer}")
        if values["W_dec"].ndim != 2 or values["W_enc"].shape != values["W_dec"].shape:
            raise ValueError(f"E8b compact transcoder C/M/D vector shapes differ at layer {layer}")
        if values["b_enc"].ndim != 1 or values["b_enc"].shape[0] != len(indices):
            raise ValueError(f"E8b compact transcoder bias shape differs at layer {layer}")
        if values["W_dec"].shape[0] != len(indices):
            raise ValueError(f"E8b compact transcoder row count differs at layer {layer}")
        for row_index, feature_index in enumerate(indices):
            feature = f"L{layer}:F{feature_index}"
            if feature not in REQUIRED_FEATURE_ROWS:
                raise ValueError(f"Unexpected E8b compact transcoder feature: {feature}")
            table[feature] = {
                "feature": feature,
                "layer": layer,
                "feature_index": feature_index,
                "role": REQUIRED_FEATURE_ROWS[feature][2],
                "encoder": values["W_enc"][row_index].contiguous(),
                "decoder": values["W_dec"][row_index].contiguous(),
                "bias": values["b_enc"][row_index].reshape(()).contiguous(),
            }
    if set(table) != set(REQUIRED_FEATURE_ROWS):
        raise ValueError("E8b compact transcoder feature closure differs")
    return table


def route_effects(
    worlds: Mapping[str, Mapping[str, float]], *, invariant_atol: float = 1e-7
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        y0 = float(worlds["Y0"][metric])
        ys = float(worlds["YS"][metric])
        ysr0 = float(worlds["YS_R0"][metric])
        yrs = float(worlds["YR_S"][metric])
        values = (y0, ys, ysr0, yrs)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"E8b non-finite endpoint world for {metric}")
        total = ys - y0
        dependence = ys - ysr0
        bypass = ysr0 - y0
        rescue = yrs - y0
        invariant = total - dependence - bypass
        # Preserve a finite observed threshold miss for preregistered analysis.
        _ = invariant_atol
        output[metric] = {
            "Y0": y0,
            "YS": ys,
            "YS_R0": ysr0,
            "YR_S": yrs,
            "T_S": total,
            "D_R": dependence,
            "B_R": bypass,
            "Q_R": rescue,
            "invariant_error": abs(invariant),
        }
    return output


def completed_forward_count(
    payload: Mapping[str, Any], *, verify_recorded: bool = False
) -> int:
    prompt_count = len(payload.get("prompt_records", []))
    sender_count = len(payload.get("sender_worlds", []))
    route_count = len(payload.get("route_cells", []))
    if prompt_count > EXPECTED_PROMPT_RECORDS or sender_count > EXPECTED_SENDER_WORLDS or route_count > EXPECTED_ROUTE_CELLS:
        raise ValueError("E8b scientific result-table counts exceed their frozen sizes")
    derived = prompt_count + sender_count + 2 * route_count
    if verify_recorded:
        accounting = payload.get("forward_accounting", {})
        recorded = accounting.get(
            "completed_scientific", accounting.get("completed")
        )
        if recorded != derived:
            raise ValueError("E8b cumulative completed-forward accounting differs")
    return derived


def _refresh_accounting(payload: dict[str, Any]) -> None:
    accounting = payload["forward_accounting"]
    completed = completed_forward_count(payload)
    cumulative = int(accounting["cumulative_total_across_resumes"])
    if cumulative < completed:
        raise ValueError("E8b cumulative counter is below completed scientific forwards")
    accounting.update(
        {
            "planned_scientific": EXPECTED_FORWARD_TOTAL,
            "completed_scientific": completed,
            "audit_forwards": cumulative - completed,
            "audit_ceiling": AUDIT_FORWARD_CEILING,
            "audit_ceiling_exceeded": cumulative > AUDIT_FORWARD_CEILING,
            "counter_reset_on_resume": False,
        }
    )


def _begin_forward(payload: dict[str, Any], output_path: Path, operation_id: str) -> None:
    accounting = payload["forward_accounting"]
    accounting["cumulative_total_across_resumes"] = int(
        accounting["cumulative_total_across_resumes"]
    ) + 1
    payload["resume_state"]["inflight_operation_id"] = operation_id
    _refresh_accounting(payload)
    _atomic(output_path, payload)
    if accounting["audit_ceiling_exceeded"]:
        raise RuntimeError("E8b cumulative forward audit ceiling exceeded")


def _complete_operation(payload: dict[str, Any]) -> None:
    payload["resume_state"]["inflight_operation_id"] = None
    _refresh_accounting(payload)


def _state_record(tensor: torch.Tensor, raw_key: str) -> dict[str, Any]:
    return {
        "raw_tensor_key": raw_key,
        "sha256": _tensor_sha256(tensor),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "l2": float(tensor.float().norm()),
    }


def _store_world_states(
    world: Mapping[str, Any], *, prefix: str, raw_tensors: dict[str, torch.Tensor]
) -> dict[str, Any]:
    records: dict[str, Any] = {"receiver_input": {}, "receiver_output": {}}
    source_key = f"{prefix}.L31.module_input.{FINAL_POSITION}"
    source = world["states"]["source_input_final"].contiguous().cpu()
    if source_key in raw_tensors:
        raise ValueError(f"Duplicate E8b raw tensor key: {source_key}")
    raw_tensors[source_key] = source
    records["source_input_final"] = _state_record(source, source_key)
    for boundary in ("receiver_input", "receiver_output"):
        for position in POSITIONS:
            tensor = world["states"][boundary][position].contiguous().cpu()
            key = f"{prefix}.L32.{boundary}.{position}"
            if key in raw_tensors:
                raise ValueError(f"Duplicate E8b raw tensor key: {key}")
            raw_tensors[key] = tensor
            records[boundary][position] = _state_record(tensor, key)
    return records


def _restore_world_states(
    records: Mapping[str, Any], raw_tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    def restore(record: Mapping[str, Any]) -> torch.Tensor:
        key = record["raw_tensor_key"]
        if key not in raw_tensors:
            raise ValueError(f"E8b raw tensor is missing: {key}")
        tensor = raw_tensors[key]
        if _tensor_sha256(tensor) != record["sha256"]:
            raise ValueError(f"E8b raw tensor hash differs: {key}")
        if str(tensor.dtype).removeprefix("torch.") != record["dtype"] or list(tensor.shape) != record["shape"]:
            raise ValueError(f"E8b raw tensor metadata differs: {key}")
        return tensor

    return {
        "source_input_final": restore(records["source_input_final"]),
        "receiver_input": {
            position: restore(records["receiver_input"][position])
            for position in POSITIONS
        },
        "receiver_output": {
            position: restore(records["receiver_output"][position])
            for position in POSITIONS
        },
    }


def _save_raw(
    path: Path, tensors: Mapping[str, torch.Tensor], execution_plan_hash: str
) -> dict[str, Any] | None:
    if not tensors:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_deterministic_safetensors(
        dict(tensors),
        temporary,
        metadata={"execution_plan_payload_sha256": execution_plan_hash},
    )
    digest = sha256_file(temporary)
    checkpoint = path.with_name(f"{path.stem}.{digest}{path.suffix}")
    if checkpoint.exists():
        if sha256_file(checkpoint) != digest:
            raise ValueError("E8b content-addressed raw checkpoint hash differs")
        temporary.unlink()
    else:
        temporary.replace(checkpoint)
    return {
        "path": checkpoint.name,
        "sha256": digest,
        "tensor_count": len(tensors),
        "public": False,
        "contains_prompt_or_token_text": False,
    }


def raw_checkpoint_path(base_path: Path, record: Mapping[str, Any]) -> Path:
    """Resolve only the content-addressed raw file named by the JSON pointer."""

    filename = record.get("path")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError("E8b raw checkpoint filename is malformed")
    expected_hash = record.get("sha256")
    expected_name = f"{base_path.stem}.{expected_hash}{base_path.suffix}"
    if filename != expected_name:
        raise ValueError("E8b raw checkpoint is not content-addressed")
    return base_path.parent / filename


def _checkpoint(
    *,
    output: dict[str, Any],
    output_path: Path,
    raw_tensors: Mapping[str, torch.Tensor],
    raw_path: Path,
    execution_plan_hash: str,
) -> None:
    output["private_raw_tensor_bundle"] = _save_raw(
        raw_path, raw_tensors, execution_plan_hash
    )
    _complete_operation(output)
    _atomic(output_path, output)


def _receiver_state(
    baseline: torch.Tensor,
    sender: torch.Tensor,
    baseline_record: Mapping[str, Any],
    sender_record: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline_record["dtype"] != sender_record["dtype"] or baseline_record["shape"] != sender_record["shape"]:
        raise ValueError("E8b receiver state metadata differs across worlds")
    delta = sender.float() - baseline.float()
    return {
        "baseline_tensor_sha256": baseline_record["sha256"],
        "sender_tensor_sha256": sender_record["sha256"],
        "baseline_raw_tensor_key": baseline_record["raw_tensor_key"],
        "sender_raw_tensor_key": sender_record["raw_tensor_key"],
        "dtype": baseline_record["dtype"],
        "shape": baseline_record["shape"],
        "baseline_l2": baseline_record["l2"],
        "sender_l2": sender_record["l2"],
        "state_delta_l2": float(delta.norm()),
        "state_delta_max_abs": float(delta.abs().max()),
        "receiver_moved": bool(float(delta.norm()) > 0.0),
    }


def assert_public_safe(value: Any) -> None:
    forbidden = {
        "prompt",
        "prompt_text",
        "completion",
        "completion_text",
        "token_text",
        "token_ids",
        "chat_text",
        "input_string",
        "generated_text",
        "response_text",
        "receiver_tensor",
        "raw_tensor",
        "core",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                raise ValueError(f"Public E8b payload contains forbidden key: {key}")
            assert_public_safe(item)
    elif isinstance(value, list):
        for item in value:
            assert_public_safe(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Public E8b payload contains a non-finite value")


def _load_selected_prompts(
    path: Path, e8a_plan: Mapping[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    all_rows = {row.get("prompt_key"): row for row in rows}
    if len(all_rows) != len(rows):
        raise ValueError("E8b private prompt source contains duplicate keys")
    expected: dict[str, dict[str, Any]] = {}
    for pair in e8a_plan["intent_pairs"]:
        for side in ("source_prompt", "target_prompt"):
            record = pair[side]
            expected[record["prompt_key"]] = {
                "prompt_key": record["prompt_key"],
                "prompt_sha256": record["prompt_sha256"],
                "intent_id": pair["intent_id"],
                "risk_class": pair["risk_class"],
                "category": pair["category"],
                "partition": pair["partition"],
                "frame_id": record["frame_id"],
            }
    if len(expected) != EXPECTED_PROMPT_RECORDS:
        raise ValueError("E8b selected prompt closure differs")
    selected: dict[str, dict[str, str]] = {}
    for key, record in expected.items():
        if key not in all_rows:
            raise ValueError(f"E8b selected private prompt is missing: {key}")
        row = all_rows[key]
        if sha256_text(row.get("prompt", "")) != record["prompt_sha256"]:
            raise ValueError(f"E8b selected private prompt hash differs: {key}")
        for field in (
            "prompt_key",
            "prompt_sha256",
            "intent_id",
            "risk_class",
            "category",
            "partition",
            "frame_id",
        ):
            if row.get(field) != record[field]:
                raise ValueError(f"E8b selected prompt metadata differs: {key}.{field}")
        if row["partition"] != "validation_expansion":
            raise ValueError("E8b may load only validation-expansion prompts")
        selected[key] = row
    return selected, expected


def _semantic_paths(args: argparse.Namespace) -> dict[str, Path]:
    here = Path(__file__).resolve()
    return {
        "e8a_manifest": args.e8a_manifest,
        "e8a_plan": args.e8a_plan,
        "semantic_erratum": args.semantic_erratum,
        "private_prompts": args.prompts,
        "probe_direction": args.probe_direction,
        "transcoder_row_bundle": args.transcoder_row_bundle,
        "transcoder_row_manifest": args.transcoder_row_manifest,
        "transcoder_bundle_builder": here.with_name("build_s9_e8b_transcoder_bundle.py"),
        "range_helper": here.with_name("extract_s9_e8b_transcoder_rows.py"),
        "runner": here,
        "colab_runner": here.with_name("run_s9_e8b_colab.py"),
        "runtime_probe": here.with_name("probe_s9_e8b_runtime_colab.py"),
        "analyzer": here.with_name("analyze_s9_e8b.py"),
        "runtime_installer": here.with_name("install_s9_colab_runtime.py"),
        "base_runner": Path(base.__file__).resolve(),
        "base_contract": here.with_name("s9_contract.py"),
        "e8_contract": here.with_name("s9_e8_contract.py"),
        "e8b_contract": here.with_name("s9_e8b_contract.py"),
        "execution_plan_builder": here.with_name("build_s9_e8b_execution_plan.py"),
    }


def _load_runtime_proof(
    path: Path,
    execution_plan: Mapping[str, Any],
    actual_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    validate_stamped_payload(proof)
    if proof.get("status") != RUNTIME_PROBE_STATUS:
        raise ValueError("E8b runtime probe status differs")
    if proof.get("model_forward_executed_by_this_probe") is not False:
        raise ValueError("E8b runtime probe executed a model forward")
    if proof.get("execution_plan_payload_sha256") != execution_plan.get(
        "payload_sha256"
    ):
        raise ValueError("E8b runtime probe references another execution plan")
    expected_hashes = {
        name: record["sha256"] for name, record in execution_plan["inputs"].items()
    }
    if proof.get("uploaded_input_hashes") != expected_hashes:
        raise ValueError("E8b runtime probe uploaded-input closure differs")
    if proof.get("runtime") != actual_runtime:
        raise ValueError("E8b runtime probe fingerprint differs from runner runtime")
    return {
        "path": path.name,
        "file_sha256": sha256_file(path),
        "payload_sha256": proof["payload_sha256"],
        "status": proof["status"],
        "public": True,
        "model_forward_executed_by_probe": False,
    }


def _validate_partial(
    payload: Mapping[str, Any], execution_plan: Mapping[str, Any]
) -> None:
    validate_stamped_payload(payload)
    if payload.get("status") != E8B_RUNNING_STATUS:
        raise ValueError("E8b resumable result status differs")
    if payload.get("plan_payload_sha256") != execution_plan.get("payload_sha256"):
        raise ValueError("E8b resumable result references another execution plan")
    for key in ("prompt_records", "dose_audits", "sender_worlds", "route_cells"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"E8b resumable {key} table is missing")
    for key, id_field in (
        ("prompt_records", "prompt_key"),
        ("dose_audits", "dose_audit_id"),
        ("sender_worlds", "sender_world_id"),
        ("route_cells", "cell_id"),
    ):
        ids = [row.get(id_field) for row in payload[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"E8b resumable {key} contains duplicate identities")
    completed_forward_count(payload, verify_recorded=True)
    accounting = payload.get("forward_accounting", {})
    if accounting.get("counter_reset_on_resume") is not False:
        raise ValueError("E8b cumulative counter was reset on resume")
    cumulative = accounting.get("cumulative_total_across_resumes")
    if isinstance(cumulative, bool) or not isinstance(cumulative, int) or cumulative < 0:
        raise ValueError("E8b cumulative resume counter is malformed")
    resume_state = payload.get("resume_state")
    if not isinstance(resume_state, Mapping) or set(resume_state) != {
        "inflight_operation_id"
    }:
        raise ValueError("E8b resumable operation state is malformed")
    inflight = resume_state["inflight_operation_id"]
    if inflight is not None and (not isinstance(inflight, str) or not inflight):
        raise ValueError("E8b resumable in-flight operation identity is malformed")
    assert_public_safe(payload)


def _world_from_record(
    row: Mapping[str, Any], raw_tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    return {
        "metrics": row["metrics"],
        "diagnostics": row["diagnostics"],
        "states": _restore_world_states(row["state_records"], raw_tensors),
    }


def _runtime_limits(plan: Mapping[str, Any]) -> tuple[list[str], float]:
    runtime = plan["runtime_contract"]
    preferred = list(runtime.get("preferred_accelerators", ("H100", "A100")))
    minimum = float(runtime.get("minimum_cuda_memory_gib", 39.0))
    return preferred, minimum


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_hf_snapshot(
    snapshot_path: Path, *, repo_id: str, expected_revision: str
) -> dict[str, Any]:
    """Authenticate every cached file used from one exact Hub snapshot."""

    if snapshot_path.name != expected_revision or not snapshot_path.is_dir():
        raise ValueError("E8b model snapshot did not resolve to the frozen revision")
    files: list[dict[str, Any]] = []
    for logical_path in sorted(snapshot_path.rglob("*")):
        if logical_path.is_dir():
            continue
        if not logical_path.is_symlink():
            raise ValueError("E8b model snapshot contains an unauthenticated cache file")
        blob_path = logical_path.resolve(strict=True)
        cache_object = blob_path.name
        if _SHA256_RE.fullmatch(cache_object):
            algorithm = "sha256"
            observed_cache_object = sha256_file(blob_path)
            content_sha256 = observed_cache_object
        elif _GIT_OBJECT_RE.fullmatch(cache_object):
            algorithm = "git_blob_sha1"
            observed_cache_object = _git_blob_sha1(blob_path)
            content_sha256 = sha256_file(blob_path)
        else:
            raise ValueError("E8b model cache object name is not content-addressed")
        if observed_cache_object != cache_object:
            raise ValueError("E8b model cache object content hash differs")
        files.append(
            {
                "path": logical_path.relative_to(snapshot_path).as_posix(),
                "size_bytes": blob_path.stat().st_size,
                "content_sha256": content_sha256,
                "cache_object_algorithm": algorithm,
                "cache_object": cache_object,
            }
        )
    names = {record["path"] for record in files}
    if "config.json" not in names or "tokenizer.json" not in names:
        raise ValueError("E8b model snapshot lacks its loaded config/tokenizer closure")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("E8b model snapshot lacks authenticated model weights")
    return {
        "repo_id": repo_id,
        "resolved_revision": expected_revision,
        "verified_before_first_forward": True,
        "all_cache_objects_content_authenticated": True,
        "file_count": len(files),
        "total_size_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
        "file_closure_sha256": canonical_sha256({"files": files}),
    }


def apply_runtime_determinism(
    runtime_contract: Mapping[str, Any], *, torch_module: Any = torch
) -> dict[str, Any]:
    """Apply and return the exact frozen runtime determinism record."""

    seed = int(runtime_contract["random_seed"])
    determinism = dict(runtime_contract["determinism"])
    expected_workspace = determinism["cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_workspace:
        raise RuntimeError(
            "E8b CUBLAS_WORKSPACE_CONFIG was not frozen before torch import"
        )
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.benchmark = bool(
        determinism["cudnn_benchmark"]
    )
    torch_module.backends.cuda.matmul.allow_tf32 = bool(
        determinism["allow_tf32"]
    )
    torch_module.backends.cudnn.allow_tf32 = bool(determinism["allow_tf32"])
    torch_module.use_deterministic_algorithms(
        bool(determinism["deterministic_algorithms"])
    )
    torch_module.set_grad_enabled(bool(determinism["gradient_tracking"]))
    return determinism


def _nvidia_driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not versions or len(set(versions)) != 1:
        raise ValueError("E8b NVIDIA driver fingerprint is unavailable or heterogeneous")
    return versions[0]


def build_runtime_fingerprint(
    runtime_contract: Mapping[str, Any], actual_packages: Mapping[str, str]
) -> dict[str, Any]:
    """Return the exact cross-resume software/accelerator fingerprint."""

    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_gib": (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        ),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "nvidia_driver_version": _nvidia_driver_version(),
        "packages": dict(actual_packages),
        "dtype": runtime_contract["dtype"],
        "random_seed": runtime_contract["random_seed"],
        "determinism": dict(runtime_contract["determinism"]),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    fields = runtime_contract["runtime_fingerprint_fields"]
    if set(fields) != set(runtime):
        raise ValueError("E8b runtime fingerprint field closure differs")
    runtime["runtime_fingerprint_sha256"] = canonical_sha256(
        {"runtime_fingerprint": {key: runtime[key] for key in fields}}
    )
    return runtime


def result_identity_fields(execution_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical immutable identity/authorization fields for a result."""

    return {
        "schema_version": E8B_SCHEMA_VERSION,
        "analysis_semantics_version": E8B_RESULT_SEMANTICS_VERSION,
        "plan_payload_sha256": execution_plan["payload_sha256"],
        "frozen_e8a_snapshot_sha256": execution_plan["frozen_e8a_snapshot"][
            "scientific_snapshot_sha256"
        ],
        "model": dict(execution_plan["model"]),
        "transcoder": dict(execution_plan["transcoder"]),
        "semantic_resolution": dict(execution_plan["semantic_resolution"]),
        "generation_executed": False,
        "backward_pass_executed": False,
        EXCLUDED_STAGE_EXECUTED_KEY: False,
        "confirmation_boundary": execution_plan["confirmation_boundary"],
        "privacy": execution_plan["privacy"],
    }


def finalize_result_contract_fields(
    payload: dict[str, Any], execution_plan: Mapping[str, Any]
) -> None:
    """Close transient resume state and apply the complete result identity."""

    payload.update(result_identity_fields(execution_plan))
    payload.pop("resume_state", None)
    payload["forward_accounting"].pop("inflight_operation_id", None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--e8a-manifest", type=Path, required=True)
    parser.add_argument("--e8a-plan", type=Path, required=True)
    parser.add_argument("--semantic-erratum", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--probe-direction", type=Path, required=True)
    parser.add_argument("--transcoder-row-bundle", type=Path, required=True)
    parser.add_argument("--transcoder-row-manifest", type=Path, required=True)
    parser.add_argument("--runtime-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-tensors", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E8b")
    execution_plan = json.loads(args.execution_plan.read_text(encoding="utf-8"))
    e8a_manifest = json.loads(args.e8a_manifest.read_text(encoding="utf-8"))
    e8a_plan = json.loads(args.e8a_plan.read_text(encoding="utf-8"))
    validate_e8a_manifest(e8a_manifest)
    validate_e8a_plan(e8a_plan, e8a_manifest)

    validate_e8b_execution_plan(execution_plan, e8a_manifest, e8a_plan)
    paths = _semantic_paths(args)
    if set(paths) != set(E8B_INPUT_NAMES) or set(execution_plan["inputs"]) != set(
        E8B_INPUT_NAMES
    ):
        raise ValueError("E8b execution semantic-input closure differs")
    execution_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"E8b semantic input is missing: {name}")
        digest = sha256_file(path)
        if digest != execution_plan["inputs"][name]["sha256"]:
            raise ValueError(f"E8b semantic input hash differs: {name}")
        execution_hashes[name] = digest

    row_manifest = json.loads(args.transcoder_row_manifest.read_text(encoding="utf-8"))
    validate_transcoder_bundle_manifest(
        row_manifest,
        e8a_manifest=e8a_manifest,
        e8a_plan=e8a_plan,
        e8a_manifest_path=args.e8a_manifest,
        e8a_plan_path=args.e8a_plan,
        bundle_path=args.transcoder_row_bundle,
    )
    if row_manifest["payload_sha256"] != execution_plan.get(
        "transcoder_row_manifest_payload_sha256", row_manifest["payload_sha256"]
    ):
        raise ValueError("E8b execution plan references another transcoder manifest")

    package_versions = execution_plan["runtime_contract"]["package_versions"]
    actual_packages = {
        package: importlib.metadata.version(package) for package in package_versions
    }
    if actual_packages != package_versions:
        raise ValueError("E8b runtime package contract differs")
    device_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    preferred, minimum_memory = _runtime_limits(execution_plan)
    if not any(name in device_name for name in preferred):
        raise ValueError(f"Unsupported E8b accelerator: {device_name}")
    if memory < minimum_memory:
        raise ValueError(f"E8b GPU memory below contract: {memory:.2f} GiB")
    _ = apply_runtime_determinism(execution_plan["runtime_contract"])
    runtime = build_runtime_fingerprint(
        execution_plan["runtime_contract"], actual_packages
    )
    runtime_proof = _load_runtime_proof(
        args.runtime_probe, execution_plan, runtime
    )

    selected_prompts, prompt_metadata = _load_selected_prompts(
        args.prompts, e8a_plan
    )
    feature_table = load_compact_feature_table(
        args.transcoder_row_bundle,
        expected_file_sha256=execution_plan["inputs"]["transcoder_row_bundle"][
            "sha256"
        ],
    )

    from transformer_lens import HookedTransformer

    device = torch.device("cuda")
    print(f"Loading {execution_plan['model']['id']} on {device_name}", flush=True)
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

    model_snapshot_path = Path(
        snapshot_download(
            repo_id=execution_plan["model"]["id"],
            revision=execution_plan["model"]["revision"],
            local_files_only=True,
        )
    )
    model_snapshot_proof = authenticate_hf_snapshot(
        model_snapshot_path,
        repo_id=execution_plan["model"]["id"],
        expected_revision=execution_plan["model"]["revision"],
    )

    for record in feature_table.values():
        for field in ("encoder", "decoder", "bias"):
            tensor = record[field]
            if tensor.dtype != torch.bfloat16:
                raise ValueError("E8b compact transcoder row storage dtype differs")
            record[field] = tensor.to(device=device, dtype=torch.bfloat16)
        if any(record[field].device.type != "cuda" for field in ("encoder", "decoder", "bias")):
            raise ValueError("E8b feature activation tensors are not on CUDA")

    hidden_size = int(model.cfg.d_model)
    if any(record["decoder"].numel() != hidden_size for record in feature_table.values()):
        raise ValueError("E8b compact transcoder width differs from model d_model")
    probe = np.load(args.probe_direction)
    probe_direction = torch.tensor(
        probe["direction_unit"].astype(np.float32),
        device=device,
        dtype=torch.float32,
    )
    probe_bias = float(probe["bias_unit"])

    tokens_by_prompt: dict[str, torch.Tensor] = {}
    runtime_prompt_metadata = {key: dict(value) for key, value in prompt_metadata.items()}
    for key in sorted(selected_prompts):
        chat = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": selected_prompts[key]["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = base.tokenize_chat(model, chat)
        if tokens.shape[-1] < 2:
            raise ValueError(f"E8b prompt has no penultimate token position: {key}")
        tokens_by_prompt[key] = tokens
        runtime_prompt_metadata[key]["chat_text_sha256"] = sha256_text(chat)
        runtime_prompt_metadata[key]["input_token_count"] = int(tokens.shape[-1])
        runtime_prompt_metadata[key]["input_token_tensor_sha256"] = (
            tensor_identity_sha256(tokens)
        )

    prior: dict[str, Any] | None = None
    if args.output.exists():
        if not args.resume:
            raise ValueError(f"E8b output exists; use --resume: {args.output}")
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("status") == E8B_RESULT_STATUS:
            validate_e8b_result(prior, execution_plan, e8a_manifest, e8a_plan)
            raw = prior.get("private_raw_tensor_bundle")
            if not isinstance(raw, Mapping):
                raise ValueError("Completed E8b private tensor metadata differs")
            raw_path = raw_checkpoint_path(args.raw_tensors, raw)
            if not raw_path.is_file() or sha256_file(raw_path) != raw.get("sha256"):
                raise ValueError("Completed E8b private tensor bundle differs")
            print(json.dumps({"status": "already_complete"}, indent=2))
            return
        _validate_partial(prior, execution_plan)

    if prior is None:
        output: dict[str, Any] = {
            **result_identity_fields(execution_plan),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "status": E8B_RUNNING_STATUS,
            "mode": e8a_plan["mode"],
            "objective": e8a_plan["objective"],
            "execution_plan_file_sha256": sha256_file(args.execution_plan),
            "e8a_manifest_payload_sha256": e8a_manifest["payload_sha256"],
            "e8a_plan_payload_sha256": e8a_plan["payload_sha256"],
            "runner_file_sha256": sha256_file(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "runtime": runtime,
            "runtime_proof": runtime_proof,
            "model_snapshot_proof": model_snapshot_proof,
            "execution_input_hashes": execution_hashes,
            "prompt_records": [],
            "dose_audits": [],
            "sender_worlds": [],
            "route_cells": [],
            "forward_accounting": {
                "planned_scientific": EXPECTED_FORWARD_TOTAL,
                "completed_scientific": 0,
                "audit_forwards": 0,
                "cumulative_total_across_resumes": 0,
                "audit_ceiling": AUDIT_FORWARD_CEILING,
                "audit_ceiling_exceeded": False,
                "counter_reset_on_resume": False,
            },
            "resume_state": {"inflight_operation_id": None},
            "private_raw_tensor_bundle": None,
        }
        raw_tensors: dict[str, torch.Tensor] = {}
        _atomic(args.output, output)
    else:
        output = prior
        if output.get("runtime") != runtime:
            raise ValueError("E8b cross-resume runtime fingerprint differs")
        if output.get("runtime_proof") != runtime_proof:
            raise ValueError("E8b cross-resume runtime proof differs")
        if output.get("model_snapshot_proof") != model_snapshot_proof:
            raise ValueError("E8b resumed model snapshot proof differs")
        if output.get("execution_input_hashes") != execution_hashes:
            raise ValueError("E8b resumed execution-input hashes differ")
        if output.get("private_raw_tensor_bundle") is None:
            if completed_forward_count(output) != 0:
                raise ValueError("E8b resumable result lacks its private state bundle")
            raw_tensors = {}
        else:
            raw_path = raw_checkpoint_path(
                args.raw_tensors, output["private_raw_tensor_bundle"]
            )
            if not raw_path.is_file() or sha256_file(raw_path) != output["private_raw_tensor_bundle"]["sha256"]:
                raise ValueError("E8b resumable private tensor bundle differs")
            raw_tensors = dict(load_file(str(raw_path), device="cpu"))

    baselines: dict[str, dict[str, Any]] = {}
    for row in output["prompt_records"]:
        key = row["prompt_key"]
        expected = runtime_prompt_metadata[key]
        if any(
            row[field] != expected[field]
            for field in (
                "chat_text_sha256",
                "input_token_count",
                "input_token_tensor_sha256",
            )
        ):
            raise ValueError("E8b resumed prompt tokenization differs")
        baselines[key] = _world_from_record(row, raw_tensors)

    completed_prompts = set(baselines)
    for index, key in enumerate(sorted(selected_prompts), 1):
        if key in completed_prompts:
            continue
        _begin_forward(output, args.output, f"prompt_cache:{key}")
        world = _run_world(
            model, tokens_by_prompt[key], probe_direction, probe_bias, []
        )
        records = _store_world_states(
            world,
            prefix=f"prompt.{key}",
            raw_tensors=raw_tensors,
        )
        row = {
            **runtime_prompt_metadata[key],
            "metrics": world["metrics"],
            "diagnostics": world["diagnostics"],
            "state_records": records,
        }
        output["prompt_records"].append(row)
        output["prompt_records"].sort(key=lambda item: item["prompt_key"])
        baselines[key] = world
        _checkpoint(
            output=output,
            output_path=args.output,
            raw_tensors=raw_tensors,
            raw_path=args.raw_tensors,
            execution_plan_hash=execution_plan["payload_sha256"],
        )
        print(f"prompt cache {index}/{EXPECTED_PROMPT_RECORDS}: {key}", flush=True)

    pairs = sorted(e8a_plan["intent_pairs"], key=lambda row: row["intent_id"])
    computed_doses: list[dict[str, Any]] = []
    for pair in pairs:
        intent_id = pair["intent_id"]
        source_key = pair["source_prompt"]["prompt_key"]
        target_key = pair["target_prompt"]["prompt_key"]
        natural = compute_natural_activation_delta(
            encoder=feature_table["L31:F49541"]["encoder"],
            bias=feature_table["L31:F49541"]["bias"],
            source_input=baselines[source_key]["states"]["source_input_final"],
            target_input=baselines[target_key]["states"]["source_input_final"],
        )
        dose = compute_dose_audit(
            nominated_activation_delta=natural["activation_delta"],
            nominated_decoder=feature_table["L31:F49541"]["decoder"],
            control_decoder=feature_table["L31:F139261"]["decoder"],
        )
        computed_doses.append(
            {
                "dose_audit_id": f"e8b:{intent_id}:dose",
                "pair_id": pair["pair_id"],
                "intent_id": intent_id,
                "risk_class": pair["risk_class"],
                "source_prompt_key": source_key,
                "target_prompt_key": target_key,
                "nominated_source_feature": "L31:F49541",
                "control_source_feature": "L31:F139261",
                "natural_c": natural,
                **dose,
                "zero_nominated_delta_retained": natural["activation_delta"] == 0.0,
            }
        )
    if len(computed_doses) != EXPECTED_DOSE_AUDITS:
        raise ValueError("E8b intent-dose closure differs")
    if output["dose_audits"] and output["dose_audits"] != computed_doses:
        raise ValueError("E8b resumed intent doses differ")
    output["dose_audits"] = computed_doses
    _atomic(args.output, output)
    doses_by_intent = {row["intent_id"]: row for row in computed_doses}

    def edit_vector(intent_id: str, arm: str) -> tuple[str, float, torch.Tensor, dict[str, Any]]:
        dose = doses_by_intent[intent_id]
        if arm == "nominated_c":
            feature = "L31:F49541"
            activation_delta = float(dose["nominated_activation_delta"])
            nominal_l2 = float(dose["nominal_nominated_edit_l2"])
            realized_l2 = float(dose["realized_bfloat16_nominated_edit_l2"])
        elif arm == "equal_l2_control":
            feature = "L31:F139261"
            activation_delta = float(dose["control_activation_delta"])
            nominal_l2 = float(dose["nominal_control_edit_l2"])
            realized_l2 = float(dose["realized_bfloat16_control_edit_l2"])
        else:
            raise ValueError(f"Unexpected E8b source arm: {arm}")
        vector = feature_table[feature]["decoder"].float() * activation_delta
        return feature, activation_delta, vector, {
            "nominal_applied_residual_l2": nominal_l2,
            "realized_bfloat16_applied_residual_l2": realized_l2,
        }

    senders: dict[tuple[str, str], dict[str, Any]] = {}
    for row in output["sender_worlds"]:
        senders[(row["intent_id"], row["arm"])] = _world_from_record(row, raw_tensors)
    completed_senders = set(senders)
    sender_index = 0
    for pair in pairs:
        intent_id = pair["intent_id"]
        target_key = pair["target_prompt"]["prompt_key"]
        for arm in ("nominated_c", "equal_l2_control"):
            sender_index += 1
            if (intent_id, arm) in completed_senders:
                continue
            feature, activation_delta, vector, norms = edit_vector(intent_id, arm)
            sender_id = f"e8b:{intent_id}:{arm}:sender"
            _begin_forward(output, args.output, f"sender_world:{sender_id}")
            world = _run_world(
                model,
                tokens_by_prompt[target_key],
                probe_direction,
                probe_bias,
                [("blocks.31.mlp.hook_out", _source_edit_hook(vector))],
            )
            records = _store_world_states(
                world,
                prefix=f"sender.{intent_id}.{arm}",
                raw_tensors=raw_tensors,
            )
            sender_row = {
                "sender_world_id": sender_id,
                "dose_audit_id": f"e8b:{intent_id}:dose",
                "pair_id": pair["pair_id"],
                "intent_id": intent_id,
                "risk_class": pair["risk_class"],
                "target_prompt_key": target_key,
                "arm": arm,
                "source_feature": feature,
                "source_activation_delta": activation_delta,
                **norms,
                "endpoints": {
                    metric: {
                        "Y0": float(baselines[target_key]["metrics"][metric]),
                        "YS": float(world["metrics"][metric]),
                        "T_S": float(world["metrics"][metric])
                        - float(baselines[target_key]["metrics"][metric]),
                    }
                    for metric in METRICS
                },
                "metrics": world["metrics"],
                "diagnostics": world["diagnostics"],
                "state_records": records,
            }
            output["sender_worlds"].append(sender_row)
            output["sender_worlds"].sort(key=lambda item: item["sender_world_id"])
            senders[(intent_id, arm)] = world
            _checkpoint(
                output=output,
                output_path=args.output,
                raw_tensors=raw_tensors,
                raw_path=args.raw_tensors,
                execution_plan_hash=execution_plan["payload_sha256"],
            )
            print(
                f"sender world {sender_index}/{EXPECTED_SENDER_WORLDS}: {sender_id}",
                flush=True,
            )

    prompt_state_records = {
        row["prompt_key"]: row["state_records"] for row in output["prompt_records"]
    }
    sender_state_records = {
        (row["intent_id"], row["arm"]): row["state_records"]
        for row in output["sender_worlds"]
    }
    completed_cells = {row["cell_id"] for row in output["route_cells"]}
    technical = e8a_plan["technical_evaluability"]
    mismatch_atol = float(technical["pre_replacement_max_abs_mismatch"])
    invariant_atol = float(technical["invariant_absolute_tolerance"])
    penultimate_state_atol = float(
        technical["penultimate_receiver_state_max_abs_change"]
    )
    penultimate_endpoint_atol = float(
        technical["penultimate_endpoint_abs_tolerance"]
    )
    ordered_cells = sorted(e8a_plan["route_cells"], key=lambda row: row["cell_id"])
    for index, cell in enumerate(ordered_cells, 1):
        cell_id = cell["cell_id"]
        if cell_id in completed_cells:
            continue
        intent_id = cell["intent_id"]
        arm = cell["arm"]
        position = cell["receiver_token_position"]
        target_key = cell["target_prompt_key"]
        feature, activation_delta, vector, _ = edit_vector(intent_id, arm)
        baseline = baselines[target_key]
        sender = senders[(intent_id, arm)]
        baseline_receiver = baseline["states"]["receiver_output"][position]
        sender_receiver = sender["states"]["receiver_output"][position]

        restoration_mismatch: list[float] = []
        rescue_mismatch: list[float] = []
        _begin_forward(output, args.output, f"route_restore:{cell_id}")
        restored = _run_endpoints(
            model,
            tokens_by_prompt[target_key],
            probe_direction,
            probe_bias,
            [
                ("blocks.31.mlp.hook_out", _source_edit_hook(vector)),
                (
                    "blocks.32.mlp.hook_out",
                    _replace_hook(
                        baseline_receiver,
                        expected_before=sender_receiver,
                        position=position,
                        mismatch_atol=mismatch_atol,
                        mismatch=restoration_mismatch,
                    ),
                ),
            ],
        )
        _begin_forward(output, args.output, f"route_rescue:{cell_id}")
        rescued = _run_endpoints(
            model,
            tokens_by_prompt[target_key],
            probe_direction,
            probe_bias,
            [
                (
                    "blocks.32.mlp.hook_out",
                    _replace_hook(
                        sender_receiver,
                        expected_before=baseline_receiver,
                        position=position,
                        mismatch_atol=mismatch_atol,
                        mismatch=rescue_mismatch,
                    ),
                )
            ],
        )
        world_metrics = {
            "Y0": baseline["metrics"],
            "YS": sender["metrics"],
            "YS_R0": restored["metrics"],
            "YR_S": rescued["metrics"],
        }
        endpoints = route_effects(world_metrics, invariant_atol=invariant_atol)
        baseline_record = prompt_state_records[target_key]["receiver_output"][position]
        sender_record = sender_state_records[(intent_id, arm)]["receiver_output"][position]
        receiver_state = _receiver_state(
            baseline_receiver,
            sender_receiver,
            baseline_record,
            sender_record,
        )
        gate = _gate_diagnostic(
            feature_table["L32:F50401"],
            baseline["states"]["receiver_input"][position],
            sender["states"]["receiver_input"][position],
        )
        penultimate_thresholds_passed = True
        if position == PENULTIMATE_POSITION:
            penultimate_thresholds_passed = (
                receiver_state["state_delta_max_abs"] <= penultimate_state_atol
                and all(
                    abs(values["D_R"]) <= penultimate_endpoint_atol
                    and abs(values["Q_R"]) <= penultimate_endpoint_atol
                    for values in endpoints.values()
                )
            )
        route_row = {
            "cell_id": cell_id,
            "source_cell_id": cell_id,
            "dose_audit_id": f"e8b:{intent_id}:dose",
            "sender_world_id": f"e8b:{intent_id}:{arm}:sender",
            "pair_id": cell["pair_id"],
            "intent_id": intent_id,
            "risk_class": cell["risk_class"],
            "arm": arm,
            "source_feature": feature,
            "source_activation_delta": activation_delta,
            "receiver_layer": cell["receiver_layer"],
            "receiver_boundary": cell["receiver_boundary"],
            "receiver_hook": cell["receiver_hook"],
            "receiver_token_position": position,
            "receiver_position_role": cell["receiver_position_role"],
            "included_in_empirical_analysis": cell["included_in_empirical_analysis"],
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
            "gate_diagnostic": gate,
            "receiver_audit": {
                "restoration_hook_fire_count": len(restoration_mismatch),
                "rescue_hook_fire_count": len(rescue_mismatch),
                "restoration_pre_replacement_max_abs_mismatch": max(
                    restoration_mismatch, default=0.0
                ),
                "rescue_pre_replacement_max_abs_mismatch": max(
                    rescue_mismatch, default=0.0
                ),
                "mismatch_atol": mismatch_atol,
            },
            "receiver_state": receiver_state,
            "penultimate_thresholds_passed": penultimate_thresholds_passed,
        }
        output["route_cells"].append(route_row)
        output["route_cells"].sort(key=lambda item: item["cell_id"])
        _checkpoint(
            output=output,
            output_path=args.output,
            raw_tensors=raw_tensors,
            raw_path=args.raw_tensors,
            execution_plan_hash=execution_plan["payload_sha256"],
        )
        print(f"route {index}/{EXPECTED_ROUTE_CELLS}: {cell_id}", flush=True)

    if completed_forward_count(output, verify_recorded=True) != EXPECTED_FORWARD_TOTAL:
        raise ValueError("E8b final scientific forward accounting differs")
    if output["forward_accounting"]["audit_ceiling_exceeded"]:
        raise ValueError("E8b cumulative audit ceiling was exceeded")
    candidate = copy.deepcopy(output)
    for row in candidate["route_cells"]:
        row["gate_diagnostic"] = classify_gate_diagnostic_after_cell_closure(
            row["gate_diagnostic"]
        )
    candidate["status"] = E8B_RESULT_STATUS
    candidate["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    finalize_result_contract_fields(candidate, execution_plan)
    assert_public_safe(candidate)
    candidate["payload_sha256"] = canonical_sha256(candidate)
    validate_e8b_result(candidate, execution_plan, e8a_manifest, e8a_plan)
    _atomic(args.output, candidate)
    output = candidate
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": output["status"],
                "forward_accounting": output["forward_accounting"],
                "payload_sha256": output["payload_sha256"],
                "private_raw_tensor_sha256": output["private_raw_tensor_bundle"][
                    "sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
