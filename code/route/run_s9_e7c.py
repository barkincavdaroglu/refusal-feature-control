#!/usr/bin/env python3
"""Run or resume E7c exact complete-receiver restoration and rescue."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import run_s9_calibration as base
from s9_contract import METRICS, REFUSAL_TOKEN_IDS, canonical_sha256, sha256_file, sha256_text, validate_stamped_payload
from s9_e7_contract import (
    E7C_RESULT_STATUS,
    E7C_RUNNING_STATUS,
    E7C_SEMANTICS_VERSION,
    E7_INPUT_NAMES,
    E7_PRIVACY,
    E7_SCHEMA_VERSION,
    validate_e7b_manifest,
    validate_e7b_plan,
    validate_e7c_result,
)


MISMATCH_ATOL = 1e-4


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _capture_exact(storage: dict[str, torch.Tensor], name: str):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        storage[name] = base.final_vector(activations).detach().cpu().contiguous()
        return activations

    return hook


def _diagnostics(logits: torch.Tensor) -> dict[str, Any]:
    values = base.final_vector(logits).detach().float()
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
        "refusal_top_minus_nonrefusal_top_logit_margin": float((refusal_top - nonrefusal_top).cpu()),
    }


@torch.inference_mode()
def _run_world(
    model,
    tokens: torch.Tensor,
    probe_direction: torch.Tensor,
    probe_bias: float,
    layers: list[int],
    intervention_hooks: list[tuple[str, Any]],
) -> dict[str, Any]:
    states: dict[str, torch.Tensor] = {}
    final: dict[str, torch.Tensor] = {}
    hooks = list(intervention_hooks)
    for layer in layers:
        hooks.extend(
            (
                (f"blocks.{layer}.mlp.hook_in", _capture_exact(states, f"{layer}:module_input")),
                (f"blocks.{layer}.mlp.hook_out", _capture_exact(states, f"{layer}:module_output")),
            )
        )
    hooks.append(("unembed.hook_pre", _capture_exact(final, "final")))
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    final_normalized = final["final"].to(device=logits.device)
    metrics = base.detached_metric_values(
        base.differentiable_metrics(logits, final_normalized, probe_direction, probe_bias)
    )
    return {
        "metrics": metrics,
        "diagnostics": _diagnostics(logits),
        "states": {
            layer: {
                "module_input": states[f"{layer}:module_input"],
                "module_output": states[f"{layer}:module_output"],
            }
            for layer in layers
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
    all_hooks = [*hooks, ("unembed.hook_pre", _capture_exact(final, "final"))]
    logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)
    final_normalized = final["final"].to(device=logits.device)
    return {
        "metrics": base.detached_metric_values(
            base.differentiable_metrics(logits, final_normalized, probe_direction, probe_bias)
        ),
        "diagnostics": _diagnostics(logits),
    }


def _replace_hook(
    replacement: torch.Tensor,
    *,
    expected_before: torch.Tensor | None,
    mismatch: list[float],
):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        current = base.final_vector(activations)
        if expected_before is None:
            mismatch.append(float("nan"))
        else:
            expected = expected_before.to(device=current.device, dtype=current.dtype)
            value = float((current.float() - expected.float()).abs().max().detach().cpu())
            mismatch.append(value)
            if value > MISMATCH_ATOL:
                raise ValueError(f"Receiver state drift before replacement: {value}")
        value = replacement.to(device=current.device, dtype=current.dtype)
        return base.replace_final_vector(activations, value)

    return hook


def _source_hook(feature_table: Mapping[str, Mapping[str, Any]], edit: Mapping[str, Any]):
    record = feature_table[edit["source_feature"]]
    return (
        record["output_hook"],
        base.feature_delta_hook(record["decoder"], float(edit["source_activation_delta"]), 1.0),
    )


def _state_record(tensor: torch.Tensor, raw_key: str) -> dict[str, Any]:
    return {
        "raw_tensor_key": raw_key,
        "sha256": _tensor_sha256(tensor),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "l2": float(tensor.float().norm()),
    }


def _store_world_states(
    world: Mapping[str, Any],
    *,
    prefix: str,
    raw_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer, boundaries in sorted(world["states"].items()):
        output[str(layer)] = {}
        for boundary, tensor in sorted(boundaries.items()):
            key = f"{prefix}.L{layer}.{boundary}"
            if key in raw_tensors:
                raise ValueError(f"Duplicate E7 raw tensor key: {key}")
            raw_tensors[key] = tensor.contiguous().cpu()
            output[str(layer)][boundary] = _state_record(tensor, key)
    return output


def _restore_world_states(
    records: Mapping[str, Any], raw_tensors: Mapping[str, torch.Tensor]
) -> dict[int, dict[str, torch.Tensor]]:
    output: dict[int, dict[str, torch.Tensor]] = {}
    for layer, boundaries in records.items():
        output[int(layer)] = {}
        for boundary, record in boundaries.items():
            key = record["raw_tensor_key"]
            if key not in raw_tensors:
                raise ValueError(f"E7 raw tensor is missing: {key}")
            tensor = raw_tensors[key]
            if _tensor_sha256(tensor) != record["sha256"]:
                raise ValueError(f"E7 raw tensor hash differs: {key}")
            output[int(layer)][boundary] = tensor
    return output


def _save_raw(path: Path, tensors: Mapping[str, torch.Tensor], plan_hash: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(tensors), str(temporary), metadata={"plan_payload_sha256": plan_hash})
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tensor_count": len(tensors),
        "public": False,
        "contains_prompt_or_token_text": False,
    }


def _effects(worlds: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output = {}
    for metric in METRICS:
        y0 = float(worlds["Y0"]["metrics"][metric])
        ys = float(worlds["YS"]["metrics"][metric])
        ysr0 = float(worlds["YS_R0"]["metrics"][metric])
        yrs = float(worlds["YR_S"]["metrics"][metric])
        total = ys - y0
        dependence = ys - ysr0
        bypass = ysr0 - y0
        rescue = yrs - y0
        invariant = total - dependence - bypass
        if abs(invariant) > 1e-7:
            raise ValueError(f"E7 route invariant failed for {metric}: {invariant}")
        output[metric] = {
            "Y0": y0,
            "YS": ys,
            "YS_R0": ysr0,
            "YR_S": yrs,
            "T_S": total,
            "D_R": dependence,
            "B_R": bypass,
            "Q_R": rescue,
            "invariant_error": invariant,
        }
    return output


def _state_delta(
    baseline: torch.Tensor,
    sender: torch.Tensor,
    baseline_record: Mapping[str, Any],
    sender_record: Mapping[str, Any],
) -> dict[str, Any]:
    delta = sender.float() - baseline.float()
    return {
        "baseline_tensor_sha256": baseline_record["sha256"],
        "sender_tensor_sha256": sender_record["sha256"],
        "baseline_raw_tensor_key": baseline_record["raw_tensor_key"],
        "sender_raw_tensor_key": sender_record["raw_tensor_key"],
        "state_delta_l2": float(delta.norm()),
        "state_delta_max_abs": float(delta.abs().max()),
        "receiver_moved": bool(float(delta.norm()) > 0.0),
    }


def _feature_diagnostic(
    feature_table: Mapping[str, Mapping[str, Any]],
    feature: str | None,
    baseline_input: torch.Tensor,
    sender_input: torch.Tensor,
) -> dict[str, Any] | None:
    if feature is None:
        return None
    record = feature_table[feature]

    def values(tensor: torch.Tensor) -> tuple[float, float]:
        vector = tensor.to(device=record["encoder"].device, dtype=record["encoder"].dtype)
        pre = torch.dot(record["encoder"], vector) + record["bias"]
        activation = torch.relu(pre)
        return float(pre.detach().float().cpu()), float(activation.detach().float().cpu())

    baseline_pre, baseline_activation = values(baseline_input)
    sender_pre, sender_activation = values(sender_input)
    return {
        "feature": feature,
        "baseline_preactivation": baseline_pre,
        "sender_preactivation": sender_pre,
        "preactivation_delta": sender_pre - baseline_pre,
        "baseline_activation": baseline_activation,
        "sender_activation": sender_activation,
        "activation_delta": sender_activation - baseline_activation,
        "baseline_active": baseline_activation > 0.0,
        "sender_active": sender_activation > 0.0,
        "finite_gate_crossing": (baseline_activation > 0.0) != (sender_activation > 0.0),
    }


def _single_route(
    *,
    result_id: str,
    row: Mapping[str, Any],
    boundary: str,
    model,
    tokens: torch.Tensor,
    feature_table: Mapping[str, Mapping[str, Any]],
    edit: Mapping[str, Any],
    baseline: Mapping[str, Any],
    sender: Mapping[str, Any],
    baseline_records: Mapping[str, Any],
    sender_records: Mapping[str, Any],
    probe_direction: torch.Tensor,
    probe_bias: float,
) -> dict[str, Any]:
    layer = int(row["receiver_layer"])
    hook_name = f"blocks.{layer}.mlp.hook_{'out' if boundary == 'module_output' else 'in'}"
    baseline_tensor = baseline["states"][layer][boundary]
    sender_tensor = sender["states"][layer][boundary]
    restore_mismatch: list[float] = []
    rescue_mismatch: list[float] = []
    restored = _run_endpoints(
        model,
        tokens,
        probe_direction,
        probe_bias,
        [
            _source_hook(feature_table, edit),
            (
                hook_name,
                _replace_hook(
                    baseline_tensor,
                    expected_before=sender_tensor,
                    mismatch=restore_mismatch,
                ),
            ),
        ],
    )
    rescued = _run_endpoints(
        model,
        tokens,
        probe_direction,
        probe_bias,
        [
            (
                hook_name,
                _replace_hook(
                    sender_tensor,
                    expected_before=baseline_tensor,
                    mismatch=rescue_mismatch,
                ),
            )
        ],
    )
    if len(restore_mismatch) != 1 or len(rescue_mismatch) != 1:
        raise RuntimeError(f"E7 receiver hook did not fire exactly once: {result_id}")
    worlds = {
        "Y0": {"metrics": baseline["metrics"], "diagnostics": baseline["diagnostics"]},
        "YS": {"metrics": sender["metrics"], "diagnostics": sender["diagnostics"]},
        "YS_R0": restored,
        "YR_S": rescued,
    }
    baseline_record = baseline_records[str(layer)][boundary]
    sender_record = sender_records[str(layer)][boundary]
    return {
        "result_id": result_id,
        "source_cell_id": row.get("source_cell_id", row.get("cell_id")),
        "panel": row.get("panel"),
        "direction_id": row["direction_id"],
        "source_edit_id": row["source_edit_id"],
        "source_feature": row["source_feature"],
        "receiver_layer": layer,
        "receiver_boundary": boundary,
        "receiver_hook": hook_name,
        "receiver_token_position": "final_prompt_position",
        "nominated_receiver_feature": row.get("nominated_receiver_feature"),
        "role": row.get("role"),
        "topology": row.get("topology"),
        "worlds": worlds,
        "endpoints": _effects(worlds),
        "receiver_state": _state_delta(
            baseline_tensor, sender_tensor, baseline_record, sender_record
        ),
        "receiver_feature_diagnostic": _feature_diagnostic(
            feature_table,
            row.get("nominated_receiver_feature"),
            baseline["states"][layer]["module_input"],
            sender["states"][layer]["module_input"],
        ),
        "hook_audit": {
            "restoration_pre_replacement_max_abs_mismatch": restore_mismatch[0],
            "rescue_pre_replacement_max_abs_mismatch": rescue_mismatch[0],
            "mismatch_atol": MISMATCH_ATOL,
        },
    }


def _joint_route(
    *,
    row: Mapping[str, Any],
    source_cell: Mapping[str, Any],
    model,
    tokens: torch.Tensor,
    feature_table: Mapping[str, Mapping[str, Any]],
    edit: Mapping[str, Any],
    baseline: Mapping[str, Any],
    sender: Mapping[str, Any],
    baseline_records: Mapping[str, Any],
    sender_records: Mapping[str, Any],
    probe_direction: torch.Tensor,
    probe_bias: float,
) -> dict[str, Any]:
    layers = list(row["receiver_layers"])
    restore_mismatch: list[float] = []
    rescue_mismatch: list[float] = []
    restore_hooks = [_source_hook(feature_table, edit)]
    rescue_hooks = []
    for layer in layers:
        restore_hooks.append(
            (
                f"blocks.{layer}.mlp.hook_out",
                _replace_hook(
                    baseline["states"][layer]["module_output"],
                    expected_before=None,
                    mismatch=restore_mismatch,
                ),
            )
        )
        rescue_hooks.append(
            (
                f"blocks.{layer}.mlp.hook_out",
                _replace_hook(
                    sender["states"][layer]["module_output"],
                    expected_before=None,
                    mismatch=rescue_mismatch,
                ),
            )
        )
    restored = _run_endpoints(model, tokens, probe_direction, probe_bias, restore_hooks)
    rescued = _run_endpoints(model, tokens, probe_direction, probe_bias, rescue_hooks)
    if len(restore_mismatch) != len(layers) or len(rescue_mismatch) != len(layers):
        raise RuntimeError(f"E7 joint receiver hook count differs: {row['cell_id']}")
    worlds = {
        "Y0": {"metrics": baseline["metrics"], "diagnostics": baseline["diagnostics"]},
        "YS": {"metrics": sender["metrics"], "diagnostics": sender["diagnostics"]},
        "YS_R0": restored,
        "YR_S": rescued,
    }
    states = []
    for layer in layers:
        boundary = "module_output"
        states.append(
            {
                "receiver_layer": layer,
                **_state_delta(
                    baseline["states"][layer][boundary],
                    sender["states"][layer][boundary],
                    baseline_records[str(layer)][boundary],
                    sender_records[str(layer)][boundary],
                ),
            }
        )
    return {
        "result_id": row["cell_id"],
        "source_cell_id": row["source_cell_id"],
        "panel": "joint_receiver_set",
        "direction_id": row["direction_id"],
        "source_edit_id": row["source_edit_id"],
        "source_feature": row["source_feature"],
        "receiver_layers": layers,
        "receiver_boundary": "module_output",
        "receiver_hooks": [f"blocks.{layer}.mlp.hook_out" for layer in layers],
        "receiver_token_position": "final_prompt_position",
        "interpretation": row["interpretation"],
        "topology": source_cell.get("topology"),
        "worlds": worlds,
        "endpoints": _effects(worlds),
        "receiver_states": states,
        "hook_audit": {
            "pre_replacement_mismatch_enforced": False,
            "reason": "an earlier joint receiver replacement changes later pre-replacement state",
            "receiver_hook_count_restoration": len(restore_mismatch),
            "receiver_hook_count_rescue": len(rescue_mismatch),
        },
    }


def _semantic_paths(args: argparse.Namespace) -> dict[str, Path]:
    here = Path(__file__).resolve()
    return {
        "prompts": args.prompts,
        "annotations": args.annotations,
        "probe_direction": args.probe_direction,
        "candidate_manifest": args.manifest,
        "manifest_builder": here.with_name("build_s9_e7b_candidate_manifest.py"),
        "topology_source": here.with_name("e7_topology.py"),
        "e7a_semantics": here.with_name("receiver_route_semantics.py"),
        "e7_contract": here.with_name("s9_e7_contract.py"),
        "plan_builder": here.with_name("build_s9_e7b_plan.py"),
        "runner": here,
        "colab_runner": here.with_name("run_s9_e7c_colab.py"),
        "runtime_probe": here.with_name("probe_s9_e7c_runtime_colab.py"),
        "analyzer": here.with_name("analyze_s9_e7c.py"),
        "runtime_installer": here.with_name("install_s9_colab_runtime.py"),
        "base_runner": Path(base.__file__).resolve(),
        "base_contract": here.with_name("s9_contract.py"),
    }


def _validate_partial(payload: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_stamped_payload(payload)
    if payload.get("status") != E7C_RUNNING_STATUS:
        raise ValueError("E7c resumable status differs")
    if payload.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("E7c resumable output references another plan")
    for key in (
        "prompt_baselines",
        "sender_worlds",
        "primary_route_cells",
        "input_calibrations",
        "low_coverage_controls",
        "joint_receiver_sets",
    ):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"E7c resumable {key} is missing")
    for key in (
        "primary_route_cells",
        "input_calibrations",
        "low_coverage_controls",
        "joint_receiver_sets",
    ):
        ids = [row.get("result_id") for row in payload[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"E7c resumable {key} contains duplicate rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--probe-direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-tensors", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E7c")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_e7b_plan(plan)
    validate_e7b_manifest(manifest)
    if plan["candidate_manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("E7b plan references another candidate manifest")
    paths = _semantic_paths(args)
    if set(paths) != set(E7_INPUT_NAMES):
        raise AssertionError("E7c runtime semantic-input closure differs")
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != plan["inputs"][name]["sha256"]:
            raise ValueError(f"E7c runtime semantic input differs: {name}")
    for package, expected in plan["runtime_contract"]["package_versions"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"E7c package differs: {package}={actual} != {expected}")
    device_name = torch.cuda.get_device_name(0)
    if not any(name in device_name for name in plan["runtime_contract"]["preferred_accelerators"]):
        raise ValueError(f"Unsupported E7c accelerator: {device_name}")
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory < plan["runtime_contract"]["minimum_cuda_memory_gib"]:
        raise ValueError(f"E7c GPU memory below contract: {memory:.2f} GiB")

    from transformer_lens import HookedTransformer

    device = torch.device("cuda")
    torch.manual_seed(int(plan["random_seed"]))
    torch.cuda.manual_seed_all(int(plan["random_seed"]))
    print(f"Loading {plan['model']['id']} on {device_name}", flush=True)
    model = HookedTransformer.from_pretrained(
        plan["model"]["id"],
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        dtype=torch.bfloat16,
        revision=plan["model"]["revision"],
    )
    for block in model.blocks:
        block.mlp = base.ExactHookedMLP(block.mlp)
    model.unembed = base.ExactHookedUnembed(model.unembed)
    model.setup()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    probe = np.load(args.probe_direction)
    probe_direction = torch.tensor(
        probe["direction_unit"].astype(np.float32), device=device, dtype=torch.float32
    )
    probe_bias = float(probe["bias_unit"])
    feature_table = base.load_feature_table(model, plan, device)
    for edit in plan["source_edits"]:
        record = feature_table[edit["source_feature"]]
        actual_l2 = float(
            (record["decoder"].float() * float(edit["source_activation_delta"])).norm().cpu()
        )
        if not math.isclose(actual_l2, float(edit["source_edit_l2"]), rel_tol=2e-4, abs_tol=2e-3):
            raise ValueError(f"E7 source-edit L2 differs: {edit['source_edit_id']}")

    prompt_rows = base.load_prompt_rows(args.prompts, plan)
    prompt_plan = {row["prompt_key"]: dict(row) for row in plan["prompt_records"]}
    tokens_by_prompt: dict[str, torch.Tensor] = {}
    for key in sorted(prompt_rows):
        chat = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_rows[key]["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens_by_prompt[key] = base.tokenize_chat(model, chat)
        prompt_plan[key]["chat_text_sha256"] = sha256_text(chat)
        prompt_plan[key]["input_token_count"] = int(tokens_by_prompt[key].shape[-1])

    all_cells = [*plan["mechanics_cells"], *plan["topology_cells"]]
    cells_by_id = {row["cell_id"]: row for row in all_cells}
    edits_by_id = {row["source_edit_id"]: row for row in plan["source_edits"]}
    layers = sorted(
        {
            *[int(row["receiver_layer"]) for row in all_cells],
            *[int(row["receiver_layer"]) for row in plan["low_coverage_controls"]],
            *[int(layer) for row in plan["joint_receiver_sets"] for layer in row["receiver_layers"]],
        }
    )

    prior: dict[str, Any] | None = None
    if args.output.exists():
        if not args.resume:
            raise ValueError(f"E7c output exists; use --resume: {args.output}")
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("status") == E7C_RESULT_STATUS:
            validate_e7c_result(prior, plan)
            if not args.raw_tensors.is_file() or sha256_file(args.raw_tensors) != prior["private_raw_tensor_bundle"]["sha256"]:
                raise ValueError("Completed E7c private tensor bundle differs")
            print(json.dumps({"status": "already_complete"}, indent=2))
            return
        _validate_partial(prior, plan)
        if not args.raw_tensors.is_file() or sha256_file(args.raw_tensors) != prior["private_raw_tensor_bundle"]["sha256"]:
            raise ValueError("Resumable E7c private tensor bundle differs")

    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device_name,
        "cuda_total_memory_gib": memory,
        "packages": {
            package: importlib.metadata.version(package)
            for package in plan["runtime_contract"]["package_versions"]
        },
    }
    if prior is None:
        output: dict[str, Any] = {
            "schema_version": E7_SCHEMA_VERSION,
            "analysis_semantics_version": E7C_SEMANTICS_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "status": E7C_RUNNING_STATUS,
            "mode": plan["mode"],
            "objective": plan["objective"],
            "plan_payload_sha256": plan["payload_sha256"],
            "plan_file_sha256": sha256_file(args.plan),
            "runner_file_sha256": sha256_file(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "runtime": runtime,
            "generation_executed": False,
            "prompt_baselines": [],
            "sender_worlds": [],
            "primary_route_cells": [],
            "input_calibrations": [],
            "low_coverage_controls": [],
            "joint_receiver_sets": [],
            "forward_accounting": {
                "planned": plan["expected_forwards"]["total"],
                "audit_ceiling": plan["expected_forwards"]["audit_ceiling"],
                "completed": 0,
                "audit_ceiling_exceeded": False,
            },
            "private_raw_tensor_bundle": None,
            "privacy": E7_PRIVACY,
        }
        raw_tensors: dict[str, torch.Tensor] = {}
        baselines: dict[str, dict[str, Any]] = {}
        for index, key in enumerate(sorted(prompt_rows)):
            world = _run_world(
                model, tokens_by_prompt[key], probe_direction, probe_bias, layers, []
            )
            records = _store_world_states(
                world, prefix=f"baseline.{index:03d}", raw_tensors=raw_tensors
            )
            baselines[key] = world
            output["prompt_baselines"].append(
                {
                    **prompt_plan[key],
                    "metrics": world["metrics"],
                    "diagnostics": world["diagnostics"],
                    "state_records": records,
                }
            )
            output["forward_accounting"]["completed"] += 1
            print(f"baseline {index + 1}/{len(prompt_rows)}: {key}", flush=True)
        senders: dict[str, dict[str, Any]] = {}
        for index, edit in enumerate(plan["source_edits"]):
            key = edit["target_prompt_key"]
            world = _run_world(
                model,
                tokens_by_prompt[key],
                probe_direction,
                probe_bias,
                layers,
                [_source_hook(feature_table, edit)],
            )
            records = _store_world_states(
                world, prefix=f"sender.{index:03d}", raw_tensors=raw_tensors
            )
            senders[edit["source_edit_id"]] = world
            output["sender_worlds"].append(
                {
                    **edit,
                    "metrics": world["metrics"],
                    "diagnostics": world["diagnostics"],
                    "state_records": records,
                }
            )
            output["forward_accounting"]["completed"] += 1
            print(f"sender {index + 1}/{len(plan['source_edits'])}: {edit['source_edit_id']}", flush=True)
        output["private_raw_tensor_bundle"] = _save_raw(
            args.raw_tensors, raw_tensors, plan["payload_sha256"]
        )
        _atomic(args.output, output)
    else:
        output = prior
        raw_tensors = load_file(str(args.raw_tensors), device="cpu")
        baselines = {}
        for row in output["prompt_baselines"]:
            baselines[row["prompt_key"]] = {
                "metrics": row["metrics"],
                "diagnostics": row["diagnostics"],
                "states": _restore_world_states(row["state_records"], raw_tensors),
            }
            expected = prompt_plan[row["prompt_key"]]
            if row["chat_text_sha256"] != expected["chat_text_sha256"] or row["input_token_count"] != expected["input_token_count"]:
                raise ValueError("E7c resumed prompt tokenization differs")
        senders = {}
        for row in output["sender_worlds"]:
            senders[row["source_edit_id"]] = {
                "metrics": row["metrics"],
                "diagnostics": row["diagnostics"],
                "states": _restore_world_states(row["state_records"], raw_tensors),
            }

    baseline_records = {
        row["prompt_key"]: row["state_records"] for row in output["prompt_baselines"]
    }
    sender_records = {
        row["source_edit_id"]: row["state_records"] for row in output["sender_worlds"]
    }

    def finish_row(table: str, result: dict[str, Any]) -> None:
        output[table].append(result)
        output[table].sort(key=lambda row: row["result_id"])
        output["forward_accounting"]["completed"] += 2
        output["forward_accounting"]["audit_ceiling_exceeded"] = (
            output["forward_accounting"]["completed"]
            > output["forward_accounting"]["audit_ceiling"]
        )
        if output["forward_accounting"]["audit_ceiling_exceeded"]:
            raise RuntimeError("E7c audit forward ceiling exceeded")
        _atomic(args.output, output)

    completed = {row["result_id"] for row in output["primary_route_cells"]}
    for index, cell in enumerate(sorted(all_cells, key=lambda row: row["cell_id"]), 1):
        result_id = f"primary:{cell['cell_id']}"
        if result_id in completed:
            continue
        edit_id = cell["source_edit_id"]
        edit = edits_by_id[edit_id]
        target = edit["target_prompt_key"]
        result = _single_route(
            result_id=result_id,
            row=cell,
            boundary="module_output",
            model=model,
            tokens=tokens_by_prompt[target],
            feature_table=feature_table,
            edit=edit,
            baseline=baselines[target],
            sender=senders[edit_id],
            baseline_records=baseline_records[target],
            sender_records=sender_records[edit_id],
            probe_direction=probe_direction,
            probe_bias=probe_bias,
        )
        finish_row("primary_route_cells", result)
        print(f"primary {index}/{len(all_cells)}: {cell['cell_id']}", flush=True)

    completed = {row["result_id"] for row in output["input_calibrations"]}
    for index, cell_id in enumerate(plan["input_calibration_cell_ids"], 1):
        result_id = f"input_calibration:{cell_id}"
        if result_id in completed:
            continue
        cell = cells_by_id[cell_id]
        edit_id = cell["source_edit_id"]
        edit = edits_by_id[edit_id]
        target = edit["target_prompt_key"]
        result = _single_route(
            result_id=result_id,
            row=cell,
            boundary="module_input",
            model=model,
            tokens=tokens_by_prompt[target],
            feature_table=feature_table,
            edit=edit,
            baseline=baselines[target],
            sender=senders[edit_id],
            baseline_records=baseline_records[target],
            sender_records=sender_records[edit_id],
            probe_direction=probe_direction,
            probe_bias=probe_bias,
        )
        finish_row("input_calibrations", result)
        print(f"input calibration {index}/{len(plan['input_calibration_cell_ids'])}: {cell_id}", flush=True)

    completed = {row["result_id"] for row in output["low_coverage_controls"]}
    for index, control in enumerate(plan["low_coverage_controls"], 1):
        result_id = control["control_id"]
        if result_id in completed:
            continue
        source_cell = cells_by_id[control["source_cell_id"]]
        row = {
            **control,
            "cell_id": control["control_id"],
            "source_feature": source_cell["source_feature"],
            "nominated_receiver_feature": None,
            "role": "operational_low_coverage_module_control",
            "topology": {"unsigned_layer_selection": control["selection"]},
        }
        edit_id = control["source_edit_id"]
        edit = edits_by_id[edit_id]
        target = edit["target_prompt_key"]
        result = _single_route(
            result_id=result_id,
            row=row,
            boundary="module_output",
            model=model,
            tokens=tokens_by_prompt[target],
            feature_table=feature_table,
            edit=edit,
            baseline=baselines[target],
            sender=senders[edit_id],
            baseline_records=baseline_records[target],
            sender_records=sender_records[edit_id],
            probe_direction=probe_direction,
            probe_bias=probe_bias,
        )
        finish_row("low_coverage_controls", result)
        print(f"low coverage {index}/{len(plan['low_coverage_controls'])}: {control['control_id']}", flush=True)

    completed = {row["result_id"] for row in output["joint_receiver_sets"]}
    for index, joint in enumerate(plan["joint_receiver_sets"], 1):
        if joint["cell_id"] in completed:
            continue
        source_cell = cells_by_id[joint["source_cell_id"]]
        edit_id = joint["source_edit_id"]
        edit = edits_by_id[edit_id]
        target = edit["target_prompt_key"]
        result = _joint_route(
            row=joint,
            source_cell=source_cell,
            model=model,
            tokens=tokens_by_prompt[target],
            feature_table=feature_table,
            edit=edit,
            baseline=baselines[target],
            sender=senders[edit_id],
            baseline_records=baseline_records[target],
            sender_records=sender_records[edit_id],
            probe_direction=probe_direction,
            probe_bias=probe_bias,
        )
        finish_row("joint_receiver_sets", result)
        print(f"joint {index}/{len(plan['joint_receiver_sets'])}: {joint['cell_id']}", flush=True)

    if output["forward_accounting"]["completed"] != plan["expected_forwards"]["total"]:
        raise ValueError("E7c final forward accounting differs")
    output["status"] = E7C_RESULT_STATUS
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic(args.output, output)
    validate_e7c_result(output, plan)
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": output["status"],
                "forward_accounting": output["forward_accounting"],
                "payload_sha256": output["payload_sha256"],
                "private_raw_tensor_sha256": output["private_raw_tensor_bundle"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
