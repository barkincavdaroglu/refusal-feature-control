#!/usr/bin/env python3
"""Run S9 E1 attribution patching and E2 exact-patch calibration on one GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from safetensors import safe_open
from transformer_lens.hook_points import HookPoint

from s9_contract import (
    ANALYSIS_SEMANTICS_VERSION,
    METRICS,
    REFUSAL_TOKEN_IDS,
    SCHEMA_VERSION,
    canonical_sha256,
    component_id,
    parse_feature,
    select_component_candidates,
    sha256_file,
    sha256_text,
    validate_plan,
    validate_result_shape,
)


TensorHook = tuple[str, Callable[..., torch.Tensor]]


class ExactHookedMLP(nn.Module):
    """Expose MLP input/output hooks without changing forward or backward math."""

    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        self.original_mlp = original_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.hook_out(self.original_mlp(self.hook_in(activations)))


class ExactHookedUnembed(nn.Module):
    """Expose the final normalized residual without changing logits or gradients."""

    def __init__(self, original_unembed: nn.Module):
        super().__init__()
        self.original_unembed = original_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.hook_post(self.original_unembed(self.hook_pre(activations)))


def sha256_python_tree(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*.py") if item.is_file())
    if not files:
        raise ValueError(f"No Python files below {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_prompt_rows(path: Path, plan: dict[str, Any]) -> dict[str, dict[str, str]]:
    all_rows = {row["prompt_key"]: row for row in read_csv(path)}
    expected = {row["prompt_key"]: row for row in plan["prompt_records"]}
    if any(key not in all_rows for key in expected):
        raise ValueError("One or more planned prompt keys are absent")
    selected = {}
    for key, record in expected.items():
        row = all_rows[key]
        if sha256_text(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"Prompt bytes do not match their hash for {key}")
        for field in ("intent_id", "risk_class", "frame_id", "frame_class", "prompt_sha256"):
            if row[field] != record[field]:
                raise ValueError(f"Prompt metadata mismatch for {key}.{field}")
        selected[key] = row
    return selected


def tokenize_chat(model, chat_text: str) -> torch.Tensor:
    """Tokenize an already-formatted chat without adding another special token."""

    encoded = model.tokenizer(
        chat_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    tokens = encoded.input_ids
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"Expected one tokenized sequence, got {tokens.shape}")
    return tokens.to(model.cfg.device)


def final_vector(activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim == 1:
        return activations
    if activations.ndim == 3:
        if activations.shape[0] != 1:
            raise ValueError("S9 calibration requires batch size one")
        return activations[0, -1]
    if activations.ndim == 2:
        return activations[-1]
    raise ValueError(f"Expected rank-two or rank-three activation, got {activations.shape}")


def replace_final_vector(activations: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    output = activations.clone()
    if output.ndim == 3:
        output[0, -1] = replacement
    elif output.ndim == 2:
        output[-1] = replacement
    else:
        raise ValueError(f"Expected rank-two or rank-three activation, got {output.shape}")
    return output


def add_final_vector(activations: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    output = activations.clone()
    if output.ndim == 3:
        output[0, -1] = output[0, -1] + delta
    elif output.ndim == 2:
        output[-1] = output[-1] + delta
    else:
        raise ValueError(f"Expected rank-two or rank-three activation, got {output.shape}")
    return output


def capture_final(storage: dict[str, torch.Tensor], name: str):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        storage[name] = final_vector(activations).detach().float().cpu()
        return activations

    return hook


def capture_graph_tensor(storage: dict[str, torch.Tensor], name: str):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        if not activations.requires_grad:
            raise RuntimeError(f"Gradient capture tensor does not require grad: {name}")
        storage[name] = activations
        return activations

    return hook


def seed_residual_gradient(activations: torch.Tensor, hook) -> torch.Tensor:
    del hook
    return activations.detach().requires_grad_(True)


def differentiable_metrics(
    logits: torch.Tensor,
    final_normalized: torch.Tensor,
    probe_direction: torch.Tensor,
    probe_bias: float,
) -> dict[str, torch.Tensor]:
    final = final_vector(final_normalized).float()
    final_logits = final_vector(logits).float()
    mask = torch.ones(final_logits.shape[-1], dtype=torch.bool, device=final_logits.device)
    mask[list(REFUSAL_TOKEN_IDS)] = False
    return {
        "probe_score": final @ probe_direction.float() + probe_bias,
        "next_token_refusal_log_odds": torch.logsumexp(
            final_logits[list(REFUSAL_TOKEN_IDS)], dim=-1
        )
        - torch.logsumexp(final_logits[mask], dim=-1),
    }


def detached_metric_values(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().float().cpu()) for name, value in metrics.items()}


def read_rows(path: str, key: str, indices: list[int], device: torch.device) -> torch.Tensor:
    rows = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"{key} missing from {path}")
        values = handle.get_slice(key)
        for index in indices:
            rows.append(values[index : index + 1])
    return torch.cat(rows).to(device=device, dtype=torch.bfloat16)


def build_hook_table(model, plan: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    component_hooks: dict[str, str] = {}
    for layer in plan["component_layers"]:
        component_hooks[component_id("mlp_output", layer)] = f"blocks.{layer}.mlp.hook_out"
        component_hooks[component_id("attention_output", layer)] = f"blocks.{layer}.hook_attn_out"
    missing = [name for name in component_hooks.values() if name not in model.hook_dict]
    if missing:
        raise KeyError(f"Required component hooks missing: {missing[:5]}")
    seed_hook = "blocks.0.hook_resid_pre"
    final_hook = "unembed.hook_pre"
    for name in (seed_hook, final_hook):
        if name not in model.hook_dict:
            raise KeyError(f"Required hook missing: {name}")
    metadata: dict[str, dict[str, Any]] = {}
    for node_id in component_hooks:
        kind, layer_text = node_id.split(":L")
        metadata[node_id] = {"node_id": node_id, "kind": kind, "layer": int(layer_text)}
    return component_hooks, metadata


def load_feature_table(model, plan: dict[str, Any], device: torch.device) -> dict[str, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in plan["candidate_features"]:
        grouped[int(record["layer"])].append(record)
    table: dict[str, dict[str, Any]] = {}
    for layer, records in sorted(grouped.items()):
        records = sorted(records, key=lambda row: int(row["feature_index"]))
        path = hf_hub_download(
            repo_id=plan["transcoder"]["id"],
            filename=f"layer_{layer}.safetensors",
            revision=plan["transcoder"]["revision"],
        )
        if plan["transcoder"]["revision"] not in str(path):
            raise ValueError(f"Unexpected transcoder snapshot at layer {layer}: {path}")
        indices = [int(row["feature_index"]) for row in records]
        encoder = read_rows(str(path), "W_enc", indices, device)
        decoder = read_rows(str(path), "W_dec", indices, device)
        bias = read_rows(str(path), "b_enc", indices, device)
        threshold = None
        activation_function = "ReLU"
        input_hook = f"blocks.{layer}.mlp.hook_in"
        output_hook = f"blocks.{layer}.mlp.hook_out"
        if input_hook not in model.hook_dict or output_hook not in model.hook_dict:
            raise KeyError(f"Feature hook missing at layer {layer}")
        for index, record in enumerate(records):
            table[record["feature"]] = {
                **record,
                "encoder": encoder[index],
                "decoder": decoder[index],
                "bias": bias[index],
                "threshold": None,
                "activation_function": activation_function,
                "input_hook": input_hook,
                "output_hook": output_hook,
                "component_node_id": component_id("mlp_output", layer),
            }
    return table


def feature_activation(input_vector: torch.Tensor, record: dict[str, Any]) -> float:
    vector = input_vector.to(device=record["encoder"].device, dtype=record["encoder"].dtype)
    pre = torch.dot(record["encoder"], vector) + record["bias"]
    if record["threshold"] is None:
        value = F.relu(pre)
    else:
        value = pre * (pre > record["threshold"])
    return float(value.detach().float().cpu())


@torch.inference_mode()
def cache_prompt(
    model,
    tokens: torch.Tensor,
    component_hooks: dict[str, str],
    feature_input_hooks: dict[int, str],
    probe_direction: torch.Tensor,
    probe_bias: float,
) -> dict[str, Any]:
    components: dict[str, torch.Tensor] = {}
    feature_inputs: dict[str, torch.Tensor] = {}
    final_storage: dict[str, torch.Tensor] = {}
    hooks: list[TensorHook] = [
        (hook_name, capture_final(components, node_id))
        for node_id, hook_name in component_hooks.items()
    ]
    hooks.extend(
        (hook_name, capture_final(feature_inputs, str(layer)))
        for layer, hook_name in feature_input_hooks.items()
    )
    hooks.append(("unembed.hook_pre", capture_final(final_storage, "final")))
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    final_normalized = final_storage["final"].to(device=logits.device)
    metrics = differentiable_metrics(logits, final_normalized, probe_direction, probe_bias)
    final_logits = final_vector(logits).detach().float().cpu()
    return {
        "components": components,
        "feature_inputs": {int(layer): value for layer, value in feature_inputs.items()},
        "metrics": detached_metric_values(metrics),
        "top_token_id": int(final_logits.argmax()),
        "final_logit_l2": float(final_logits.norm()),
    }


def target_gradients(
    model,
    tokens: torch.Tensor,
    component_hooks: dict[str, str],
    probe_direction: torch.Tensor,
    probe_bias: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    captured: dict[str, torch.Tensor] = {}
    final_storage: dict[str, torch.Tensor] = {}
    hooks: list[TensorHook] = [("blocks.0.hook_resid_pre", seed_residual_gradient)]
    hooks.extend(
        (hook_name, capture_graph_tensor(captured, node_id))
        for node_id, hook_name in component_hooks.items()
    )

    def capture_final_graph(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        final_storage["final"] = activations
        return activations

    hooks.append(("unembed.hook_pre", capture_final_graph))
    with torch.enable_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        metrics = differentiable_metrics(logits, final_storage["final"], probe_direction, probe_bias)
        ordered_ids = sorted(captured)
        tensors = tuple(captured[node_id] for node_id in ordered_ids)
        gradients: dict[str, dict[str, torch.Tensor]] = {node_id: {} for node_id in ordered_ids}
        for metric_index, metric_name in enumerate(METRICS):
            values = torch.autograd.grad(
                metrics[metric_name],
                tensors,
                retain_graph=metric_index < len(METRICS) - 1,
                allow_unused=False,
            )
            for node_id, value in zip(ordered_ids, values):
                gradients[node_id][metric_name] = final_vector(value).detach().float().cpu()
        metric_values = detached_metric_values(metrics)
    del logits, metrics, tensors, captured, final_storage
    return gradients, metric_values


@torch.inference_mode()
def run_metrics_only(
    model,
    tokens: torch.Tensor,
    probe_direction: torch.Tensor,
    probe_bias: float,
    intervention_hooks: list[TensorHook],
) -> tuple[dict[str, float], int]:
    final_storage: dict[str, torch.Tensor] = {}
    hooks = list(intervention_hooks)
    hooks.append(("unembed.hook_pre", capture_final(final_storage, "final")))
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    final_normalized = final_storage["final"].to(device=logits.device)
    metrics = detached_metric_values(
        differentiable_metrics(logits, final_normalized, probe_direction, probe_bias)
    )
    top_token_id = int(final_vector(logits).detach().argmax())
    return metrics, top_token_id


def component_patch_hook(
    target: torch.Tensor,
    source: torch.Tensor,
    dose: float,
    mismatch_storage: list[float],
):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        current = final_vector(activations)
        expected = target.to(device=current.device, dtype=current.dtype)
        mismatch = float((current.float() - expected.float()).abs().max().detach().cpu())
        mismatch_storage.append(mismatch)
        if mismatch > 1e-4:
            raise ValueError(f"Target component cache drift before patch: {mismatch}")
        replacement = expected + dose * (
            source.to(device=current.device, dtype=current.dtype) - expected
        )
        return replace_final_vector(activations, replacement)

    return hook


def feature_delta_hook(decoder: torch.Tensor, activation_delta: float, dose: float):
    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        delta = decoder.to(device=activations.device, dtype=activations.dtype) * (
            activation_delta * dose
        )
        return add_final_vector(activations, delta)

    return hook


def finite_effects(patched: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    effects = {metric: patched[metric] - baseline[metric] for metric in METRICS}
    if not all(math.isfinite(value) for value in effects.values()):
        raise ValueError("Exact patch produced non-finite effects")
    return effects


def execute_direction(
    *,
    model,
    plan: dict[str, Any],
    direction: dict[str, Any],
    tokens_by_prompt: dict[str, torch.Tensor],
    caches: dict[str, dict[str, Any]],
    component_hooks: dict[str, str],
    component_metadata: dict[str, dict[str, Any]],
    feature_table: dict[str, dict[str, Any]],
    feature_activations: dict[str, dict[str, float]],
    prompt_metadata: dict[str, dict[str, Any]],
    probe_direction: torch.Tensor,
    probe_bias: float,
) -> dict[str, Any]:
    started = time.monotonic()
    target_key = direction["target_prompt_key"]
    source_key = direction["source_prompt_key"]
    target_cache = caches[target_key]
    source_cache = caches[source_key]
    gradients, gradient_metrics = target_gradients(
        model,
        tokens_by_prompt[target_key],
        component_hooks,
        probe_direction,
        probe_bias,
    )
    for metric in METRICS:
        if not math.isclose(
            gradient_metrics[metric], target_cache["metrics"][metric], rel_tol=0.0, abs_tol=2e-5
        ):
            raise ValueError(f"Gradient and cached target metrics differ for {target_key}/{metric}")

    component_rows = []
    for node_id in sorted(component_hooks):
        delta = source_cache["components"][node_id] - target_cache["components"][node_id]
        atp = {
            metric: float(torch.dot(delta.float(), gradients[node_id][metric].float()))
            for metric in METRICS
        }
        component_rows.append(
            {
                **component_metadata[node_id],
                "activation_delta_l2": float(delta.norm()),
                "target_activation_l2": float(target_cache["components"][node_id].norm()),
                "gradient_l2": {
                    metric: float(gradients[node_id][metric].norm()) for metric in METRICS
                },
                "attribution_patch": atp,
            }
        )

    feature_rows = []
    for feature in sorted(feature_table, key=parse_feature):
        record = feature_table[feature]
        activation_delta = feature_activations[source_key][feature] - feature_activations[target_key][feature]
        output_gradient = gradients[record["component_node_id"]]
        decoder_cpu = record["decoder"].detach().float().cpu()
        atp = {
            metric: float(activation_delta * torch.dot(decoder_cpu, output_gradient[metric]))
            for metric in METRICS
        }
        feature_rows.append(
            {
                "feature": feature,
                "layer": int(record["layer"]),
                "feature_index": int(record["feature_index"]),
                "roles": record["roles"],
                "activation_function": record["activation_function"],
                "target_activation": feature_activations[target_key][feature],
                "source_activation": feature_activations[source_key][feature],
                "activation_delta": activation_delta,
                "decoder_delta_l2": abs(activation_delta) * float(decoder_cpu.norm()),
                "attribution_patch": atp,
            }
        )

    selection_cfg = plan["exact_patch_selection"]
    selected_components = select_component_candidates(
        component_rows,
        top_per_metric=selection_cfg["top_per_metric"],
        low_both=selection_cfg["low_both"],
        disagreements=selection_cfg["disagreements"],
    )
    component_atp = {row["node_id"]: row["attribution_patch"] for row in component_rows}
    exact_patches = []
    for selected in selected_components:
        node_id = selected["node_id"]
        for dose in selection_cfg["doses"]:
            mismatch: list[float] = []
            patched, top_token_id = run_metrics_only(
                model,
                tokens_by_prompt[target_key],
                probe_direction,
                probe_bias,
                [
                    (
                        component_hooks[node_id],
                        component_patch_hook(
                            target_cache["components"][node_id],
                            source_cache["components"][node_id],
                            float(dose),
                            mismatch,
                        ),
                    )
                ],
            )
            if len(mismatch) != 1:
                raise RuntimeError(f"Component hook did not fire exactly once for {node_id}")
            exact_patches.append(
                {
                    "entity_type": "model_component",
                    "entity_id": node_id,
                    "kind": selected["kind"],
                    "layer": selected["layer"],
                    "selection_reasons": selected["reasons"],
                    "dose": float(dose),
                    "cache_mismatch_max_abs": mismatch[0],
                    "attribution_patch_prediction": {
                        metric: float(dose) * component_atp[node_id][metric] for metric in METRICS
                    },
                    "patched_metrics": patched,
                    "exact_effect": finite_effects(patched, target_cache["metrics"]),
                    "top_token_id": top_token_id,
                }
            )

    feature_atp = {row["feature"]: row for row in feature_rows}
    for feature in sorted(feature_table, key=parse_feature):
        record = feature_table[feature]
        activation_delta = feature_activations[source_key][feature] - feature_activations[target_key][feature]
        for dose in selection_cfg["doses"]:
            patched, top_token_id = run_metrics_only(
                model,
                tokens_by_prompt[target_key],
                probe_direction,
                probe_bias,
                [
                    (
                        record["output_hook"],
                        feature_delta_hook(record["decoder"], activation_delta, float(dose)),
                    )
                ],
            )
            exact_patches.append(
                {
                    "entity_type": "transcoder_feature_direction",
                    "entity_id": feature,
                    "kind": "feature_decoder_contribution",
                    "layer": int(record["layer"]),
                    "selection_reasons": record["roles"],
                    "dose": float(dose),
                    "attribution_patch_prediction": {
                        metric: float(dose) * feature_atp[feature]["attribution_patch"][metric]
                        for metric in METRICS
                    },
                    "patched_metrics": patched,
                    "exact_effect": finite_effects(patched, target_cache["metrics"]),
                    "top_token_id": top_token_id,
                }
            )

    del gradients
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **direction,
        "target": prompt_metadata[target_key],
        "source": prompt_metadata[source_key],
        "target_baseline_metrics": target_cache["metrics"],
        "source_baseline_metrics": source_cache["metrics"],
        "component_attribution": component_rows,
        "feature_attribution": feature_rows,
        "selected_components": selected_components,
        "exact_patches": exact_patches,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--probe-direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for S9 E1/E2 calibration")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    expected_inputs = {
        "prompts": args.prompts,
        "annotations": args.annotations,
        "probe_direction": args.probe_direction,
        "runner": Path(__file__).resolve(),
        "contract": Path(__file__).resolve().with_name("s9_contract.py"),
    }
    for name, path in expected_inputs.items():
        expected = plan["inputs"][name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"{name} hash mismatch")
    for package, expected in plan["runtime_contract"]["package_versions"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"Runtime package mismatch for {package}: {actual} != {expected}")
    device_name = torch.cuda.get_device_name(0)
    if not any(accelerator in device_name for accelerator in plan["runtime_contract"]["preferred_accelerators"]):
        raise ValueError(f"Unsupported CUDA device for this plan: {device_name}")
    total_memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_memory_gib < plan["runtime_contract"]["minimum_cuda_memory_gib"]:
        raise ValueError(f"GPU memory {total_memory_gib:.2f} GiB is below the plan minimum")

    probe = np.load(args.probe_direction)
    direction_np = probe["direction_unit"].astype(np.float32)
    probe_direction = torch.tensor(direction_np, device="cuda", dtype=torch.float32)
    probe_bias = float(probe["bias_unit"])
    prompt_rows = load_prompt_rows(args.prompts, plan)

    import circuit_tracer
    from transformer_lens import HookedTransformer

    tree_sha = sha256_python_tree(Path(circuit_tracer.__file__).resolve().parent)
    if tree_sha != plan["circuit_tracer"]["python_tree_sha256"]:
        raise ValueError(f"Circuit Tracer source tree mismatch: {tree_sha}")

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    device = torch.device("cuda")
    print(f"Loading plain TransformerLens {plan['model']['id']} on {device_name}", flush=True)
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
        block.mlp = ExactHookedMLP(block.mlp)
    model.unembed = ExactHookedUnembed(model.unembed)
    model.setup()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    component_hooks, component_metadata = build_hook_table(model, plan)
    feature_table = load_feature_table(model, plan, device)
    feature_input_hooks = {
        int(record["layer"]): record["input_hook"] for record in feature_table.values()
    }
    prompt_metadata = {row["prompt_key"]: dict(row) for row in plan["prompt_records"]}
    tokens_by_prompt: dict[str, torch.Tensor] = {}
    caches: dict[str, dict[str, Any]] = {}
    feature_activations: dict[str, dict[str, float]] = {}
    for prompt_key in sorted(prompt_rows):
        chat_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_rows[prompt_key]["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenize_chat(model, chat_text)
        tokens_by_prompt[prompt_key] = tokens
        cache = cache_prompt(
            model,
            tokens,
            component_hooks,
            feature_input_hooks,
            probe_direction,
            probe_bias,
        )
        caches[prompt_key] = cache
        feature_activations[prompt_key] = {
            feature: feature_activation(cache["feature_inputs"][int(record["layer"])], record)
            for feature, record in feature_table.items()
        }
        prompt_metadata[prompt_key]["chat_text_sha256"] = sha256_text(chat_text)
        prompt_metadata[prompt_key]["input_token_count"] = int(tokens.shape[-1])
        prompt_metadata[prompt_key]["baseline_metrics"] = cache["metrics"]
        prompt_metadata[prompt_key]["top_token_id"] = cache["top_token_id"]
        print(f"cached {prompt_key}", flush=True)

    completed_rows: list[dict[str, Any]] = []
    if args.output.exists():
        if not args.resume:
            raise ValueError(f"Output exists; use --resume: {args.output}")
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        validate_result_shape(prior, plan)
        completed_rows = prior["directions"]
    completed = {row["direction_id"] for row in completed_rows}
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device_name,
        "cuda_total_memory_gib": total_memory_gib,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "transformer-lens", "circuit-tracer", "safetensors")
        },
        "circuit_tracer_python_tree_sha256": tree_sha,
    }
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_semantics_version": ANALYSIS_SEMANTICS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": "adaptive_exploration",
        "objective": plan["objective"],
        "plan_sha256": sha256_file(args.plan),
        "plan_payload_sha256": plan["payload_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "runtime": runtime,
        "model": plan["model"],
        "transcoder": plan["transcoder"],
        "method": {
            "attribution_patch": "first_order_source_minus_target_dot_target_gradient",
            "gradient_semantics": "plain_transformerlens_exact_original_model_backward",
            "atp_star": False,
            "component_exact_patch": "interpolated final-position source component replacement",
            "feature_exact_patch": "interpolated source-minus-target feature decoder contribution",
            "edge_or_path_patch": False,
        },
        "privacy": plan["privacy"],
        "prompt_baselines": [prompt_metadata[key] for key in sorted(prompt_metadata)],
        "directions": completed_rows,
    }
    for direction in plan["directions"]:
        if direction["direction_id"] in completed:
            print(f"resume skip {direction['direction_id']}", flush=True)
            continue
        row = execute_direction(
            model=model,
            plan=plan,
            direction=direction,
            tokens_by_prompt=tokens_by_prompt,
            caches=caches,
            component_hooks=component_hooks,
            component_metadata=component_metadata,
            feature_table=feature_table,
            feature_activations=feature_activations,
            prompt_metadata=prompt_metadata,
            probe_direction=probe_direction,
            probe_bias=probe_bias,
        )
        output["directions"].append(row)
        completed.add(direction["direction_id"])
        atomic_write_json(args.output, output)
        print(
            f"direction {len(output['directions'])}/{len(plan['directions'])}: "
            f"{direction['direction_id']} ({row['elapsed_seconds']:.1f}s)",
            flush=True,
        )
    output["status"] = "complete"
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output["direction_count"] = len(output["directions"])
    atomic_write_json(args.output, output)
    validate_result_shape(output, plan)
    print(json.dumps({"status": "complete", "directions": len(output["directions"]), "payload_sha256": output["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
