#!/usr/bin/env python3
"""Freeze E7b v1's exact execution plan before any E7 model outcome."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from s9_contract import METRICS, canonical_sha256, parse_feature, sha256_file
from s9_e7_contract import (
    E7B_PLAN_SEMANTICS_VERSION,
    E7B_PLAN_STATUS,
    E7_INPUT_NAMES,
    E7_PRIVACY,
    E7_SCHEMA_VERSION,
    validate_e7b_manifest,
    validate_e7b_plan,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def _input(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.name, "sha256": sha256_file(path)}


def _candidate_features(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    roles: dict[str, set[str]] = defaultdict(set)
    for row in [*manifest["mechanics_cells"], *manifest["topology_cells"]]:
        roles[row["source_feature"]].add("sender_edit_decoder")
        receiver = row.get("nominated_receiver_feature")
        if receiver is not None:
            roles[receiver].add("receiver_topology_marker_diagnostic")
    output = []
    for feature in sorted(roles, key=parse_feature):
        layer, index = parse_feature(feature)
        output.append(
            {
                "feature": feature,
                "layer": layer,
                "feature_index": index,
                "roles": sorted(roles[feature]),
            }
        )
    return output


def build_plan(
    *,
    manifest_path: Path,
    semantic_inputs: Mapping[str, Path],
) -> dict[str, Any]:
    if set(semantic_inputs) != set(E7_INPUT_NAMES):
        raise ValueError("E7b builder semantic-input mapping differs")
    if semantic_inputs["candidate_manifest"].resolve() != manifest_path.resolve():
        raise ValueError("E7b candidate-manifest path differs")
    manifest = _read(manifest_path)
    validate_e7b_manifest(manifest)
    features = _candidate_features(manifest)
    primary_cells = [*manifest["mechanics_cells"], *manifest["topology_cells"]]
    source_edits = sorted(
        (
            {
                "source_edit_id": row["source_edit_id"],
                "direction_id": row["direction_id"],
                "target_prompt_key": row["target_prompt_key"],
                "source_prompt_key": row["source_prompt_key"],
                "source_feature": row["source_feature"],
                "source_edit_policy": row["source_edit_policy"],
                "source_activation_delta": row["source_activation_delta"],
                "source_edit_l2": row["source_edit_l2"],
            }
            for row in primary_cells
        ),
        key=lambda row: row["source_edit_id"],
    )
    if len(source_edits) != len({row["source_edit_id"] for row in source_edits}):
        raise ValueError("E7b source-edit closure contains aliases")
    payload = {
        "schema_version": E7_SCHEMA_VERSION,
        "analysis_semantics_version": E7B_PLAN_SEMANTICS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": E7B_PLAN_STATUS,
        "mode": "adaptive_exploration",
        "research_stage": "understanding_testing_hypotheses",
        "objective": (
            "measure exact original-model dependence on complete receiver MLP "
            "outputs, with MLP-input calibration and controls, before any E7 outcome"
        ),
        "authorization": {
            "explicit_user_confirmation_received": True,
            "gpu_allocation_allowed": True,
            "confirmation_received_after_e7a_and_e7b_v0_1": True,
            "e7d_allowed": False,
        },
        "selection_used_e7_outcomes": False,
        "candidate_manifest_payload_sha256": manifest["payload_sha256"],
        "candidate_manifest_file_sha256": sha256_file(manifest_path),
        "model": manifest["model"],
        "transcoder": manifest["transcoder"],
        "runtime_contract": manifest["runtime_contract"],
        "random_seed": 90701,
        "prompt_records": manifest["prompt_records"],
        "candidate_features": features,
        "source_edits": source_edits,
        "mechanics_cells": manifest["mechanics_cells"],
        "topology_cells": manifest["topology_cells"],
        "topology_contrasts": manifest["topology_contrasts"],
        "input_calibration_cell_ids": manifest["input_calibration_cell_ids"],
        "low_coverage_controls": manifest["low_coverage_controls"],
        "joint_receiver_sets": manifest["joint_receiver_sets"],
        "expected_counts": manifest["expected_counts"],
        "expected_forwards": manifest["expected_forwards"],
        "endpoints": {
            "primary": list(METRICS),
            "same_forward_diagnostics": [
                "top_token_id",
                "final_logit_l2",
                "top1_minus_top2_logit_margin",
                "refusal_top_minus_nonrefusal_top_logit_margin",
            ],
            "s10_four_view_score": {
                "included": False,
                "reason": (
                    "the S10 target is not a same-forward endpoint and failed two "
                    "portability clauses; adding it would change timing and compute"
                ),
            },
            "generation_executed": False,
        },
        "exact_method": {
            "sender_edit": (
                "add source_activation_delta times the pinned transcoder decoder to "
                "the final-position source MLP output"
            ),
            "primary_receiver_boundary": "blocks.L.mlp.hook_out",
            "calibration_receiver_boundary": "blocks.L.mlp.hook_in",
            "receiver_state_contract": "complete final-position d_model vector",
            "receiver_feature_intervened": False,
            "receiver_feature_role": "topology marker and gate diagnostic only",
            "restoration": "sender edit plus replacement by target-baseline receiver state",
            "rescue": "target baseline plus replacement by sender-world receiver state",
            "joint_receiver_order": "ascending causal layer order",
            "position": "final prompt token",
            "dose": 1.0,
            "generation": False,
        },
        "estimands": {
            "T_S": "YS - Y0",
            "D_R": "YS - YS_R0",
            "B_R": "YS_R0 - Y0",
            "Q_R": "YR_S - Y0",
            "invariant": "T_S = D_R + B_R",
            "ratios_primary": False,
        },
        "execution_contract": {
            "resume_after_each_route_row": True,
            "cache_prompt_baselines": True,
            "cache_unique_sender_worlds": True,
            "exact_planned_forwards": 223,
            "audit_ceiling": 246,
            "no_backward_pass": True,
            "no_generation": True,
            "public_result_excludes_raw_tensors": True,
            "private_safetensors_bundle_required": True,
        },
        "analysis_contract": {
            "development_only": True,
            "generalization_unit": "independent intent, not route cell",
            "primary_outputs": ["T_S", "D_R", "B_R", "Q_R"],
            "single_receiver_does_not_establish_redundancy": True,
            "joint_ordered_receivers_are_receiver_set_dependence": True,
            "weak_dependence_is_not_automatically_bypass": True,
            "e7d_incremental_topology_claim_allowed": False,
        },
        "inputs": {name: _input(path) for name, path in sorted(semantic_inputs.items())},
        "privacy": E7_PRIVACY,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    validate_e7b_plan(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    for name in sorted(E7_INPUT_NAMES):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    semantic = {name: getattr(args, name) for name in E7_INPUT_NAMES}
    payload = build_plan(manifest_path=args.manifest, semantic_inputs=semantic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_features": len(payload["candidate_features"]),
                "expected_counts": payload["expected_counts"],
                "expected_forwards": payload["expected_forwards"],
                "payload_sha256": payload["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
