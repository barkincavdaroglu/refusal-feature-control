"""Fail-closed contracts for S9 E9c's CPU-only runner freeze.

E9c implements and hash-closes the future fixed-dose route-selectivity runner.
It does not authorize external transfer, tokenizer/model loading, a model
forward, Colab, GPU allocation, generation, backward computation, or E7d.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from s9_contract import canonical_sha256, sha256_file, validate_stamped_payload
from s9_e9b_contract import validate_e9b_private_panel, validate_e9b_public_manifest


E9C_SCHEMA_VERSION = 1
E9C_SOURCE_BUNDLE_SEMANTICS_VERSION = "s9-e9c-minimal-source-decoder-bundle-v1"
E9C_SOURCE_BUNDLE_STATUS = "complete_authenticated_minimal_e9c_source_bundle_v1"
E9C_PLAN_SEMANTICS_VERSION = "s9-e9c-cpu-runner-execution-plan-v3"
E9C_PLAN_STATUS = "frozen_cpu_only_runner_plan_after_hook_portability_erratum_v3"
E9C_RESULT_SEMANTICS_VERSION = "s9-e9c-fixed-dose-route-result-v2"
E9C_RUNNING_STATUS = "running_fixed_dose_route_selectivity_v2"
E9C_RESULT_STATUS = "complete_fixed_dose_route_selectivity_v2"
E9C_ANALYSIS_SEMANTICS_VERSION = "s9-e9c-preregistered-route-selectivity-analysis-v1"
E9C_ANALYSIS_STATUS = "complete_preregistered_route_selectivity_analysis_v1"

E9A_DESIGN_FILE_SHA256 = (
    "dfb1f1ad5de297997301a74022d38d27421acc772426d4314eebc9996a777788"
)
E9A_DESIGN_PAYLOAD_SHA256 = (
    "0f3c01580c8a8837fe5a1819ea63c041ec5829f5acda24571f5ff6132aaf70f8"
)
E9B_PRIVATE_PANEL_FILE_SHA256 = (
    "d003b98d170721fedbc58f0dd5f225f7e00261a4d7daeb2ab6ae9803572202fa"
)
E9B_PRIVATE_PANEL_PAYLOAD_SHA256 = (
    "da3657aba93bafe307d960a5c6c8415185c3925cd8a744dcf281fb082e298356"
)
E9B_PUBLIC_MANIFEST_FILE_SHA256 = (
    "aeace06bdfdf637fce6e9d6190eb40fffeec28a16d964dd57ad6e0f8b1f9943b"
)
E9B_PUBLIC_MANIFEST_PAYLOAD_SHA256 = (
    "a0aad6c7904ff08e05326ffb808c8d2fd8c3d468e927de8285df4dc5596e4f97"
)
E8B_SOURCE_BUNDLE_FILE_SHA256 = (
    "8a2de5629bcdbd138d23b9ed9705b6c4a2ac78d9995912ded0c820a0b0f23ffc"
)
E8B_SOURCE_MANIFEST_FILE_SHA256 = (
    "c19d0722826a475fced422f73a87f9173948c43cc26d42a3751c34ecf337ef20"
)
E8B_SOURCE_MANIFEST_PAYLOAD_SHA256 = (
    "de8be0a4f475c477d5bcdbb90c2edfcf921d75cf470ea89758d7250f87b5907e"
)

E9C_MODEL = {
    "id": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
}
E9C_SOURCE_FEATURES = (
    ("L31:F49541", 49541, 0, "nominated_c"),
    ("L31:F139261", 139261, 1, "equal_l2_control"),
)
E9C_SOURCE_BUNDLE_KEYS = ("decoder_rows", "feature_indices")
E9C_ARMS = ("nominated_c", "equal_l2_control")
E9C_POSITIONS = ("final_prompt_position", "penultimate_prompt_position")
E9C_FIXED_C_ACTIVATION_DELTA = 18.625
E9C_REQUIRED_PROMPTS = 24
E9C_REQUIRED_SENDER_WORLDS = 48
E9C_REQUIRED_ROUTE_CELLS = 96
E9C_PLANNED_SCIENTIFIC_FORWARDS = 264
E9C_AUDIT_FORWARD_CEILING = 291

E9C_AUTHORIZATION_REFERENCE = (
    "user-message:apply-S-20260824T044238670002Z-af880de116-and-proceed-with-e9c"
)
E9C_AUTHORIZATION = {
    "explicit_user_confirmation_received": True,
    "confirmation_reference": E9C_AUTHORIZATION_REFERENCE,
    "scope": "implement_cpu_only_e9c_runner_and_hash_pinned_execution_plan_only",
    "local_private_prompt_read_allowed": True,
    "runner_implementation_allowed": True,
    "execution_plan_implementation_allowed": True,
    "source_decoder_bundle_implementation_allowed": True,
    "execution_allowed": False,
    "private_prompt_upload_allowed": False,
    "tokenizer_load_allowed": False,
    "target_model_load_allowed": False,
    "model_forward_allowed": False,
    "colab_connection_allowed": False,
    "gpu_allocation_allowed": False,
    "generation_allowed": False,
    "backward_pass_allowed": False,
    "e7d_allowed": False,
    "outcome_based_panel_replacement_allowed": False,
    "fresh_user_confirmation_required_for_e9d": True,
}

E9C_INPUT_NAMES = (
    "e9a_design",
    "e9a_report",
    "e9b_private_panel",
    "e9b_public_manifest",
    "e9b_report",
    "decision_d006",
    "question_q002",
    "probe_direction",
    "source_decoder_bundle",
    "source_decoder_manifest",
    "source_bundle_builder",
    "runner",
    "analyzer",
    "runtime_installer",
    "base_runner",
    "base_contract",
    "e9_contract",
    "e9b_contract",
    "e9c_contract",
    "execution_plan_builder",
)

E9C_PRIVACY = {
    "private_prompt_text_in_public_artifacts": False,
    "core_request_text_in_public_artifacts": False,
    "completion_text_in_public_artifacts": False,
    "token_text_or_ids_in_public_artifacts": False,
    "raw_receiver_tensors_in_public_artifacts": False,
    "input_token_tensor_hashes_in_public_result": True,
    "private_raw_tensor_bundle_required_for_execution": True,
}

E9C_RUNTIME_CONTRACT = {
    "status": "future_runtime_specification_not_execution_authority",
    "preferred_accelerators": ["A100"],
    "minimum_cuda_memory_gib": 39.0,
    "package_versions": {
        "torch": "2.11.0+cu128",
        "transformers": "4.57.3",
        "transformer-lens": "3.2.1",
        "circuit-tracer": "0.5.2",
        "safetensors": "0.8.0",
    },
    "dtype": "bfloat16",
    "random_seed": 90901,
    "determinism": {
        "model_eval": True,
        "gradient_tracking": False,
        "parameters_require_grad": False,
        "torch_manual_seed": 90901,
        "cuda_manual_seed_all": 90901,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    },
    "model_load": {
        "fold_ln": False,
        "center_writing_weights": False,
        "center_unembed": False,
    },
    "chat_template": {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": False,
    },
    "uploaded_hashes_verified_before_model_load": True,
    "model_snapshot_authenticated_before_first_forward": True,
    "cross_resume_runtime_fingerprint_required": True,
}

E9C_NUMERICAL_CONTRACT = {
    "source_decoder_storage_dtype": "bfloat16",
    "nominal_numeric_dtype": "float32",
    "applied_edit_dtype": "bfloat16",
    "fixed_c_activation_delta": E9C_FIXED_C_ACTIVATION_DELTA,
    "fixed_positive_dose_for_every_prompt": True,
    "prompt_activation_screen_used": False,
    "nominal_equal_l2_relative_tolerance": 1e-6,
    "applied_edit_identity_contract": "exact_bfloat16_tensor_sha256",
    "cross_platform_float_summary_relative_tolerance": 1e-6,
    "invariant_absolute_tolerance": 1e-7,
    "pre_replacement_max_abs_mismatch": 1e-4,
    "penultimate_receiver_state_max_abs_change": 1e-4,
    "penultimate_endpoint_abs_tolerance": 1e-4,
    "threshold_failure_action": "retain_complete_result_and_mark_not_evaluable",
}

E9C_FORWARD_CONTRACT = {
    "target_prompt_baselines": 24,
    "sender_worlds": 48,
    "receiver_restoration_and_rescue_worlds": 192,
    "planned_scientific": E9C_PLANNED_SCIENTIFIC_FORWARDS,
    "audit_ceiling": E9C_AUDIT_FORWARD_CEILING,
    "maximum_audit_forwards": 27,
    "scientific_forwards_must_equal_plan": True,
    "cumulative_across_resumes": True,
    "counter_reset_on_resume_allowed": False,
    "new_scientific_cells_from_audit_budget_allowed": False,
}

E9C_ANALYSIS_CONTRACT = {
    "primary_endpoint": "next_token_refusal_log_odds",
    "secondary_endpoint": "probe_score",
    "harmful_independent_unit": "intent_id",
    "selectivity_independent_unit": "topic_pair_id",
    "strict_win_threshold": 0.0,
    "harmful_dependence_minimum_wins": 10,
    "harmful_rescue_minimum_wins": 10,
    "risk_differential_minimum_positive_paired_D": 10,
    "risk_differential_minimum_positive_paired_Q": 10,
    "benign_relative_smallness_maximum_ratio": 0.25,
    "benign_relative_smallness_minimum_wins": 10,
    "denominator": 12,
    "ties_are_nonwins": True,
    "all_frozen_intents_remain_in_denominators": True,
    "route_cells_are_independent_samples": False,
    "exact_one_sided_sign_test_reported": True,
    "behavior_claim_allowed": False,
    "benign_behavior_sparing_claim_allowed": False,
    "threshold_refit_allowed": False,
    "structural_audit_must_pass_before_scientific_analysis": True,
}

E9C_OUTPUT_CONTRACT = {
    "required_prompt_records": E9C_REQUIRED_PROMPTS,
    "required_sender_worlds": E9C_REQUIRED_SENDER_WORLDS,
    "required_route_cells": E9C_REQUIRED_ROUTE_CELLS,
    "public_result_required_after_authorized_execution": True,
    "private_raw_tensor_bundle_required_after_authorized_execution": True,
    "resumable_checkpoint_required": True,
    "analysis_implemented_before_outcomes": True,
    "public_prompt_or_completion_text_allowed": False,
    "public_raw_receiver_tensors_allowed": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def tensor_identity_sha256(tensor: torch.Tensor) -> str:
    materialized = tensor.detach().contiguous().cpu()
    identity = {
        "dtype": str(materialized.dtype).removeprefix("torch."),
        "shape": list(materialized.shape),
        "bytes_sha256": __import__("hashlib").sha256(
            materialized.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    return canonical_sha256(identity)


def expected_source_features() -> list[dict[str, Any]]:
    return [
        {
            "feature": feature,
            "feature_index": feature_index,
            "row_index": row_index,
            "role": role,
        }
        for feature, feature_index, row_index, role in E9C_SOURCE_FEATURES
    ]


def validate_e9c_source_bundle_manifest(
    payload: Mapping[str, Any],
    *,
    e9a_design: Mapping[str, Any],
    bundle_path: Path | None = None,
) -> None:
    validate_stamped_payload(payload)
    validate_stamped_payload(e9a_design)
    if e9a_design.get("payload_sha256") != E9A_DESIGN_PAYLOAD_SHA256:
        raise ValueError("E9c source bundle references changed E9a science")
    if payload.get("schema_version") != E9C_SCHEMA_VERSION:
        raise ValueError("E9c source bundle schema differs")
    if payload.get("analysis_semantics_version") != E9C_SOURCE_BUNDLE_SEMANTICS_VERSION:
        raise ValueError("E9c source bundle semantics differ")
    if payload.get("status") != E9C_SOURCE_BUNDLE_STATUS:
        raise ValueError("E9c source bundle status differs")
    if payload.get("features") != expected_source_features():
        raise ValueError("E9c source feature closure differs")
    builder = Path(__file__).with_name("build_s9_e9c_source_bundle.py")
    if payload.get("producer") != {
        "path": builder.name,
        "file_sha256": sha256_file(builder),
    }:
        raise ValueError("E9c source-bundle producer bytes differ")
    if payload.get("source") != {
        "bundle_file_sha256": E8B_SOURCE_BUNDLE_FILE_SHA256,
        "manifest_file_sha256": E8B_SOURCE_MANIFEST_FILE_SHA256,
        "manifest_payload_sha256": E8B_SOURCE_MANIFEST_PAYLOAD_SHA256,
        "selected_layer": 31,
        "selected_tensor": "W_dec",
        "unused_e8b_rows_copied": False,
    }:
        raise ValueError("E9c source-bundle provenance differs")
    if payload.get("e9a_design") != {
        "file_sha256": E9A_DESIGN_FILE_SHA256,
        "payload_sha256": E9A_DESIGN_PAYLOAD_SHA256,
    }:
        raise ValueError("E9c source-bundle E9a identity differs")
    if payload.get("privacy") != {
        "raw_decoder_values_in_public_manifest": False,
        "private_bundle_external_transfer_allowed": False,
        "target_model_contact": False,
    }:
        raise ValueError("E9c source-bundle privacy boundary differs")
    bundle = payload.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("E9c source-bundle record is missing")
    if (
        bundle.get("path") != "s9_e9c_source_decoders.safetensors"
        or bundle.get("keys") != list(E9C_SOURCE_BUNDLE_KEYS)
        or bundle.get("public") is not False
    ):
        raise ValueError("E9c source-bundle key/privacy closure differs")
    tensor_hashes = bundle.get("tensor_identity_sha256")
    if not isinstance(tensor_hashes, Mapping) or set(tensor_hashes) != set(
        E9C_SOURCE_BUNDLE_KEYS
    ) or not all(_is_sha256(value) for value in tensor_hashes.values()):
        raise ValueError("E9c source-bundle tensor hashes differ")
    if bundle_path is None:
        return
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file() or sha256_file(bundle_path) != bundle.get(
        "file_sha256"
    ):
        raise ValueError("E9c source-decoder bundle file hash differs")
    with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(E9C_SOURCE_BUNDLE_KEYS):
            raise ValueError("E9c source-decoder bundle contains extra tensors")
        indices = handle.get_tensor("feature_indices")
        decoders = handle.get_tensor("decoder_rows")
    if indices.dtype != torch.int64 or indices.tolist() != [49541, 139261]:
        raise ValueError("E9c source-decoder feature indices differ")
    if decoders.dtype != torch.bfloat16 or list(decoders.shape) != [2, 2560]:
        raise ValueError("E9c source-decoder tensor semantics differ")
    observed = {
        "feature_indices": tensor_identity_sha256(indices),
        "decoder_rows": tensor_identity_sha256(decoders),
    }
    if observed != dict(tensor_hashes):
        raise ValueError("E9c source-decoder tensor identity differs")


def expected_prompt_records(public_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in public_manifest["topic_pairs"]:
        for risk in ("harmful", "benign"):
            record = pair[risk]
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "topic_category": pair["topic_category"],
                    "intent_id": record["intent_id"],
                    "intent_category": record["intent_category"],
                    "risk_class": risk,
                    "prompt_key": record["prompt_key"],
                    "frame_id": record["frame_id"],
                    "core_sha256": record["core_sha256"],
                    "prompt_sha256": record["prompt_sha256"],
                }
            )
    return sorted(rows, key=lambda row: row["intent_id"])


def expected_sender_worlds(public_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_by_arm = {
        "nominated_c": "L31:F49541",
        "equal_l2_control": "L31:F139261",
    }
    for prompt in expected_prompt_records(public_manifest):
        for arm in E9C_ARMS:
            rows.append(
                {
                    "sender_world_id": f"e9c:{prompt['intent_id']}:{arm}:sender",
                    "fixed_dose_id": "e9c:fixed-dose:v1",
                    "pair_id": prompt["pair_id"],
                    "intent_id": prompt["intent_id"],
                    "risk_class": prompt["risk_class"],
                    "target_prompt_key": prompt["prompt_key"],
                    "arm": arm,
                    "source_feature": feature_by_arm[arm],
                }
            )
    return sorted(rows, key=lambda row: row["sender_world_id"])


def expected_route_cells(public_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_by_arm = {
        "nominated_c": "L31:F49541",
        "equal_l2_control": "L31:F139261",
    }
    for prompt in expected_prompt_records(public_manifest):
        for arm in E9C_ARMS:
            for position in E9C_POSITIONS:
                final = position == "final_prompt_position"
                rows.append(
                    {
                        "cell_id": f"e9c:{prompt['intent_id']}:{arm}:{position}",
                        "sender_world_id": f"e9c:{prompt['intent_id']}:{arm}:sender",
                        "fixed_dose_id": "e9c:fixed-dose:v1",
                        "pair_id": prompt["pair_id"],
                        "intent_id": prompt["intent_id"],
                        "risk_class": prompt["risk_class"],
                        "target_prompt_key": prompt["prompt_key"],
                        "arm": arm,
                        "source_feature": feature_by_arm[arm],
                        "source_hook": "blocks.31.mlp.hook_out",
                        "source_token_position": "final_prompt_position",
                        "receiver_layer": 32,
                        "receiver_boundary": "module_output",
                        "receiver_hook": "blocks.32.mlp.hook_out",
                        "receiver_token_position": position,
                        "receiver_position_role": (
                            "empirical_receiver"
                            if final
                            else "same_module_wrong_time_structural_audit"
                        ),
                        "included_in_empirical_analysis": final,
                    }
                )
    return sorted(rows, key=lambda row: row["cell_id"])


def _finite_nonnegative(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"E9c {label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"E9c {label} is not finite/nonnegative")
    return result


def validate_fixed_dose(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("E9c fixed-dose record is missing")
    required = {
        "fixed_dose_id",
        "nominated_source_feature",
        "control_source_feature",
        "nominated_activation_delta",
        "control_activation_delta",
        "nominal_nominated_edit_l2",
        "nominal_control_edit_l2",
        "nominal_equal_l2_relative_error",
        "realized_bfloat16_nominated_edit_l2",
        "realized_bfloat16_control_edit_l2",
        "nominated_edit_tensor_sha256",
        "control_edit_tensor_sha256",
        "prompt_activation_screen_used",
        "same_fixed_dose_for_all_24_prompts",
    }
    if set(value) != required:
        raise ValueError("E9c fixed-dose fields differ")
    if value.get("fixed_dose_id") != "e9c:fixed-dose:v1":
        raise ValueError("E9c fixed-dose identity differs")
    if value.get("nominated_source_feature") != "L31:F49541" or value.get(
        "control_source_feature"
    ) != "L31:F139261":
        raise ValueError("E9c fixed-dose feature identity differs")
    if value.get("nominated_activation_delta") != E9C_FIXED_C_ACTIVATION_DELTA:
        raise ValueError("E9c nominated fixed dose differs")
    if value.get("prompt_activation_screen_used") is not False or value.get(
        "same_fixed_dose_for_all_24_prompts"
    ) is not True:
        raise ValueError("E9c fixed-dose freshness semantics differ")
    control = _finite_nonnegative(value.get("control_activation_delta"), "control dose")
    if control <= 0:
        raise ValueError("E9c control dose must be positive")
    nominated_l2 = _finite_nonnegative(
        value.get("nominal_nominated_edit_l2"), "nominated L2"
    )
    control_l2 = _finite_nonnegative(value.get("nominal_control_edit_l2"), "control L2")
    relative = _finite_nonnegative(
        value.get("nominal_equal_l2_relative_error"), "equal-L2 error"
    )
    if nominated_l2 <= 0 or control_l2 <= 0 or relative > 1e-6:
        raise ValueError("E9c fixed-dose equal-L2 contract failed")
    for name in (
        "realized_bfloat16_nominated_edit_l2",
        "realized_bfloat16_control_edit_l2",
    ):
        if _finite_nonnegative(value.get(name), name) <= 0:
            raise ValueError("E9c realized edit must be positive")
    if not _is_sha256(value.get("nominated_edit_tensor_sha256")) or not _is_sha256(
        value.get("control_edit_tensor_sha256")
    ):
        raise ValueError("E9c fixed-dose edit hashes differ")


def expected_fixed_dose_from_bundle(bundle_path: Path) -> dict[str, Any]:
    with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
        decoders = handle.get_tensor("decoder_rows")
    nominated = decoders[0].float() * E9C_FIXED_C_ACTIVATION_DELTA
    nominated_l2 = float(nominated.norm())
    control_norm = float(decoders[1].float().norm())
    if nominated_l2 <= 0 or control_norm <= 0:
        raise ValueError("E9c frozen source decoder norm is zero")
    control_delta = nominated_l2 / control_norm
    control = decoders[1].float() * control_delta
    control_l2 = float(control.norm())
    nominated_bf16 = nominated.to(torch.bfloat16)
    control_bf16 = control.to(torch.bfloat16)
    return {
        "fixed_dose_id": "e9c:fixed-dose:v1",
        "nominated_source_feature": "L31:F49541",
        "control_source_feature": "L31:F139261",
        "nominated_activation_delta": E9C_FIXED_C_ACTIVATION_DELTA,
        "control_activation_delta": control_delta,
        "nominal_nominated_edit_l2": nominated_l2,
        "nominal_control_edit_l2": control_l2,
        "nominal_equal_l2_relative_error": abs(control_l2 - nominated_l2)
        / nominated_l2,
        "realized_bfloat16_nominated_edit_l2": float(
            nominated_bf16.float().norm()
        ),
        "realized_bfloat16_control_edit_l2": float(control_bf16.float().norm()),
        "nominated_edit_tensor_sha256": tensor_identity_sha256(nominated_bf16),
        "control_edit_tensor_sha256": tensor_identity_sha256(control_bf16),
        "prompt_activation_screen_used": False,
        "same_fixed_dose_for_all_24_prompts": True,
    }


def _walk_public(value: Any) -> None:
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
            raise ValueError("E9c public plan contains private text/token/tensor fields")
        for child in value.values():
            _walk_public(child)
    elif isinstance(value, list):
        for child in value:
            _walk_public(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("E9c public plan contains non-finite values")


def validate_e9c_execution_plan(
    payload: Mapping[str, Any],
    e9a_design: Mapping[str, Any],
    e9b_private_panel: Mapping[str, Any],
    e9b_public_manifest: Mapping[str, Any],
) -> None:
    validate_stamped_payload(payload)
    validate_stamped_payload(e9a_design)
    validate_e9b_private_panel(e9b_private_panel)
    validate_e9b_public_manifest(e9b_public_manifest)
    if e9a_design.get("payload_sha256") != E9A_DESIGN_PAYLOAD_SHA256:
        raise ValueError("E9c plan E9a frozen payload differs")
    if e9b_private_panel.get("payload_sha256") != E9B_PRIVATE_PANEL_PAYLOAD_SHA256:
        raise ValueError("E9c plan private E9b payload differs")
    if e9b_public_manifest.get("payload_sha256") != E9B_PUBLIC_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("E9c plan public E9b payload differs")
    if payload.get("schema_version") != E9C_SCHEMA_VERSION:
        raise ValueError("E9c plan schema differs")
    if payload.get("analysis_semantics_version") != E9C_PLAN_SEMANTICS_VERSION:
        raise ValueError("E9c plan semantics differ")
    if payload.get("status") != E9C_PLAN_STATUS:
        raise ValueError("E9c plan status differs")
    if payload.get("mode") != "fixed_dose_matched_risk_route_selectivity":
        raise ValueError("E9c plan mode differs")
    if payload.get("research_stage") != "understanding_testing_hypotheses":
        raise ValueError("E9c plan research stage differs")
    if payload.get("authorization") != E9C_AUTHORIZATION:
        raise ValueError("E9c plan authorization differs")
    if payload.get("model") != E9C_MODEL:
        raise ValueError("E9c model snapshot differs")
    if payload.get("runtime_contract") != E9C_RUNTIME_CONTRACT:
        raise ValueError("E9c future-runtime contract differs")
    if payload.get("numerical_contract") != E9C_NUMERICAL_CONTRACT:
        raise ValueError("E9c numerical contract differs")
    if payload.get("forward_contract") != E9C_FORWARD_CONTRACT:
        raise ValueError("E9c forward contract differs")
    if payload.get("analysis_contract") != E9C_ANALYSIS_CONTRACT:
        raise ValueError("E9c analysis contract differs")
    if payload.get("output_contract") != E9C_OUTPUT_CONTRACT:
        raise ValueError("E9c output contract differs")
    if payload.get("privacy") != E9C_PRIVACY:
        raise ValueError("E9c privacy contract differs")
    if payload.get("route_definition") != e9a_design.get("route_contract"):
        raise ValueError("E9c route definition differs from frozen E9a")
    if payload.get("excluded_arms") != e9a_design.get("excluded_arms"):
        raise ValueError("E9c excluded-arm closure differs from frozen E9a")
    if payload.get("execution_gate") != {
        "authorization_receipt_materialized": False,
        "execution_plan_is_scientific_and_byte_closure_not_execution_authority": True,
        "preflight_only_is_currently_permitted": True,
        "fresh_exact_user_confirmation_required_before_external_transfer": True,
    }:
        raise ValueError("E9c execution gate differs")
    if payload.get("freshness") != {
        "preferred_label": "project_untouched_target_model_panel",
        "pretraining_unseen_claim_allowed": False,
        "target_model_calls_on_exact_prompt_bytes": 0,
        "target_model_screening_used": False,
        "selection_or_replacement_after_outcomes": False,
    }:
        raise ValueError("E9c freshness boundary differs")
    frozen = payload.get("frozen_inputs")
    if frozen != {
        "e9a_design_file_sha256": E9A_DESIGN_FILE_SHA256,
        "e9a_design_payload_sha256": E9A_DESIGN_PAYLOAD_SHA256,
        "e9b_private_panel_file_sha256": E9B_PRIVATE_PANEL_FILE_SHA256,
        "e9b_private_panel_payload_sha256": E9B_PRIVATE_PANEL_PAYLOAD_SHA256,
        "e9b_public_manifest_file_sha256": E9B_PUBLIC_MANIFEST_FILE_SHA256,
        "e9b_public_manifest_payload_sha256": E9B_PUBLIC_MANIFEST_PAYLOAD_SHA256,
        "exact_target_model_calls_on_prompt_bytes_before_e9c": 0,
    }:
        raise ValueError("E9c frozen scientific/panel snapshot differs")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(E9C_INPUT_NAMES):
        raise ValueError("E9c execution input closure differs")
    for name, record in inputs.items():
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "upload_name",
        }:
            raise ValueError(f"E9c input record differs: {name}")
        if not isinstance(record.get("path"), str) or not _is_sha256(
            record.get("sha256")
        ) or not isinstance(record.get("upload_name"), str) or not record.get(
            "upload_name"
        ):
            raise ValueError(f"E9c input identity differs: {name}")
    expected_frozen_input_hashes = {
        "e9a_design": E9A_DESIGN_FILE_SHA256,
        "e9b_private_panel": E9B_PRIVATE_PANEL_FILE_SHA256,
        "e9b_public_manifest": E9B_PUBLIC_MANIFEST_FILE_SHA256,
    }
    for name, digest in expected_frozen_input_hashes.items():
        if inputs[name]["sha256"] != digest:
            raise ValueError(f"E9c frozen execution input hash differs: {name}")
    if payload.get("prompt_records") != expected_prompt_records(e9b_public_manifest):
        raise ValueError("E9c prompt-record closure differs")
    if payload.get("sender_worlds") != expected_sender_worlds(e9b_public_manifest):
        raise ValueError("E9c sender-world closure differs")
    if payload.get("route_cells") != expected_route_cells(e9b_public_manifest):
        raise ValueError("E9c route-cell closure differs")
    validate_fixed_dose(payload.get("fixed_dose"))
    source_manifest_path = Path(inputs["source_decoder_manifest"]["path"])
    source_bundle_path = Path(inputs["source_decoder_bundle"]["path"])
    if source_manifest_path.is_file() and source_bundle_path.is_file():
        if sha256_file(source_manifest_path) != inputs["source_decoder_manifest"][
            "sha256"
        ] or sha256_file(source_bundle_path) != inputs["source_decoder_bundle"][
            "sha256"
        ]:
            raise ValueError("E9c source decoder input hash differs")
        source_manifest = __import__("json").loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        validate_e9c_source_bundle_manifest(
            source_manifest,
            e9a_design=e9a_design,
            bundle_path=source_bundle_path,
        )
        if payload.get("source_decoder_manifest_payload_sha256") != source_manifest.get(
            "payload_sha256"
        ):
            raise ValueError("E9c plan references another source-decoder manifest")
        if payload.get("fixed_dose") != expected_fixed_dose_from_bundle(
            source_bundle_path
        ):
            raise ValueError("E9c fixed dose differs from pinned decoder bytes")
    _walk_public(payload)


__all__ = [
    "E9C_ANALYSIS_CONTRACT",
    "E9C_ANALYSIS_SEMANTICS_VERSION",
    "E9C_ANALYSIS_STATUS",
    "E9C_ARMS",
    "E9C_AUDIT_FORWARD_CEILING",
    "E9C_AUTHORIZATION",
    "E9C_AUTHORIZATION_REFERENCE",
    "E9C_FORWARD_CONTRACT",
    "E9C_INPUT_NAMES",
    "E9C_MODEL",
    "E9C_NUMERICAL_CONTRACT",
    "E9C_OUTPUT_CONTRACT",
    "E9C_PLAN_SEMANTICS_VERSION",
    "E9C_PLAN_STATUS",
    "E9C_PLANNED_SCIENTIFIC_FORWARDS",
    "E9C_POSITIONS",
    "E9C_PRIVACY",
    "E9C_RESULT_STATUS",
    "E9C_RESULT_SEMANTICS_VERSION",
    "E9C_RUNNING_STATUS",
    "E9C_RUNTIME_CONTRACT",
    "E9C_SCHEMA_VERSION",
    "E9C_SOURCE_BUNDLE_KEYS",
    "E9C_SOURCE_BUNDLE_SEMANTICS_VERSION",
    "E9C_SOURCE_BUNDLE_STATUS",
    "expected_prompt_records",
    "expected_fixed_dose_from_bundle",
    "expected_route_cells",
    "expected_sender_worlds",
    "expected_source_features",
    "tensor_identity_sha256",
    "validate_e9c_execution_plan",
    "validate_e9c_source_bundle_manifest",
    "validate_fixed_dose",
]
