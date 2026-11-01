"""Pure public-safe contracts for S9 E8b execution and exact route outcomes.

E8b is an operational freeze around the immutable E8a manifest and scientific
plan.  This module deliberately imports no tensor, model, GPU, generation, or
runtime package.  It authenticates the complete E8a snapshot, constrains the
authorized runtime bundle, and validates a future text-free result.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from s9_contract import (
    METRICS,
    MODEL_ID,
    MODEL_REVISION,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    canonical_sha256,
    require_unique,
    validate_stamped_payload,
)
from s9_e8_contract import (
    E8A_CONTROL_FEATURE,
    E8A_EXPECTED_INTENT_IDS,
    E8A_NOMINATED_SOURCE_FEATURE,
    E8A_PRIVACY,
    E8_SCHEMA_VERSION,
    validate_e8a_manifest,
    validate_e8a_plan,
)


E8B_SCHEMA_VERSION = E8_SCHEMA_VERSION
E8B_PLAN_SEMANTICS_VERSION = "s9-e8b-authorized-route-execution-bundle-v2"
E8B_PLAN_STATUS = "frozen_authorized_route_execution_bundle_v2"
E8B_RESULT_SEMANTICS_VERSION = "s9-e8b-exact-c-to-d-route-outcomes-v2"
E8B_RUNNING_STATUS = "running_exact_c_to_d_route_replication_v1"
E8B_RESULT_STATUS = "complete_exact_c_to_d_route_replication_v1"
E8B_ANALYSIS_SEMANTICS_VERSION = "s9-e8b-preregistered-c-to-d-analysis-v2"
E8B_ANALYSIS_STATUS = "complete_preregistered_c_to_d_replication_analysis_v1"

E8A_FROZEN_MANIFEST_FILE_SHA256 = (
    "110932011bbb087a16bff0b2128b923e545534c9227e12648bfdb9bbf250744a"
)
E8A_FROZEN_MANIFEST_PAYLOAD_SHA256 = (
    "0d0de4fe9983c408b99158b74abd93e63421e0d098e8f07b25256520aad841c4"
)
E8A_FROZEN_PLAN_FILE_SHA256 = (
    "76a6879af6ac9fd2f2595ab41b275318720adf0d47320455c0b16c112e3503ed"
)
E8A_FROZEN_PLAN_PAYLOAD_SHA256 = (
    "270167e9e8bb9c35f997e5795529c265e88b57bf138164b2811c6346355b794b"
)
E8A_FROZEN_SCIENTIFIC_SNAPSHOT_SHA256 = (
    "4cd5a30b185400d095672a3ab39574ed639165f0ec3f1c50db2a293f6e9848a4"
)
E8B_SEMANTIC_ERRATUM_FILE_SHA256 = (
    "2a2566d260cb1bf5341f2b3cfc2bd1ea073cd9a9d013f9fa08a9e8a4cfe27691"
)

E8B_AUTHORIZATION_SCOPE = (
    "freeze_the_e8b_bundle_and_run_only_the_frozen_e8a_180_forward_"
    "original_model_route_replication_under_the_198_forward_ceiling"
)

E8B_INPUT_NAMES = (
    "e8a_manifest",
    "e8a_plan",
    "semantic_erratum",
    "private_prompts",
    "probe_direction",
    "transcoder_row_bundle",
    "transcoder_row_manifest",
    "transcoder_bundle_builder",
    "range_helper",
    "runner",
    "colab_runner",
    "runtime_probe",
    "analyzer",
    "runtime_installer",
    "base_runner",
    "base_contract",
    "e8_contract",
    "e8b_contract",
    "execution_plan_builder",
)

E8B_PRIVACY = {
    **E8A_PRIVACY,
    "raw_transcoder_rows_in_public_artifacts": False,
    "input_token_ids_in_public_artifacts": False,
    "input_token_tensor_hashes_in_public_artifacts": True,
}

E8B_RUNTIME_CONTRACT = {
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
    "random_seed": 90801,
    "determinism": {
        "model_eval": True,
        "gradient_tracking": False,
        "parameters_require_grad": False,
        "torch_manual_seed": 90801,
        "cuda_manual_seed_all": 90801,
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
    "runtime_probe_before_model_load": True,
    "uploaded_hashes_verified_before_model_load": True,
    "downloaded_model_and_transcoder_hashes_verified_before_first_forward": True,
    "cross_resume_runtime_fingerprint_required": True,
    "runtime_fingerprint_fields": [
        "python",
        "platform",
        "device",
        "device_capability",
        "cuda_total_memory_gib",
        "torch_cuda_version",
        "cudnn_version",
        "nvidia_driver_version",
        "packages",
        "dtype",
        "random_seed",
        "determinism",
        "cublas_workspace_config",
    ],
    "gpu_instance_identity_policy": (
        "instance_uuid_not_recorded_but_accelerator_capability_driver_and_software_"
        "fingerprint_must_match_exactly_across_resume"
    ),
}

E8B_NUMERICAL_CONTRACT = {
    "feature_row_storage_dtype": "bfloat16",
    "feature_activation_operands_and_output_dtype": "bfloat16",
    "feature_activation_device": "cuda",
    "feature_activation_reference": "match_developed_e7c_operational_path",
    "nominal_numeric_dtype": "float32",
    "realized_numeric_dtype": "bfloat16",
    "hook_fire_count_per_declared_capture": 1,
    "invariant_absolute_tolerance": 1e-7,
    "pre_replacement_max_abs_mismatch": 1e-4,
    "penultimate_receiver_state_max_abs_change": 1e-4,
    "penultimate_endpoint_abs_tolerance": 1e-4,
    "scientific_threshold_miss_action": (
        "retain_complete_result_and_mark_analysis_not_evaluable"
    ),
    "zero_dose": {
        "definition": (
            "nominated_activation_delta_exactly_zero_and_nominal_"
            "nominated_edit_l2_exactly_zero"
        ),
        "control_activation_delta": 0.0,
        "nominal_control_edit_l2": 0.0,
        "nominal_equal_l2_relative_error": 0.0,
        "retain_all_four_route_cells": True,
        "retuning_allowed": False,
    },
    "nominal_equal_l2_relative_error_formula": (
        "abs(nominal_control_edit_l2-nominal_nominated_edit_l2)/"
        "nominal_nominated_edit_l2"
    ),
    "nominal_equal_l2_relative_tolerance": 1e-6,
    "realized_bfloat16_equal_l2_role": "reported_not_gated",
    "topic_dose_ratio": {
        "orientation": "harmful_over_benign",
        "formula": (
            "nominal_nominated_edit_l2[harmful]/"
            "nominal_nominated_edit_l2[benign]"
        ),
        "inclusive_interval": [0.8, 1.25],
        "zero_benign_denominator_action": "not_evaluable",
        "outside_interval_action": (
            "semantic_differential_not_evaluable_but_raw_results_retained"
        ),
    },
}

E8B_FORWARD_CONTRACT = {
    "planned_scientific": 180,
    "audit_ceiling": 198,
    "maximum_audit_forwards": 18,
    "scientific_forwards_must_equal_plan": True,
    "cumulative_across_resumes": True,
    "counter_reset_on_resume_allowed": False,
    "resume_policy": "authenticated_checkpoint_without_replaying_completed_forwards",
    "audit_forward_policy": "predeclared_runtime_or_hash_diagnostics_only",
    "new_scientific_cells_from_audit_budget_allowed": False,
    "raw_checkpoint_storage": "content_addressed_versioned_safetensors",
    "prior_raw_checkpoint_preserved_until_json_pointer_commit": True,
}

E8B_OUTPUT_CONTRACT = {
    "public_result_required": True,
    "runtime_proof_required": True,
    "private_raw_tensor_bundle_required": True,
    "resumable_checkpoint_required": True,
    "required_prompt_records": 30,
    "required_dose_audits": 15,
    "required_sender_worlds": 30,
    "required_route_cells": 60,
    "public_prompt_or_completion_text_allowed": False,
    "public_raw_receiver_tensors_allowed": False,
    "public_input_token_tensor_hash_required": True,
    "analysis_implemented_in_this_stage": True,
}

E8B_SEMANTIC_RESOLUTION = {
    "status": "frozen_preoutcome_semantic_erratum_v1",
    "erratum_file_sha256": E8B_SEMANTIC_ERRATUM_FILE_SHA256,
    "primary_compound_and_topic_win_threshold": 0.0,
    "primary_win_operator": "strictly_greater_than",
    "scalar_sign_test_nonwin_threshold": 1e-7,
    "scalar_sign_tests_decide_compound_gate": False,
    "raw_nominated_median_positivity_role": (
        "reported_redundant_full_pass_consistency_check"
    ),
    "feature_activation_semantics": (
        "cuda_bfloat16_encoder_state_bias_operands_and_bfloat16_output"
    ),
    "nominal_decoded_edit_and_l2_semantics": "float32",
    "applied_edit_semantics": "bfloat16",
}

_FROZEN_SNAPSHOT = {
    "scope": "entire_validated_e8a_manifest_and_plan_payloads",
    "manifest_file_sha256": E8A_FROZEN_MANIFEST_FILE_SHA256,
    "manifest_payload_sha256": E8A_FROZEN_MANIFEST_PAYLOAD_SHA256,
    "plan_file_sha256": E8A_FROZEN_PLAN_FILE_SHA256,
    "plan_payload_sha256": E8A_FROZEN_PLAN_PAYLOAD_SHA256,
    "scientific_snapshot_sha256": E8A_FROZEN_SCIENTIFIC_SNAPSHOT_SHA256,
}

_AUTHORIZATION_WITHOUT_REFERENCE = {
    "explicit_user_confirmation_received": True,
    "scope": E8B_AUTHORIZATION_SCOPE,
    "bundle_implementation_allowed": True,
    "gpu_allocation_allowed": True,
    "original_model_load_allowed": True,
    "original_model_outcomes_allowed": True,
    "validation_expansion_prompt_text_to_model_allowed": True,
    "generation_allowed": False,
    "e7d_allowed": False,
    "backward_pass_allowed": False,
    "confirmation_partition_allowed": False,
    "new_scientific_cells_allowed": False,
    "excluded_arms_allowed": False,
}

_KNOWN_FIXED_INPUT_HASHES = {
    "e8a_manifest": E8A_FROZEN_MANIFEST_FILE_SHA256,
    "e8a_plan": E8A_FROZEN_PLAN_FILE_SHA256,
    "semantic_erratum": E8B_SEMANTIC_ERRATUM_FILE_SHA256,
    "private_prompts": (
        "f5f65a41a2ca6cf479a58f9d21e4cfa50becadd28fc033ceeaa3b88eb4bd41e8"
    ),
    "probe_direction": (
        "3d2048a873ed05b334545e703e0253707aad9b3de76176eb1fc9d8a71b30d72e"
    ),
    "base_contract": (
        "35717f36e91ec3bcdda1c6adcd220477dc899f7ac4b1d17de8aaee9a0e07753b"
    ),
    "e8_contract": (
        "6fc5c7ea89b9549e6d558ab7fb5f06f1be5315da07bbceace8c60fc1e38d2253"
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_GATE_CLASSES = {
    "inactive_to_active",
    "active_to_inactive",
    "active_stable",
    "inactive_stable",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _walk_public(value: Any) -> None:
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
            _walk_public(item)
    elif isinstance(value, list):
        for item in value:
            _walk_public(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Public E8b payload contains a non-finite value")


def e8a_scientific_snapshot_sha256(
    manifest: Mapping[str, Any], plan: Mapping[str, Any]
) -> str:
    """Hash every validated E8a manifest and plan field, including boundaries."""

    validate_e8a_plan(plan, manifest)
    return canonical_sha256({"manifest": manifest, "plan": plan})


def _validate_immutable_e8a(
    manifest: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    validate_e8a_plan(plan, manifest)
    if manifest.get("payload_sha256") != E8A_FROZEN_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("E8a manifest is not the immutable frozen payload")
    if plan.get("payload_sha256") != E8A_FROZEN_PLAN_PAYLOAD_SHA256:
        raise ValueError("E8a plan is not the immutable frozen payload")
    if (
        e8a_scientific_snapshot_sha256(manifest, plan)
        != E8A_FROZEN_SCIENTIFIC_SNAPSHOT_SHA256
    ):
        raise ValueError("E8a complete scientific snapshot hash differs")


def _validate_inputs(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(E8B_INPUT_NAMES):
        raise ValueError("E8b execution-bundle input closure differs")
    for name in E8B_INPUT_NAMES:
        record = value[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"E8b input {name} is malformed")
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError(f"E8b input {name} path is missing")
        if not _is_sha256(record.get("sha256")):
            raise ValueError(f"E8b input {name} hash is malformed")
    for name, expected in _KNOWN_FIXED_INPUT_HASHES.items():
        if value[name]["sha256"] != expected:
            raise ValueError(f"E8b fixed input hash differs: {name}")
    bundle_path = value["transcoder_row_bundle"]["path"]
    if "/private/" not in bundle_path.replace("\\", "/"):
        raise ValueError("E8b raw transcoder rows must remain in private storage")


def _validate_authorization(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("E8b authorization is missing")
    if set(value) != {*_AUTHORIZATION_WITHOUT_REFERENCE, "confirmation_reference"}:
        raise ValueError("E8b authorization scope differs")
    reference = value.get("confirmation_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("E8b exact user confirmation reference is missing")
    observed = {key: value[key] for key in _AUTHORIZATION_WITHOUT_REFERENCE}
    if observed != _AUTHORIZATION_WITHOUT_REFERENCE:
        raise ValueError("E8b authorization expanded or narrowed")


def validate_e8b_execution_plan(
    payload: Mapping[str, Any],
    e8a_manifest: Mapping[str, Any],
    e8a_plan: Mapping[str, Any],
) -> None:
    """Validate E8b's authorized execution bundle against immutable E8a."""

    _validate_immutable_e8a(e8a_manifest, e8a_plan)
    validate_stamped_payload(payload)
    _walk_public(payload)
    if payload.get("schema_version") != E8B_SCHEMA_VERSION:
        raise ValueError("E8b execution-plan schema differs")
    if payload.get("analysis_semantics_version") != E8B_PLAN_SEMANTICS_VERSION:
        raise ValueError("E8b execution-plan semantics differ")
    if payload.get("status") != E8B_PLAN_STATUS:
        raise ValueError("E8b execution plan is not frozen")
    if payload.get("mode") != "prospective_route_replication":
        raise ValueError("E8b execution-plan mode differs")
    if payload.get("research_stage") != "understanding_testing_hypotheses":
        raise ValueError("E8b research stage differs")
    if not isinstance(payload.get("objective"), str) or not payload["objective"]:
        raise ValueError("E8b objective is missing")
    if payload.get("frozen_e8a_snapshot") != _FROZEN_SNAPSHOT:
        raise ValueError("E8b complete frozen E8a scientific snapshot differs")
    if payload.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("E8b model identity differs")
    if payload.get("transcoder") != {
        "id": TRANSCODER_ID,
        "revision": TRANSCODER_REVISION,
    }:
        raise ValueError("E8b transcoder identity differs")
    _validate_authorization(payload.get("authorization"))
    if payload.get("confirmation_boundary") != e8a_plan.get("confirmation_boundary"):
        raise ValueError("E8b confirmation boundary differs from immutable E8a")
    if payload.get("outcome_disclosure") != e8a_plan.get("outcome_disclosure"):
        raise ValueError("E8b outcome disclosure differs from immutable E8a")
    if payload.get("excluded_arms") != e8a_plan.get("excluded_arms"):
        raise ValueError("E8b excluded scientific arms differ from immutable E8a")
    if payload.get("semantic_resolution") != E8B_SEMANTIC_RESOLUTION:
        raise ValueError("E8b pre-outcome semantic resolution differs")
    if payload.get("runtime_contract") != E8B_RUNTIME_CONTRACT:
        raise ValueError("E8b runtime contract differs")
    if payload.get("numerical_contract") != E8B_NUMERICAL_CONTRACT:
        raise ValueError("E8b numerical contract differs")
    if payload.get("forward_contract") != E8B_FORWARD_CONTRACT:
        raise ValueError("E8b cumulative forward contract differs")
    if payload.get("output_contract") != E8B_OUTPUT_CONTRACT:
        raise ValueError("E8b output contract differs")
    _validate_inputs(payload.get("inputs"))
    if payload.get("privacy") != E8B_PRIVACY:
        raise ValueError("E8b privacy contract differs")


def _validate_endpoint_record(record: Any, name: str) -> dict[str, float]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} endpoint record is missing")
    required = (
        "Y0",
        "YS",
        "YS_R0",
        "YR_S",
        "T_S",
        "D_R",
        "B_R",
        "Q_R",
        "invariant_error",
    )
    if not set(required) <= set(record):
        raise ValueError(f"{name} endpoint shape differs")
    values = {key: _finite(record[key], f"{name}.{key}") for key in required}
    if values["invariant_error"] < 0.0:
        raise ValueError(f"{name}.invariant_error must be absolute")
    return values


def _validate_prompt_records(value: Any, manifest: Mapping[str, Any]) -> None:
    if not isinstance(value, list) or len(value) != 30:
        raise ValueError("E8b result requires exactly 30 prompt records")
    expected: dict[str, tuple[str, str]] = {}
    for pair in manifest["intent_pairs"]:
        for side in ("source_prompt", "target_prompt"):
            prompt = pair[side]
            expected[prompt["prompt_key"]] = (
                prompt["prompt_sha256"],
                prompt["frame_id"],
            )
    require_unique((row.get("prompt_key") for row in value), "E8b prompt-record keys")
    if {row.get("prompt_key") for row in value} != set(expected):
        raise ValueError("E8b result prompt identities differ")
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("E8b prompt record is malformed")
        key = row["prompt_key"]
        prompt_sha, frame = expected[key]
        if row.get("prompt_sha256") != prompt_sha or row.get("frame_id") != frame:
            raise ValueError(f"E8b prompt metadata differs: {key}")
        if not _is_sha256(row.get("chat_text_sha256")):
            raise ValueError(f"E8b chat hash is missing: {key}")
        if not _is_sha256(row.get("input_token_tensor_sha256")):
            raise ValueError(f"E8b token-tensor hash is missing: {key}")
        count = row.get("input_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"E8b token count is malformed: {key}")


def _validate_dose_audits(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 15:
        raise ValueError("E8b result requires exactly 15 dose audits")
    require_unique((row.get("intent_id") for row in value), "E8b dose-audit intents")
    by_intent = {row.get("intent_id"): row for row in value}
    if set(by_intent) != set(E8A_EXPECTED_INTENT_IDS):
        raise ValueError("E8b dose-audit intent closure differs")
    for intent_id, row in by_intent.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"E8b {intent_id} dose audit is malformed")
        if row.get("nominal_numeric_dtype") != "float32":
            raise ValueError(f"E8b {intent_id} nominal dose dtype differs")
        if row.get("realized_numeric_dtype") != "bfloat16":
            raise ValueError(f"E8b {intent_id} realized dose dtype differs")
        natural = row.get("natural_c")
        if not isinstance(natural, Mapping):
            raise ValueError(f"E8b {intent_id} natural-C audit is missing")
        if natural.get("numeric_dtype") != "bfloat16":
            raise ValueError(f"E8b {intent_id} natural-C dtype differs")
        if natural.get("numeric_device") != "cuda":
            raise ValueError(f"E8b {intent_id} natural-C device differs")
        source_pre = _finite(
            natural.get("source_preactivation"), f"{intent_id}.source_preactivation"
        )
        target_pre = _finite(
            natural.get("target_preactivation"), f"{intent_id}.target_preactivation"
        )
        source_activation = _nonnegative(
            natural.get("source_activation"), f"{intent_id}.source_activation"
        )
        target_activation = _nonnegative(
            natural.get("target_activation"), f"{intent_id}.target_activation"
        )
        natural_delta = _finite(
            natural.get("activation_delta"), f"{intent_id}.natural_activation_delta"
        )
        if source_activation != max(source_pre, 0.0) or target_activation != max(
            target_pre, 0.0
        ):
            raise ValueError(f"E8b {intent_id} natural-C ReLU arithmetic differs")
        if natural_delta != source_activation - target_activation:
            raise ValueError(f"E8b {intent_id} natural-C delta arithmetic differs")
        delta_c = _finite(
            row.get("nominated_activation_delta"), f"{intent_id}.nominated_delta"
        )
        if delta_c != natural_delta:
            raise ValueError(f"E8b {intent_id} nominated dose differs from natural C")
        delta_m = _finite(
            row.get("control_activation_delta"), f"{intent_id}.control_delta"
        )
        nominal_c = _nonnegative(
            row.get("nominal_nominated_edit_l2"), f"{intent_id}.nominal_C_l2"
        )
        nominal_m = _nonnegative(
            row.get("nominal_control_edit_l2"), f"{intent_id}.nominal_M_l2"
        )
        nominal_error = _nonnegative(
            row.get("nominal_equal_l2_relative_error"),
            f"{intent_id}.nominal_equal_l2_error",
        )
        realized_c = _nonnegative(
            row.get("realized_bfloat16_nominated_edit_l2"),
            f"{intent_id}.realized_C_l2",
        )
        realized_m = _nonnegative(
            row.get("realized_bfloat16_control_edit_l2"),
            f"{intent_id}.realized_M_l2",
        )
        realized_error = _nonnegative(
            row.get("realized_bfloat16_equal_l2_relative_error"),
            f"{intent_id}.realized_equal_l2_error",
        )
        zero = row.get("zero_dose")
        if not isinstance(zero, bool):
            raise ValueError(f"E8b {intent_id} zero-dose flag is malformed")
        # These are observed scientific audit values.  Their preregistered
        # relations (zero-dose classification, comparator sign/equal-L2, and
        # reported norm errors) are evaluated by the deterministic analyzer.
        # A threshold miss is a complete but non-evaluable run, not a malformed
        # result that should be discarded before analysis.
        _ = (delta_c, delta_m, nominal_c, nominal_m, nominal_error)
        _ = (realized_c, realized_m, realized_error, zero)
    return by_intent


def _validate_sender_worlds(
    value: Any, plan: Mapping[str, Any], doses: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 30:
        raise ValueError("E8b result requires exactly 30 sender worlds")
    require_unique(
        (row.get("sender_world_id") for row in value), "E8b sender-world ids"
    )
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    risk = {row["intent_id"]: row["risk_class"] for row in plan["intent_pairs"]}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("E8b sender world is malformed")
        intent_id = row.get("intent_id")
        arm = row.get("arm")
        key = (intent_id, arm)
        if intent_id not in risk or arm not in {"nominated_c", "equal_l2_control"}:
            raise ValueError("E8b sender-world identity differs")
        if key in by_key:
            raise ValueError("E8b sender-world intent/arm pairs must be unique")
        if row.get("risk_class") != risk[intent_id]:
            raise ValueError("E8b sender-world risk class differs")
        expected_feature = (
            E8A_NOMINATED_SOURCE_FEATURE
            if arm == "nominated_c"
            else E8A_CONTROL_FEATURE
        )
        if row.get("source_feature") != expected_feature:
            raise ValueError("E8b sender-world feature differs")
        dose = doses[intent_id]
        expected_delta = dose[
            "nominated_activation_delta"
            if arm == "nominated_c"
            else "control_activation_delta"
        ]
        expected_nominal = dose[
            "nominal_nominated_edit_l2"
            if arm == "nominated_c"
            else "nominal_control_edit_l2"
        ]
        expected_realized = dose[
            "realized_bfloat16_nominated_edit_l2"
            if arm == "nominated_c"
            else "realized_bfloat16_control_edit_l2"
        ]
        for field, expected in (
            ("source_activation_delta", expected_delta),
            ("nominal_applied_residual_l2", expected_nominal),
            ("realized_bfloat16_applied_residual_l2", expected_realized),
        ):
            actual = _finite(row.get(field), f"{intent_id}.{arm}.{field}")
            if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"E8b sender-world {field} differs from dose audit")
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(METRICS):
            raise ValueError("E8b sender-world endpoints differ")
        for metric in METRICS:
            record = endpoints[metric]
            if not isinstance(record, Mapping) or not {
                "Y0",
                "YS",
                "T_S",
            } <= set(record):
                raise ValueError("E8b sender-world endpoint shape differs")
            y0 = _finite(record["Y0"], f"{intent_id}.{arm}.{metric}.Y0")
            ys = _finite(record["YS"], f"{intent_id}.{arm}.{metric}.YS")
            ts = _finite(record["T_S"], f"{intent_id}.{arm}.{metric}.T_S")
            _ = (y0, ys, ts)
        by_key[key] = row
    expected_keys = {
        (intent_id, arm)
        for intent_id in E8A_EXPECTED_INTENT_IDS
        for arm in ("nominated_c", "equal_l2_control")
    }
    if set(by_key) != expected_keys:
        raise ValueError("E8b sender-world closure differs")
    return by_key


def _validate_gate_diagnostic(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} gate diagnostic is missing")
    if value.get("feature") != "L32:F50401":
        raise ValueError(f"{name} D gate feature differs")
    if value.get("numeric_dtype") != "bfloat16" or value.get(
        "numeric_device"
    ) != "cuda":
        raise ValueError(f"{name} D gate arithmetic must use CUDA BF16")
    pre0 = _finite(value.get("baseline_preactivation"), f"{name}.baseline_pre")
    pre_s = _finite(value.get("sender_preactivation"), f"{name}.sender_pre")
    pre_delta = _finite(value.get("preactivation_delta"), f"{name}.pre_delta")
    act0 = _finite(value.get("baseline_activation"), f"{name}.baseline_activation")
    act_s = _finite(value.get("sender_activation"), f"{name}.sender_activation")
    if value.get("crossing_class") not in _GATE_CLASSES:
        raise ValueError(f"{name} D gate-crossing class is malformed")
    if act0 < 0.0 or act_s < 0.0:
        raise ValueError(f"{name} D post-ReLU activation is negative")
    if pre_delta != pre_s - pre0:
        raise ValueError(f"{name} D preactivation delta arithmetic differs")
    if act0 != max(pre0, 0.0) or act_s != max(pre_s, 0.0):
        raise ValueError(f"{name} D ReLU arithmetic differs")
    if "gate_crossing" in value and not isinstance(value["gate_crossing"], bool):
        raise ValueError(f"{name} gate-crossing boolean is malformed")
    _ = (pre0, pre_s, act0, act_s)


def _validate_receiver_state(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} receiver-state metadata is missing")
    for key in ("baseline_tensor_sha256", "sender_tensor_sha256"):
        if not _is_sha256(value.get(key)):
            raise ValueError(f"{name}.{key} is malformed")
    for key in ("baseline_raw_tensor_key", "sender_raw_tensor_key"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{name}.{key} is missing")
    if value.get("dtype") != "bfloat16":
        raise ValueError(f"{name} receiver-state dtype differs")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or shape != [2560]
        or any(isinstance(item, bool) for item in shape)
    ):
        raise ValueError(f"{name} must reference one complete 2560-vector")
    output = {
        key: _nonnegative(value.get(key), f"{name}.{key}")
        for key in ("baseline_l2", "sender_l2", "state_delta_l2", "state_delta_max_abs")
    }
    return output


def _validate_receiver_audit(value: Any, name: str, mismatch_atol: float) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} receiver audit is missing")
    for key in ("restoration_hook_fire_count", "rescue_hook_fire_count"):
        observed = value.get(key)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(f"{name} hook fire count is malformed")
    for key in (
        "restoration_pre_replacement_max_abs_mismatch",
        "rescue_pre_replacement_max_abs_mismatch",
    ):
        _nonnegative(value.get(key), f"{name}.{key}")
    if "mismatch_atol" in value and not math.isclose(
        _finite(value["mismatch_atol"], f"{name}.mismatch_atol"),
        mismatch_atol,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"{name} mismatch tolerance differs")


def _validate_route_cells(
    value: Any,
    plan: Mapping[str, Any],
    sender_worlds: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    if not isinstance(value, list) or len(value) != 60:
        raise ValueError("E8b result requires exactly 60 route cells")
    require_unique((row.get("cell_id") for row in value), "E8b result cell ids")
    expected = {row["cell_id"]: row for row in plan["route_cells"]}
    if {row.get("cell_id") for row in value} != set(expected):
        raise ValueError("E8b result route-cell identities differ")
    mismatch_atol = float(
        plan["technical_evaluability"]["pre_replacement_max_abs_mismatch"]
    )
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("E8b route result is malformed")
        cell = expected[row["cell_id"]]
        if row.get("source_cell_id") != row["cell_id"]:
            raise ValueError(f"E8b {row['cell_id']} source-cell identity differs")
        for key in (
            "intent_id",
            "risk_class",
            "arm",
            "receiver_token_position",
            "receiver_position_role",
            "included_in_empirical_analysis",
        ):
            if row.get(key) != cell.get(key):
                raise ValueError(f"E8b {row['cell_id']} metadata differs: {key}")
        _validate_gate_diagnostic(row.get("gate_diagnostic"), row["cell_id"])
        state = _validate_receiver_state(row.get("receiver_state"), row["cell_id"])
        _validate_receiver_audit(
            row.get("receiver_audit"), row["cell_id"], mismatch_atol
        )
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(METRICS):
            raise ValueError(f"E8b {row['cell_id']} endpoints differ")
        for metric in METRICS:
            _validate_endpoint_record(endpoints[metric], f"{row['cell_id']}.{metric}")
        _ = (state, sender_worlds[(row["intent_id"], row["arm"])])


def _validate_forward_accounting(value: Any) -> None:
    expected_keys = {
        "planned_scientific",
        "completed_scientific",
        "audit_forwards",
        "cumulative_total_across_resumes",
        "audit_ceiling",
        "audit_ceiling_exceeded",
        "counter_reset_on_resume",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("E8b forward accounting shape differs")
    for key in (
        "planned_scientific",
        "completed_scientific",
        "audit_forwards",
        "cumulative_total_across_resumes",
        "audit_ceiling",
    ):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] < 0
        ):
            raise ValueError(f"E8b forward accounting {key} is malformed")
    if value["planned_scientific"] != 180 or value["completed_scientific"] != 180:
        raise ValueError("E8b must complete exactly 180 scientific forwards")
    if not 0 <= value["audit_forwards"] <= 18:
        raise ValueError("E8b audit-forward allowance differs")
    if value["cumulative_total_across_resumes"] != (
        value["completed_scientific"] + value["audit_forwards"]
    ):
        raise ValueError("E8b cumulative forward count does not close")
    if value["audit_ceiling"] != 198:
        raise ValueError("E8b audit ceiling differs")
    if value["cumulative_total_across_resumes"] > 198:
        raise ValueError("E8b cumulative forward ceiling exceeded")
    if value["audit_ceiling_exceeded"] is not False:
        raise ValueError("E8b audit ceiling was exceeded")
    if value["counter_reset_on_resume"] is not False:
        raise ValueError("E8b forward counter reset on resume")


def _validate_runtime(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("E8b runtime record is missing")
    device = value.get("device")
    if not isinstance(device, str) or not any(
        name in device for name in E8B_RUNTIME_CONTRACT["preferred_accelerators"]
    ):
        raise ValueError("E8b runtime accelerator differs")
    memory = _finite(value.get("cuda_total_memory_gib"), "runtime.cuda_memory")
    if memory < float(E8B_RUNTIME_CONTRACT["minimum_cuda_memory_gib"]):
        raise ValueError("E8b runtime CUDA memory is below contract")
    if value.get("dtype") != E8B_RUNTIME_CONTRACT["dtype"]:
        raise ValueError("E8b runtime dtype differs")
    if value.get("random_seed") != E8B_RUNTIME_CONTRACT["random_seed"]:
        raise ValueError("E8b runtime seed differs")
    if value.get("packages") != E8B_RUNTIME_CONTRACT["package_versions"]:
        raise ValueError("E8b runtime package versions differ")
    if value.get("determinism") != E8B_RUNTIME_CONTRACT["determinism"]:
        raise ValueError("E8b runtime determinism record differs")
    if value.get("cublas_workspace_config") != E8B_RUNTIME_CONTRACT[
        "determinism"
    ]["cublas_workspace_config"]:
        raise ValueError("E8b CUBLAS deterministic workspace differs")
    for key in ("python", "platform", "torch_cuda_version", "nvidia_driver_version"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"E8b runtime fingerprint field is missing: {key}")
    capability = value.get("device_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability)
    ):
        raise ValueError("E8b runtime device capability is malformed")
    cudnn = value.get("cudnn_version")
    if isinstance(cudnn, bool) or not isinstance(cudnn, int) or cudnn <= 0:
        raise ValueError("E8b cuDNN runtime fingerprint is malformed")
    expected_fingerprint = canonical_sha256(
        {
            "runtime_fingerprint": {
                key: value.get(key)
                for key in E8B_RUNTIME_CONTRACT["runtime_fingerprint_fields"]
            }
        }
    )
    if value.get("runtime_fingerprint_sha256") != expected_fingerprint:
        raise ValueError("E8b runtime fingerprint hash differs")


def _validate_model_snapshot_proof(
    value: Any, model: Mapping[str, Any]
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("E8b model snapshot proof is missing")
    if value.get("repo_id") != model.get("id") or value.get(
        "resolved_revision"
    ) != model.get("revision"):
        raise ValueError("E8b model snapshot identity differs")
    if value.get("verified_before_first_forward") is not True or value.get(
        "all_cache_objects_content_authenticated"
    ) is not True:
        raise ValueError("E8b model snapshot was not authenticated before execution")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("E8b model snapshot file closure is missing")
    paths: list[str] = []
    total_size = 0
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("E8b model snapshot file record is malformed")
        path = record.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("E8b model snapshot file path is malformed")
        paths.append(path)
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("E8b model snapshot file size is malformed")
        total_size += size
        content_sha256 = record.get("content_sha256")
        if not _is_sha256(content_sha256):
            raise ValueError("E8b model snapshot content SHA-256 is malformed")
        algorithm = record.get("cache_object_algorithm")
        cache_object = record.get("cache_object")
        if algorithm == "sha256":
            if cache_object != content_sha256:
                raise ValueError("E8b model snapshot SHA-256 cache object differs")
        elif algorithm == "git_blob_sha1":
            if not isinstance(cache_object, str) or not _SHA1_RE.fullmatch(
                cache_object
            ):
                raise ValueError("E8b model snapshot Git cache object is malformed")
        else:
            raise ValueError("E8b model snapshot cache-object algorithm differs")
    require_unique(paths, "E8b model snapshot paths")
    if value.get("file_count") != len(files):
        raise ValueError("E8b model snapshot file count differs")
    if value.get("total_size_bytes") != total_size:
        raise ValueError("E8b model snapshot byte count differs")
    if value.get("file_closure_sha256") != canonical_sha256({"files": files}):
        raise ValueError("E8b model snapshot file-closure hash differs")
    if "config.json" not in paths or "tokenizer.json" not in paths or not any(
        path.endswith(".safetensors") for path in paths
    ):
        raise ValueError("E8b model snapshot loaded-file closure differs")


def validate_e8b_result(
    payload: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    e8a_manifest: Mapping[str, Any],
    e8a_plan: Mapping[str, Any],
) -> None:
    """Validate a complete E8b public result and every frozen runtime gate."""

    validate_e8b_execution_plan(execution_plan, e8a_manifest, e8a_plan)
    validate_stamped_payload(payload)
    _walk_public(payload)
    if payload.get("schema_version") != E8B_SCHEMA_VERSION:
        raise ValueError("E8b result schema differs")
    if payload.get("analysis_semantics_version") != E8B_RESULT_SEMANTICS_VERSION:
        raise ValueError("E8b result semantics differ")
    if payload.get("status") != E8B_RESULT_STATUS:
        raise ValueError("E8b result is incomplete")
    if payload.get("plan_payload_sha256") != execution_plan.get("payload_sha256"):
        raise ValueError("E8b result references another execution plan")
    if payload.get("frozen_e8a_snapshot_sha256") != (
        execution_plan["frozen_e8a_snapshot"]["scientific_snapshot_sha256"]
    ):
        raise ValueError("E8b result references another scientific snapshot")
    if payload.get("model") != execution_plan.get("model"):
        raise ValueError("E8b result model differs")
    if payload.get("transcoder") != execution_plan.get("transcoder"):
        raise ValueError("E8b result transcoder differs")
    if payload.get("semantic_resolution") != execution_plan.get(
        "semantic_resolution"
    ):
        raise ValueError("E8b result semantic resolution differs")
    if payload.get("confirmation_boundary") != execution_plan.get(
        "confirmation_boundary"
    ):
        raise ValueError("E8b result confirmation boundary differs")
    for key in ("generation_executed", "backward_pass_executed", "e7d_executed"):
        if payload.get(key) is not False:
            raise ValueError(f"E8b result authorization boundary differs: {key}")
    if payload.get("privacy") != E8B_PRIVACY:
        raise ValueError("E8b result privacy differs")
    _validate_prompt_records(payload.get("prompt_records"), e8a_manifest)
    doses = _validate_dose_audits(payload.get("dose_audits"))
    sender_worlds = _validate_sender_worlds(
        payload.get("sender_worlds"), e8a_plan, doses
    )
    _validate_route_cells(payload.get("route_cells"), e8a_plan, sender_worlds)
    _validate_forward_accounting(payload.get("forward_accounting"))
    _validate_runtime(payload.get("runtime"))
    _validate_model_snapshot_proof(
        payload.get("model_snapshot_proof"), execution_plan["model"]
    )
    proof = payload.get("runtime_proof")
    if (
        not isinstance(proof, Mapping)
        or not _is_sha256(proof.get("file_sha256"))
        or not _is_sha256(proof.get("payload_sha256"))
        or proof.get("public") is not True
        or proof.get("model_forward_executed_by_probe") is not False
    ):
        raise ValueError("E8b authenticated pre-load runtime proof is missing")
    expected_hashes = {
        name: record["sha256"] for name, record in execution_plan["inputs"].items()
    }
    if payload.get("execution_input_hashes") != expected_hashes:
        raise ValueError("E8b result execution-input hashes differ")
    raw = payload.get("private_raw_tensor_bundle")
    if (
        not isinstance(raw, Mapping)
        or not isinstance(raw.get("path"), str)
        or not raw["path"]
        or not _is_sha256(raw.get("sha256"))
        or raw.get("public") is not False
        or isinstance(raw.get("tensor_count"), bool)
        or not isinstance(raw.get("tensor_count"), int)
        or raw["tensor_count"] <= 0
        or raw.get("contains_prompt_or_token_text") is not False
    ):
        raise ValueError("E8b private raw tensor-bundle authentication is missing")
    expected_raw_name = f"s9_e8b_raw_tensors.{raw['sha256']}.safetensors"
    if raw.get("path") != expected_raw_name:
        raise ValueError("E8b private raw tensor bundle is not content-addressed")


__all__ = [
    "E8B_SCHEMA_VERSION",
    "E8B_PLAN_SEMANTICS_VERSION",
    "E8B_PLAN_STATUS",
    "E8B_RESULT_SEMANTICS_VERSION",
    "E8B_RUNNING_STATUS",
    "E8B_RESULT_STATUS",
    "E8B_ANALYSIS_SEMANTICS_VERSION",
    "E8B_ANALYSIS_STATUS",
    "E8A_FROZEN_MANIFEST_FILE_SHA256",
    "E8A_FROZEN_MANIFEST_PAYLOAD_SHA256",
    "E8A_FROZEN_PLAN_FILE_SHA256",
    "E8A_FROZEN_PLAN_PAYLOAD_SHA256",
    "E8A_FROZEN_SCIENTIFIC_SNAPSHOT_SHA256",
    "E8B_SEMANTIC_ERRATUM_FILE_SHA256",
    "E8B_AUTHORIZATION_SCOPE",
    "E8B_INPUT_NAMES",
    "E8B_PRIVACY",
    "E8B_RUNTIME_CONTRACT",
    "E8B_NUMERICAL_CONTRACT",
    "E8B_FORWARD_CONTRACT",
    "E8B_OUTPUT_CONTRACT",
    "E8B_SEMANTIC_RESOLUTION",
    "e8a_scientific_snapshot_sha256",
    "validate_e8b_execution_plan",
    "validate_e8b_result",
]
