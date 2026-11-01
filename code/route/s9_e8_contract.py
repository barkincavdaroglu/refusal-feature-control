"""Pure CPU-only contracts for S9 E8a's C-to-D route preregistration.

E8a freezes a route-outcome-fresh replication panel.  The selected S10
validation-expansion prompts already have behavior and target outcomes, so the
contract requires that limitation to be disclosed and never calls the panel a
pristine confirmation set.  The untouched S10 confirmation partition remains
sealed and excluded.

The empirical receiver is the complete layer-32 MLP output at the final prompt
position.  Replacing the same complete output at the penultimate position is a
causal-order/runtime audit only: it is neither an empirical control nor another
sample.  The nominated D feature is a gate diagnostic and is never the patched
receiver tensor.

This module imports no model, tensor, GPU, generation, or runtime code.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from s9_contract import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    require_unique,
    validate_stamped_payload,
)


E8_SCHEMA_VERSION = 1
E8A_MANIFEST_SEMANTICS_VERSION = "s9-e8a-c-to-d-replication-manifest-v1"
E8A_MANIFEST_STATUS = "complete_preoutcome_route_replication_manifest_v1"
E8A_PLAN_SEMANTICS_VERSION = "s9-e8a-c-to-d-route-preregistration-v1"
E8A_PLAN_STATUS = "frozen_cpu_only_route_preregistration_v1"

E8A_EXPECTED_INTENT_IDS = (
    *(f"B{index:02d}" for index in range(15, 20)),
    *(f"H{index:02d}" for index in range(31, 41)),
)
E8A_PRIOR_E7_INTENT_IDS = frozenset({"B01", "B04", "H01", "H03", "H07", "H09"})
E8A_CONFIRMATION_KEY_SET_SHA256 = (
    "fcc7e8899fd86400ee9761ca9adea04a4f542d8f82cd48662226e717b1bfcac5"
)

E8A_NOMINATED_SOURCE_FEATURE = "L31:F49541"
E8A_CONTROL_FEATURE = "L31:F139261"
E8A_RECEIVER_FEATURE = "L32:F50401"
E8A_NOMINATED_DELTA_FORMULA = (
    "relu(e_C @ h_in31(D0) + b_C) - relu(e_C @ h_in31(F0) + b_C)"
)
E8A_CONTROL_DELTA_FORMULA = (
    "sign(delta_C) * l2(delta_C * d_C) / l2(d_control)"
)

E8A_MANIFEST_INPUT_NAMES = (
    "s10_expansion_plan",
    "s10_expansion_analysis",
    "s10_validation_plan",
    "e7b_plan",
    "e7c_result",
    "e7c_interpretation_audit",
    "manifest_builder",
    "e8_contract",
)
E8A_PLAN_INPUT_NAMES = (
    "manifest",
    "plan_builder",
    "e8_contract",
    "base_contract",
    "probe_direction",
    "s10_expansion_plan",
    "s10_expansion_analysis",
    "s10_validation_plan",
    "e7b_plan",
    "e7c_result",
    "e7c_interpretation_audit",
)

E8A_PRIVACY = {
    "prompt_text_in_public_artifacts": False,
    "completion_text_in_public_artifacts": False,
    "token_text_in_public_artifacts": False,
    "raw_receiver_tensors_in_public_artifacts": False,
    "confirmation_prompt_text_read": False,
}

_MANIFEST_AUTHORIZATION = {
    "local_cpu_only": True,
    "gpu_allocation_allowed": False,
    "original_model_outcomes_allowed": False,
    "generation_allowed": False,
    "e7d_allowed": False,
    "fresh_user_confirmation_required": True,
}
_PLAN_AUTHORIZATION = {
    **_MANIFEST_AUTHORIZATION,
    "execution_bundle_frozen": False,
}
_OUTCOME_DISCLOSURE = {
    "freshness_scope": "route_outcomes_only",
    "s10_behavior_and_target_outcomes_already_exist": True,
    "s10_cohort_level_outcomes_known_to_designers": True,
    "record_level_s10_outcomes_used_for_selection": False,
    "e8_route_outcomes_seen_by_builder": False,
    "pristine_model_unseen_confirmation": False,
    "selected_partition": "validation_expansion",
}
_SELECTION_CONTRACT = {
    "fixed_intent_ids": list(E8A_EXPECTED_INTENT_IDS),
    "all_validation_expansion_intents_retained": True,
    "selection_used_behavior_or_target_outcomes": False,
    "selection_used_e7_or_e8_route_outcomes": False,
    "source_frame_id": "D0",
    "target_frame_id": "F0",
    "one_pair_per_intent": True,
    "fixed_full_panel_no_sequential_stopping": True,
}
_EXPECTED_COUNTS = {
    "intent_pairs": 15,
    "harmful_intents": 10,
    "benign_intents": 5,
    "prompt_records": 30,
    "route_arms_per_intent": 2,
    "receiver_positions_per_arm": 2,
    "route_cells": 60,
}
_ROUTE_DEFINITION = {
    "source_hook": "blocks.31.mlp.hook_out",
    "source_token_position": "final_prompt_position",
    "nominated_source_feature": E8A_NOMINATED_SOURCE_FEATURE,
    "nominated_activation_delta_formula": E8A_NOMINATED_DELTA_FORMULA,
    "control_source_feature": E8A_CONTROL_FEATURE,
    "control_activation_delta_formula": E8A_CONTROL_DELTA_FORMULA,
    "control_matching": "exact_applied_residual_l2_with_nominated_sign",
    "receiver_layer": 32,
    "receiver_boundary": "module_output",
    "receiver_hook": "blocks.32.mlp.hook_out",
    "empirical_receiver_token_position": "final_prompt_position",
    "structural_audit_token_position": "penultimate_prompt_position",
    "structural_audit_role": "causal_order_runtime_audit_not_empirical_control_or_sample",
    "receiver_state_contract": "complete_d_model_vector_at_declared_token_position",
    "nominated_receiver_feature": E8A_RECEIVER_FEATURE,
    "receiver_feature_role": "gate_diagnostic_only",
    "receiver_feature_intervened": False,
    "gate_crossing_definition": (
        "baseline_preactivation_lte_zero_and_sender_preactivation_gt_zero"
    ),
    "run_all_cells_before_gate_classification": True,
    "zero_nominated_delta_action": "retain_zero_dose_cell_without_retuning",
}
_ENDPOINTS = {
    "primary": ["next_token_refusal_log_odds"],
    "secondary": ["probe_score"],
    "raw_reported": ["probe_score", "next_token_refusal_log_odds"],
    "generation_executed": False,
}
_ESTIMANDS = {
    "T_S": "YS - Y0",
    "D_R": "YS - YS_R0",
    "B_R": "YS_R0 - Y0",
    "Q_R": "YR_S - Y0",
    "invariant": "T_S = D_R + B_R",
    "paired_D": "D_R[nominated_c] - D_R[equal_l2_control]",
    "paired_Q": "Q_R[nominated_c] - Q_R[equal_l2_control]",
    "ratios_primary": False,
    "absolute_effects_primary": False,
}
_ANALYSIS_CONTRACT = {
    "independent_unit": "intent_id",
    "all_frozen_cells_reported": True,
    "gate_crossing_is_post_intervention_diagnostic": True,
    "gate_crossing_role": "descriptive_predeclared_stratum_only",
    "panel_evaluability_depends_on_gate_crossing": False,
    "primary_population": "10_harmful_intents_final_position_only",
    "primary_endpoint": "next_token_refusal_log_odds",
    "primary_hierarchy": ["paired_D", "paired_Q"],
    "test": "exact_one_sided_intent_level_sign_test",
    "alpha": 0.05,
    "ties_at_or_below_invariant_atol_are_nonwins": True,
    "raw_nominated_medians_must_be_positive": ["D_R", "Q_R"],
    "risk_strata_reported_separately": True,
    "no_population_claim_from_route_cells": True,
    "no_threshold_refit": True,
}
_TECHNICAL_EVALUABILITY = {
    "required_route_cells": 60,
    "required_model_forwards": 180,
    "missing_intents_may_be_dropped": False,
    "hook_fire_count_per_declared_capture": 1,
    "invariant_absolute_tolerance": 1e-7,
    "pre_replacement_max_abs_mismatch": 1e-4,
    "equal_l2_relative_tolerance": 1e-6,
    "penultimate_receiver_state_max_abs_change": 1e-4,
    "penultimate_endpoint_abs_tolerance": 1e-4,
    "penultimate_violation_role": (
        "semantic_or_runtime_failure_not_mechanism_evidence"
    ),
}
_DECISION_CONTRACT = {
    "harmful_intent_ids": [f"H{index:02d}" for index in range(31, 41)],
    "dependence_success": {
        "per_intent": "T_S_C_gt_0_and_D_R_C_gt_0_and_paired_D_gt_0",
        "minimum_wins": 9,
        "denominator": 10,
    },
    "rescue_success": {
        "tested_only_after_dependence_success": True,
        "per_intent": "dependence_win_and_Q_R_C_gt_0_and_paired_Q_gt_0",
        "minimum_wins": 9,
        "denominator": 10,
    },
    "gate_crossing_advancement": {
        "crossing_definition": (
            "baseline_preactivation_lte_zero_and_sender_preactivation_gt_zero"
        ),
        "minimum_harmful_nominated_crossings": 8,
        "minimum_joint_route_wins_among_crossings": 7,
        "changes_panel_evaluability": False,
        "threshold_role": "decision_threshold_not_natural_statistical_boundary",
    },
    "topic_matched_pairs": [
        {
            "harmful_intent_id": f"H{index:02d}",
            "benign_intent_id": f"B{index - 16:02d}",
        }
        for index in range(31, 36)
    ],
    "topic_differential": {
        "paired_D_formula": "paired_D[harmful] - paired_D[benign]",
        "paired_Q_formula": "paired_Q[harmful] - paired_Q[benign]",
        "minimum_positive_paired_D": 5,
        "denominator": 5,
        "paired_Q_role": "corroboration",
        "nominated_edit_l2_ratio_interval": [0.8, 1.25],
        "outside_interval_action": (
            "semantic_differential_not_evaluable_but_raw_results_retained"
        ),
    },
    "equal_l2_comparator_claim_boundary": (
        "magnitude_matched_sender_comparator_not_sender_total_matched_"
        "route_specificity_control"
    ),
    "benign_sparing_claim_allowed": False,
    "behavior_claim_allowed": False,
}
_EXECUTION_BOUNDARY = {
    "scientific_design_frozen": True,
    "execution_bundle_frozen": False,
    "future_execution_stage": "E8b",
    "runner_present": False,
    "runtime_hashes_present": False,
    "requires_fresh_user_confirmation": True,
    "e8b_may_change_scientific_fields": False,
}
_EXCLUDED_ARMS = {
    "module_input": True,
    "graph_low_coverage_receiver": True,
    "joint_receiver_set": True,
    "topology_selection": True,
    "e7d": True,
    "generation": True,
    "backward_pass": True,
}
_EXPECTED_FORWARDS = {
    "d0_f0_baseline_and_activation_caches": 30,
    "sender_patched_worlds": 30,
    "module_output_restoration_and_rescue": 120,
    "total": 180,
    "audit_ceiling": 198,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


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
                raise ValueError(f"Public E8a payload contains forbidden key: {key}")
            _walk_public(item)
    elif isinstance(value, list):
        for item in value:
            _walk_public(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Public E8a payload contains a non-finite value")


def _validate_inputs(value: Any, names: tuple[str, ...], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"E8a {label} input closure differs")
    for name in names:
        record = value[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"E8a {label} input {name} is malformed")
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError(f"E8a {label} input {name} path is missing")
        if not _is_sha256(record.get("sha256")):
            raise ValueError(f"E8a {label} input {name} hash is malformed")


def _validate_confirmation_boundary(value: Any) -> None:
    expected = {
        "partition": "confirmation",
        "sealed": True,
        "prompt_text_read": False,
        "prompt_text_sent_to_model": False,
        "model_outcomes": 0,
        "selected_intent_overlap": [],
        "prompt_key_set_sha256": E8A_CONFIRMATION_KEY_SET_SHA256,
    }
    if value != expected:
        raise ValueError("E8a confirmation partition must remain sealed and excluded")


def _validate_intent_pairs(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 15:
        raise ValueError("E8a manifest requires exactly 15 intent pairs")
    pairs: list[Mapping[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("E8a intent pair is malformed")
        pairs.append(row)
    require_unique((row.get("pair_id") for row in pairs), "E8a pair ids")
    require_unique((row.get("intent_id") for row in pairs), "E8a intent ids")

    intent_ids = {row.get("intent_id") for row in pairs}
    if intent_ids != set(E8A_EXPECTED_INTENT_IDS):
        overlap = sorted(intent_ids & E8A_PRIOR_E7_INTENT_IDS)
        raise ValueError(
            "E8a manifest intent identities differ or overlap prior E7 intents: "
            f"{overlap}"
        )
    risk_counts = Counter(row.get("risk_class") for row in pairs)
    if risk_counts != Counter({"harmful": 10, "benign": 5}):
        raise ValueError("E8a intent risk balance differs")
    categories = [row.get("category") for row in pairs]
    if any(not isinstance(item, str) or not item for item in categories):
        raise ValueError("E8a intent category is missing")
    require_unique(categories, "E8a intent categories")

    prompt_keys: list[Any] = []
    prompt_hashes: list[Any] = []
    core_hashes: list[Any] = []
    for row in pairs:
        intent_id = row["intent_id"]
        if row.get("pair_id") != f"e8a:{intent_id}:d0_to_f0":
            raise ValueError(f"E8a {intent_id} pair id differs")
        if row.get("partition") != "validation_expansion":
            raise ValueError(
                "E8a may use only validation_expansion; confirmation is sealed"
            )
        if not _is_sha256(row.get("core_sha256")):
            raise ValueError(f"E8a {intent_id} core hash is malformed")
        core_hashes.append(row["core_sha256"])
        for side, frame in (("source_prompt", "D0"), ("target_prompt", "F0")):
            prompt = row.get(side)
            if not isinstance(prompt, Mapping) or set(prompt) != {
                "prompt_key",
                "frame_id",
                "prompt_sha256",
            }:
                raise ValueError(f"E8a {intent_id} {side} metadata is malformed")
            expected_key = f"S10BX-{intent_id}-{frame}"
            if prompt.get("frame_id") != frame or prompt.get("prompt_key") != expected_key:
                raise ValueError(f"E8a {intent_id} must pair D0 source with F0 target")
            if not _is_sha256(prompt.get("prompt_sha256")):
                raise ValueError(f"E8a {intent_id} {side} hash is malformed")
            prompt_keys.append(prompt["prompt_key"])
            prompt_hashes.append(prompt["prompt_sha256"])
    require_unique(prompt_keys, "E8a selected prompt keys")
    require_unique(prompt_hashes, "E8a selected prompt hashes")
    require_unique(core_hashes, "E8a intent core hashes")
    return pairs


def validate_e8a_manifest(payload: Mapping[str, Any]) -> None:
    """Validate E8a's public, route-outcome-fresh intent manifest."""

    validate_stamped_payload(payload)
    _walk_public(payload)
    if payload.get("schema_version") != E8_SCHEMA_VERSION:
        raise ValueError("E8a manifest schema differs")
    if payload.get("analysis_semantics_version") != E8A_MANIFEST_SEMANTICS_VERSION:
        raise ValueError("E8a manifest semantics differ")
    if payload.get("status") != E8A_MANIFEST_STATUS:
        raise ValueError("E8a manifest status differs")
    if payload.get("mode") != "prospective_route_replication":
        raise ValueError("E8a manifest mode differs")
    if payload.get("research_stage") != "understanding_testing_hypotheses":
        raise ValueError("E8a research stage differs")
    if not isinstance(payload.get("objective"), str) or not payload["objective"]:
        raise ValueError("E8a objective is missing")
    if payload.get("authorization") != _MANIFEST_AUTHORIZATION:
        raise ValueError("E8a CPU-only authorization boundary differs")
    if payload.get("outcome_disclosure") != _OUTCOME_DISCLOSURE:
        raise ValueError(
            "E8a outcome disclosure must say route outcomes only, with prior S10 outcomes"
        )
    _validate_confirmation_boundary(payload.get("confirmation_boundary"))
    if payload.get("selection_contract") != _SELECTION_CONTRACT:
        raise ValueError("E8a fixed selection contract differs")
    _validate_intent_pairs(payload.get("intent_pairs"))
    if payload.get("expected_counts") != _EXPECTED_COUNTS:
        raise ValueError("E8a manifest expected counts differ")

    private_source = payload.get("private_prompt_source")
    if not isinstance(private_source, Mapping):
        raise ValueError("E8a private prompt source is missing")
    if not isinstance(private_source.get("path"), str) or not private_source["path"].endswith(
        "s10_phase_b_validation_expansion_prompts.csv"
    ):
        raise ValueError("E8a private prompt source must be the validation expansion")
    if (
        not _is_sha256(private_source.get("sha256"))
        or private_source.get("public") is not False
        or private_source.get("opened_by_e8a_builder") is not False
        or private_source.get("hash_copied_from")
        != "authenticated_s10_expansion_plan_input"
    ):
        raise ValueError("E8a private prompt source authentication differs")
    _validate_inputs(payload.get("inputs"), E8A_MANIFEST_INPUT_NAMES, "manifest")
    if payload.get("privacy") != E8A_PRIVACY:
        raise ValueError("E8a manifest privacy contract differs")


def _validate_route_cells(value: Any, pairs: list[Mapping[str, Any]]) -> None:
    if not isinstance(value, list) or len(value) != 60:
        raise ValueError("E8a plan must contain all 60 fixed route cells")
    cells: list[Mapping[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("E8a route cell is malformed")
        cells.append(row)
    require_unique((row.get("cell_id") for row in cells), "E8a route cell ids")

    pairs_by_id = {row["pair_id"]: row for row in pairs}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cells:
        if "source_activation_delta" in row or "control_activation_delta" in row:
            raise ValueError(
                "E8a source deltas must remain formula-defined before model outcomes"
            )
        pair_id = row.get("pair_id")
        if pair_id not in pairs_by_id:
            raise ValueError("E8a route cell references an unknown intent pair")
        pair = pairs_by_id[pair_id]
        if (
            row.get("intent_id") != pair["intent_id"]
            or row.get("risk_class") != pair["risk_class"]
            or row.get("source_prompt_key") != pair["source_prompt"]["prompt_key"]
            or row.get("target_prompt_key") != pair["target_prompt"]["prompt_key"]
        ):
            raise ValueError("E8a route cell prompt or intent metadata differs")

        arm = row.get("arm")
        if arm == "nominated_c":
            source_feature = E8A_NOMINATED_SOURCE_FEATURE
            policy = "natural_counterfactual_activation_delta_d0_minus_f0"
        elif arm == "equal_l2_control":
            source_feature = E8A_CONTROL_FEATURE
            policy = "equal_l2_residual_direction_control_matched_to_nominated_c"
        else:
            raise ValueError(f"E8a route arm differs: {arm!r}")
        if row.get("source_feature") != source_feature or row.get("source_edit_policy") != policy:
            raise ValueError("E8a route arm source operation differs")
        dose = row.get("dose_multiplier")
        if isinstance(dose, bool) or not isinstance(dose, (int, float)) or float(dose) != 1.0:
            raise ValueError("E8a dose multiplier must remain fixed at 1.0")

        if (
            row.get("receiver_layer") != 32
            or row.get("receiver_boundary") != "module_output"
            or row.get("receiver_hook") != "blocks.32.mlp.hook_out"
        ):
            raise ValueError("E8a controls must use the same complete receiver cut")
        if row.get("nominated_receiver_feature") != E8A_RECEIVER_FEATURE:
            raise ValueError("E8a nominated receiver feature differs")
        if row.get("receiver_feature_intervened") is not False:
            raise ValueError("E8a receiver feature must remain diagnostic only")

        position = row.get("receiver_token_position")
        if position == "final_prompt_position":
            expected_role = "empirical_receiver"
            expected_inclusion = True
        elif position == "penultimate_prompt_position":
            expected_role = "causal_order_runtime_audit"
            expected_inclusion = False
        else:
            raise ValueError("E8a receiver token position differs")
        if (
            row.get("receiver_position_role") != expected_role
            or row.get("included_in_empirical_analysis") is not expected_inclusion
        ):
            raise ValueError(
                "E8a penultimate receiver is a structural audit, never an empirical control or sample"
            )
        expected_id = f"e8a:{pair['intent_id']}:{arm}:{position}"
        if row.get("cell_id") != expected_id:
            raise ValueError("E8a route cell identity differs")
        grouped[pair_id].append(row)

    expected_combinations = {
        (arm, position)
        for arm in ("nominated_c", "equal_l2_control")
        for position in ("final_prompt_position", "penultimate_prompt_position")
    }
    for pair_id, rows in grouped.items():
        combinations = {
            (row.get("arm"), row.get("receiver_token_position")) for row in rows
        }
        if combinations != expected_combinations or len(rows) != 4:
            raise ValueError(f"E8a {pair_id} route-cell closure differs")
    if set(grouped) != set(pairs_by_id):
        raise ValueError("E8a route cells do not cover every frozen intent")


def validate_e8a_plan(payload: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate E8a's CPU-only preregistration against its frozen manifest."""

    validate_e8a_manifest(manifest)
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E8_SCHEMA_VERSION:
        raise ValueError("E8a plan schema differs")
    if payload.get("analysis_semantics_version") != E8A_PLAN_SEMANTICS_VERSION:
        raise ValueError("E8a plan semantics differ")
    if payload.get("status") != E8A_PLAN_STATUS:
        raise ValueError("E8a plan status differs")
    if payload.get("mode") != "prospective_route_replication":
        raise ValueError("E8a plan mode differs")
    if payload.get("research_stage") != "understanding_testing_hypotheses":
        raise ValueError("E8a plan research stage differs")
    if not isinstance(payload.get("objective"), str) or not payload["objective"]:
        raise ValueError("E8a plan objective is missing")
    if payload.get("manifest_payload_sha256") != manifest.get("payload_sha256"):
        raise ValueError("E8a plan references another manifest")
    if not _is_sha256(payload.get("manifest_file_sha256")):
        raise ValueError("E8a manifest file authentication is missing")
    if payload.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("E8a model identity differs")
    if payload.get("transcoder") != {
        "id": TRANSCODER_ID,
        "revision": TRANSCODER_REVISION,
    }:
        raise ValueError("E8a transcoder identity differs")
    if payload.get("authorization") != _PLAN_AUTHORIZATION:
        raise ValueError("E8a plan authorization boundary differs")
    if payload.get("outcome_disclosure") != manifest.get("outcome_disclosure"):
        raise ValueError("E8a plan outcome disclosure differs from the manifest")
    if payload.get("confirmation_boundary") != manifest.get("confirmation_boundary"):
        raise ValueError("E8a plan confirmation boundary differs from the manifest")
    _validate_confirmation_boundary(payload.get("confirmation_boundary"))
    if payload.get("route_definition") != _ROUTE_DEFINITION:
        raise ValueError(
            "E8a route definition must run all cells before gate classification and preserve the structural audit"
        )
    if payload.get("intent_pairs") != manifest.get("intent_pairs"):
        raise ValueError("E8a plan intent pairs differ from the manifest")
    pairs = _validate_intent_pairs(payload.get("intent_pairs"))
    _validate_route_cells(payload.get("route_cells"), pairs)
    if payload.get("endpoints") != _ENDPOINTS:
        raise ValueError("E8a endpoint contract differs")
    if payload.get("estimands") != _ESTIMANDS:
        raise ValueError("E8a signed paired estimands differ")
    if payload.get("analysis_contract") != _ANALYSIS_CONTRACT:
        raise ValueError("E8a intent-level signed paired analysis contract differs")
    if payload.get("technical_evaluability") != _TECHNICAL_EVALUABILITY:
        raise ValueError("E8a technical evaluability contract differs")
    if payload.get("decision_contract") != _DECISION_CONTRACT:
        raise ValueError(
            "E8a decision contract must preserve the 9 of 10 hierarchy and dose boundary"
        )
    if payload.get("execution_boundary") != _EXECUTION_BOUNDARY:
        raise ValueError(
            "E8a execution boundary keeps E8b unfrozen and scientific fields immutable"
        )
    if payload.get("excluded_arms") != _EXCLUDED_ARMS:
        raise ValueError("E8a excluded execution arms differ")
    if payload.get("expected_counts") != manifest.get("expected_counts"):
        raise ValueError("E8a plan expected counts differ from the manifest")
    if payload.get("expected_counts") != _EXPECTED_COUNTS:
        raise ValueError("E8a plan expected counts differ")
    if payload.get("expected_forwards") != _EXPECTED_FORWARDS:
        raise ValueError("E8a forward accounting must close exactly at 180")
    _validate_inputs(payload.get("inputs"), E8A_PLAN_INPUT_NAMES, "plan")
    if payload.get("privacy") != E8A_PRIVACY:
        raise ValueError("E8a plan privacy contract differs")
    _walk_public(payload)


__all__ = [
    "E8_SCHEMA_VERSION",
    "E8A_MANIFEST_SEMANTICS_VERSION",
    "E8A_MANIFEST_STATUS",
    "E8A_PLAN_SEMANTICS_VERSION",
    "E8A_PLAN_STATUS",
    "E8A_EXPECTED_INTENT_IDS",
    "E8A_PRIOR_E7_INTENT_IDS",
    "E8A_CONFIRMATION_KEY_SET_SHA256",
    "E8A_NOMINATED_SOURCE_FEATURE",
    "E8A_CONTROL_FEATURE",
    "E8A_RECEIVER_FEATURE",
    "E8A_NOMINATED_DELTA_FORMULA",
    "E8A_CONTROL_DELTA_FORMULA",
    "E8A_MANIFEST_INPUT_NAMES",
    "E8A_PLAN_INPUT_NAMES",
    "E8A_PRIVACY",
    "validate_e8a_manifest",
    "validate_e8a_plan",
]
