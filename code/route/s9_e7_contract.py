"""Pure, public-safe contracts for S9 E7 receiver-route reliability.

The contract deliberately keeps complete MLP-output restoration separate from
the MLP-input calibration and treats a receiver feature only as a topology
marker.  It imports no model runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from s9_contract import (
    METRICS,
    MODEL_ID,
    MODEL_REVISION,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    parse_feature,
    require_unique,
    validate_stamped_payload,
)


E7_SCHEMA_VERSION = 1
E7B_MANIFEST_SEMANTICS_VERSION = "s9-e7b-receiver-route-panel-v1"
E7B_MANIFEST_STATUS = "complete_preoutcome_route_panel_manifest_v1"
E7B_PLAN_SEMANTICS_VERSION = "s9-e7b-exact-receiver-route-plan-v1"
E7B_PLAN_STATUS = "frozen_preoutcome_route_execution_plan_v1"
E7C_SEMANTICS_VERSION = "s9-e7c-exact-original-model-route-outcomes-v1"
E7C_RUNNING_STATUS = "running_exact_original_model_route_outcomes_v1"
E7C_RESULT_STATUS = "complete_exact_original_model_route_outcomes_v1"
E7_BOUNDARIES = ("module_output", "module_input")
E7_PRIMARY_BOUNDARY = "module_output"
E7_INPUT_NAMES = (
    "prompts",
    "annotations",
    "probe_direction",
    "candidate_manifest",
    "manifest_builder",
    "topology_source",
    "e7a_semantics",
    "e7_contract",
    "plan_builder",
    "runner",
    "colab_runner",
    "runtime_probe",
    "analyzer",
    "runtime_installer",
    "base_runner",
    "base_contract",
)
E7_PRIVACY = {
    "prompt_text_in_public_artifacts": False,
    "completion_text_in_public_artifacts": False,
    "token_text_in_public_artifacts": False,
    "raw_receiver_tensors_in_public_artifacts": False,
}
INVARIANT_ATOL = 1e-7


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
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
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                raise ValueError(f"Public E7 payload contains forbidden key: {key}")
            _walk_public(item)
    elif isinstance(value, list):
        for item in value:
            _walk_public(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Public E7 payload contains a non-finite value")


def _primary_cells(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    mechanics = payload.get("mechanics_cells")
    topology = payload.get("topology_cells")
    if not isinstance(mechanics, list) or not isinstance(topology, list):
        raise ValueError("E7b primary cell tables are missing")
    return [*mechanics, *topology]


def validate_e7b_manifest(payload: Mapping[str, Any]) -> None:
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E7_SCHEMA_VERSION:
        raise ValueError("E7b manifest schema differs")
    if payload.get("analysis_semantics_version") != E7B_MANIFEST_SEMANTICS_VERSION:
        raise ValueError("E7b manifest semantics differ")
    if payload.get("status") != E7B_MANIFEST_STATUS:
        raise ValueError("E7b manifest is not the frozen pre-outcome panel")
    if payload.get("mode") != "adaptive_exploration":
        raise ValueError("E7 must remain adaptive exploration")
    if payload.get("authorization", {}).get("explicit_user_confirmation_received") is not True:
        raise ValueError("E7c authorization is absent")
    if payload.get("authorization", {}).get("e7_outcomes_seen_by_builder") is not False:
        raise ValueError("E7b selection must precede E7 outcomes")
    if payload.get("model", {}).get("id") != MODEL_ID or payload.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("E7 model identity differs")
    if payload.get("transcoder", {}).get("id") != TRANSCODER_ID or payload.get("transcoder", {}).get("revision") != TRANSCODER_REVISION:
        raise ValueError("E7 transcoder identity differs")
    if payload.get("privacy") != {
        "prompt_text_in_public_artifacts": False,
        "completion_text_in_public_artifacts": False,
        "token_text_in_public_artifacts": False,
    }:
        raise ValueError("E7b manifest privacy contract differs")

    cells = _primary_cells(payload)
    counts = payload.get("expected_counts", {})
    expected = {
        "mechanics_primary_cells": len(payload["mechanics_cells"]),
        "topology_primary_cells": len(payload["topology_cells"]),
        "primary_cells": len(cells),
        "input_calibration_cells": len(payload.get("input_calibration_cell_ids", [])),
        "low_coverage_controls": len(payload.get("low_coverage_controls", [])),
        "joint_receiver_sets": len(payload.get("joint_receiver_sets", [])),
        "unique_source_edits": len({row.get("source_edit_id") for row in cells}),
        "unique_prompts": len(payload.get("prompt_records", [])),
    }
    if counts != expected:
        raise ValueError(f"E7b manifest counts differ: {counts} != {expected}")
    if expected != {
        "mechanics_primary_cells": 30,
        "topology_primary_cells": 18,
        "primary_cells": 48,
        "input_calibration_cells": 21,
        "low_coverage_controls": 11,
        "joint_receiver_sets": 2,
        "unique_source_edits": 48,
        "unique_prompts": 11,
    }:
        raise ValueError("E7b v1 panel size differs")
    require_unique((row.get("cell_id") for row in cells), "E7b cell ids")
    require_unique((row.get("route_signature") for row in cells), "E7b module route signatures")
    require_unique((row.get("prompt_key") for row in payload["prompt_records"]), "E7b prompt keys")
    cell_ids = {row["cell_id"] for row in cells}
    for row in cells:
        if row.get("receiver_boundary") != E7_PRIMARY_BOUNDARY:
            raise ValueError("Every E7b primary cell must use a complete module output")
        source_layer, _ = parse_feature(row.get("source_feature"))
        receiver_layer = row.get("receiver_layer")
        if isinstance(receiver_layer, bool) or not isinstance(receiver_layer, int):
            raise ValueError("E7b receiver layer must be an integer")
        if source_layer >= receiver_layer:
            raise ValueError("E7b sender must precede its receiver")
        _finite(row.get("source_activation_delta"), f"{row['cell_id']}.source_activation_delta")
        _finite(row.get("source_edit_l2"), f"{row['cell_id']}.source_edit_l2")
    input_ids = payload.get("input_calibration_cell_ids")
    if not isinstance(input_ids, list) or not set(input_ids) <= cell_ids:
        raise ValueError("E7b input calibrations do not reference primary cells")
    require_unique(input_ids, "E7b input calibration ids")
    for row in payload.get("low_coverage_controls", []):
        if row.get("source_cell_id") not in cell_ids or row.get("receiver_boundary") != E7_PRIMARY_BOUNDARY:
            raise ValueError("E7b low-coverage control differs")
    require_unique(
        (row.get("control_id") for row in payload.get("low_coverage_controls", [])),
        "E7b low-coverage control ids",
    )
    for row in payload.get("joint_receiver_sets", []):
        layers = row.get("receiver_layers")
        if row.get("source_cell_id") not in cell_ids or row.get("receiver_boundary") != E7_PRIMARY_BOUNDARY:
            raise ValueError("E7b joint receiver set differs")
        if not isinstance(layers, list) or len(layers) < 2 or layers != sorted(set(layers)):
            raise ValueError("E7b joint receiver layers differ")
    require_unique(
        (row.get("cell_id") for row in payload.get("joint_receiver_sets", [])),
        "E7b joint receiver-set ids",
    )
    forwards = payload.get("expected_forwards", {})
    components = [value for key, value in forwards.items() if key not in {"total", "audit_ceiling"}]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in components):
        raise ValueError("E7b forward accounting is malformed")
    if sum(components) != forwards.get("total") or forwards.get("total") != 223:
        raise ValueError("E7b forward accounting does not close at 223")
    if forwards.get("audit_ceiling") != 246:
        raise ValueError("E7b audit ceiling differs")
    _walk_public(payload)


def validate_e7b_plan(payload: Mapping[str, Any]) -> None:
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E7_SCHEMA_VERSION:
        raise ValueError("E7b plan schema differs")
    if payload.get("analysis_semantics_version") != E7B_PLAN_SEMANTICS_VERSION:
        raise ValueError("E7b plan semantics differ")
    if payload.get("status") != E7B_PLAN_STATUS:
        raise ValueError("E7b plan is not frozen")
    if payload.get("mode") != "adaptive_exploration":
        raise ValueError("E7b plan mode differs")
    if payload.get("authorization", {}).get("explicit_user_confirmation_received") is not True:
        raise ValueError("E7c is not authorized")
    if payload.get("authorization", {}).get("gpu_allocation_allowed") is not True:
        raise ValueError("E7c GPU allocation is not authorized")
    if payload.get("selection_used_e7_outcomes") is not False:
        raise ValueError("E7b execution plan must precede E7 outcomes")
    if payload.get("model", {}).get("id") != MODEL_ID or payload.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("E7b plan model identity differs")
    if payload.get("transcoder", {}).get("id") != TRANSCODER_ID or payload.get("transcoder", {}).get("revision") != TRANSCODER_REVISION:
        raise ValueError("E7b plan transcoder identity differs")
    if set(payload.get("inputs", {})) != set(E7_INPUT_NAMES):
        raise ValueError("E7b semantic-input closure differs")
    if payload.get("privacy") != E7_PRIVACY:
        raise ValueError("E7b plan privacy differs")
    endpoints = payload.get("endpoints", {})
    if endpoints.get("primary") != list(METRICS):
        raise ValueError("E7b primary endpoints differ")
    if endpoints.get("generation_executed") is not False:
        raise ValueError("E7b must not execute generation")
    method = payload.get("exact_method", {})
    if method.get("primary_receiver_boundary") != "blocks.L.mlp.hook_out":
        raise ValueError("E7b primary receiver hook differs")
    if method.get("calibration_receiver_boundary") != "blocks.L.mlp.hook_in":
        raise ValueError("E7b calibration receiver hook differs")
    if method.get("receiver_feature_intervened") is not False:
        raise ValueError("E7b must intervene on a complete module, not one feature")
    if payload.get("expected_counts", {}).get("primary_cells") != 48:
        raise ValueError("E7b plan primary cell count differs")
    if payload.get("expected_forwards", {}).get("total") != 223:
        raise ValueError("E7b plan forward count differs")
    features = payload.get("candidate_features")
    if not isinstance(features, list) or not features:
        raise ValueError("E7b feature closure is missing")
    require_unique((row.get("feature") for row in features), "E7b candidate features")
    for row in features:
        layer, index = parse_feature(row.get("feature"))
        if row.get("layer") != layer or row.get("feature_index") != index:
            raise ValueError("E7b feature coordinate differs")
    _walk_public(payload)


def _validate_endpoint_record(record: Mapping[str, Any], name: str) -> None:
    required = ("Y0", "YS", "YS_R0", "YR_S", "T_S", "D_R", "B_R", "Q_R", "invariant_error")
    values = {key: _finite(record.get(key), f"{name}.{key}") for key in required}
    if not math.isclose(values["T_S"], values["YS"] - values["Y0"], rel_tol=0.0, abs_tol=INVARIANT_ATOL):
        raise ValueError(f"{name}: T_S differs")
    if not math.isclose(values["D_R"], values["YS"] - values["YS_R0"], rel_tol=0.0, abs_tol=INVARIANT_ATOL):
        raise ValueError(f"{name}: D_R differs")
    if not math.isclose(values["B_R"], values["YS_R0"] - values["Y0"], rel_tol=0.0, abs_tol=INVARIANT_ATOL):
        raise ValueError(f"{name}: B_R differs")
    if not math.isclose(values["Q_R"], values["YR_S"] - values["Y0"], rel_tol=0.0, abs_tol=INVARIANT_ATOL):
        raise ValueError(f"{name}: Q_R differs")
    expected_error = values["T_S"] - values["D_R"] - values["B_R"]
    if not math.isclose(values["invariant_error"], expected_error, rel_tol=0.0, abs_tol=INVARIANT_ATOL):
        raise ValueError(f"{name}: recorded invariant error differs")
    if abs(values["invariant_error"]) > INVARIANT_ATOL:
        raise ValueError(f"{name}: T_S = D_R + B_R invariant failed")


def _validate_route_rows(rows: Any, expected: int, name: str) -> None:
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"E7c {name} row count differs")
    require_unique((row.get("result_id") for row in rows), f"E7c {name} result ids")
    for row in rows:
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(METRICS):
            raise ValueError(f"E7c {name} endpoints differ")
        for metric in METRICS:
            _validate_endpoint_record(endpoints[metric], f"{row.get('result_id')}.{metric}")


def validate_e7c_result(payload: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_e7b_plan(plan)
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E7_SCHEMA_VERSION:
        raise ValueError("E7c result schema differs")
    if payload.get("analysis_semantics_version") != E7C_SEMANTICS_VERSION:
        raise ValueError("E7c result semantics differ")
    if payload.get("status") != E7C_RESULT_STATUS:
        raise ValueError("E7c result is incomplete")
    if payload.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("E7c result references another plan")
    if payload.get("privacy") != E7_PRIVACY:
        raise ValueError("E7c result privacy differs")
    if payload.get("generation_executed") is not False:
        raise ValueError("E7c generation boundary differs")
    counts = plan["expected_counts"]
    if len(payload.get("prompt_baselines", [])) != counts["unique_prompts"]:
        raise ValueError("E7c prompt baseline count differs")
    if len(payload.get("sender_worlds", [])) != counts["unique_source_edits"]:
        raise ValueError("E7c sender-world count differs")
    _validate_route_rows(payload.get("primary_route_cells"), counts["primary_cells"], "primary")
    _validate_route_rows(payload.get("input_calibrations"), counts["input_calibration_cells"], "input calibration")
    _validate_route_rows(payload.get("low_coverage_controls"), counts["low_coverage_controls"], "low-coverage control")
    _validate_route_rows(payload.get("joint_receiver_sets"), counts["joint_receiver_sets"], "joint receiver set")
    actual = payload.get("forward_accounting", {})
    if actual.get("completed") != plan["expected_forwards"]["total"]:
        raise ValueError("E7c completed forward count differs")
    if actual.get("audit_ceiling_exceeded") is not False:
        raise ValueError("E7c audit ceiling was exceeded")
    raw = payload.get("private_raw_tensor_bundle")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("sha256"), str) or raw.get("public") is not False:
        raise ValueError("E7c private tensor-bundle authentication is missing")
    _walk_public(payload)
