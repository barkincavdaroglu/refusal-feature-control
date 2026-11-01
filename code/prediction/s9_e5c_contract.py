"""Pure, public-safe contracts for S9 E5c's two-stage topology test."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from s9_contract import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    build_directions,
    canonical_sha256,
    parse_feature,
    require_unique,
    validate_stamped_payload,
)


E5C_SCHEMA_VERSION = 1
E5C_POOL_SEMANTICS_VERSION = "s9-e5c-graph-screen-pool-v1"
E5C_POOL_STATUS = "complete_graph_only_pool_before_e5c_model_screen"
E5C_SCREEN_SEMANTICS_VERSION = "s9-e5c-original-model-atp-screen-v1"
E5C_SCREEN_PLAN_STATUS = "screen_plan_before_e5c_atp_or_exact_outcomes"
E5C_SCREEN_RESULT_STATUS = "complete_atp_screen_without_exact_receiver_outcomes"
E5C_EXACT_PANEL_SEMANTICS_VERSION = "s9-e5c-disagreement-panel-v1"
E5C_EXACT_PANEL_STATUS = "complete_predictor_only_panel_before_exact_outcomes"
E5C_EXACT_SEMANTICS_VERSION = "s9-e5c-exact-receiver-adjudication-v1"
E5C_EXACT_RESULT_STATUS = "complete_exact_receiver_adjudication"
E5C_ORDER_ATOL = 1e-12

E5C_PAIR_SPECS = (
    {
        "pair_id": "flip_h07_d0_f0",
        "pair_type": "harmful_behavior_flip",
        "a": "LC-H07-D0",
        "b": "LC-H07-F0",
    },
    {
        "pair_id": "same_h01_d0_f1",
        "pair_type": "harmful_same_behavior",
        "a": "LC-H01-D0",
        "b": "LC-H01-F1",
    },
    {
        "pair_id": "benign_b01_d0_f0",
        "pair_type": "benign_same_behavior",
        "a": "LC-B01-D0",
        "b": "LC-B01-F0",
    },
)
E5C_DIRECTIONS = tuple(build_directions(E5C_PAIR_SPECS))
E5C_GRAPH_IDS = tuple(
    sorted({row[side] for row in E5C_PAIR_SPECS for side in ("a", "b")})
)
E5C_POOL_ROLES = (
    "anchor_high",
    "middle_same_source",
    "weak_same_source",
    "weak_same_receiver",
    "edge_permuted_same_receiver",
)
E5C_RECEIVER_BANDS = ("early", "middle", "late")
E5C_CONTROL_FAMILIES = (
    "same_source",
    "same_receiver",
    "edge_permuted",
)
E5C_SCREEN_INPUT_NAMES = (
    "prompts",
    "annotations",
    "pool_manifest",
    "transcoder_row_bundle",
    "transcoder_row_manifest",
    "runtime_reference_plan",
    "runner",
    "base_runner",
    "feature_path_runner",
    "base_contract",
    "path_contract",
    "e5c_contract",
    "e5b_contract",
    "pool_selector",
    "base_pool_selector",
    "pool_builder",
    "range_extractor",
    "range_helper",
    "screen_builder",
    "colab_runner",
    "runtime_probe",
    "runtime_installer",
)
E5C_EXACT_INPUT_NAMES = E5C_SCREEN_INPUT_NAMES + (
    "screen_plan",
    "screen_result",
    "screen_contract",
    "exact_contract",
    "exact_builder",
    "exact_runner",
    "exact_colab_runner",
    "exact_runtime_probe",
    "exact_analyzer",
)
E5C_PRIVACY = {
    "prompt_text_in_public_artifacts": False,
    "completion_text_in_public_artifacts": False,
    "token_text_in_public_artifacts": False,
}


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
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                raise ValueError(f"Public E5c payload contains forbidden key: {key}")
            _walk_public(item)
    elif isinstance(value, list):
        for item in value:
            _walk_public(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Public E5c payload contains a non-finite value")


def candidate_by_id(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    panels = payload.get("graph_panels")
    if not isinstance(panels, list):
        raise ValueError("E5c graph_panels must be a list")
    rows = [row for panel in panels for row in panel.get("candidates", [])]
    require_unique((row.get("candidate_id") for row in rows), "E5c candidate ids")
    return {row["candidate_id"]: row for row in rows}


def validate_e5c_pool_manifest(payload: Mapping[str, Any]) -> None:
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E5C_SCHEMA_VERSION:
        raise ValueError("E5c pool schema differs")
    if payload.get("analysis_semantics_version") != E5C_POOL_SEMANTICS_VERSION:
        raise ValueError("E5c pool semantics differ")
    if payload.get("status") != E5C_POOL_STATUS:
        raise ValueError("E5c pool status differs")
    if payload.get("mode") != "adaptive_exploration":
        raise ValueError("E5c must remain adaptive exploration")
    if payload.get("pair_specs") != list(E5C_PAIR_SPECS):
        raise ValueError("E5c pair specifications differ")
    if payload.get("directions") != list(E5C_DIRECTIONS):
        raise ValueError("E5c direction table differs")
    if payload.get("selection_used_atp_or_exact_outcomes") is not False:
        raise ValueError("E5c graph pool must precede AtP and exact outcomes")
    if payload.get("privacy") != E5C_PRIVACY:
        raise ValueError("E5c privacy contract differs")

    panels = payload.get("graph_panels")
    if not isinstance(panels, list) or len(panels) != len(E5C_GRAPH_IDS):
        raise ValueError("E5c graph-panel count differs")
    if {panel.get("graph_id") for panel in panels} != set(E5C_GRAPH_IDS):
        raise ValueError("E5c target-graph identities differ")
    all_candidates: list[Mapping[str, Any]] = []
    for panel in panels:
        candidates = panel.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 45:
            raise ValueError(f"{panel.get('graph_id')}: E5c pool must contain 45 candidates")
        by_quartet: dict[str, list[Mapping[str, Any]]] = {}
        for row in candidates:
            by_quartet.setdefault(row.get("quartet_id"), []).append(row)
            for field in ("candidate_id", "graph_id", "quartet_id", "pool_role"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"E5c candidate lacks {field}")
            if row["graph_id"] != panel["graph_id"]:
                raise ValueError("E5c candidate graph identity differs")
            if row["pool_role"] not in E5C_POOL_ROLES:
                raise ValueError("E5c pool role differs")
            source_layer, source_index = parse_feature(row.get("source_feature"))
            receiver_layer, receiver_index = parse_feature(row.get("receiver_feature"))
            if source_layer >= receiver_layer:
                raise ValueError("E5c candidate is not forward ordered")
            expected = {
                "source_layer": source_layer,
                "source_feature_index": source_index,
                "receiver_layer": receiver_layer,
                "receiver_feature_index": receiver_index,
                "layer_gap": receiver_layer - source_layer,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("E5c candidate feature coordinates differ")
            for field in (
                "source_reference_activation",
                "receiver_reference_activation",
                "source_node_mass",
                "receiver_node_mass",
                "raw_signed_edge_weight",
                "normalized_absolute_edge_weight",
                "edge_influence",
            ):
                _finite(row.get(field), f"{row['candidate_id']}.{field}")
            if row.get("anchor_candidate_id") is None:
                raise ValueError("E5c candidate lacks anchor identity")
            all_candidates.append(row)
        if len(by_quartet) != 9:
            raise ValueError("E5c graph panel must contain nine structural quartets")
        for quartet, rows in by_quartet.items():
            roles = Counter(row["pool_role"] for row in rows)
            if roles != Counter(E5C_POOL_ROLES):
                raise ValueError(f"E5c quartet {quartet} role grid differs")
            anchor = next(row for row in rows if row["pool_role"] == "anchor_high")
            if any(row["anchor_candidate_id"] != anchor["candidate_id"] for row in rows):
                raise ValueError(f"E5c quartet {quartet} anchor links differ")
    require_unique((row["candidate_id"] for row in all_candidates), "E5c candidate ids")
    if payload.get("expected_candidates") != len(all_candidates):
        raise ValueError("E5c expected candidate count differs")

    features = payload.get("candidate_features")
    if not isinstance(features, list) or not features:
        raise ValueError("E5c candidate feature closure is missing")
    require_unique((row.get("feature") for row in features), "E5c candidate features")
    for row in features:
        layer, index = parse_feature(row.get("feature"))
        if row.get("layer") != layer or row.get("feature_index") != index:
            raise ValueError("E5c candidate feature record differs")
    expected_features = {
        row[field] for row in all_candidates for field in ("source_feature", "receiver_feature")
    }
    if {row["feature"] for row in features} != expected_features:
        raise ValueError("E5c feature closure differs from candidate endpoints")
    if payload.get("expected_unique_features") != len(features):
        raise ValueError("E5c unique-feature count differs")
    _walk_public(payload)


def validate_e5c_screen_plan(payload: Mapping[str, Any]) -> None:
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E5C_SCHEMA_VERSION:
        raise ValueError("E5c screen-plan schema differs")
    if payload.get("analysis_semantics_version") != E5C_SCREEN_SEMANTICS_VERSION:
        raise ValueError("E5c screen-plan semantics differ")
    if payload.get("status") != E5C_SCREEN_PLAN_STATUS:
        raise ValueError("E5c screen-plan status differs")
    if payload.get("model", {}).get("id") != MODEL_ID or payload.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("E5c screen model identity differs")
    if payload.get("transcoder", {}).get("id") != TRANSCODER_ID or payload.get("transcoder", {}).get("revision") != TRANSCODER_REVISION:
        raise ValueError("E5c screen transcoder identity differs")
    if payload.get("pair_specs") != list(E5C_PAIR_SPECS) or payload.get("directions") != list(E5C_DIRECTIONS):
        raise ValueError("E5c screen pair grid differs")
    if payload.get("privacy") != E5C_PRIVACY:
        raise ValueError("E5c screen privacy differs")
    if payload.get("exact_receiver_outcomes_allowed") is not False:
        raise ValueError("E5c screen plan must prohibit exact receiver outcomes")
    if set(payload.get("inputs", {})) != set(E5C_SCREEN_INPUT_NAMES):
        raise ValueError("E5c screen semantic-input closure differs")
    features = payload.get("candidate_features")
    if not isinstance(features, list) or not features:
        raise ValueError("E5c screen feature closure is missing")
    require_unique((row.get("feature") for row in features), "E5c screen features")
    _walk_public(payload)


def validate_e5c_screen_result(payload: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_e5c_screen_plan(plan)
    validate_stamped_payload(payload)
    if payload.get("analysis_semantics_version") != E5C_SCREEN_SEMANTICS_VERSION:
        raise ValueError("E5c screen-result semantics differ")
    if payload.get("status") != E5C_SCREEN_RESULT_STATUS:
        raise ValueError("E5c screen result is incomplete")
    if payload.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("E5c screen result references another plan")
    if payload.get("exact_source_intervention_for_receiver_outcome_executed") is not False:
        raise ValueError("E5c screen result contains an exact-outcome stage")
    rows = payload.get("direction_predictions")
    if not isinstance(rows, list):
        raise ValueError("E5c direction predictions must be a list")
    expected = len(E5C_DIRECTIONS) * 45
    if len(rows) != expected:
        raise ValueError(f"E5c screen has {len(rows)} rows, expected {expected}")
    require_unique(
        ((row.get("direction_id"), row.get("candidate_id")) for row in rows),
        "E5c direction/candidate rows",
    )
    for row in rows:
        for field in (
            "applied_activation_delta",
            "source_edit_l2",
            "graph_predicted_receiver_preactivation_delta",
            "atp_predicted_receiver_preactivation_delta",
        ):
            _finite(row.get(field), f"E5c screen {field}")
        if "exact_receiver_response" in row:
            raise ValueError("E5c screen row contains a forbidden exact outcome")
    if payload.get("privacy") != E5C_PRIVACY:
        raise ValueError("E5c screen-result privacy differs")
    _walk_public(payload)


def source_edit_identity(row: Mapping[str, Any]) -> str:
    direction = row.get("direction_id")
    source = row.get("source_feature")
    policy = row.get("source_edit_policy")
    if not all(isinstance(value, str) and value for value in (direction, source, policy)):
        raise ValueError("E5c source-edit identity fields are missing")
    if policy == "natural_counterfactual":
        return f"{direction}:{source}:natural"
    if policy == "l2_matched_to_anchor_source_edit":
        anchor = row.get("anchor_candidate_id")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError("E5c L2-matched source edit lacks an anchor")
        return f"{direction}:{source}:l2_matched_to:{anchor}"
    raise ValueError(f"Unsupported E5c source-edit policy: {policy}")


def _predicted_order(margin: float) -> str:
    if abs(margin) <= E5C_ORDER_ATOL:
        return "tie"
    return "anchor_larger" if margin > 0 else "control_larger"


def _predictor_relation(graph_margin: float, atp_margin: float) -> str:
    graph_order = _predicted_order(graph_margin)
    atp_order = _predicted_order(atp_margin)
    if "tie" in {graph_order, atp_order}:
        return "tie"
    return "agreement" if graph_order == atp_order else "disagreement"


def validate_e5c_exact_plan(payload: Mapping[str, Any]) -> None:
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E5C_SCHEMA_VERSION:
        raise ValueError("E5c exact-plan schema differs")
    if payload.get("analysis_semantics_version") != E5C_EXACT_PANEL_SEMANTICS_VERSION:
        raise ValueError("E5c exact-plan semantics differ")
    if payload.get("status") != E5C_EXACT_PANEL_STATUS:
        raise ValueError("E5c exact-plan status differs")
    if payload.get("mode") != "adaptive_exploration":
        raise ValueError("E5c exact stage must remain adaptive exploration")
    if payload.get("model", {}).get("id") != MODEL_ID or payload.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("E5c exact model identity differs")
    if payload.get("transcoder", {}).get("id") != TRANSCODER_ID or payload.get("transcoder", {}).get("revision") != TRANSCODER_REVISION:
        raise ValueError("E5c exact transcoder identity differs")
    if payload.get("pair_specs") != list(E5C_PAIR_SPECS) or payload.get("directions") != list(E5C_DIRECTIONS):
        raise ValueError("E5c exact pair grid differs")
    if payload.get("selection_used_exact_receiver_outcomes") is not False:
        raise ValueError("E5c exact plan must precede exact receiver outcomes")
    if payload.get("screen_outcomes_seen") is not True:
        raise ValueError("E5c exact plan must disclose predictor-screen observation")
    if set(payload.get("inputs", {})) != set(E5C_EXACT_INPUT_NAMES):
        raise ValueError("E5c exact semantic-input closure differs")
    candidates = payload.get("exact_candidates")
    if not isinstance(candidates, list) or len(candidates) != 270:
        raise ValueError("E5c exact plan must carry the full 270-row screen pool")
    require_unique(
        ((row.get("direction_id"), row.get("candidate_id")) for row in candidates),
        "E5c exact candidate keys",
    )
    candidate_lookup = {
        (row.get("direction_id"), row.get("candidate_id")): row for row in candidates
    }
    for row in candidates:
        expected = source_edit_identity(row)
        if row.get("source_edit_id") != expected:
            raise ValueError("E5c exact candidate source-edit identity differs")
        for field in (
            "applied_activation_delta",
            "source_edit_l2",
            "graph_predicted_receiver_preactivation_delta",
            "atp_predicted_receiver_preactivation_delta",
        ):
            _finite(row.get(field), f"E5c exact candidate {field}")
        if "exact_receiver_response" in row:
            raise ValueError("E5c exact plan already contains an exact outcome")
    contrasts = payload.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != 216:
        raise ValueError("E5c exact plan must contain all 216 anchor/control contrasts")
    require_unique((row.get("contrast_id") for row in contrasts), "E5c contrast ids")
    classifications = Counter(row.get("predictor_relation") for row in contrasts)
    if sum(classifications.values()) != 216 or any(
        key not in {"disagreement", "agreement", "tie"} for key in classifications
    ):
        raise ValueError("E5c predictor-relation labels differ")
    family_relations = Counter()
    for row in contrasts:
        graph_margin = _finite(
            row.get("graph_absolute_margin_anchor_minus_control"),
            "E5c graph contrast margin",
        )
        atp_margin = _finite(
            row.get("atp_absolute_margin_anchor_minus_control"),
            "E5c AtP contrast margin",
        )
        if row.get("graph_predicted_order") != _predicted_order(graph_margin):
            raise ValueError("E5c graph predicted order differs from its margin")
        if row.get("atp_predicted_order") != _predicted_order(atp_margin):
            raise ValueError("E5c AtP predicted order differs from its margin")
        relation = _predictor_relation(graph_margin, atp_margin)
        if row.get("predictor_relation") != relation:
            raise ValueError("E5c predictor relation differs from its two orderings")
        direction = row.get("direction_id")
        anchor = candidate_lookup.get((direction, row.get("anchor_candidate_id")))
        control = candidate_lookup.get((direction, row.get("control_candidate_id")))
        if anchor is None or control is None:
            raise ValueError("E5c contrast references a missing candidate")
        if anchor.get("pool_role") != "anchor_high":
            raise ValueError("E5c contrast anchor is not the quartet anchor")
        if anchor.get("quartet_id") != control.get("quartet_id"):
            raise ValueError("E5c contrast crosses quartets")
        if anchor.get("pair_id") != row.get("pair_id") or control.get("pair_id") != row.get("pair_id"):
            raise ValueError("E5c contrast pair identity differs")
        family = row.get("control_family")
        if family not in E5C_CONTROL_FAMILIES:
            raise ValueError("E5c contrast control family differs")
        family_relations[(family, relation)] += 1
    recorded = payload.get("contrast_counts", {})
    if any(
        recorded.get(key) != value
        for key, value in {
            "total": 216,
            "agreement": classifications["agreement"],
            "disagreement": classifications["disagreement"],
            "tie": classifications["tie"],
        }.items()
    ):
        raise ValueError("E5c contrast counts differ")
    expected_family_relations = {
        f"{family}:{relation}": count
        for (family, relation), count in sorted(family_relations.items())
    }
    if recorded.get("by_control_family_and_relation") != expected_family_relations:
        raise ValueError("E5c contrast family counts differ")
    expected_counts = payload.get("expected_counts", {})
    if expected_counts != {
        "candidates": len(candidates),
        "source_edits": len({row["source_edit_id"] for row in candidates}),
        "contrasts": len(contrasts),
    }:
        raise ValueError("E5c exact expected counts differ")
    if payload.get("privacy") != E5C_PRIVACY:
        raise ValueError("E5c exact-plan privacy differs")
    _walk_public(payload)


def validate_e5c_exact_result(payload: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_e5c_exact_plan(plan)
    validate_stamped_payload(payload)
    if payload.get("schema_version") != E5C_SCHEMA_VERSION:
        raise ValueError("E5c exact-result schema differs")
    if payload.get("analysis_semantics_version") != E5C_EXACT_SEMANTICS_VERSION:
        raise ValueError("E5c exact-result semantics differ")
    if payload.get("status") != E5C_EXACT_RESULT_STATUS:
        raise ValueError("E5c exact result is incomplete")
    if payload.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("E5c exact result references another plan")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 270:
        raise ValueError("E5c exact result must contain 270 candidates")
    require_unique(
        ((row.get("direction_id"), row.get("candidate_id")) for row in candidates),
        "E5c exact-result candidate keys",
    )
    expected = {
        (row["direction_id"], row["candidate_id"]) for row in plan["exact_candidates"]
    }
    if {(row.get("direction_id"), row.get("candidate_id")) for row in candidates} != expected:
        raise ValueError("E5c exact-result candidate grid differs")
    planned_candidates = {
        (row["direction_id"], row["candidate_id"]): row
        for row in plan["exact_candidates"]
    }
    immutable_fields = (
        "candidate_id", "graph_id", "direction_id", "pair_id", "pair_type",
        "receiver_band", "quartet_id", "pool_role", "source_feature",
        "receiver_feature", "source_layer", "receiver_layer", "layer_gap",
        "source_edit_policy", "anchor_candidate_id", "source_edit_id",
        "applied_activation_delta", "source_edit_l2",
        "graph_predicted_receiver_preactivation_delta",
        "atp_predicted_receiver_preactivation_delta",
    )
    for row in candidates:
        planned = planned_candidates[(row["direction_id"], row["candidate_id"])]
        if any(row.get(field) != planned.get(field) for field in immutable_fields):
            raise ValueError("E5c exact-result candidate metadata differs from the plan")
        response = row.get("exact_receiver_response")
        if not isinstance(response, Mapping):
            raise ValueError("E5c exact candidate lacks receiver response")
        for field in (
            "target_preactivation", "edited_preactivation", "preactivation_delta",
            "target_activation", "edited_activation", "activation_delta",
        ):
            _finite(response.get(field), f"E5c exact receiver {field}")
    source_edits = payload.get("source_edits")
    if not isinstance(source_edits, list) or not source_edits:
        raise ValueError("E5c exact source-edit rows are missing")
    require_unique((row.get("source_edit_id") for row in source_edits), "E5c exact source edits")
    planned_edits = {row["source_edit_id"] for row in plan["exact_candidates"]}
    if {row.get("source_edit_id") for row in source_edits} != planned_edits:
        raise ValueError("E5c exact source-edit closure differs")
    planned_by_edit: dict[str, list[Mapping[str, Any]]] = {}
    for row in plan["exact_candidates"]:
        planned_by_edit.setdefault(row["source_edit_id"], []).append(row)
    for row in source_edits:
        group = planned_by_edit[row["source_edit_id"]]
        representative = group[0]
        for field in (
            "direction_id", "source_feature", "source_edit_policy",
            "anchor_candidate_id", "natural_activation_delta",
            "applied_activation_delta", "source_edit_l2",
        ):
            if row.get(field) != representative.get(field):
                raise ValueError("E5c exact source-edit metadata differs from the plan")
        if row.get("candidate_ids") != sorted(item["candidate_id"] for item in group):
            raise ValueError("E5c exact source-edit candidate membership differs")
    if payload.get("privacy") != E5C_PRIVACY:
        raise ValueError("E5c exact-result privacy differs")
    _walk_public(payload)


def stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload
