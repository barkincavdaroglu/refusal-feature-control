"""Public-safe contracts for the S9 E3 feature-path falsification study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from s9_contract import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    parse_feature,
    require_unique,
    validate_stamped_payload,
)


PATH_SCHEMA_VERSION = 1
PATH_SEMANTICS_VERSION = "s9-e3-feature-path-v1"
PRIVACY = {
    "prompt_text_in_public_artifacts": False,
    "completion_text_in_public_artifacts": False,
    "token_text_in_public_artifacts": False,
}


def validate_path_plan(plan: Mapping[str, Any]) -> None:
    validate_stamped_payload(plan)
    if plan.get("schema_version") != PATH_SCHEMA_VERSION:
        raise ValueError("Unsupported S9 path plan schema")
    if plan.get("analysis_semantics_version") != PATH_SEMANTICS_VERSION:
        raise ValueError("Unsupported S9 path semantics")
    if plan.get("mode") != "adaptive_exploration":
        raise ValueError("S9 E3 must remain adaptive_exploration")
    if plan.get("model", {}).get("id") != MODEL_ID:
        raise ValueError("Path plan model differs")
    if plan.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("Path plan model revision differs")
    if plan.get("transcoder", {}).get("id") != TRANSCODER_ID:
        raise ValueError("Path plan transcoder differs")
    if plan.get("transcoder", {}).get("revision") != TRANSCODER_REVISION:
        raise ValueError("Path plan transcoder revision differs")
    if plan.get("privacy") != PRIVACY:
        raise ValueError("Path plan privacy contract differs")

    contexts = plan.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Path plan requires contexts")
    require_unique((row.get("context_id") for row in contexts), "path context ids")
    for row in contexts:
        for key in ("context_id", "target_prompt_key", "delta_target_prompt_key", "delta_source_prompt_key"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Malformed path context field: {key}")

    relations = plan.get("relations")
    if not isinstance(relations, list) or not relations:
        raise ValueError("Path plan requires relations")
    require_unique((row.get("relation_id") for row in relations), "path relation ids")
    relation_ids = {row["relation_id"] for row in relations}
    for row in relations:
        source_layer, _ = parse_feature(row.get("source_feature"))
        receiver_layer, _ = parse_feature(row.get("receiver_feature"))
        if source_layer >= receiver_layer:
            raise ValueError(f"Path relation is not forward-ordered: {row['relation_id']}")
        reference = row.get("l2_match_reference_relation")
        if reference is not None and reference not in relation_ids:
            raise ValueError(f"Unknown control reference relation: {reference}")
        if row.get("graph_status") not in {"graph_proposed", "matched_source_control"}:
            raise ValueError(f"Unsupported graph status: {row.get('graph_status')}")

    features = plan.get("candidate_features")
    if not isinstance(features, list) or not features:
        raise ValueError("Path plan requires candidate features")
    require_unique((row.get("feature") for row in features), "path candidate features")
    available = {row["feature"] for row in features}
    required = {
        feature
        for relation in relations
        for feature in (relation["source_feature"], relation["receiver_feature"])
    }
    if not required <= available:
        raise ValueError("A path relation references an unavailable feature")
    for feature in available:
        parse_feature(feature)

    decomposition = plan.get("module_decomposition_relation_ids")
    if not isinstance(decomposition, list) or not set(decomposition) <= relation_ids:
        raise ValueError("Module decomposition relations differ")
    if plan.get("expected_relation_rows") != len(contexts) * len(relations):
        raise ValueError("Expected path result grid differs")


def validate_path_result(result: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_stamped_payload(result)
    validate_path_plan(plan)
    if result.get("schema_version") != PATH_SCHEMA_VERSION:
        raise ValueError("Unsupported S9 path result schema")
    if result.get("analysis_semantics_version") != PATH_SEMANTICS_VERSION:
        raise ValueError("Path result semantics differ")
    if result.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("Path result references a different plan")
    if result.get("privacy") != PRIVACY:
        raise ValueError("Path result privacy contract differs")
    rows = result.get("relations")
    if not isinstance(rows, list):
        raise ValueError("Path result relations must be a list")
    if result.get("status") == "complete":
        expected = {
            (context["context_id"], relation["relation_id"])
            for context in plan["contexts"]
            for relation in plan["relations"]
        }
        actual = {(row.get("context_id"), row.get("relation_id")) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("Complete path result does not contain the exact grid")
    forbidden = {"prompt", "prompt_text", "completion", "token_text", "chat_text"}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in forbidden:
                    raise ValueError(f"Public path result contains forbidden key: {key}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
