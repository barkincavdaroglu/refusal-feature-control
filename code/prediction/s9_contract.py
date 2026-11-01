"""Pure contracts for the S9 exploratory attribution/patch calibration.

This module intentionally contains no model imports.  It defines the public,
text-free schema and deterministic candidate selection used by both the local
tests and the GPU runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ANALYSIS_SEMANTICS_VERSION = "s9-e0-e2-v2"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TRANSCODER_ID = "mwhanna/qwen3-4b-transcoders"
TRANSCODER_REVISION = "94d176260ac39ce2f882b8b09aba8c118df29bb3"
REFUSAL_TOKEN_IDS = (40, 19152)
METRICS = ("probe_score", "next_token_refusal_log_odds")
COMPONENT_KINDS = ("mlp_output", "attention_output")
PAIR_TYPES = (
    "harmful_behavior_flip",
    "harmful_same_behavior",
    "benign_same_behavior",
)
FEATURE_RE = re.compile(r"^L(?P<layer>\d+):F(?P<feature>\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stamp_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_stamped_payload(payload: Mapping[str, Any]) -> None:
    recorded = payload.get("payload_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("payload_sha256 is missing or malformed")
    if recorded != canonical_sha256(payload):
        raise ValueError("payload_sha256 does not match canonical payload bytes")


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def parse_feature(value: str) -> tuple[int, int]:
    match = FEATURE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid feature identity: {value}")
    return int(match.group("layer")), int(match.group("feature"))


def coarse_label(label: str) -> str:
    if label in {"R", "S"}:
        return "refusal_like"
    if label in {"P", "C"}:
        return "compliance_like"
    raise ValueError(f"Unsupported manual label: {label}")


def component_id(kind: str, layer: int) -> str:
    if kind not in COMPONENT_KINDS:
        raise ValueError(f"Unsupported component kind: {kind}")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError(f"Layer must be a non-negative integer, got {layer!r}")
    return f"{kind}:L{layer}"


def build_directions(pair_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    directions: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    seen_prompts: set[str] = set()
    for pair in pair_specs:
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Every pair requires a non-empty pair_id")
        if pair_id in seen_pairs:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        seen_pairs.add(pair_id)
        pair_type = pair.get("pair_type")
        if pair_type not in PAIR_TYPES:
            raise ValueError(f"Unsupported pair_type for {pair_id}: {pair_type}")
        a = pair.get("a")
        b = pair.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b or a == b:
            raise ValueError(f"Pair {pair_id} requires two distinct prompt keys")
        if a in seen_prompts or b in seen_prompts:
            raise ValueError("The initial calibration requires prompt-disjoint pairs")
        seen_prompts.update((a, b))
        directions.extend(
            (
                {
                    "direction_id": f"{pair_id}:a_from_b",
                    "pair_id": pair_id,
                    "pair_type": pair_type,
                    "target_prompt_key": a,
                    "source_prompt_key": b,
                },
                {
                    "direction_id": f"{pair_id}:b_from_a",
                    "pair_id": pair_id,
                    "pair_type": pair_type,
                    "target_prompt_key": b,
                    "source_prompt_key": a,
                },
            )
        )
    return directions


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    atp = row.get("attribution_patch")
    if not isinstance(atp, Mapping):
        raise ValueError("component row lacks attribution_patch mapping")
    return finite_number(atp.get(metric), f"attribution_patch.{metric}")


def select_component_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_per_metric: int,
    low_both: int,
    disagreements: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, deliberately non-consensus exact-patch panel.

    Selection includes the largest absolute AtP estimates for each metric,
    components small under both metrics, and components whose normalized
    magnitudes or signs disagree.  The result is exploratory and intentionally
    not an unbiased sample for estimating global AtP calibration.
    """

    if not rows:
        raise ValueError("Cannot select candidates from an empty component table")
    for name, value in (
        ("top_per_metric", top_per_metric),
        ("low_both", low_both),
        ("disagreements", disagreements),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        node_id = row.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Every component row needs a non-empty node_id")
        if node_id in by_id:
            raise ValueError(f"Duplicate component row: {node_id}")
        for metric in METRICS:
            _metric_value(row, metric)
        by_id[node_id] = row

    reasons: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    for metric in METRICS:
        ranked = sorted(
            by_id.values(),
            key=lambda row: (-abs(_metric_value(row, metric)), str(row["node_id"])),
        )
        for row in ranked[:top_per_metric]:
            reasons[str(row["node_id"])].add(f"top_abs_{metric}")

    maxima = {
        metric: max(abs(_metric_value(row, metric)) for row in by_id.values())
        for metric in METRICS
    }

    def normalized(row: Mapping[str, Any], metric: str) -> float:
        maximum = maxima[metric]
        if maximum == 0:
            return 0.0
        return abs(_metric_value(row, metric)) / maximum

    low_ranked = sorted(
        by_id.values(),
        key=lambda row: (
            max(normalized(row, metric) for metric in METRICS),
            str(row["node_id"]),
        ),
    )
    for row in low_ranked[:low_both]:
        reasons[str(row["node_id"])].add("low_both_metrics")

    def disagreement_score(row: Mapping[str, Any]) -> float:
        values = [_metric_value(row, metric) for metric in METRICS]
        magnitude_gap = abs(normalized(row, METRICS[0]) - normalized(row, METRICS[1]))
        sign_gap = 1.0 if values[0] * values[1] < 0 else 0.0
        return sign_gap + magnitude_gap

    disagreement_ranked = sorted(
        by_id.values(),
        key=lambda row: (-disagreement_score(row), str(row["node_id"])),
    )
    for row in disagreement_ranked[:disagreements]:
        reasons[str(row["node_id"])].add("metric_disagreement")

    selected = []
    for node_id in sorted(node_id for node_id, why in reasons.items() if why):
        row = by_id[node_id]
        selected.append(
            {
                "node_id": node_id,
                "kind": row["kind"],
                "layer": row["layer"],
                "reasons": sorted(reasons[node_id]),
            }
        )
    return selected


def require_unique(values: Iterable[str], name: str) -> None:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{name} must be unique")


def validate_plan(plan: Mapping[str, Any]) -> None:
    validate_stamped_payload(plan)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported S9 plan schema_version")
    if plan.get("analysis_semantics_version") != ANALYSIS_SEMANTICS_VERSION:
        raise ValueError("Unsupported S9 analysis semantics")
    if plan.get("mode") != "adaptive_exploration":
        raise ValueError("S9 calibration must remain labelled adaptive_exploration")
    if plan.get("model", {}).get("id") != MODEL_ID:
        raise ValueError("Model identity differs")
    if plan.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("Model revision differs")
    if plan.get("transcoder", {}).get("id") != TRANSCODER_ID:
        raise ValueError("Transcoder identity differs")
    if plan.get("transcoder", {}).get("revision") != TRANSCODER_REVISION:
        raise ValueError("Transcoder revision differs")
    directions = plan.get("directions")
    if not isinstance(directions, list) or not directions:
        raise ValueError("Plan requires non-empty directions")
    require_unique((row.get("direction_id") for row in directions), "direction ids")
    expected = build_directions(plan.get("pair_specs", []))
    if directions != expected:
        raise ValueError("Direction table does not match pair specifications")
    features = plan.get("candidate_features")
    if not isinstance(features, list) or not features:
        raise ValueError("Plan requires candidate_features")
    require_unique((row.get("feature") for row in features), "candidate features")
    for row in features:
        parse_feature(row.get("feature"))
    exact = plan.get("exact_patch_selection", {})
    for key in ("top_per_metric", "low_both", "disagreements"):
        value = exact.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"exact_patch_selection.{key} must be a non-negative integer")
    doses = exact.get("doses")
    if not isinstance(doses, list) or doses != sorted(set(doses)):
        raise ValueError("Exact-patch doses must be sorted and unique")
    if doses != [0.5, 1.0]:
        raise ValueError("Initial S9 exact-patch doses must be [0.5, 1.0]")
    privacy = plan.get("privacy", {})
    if privacy != {
        "prompt_text_in_public_artifacts": False,
        "completion_text_in_public_artifacts": False,
        "token_text_in_public_artifacts": False,
    }:
        raise ValueError("S9 public privacy contract differs")


def validate_result_shape(result: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_stamped_payload(result)
    validate_plan(plan)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported result schema_version")
    if result.get("analysis_semantics_version") != ANALYSIS_SEMANTICS_VERSION:
        raise ValueError("Result semantics differ from plan")
    if result.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("Result references a different plan payload")
    if result.get("privacy") != plan.get("privacy"):
        raise ValueError("Result privacy contract differs from plan")
    directions = result.get("directions")
    if not isinstance(directions, list):
        raise ValueError("Result directions must be a list")
    if result.get("status") == "complete":
        expected = {row["direction_id"] for row in plan["directions"]}
        actual = {row.get("direction_id") for row in directions}
        if actual != expected or len(actual) != len(directions):
            raise ValueError("Complete result does not contain the exact direction grid")
    forbidden_keys = {"prompt", "prompt_text", "completion", "token_text", "chat_text"}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in forbidden_keys:
                    raise ValueError(f"Public result contains forbidden key: {key}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
