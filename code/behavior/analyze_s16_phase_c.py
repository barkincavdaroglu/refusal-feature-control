#!/usr/bin/env python3
"""Analyze the authenticated S16 Phase-C Tier-1 generated-text experiment.

The scientific estimates are computed independently of the public cross-check. The
second-rater agreement and kappa intervals are imported from the authenticated
amendment-02 audit, as required by the frozen reporting rule.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping

import numpy as np

import s16_phase_b_contract as contract


SEED, REPS = 2522210527409210963, 10_000
PREREGISTRATION_SHA256 = "25928a120f53a277553a15a6f271463b990f52f186653773f117507ff0a6f13d"
AMENDMENT_01_SHA256 = "7f99acc0a5fb1a109d610177517b4fab876adcefb58670173f096954a62856ed"
AMENDMENT_02_SHA256 = "ba4b53484a4944ea5b662225f8218317df8283ff73c43af66ec73b903747bd44"
BASELINE_ANNOTATIONS_SHA256 = "3d0a9daec25d21b938b06587931ee4a0131d932a6d07ed7a6e70cc9b46767c43"
TIER1_ANNOTATIONS_SHA256 = "0c7a059e4f42a0c8e31feb90114742eb6eb89db7ff67d7ffd985682c38a1aed6"
SECOND_RATER_AUDIT_PAYLOAD_SHA256 = "1070eac101948f85d7e1f1a0cc672c9c9271cfb663fa8fa4c1c7da643fbe30e7"

PRIMARY_ARMS = (
    "a1_persistent",
    "a2_natural",
    "a3_benign",
    "rb_a2_natural",
    "rb_a3_benign",
)
CONTROL_ARMS = (
    "rd_a2_natural",
    "rd_a3_benign",
    "rnd1707_a1_persistent",
    "rnd1707_a2_natural",
    "rnd1707_a3_benign",
)
TIER1_ARMS = PRIMARY_ARMS + CONTROL_ARMS
EXPECTED_GENERATION_COUNTS = {
    "baseline": 204,
    "a1_persistent": 120,
    "a2_natural": 84,
    "a3_benign": 60,
    "rb_a2_natural": 84,
    "rb_a3_benign": 60,
    "rd_a2_natural": 84,
    "rd_a3_benign": 60,
    "rnd1707_a1_persistent": 120,
    "rnd1707_a2_natural": 84,
    "rnd1707_a3_benign": 60,
}
DIRECT_VARIANTS, FICTIONAL_VARIANTS = {"D0", "D1", "D2"}, {"F0", "F1", "F2"}
VALID_LABELS = {"R", "S", "P", "C", "U"}
ESTIMAND_KEYS = (
    "necessity_A1_direct",
    "necessity_A1_all_refusing",
    "necessity_control_RND_A1_direct",
    "necessity_control_RND_A1_all_refusing",
    "sufficiency_A2_fictional_complying",
    "sufficiency_A2_fresh_complying",
    "sufficiency_RB_A2_fictional_complying",
    "sufficiency_RB_A2_fresh_complying",
    "sufficiency_RD_A2_fictional_complying",
    "sufficiency_RD_A2_fresh_complying",
    "sufficiency_RND_A2_fictional_complying",
    "sufficiency_RND_A2_fresh_complying",
    "benign_A3_refusal_like",
    "benign_RB_A3_refusal_like",
    "benign_RD_A3_refusal_like",
    "benign_RND_A3_refusal_like",
)


def csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def indexed_annotations(path: pathlib.Path) -> dict[str, dict[str, str]]:
    rows = csv_rows(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        record_id = row.get("record_id", "")
        label = row.get("manual_label", "").strip().upper()
        if not record_id or record_id in result:
            raise contract.ContractError(f"duplicate or missing annotation record id: {path}")
        if label not in VALID_LABELS:
            raise contract.ContractError(f"invalid annotation label for {record_id}: {label!r}")
        result[record_id] = {**row, "manual_label": label}
    return result


def refusal(label: str) -> bool:
    return label in {"R", "S"}


def binary(label: str) -> bool | None:
    return None if label == "U" else refusal(label)


def clustered(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Frozen analyzer bootstrap: sorted intent clusters, NumPy PCG64, percentiles."""

    if not rows:
        return {"n": 0, "n_intents": 0}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["intent_id"]].append(row)
    ids = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(SEED))
    values: list[float] = []
    estimate = float(np.mean([row[key] for row in rows]))
    for _ in range(REPS):
        sample = [
            row
            for group_id in rng.choice(ids, len(ids), replace=True)
            for row in groups[group_id]
        ]
        value = float(np.mean([row[key] for row in sample]))
        if math.isfinite(value):
            values.append(value)
    return {
        "count": int(sum(row[key] for row in rows)),
        "n": len(rows),
        "n_intents": len(ids),
        "estimate": estimate,
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "defined_replicates": len(values),
        "requested_replicates": REPS,
    }


def declared_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Serializer used by the Tier-1 audit and independent cross-check tools."""

    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def load_second_rater_audit(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise contract.ContractError(f"second-rater audit is absent or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or declared_payload_sha256(payload) != payload.get("payload_sha256")
    ):
        raise contract.ContractError("Tier-1 second-rater audit payload authentication failed")
    if payload["payload_sha256"] != SECOND_RATER_AUDIT_PAYLOAD_SHA256:
        raise contract.ContractError("Tier-1 second-rater audit payload differs from the audited value")
    if payload.get("amendment_02_sha256") != AMENDMENT_02_SHA256:
        raise contract.ContractError("Tier-1 second-rater audit amendment binding differs")
    return payload


def main() -> int:
    step = pathlib.Path(__file__).resolve().parents[1]
    output = step / "outputs/phase_c_local"
    preregistration = step / "PHASE_C_PREREGISTRATION.md"
    amendment_01 = step / "PHASE_C_AMENDMENT_01.md"
    amendment_02 = step / "PHASE_C_AMENDMENT_02.md"
    if contract.file_sha256(preregistration) != PREREGISTRATION_SHA256:
        raise contract.ContractError("Phase-C preregistration hash differs")
    if contract.file_sha256(amendment_01) != AMENDMENT_01_SHA256:
        raise contract.ContractError("Phase-C amendment-01 hash differs")
    if contract.file_sha256(amendment_02) != AMENDMENT_02_SHA256:
        raise contract.ContractError("Phase-C amendment-02 hash differs")

    manifest_path = output / "s16_phase_c_run_manifest.json"
    binary_gate_path = output / "s16_phase_c_baseline_binary_gate.json"
    exact_gate_path = output / "s16_phase_c_baseline_gate.json"
    tier1_receipt_path = output / "s16_phase_c_tier1_annotation_receipt.json"
    baseline_annotations_path = output / "s16_phase_c_baseline_annotations.csv"
    tier1_annotations_path = output / "s16_phase_c_tier1_annotations.csv"
    second_annotations_path = output / "s16_phase_c_second_rater_annotations.csv"
    second_audit_path = output / "s16_phase_c_tier1_second_rater_audit.json"

    manifest = contract.load_authenticated_json(
        manifest_path, schema="s16-phase-c-local-run/1"
    )
    binary_gate = contract.load_authenticated_json(
        binary_gate_path, schema="s16-phase-c-amendment-01-binary-gate/1"
    )
    exact_gate = contract.load_authenticated_json(exact_gate_path)
    tier1_receipt = contract.load_authenticated_json(
        tier1_receipt_path, schema="s16-phase-c-annotations/1"
    )
    audit = load_second_rater_audit(second_audit_path)

    if manifest.get("status") != "tier1_generation_complete_pending_blind_labels":
        raise contract.ContractError("Phase-C Tier-1 generation closure is incomplete")
    if manifest.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise contract.ContractError("Phase-C manifest preregistration binding differs")
    if manifest.get("amendment_01_sha256") != AMENDMENT_01_SHA256:
        raise contract.ContractError("Phase-C manifest amendment-01 binding differs")
    if manifest.get("binary_gate_sha256") != contract.file_sha256(binary_gate_path):
        raise contract.ContractError("Phase-C manifest binary-gate binding differs")
    if manifest.get("baseline_gate_sha256") != contract.file_sha256(exact_gate_path):
        raise contract.ContractError("Phase-C manifest exact-gate binding differs")
    if binary_gate.get("status") != "passed" or binary_gate.get("agreement_rate", 0) < 0.80:
        raise contract.ContractError("Phase-C amended binary baseline gate is not passed")
    if exact_gate.get("status") != "failed" or exact_gate.get("agreement_rate", 1) >= 0.80:
        raise contract.ContractError("Phase-C original exact-label gate history differs")
    if tier1_receipt.get("status") != "complete" or tier1_receipt.get("record_count") != 816:
        raise contract.ContractError("Phase-C Tier-1 annotation closure is incomplete")

    if contract.file_sha256(baseline_annotations_path) != BASELINE_ANNOTATIONS_SHA256:
        raise contract.ContractError("Phase-C frozen baseline annotation hash differs")
    if binary_gate["input_sha256"]["annotations"] != BASELINE_ANNOTATIONS_SHA256:
        raise contract.ContractError("Phase-C binary gate does not bind the frozen baseline labels")
    if contract.file_sha256(tier1_annotations_path) != TIER1_ANNOTATIONS_SHA256:
        raise contract.ContractError("Phase-C Tier-1 primary annotation hash differs")
    if tier1_receipt.get("annotations_sha256") != TIER1_ANNOTATIONS_SHA256:
        raise contract.ContractError("Phase-C Tier-1 receipt annotation binding differs")

    baseline_annotations = indexed_annotations(baseline_annotations_path)
    tier1_annotations = indexed_annotations(tier1_annotations_path)
    second_annotations = indexed_annotations(second_annotations_path)
    if len(baseline_annotations) != 204 or len(tier1_annotations) != 816:
        raise contract.ContractError("Phase-C baseline/Tier-1 annotation counts differ")
    if len(second_annotations) != 55:
        raise contract.ContractError("Phase-C amendment-02 second-rater count differs")
    if Counter(row["manual_label"] for row in tier1_annotations.values()) != Counter(
        tier1_receipt["label_counts"]
    ):
        raise contract.ContractError("Phase-C Tier-1 label counts differ from the receipt")
    if (
        contract.file_sha256(second_annotations_path)
        != audit["inputs_sha256"]["public_second_rater_csv"]
    ):
        raise contract.ContractError("Phase-C second-rater annotations differ from the audit binding")
    if audit["selection"]["candidate_arm_ids"] != list(PRIMARY_ARMS):
        raise contract.ContractError("Phase-C second-rater audit primary-arm universe differs")
    if audit["bootstrap"] != {
        "seed": SEED,
        "replicates": REPS,
        "unit": "item",
        "percentile": [2.5, 97.5],
    }:
        raise contract.ContractError("Phase-C second-rater audit bootstrap declaration differs")

    records: dict[str, dict[str, Any]] = {}
    for item in manifest.get("completed", []):
        record_id = item.get("record_id")
        if not record_id or record_id in records:
            raise contract.ContractError("Phase-C manifest has a duplicate or missing record id")
        if item.get("public_path") != f"records/{record_id}.json":
            raise contract.ContractError(f"Phase-C public path differs for {record_id}")
        path = output / item["public_path"]
        if contract.file_sha256(path) != item.get("public_sha256"):
            raise contract.ContractError(f"Phase-C public record hash differs for {record_id}")
        record = contract.load_authenticated_json(
            path, schema="s16-phase-c-public-record/1"
        )
        if record.get("record_id") != record_id:
            raise contract.ContractError(f"Phase-C record identity differs for {record_id}")
        records[record_id] = record

    if len(records) != 1020:
        raise contract.ContractError("Phase-C Tier-1 manifest must bind exactly 1,020 records")
    arm_counts = Counter(record["arm"]["id"] for record in records.values())
    if arm_counts != Counter(EXPECTED_GENERATION_COUNTS):
        raise contract.ContractError(
            f"Phase-C Tier-1 arm counts differ: {dict(sorted(arm_counts.items()))}"
        )
    baseline_ids = {
        record_id
        for record_id, record in records.items()
        if record["arm"]["id"] == "baseline"
    }
    tier1_ids = set(records) - baseline_ids
    if baseline_ids != set(baseline_annotations) or tier1_ids != set(tier1_annotations):
        raise contract.ContractError("Phase-C record and annotation identities differ")

    primary_labels: dict[str, str] = {}
    for annotations in (baseline_annotations, tier1_annotations):
        for record_id, row in annotations.items():
            record = records[record_id]
            if row.get("completion_sha256") != record.get("completion_sha256"):
                raise contract.ContractError(
                    f"Phase-C annotation completion binding differs for {record_id}"
                )
            primary_labels[record_id] = row["manual_label"]

    baseline = {
        record["row_id"]: record
        for record in records.values()
        if record["arm"]["id"] == "baseline"
    }
    arms: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records.values():
        arm_id, row_id = record["arm"]["id"], record["row_id"]
        if row_id in arms[arm_id]:
            raise contract.ContractError(f"duplicate Phase-C row in arm {arm_id}: {row_id}")
        arms[arm_id][row_id] = record
    if len(baseline) != 204 or any(
        record["row_id"] not in baseline for record in records.values()
    ):
        raise contract.ContractError("Phase-C same-row baseline closure differs")

    sensitivity_labels = dict(primary_labels)
    reclassified: list[dict[str, str]] = []
    for record_id, row in second_annotations.items():
        if record_id not in tier1_annotations:
            raise contract.ContractError(f"second-rater record is outside Tier 1: {record_id}")
        if row.get("completion_sha256") != records[record_id].get("completion_sha256"):
            raise contract.ContractError(
                f"second-rater completion binding differs for {record_id}"
            )
        original, replacement = primary_labels[record_id], row["manual_label"]
        if binary(replacement) is not None and binary(replacement) != binary(original):
            sensitivity_labels[record_id] = replacement
            reclassified.append(
                {
                    "record_id": record_id,
                    "agent_label": original,
                    "barkin_label": replacement,
                }
            )
    audited_disagreements = {
        (row["record_id"], row["agent_label"], row["barkin_label"])
        for row in audit["binary_disagreements"]
    }
    observed_disagreements = {
        (row["record_id"], row["agent_label"], row["barkin_label"])
        for row in reclassified
    }
    if observed_disagreements != audited_disagreements:
        raise contract.ContractError(
            "Phase-C amendment-02 sensitivity records differ from the audit"
        )
    if len(reclassified) != audit["rule_5_7"]["sensitivity_reclassified_record_count"]:
        raise contract.ContractError(
            "Phase-C amendment-02 sensitivity count differs from the audit"
        )

    def paired(
        arm_id: str,
        predicate: Callable[[dict[str, Any]], bool],
        labels: Mapping[str, str],
        event: str,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row_id, arm in arms[arm_id].items():
            base = baseline[row_id]
            label = labels[arm["record_id"]]
            if not predicate(base) or label == "U":
                continue
            hit = not refusal(label) if event == "compliance" else refusal(label)
            selected.append({"intent_id": base["intent_id"], "event": int(hit)})
        return selected

    def all_estimands(labels: Mapping[str, str]) -> dict[str, Any]:
        base_label = lambda row: primary_labels[row["record_id"]]
        direct_refusing = lambda row: (
            row["risk_class"] == "harmful"
            and row["variant_id"] in DIRECT_VARIANTS
            and refusal(base_label(row))
        )
        all_harmful_refusing = lambda row: (
            row["risk_class"] == "harmful" and refusal(base_label(row))
        )
        fictional_complying = lambda row: (
            row["risk_class"] == "harmful"
            and row["variant_id"] in FICTIONAL_VARIANTS
            and not refusal(base_label(row))
        )
        fresh_complying = lambda row: (
            row["source_id"] == "s16_fresh_reference_pairs"
            and not refusal(base_label(row))
        )
        benign_s10 = lambda row: (
            row["risk_class"] == "benign"
            and row["source_id"] != "s16_fresh_reference_pairs"
        )
        estimands: dict[str, Any] = {
            "denominators": {
                "harmful_direct_refusing": sum(
                    direct_refusing(row) for row in baseline.values()
                ),
                "harmful_fictional_complying": sum(
                    fictional_complying(row) for row in baseline.values()
                ),
                "harmful_fictional_all": sum(
                    row["risk_class"] == "harmful"
                    and row["variant_id"] in FICTIONAL_VARIANTS
                    for row in baseline.values()
                ),
                "fresh_complying": sum(
                    fresh_complying(row) for row in baseline.values()
                ),
                "benign": sum(
                    row["risk_class"] == "benign" for row in baseline.values()
                ),
                "harmful_all_refusing": sum(
                    all_harmful_refusing(row) for row in baseline.values()
                ),
            }
        }
        specifications = (
            ("necessity_A1_direct", "a1_persistent", direct_refusing, "compliance"),
            ("necessity_A1_all_refusing", "a1_persistent", all_harmful_refusing, "compliance"),
            (
                "necessity_control_RND_A1_direct",
                "rnd1707_a1_persistent",
                direct_refusing,
                "compliance",
            ),
            (
                "necessity_control_RND_A1_all_refusing",
                "rnd1707_a1_persistent",
                all_harmful_refusing,
                "compliance",
            ),
            (
                "sufficiency_A2_fictional_complying",
                "a2_natural",
                fictional_complying,
                "refusal",
            ),
            (
                "sufficiency_A2_fresh_complying",
                "a2_natural",
                fresh_complying,
                "refusal",
            ),
            (
                "sufficiency_RB_A2_fictional_complying",
                "rb_a2_natural",
                fictional_complying,
                "refusal",
            ),
            (
                "sufficiency_RB_A2_fresh_complying",
                "rb_a2_natural",
                fresh_complying,
                "refusal",
            ),
            (
                "sufficiency_RD_A2_fictional_complying",
                "rd_a2_natural",
                fictional_complying,
                "refusal",
            ),
            (
                "sufficiency_RD_A2_fresh_complying",
                "rd_a2_natural",
                fresh_complying,
                "refusal",
            ),
            (
                "sufficiency_RND_A2_fictional_complying",
                "rnd1707_a2_natural",
                fictional_complying,
                "refusal",
            ),
            (
                "sufficiency_RND_A2_fresh_complying",
                "rnd1707_a2_natural",
                fresh_complying,
                "refusal",
            ),
            ("benign_A3_refusal_like", "a3_benign", benign_s10, "refusal"),
            (
                "benign_RB_A3_refusal_like",
                "rb_a3_benign",
                benign_s10,
                "refusal",
            ),
            (
                "benign_RD_A3_refusal_like",
                "rd_a3_benign",
                benign_s10,
                "refusal",
            ),
            (
                "benign_RND_A3_refusal_like",
                "rnd1707_a3_benign",
                benign_s10,
                "refusal",
            ),
        )
        for name, arm_id, predicate, event in specifications:
            estimands[name] = clustered(
                paired(arm_id, predicate, labels, event), "event"
            )

        def route_share(arm_key: str, rb_key: str) -> dict[str, Any]:
            arm_effect, rb_effect = estimands[arm_key], estimands[rb_key]
            eligible = arm_effect["estimate"] >= 0.10 and arm_effect["ci95"][0] > 0
            if not eligible:
                return {
                    "route_share": None,
                    "reason": "arm effect interval includes 0 or size < 0.10",
                    "arm_effect": arm_effect["estimate"],
                    "rb_effect": rb_effect["estimate"],
                }
            return {
                "route_share": 1 - rb_effect["estimate"] / arm_effect["estimate"],
                "arm_effect": arm_effect["estimate"],
                "rb_effect": rb_effect["estimate"],
            }

        estimands["route_share_A2_fictional"] = route_share(
            "sufficiency_A2_fictional_complying",
            "sufficiency_RB_A2_fictional_complying",
        )
        estimands["route_share_A3_benign"] = route_share(
            "benign_A3_refusal_like", "benign_RB_A3_refusal_like"
        )
        return estimands

    primary = all_estimands(primary_labels)
    sensitivity = all_estimands(sensitivity_labels)
    if any(
        primary[key].get("requested_replicates") != REPS for key in ESTIMAND_KEYS
    ):
        raise contract.ContractError("Phase-C Tier-1 bootstrap replicate count differs")

    label_distribution = {
        arm_id: dict(
            sorted(
                Counter(
                    primary_labels[record["record_id"]]
                    for record in records.values()
                    if record["arm"]["id"] == arm_id
                ).items()
            )
        )
        for arm_id in sorted(TIER1_ARMS)
    }
    numeric_continuity: dict[str, dict[str, Any]] = {}
    for arm_id in sorted(TIER1_ARMS):
        primary_deltas: list[float] = []
        probe_deltas: list[float] = []
        for arm in arms[arm_id].values():
            base = baseline[arm["row_id"]]
            primary_deltas.append(
                float(arm["primary_score"] - base["primary_score"])
            )
            probe_deltas.append(float(arm["probe_value"] - base["probe_value"]))
        numeric_continuity[arm_id] = {
            "n": len(primary_deltas),
            "median_delta_primary_score": statistics.median(primary_deltas),
            "median_delta_probe": statistics.median(probe_deltas),
        }

    headline_variant = (
        "sensitivity_barkin_reclassified"
        if audit["rule_5_7"]["headline_is_sensitivity"]
        else "primary_agent_labels"
    )
    result = {
        "schema_version": "s16-phase-c-tier1-result/1",
        "status": "complete_preregistered_phase_c_tier1",
        "claim_boundary": (
            "Tier-1 generated-text intervention evidence only; no unique native-variable, "
            "mediation, selectivity, deferred-arm, Phase-D, or population-general claim."
        ),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "amendment_01_sha256": AMENDMENT_01_SHA256,
        "amendment_02_sha256": AMENDMENT_02_SHA256,
        "bootstrap": {
            "unit": "intent",
            "seed": SEED,
            "replicates": REPS,
            "implementation": (
                "numpy.random.Generator(PCG64), sorted intent ids, numpy percentile"
            ),
        },
        "generation_counts": dict(sorted(arm_counts.items())),
        "baseline_binary_gate": {
            key: binary_gate[key]
            for key in (
                "comparison_count",
                "agreement_count",
                "agreement_rate",
                "exact_four_class_agreement_count",
                "exact_four_class_agreement_rate",
            )
        },
        "second_rater": {
            "source": "s16_phase_c_tier1_second_rater_audit.json",
            "audit_payload_sha256": audit["payload_sha256"],
            "agreement": audit["agreement"],
            "per_arm": audit["per_arm"],
            "flip_sides": audit["flip_sides"],
            "rule_5_7": audit["rule_5_7"],
            "annotations_sha256": contract.file_sha256(second_annotations_path),
        },
        "headline_variant": headline_variant,
        "primary_agent_labels": primary,
        "sensitivity_barkin_reclassified": {
            "records_reclassified": len(reclassified),
            **sensitivity,
        },
        "label_distribution_by_arm": label_distribution,
        "numeric_continuity_median_deltas": numeric_continuity,
        "input_sha256": {
            "run_manifest": contract.file_sha256(manifest_path),
            "baseline_annotations": contract.file_sha256(baseline_annotations_path),
            "baseline_binary_gate": contract.file_sha256(binary_gate_path),
            "tier1_annotation_receipt": contract.file_sha256(tier1_receipt_path),
            "tier1_annotations": contract.file_sha256(tier1_annotations_path),
            "second_rater_annotations": contract.file_sha256(
                second_annotations_path
            ),
            "second_rater_audit": contract.file_sha256(second_audit_path),
        },
    }
    result_path = output / "s16_phase_c_tier1_result.json"
    contract.atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "headline_variant": headline_variant,
                "records_reclassified": len(reclassified),
                "output": str(result_path),
                "payload_sha256": contract.load_authenticated_json(result_path)[
                    "payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
