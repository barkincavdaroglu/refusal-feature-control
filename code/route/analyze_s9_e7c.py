#!/usr/bin/env python3
"""Produce E7c's descriptive, intent-bounded route analysis.

This analyzer intentionally does not run E7d regression or claim independent
route-cell generalization.  It authenticates the frozen plan/result and reports
raw continuous contrasts, matched controls, boundary asymmetry, receiver sets,
and descriptive topology associations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from s9_contract import METRICS, canonical_sha256, sha256_file
from s9_e7_contract import validate_e7b_plan, validate_e7c_result


ANALYSIS_SEMANTICS_VERSION = "s9-e7c-descriptive-route-analysis-v1"
ANALYSIS_STATUS = "complete_descriptive_e7c_analysis_without_e7d"


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized or not all(math.isfinite(value) for value in materialized):
        raise ValueError("E7c descriptive summary received invalid values")
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "minimum": min(materialized),
        "q25": _quantile(materialized, 0.25),
        "median": statistics.median(materialized),
        "q75": _quantile(materialized, 0.75),
        "maximum": max(materialized),
        "mean_absolute": statistics.fmean(abs(value) for value in materialized),
        "median_absolute": statistics.median(abs(value) for value in materialized),
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in order[cursor:end]:
            output[position] = rank
        cursor = end
    return output


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_scale == 0.0 or y_scale == 0.0:
        return None
    return numerator / (x_scale * y_scale)


def _effect(row: Mapping[str, Any], metric: str, estimand: str) -> float:
    return float(row["endpoints"][metric][estimand])


def _summaries(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for metric in METRICS:
        output[metric] = {
            estimand: _describe(_effect(row, metric, estimand) for row in rows)
            for estimand in ("T_S", "D_R", "B_R", "Q_R")
        }
    return output


def analyze(plan: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    validate_e7b_plan(plan)
    validate_e7c_result(result, plan)
    primary = result["primary_route_cells"]
    primary_by_cell = {row["source_cell_id"]: row for row in primary}
    plan_cells = {
        row["cell_id"]: row for row in [*plan["mechanics_cells"], *plan["topology_cells"]]
    }

    panels: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    roles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        panels[str(row["panel"])].append(row)
        roles[str(row.get("role"))].append(row)

    input_output = []
    for input_row in result["input_calibrations"]:
        output_row = primary_by_cell[input_row["source_cell_id"]]
        comparison = {
            "source_cell_id": input_row["source_cell_id"],
            "direction_id": input_row["direction_id"],
            "metrics": {},
        }
        for metric in METRICS:
            comparison["metrics"][metric] = {
                estimand: {
                    "module_output": _effect(output_row, metric, estimand),
                    "module_input": _effect(input_row, metric, estimand),
                    "input_minus_output": _effect(input_row, metric, estimand)
                    - _effect(output_row, metric, estimand),
                }
                for estimand in ("D_R", "B_R", "Q_R")
            }
        input_output.append(comparison)

    low_controls = []
    for control in result["low_coverage_controls"]:
        source = primary_by_cell[control["source_cell_id"]]
        row = {
            "control_id": control["result_id"],
            "source_cell_id": control["source_cell_id"],
            "source_receiver_layer": source["receiver_layer"],
            "control_receiver_layer": control["receiver_layer"],
            "metrics": {},
        }
        for metric in METRICS:
            row["metrics"][metric] = {
                estimand: {
                    "source_receiver": _effect(source, metric, estimand),
                    "low_coverage_receiver": _effect(control, metric, estimand),
                    "absolute_source_minus_absolute_control": abs(_effect(source, metric, estimand))
                    - abs(_effect(control, metric, estimand)),
                }
                for estimand in ("D_R", "B_R", "Q_R")
            }
        low_controls.append(row)

    receiver_sets = []
    for joint in result["joint_receiver_sets"]:
        source = primary_by_cell[joint["source_cell_id"]]
        row = {
            "joint_result_id": joint["result_id"],
            "source_cell_id": joint["source_cell_id"],
            "receiver_layers": joint["receiver_layers"],
            "interpretation": joint["interpretation"],
            "metrics": {},
        }
        for metric in METRICS:
            row["metrics"][metric] = {
                estimand: {
                    "single_receiver": _effect(source, metric, estimand),
                    "ordered_receiver_set": _effect(joint, metric, estimand),
                    "set_minus_single": _effect(joint, metric, estimand)
                    - _effect(source, metric, estimand),
                }
                for estimand in ("D_R", "B_R", "Q_R")
            }
        receiver_sets.append(row)

    topology_cells = [
        row for row in primary if row["panel"] == "topology_discrimination"
    ]
    topology_correlations = {}
    predictor_names = {
        "unsigned_receiver_coverage": lambda row: float(row["topology"]["unsigned"]["receiver_coverage"]),
        "unsigned_bypass_share": lambda row: float(row["topology"]["unsigned"]["bypass_share"]),
        "signed_absolute_predicted_delta": lambda row: abs(float(row["topology"]["signed"]["predicted_receiver_preactivation_delta"])),
        "signed_cancellation_ratio": lambda row: float(row["topology"]["signed"]["cancellation_ratio"]),
    }
    for metric in METRICS:
        topology_correlations[metric] = {}
        for estimand in ("D_R", "B_R", "Q_R"):
            target = [abs(_effect(row, metric, estimand)) for row in topology_cells]
            topology_correlations[metric][estimand] = {
                name: {
                    "n_cells": len(target),
                    "spearman_r": _pearson(_ranks([reader(row) for row in topology_cells]), _ranks(target)),
                    "claim_boundary": "descriptive across correlated development cells; not E7d or intent-level validation",
                }
                for name, reader in predictor_names.items()
            }

    topology_contrasts = []
    topology_plan_cells = [row for row in plan["topology_cells"]]
    for contrast in plan["topology_contrasts"]:
        first_plan = next(
            row for row in topology_plan_cells
            if row["direction_id"] == contrast["direction_id"]
            and row["cell_id"].endswith(":" + contrast["first_candidate_id"])
        )
        second_plan = next(
            row for row in topology_plan_cells
            if row["direction_id"] == contrast["direction_id"]
            and row["cell_id"].endswith(":" + contrast["second_candidate_id"])
        )
        first = primary_by_cell[first_plan["cell_id"]]
        second = primary_by_cell[second_plan["cell_id"]]
        row = {**contrast, "exact_ordering": {}}
        for metric in METRICS:
            row["exact_ordering"][metric] = {}
            for estimand in ("D_R", "B_R", "Q_R"):
                margin = abs(_effect(first, metric, estimand)) - abs(_effect(second, metric, estimand))
                row["exact_ordering"][metric][estimand] = {
                    "absolute_margin_first_minus_second": margin,
                    "order": "tie" if margin == 0.0 else ("first_larger" if margin > 0 else "second_larger"),
                }
        topology_contrasts.append(row)

    receiver_movement = {
        "primary_receiver_moved": sum(bool(row["receiver_state"]["receiver_moved"]) for row in primary),
        "primary_receiver_unchanged": sum(not bool(row["receiver_state"]["receiver_moved"]) for row in primary),
        "nominated_feature_gate_crossings": sum(
            bool(row["receiver_feature_diagnostic"]["finite_gate_crossing"])
            for row in primary
            if row.get("receiver_feature_diagnostic") is not None
        ),
        "changed_receivers_with_near_zero_both_endpoints_D_and_Q": sum(
            bool(row["receiver_state"]["receiver_moved"])
            and all(
                abs(_effect(row, metric, "D_R")) <= 1e-8
                and abs(_effect(row, metric, "Q_R")) <= 1e-8
                for metric in METRICS
            )
            for row in primary
        ),
        "near_zero_definition": "absolute raw contrast <= 1e-8; numerical diagnostic only, not a scientific effect threshold",
    }

    intent_ids = sorted({row["intent_id"] for row in plan["prompt_records"]})
    payload = {
        "analysis_semantics_version": ANALYSIS_SEMANTICS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": ANALYSIS_STATUS,
        "plan_payload_sha256": plan["payload_sha256"],
        "result_payload_sha256": result["payload_sha256"],
        "claim_boundary": {
            "development_only": True,
            "independent_intents": intent_ids,
            "independent_intent_count": len(intent_ids),
            "route_cells_are_correlated": True,
            "e7d_incremental_topology_analysis_executed": False,
            "population_predictive_claim_allowed": False,
            "generation_or_behavioral_flip_tested": False,
        },
        "forward_accounting": result["forward_accounting"],
        "primary_summary": _summaries(primary),
        "panel_summaries": {name: _summaries(rows) for name, rows in sorted(panels.items())},
        "role_summaries": {name: _summaries(rows) for name, rows in sorted(roles.items())},
        "input_output_boundary_comparisons": input_output,
        "low_coverage_control_comparisons": low_controls,
        "ordered_receiver_set_comparisons": receiver_sets,
        "topology_contrast_adjudications": topology_contrasts,
        "topology_descriptive_correlations": topology_correlations,
        "receiver_movement": receiver_movement,
        "invariant_audit": {
            "rows": sum(
                len(result[key])
                for key in (
                    "primary_route_cells",
                    "input_calibrations",
                    "low_coverage_controls",
                    "joint_receiver_sets",
                )
            ),
            "metrics_per_row": len(METRICS),
            "all_T_equals_D_plus_B": True,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    payload = analyze(plan, result)
    payload["input_file_sha256"] = {
        "plan": sha256_file(args.plan),
        "result": sha256_file(args.result),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "independent_intent_count": payload["claim_boundary"]["independent_intent_count"],
                "e7d_executed": payload["claim_boundary"]["e7d_incremental_topology_analysis_executed"],
                "payload_sha256": payload["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
