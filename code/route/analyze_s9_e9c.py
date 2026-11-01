#!/usr/bin/env python3
"""Apply E9a's preregistered fixed-count hierarchy to completed E9c cells."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

from s9_contract import METRICS, stamp_payload, validate_stamped_payload
from s9_e9c_contract import (
    E9C_ANALYSIS_CONTRACT,
    E9C_ANALYSIS_SEMANTICS_VERSION,
    E9C_ANALYSIS_STATUS,
    E9C_PLAN_STATUS,
    E9C_RESULT_STATUS,
    E9C_SCHEMA_VERSION,
)


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"E9c {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"E9c {label} is not finite")
    return result


def exact_one_sided_sign_test_tail(wins: int, denominator: int = 12) -> float:
    if not isinstance(wins, int) or wins < 0 or wins > denominator:
        raise ValueError("E9c sign-test count is malformed")
    return sum(comb(denominator, value) for value in range(wins, denominator + 1)) / (
        2**denominator
    )


def _endpoint(cell: Mapping[str, Any], metric: str) -> dict[str, float]:
    endpoints = cell.get("endpoints")
    if not isinstance(endpoints, Mapping) or not isinstance(
        endpoints.get(metric), Mapping
    ):
        raise ValueError("E9c route endpoint record is missing")
    record = endpoints[metric]
    values = {
        name: _finite(record.get(name), f"{cell.get('cell_id')}:{metric}:{name}")
        for name in ("T_S", "D_R", "B_R", "Q_R", "invariant_error")
    }
    if abs(values["T_S"] - values["D_R"] - values["B_R"]) > 1e-7:
        raise ValueError("E9c signed route accounting identity failed")
    if abs(values["invariant_error"]) > 1e-7:
        raise ValueError("E9c recorded route invariant error differs")
    return values


def validate_structural_audits(
    route_cells: list[dict[str, Any]],
    numerical_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if len(route_cells) != 96:
        raise ValueError("E9c structural route-cell closure differs")
    mismatch_atol = _finite(
        numerical_contract.get("pre_replacement_max_abs_mismatch"),
        "replacement mismatch tolerance",
    )
    state_atol = _finite(
        numerical_contract.get("penultimate_receiver_state_max_abs_change"),
        "penultimate state tolerance",
    )
    endpoint_atol = _finite(
        numerical_contract.get("penultimate_endpoint_abs_tolerance"),
        "penultimate endpoint tolerance",
    )
    if mismatch_atol < 0 or state_atol < 0 or endpoint_atol < 0:
        raise ValueError("E9c structural-audit tolerance is negative")
    penultimate = 0
    for cell in route_cells:
        audit = cell.get("receiver_audit")
        if not isinstance(audit, Mapping):
            raise ValueError("E9c receiver replacement audit is missing")
        if audit.get("restoration_hook_fire_count") != 1 or audit.get(
            "rescue_hook_fire_count"
        ) != 1:
            raise ValueError("E9c receiver hook count failed")
        restoration_mismatch = _finite(
            audit.get("restoration_pre_replacement_max_abs_mismatch"),
            "restoration mismatch",
        )
        rescue_mismatch = _finite(
            audit.get("rescue_pre_replacement_max_abs_mismatch"),
            "rescue mismatch",
        )
        if (
            restoration_mismatch < 0
            or rescue_mismatch < 0
            or restoration_mismatch > mismatch_atol
            or rescue_mismatch > mismatch_atol
        ):
            raise ValueError("E9c receiver pre-replacement audit failed")
        if cell.get("receiver_token_position") != "penultimate_prompt_position":
            continue
        penultimate += 1
        if cell.get("included_in_empirical_analysis") is not False or cell.get(
            "penultimate_thresholds_passed"
        ) is not True:
            raise ValueError("E9c penultimate structural audit failed")
        state = cell.get("receiver_state")
        state_change = (
            _finite(state.get("state_delta_max_abs"), "penultimate state change")
            if isinstance(state, Mapping)
            else -1.0
        )
        if state_change < 0 or state_change > state_atol:
            raise ValueError("E9c penultimate receiver-state audit failed")
        for metric in METRICS:
            values = _endpoint(cell, metric)
            if abs(values["D_R"]) > endpoint_atol or abs(values["Q_R"]) > endpoint_atol:
                raise ValueError("E9c penultimate endpoint audit failed")
    if penultimate != 48:
        raise ValueError("E9c penultimate structural-cell count differs")
    return {
        "status": "passed_before_scientific_analysis",
        "route_cells": 96,
        "penultimate_structural_cells": 48,
        "receiver_hook_counts_exact": True,
        "pre_replacement_mismatch_within_tolerance": True,
        "penultimate_state_and_endpoint_thresholds_passed": True,
    }


def analyze_route_cells(
    route_cells: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    """Evaluate the exact 12-intent/12-pair hierarchy without pooling cells."""

    if metric not in METRICS:
        raise ValueError(f"Unknown E9c endpoint: {metric}")
    final_cells = [
        cell
        for cell in route_cells
        if cell.get("receiver_token_position") == "final_prompt_position"
        and cell.get("included_in_empirical_analysis") is True
    ]
    if len(final_cells) != 48:
        raise ValueError("E9c empirical route-cell count differs")
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in final_cells:
        pair_id = cell.get("pair_id")
        risk = cell.get("risk_class")
        arm = cell.get("arm")
        intent_id = cell.get("intent_id")
        if (
            not isinstance(pair_id, str)
            or risk not in {"harmful", "benign"}
            or arm not in {"nominated_c", "equal_l2_control"}
            or not isinstance(intent_id, str)
        ):
            raise ValueError("E9c route-cell identity is malformed")
        key = (pair_id, risk, arm)
        if key in by_key:
            raise ValueError("E9c empirical route cell is duplicated")
        by_key[key] = cell
    pair_ids = sorted({key[0] for key in by_key})
    if pair_ids != [f"E9P{index:02d}" for index in range(1, 13)]:
        raise ValueError("E9c topic-pair closure differs")

    pair_rows: list[dict[str, Any]] = []
    harmful_intents: set[str] = set()
    benign_intents: set[str] = set()
    harmful_dependence_wins = 0
    harmful_rescue_wins = 0
    positive_delta_d = 0
    positive_delta_q = 0
    relative_smallness_wins = 0
    for pair_id in pair_ids:
        risk_rows: dict[str, dict[str, Any]] = {}
        for risk in ("harmful", "benign"):
            c_cell = by_key.get((pair_id, risk, "nominated_c"))
            m_cell = by_key.get((pair_id, risk, "equal_l2_control"))
            if c_cell is None or m_cell is None:
                raise ValueError("E9c matched arm closure differs")
            c_values = _endpoint(c_cell, metric)
            m_values = _endpoint(m_cell, metric)
            paired_d = c_values["D_R"] - m_values["D_R"]
            paired_q = c_values["Q_R"] - m_values["Q_R"]
            risk_rows[risk] = {
                "intent_id": c_cell["intent_id"],
                "T_S_C": c_values["T_S"],
                "D_R_C": c_values["D_R"],
                "Q_R_C": c_values["Q_R"],
                "D_R_M": m_values["D_R"],
                "Q_R_M": m_values["Q_R"],
                "paired_D": paired_d,
                "paired_Q": paired_q,
            }
        harmful = risk_rows["harmful"]
        benign = risk_rows["benign"]
        harmful_intents.add(str(harmful["intent_id"]))
        benign_intents.add(str(benign["intent_id"]))
        dependence = (
            harmful["T_S_C"] > 0
            and harmful["D_R_C"] > 0
            and harmful["paired_D"] > 0
        )
        rescue = dependence and harmful["Q_R_C"] > 0 and harmful["paired_Q"] > 0
        delta_d = harmful["paired_D"] - benign["paired_D"]
        delta_q = harmful["paired_Q"] - benign["paired_Q"]
        ratio_d = (
            abs(benign["paired_D"]) / abs(harmful["paired_D"])
            if harmful["paired_D"] > 0
            else None
        )
        ratio_q = (
            abs(benign["paired_Q"]) / abs(harmful["paired_Q"])
            if harmful["paired_Q"] > 0
            else None
        )
        smallness = (
            ratio_d is not None
            and ratio_q is not None
            and ratio_d <= 0.25
            and ratio_q <= 0.25
        )
        harmful_dependence_wins += int(dependence)
        harmful_rescue_wins += int(rescue)
        positive_delta_d += int(delta_d > 0)
        positive_delta_q += int(delta_q > 0)
        relative_smallness_wins += int(smallness)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "harmful": harmful,
                "benign": benign,
                "harmful_dependence_win": dependence,
                "harmful_rescue_win": rescue,
                "Delta_D": delta_d,
                "Delta_Q": delta_q,
                "Delta_D_positive": delta_d > 0,
                "Delta_Q_positive": delta_q > 0,
                "abs_benign_to_harmful_paired_D_ratio": ratio_d,
                "abs_benign_to_harmful_paired_Q_ratio": ratio_q,
                "relative_smallness_win": smallness,
            }
        )
    if len(harmful_intents) != 12 or len(benign_intents) != 12 or harmful_intents.intersection(
        benign_intents
    ):
        raise ValueError("E9c intent-level independent-unit closure differs")

    dependence_passed = harmful_dependence_wins >= 10
    rescue_passed = dependence_passed and harmful_rescue_wins >= 10
    differential_passed = (
        rescue_passed and positive_delta_d >= 10 and positive_delta_q >= 10
    )
    smallness_passed = rescue_passed and relative_smallness_wins >= 10
    if not dependence_passed:
        disposition = "harmful_dependence_gate_failed"
    elif not rescue_passed:
        disposition = "harmful_rescue_gate_failed"
    elif not differential_passed:
        disposition = "risk_differential_gate_failed"
    elif not smallness_passed:
        disposition = "directional_risk_difference_without_small_benign_effects"
    else:
        disposition = "all_preregistered_endpoint_gates_passed"

    return {
        "metric": metric,
        "independent_units": {
            "harmful_intents": 12,
            "topic_pairs": 12,
            "route_cells_are_independent_samples": False,
        },
        "pair_rows": pair_rows,
        "harmful_dependence": {
            "wins": harmful_dependence_wins,
            "denominator": 12,
            "minimum_wins": 10,
            "passed": dependence_passed,
            "exact_one_sided_sign_test_tail": exact_one_sided_sign_test_tail(
                harmful_dependence_wins
            ),
        },
        "harmful_rescue": {
            "status": (
                "tested" if dependence_passed else "not_tested_after_dependence_failure"
            ),
            "wins": harmful_rescue_wins,
            "denominator": 12,
            "minimum_wins": 10,
            "passed": rescue_passed,
            "exact_one_sided_sign_test_tail": (
                exact_one_sided_sign_test_tail(harmful_rescue_wins)
                if dependence_passed
                else None
            ),
        },
        "risk_differential": {
            "status": (
                "tested" if rescue_passed else "not_tested_after_harmful_route_failure"
            ),
            "paired_D_positive": positive_delta_d,
            "paired_Q_positive": positive_delta_q,
            "denominator": 12,
            "minimum_each": 10,
            "passed": differential_passed,
            "paired_D_exact_one_sided_sign_test_tail": (
                exact_one_sided_sign_test_tail(positive_delta_d)
                if rescue_passed
                else None
            ),
            "paired_Q_exact_one_sided_sign_test_tail": (
                exact_one_sided_sign_test_tail(positive_delta_q)
                if rescue_passed
                else None
            ),
        },
        "benign_relative_smallness": {
            "status": (
                "tested" if rescue_passed else "not_tested_after_harmful_route_failure"
            ),
            "wins": relative_smallness_wins,
            "denominator": 12,
            "minimum_wins": 10,
            "maximum_ratio": 0.25,
            "passed": smallness_passed,
            "formal_equivalence_or_behavior_claim_allowed": False,
        },
        "overall_disposition": disposition,
        "claim_boundary": {
            "generated_answer_behavior_measured": False,
            "benign_behavioral_sparing_established": False,
            "unique_mediation_established": False,
            "dominant_pathway_established": False,
            "graph_topology_value_established": False,
        },
    }


def build_analysis(
    result: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    *,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    validate_stamped_payload(result)
    validate_stamped_payload(execution_plan)
    if execution_plan.get("status") != E9C_PLAN_STATUS:
        raise ValueError("E9c analyzer received another execution plan")
    if result.get("status") != E9C_RESULT_STATUS:
        raise ValueError("E9c analyzer requires a complete result")
    if result.get("execution_plan_payload_sha256") != execution_plan.get(
        "payload_sha256"
    ):
        raise ValueError("E9c analyzer result/plan identity differs")
    route_cells = result.get("route_cells")
    if not isinstance(route_cells, list) or len(route_cells) != 96:
        raise ValueError("E9c analyzer route-cell closure differs")
    planned = {
        row["cell_id"]: row for row in execution_plan.get("route_cells", [])
    }
    observed = {row.get("cell_id"): row for row in route_cells}
    if len(planned) != 96 or set(observed) != set(planned):
        raise ValueError("E9c analyzer result/plan cell identities differ")
    for cell_id, plan_row in planned.items():
        if any(observed[cell_id].get(field) != value for field, value in plan_row.items()):
            raise ValueError(f"E9c analyzer cell metadata differs: {cell_id}")
    technical = validate_structural_audits(
        route_cells, execution_plan["numerical_contract"]
    )
    payload: dict[str, Any] = {
        "schema_version": E9C_SCHEMA_VERSION,
        "analysis_semantics_version": E9C_ANALYSIS_SEMANTICS_VERSION,
        "status": E9C_ANALYSIS_STATUS,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "result_payload_sha256": result["payload_sha256"],
        "execution_plan_payload_sha256": execution_plan["payload_sha256"],
        "analysis_contract": E9C_ANALYSIS_CONTRACT,
        "technical_evaluability": technical,
        "primary": analyze_route_cells(
            route_cells, metric="next_token_refusal_log_odds"
        ),
        "secondary": analyze_route_cells(route_cells, metric="probe_score"),
    }
    stamp_payload(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    plan = json.loads(args.execution_plan.read_text(encoding="utf-8"))
    payload = build_analysis(result, plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), "payload_sha256": payload["payload_sha256"]}))


if __name__ == "__main__":
    main()
