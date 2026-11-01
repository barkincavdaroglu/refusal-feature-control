#!/usr/bin/env python3
"""Deterministic, intent-level analysis for the frozen S9 E8 replication.

The compound 9/10 dependence and rescue rules are count thresholds, not sign
tests.  Exact one-sided sign tests are reported only for the two scalar paired
contrasts.  Gate crossing is a separate advancement diagnostic, and the
topic-matched analysis fails closed when its harmful/benign dose ratio cannot
be evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from s9_contract import canonical_sha256, sha256_file
from s9_e8b_contract import (
    E8B_ANALYSIS_SEMANTICS_VERSION,
    E8B_ANALYSIS_STATUS,
    E8B_NUMERICAL_CONTRACT,
    E8B_SEMANTIC_RESOLUTION,
    validate_e8b_result,
)


ANALYSIS_SEMANTICS_VERSION = E8B_ANALYSIS_SEMANTICS_VERSION
ANALYSIS_STATUS = E8B_ANALYSIS_STATUS
PRIMARY_ENDPOINT = "next_token_refusal_log_odds"
ARMS = ("nominated_c", "equal_l2_control")
POSITIONS = ("final_prompt_position", "penultimate_prompt_position")
ESTIMANDS = ("T_S", "D_R", "B_R", "Q_R")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"E8b analysis requires numeric {label}")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"E8b analysis requires finite {label}")
    return output


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    materialized = [_number(value, "descriptive value") for value in values]
    if not materialized:
        raise ValueError("E8b analysis cannot summarize an empty group")
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "minimum": min(materialized),
        "q25": _quantile(materialized, 0.25),
        "median": statistics.median(materialized),
        "q75": _quantile(materialized, 0.75),
        "maximum": max(materialized),
    }


def _is_win(value: float, threshold: float = 0.0) -> bool:
    """Apply the frozen strict-greater-than rule at an explicit threshold."""

    return value > threshold


def _exact_one_sided_sign_test(values: Sequence[float], threshold: float) -> dict[str, Any]:
    """Return the fixed-denominator exact binomial upper tail.

    E8a keeps all ten harmful intents in the denominator. Values at or below
    the explicit scalar threshold, including exact ties, are nonwins rather
    than dropped observations.
    """

    wins = sum(_is_win(value, threshold) for value in values)
    trials = len(values)
    p_value = sum(math.comb(trials, k) for k in range(wins, trials + 1)) / (2**trials)
    return {
        "test": "exact_one_sided_fixed_denominator_sign_test",
        "independent_unit": "intent_id",
        "wins": wins,
        "nonwins_including_ties": trials - wins,
        "denominator": trials,
        "win_definition": f"scalar paired estimand > {threshold:g}",
        "nonwin_threshold": threshold,
        "p_value_exact_one_sided": p_value,
    }


def _validate_minimal_shape(
    plan: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    plan_cells = plan.get("route_cells")
    result_cells = result.get("route_cells")
    if not isinstance(plan_cells, list) or len(plan_cells) != 60:
        raise ValueError("E8b analysis requires the frozen 60-cell E8a plan")
    if not isinstance(result_cells, list) or len(result_cells) != 60:
        raise ValueError("E8b route-cell closure requires exactly 60 result rows")
    if not all(isinstance(row, Mapping) for row in result_cells):
        raise ValueError("E8b route-cell result is malformed")

    plan_by_id = {row.get("cell_id"): row for row in plan_cells}
    if len(plan_by_id) != 60 or None in plan_by_id:
        raise ValueError("E8a plan contains duplicate or missing route-cell identities")
    result_ids = [row.get("source_cell_id", row.get("cell_id")) for row in result_cells]
    if len(set(result_ids)) != 60:
        raise ValueError("E8b result contains duplicate route-cell identities")
    if set(result_ids) != set(plan_by_id):
        raise ValueError("E8b result route-cell closure differs from the frozen plan")

    for row, source_id in zip(result_cells, result_ids):
        planned = plan_by_id[source_id]
        for key in ("intent_id", "risk_class", "arm", "receiver_token_position"):
            if row.get(key) != planned.get(key):
                raise ValueError(f"E8b route row {source_id} differs on {key}")
        endpoints = row.get("endpoints")
        expected_metrics = set(plan.get("endpoints", {}).get("raw_reported", ()))
        if not isinstance(endpoints, Mapping) or set(endpoints) != expected_metrics:
            raise ValueError(f"E8b route row {source_id} endpoint set differs")
        for metric in expected_metrics:
            endpoint = endpoints[metric]
            if not isinstance(endpoint, Mapping):
                raise ValueError(f"E8b route row {source_id}.{metric} is malformed")
            for field in ("Y0", "YS", "YS_R0", "YR_S", *ESTIMANDS, "invariant_error"):
                value = _number(endpoint.get(field), f"{source_id}.{metric}.{field}")
                if field == "invariant_error" and value < 0.0:
                    raise ValueError(
                        f"E8b route row {source_id}.{metric}.invariant_error must be absolute"
                    )
        if not isinstance(row.get("gate_diagnostic"), Mapping):
            raise ValueError(f"E8b route row {source_id} lacks gate diagnostics")
        if not isinstance(row.get("receiver_audit"), Mapping):
            raise ValueError(f"E8b route row {source_id} lacks receiver audit")

    doses = result.get("dose_audits")
    intent_pairs = plan.get("intent_pairs")
    if not isinstance(intent_pairs, list) or len(intent_pairs) != 15:
        raise ValueError("E8b analysis requires all 15 frozen intent pairs")
    if not isinstance(doses, list) or len(doses) != 15 or not all(
        isinstance(row, Mapping) for row in doses
    ):
        raise ValueError("E8b analysis requires exactly 15 dose audits")
    dose_by_intent = {row.get("intent_id"): row for row in doses}
    expected_intents = {row.get("intent_id") for row in intent_pairs}
    if len(dose_by_intent) != 15 or set(dose_by_intent) != expected_intents:
        raise ValueError("E8b dose-audit intent closure differs")
    for intent_id, dose in dose_by_intent.items():
        for field in (
            "nominated_activation_delta",
            "control_activation_delta",
            "nominal_nominated_edit_l2",
            "nominal_control_edit_l2",
            "realized_bfloat16_nominated_edit_l2",
            "realized_bfloat16_control_edit_l2",
            "nominal_equal_l2_relative_error",
            "realized_bfloat16_equal_l2_relative_error",
        ):
            _number(dose.get(field), f"{intent_id}.{field}")

    if not isinstance(result.get("prompt_records"), list) or len(result["prompt_records"]) != 30:
        raise ValueError("E8b result must contain 30 authenticated prompt records")
    sender_worlds = result.get("sender_worlds")
    if not isinstance(sender_worlds, list) or len(sender_worlds) != 30:
        raise ValueError("E8b result must contain 30 sender worlds")
    if not all(isinstance(row, Mapping) for row in sender_worlds):
        raise ValueError("E8b sender-world record is malformed")
    sender_keys = [(row.get("intent_id"), row.get("arm")) for row in sender_worlds]
    expected_sender_keys = {
        (intent_id, arm) for intent_id in expected_intents for arm in ARMS
    }
    if len(set(sender_keys)) != 30 or set(sender_keys) != expected_sender_keys:
        raise ValueError("E8b sender-world intent/arm closure differs")
    expected_metrics = set(plan.get("endpoints", {}).get("raw_reported", ()))
    for sender in sender_worlds:
        endpoints = sender.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != expected_metrics:
            raise ValueError("E8b sender-world endpoint set differs")
        for metric in expected_metrics:
            record = endpoints[metric]
            if not isinstance(record, Mapping):
                raise ValueError("E8b sender-world endpoint is malformed")
            for field in ("Y0", "YS", "T_S"):
                _number(record.get(field), f"sender world {metric}.{field}")
    return result_cells, dose_by_intent


def _endpoint(row: Mapping[str, Any], metric: str, estimand: str) -> float:
    return _number(
        row["endpoints"][metric][estimand],
        f"{row.get('source_cell_id', row.get('cell_id'))}.{metric}.{estimand}",
    )


def _technical_evaluability(
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    doses: Mapping[str, Mapping[str, Any]],
    authoritative_validator_executed: bool,
) -> dict[str, Any]:
    contract = plan["technical_evaluability"]
    invariant_tol = _number(contract["invariant_absolute_tolerance"], "invariant tolerance")
    mismatch_tol = _number(
        contract["pre_replacement_max_abs_mismatch"], "replacement mismatch tolerance"
    )
    l2_tol = _number(contract["equal_l2_relative_tolerance"], "equal-L2 tolerance")
    state_tol = _number(
        contract["penultimate_receiver_state_max_abs_change"], "penultimate state tolerance"
    )
    endpoint_tol = _number(
        contract["penultimate_endpoint_abs_tolerance"], "penultimate endpoint tolerance"
    )

    forward = result.get("forward_accounting", {})
    forward_ok = (
        isinstance(forward, Mapping)
        and forward.get("planned_scientific") == 180
        and forward.get("completed_scientific") == 180
        and isinstance(forward.get("audit_forwards"), int)
        and 0 <= forward["audit_forwards"] <= 18
        and forward.get("cumulative_total_across_resumes")
        == 180 + forward["audit_forwards"]
        and forward.get("audit_ceiling") == 198
        and forward.get("audit_ceiling_exceeded") is False
        and forward.get("counter_reset_on_resume") is False
    )

    max_world_error = 0.0
    max_invariant_error = 0.0
    max_reported_invariant_error = 0.0
    max_reported_invariant_disagreement = 0.0
    max_mismatch = 0.0
    hooks_ok = True
    penultimate_max_state = 0.0
    penultimate_max_endpoint = 0.0
    cross_position_max_error = 0.0
    sender_world_max_error = 0.0
    across_arm_y0_max_error = 0.0
    gate_labels_ok = True
    gate_activation_ok = True
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    y0_by_intent_metric: dict[tuple[str, str], list[float]] = {}
    sender_worlds = {
        (str(row["intent_id"]), str(row["arm"])): row
        for row in result["sender_worlds"]
    }

    for row in rows:
        audit = row["receiver_audit"]
        hooks_ok = hooks_ok and all(
            audit.get(name) == 1
            for name in ("restoration_hook_fire_count", "rescue_hook_fire_count")
        )
        for name in (
            "restoration_pre_replacement_max_abs_mismatch",
            "rescue_pre_replacement_max_abs_mismatch",
        ):
            max_mismatch = max(max_mismatch, abs(_number(audit.get(name), name)))

        gate = row["gate_diagnostic"]
        pre0 = _number(gate.get("baseline_preactivation"), "baseline gate preactivation")
        pre_s = _number(gate.get("sender_preactivation"), "sender gate preactivation")
        expected_class = (
            "inactive_to_active"
            if pre0 <= 0 < pre_s
            else "active_to_inactive"
            if pre_s <= 0 < pre0
            else "active_stable"
            if pre0 > 0 and pre_s > 0
            else "inactive_stable"
        )
        gate_labels_ok = gate_labels_ok and gate.get("crossing_class") == expected_class
        if "gate_crossing" in gate:
            gate_labels_ok = gate_labels_ok and gate["gate_crossing"] is (
                expected_class == "inactive_to_active"
            )
        act0 = _number(gate.get("baseline_activation"), "baseline gate activation")
        act_s = _number(gate.get("sender_activation"), "sender gate activation")
        gate_activation_ok = gate_activation_ok and (
            abs(act0 - max(pre0, 0.0)) <= 1e-6
            and abs(act_s - max(pre_s, 0.0)) <= 1e-6
        )

        for metric, endpoint in row["endpoints"].items():
            y0 = _number(endpoint["Y0"], "Y0")
            ys = _number(endpoint["YS"], "YS")
            ys_r0 = _number(endpoint["YS_R0"], "YS_R0")
            yr_s = _number(endpoint["YR_S"], "YR_S")
            t_s = _number(endpoint["T_S"], "T_S")
            d_r = _number(endpoint["D_R"], "D_R")
            b_r = _number(endpoint["B_R"], "B_R")
            q_r = _number(endpoint["Q_R"], "Q_R")
            max_world_error = max(
                max_world_error,
                abs(t_s - (ys - y0)),
                abs(d_r - (ys - ys_r0)),
                abs(b_r - (ys_r0 - y0)),
                abs(q_r - (yr_s - y0)),
            )
            computed_invariant_error = abs(t_s - d_r - b_r)
            max_invariant_error = max(max_invariant_error, computed_invariant_error)
            reported_invariant_error = _number(
                endpoint["invariant_error"], "reported invariant error"
            )
            max_reported_invariant_error = max(
                max_reported_invariant_error,
                reported_invariant_error,
            )
            max_reported_invariant_disagreement = max(
                max_reported_invariant_disagreement,
                abs(reported_invariant_error - computed_invariant_error),
            )
            y0_by_intent_metric.setdefault((str(row["intent_id"]), metric), []).append(y0)
            if row["receiver_token_position"] == "penultimate_prompt_position":
                penultimate_max_endpoint = max(
                    penultimate_max_endpoint, abs(d_r), abs(q_r)
                )

        if row["receiver_token_position"] == "penultimate_prompt_position":
            state = row.get("receiver_state")
            if not isinstance(state, Mapping):
                raise ValueError("E8b penultimate route row lacks receiver-state metadata")
            state_change = _number(
                state.get("state_delta_max_abs"), "receiver state delta max abs"
            )
            penultimate_max_state = max(penultimate_max_state, abs(state_change))

        key = (str(row["intent_id"]), str(row["arm"]))
        grouped.setdefault(key, {})[str(row["receiver_token_position"])] = row
        sender = sender_worlds[key]
        for metric in row["endpoints"]:
            for field in ("Y0", "YS", "T_S"):
                sender_world_max_error = max(
                    sender_world_max_error,
                    abs(
                        _endpoint(row, metric, field)
                        - _number(
                            sender["endpoints"][metric][field],
                            f"sender world {metric}.{field}",
                        )
                    ),
                )

    for position_rows in grouped.values():
        final = position_rows["final_prompt_position"]
        penultimate = position_rows["penultimate_prompt_position"]
        for metric in final["endpoints"]:
            for field in ("Y0", "YS", "T_S"):
                cross_position_max_error = max(
                    cross_position_max_error,
                    abs(_endpoint(final, metric, field) - _endpoint(penultimate, metric, field)),
                )

    for values in y0_by_intent_metric.values():
        across_arm_y0_max_error = max(
            across_arm_y0_max_error, max(values) - min(values)
        )

    max_equal_l2_error = 0.0
    dose_fields_consistent = True
    zero_denominator_mismatch = False
    zero_dose_contract_ok = True
    control_dose_sign_ok = True
    realized_report_consistent = True
    for dose in doses.values():
        delta_c = _number(dose["nominated_activation_delta"], "nominated activation delta")
        delta_m = _number(dose["control_activation_delta"], "control activation delta")
        nominated = _number(dose["nominal_nominated_edit_l2"], "nominal nominated L2")
        control = _number(dose["nominal_control_edit_l2"], "nominal control L2")
        if nominated == 0.0:
            zero_denominator_mismatch = zero_denominator_mismatch or control != 0.0
            computed = 0.0
        else:
            computed = abs(control - nominated) / abs(nominated)
        reported = abs(
            _number(
                dose["nominal_equal_l2_relative_error"],
                "nominal equal-L2 relative error",
            )
        )
        max_equal_l2_error = max(max_equal_l2_error, computed, reported)
        dose_fields_consistent = dose_fields_consistent and abs(computed - reported) <= invariant_tol
        realized_c = _number(
            dose["realized_bfloat16_nominated_edit_l2"], "realized nominated L2"
        )
        realized_m = _number(
            dose["realized_bfloat16_control_edit_l2"], "realized control L2"
        )
        realized_report = _number(
            dose["realized_bfloat16_equal_l2_relative_error"],
            "realized equal-L2 relative error",
        )
        if realized_c == 0.0:
            realized_computed = 0.0 if realized_m == 0.0 else None
        else:
            realized_computed = abs(realized_m - realized_c) / abs(realized_c)
        realized_report_consistent = realized_report_consistent and (
            realized_computed is not None
            and abs(realized_computed - realized_report) <= invariant_tol
        )
        zero_expected = delta_c == 0.0 and nominated == 0.0
        zero_dose_contract_ok = zero_dose_contract_ok and (
            dose.get("zero_dose") is zero_expected
        )
        if zero_expected:
            zero_dose_contract_ok = zero_dose_contract_ok and all(
                value == 0.0
                for value in (
                    delta_m,
                    control,
                    reported,
                    realized_c,
                    realized_m,
                    realized_report,
                )
            )
        else:
            zero_dose_contract_ok = zero_dose_contract_ok and nominated > 0.0
            control_dose_sign_ok = control_dose_sign_ok and delta_c * delta_m >= 0.0

    checks = {
        "route_cell_closure": True,
        "forward_accounting": forward_ok,
        "endpoint_world_identities": max_world_error <= invariant_tol,
        "T_equals_D_plus_B": (
            max_invariant_error <= invariant_tol
            and max_reported_invariant_error <= invariant_tol
            and max_reported_invariant_disagreement <= invariant_tol
        ),
        "cross_position_sender_world_identity": cross_position_max_error <= invariant_tol,
        "sender_world_route_identity": sender_world_max_error <= invariant_tol,
        "across_arm_baseline_Y0_identity": across_arm_y0_max_error <= invariant_tol,
        "one_hook_fire_per_capture": hooks_ok,
        "pre_replacement_receiver_match": max_mismatch <= mismatch_tol,
        "nominal_float32_equal_l2": (
            max_equal_l2_error <= l2_tol
            and dose_fields_consistent
            and not zero_denominator_mismatch
        ),
        "zero_dose_contract": zero_dose_contract_ok,
        "control_dose_sign": control_dose_sign_ok,
        "realized_bfloat16_l2_report_consistency": realized_report_consistent,
        "penultimate_causal_order_audit": (
            penultimate_max_state <= state_tol
            and penultimate_max_endpoint <= endpoint_tol
        ),
        "gate_classification_consistency": gate_labels_ok and gate_activation_ok,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "evaluable" if not failed else "not_evaluable",
        "checks": checks,
        "failed_checks": failed,
        "authoritative_e8b_contract_validator_executed": authoritative_validator_executed,
        "thresholds": {
            "invariant_absolute_tolerance": invariant_tol,
            "pre_replacement_max_abs_mismatch": mismatch_tol,
            "nominal_float32_equal_l2_relative_tolerance": l2_tol,
            "penultimate_receiver_state_max_abs_change": state_tol,
            "penultimate_endpoint_abs_tolerance": endpoint_tol,
        },
        "observed_maxima": {
            "endpoint_world_identity_error": max_world_error,
            "T_minus_D_minus_B_absolute_error": max_invariant_error,
            "reported_invariant_error": max_reported_invariant_error,
            "reported_vs_computed_invariant_error": (
                max_reported_invariant_disagreement
            ),
            "cross_position_sender_world_error": cross_position_max_error,
            "sender_world_route_error": sender_world_max_error,
            "across_arm_baseline_Y0_error": across_arm_y0_max_error,
            "pre_replacement_receiver_mismatch": max_mismatch,
            "nominal_float32_equal_l2_relative_error": max_equal_l2_error,
            "penultimate_receiver_state_max_abs_change": penultimate_max_state,
            "penultimate_endpoint_absolute_effect": penultimate_max_endpoint,
        },
        "invalid_runtime_interpretation": (
            "audit failures are semantic/runtime failures, not negative mechanism evidence"
        ),
    }


def _intent_rows(
    plan: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    doses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final = {
        (str(row["intent_id"]), str(row["arm"])): row
        for row in rows
        if row["receiver_token_position"] == "final_prompt_position"
    }
    output = []
    for pair in plan["intent_pairs"]:
        intent_id = str(pair["intent_id"])
        nominated = final[(intent_id, "nominated_c")]
        control = final[(intent_id, "equal_l2_control")]
        nominated_endpoint = nominated["endpoints"][PRIMARY_ENDPOINT]
        control_endpoint = control["endpoints"][PRIMARY_ENDPOINT]
        paired_d = _number(nominated_endpoint["D_R"], "nominated D_R") - _number(
            control_endpoint["D_R"], "control D_R"
        )
        paired_q = _number(nominated_endpoint["Q_R"], "nominated Q_R") - _number(
            control_endpoint["Q_R"], "control Q_R"
        )
        t_win = _is_win(_number(nominated_endpoint["T_S"], "nominated T_S"))
        d_positive = _is_win(_number(nominated_endpoint["D_R"], "nominated D_R"))
        paired_d_win = _is_win(paired_d)
        dependence_win = t_win and d_positive and paired_d_win
        q_positive = _is_win(_number(nominated_endpoint["Q_R"], "nominated Q_R"))
        paired_q_win = _is_win(paired_q)
        joint_win = dependence_win and q_positive and paired_q_win
        gate = nominated["gate_diagnostic"]
        pre0 = _number(gate["baseline_preactivation"], "baseline D preactivation")
        pre_s = _number(gate["sender_preactivation"], "sender D preactivation")
        dose = doses[intent_id]
        output.append(
            {
                "intent_id": intent_id,
                "risk_class": pair["risk_class"],
                "category": pair["category"],
                "dose": {
                    "nominated_activation_delta": _number(
                        dose["nominated_activation_delta"], "nominated activation delta"
                    ),
                    "control_activation_delta": _number(
                        dose["control_activation_delta"], "control activation delta"
                    ),
                    "nominal_nominated_edit_l2": _number(
                        dose["nominal_nominated_edit_l2"], "nominal nominated L2"
                    ),
                    "nominal_control_edit_l2": _number(
                        dose["nominal_control_edit_l2"], "nominal control L2"
                    ),
                    "realized_bfloat16_nominated_edit_l2": _number(
                        dose["realized_bfloat16_nominated_edit_l2"], "realized nominated L2"
                    ),
                    "realized_bfloat16_control_edit_l2": _number(
                        dose["realized_bfloat16_control_edit_l2"], "realized control L2"
                    ),
                    "nominal_equal_l2_relative_error": _number(
                        dose["nominal_equal_l2_relative_error"],
                        "nominal equal-L2 relative error",
                    ),
                    "realized_bfloat16_equal_l2_relative_error": _number(
                        dose["realized_bfloat16_equal_l2_relative_error"],
                        "realized bfloat16 equal-L2 relative error",
                    ),
                    "zero_dose": bool(dose.get("zero_dose")),
                },
                "final_receiver": {
                    arm: {
                        metric: {
                            estimand: _endpoint(final[(intent_id, arm)], metric, estimand)
                            for estimand in ESTIMANDS
                        }
                        for metric in final[(intent_id, arm)]["endpoints"]
                    }
                    for arm in ARMS
                },
                "paired_D": paired_d,
                "paired_Q": paired_q,
                "dependence_components": {
                    "nominated_T_S_positive": t_win,
                    "nominated_D_R_positive": d_positive,
                    "paired_D_positive": paired_d_win,
                    "compound_win": dependence_win,
                },
                "rescue_components": {
                    "nominated_Q_R_positive": q_positive,
                    "paired_Q_positive": paired_q_win,
                    "compound_joint_win": joint_win,
                },
                "nominated_D_gate": {
                    "baseline_preactivation": pre0,
                    "sender_preactivation": pre_s,
                    "crossing_class": gate["crossing_class"],
                    "inactive_to_active": pre0 <= 0 < pre_s,
                    "diagnostic_only": True,
                },
            }
        )
    return output


def _risk_summaries(intent_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for risk in ("harmful", "benign"):
        group = [row for row in intent_rows if row["risk_class"] == risk]
        output[risk] = {
            "intent_count": len(group),
            "paired_D": _describe([row["paired_D"] for row in group]),
            "paired_Q": _describe([row["paired_Q"] for row in group]),
            "nominal_nominated_edit_l2": _describe(
                [row["dose"]["nominal_nominated_edit_l2"] for row in group]
            ),
            "nominated_gate_crossing_classes": dict(
                sorted(Counter(row["nominated_D_gate"]["crossing_class"] for row in group).items())
            ),
            "final_receiver": {
                arm: {
                    metric: {
                        estimand: _describe(
                            [row["final_receiver"][arm][metric][estimand] for row in group]
                        )
                        for estimand in ESTIMANDS
                    }
                    for metric in group[0]["final_receiver"][arm]
                }
                for arm in ARMS
            },
        }
    return output


def _hierarchical_decision(
    plan: Mapping[str, Any], intent_rows: list[Mapping[str, Any]], technically_evaluable: bool
) -> dict[str, Any]:
    harmful = [row for row in intent_rows if row["risk_class"] == "harmful"]
    contract = plan["decision_contract"]
    dependence_wins = sum(row["dependence_components"]["compound_win"] for row in harmful)
    joint_wins = sum(row["rescue_components"]["compound_joint_win"] for row in harmful)
    dependence_threshold = int(contract["dependence_success"]["minimum_wins"])
    rescue_threshold = int(contract["rescue_success"]["minimum_wins"])
    dependence_threshold_met = dependence_wins >= dependence_threshold
    rescue_threshold_met = joint_wins >= rescue_threshold
    rescue_tested = technically_evaluable and dependence_threshold_met
    median_d = statistics.median(
        row["final_receiver"]["nominated_c"][PRIMARY_ENDPOINT]["D_R"]
        for row in harmful
    )
    median_q = statistics.median(
        row["final_receiver"]["nominated_c"][PRIMARY_ENDPOINT]["Q_R"]
        for row in harmful
    )

    if not technically_evaluable:
        status = "not_evaluable"
    elif not dependence_threshold_met:
        status = "dependence_gate_failed"
    elif not rescue_threshold_met:
        status = "dependence_passed_rescue_failed"
    else:
        status = "full_phenotype_replicated"
    return {
        "status": status,
        "decision_type": "compound_preregistered_count_threshold",
        "independent_unit": "intent_id",
        "receiver_dependence": {
            "compound_wins": dependence_wins,
            "minimum_wins": dependence_threshold,
            "denominator": 10,
            "threshold_met_descriptively": dependence_threshold_met,
            "passed": dependence_threshold_met if technically_evaluable else None,
            "not_a_sign_test": True,
        },
        "receiver_rescue": {
            "compound_joint_wins": joint_wins,
            "minimum_wins": rescue_threshold,
            "denominator": 10,
            "threshold_met_descriptively": rescue_threshold_met,
            "tested": rescue_tested,
            "passed": rescue_threshold_met if rescue_tested else None,
            "cannot_rescue_failed_dependence": True,
            "not_a_sign_test": True,
        },
        "raw_nominated_medians": {
            "D_R": median_d,
            "Q_R": median_q,
            "D_R_strictly_positive": median_d > 0.0,
            "Q_R_strictly_positive": median_q > 0.0,
            "role": E8B_SEMANTIC_RESOLUTION[
                "raw_nominated_median_positivity_role"
            ],
        },
    }


def _gate_advancement(
    plan: Mapping[str, Any], intent_rows: list[Mapping[str, Any]], technically_evaluable: bool
) -> dict[str, Any]:
    harmful = [row for row in intent_rows if row["risk_class"] == "harmful"]
    crossings = [row for row in harmful if row["nominated_D_gate"]["inactive_to_active"]]
    joint = sum(row["rescue_components"]["compound_joint_win"] for row in crossings)
    contract = plan["decision_contract"]["gate_crossing_advancement"]
    threshold_met = (
        len(crossings) >= int(contract["minimum_harmful_nominated_crossings"])
        and joint >= int(contract["minimum_joint_route_wins_among_crossings"])
    )
    return {
        "status": (
            "not_evaluable"
            if not technically_evaluable
            else "advance"
            if threshold_met
            else "do_not_advance"
        ),
        "harmful_nominated_crossings": len(crossings),
        "minimum_harmful_nominated_crossings": int(
            contract["minimum_harmful_nominated_crossings"]
        ),
        "joint_route_wins_among_crossings": joint,
        "minimum_joint_route_wins_among_crossings": int(
            contract["minimum_joint_route_wins_among_crossings"]
        ),
        "threshold_met_descriptively": threshold_met,
        "passed": threshold_met if technically_evaluable else None,
        "changes_panel_evaluability": False,
        "crossing_is_post_intervention_diagnostic": True,
    }


def _topic_differential(
    plan: Mapping[str, Any],
    intent_rows: list[Mapping[str, Any]],
    technically_evaluable: bool,
) -> dict[str, Any]:
    by_intent = {row["intent_id"]: row for row in intent_rows}
    contract = plan["decision_contract"]["topic_differential"]
    lower, upper = (float(value) for value in contract["nominated_edit_l2_ratio_interval"])
    pairs = []
    zero_denominators = []
    outside = []
    for pair in plan["decision_contract"]["topic_matched_pairs"]:
        harmful = by_intent[pair["harmful_intent_id"]]
        benign = by_intent[pair["benign_intent_id"]]
        l2_h = harmful["dose"]["nominal_nominated_edit_l2"]
        l2_b = benign["dose"]["nominal_nominated_edit_l2"]
        if l2_b == 0.0:
            ratio = None
            zero_denominators.append(pair["benign_intent_id"])
        else:
            ratio = l2_h / l2_b
            if not lower <= ratio <= upper:
                outside.append(
                    {
                        "harmful_intent_id": pair["harmful_intent_id"],
                        "benign_intent_id": pair["benign_intent_id"],
                        "ratio": ratio,
                    }
                )
        pairs.append(
            {
                **pair,
                "nominal_nominated_edit_l2_harmful": l2_h,
                "nominal_nominated_edit_l2_benign": l2_b,
                "nominated_edit_l2_ratio_harmful_over_benign": ratio,
                "delta_paired_D_harmful_minus_benign": harmful["paired_D"]
                - benign["paired_D"],
                "delta_paired_Q_harmful_minus_benign": harmful["paired_Q"]
                - benign["paired_Q"],
            }
        )
    if zero_denominators:
        guard_status = "not_evaluable_zero_benign_denominator"
    elif outside:
        guard_status = "not_evaluable_outside_interval"
    else:
        guard_status = "evaluable"
    positive_d = sum(
        _is_win(row["delta_paired_D_harmful_minus_benign"]) for row in pairs
    )
    positive_q = sum(
        _is_win(row["delta_paired_Q_harmful_minus_benign"]) for row in pairs
    )
    threshold_met = positive_d >= int(contract["minimum_positive_paired_D"])
    if not technically_evaluable or guard_status != "evaluable":
        status = "not_evaluable"
    elif threshold_met:
        status = "bounded_differential_supported"
    else:
        status = "bounded_differential_not_supported"
    return {
        "status": status,
        "pairs": pairs,
        "dose_guard": {
            "status": guard_status,
            "ratio_orientation": "nominal_nominated_edit_l2_harmful_over_benign",
            "inclusive_interval": [lower, upper],
            "zero_benign_denominator_intents": zero_denominators,
            "outside_interval_pairs": outside,
        },
        "positive_delta_paired_D_count": positive_d,
        "positive_delta_paired_Q_count": positive_q,
        "win_definition": "strictly_greater_than_zero",
        "denominator": len(pairs),
        "median_delta_paired_D": statistics.median(
            row["delta_paired_D_harmful_minus_benign"] for row in pairs
        ),
        "median_delta_paired_Q": statistics.median(
            row["delta_paired_Q_harmful_minus_benign"] for row in pairs
        ),
        "paired_Q_role": "corroborating_only",
        "population_claim_allowed": False,
        "benign_sparing_claim_allowed": False,
    }


def _semantic_threshold_audit(
    plan: Mapping[str, Any], intent_rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Show strict-zero decisions beside the secondary sign-test threshold."""

    secondary = float(E8B_SEMANTIC_RESOLUTION["scalar_sign_test_nonwin_threshold"])
    harmful = [row for row in intent_rows if row["risk_class"] == "harmful"]

    def counts(threshold: float) -> tuple[int, int]:
        dependence = 0
        joint = 0
        for row in harmful:
            endpoint = row["final_receiver"]["nominated_c"][PRIMARY_ENDPOINT]
            dependence_win = all(
                (
                    _is_win(endpoint["T_S"], threshold),
                    _is_win(endpoint["D_R"], threshold),
                    _is_win(row["paired_D"], threshold),
                )
            )
            joint_win = dependence_win and _is_win(
                endpoint["Q_R"], threshold
            ) and _is_win(row["paired_Q"], threshold)
            dependence += dependence_win
            joint += joint_win
        return dependence, joint

    primary_counts = counts(0.0)
    secondary_counts = counts(secondary)
    by_intent = {row["intent_id"]: row for row in intent_rows}

    def topic_counts(threshold: float) -> tuple[int, int]:
        positive_d = 0
        positive_q = 0
        for pair in plan["decision_contract"]["topic_matched_pairs"]:
            harmful_row = by_intent[pair["harmful_intent_id"]]
            benign_row = by_intent[pair["benign_intent_id"]]
            positive_d += _is_win(
                harmful_row["paired_D"] - benign_row["paired_D"], threshold
            )
            positive_q += _is_win(
                harmful_row["paired_Q"] - benign_row["paired_Q"], threshold
            )
        return positive_d, positive_q

    primary_topic_counts = topic_counts(0.0)
    secondary_topic_counts = topic_counts(secondary)
    dependence_minimum = int(
        plan["decision_contract"]["dependence_success"]["minimum_wins"]
    )
    rescue_minimum = int(
        plan["decision_contract"]["rescue_success"]["minimum_wins"]
    )
    topic_minimum = int(
        plan["decision_contract"]["topic_differential"][
            "minimum_positive_paired_D"
        ]
    )
    return {
        "primary_threshold": 0.0,
        "primary_operator": "strictly_greater_than",
        "secondary_sign_test_nonwin_threshold": secondary,
        "primary_compound_counts": {
            "dependence_wins": primary_counts[0],
            "joint_rescue_wins": primary_counts[1],
        },
        "secondary_threshold_compound_counts_reported_only": {
            "dependence_wins": secondary_counts[0],
            "joint_rescue_wins": secondary_counts[1],
        },
        "primary_topic_counts": {
            "positive_paired_D": primary_topic_counts[0],
            "positive_paired_Q": primary_topic_counts[1],
        },
        "secondary_threshold_topic_counts_reported_only": {
            "positive_paired_D": secondary_topic_counts[0],
            "positive_paired_Q": secondary_topic_counts[1],
        },
        "counts_differ": (
            primary_counts != secondary_counts
            or primary_topic_counts != secondary_topic_counts
        ),
        "gate_decision_hinges_on_threshold_difference": (
            (primary_counts[0] >= dependence_minimum)
            != (secondary_counts[0] >= dependence_minimum)
            or (
                primary_counts[0] >= dependence_minimum
                and secondary_counts[0] >= dependence_minimum
                and (primary_counts[1] >= rescue_minimum)
                != (secondary_counts[1] >= rescue_minimum)
            )
        ),
        "topic_decision_hinges_on_threshold_difference": (
            (primary_topic_counts[0] >= topic_minimum)
            != (secondary_topic_counts[0] >= topic_minimum)
        ),
        "secondary_threshold_is_not_used_for_preregistered_decisions": True,
    }


def analyze(
    e8a_plan: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any] | None = None,
    e8a_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a complete E8b result under E8a's frozen scientific contract.

    Production callers pass both ``execution_plan`` and ``e8a_manifest`` so the
    authoritative four-argument E8b result validator runs first.  Omitting both
    is supported for deterministic unit tests and applies the analyzer's strict
    scalar/identity shape checks.  Supplying only one is rejected.
    """

    if (execution_plan is None) != (e8a_manifest is None):
        raise ValueError("E8b analysis requires both execution plan and E8a manifest")
    authoritative_validation = False
    if execution_plan is not None:
        if validate_e8b_result is None:
            raise RuntimeError("s9_e8b_contract.validate_e8b_result is unavailable")
        validate_e8b_result(result, execution_plan, e8a_manifest, e8a_plan)
        authoritative_validation = True

    rows, doses = _validate_minimal_shape(e8a_plan, result)
    primary_threshold = _number(
        E8B_SEMANTIC_RESOLUTION["primary_compound_and_topic_win_threshold"],
        "primary semantic threshold",
    )
    if primary_threshold != 0.0:
        raise ValueError("E8b primary compound/topic threshold must remain zero")
    sign_test_threshold = _number(
        E8B_SEMANTIC_RESOLUTION["scalar_sign_test_nonwin_threshold"],
        "scalar sign-test threshold",
    )
    technical = _technical_evaluability(
        e8a_plan, result, rows, doses, authoritative_validation
    )
    intent_rows = _intent_rows(e8a_plan, rows, doses)
    technically_evaluable = technical["status"] == "evaluable"
    harmful = [row for row in intent_rows if row["risk_class"] == "harmful"]
    scalar_d = _exact_one_sided_sign_test(
        [row["paired_D"] for row in harmful], sign_test_threshold
    )
    scalar_q = _exact_one_sided_sign_test(
        [row["paired_Q"] for row in harmful], sign_test_threshold
    )
    if technically_evaluable:
        paired_tests = {
            "status": "evaluable",
            "interpretation_allowed": True,
            "paired_D": {
                **scalar_d,
                "status": "evaluable",
                "interpretation_allowed": True,
            },
            "paired_Q": {
                **scalar_q,
                "status": "evaluable",
                "interpretation_allowed": True,
            },
            "interpretation": (
                "These exact tests apply only to scalar paired_D and paired_Q. "
                "The compound dependence and rescue thresholds have no sign-test p-value."
            ),
        }
    else:
        invalid_d = {
            **scalar_d,
            "status": "not_evaluable",
            "interpretation_allowed": False,
            "p_value_exact_one_sided": None,
        }
        invalid_q = {
            **scalar_q,
            "status": "not_evaluable",
            "interpretation_allowed": False,
            "p_value_exact_one_sided": None,
        }
        paired_tests = {
            "status": "not_evaluable",
            "interpretation_allowed": False,
            "paired_D": invalid_d,
            "paired_Q": invalid_q,
            "interpretation": (
                "Scalar p-values are suppressed because the technical validity gate failed."
            ),
        }
    payload = {
        "schema_version": 1,
        "analysis_semantics_version": ANALYSIS_SEMANTICS_VERSION,
        "status": ANALYSIS_STATUS,
        "semantic_resolution": dict(E8B_SEMANTIC_RESOLUTION),
        "e8a_plan_payload_sha256": e8a_plan.get("payload_sha256"),
        "result_payload_sha256": result.get("payload_sha256"),
        "technical_evaluability": technical,
        "raw_intent_rows": intent_rows,
        "risk_summaries": _risk_summaries(intent_rows),
        "scalar_paired_sign_tests": paired_tests,
        "semantic_threshold_audit": _semantic_threshold_audit(
            e8a_plan, intent_rows
        ),
        "hierarchical_decision": _hierarchical_decision(
            e8a_plan, intent_rows, technically_evaluable
        ),
        "gate_crossing_advancement": _gate_advancement(
            e8a_plan, intent_rows, technically_evaluable
        ),
        "topic_matched_differential": _topic_differential(
            e8a_plan, intent_rows, technically_evaluable
        ),
        "claim_boundary": {
            "independent_unit": "intent_id",
            "independent_intent_count": 15,
            "harmful_primary_intent_count": 10,
            "benign_diagnostic_intent_count": 5,
            "route_outcome_unseen_relative_to_e7c": True,
            "intent_disjoint_from_e7c": True,
            "pristine_or_model_unseen": False,
            "template_disjoint": False,
            "receiver_estimand": "complete_layer_32_mlp_output_at_final_prompt_position",
            "nominated_D_feature_is_diagnostic_only": True,
            "equal_l2_comparator_establishes_route_specificity": False,
            "unique_sparse_feature_mediation_claim_allowed": False,
            "dominant_or_necessary_pathway_claim_allowed": False,
            "topology_validation_claim_allowed": False,
            "benign_sparing_claim_allowed": False,
            "generation_or_behavior_claim_allowed": False,
            "population_generalization_claim_allowed": False,
            "confirmation_partition_used": False,
            "e7d_executed": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    execution_plan = json.loads(args.execution_plan.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    payload = analyze(
        plan,
        result,
        execution_plan=execution_plan,
        e8a_manifest=manifest,
    )
    payload["input_file_sha256"] = {
        "manifest": sha256_file(args.manifest),
        "plan": sha256_file(args.plan),
        "execution_plan": sha256_file(args.execution_plan),
        "result": sha256_file(args.result),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "technical_evaluability": payload["technical_evaluability"]["status"],
                "replication_decision": payload["hierarchical_decision"]["status"],
                "payload_sha256": payload["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
