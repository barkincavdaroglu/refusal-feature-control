#!/usr/bin/env python3
"""Outcome-frozen severity analysis for the exact E9 route gates.

The E9 gate evaluator below mirrors the frozen S9 implementation.  Simulation
then varies only a prespecified benign-to-harmful paired-effect ratio while
retaining whole-pair dependence.  Two noise models are reported separately:

* a multivariate Gaussian model with the observed whole-pair covariance; and
* a whole-pair residual bootstrap, which preserves empirical tail/co-movement
  patterns without independently resampling route components.

Observed variance is reused to audit the completed design, so every result is
also evaluated at explicitly frozen variance multipliers.  This module scopes
future gate calibration; it does not reopen any recorded E9 scientific result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np


FEATURES = (
    "harmful_T_S_C",
    "harmful_D_R_C",
    "harmful_paired_D",
    "harmful_Q_R_C",
    "harmful_paired_Q",
    "benign_paired_D",
    "benign_paired_Q",
)
PROBABILITY_FIELDS = (
    "dependence_separate",
    "rescue_separate",
    "risk_differential_separate",
    "benign_smallness_separate",
    "dependence_hierarchical",
    "rescue_hierarchical",
    "risk_differential_hierarchical",
    "benign_smallness_hierarchical",
    "conjunction",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def exact_one_sided_sign_test_tail(wins: int, denominator: int = 12) -> float:
    if not isinstance(wins, int) or wins < 0 or wins > denominator:
        raise ValueError("E9 sign-test count is malformed")
    return sum(comb(denominator, value) for value in range(wins, denominator + 1)) / (
        2**denominator
    )


def _endpoint(cell: Mapping[str, Any], metric: str) -> dict[str, float]:
    endpoints = cell.get("endpoints")
    if not isinstance(endpoints, Mapping) or not isinstance(
        endpoints.get(metric), Mapping
    ):
        raise ValueError("E9 route endpoint record is missing")
    record = endpoints[metric]
    values = {
        name: _finite(record.get(name), f"{cell.get('cell_id')}:{metric}:{name}")
        for name in ("T_S", "D_R", "B_R", "Q_R", "invariant_error")
    }
    if abs(values["T_S"] - values["D_R"] - values["B_R"]) > 1e-7:
        raise ValueError("E9 signed route accounting identity failed")
    if abs(values["invariant_error"]) > 1e-7:
        raise ValueError("E9 recorded route invariant error differs")
    return values


def e9_route_analysis(
    route_cells: list[dict[str, Any]], *, metric: str
) -> dict[str, Any]:
    """Evaluate the frozen 12-pair hierarchy exactly as S9 E9c/E9d did."""

    if metric not in {"next_token_refusal_log_odds", "probe_score"}:
        raise ValueError(f"Unknown E9 endpoint: {metric}")
    final_cells = [
        cell
        for cell in route_cells
        if cell.get("receiver_token_position") == "final_prompt_position"
        and cell.get("included_in_empirical_analysis") is True
    ]
    if len(final_cells) != 48:
        raise ValueError("E9 empirical route-cell count differs")
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
            raise ValueError("E9 route-cell identity is malformed")
        key = (pair_id, risk, arm)
        if key in by_key:
            raise ValueError("E9 empirical route cell is duplicated")
        by_key[key] = cell
    pair_ids = sorted({key[0] for key in by_key})
    if pair_ids != [f"E9P{index:02d}" for index in range(1, 13)]:
        raise ValueError("E9 topic-pair closure differs")

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
                raise ValueError("E9 matched arm closure differs")
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
    if (
        len(harmful_intents) != 12
        or len(benign_intents) != 12
        or harmful_intents.intersection(benign_intents)
    ):
        raise ValueError("E9 intent-level independent-unit closure differs")

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


@dataclass(frozen=True)
class GateConfig:
    name: str
    dependence_minimum_wins: int = 10
    rescue_minimum_wins: int = 10
    differential_minimum_each: int = 10
    smallness_minimum_wins: int = 10
    smallness_maximum_ratio: float = 0.25

    def validate(self, n_pairs: int) -> None:
        for label, value in (
            ("dependence", self.dependence_minimum_wins),
            ("rescue", self.rescue_minimum_wins),
            ("differential", self.differential_minimum_each),
            ("smallness", self.smallness_minimum_wins),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= n_pairs:
                raise ValueError(f"{label} minimum wins is malformed")
        if not 0.0 <= self.smallness_maximum_ratio <= 1.0:
            raise ValueError("Smallness maximum ratio must lie in [0, 1]")


FROZEN_E9_GATE = GateConfig(name="frozen_e9")


def observed_pair_matrix(e9_analysis: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    primary = e9_analysis.get("primary")
    pair_rows = primary.get("pair_rows") if isinstance(primary, Mapping) else None
    if not isinstance(pair_rows, list) or len(pair_rows) != 12:
        raise ValueError("W3 requires the twelve primary E9 pair rows")
    expected_ids = [f"E9P{index:02d}" for index in range(1, 13)]
    observed_ids = [row.get("pair_id") for row in pair_rows]
    if observed_ids != expected_ids:
        raise ValueError("E9 primary pair order/closure differs")
    matrix = []
    for row in pair_rows:
        harmful = row.get("harmful")
        benign = row.get("benign")
        if not isinstance(harmful, Mapping) or not isinstance(benign, Mapping):
            raise ValueError("E9 pair row lacks harmful/benign records")
        matrix.append(
            [
                _finite(harmful.get("T_S_C"), "harmful T_S_C"),
                _finite(harmful.get("D_R_C"), "harmful D_R_C"),
                _finite(harmful.get("paired_D"), "harmful paired_D"),
                _finite(harmful.get("Q_R_C"), "harmful Q_R_C"),
                _finite(harmful.get("paired_Q"), "harmful paired_Q"),
                _finite(benign.get("paired_D"), "benign paired_D"),
                _finite(benign.get("paired_Q"), "benign paired_Q"),
            ]
        )
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (12, len(FEATURES)) or not np.isfinite(result).all():
        raise ValueError("E9 observed pair matrix is malformed")
    return result, expected_ids


def _stable_seed(root_seed: int, *parts: Any) -> int:
    material = "|".join([str(root_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _target_mean(observed: np.ndarray, ratio: float) -> np.ndarray:
    if not 0.0 < ratio < 1.0:
        raise ValueError("True benign/harmful ratio must lie strictly in (0, 1)")
    mean = observed.mean(axis=0)
    target = mean.copy()
    target[5] = ratio * mean[2]
    target[6] = ratio * mean[4]
    if abs(mean[2]) <= 1e-12 or abs(mean[4]) <= 1e-12:
        raise ValueError("Observed harmful paired-effect mean is too close to zero")
    return target


def _regularized_covariance(observed: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0.0:
        raise ValueError("Covariance ridge must be nonnegative")
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as error:
        raise RuntimeError(
            "W3 frozen parametric model requires sklearn LedoitWolf"
        ) from error
    covariance = np.asarray(
        LedoitWolf(assume_centered=False).fit(observed).covariance_, dtype=np.float64
    )
    scale = max(float(np.trace(covariance) / covariance.shape[0]), 1e-12)
    covariance = covariance + np.eye(covariance.shape[0]) * ridge * scale
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(scale * 1e-12, 1e-15)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval count is malformed")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _gate_masks(samples: np.ndarray, gate: GateConfig) -> dict[str, np.ndarray]:
    if samples.ndim != 3 or samples.shape[2] != len(FEATURES):
        raise ValueError("Severity samples must have shape [draw, pair, feature]")
    n_pairs = samples.shape[1]
    gate.validate(n_pairs)
    h_t, h_d, h_pd, h_q, h_pq, b_pd, b_pq = [
        samples[:, :, index] for index in range(len(FEATURES))
    ]
    dependence_pair = (h_t > 0.0) & (h_d > 0.0) & (h_pd > 0.0)
    rescue_pair = dependence_pair & (h_q > 0.0) & (h_pq > 0.0)
    differential_d_pair = h_pd - b_pd > 0.0
    differential_q_pair = h_pq - b_pq > 0.0
    ratio_d = np.divide(
        np.abs(b_pd),
        np.abs(h_pd),
        out=np.full_like(h_pd, np.inf),
        where=h_pd > 0.0,
    )
    ratio_q = np.divide(
        np.abs(b_pq),
        np.abs(h_pq),
        out=np.full_like(h_pq, np.inf),
        where=h_pq > 0.0,
    )
    smallness_pair = (
        (h_pd > 0.0)
        & (h_pq > 0.0)
        & (ratio_d <= gate.smallness_maximum_ratio)
        & (ratio_q <= gate.smallness_maximum_ratio)
    )
    dependence_count = dependence_pair.sum(axis=1)
    rescue_count = rescue_pair.sum(axis=1)
    differential_d_count = differential_d_pair.sum(axis=1)
    differential_q_count = differential_q_pair.sum(axis=1)
    smallness_count = smallness_pair.sum(axis=1)
    dep = dependence_count >= gate.dependence_minimum_wins
    rescue_raw = rescue_count >= gate.rescue_minimum_wins
    differential_raw = (
        (differential_d_count >= gate.differential_minimum_each)
        & (differential_q_count >= gate.differential_minimum_each)
    )
    smallness_raw = smallness_count >= gate.smallness_minimum_wins
    rescue = dep & rescue_raw
    differential = rescue & differential_raw
    smallness = rescue & smallness_raw
    return {
        "dependence_separate": dep,
        "rescue_separate": rescue_raw,
        "risk_differential_separate": differential_raw,
        "benign_smallness_separate": smallness_raw,
        "dependence_hierarchical": dep,
        "rescue_hierarchical": rescue,
        "risk_differential_hierarchical": differential,
        "benign_smallness_hierarchical": smallness,
        "conjunction": differential & smallness,
        "dependence_count": dependence_count,
        "rescue_count": rescue_count,
        "differential_D_count": differential_d_count,
        "differential_Q_count": differential_q_count,
        "smallness_count": smallness_count,
    }


def _summarize_masks(
    accumulators: Mapping[str, int],
    count_sums: Mapping[str, float],
    *,
    draws: int,
) -> dict[str, Any]:
    probabilities = {}
    for name in PROBABILITY_FIELDS:
        probability = accumulators[name] / draws
        probabilities[name] = {
            "pass_probability": probability,
            "monte_carlo_standard_error": math.sqrt(
                probability * (1.0 - probability) / draws
            ),
            "wilson_95_percent_interval": wilson_interval(
                accumulators[name], draws
            ),
        }
    return {
        "probabilities": probabilities,
        "mean_win_counts": {
            name: count_sums[name] / draws
            for name in (
                "dependence_count",
                "rescue_count",
                "differential_D_count",
                "differential_Q_count",
                "smallness_count",
            )
        },
    }


def _simulate_condition(
    observed: np.ndarray,
    *,
    ratio: float,
    method: str,
    variance_multiplier: float,
    draws: int,
    n_pairs: int,
    batch_size: int,
    covariance_ridge: float,
    seed: int,
    gates: Sequence[GateConfig],
) -> dict[str, Any]:
    if method not in {"parametric", "whole_pair_residual_bootstrap"}:
        raise ValueError(f"Unsupported W3 method: {method}")
    if variance_multiplier <= 0.0:
        raise ValueError("Variance multiplier must be positive")
    if draws <= 0 or batch_size <= 0 or n_pairs <= 0:
        raise ValueError("Simulation sizes must be positive")
    for gate in gates:
        gate.validate(n_pairs)
    target = _target_mean(observed, ratio)
    observed_mean = observed.mean(axis=0)
    residuals = observed - observed_mean
    covariance = _regularized_covariance(observed, covariance_ridge)
    rng = np.random.default_rng(seed)
    accumulators = {
        gate.name: {name: 0 for name in PROBABILITY_FIELDS} for gate in gates
    }
    count_sums = {
        gate.name: {
            name: 0.0
            for name in (
                "dependence_count",
                "rescue_count",
                "differential_D_count",
                "differential_Q_count",
                "smallness_count",
            )
        }
        for gate in gates
    }
    completed = 0
    while completed < draws:
        current = min(batch_size, draws - completed)
        if method == "parametric":
            samples = rng.multivariate_normal(
                mean=target,
                cov=covariance * variance_multiplier,
                size=(current, n_pairs),
                method="eigh",
            )
        else:
            indices = rng.integers(0, residuals.shape[0], size=(current, n_pairs))
            samples = target + math.sqrt(variance_multiplier) * residuals[indices]
        for gate in gates:
            masks = _gate_masks(samples, gate)
            for name in PROBABILITY_FIELDS:
                accumulators[gate.name][name] += int(masks[name].sum())
            for name in count_sums[gate.name]:
                count_sums[gate.name][name] += float(masks[name].sum())
        completed += current
    return {
        "true_benign_to_harmful_ratio": ratio,
        "ratio_definition": (
            "target mean benign paired_D/Q equals ratio times the corresponding "
            "observed harmful paired_D/Q mean; harmful means otherwise remain fixed"
        ),
        "method": method,
        "variance_multiplier": variance_multiplier,
        "variance_multiplier_definition": "residual variance multiplier (SD scales by square root)",
        "draws": draws,
        "pairs_per_draw": n_pairs,
        "seed": seed,
        "gates": {
            gate.name: {
                "gate_config": asdict(gate),
                **_summarize_masks(
                    accumulators[gate.name], count_sums[gate.name], draws=draws
                ),
            }
            for gate in gates
        },
    }


def run_severity_grid(
    observed: np.ndarray,
    *,
    ratios: Sequence[float],
    methods: Sequence[str],
    variance_multipliers: Sequence[float],
    draws: int,
    n_pairs: int,
    batch_size: int,
    covariance_ridge: float,
    root_seed: int,
    gates: Sequence[GateConfig],
    method_root_seeds: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    if observed.shape != (12, len(FEATURES)):
        raise ValueError("W3 observed matrix must contain exactly twelve E9 pairs")
    if n_pairs <= 0:
        raise ValueError("W3 n_pairs must be positive")
    if len({gate.name for gate in gates}) != len(gates):
        raise ValueError("Gate configuration names must be unique")
    records = []
    for ratio in ratios:
        ratio = float(ratio)
        for method in methods:
            for multiplier in variance_multipliers:
                multiplier = float(multiplier)
                method_seed = (
                    int(method_root_seeds[method])
                    if method_root_seeds is not None and method in method_root_seeds
                    else root_seed
                )
                seed = _stable_seed(
                    method_seed, method, f"{ratio:.12g}", f"{multiplier:.12g}", n_pairs
                )
                records.append(
                    _simulate_condition(
                        observed,
                        ratio=ratio,
                        method=method,
                        variance_multiplier=multiplier,
                        draws=draws,
                        n_pairs=n_pairs,
                        batch_size=batch_size,
                        covariance_ridge=covariance_ridge,
                        seed=seed,
                        gates=gates,
                    )
                )
    return records


def detectable_regions(
    records: Sequence[Mapping[str, Any]], *, power_threshold: float
) -> dict[str, Any]:
    if not 0.0 < power_threshold < 1.0:
        raise ValueError("Power threshold must lie strictly in (0, 1)")
    keys = sorted(
        {
            (str(row["method"]), float(row["variance_multiplier"]), gate_name)
            for row in records
            for gate_name in row["gates"]
        }
    )
    output = {}
    for method, multiplier, gate_name in keys:
        rows = sorted(
            (
                row
                for row in records
                if row["method"] == method
                and float(row["variance_multiplier"]) == multiplier
                and gate_name in row["gates"]
            ),
            key=lambda row: row["true_benign_to_harmful_ratio"],
        )
        passing = [
            float(row["true_benign_to_harmful_ratio"])
            for row in rows
            if row["gates"][gate_name]["probabilities"]["conjunction"][
                "pass_probability"
            ]
            >= power_threshold
        ]
        output[f"{method}|variance={multiplier:g}|gate={gate_name}"] = {
            "ratios_at_or_above_power_threshold": passing,
            "maximum_detectable_ratio_on_frozen_grid": max(passing, default=None),
        }
    return {"power_threshold": power_threshold, "regions": output}


def calibrate_future_gate(
    records: Sequence[Mapping[str, Any]],
    gates: Sequence[GateConfig],
    *,
    target_ratio: float,
    minimum_power: float,
    reference_variance_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Choose only among caller-frozen candidates; never tune a new gate here."""

    by_name = {gate.name: gate for gate in gates if gate.name != FROZEN_E9_GATE.name}
    candidates = []
    for name, gate in sorted(by_name.items()):
        powers = [
            row["gates"][name]["probabilities"]["conjunction"]["pass_probability"]
            for row in records
            if math.isclose(
                float(row["true_benign_to_harmful_ratio"]), target_ratio, abs_tol=1e-12
            )
            and math.isclose(
                float(row["variance_multiplier"]),
                reference_variance_multiplier,
                abs_tol=1e-12,
            )
            and name in row["gates"]
        ]
        if not powers:
            continue
        worst_method_power = min(float(value) for value in powers)
        candidates.append(
            {
                "name": name,
                "gate_config": asdict(gate),
                "minimum_power_across_frozen_methods": worst_method_power,
                "meets_target": worst_method_power >= minimum_power,
            }
        )
    eligible = [row for row in candidates if row["meets_target"]]
    # More count wins and a smaller ratio are stricter.  The ordering is frozen
    # and deterministic; it does not consult observed scientific outcomes.
    eligible.sort(
        key=lambda row: (
            -row["gate_config"]["dependence_minimum_wins"],
            -row["gate_config"]["rescue_minimum_wins"],
            -row["gate_config"]["differential_minimum_each"],
            -row["gate_config"]["smallness_minimum_wins"],
            row["gate_config"]["smallness_maximum_ratio"],
            row["name"],
        )
    )
    return {
        "status": "candidate_selected" if eligible else "no_frozen_candidate_meets_power",
        "target_ratio": target_ratio,
        "minimum_power": minimum_power,
        "reference_variance_multiplier": reference_variance_multiplier,
        "selection_rule": "strictest frozen candidate meeting the worst-method power target",
        "selected": eligible[0] if eligible else None,
        "candidates": candidates,
    }


def _minimum_wins(n_pairs: int, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Future gate win fraction must lie in (0, 1]")
    return int(math.ceil(fraction * n_pairs - 1e-10))


def _future_candidates(
    *,
    sample_size_grid: Sequence[int],
    ratio_threshold_grid: Sequence[float],
    minimum_win_fraction_grid: Sequence[float],
) -> list[dict[str, Any]]:
    candidates = []
    for n_pairs in sample_size_grid:
        n_pairs = int(n_pairs)
        for ratio_threshold in ratio_threshold_grid:
            ratio_threshold = float(ratio_threshold)
            for fraction in minimum_win_fraction_grid:
                fraction = float(fraction)
                wins = _minimum_wins(n_pairs, fraction)
                candidates.append(
                    {
                        "candidate_id": (
                            f"n{n_pairs}_ratio{ratio_threshold:g}_wins{wins}"
                        ),
                        "n_pairs": n_pairs,
                        "smallness_maximum_ratio": ratio_threshold,
                        "minimum_win_fraction": fraction,
                        "minimum_wins": wins,
                    }
                )
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        # Fractions may round to the same integer at small n. Preserve one
        # deterministic candidate per executable gate.
        unique = {}
        for row in candidates:
            key = (
                row["n_pairs"],
                row["smallness_maximum_ratio"],
                row["minimum_wins"],
            )
            unique.setdefault(key, row)
        candidates = list(unique.values())
    return candidates


def _future_condition_probabilities(
    observed: np.ndarray,
    *,
    true_ratio: float,
    method: str,
    variance_multiplier: float,
    draws: int,
    n_pairs: int,
    batch_size: int,
    covariance_ridge: float,
    seed: int,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every frozen candidate on one shared Monte Carlo sample stream."""

    target = _target_mean(observed, true_ratio)
    residuals = observed - observed.mean(axis=0)
    covariance = _regularized_covariance(observed, covariance_ridge)
    rng = np.random.default_rng(seed)
    thresholds = sorted(
        {float(row["smallness_maximum_ratio"]) for row in candidates}
    )
    successes = {str(row["candidate_id"]): 0 for row in candidates}
    completed = 0
    while completed < draws:
        current = min(batch_size, draws - completed)
        if method == "parametric":
            samples = rng.multivariate_normal(
                mean=target,
                cov=covariance * variance_multiplier,
                size=(current, n_pairs),
                method="eigh",
            )
        elif method == "whole_pair_residual_bootstrap":
            indices = rng.integers(0, residuals.shape[0], size=(current, n_pairs))
            samples = target + math.sqrt(variance_multiplier) * residuals[indices]
        else:
            raise ValueError(f"Unsupported W3 method: {method}")
        h_t, h_d, h_pd, h_q, h_pq, b_pd, b_pq = [
            samples[:, :, index] for index in range(len(FEATURES))
        ]
        dependence_count = ((h_t > 0) & (h_d > 0) & (h_pd > 0)).sum(axis=1)
        rescue_count = (
            (h_t > 0) & (h_d > 0) & (h_pd > 0) & (h_q > 0) & (h_pq > 0)
        ).sum(axis=1)
        differential_d_count = (h_pd - b_pd > 0).sum(axis=1)
        differential_q_count = (h_pq - b_pq > 0).sum(axis=1)
        ratio_d = np.divide(
            np.abs(b_pd),
            np.abs(h_pd),
            out=np.full_like(h_pd, np.inf),
            where=h_pd > 0,
        )
        ratio_q = np.divide(
            np.abs(b_pq),
            np.abs(h_pq),
            out=np.full_like(h_pq, np.inf),
            where=h_pq > 0,
        )
        smallness_counts = {
            threshold: (
                (h_pd > 0)
                & (h_pq > 0)
                & (ratio_d <= threshold)
                & (ratio_q <= threshold)
            ).sum(axis=1)
            for threshold in thresholds
        }
        for candidate in candidates:
            wins = int(candidate["minimum_wins"])
            passed = (
                (dependence_count >= wins)
                & (rescue_count >= wins)
                & (differential_d_count >= wins)
                & (differential_q_count >= wins)
                & (
                    smallness_counts[float(candidate["smallness_maximum_ratio"])]
                    >= wins
                )
            )
            successes[str(candidate["candidate_id"])] += int(passed.sum())
        completed += current
    output = []
    for candidate in candidates:
        count = successes[str(candidate["candidate_id"])]
        output.append(
            {
                **dict(candidate),
                "true_ratio": true_ratio,
                "method": method,
                "variance_multiplier": variance_multiplier,
                "draws": draws,
                "seed": seed,
                "passes": count,
                "pass_probability": count / draws,
                "wilson_95_percent_interval": wilson_interval(count, draws),
            }
        )
    return output


def run_future_calibration_grid(
    observed: np.ndarray,
    *,
    nonselective_null_ratio: float,
    target_selective_ratio: float,
    maximum_type_i_error: float,
    minimum_power: float,
    sample_size_grid: Sequence[int],
    ratio_threshold_grid: Sequence[float],
    minimum_win_fraction_grid: Sequence[float],
    methods: Sequence[str],
    variance_multipliers: Sequence[float],
    draws: int,
    batch_size: int,
    covariance_ridge: float,
    method_root_seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Frozen future-gate grid with robust null/type-I and target/power checks."""

    candidates = _future_candidates(
        sample_size_grid=sample_size_grid,
        ratio_threshold_grid=ratio_threshold_grid,
        minimum_win_fraction_grid=minimum_win_fraction_grid,
    )
    condition_rows = []
    for n_pairs in sorted({int(row["n_pairs"]) for row in candidates}):
        n_candidates = [row for row in candidates if int(row["n_pairs"]) == n_pairs]
        for true_ratio in (nonselective_null_ratio, target_selective_ratio):
            for method in methods:
                if method not in method_root_seeds:
                    raise ValueError(f"W3 future calibration lacks seed for {method}")
                for multiplier in variance_multipliers:
                    seed = _stable_seed(
                        int(method_root_seeds[method]),
                        "future",
                        n_pairs,
                        f"{true_ratio:.12g}",
                        f"{float(multiplier):.12g}",
                    )
                    condition_rows.extend(
                        _future_condition_probabilities(
                            observed,
                            true_ratio=float(true_ratio),
                            method=method,
                            variance_multiplier=float(multiplier),
                            draws=draws,
                            n_pairs=n_pairs,
                            batch_size=batch_size,
                            covariance_ridge=covariance_ridge,
                            seed=seed,
                            candidates=n_candidates,
                        )
                    )
    evaluations = []
    for candidate in candidates:
        rows = [
            row
            for row in condition_rows
            if row["candidate_id"] == candidate["candidate_id"]
        ]
        null_rows = [
            row
            for row in rows
            if math.isclose(row["true_ratio"], nonselective_null_ratio, abs_tol=1e-12)
        ]
        target_rows = [
            row
            for row in rows
            if math.isclose(row["true_ratio"], target_selective_ratio, abs_tol=1e-12)
        ]
        expected_conditions = len(methods) * len(variance_multipliers)
        if len(null_rows) != expected_conditions or len(target_rows) != expected_conditions:
            raise ValueError("W3 future calibration condition closure differs")
        worst_type_i = max(float(row["pass_probability"]) for row in null_rows)
        worst_power = min(float(row["pass_probability"]) for row in target_rows)
        evaluations.append(
            {
                **dict(candidate),
                "maximum_type_i_error_across_models_and_variances": worst_type_i,
                "minimum_power_across_models_and_variances": worst_power,
                "maximum_type_i_wilson_upper": max(
                    float(row["wilson_95_percent_interval"][1]) for row in null_rows
                ),
                "minimum_power_wilson_lower": min(
                    float(row["wilson_95_percent_interval"][0]) for row in target_rows
                ),
                "meets_type_i_error": worst_type_i <= maximum_type_i_error,
                "meets_power": worst_power >= minimum_power,
                "robustly_eligible": worst_type_i <= maximum_type_i_error
                and worst_power >= minimum_power,
            }
        )
    eligible = [row for row in evaluations if row["robustly_eligible"]]
    eligible.sort(
        key=lambda row: (
            row["n_pairs"],
            row["maximum_type_i_error_across_models_and_variances"],
            -row["minimum_power_across_models_and_variances"],
            row["smallness_maximum_ratio"],
            -row["minimum_wins"],
        )
    )
    return {
        "status": "candidate_selected" if eligible else "no_candidate_meets_robust_gate",
        "null_ratio": nonselective_null_ratio,
        "target_ratio": target_selective_ratio,
        "maximum_type_i_error": maximum_type_i_error,
        "minimum_power": minimum_power,
        "robust_requirement": "both_models_and_all_variance_multipliers",
        "selection_order": "smallest_n_then_smallest_type_i_error_then_highest_power_then_stricter_ratio_threshold",
        "candidate_count": len(evaluations),
        "selected": eligible[0] if eligible else None,
        "candidates": evaluations,
        "condition_probabilities": condition_rows,
    }


def binomial_threshold_curve(
    *, n: int, minimum_wins: int, success_probabilities: Sequence[float]
) -> list[dict[str, float]]:
    if not 0 <= minimum_wins <= n:
        raise ValueError("Binomial threshold is malformed")
    output = []
    for probability in success_probabilities:
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Bernoulli success probability must lie in [0, 1]")
        pass_probability = sum(
            comb(n, wins)
            * probability**wins
            * (1.0 - probability) ** (n - wins)
            for wins in range(minimum_wins, n + 1)
        )
        output.append(
            {
                "elementary_success_probability": probability,
                "gate_pass_probability": pass_probability,
            }
        )
    return output


def retro_design_worked_examples(
    e4c_analysis: Mapping[str, Any],
    e8b_analysis: Mapping[str, Any],
    *,
    success_probabilities: Sequence[float],
) -> dict[str, Any]:
    """Bounded count-threshold illustrations for E4c and E8b design sensitivity."""

    e4_h09 = e4c_analysis.get("h09_decoder_replication")
    e4_h03 = e4c_analysis.get("h03_coordinate_reversal_replication")
    e4_benign = e4c_analysis.get("benign_audit")
    if (
        not isinstance(e4_h09, Mapping)
        or not isinstance(e4_h03, Mapping)
        or not isinstance(e4_benign, Mapping)
    ):
        raise ValueError("E4c analysis lacks frozen replication rules")
    h09_rows = e4_h09.get("rows")
    minimum = e4_h09.get("working_rule_required_successes")
    if not isinstance(h09_rows, list) or not isinstance(minimum, int):
        raise ValueError("E4c H09 working rule is malformed")
    h03_arms = []
    for arm in e4_h03.get("arms", []):
        if not isinstance(arm, Mapping) or not isinstance(arm.get("rows"), list):
            raise ValueError("E4c H03 arm is malformed")
        h03_arms.append(
            {
                "arm_id": arm.get("arm_id"),
                "observed_target_only_successes": arm.get("target_only_successes"),
                "denominator": len(arm["rows"]),
                "minimum_wins": minimum,
                "illustrative_curve": binomial_threshold_curve(
                    n=len(arm["rows"]),
                    minimum_wins=minimum,
                    success_probabilities=success_probabilities,
                ),
            }
        )

    e8_decision = e8b_analysis.get("hierarchical_decision")
    if not isinstance(e8_decision, Mapping):
        raise ValueError("E8b analysis lacks its hierarchical decision")
    dependence = e8_decision.get("receiver_dependence")
    rescue = e8_decision.get("receiver_rescue")
    topic = e8b_analysis.get("topic_matched_differential")
    if not isinstance(dependence, Mapping) or not isinstance(rescue, Mapping):
        raise ValueError("E8b count gates are malformed")
    if not isinstance(topic, Mapping):
        raise ValueError("E8b analysis lacks the five topic differentials")
    false_refusals = e4_benign.get("false_refusal_comparisons", [])
    if not isinstance(false_refusals, list):
        raise ValueError("E4c benign warning rows are malformed")
    warning_contexts = {
        str(row.get("context_id")) for row in false_refusals if isinstance(row, Mapping)
    }

    return {
        "interpretation": (
            "Illustrative Bernoulli count-threshold sensitivity only. Compound route "
            "wins are not assumed to decompose into independent mechanistic events."
        ),
        "e4c": {
            "h09_decoder": {
                "observed_target_only_successes": e4_h09.get("target_only_successes"),
                "denominator": len(h09_rows),
                "minimum_wins": minimum,
                "illustrative_curve": binomial_threshold_curve(
                    n=len(h09_rows),
                    minimum_wins=minimum,
                    success_probabilities=success_probabilities,
                ),
            },
            "h03_coordinate_arms": h03_arms,
            "benign_warning": {
                "independent_contexts": int(e4_benign.get("contexts", 2)),
                "observed_contexts_with_any_warning": len(warning_contexts),
                "warning_rule": "at_least_one_false_refusal_context",
                "illustrative_curve": binomial_threshold_curve(
                    n=int(e4_benign.get("contexts", 2)),
                    minimum_wins=1,
                    success_probabilities=success_probabilities,
                ),
                "matched_comparison_rows_are_not_independent_units": True,
            },
        },
        "e8b": {
            "dependence": {
                "observed_compound_wins": dependence.get("compound_wins"),
                "denominator": int(dependence["denominator"]),
                "minimum_wins": int(dependence["minimum_wins"]),
                "illustrative_curve": binomial_threshold_curve(
                    n=int(dependence["denominator"]),
                    minimum_wins=int(dependence["minimum_wins"]),
                    success_probabilities=success_probabilities,
                ),
            },
            "joint_rescue": {
                "observed_compound_wins": rescue.get("compound_joint_wins"),
                "denominator": int(rescue["denominator"]),
                "minimum_wins": int(rescue["minimum_wins"]),
                "illustrative_curve": binomial_threshold_curve(
                    n=int(rescue["denominator"]),
                    minimum_wins=int(rescue["minimum_wins"]),
                    success_probabilities=success_probabilities,
                ),
            },
            "topic_differential": {
                "observed_positive_paired_D": topic.get(
                    "positive_delta_paired_D_count"
                ),
                "observed_positive_paired_Q": topic.get(
                    "positive_delta_paired_Q_count"
                ),
                "denominator": int(topic.get("denominator", 5)),
                "minimum_wins": 5,
                "illustrative_curve": binomial_threshold_curve(
                    n=int(topic.get("denominator", 5)),
                    minimum_wins=5,
                    success_probabilities=success_probabilities,
                ),
                "same_dose_selectivity_status": topic.get("status"),
                "same_dose_selectivity_evaluable": topic.get("status")
                != "not_evaluable",
                "guardrail": (
                    "The five topic differentials are reported at their true independent-unit "
                    "count, but the historical same-dose selectivity result remains not "
                    "evaluable when its frozen dose guard failed."
                ),
            },
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _gate_from_mapping(value: Mapping[str, Any]) -> GateConfig:
    return GateConfig(
        name=str(value["name"]),
        dependence_minimum_wins=int(value["dependence_minimum_wins"]),
        rescue_minimum_wins=int(value["rescue_minimum_wins"]),
        differential_minimum_each=int(value["differential_minimum_each"]),
        smallness_minimum_wins=int(value["smallness_minimum_wins"]),
        smallness_maximum_ratio=float(value["smallness_maximum_ratio"]),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty W3 CSV")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _historical_csv_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        gate = row["gates"]["frozen_e9"]
        flattened = {
            "true_ratio": row["true_benign_to_harmful_ratio"],
            "method": row["method"],
            "variance_multiplier": row["variance_multiplier"],
            "draws": row["draws"],
            "seed": row["seed"],
        }
        for name, probability in gate["probabilities"].items():
            flattened[f"{name}_probability"] = probability["pass_probability"]
            flattened[f"{name}_wilson_lower"] = probability[
                "wilson_95_percent_interval"
            ][0]
            flattened[f"{name}_wilson_upper"] = probability[
                "wilson_95_percent_interval"
            ][1]
        output.append(flattened)
    return output


def render_report(payload: Mapping[str, Any]) -> str:
    future = payload["future_gate_calibration"]
    lines = [
        "# S15 W3 — gate-severity audit",
        "",
        "## Basic version",
        "",
        "The historical twelve-pair gates were replayed under signed benign-to-harmful "
        "ratios from 0.10 to 0.90. Results are shown separately for a Ledoit-Wolf "
        "Gaussian model and a whole-pair residual bootstrap at every frozen variance "
        "multiplier.",
        "",
        f"Future calibration status: `{future['status']}`.",
        "",
    ]
    if future.get("selected") is not None:
        selected = future["selected"]
        lines.extend(
            [
                f"Selected future gate: n=`{selected['n_pairs']}`, ratio threshold "
                f"`{selected['smallness_maximum_ratio']}`, minimum wins "
                f"`{selected['minimum_wins']}`. Worst type-I error was "
                f"`{selected['maximum_type_i_error_across_models_and_variances']}` and "
                f"worst power was `{selected['minimum_power_across_models_and_variances']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Academic version",
            "",
            "The parametric family uses Ledoit-Wolf shrinkage of the seven-dimensional "
            "whole-pair gate vector. The nonparametric family resamples centered whole-topic-"
            "pair residual vectors, never individual route cells. Every Monte Carlo pass "
            "probability includes a Wilson 95% interval. Future candidates span the frozen "
            "n, ratio-threshold, and minimum-win-fraction grids and must meet type-I <= 0.05 "
            "at ratio 0.80 plus power >= 0.80 at ratio 0.50 under both families and all "
            "variance multipliers.",
            "",
            payload["interpretive_guardrail"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e9-analysis", type=Path, required=True)
    parser.add_argument("--e4c-analysis", type=Path, required=True)
    parser.add_argument("--e8b-analysis", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curves-csv-output", type=Path)
    parser.add_argument("--future-csv-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    e9 = _read_json(args.e9_analysis)
    e4 = _read_json(args.e4c_analysis)
    e8 = _read_json(args.e8b_analysis)
    root_config = _read_json(args.config)
    config = root_config.get("w3", root_config)
    observed, pair_ids = observed_pair_matrix(e9)
    historical = config.get("historical_gates")
    if isinstance(historical, Mapping):
        frozen_gate = GateConfig(
            name="frozen_e9",
            dependence_minimum_wins=int(historical["harmful_dependence"]["minimum_wins"]),
            rescue_minimum_wins=int(historical["harmful_rescue"]["minimum_wins"]),
            differential_minimum_each=int(historical["risk_differential"]["minimum_positive_D"]),
            smallness_minimum_wins=int(historical["relative_smallness"]["minimum_joint_wins"]),
            smallness_maximum_ratio=float(historical["relative_smallness"]["maximum_ratio"]),
        )
        if frozen_gate != FROZEN_E9_GATE:
            raise ValueError("W3 top-level config changes the exact historical E9 gate")
    else:
        frozen_gate = FROZEN_E9_GATE
    methods = ["parametric", "whole_pair_residual_bootstrap"]
    model_config = config.get("models", {})
    method_seeds = {
        "parametric": int(model_config.get("parametric", {}).get("seed", 153001)),
        "whole_pair_residual_bootstrap": int(
            model_config.get("resampling", {}).get("seed", 153002)
        ),
    }
    if "simulation" in config:
        simulation = config["simulation"]
        ratios = [float(value) for value in simulation["ratios"]]
        variances = [float(value) for value in simulation["variance_multipliers"]]
        draws = int(simulation["draws"])
        n_pairs = int(simulation["n_pairs"])
        covariance_ridge = float(simulation.get("covariance_ridge", 0.0))
        methods = [str(value) for value in simulation.get("methods", methods)]
    else:
        ratios = [float(value) for value in config["ratio_grid"]]
        variances = [float(value) for value in config["variance_multipliers"]]
        draws = int(config["monte_carlo_replicates"])
        n_pairs = int(config["sample_size"])
        covariance_ridge = 0.0
    records = run_severity_grid(
        observed,
        ratios=ratios,
        methods=methods,
        variance_multipliers=variances,
        draws=draws,
        n_pairs=n_pairs,
        batch_size=args.batch_size,
        covariance_ridge=covariance_ridge,
        root_seed=0,
        method_root_seeds=method_seeds,
        gates=[frozen_gate],
    )
    calibration = config.get("future_calibration", config.get("calibration"))
    if "sample_size_grid" in calibration:
        future = run_future_calibration_grid(
            observed,
            nonselective_null_ratio=float(calibration["nonselective_null_ratio"]),
            target_selective_ratio=float(calibration["target_selective_ratio"]),
            maximum_type_i_error=float(calibration["maximum_type_i_error"]),
            minimum_power=float(calibration["minimum_power"]),
            sample_size_grid=[int(value) for value in calibration["sample_size_grid"]],
            ratio_threshold_grid=[float(value) for value in calibration["ratio_threshold_grid"]],
            minimum_win_fraction_grid=[float(value) for value in calibration["minimum_win_fraction_grid"]],
            methods=methods,
            variance_multipliers=variances,
            draws=draws,
            batch_size=args.batch_size,
            covariance_ridge=covariance_ridge,
            method_root_seeds=method_seeds,
        )
    else:
        future = {
            "status": "legacy_config_does_not_define_frozen_future_grid",
            "selected": None,
            "candidates": [],
            "condition_probabilities": [],
        }
    retro_grid = config.get("retro_success_probability_grid", ratios)
    payload = {
        "analysis_semantics_version": "s15-w3-e9-gate-severity-v2-frozen-spec",
        "status": "complete_retro_severity_and_future_gate_calibration",
        "independent_unit": "whole matched intent pair",
        "observed_pair_ids": pair_ids,
        "observed_pair_matrix_shape": list(observed.shape),
        "observed_pair_values_exported": False,
        "simulation_config": {
            "ratios": ratios,
            "methods": methods,
            "variance_multipliers": variances,
            "draws": draws,
            "n_pairs": n_pairs,
            "batch_size": args.batch_size,
            "covariance": "ledoit_wolf",
            "method_root_seeds": method_seeds,
        },
        "gate_configs": [asdict(frozen_gate)],
        "severity_curves": records,
        "detectable_regions": detectable_regions(
            records, power_threshold=float(calibration["minimum_power"])
        ),
        "future_gate_calibration": future,
        "retro_worked_examples": retro_design_worked_examples(
            e4,
            e8,
            success_probabilities=[
                float(value) for value in retro_grid
            ],
        ),
        "interpretive_guardrail": (
            "This severity audit cannot relitigate the recorded H-005 rejection. "
            "It scopes the benign-smallness component and calibrates future designs; "
            "the recorded sign-level risk-differential result remains unchanged."
        ),
        "limitations": {
            "variance_estimated_from_audited_data": True,
            "parametric_model_may_be_misspecified": True,
            "whole_pair_bootstrap_has_only_twelve_empirical_residual_vectors": True,
            "generation_or_behavior_claim_allowed": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in {
                "e9_analysis": args.e9_analysis,
                "e4c_analysis": args.e4c_analysis,
                "e8b_analysis": args.e8b_analysis,
                "config": args.config,
            }.items()
        },
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    curves_output = args.curves_csv_output or args.output.with_name(
        args.output.stem + "_curves.csv"
    )
    future_output = args.future_csv_output or args.output.with_name(
        args.output.stem + "_future_candidates.csv"
    )
    report_output = args.report_output or args.output.with_suffix(".md")
    _write_csv(curves_output, _historical_csv_rows(records))
    if future["candidates"]:
        _write_csv(future_output, future["candidates"])
    else:
        _write_csv(future_output, [{"status": future["status"]}])
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "severity_conditions": len(records),
                "curves_csv": str(curves_output.resolve()),
                "future_candidates_csv": str(future_output.resolve()),
                "report": str(report_output.resolve()),
                "payload_sha256": payload["payload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
