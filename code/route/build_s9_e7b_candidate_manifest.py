#!/usr/bin/env python3
"""Materialize E7b v1's exact pre-outcome receiver-route panel locally.

This builder reads no E7 route outcome. It combines the completed E3 mechanics
grid with E5c's predictor-only exact plan, derives unsigned backward-walk and
signed virtual-Jacobian topology features from hash-pinned graphs, removes
feature-level contrasts that alias after promotion to a complete receiver MLP,
and emits the exact cells that a later E7c runner may execute.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from e7_topology import (
    compute_sender_hit_probabilities,
    compute_signed_transport_features,
    compute_unsigned_route_features,
    summarize_receiver_layers,
)
from s9_contract import canonical_sha256, parse_feature, sha256_file
from s9_e5c_contract import validate_e5c_exact_plan
from s9_path_contract import validate_path_plan, validate_path_result


SCHEMA_VERSION = 1
SEMANTICS_VERSION = "s9-e7b-receiver-route-panel-v1"
STATUS = "complete_preoutcome_route_panel_manifest_v1"
EDGE_THRESHOLD = 0.90
SIGNED_COHERENT_MIN = 0.85
SIGNED_CONFLICTED_MAX = 0.50
TOPOLOGY_ROLES = (
    "anchor_high",
    "weak_same_receiver",
    "edge_permuted_same_receiver",
)
DIRECTION_BAND_TARGETS = {
    "flip_h07_d0_f0:a_from_b": "middle",
    "flip_h07_d0_f0:b_from_a": "early",
    "same_h01_d0_f1:a_from_b": "late",
    "same_h01_d0_f1:b_from_a": "middle",
    "benign_b01_d0_f0:a_from_b": "late",
    "benign_b01_d0_f0:b_from_a": "early",
}
RECEIVER_BANDS = {
    "early": (1, 12),
    "middle": (13, 24),
    "late": (25, 35),
}
PRIVACY = {
    "prompt_text_in_public_artifacts": False,
    "completion_text_in_public_artifacts": False,
    "token_text_in_public_artifacts": False,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def _input(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _feature_lookup(graph) -> dict[str, int]:
    final_position = graph.n_pos - 1
    output: dict[str, int] = {}
    for node in np.flatnonzero((graph.kind == "feature") & (graph.pos == final_position)):
        label = f"L{int(graph.layer[node])}:F{int(graph.feature_id[node])}"
        if label in output:
            raise ValueError(f"{graph.name}: duplicate final-position feature {label}")
        output[label] = int(node)
    return output


def _signed_class(cancellation_ratio: float, path_strength: float) -> str:
    if path_strength <= 1e-12:
        return "no_local_transport"
    if cancellation_ratio >= SIGNED_COHERENT_MIN:
        return "coherent"
    if cancellation_ratio <= SIGNED_CONFLICTED_MAX:
        return "conflicted"
    return "mixed"


class GraphWorkspace:
    def __init__(self, path: Path, expected_sha256: str):
        from agtk.io import load_pt

        if sha256_file(path) != expected_sha256:
            raise ValueError(f"Graph bytes differ from pinned hash: {path}")
        self.graph = load_pt(path, hash_file=True)
        if self.graph.meta["file_sha256"] != expected_sha256:
            raise ValueError(f"Graph loader hash differs: {path}")
        self.pruned = self.graph.prune_edges(EDGE_THRESHOLD)
        self.normalized = self.pruned.normalized()
        self.lookup = _feature_lookup(self.graph)
        self.hit_cache: dict[int, np.ndarray] = {}

    def index(self, feature: str) -> int:
        try:
            return self.lookup[feature]
        except KeyError as error:
            raise ValueError(
                f"{self.graph.name}: final-position feature is absent: {feature}"
            ) from error

    def has_feature(self, feature: str) -> bool:
        return feature in self.lookup

    def hit(self, source: int) -> np.ndarray:
        if source not in self.hit_cache:
            self.hit_cache[source] = compute_sender_hit_probabilities(
                normalized_adjacency=self.normalized,
                source_index=source,
            )
        return self.hit_cache[source]

    def topology(
        self,
        *,
        source_feature: str,
        receiver_feature: str,
        source_activation_delta: float,
        receiver_target_preactivation: float,
    ) -> dict[str, Any]:
        source = self.index(source_feature)
        receiver = self.index(receiver_feature)
        hit = self.hit(source)
        unsigned = compute_unsigned_route_features(
            normalized_adjacency=self.normalized,
            node_kinds=self.pruned.kind,
            source_index=source,
            receiver_index=receiver,
            sender_hit_probabilities=hit,
        )
        signed = compute_signed_transport_features(
            raw_adjacency=self.graph.A,
            node_kinds=self.graph.kind,
            layers=self.graph.layer,
            positions=self.graph.pos,
            activations=self.graph.activation,
            source_index=source,
            receiver_index=receiver,
        )
        predicted_delta = signed.total_signed_jacobian * float(source_activation_delta)
        target_pre = float(receiver_target_preactivation)
        return {
            "unsigned": unsigned.to_dict(),
            "signed": {
                **signed.to_dict(),
                "class": _signed_class(
                    signed.cancellation_ratio,
                    signed.total_absolute_path_strength,
                ),
                "predicted_receiver_preactivation_delta": predicted_delta,
                "predicted_receiver_preactivation": target_pre + predicted_delta,
                "predicted_receiver_gate_crossing": (
                    (target_pre <= 0 < target_pre + predicted_delta)
                    or (target_pre > 0 >= target_pre + predicted_delta)
                ),
            },
        }

    def low_coverage_layer(
        self,
        *,
        source_feature: str,
        primary_receiver_layer: int,
        excluded_layers: Sequence[int],
    ) -> dict[str, Any]:
        source = self.index(source_feature)
        source_layer, _ = parse_feature(source_feature)
        hit = self.hit(source)
        summaries = summarize_receiver_layers(
            sender_hit_probabilities=hit,
            node_kinds=self.graph.kind,
            layers=self.graph.layer,
            positions=self.graph.pos,
            activations=self.graph.activation,
            source_layer=source_layer,
            token_position=self.graph.n_pos - 1,
        )
        band = next(
            name
            for name, (lo, hi) in RECEIVER_BANDS.items()
            if lo <= primary_receiver_layer <= hi
        )
        lo, hi = RECEIVER_BANDS[band]
        excluded = set(int(value) for value in excluded_layers)
        feasible = [
            row
            for row in summaries
            if lo <= row.layer <= hi and row.layer not in excluded
        ]
        if not feasible:
            feasible = [row for row in summaries if row.layer not in excluded]
        if not feasible:
            raise ValueError(
                f"{self.graph.name}: no later receiver layer for {source_feature}"
            )
        chosen = min(
            feasible,
            key=lambda row: (
                row.maximum_sender_coverage,
                row.mean_sender_coverage,
                row.p90_sender_coverage,
                abs(row.layer - primary_receiver_layer),
                row.layer,
            ),
        )
        return {
            **chosen.to_dict(),
            "selection_band": band,
            "primary_receiver_layer": primary_receiver_layer,
            "excluded_layers": sorted(excluded),
            "selection_semantics": (
                "minimum maximum sender coverage across active final-position "
                "feature markers in a band-matched alternate receiver MLP"
            ),
            "claim_boundary": (
                "operational low-coverage module control; not proof that the "
                "complete module is physically off-path"
            ),
        }


def _predictor_relation(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    graph_margin = abs(float(first["graph_predicted_receiver_preactivation_delta"])) - abs(
        float(second["graph_predicted_receiver_preactivation_delta"])
    )
    atp_margin = abs(float(first["atp_predicted_receiver_preactivation_delta"])) - abs(
        float(second["atp_predicted_receiver_preactivation_delta"])
    )
    if abs(graph_margin) <= 1e-12 or abs(atp_margin) <= 1e-12:
        relation = "tie"
    elif graph_margin * atp_margin < 0:
        relation = "disagreement"
    else:
        relation = "agreement"
    return {
        "relation": relation,
        "graph_absolute_margin_first_minus_second": graph_margin,
        "atp_absolute_margin_first_minus_second": atp_margin,
        "robust_margin": min(abs(graph_margin), abs(atp_margin)),
    }


def select_topology_trios(
    enriched_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one three-cell same-receiver trio per direction.

    The three route-distinct cells are the anchor, weak same-receiver source,
    and edge-permuted same-receiver source. Pairwise comparisons within the trio
    must contain at least one graph/AtP agreement and one disagreement. This
    avoids the same-source feature contrasts that alias to the identical full
    receiver-module intervention.
    """

    by_direction_quartet: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in enriched_candidates:
        if row["pool_role"] in TOPOLOGY_ROLES:
            by_direction_quartet[(row["direction_id"], row["quartet_id"])].append(row)

    selected_cells: list[dict[str, Any]] = []
    selected_contrasts: list[dict[str, Any]] = []
    for direction_id, desired_band in DIRECTION_BAND_TARGETS.items():
        options = []
        for (candidate_direction, quartet_id), trio in by_direction_quartet.items():
            if candidate_direction != direction_id or not quartet_id.startswith(f"{desired_band}_"):
                continue
            by_role = {row["pool_role"]: row for row in trio}
            if set(by_role) != set(TOPOLOGY_ROLES):
                continue
            ordered = [by_role[role] for role in TOPOLOGY_ROLES]
            if len({row["receiver_feature"] for row in ordered}) != 1:
                raise ValueError(f"{direction_id}/{quartet_id}: receiver trio differs")
            if len({row["source_edit_id"] for row in ordered}) != 3:
                raise ValueError(f"{direction_id}/{quartet_id}: route cells alias")
            l2_values = [float(row["source_edit_l2"]) for row in ordered]
            if min(l2_values) <= 0 or max(l2_values) / min(l2_values) > 1.001:
                raise ValueError(f"{direction_id}/{quartet_id}: trio is not L2 matched")
            pairs = []
            for first, second in itertools.combinations(ordered, 2):
                relation = _predictor_relation(first, second)
                pairs.append(
                    {
                        **relation,
                        "first_candidate_id": first["candidate_id"],
                        "second_candidate_id": second["candidate_id"],
                    }
                )
            relations = {row["relation"] for row in pairs}
            if not {"agreement", "disagreement"} <= relations:
                continue
            signed_classes = {row["topology"]["signed"]["class"] for row in ordered}
            coverages = [float(row["topology"]["unsigned"]["receiver_coverage"]) for row in ordered]
            conflicts = [float(row["topology"]["signed"]["conflict_fraction"]) for row in ordered]
            best_disagreement_margin = max(
                float(row["robust_margin"])
                for row in pairs
                if row["relation"] == "disagreement"
            )
            best_agreement_margin = max(
                float(row["robust_margin"])
                for row in pairs
                if row["relation"] == "agreement"
            )
            options.append(
                {
                    "quartet_id": quartet_id,
                    "cells": ordered,
                    "pairs": pairs,
                    "score": (
                        best_disagreement_margin,
                        best_agreement_margin,
                        max(conflicts),
                        len(signed_classes),
                        max(coverages) - min(coverages),
                        max(
                            float(row["topology"]["unsigned"]["route_effective_count"])
                            for row in ordered
                        ),
                    ),
                }
            )
        if not options:
            raise ValueError(
                f"{direction_id}: no non-aliased {desired_band} trio spans agreement/disagreement"
            )
        chosen = sorted(
            options,
            key=lambda row: (
                -row["score"][0],
                -row["score"][1],
                -row["score"][2],
                -row["score"][3],
                -row["score"][4],
                -row["score"][5],
                row["quartet_id"],
            ),
        )[0]
        selected_cells.extend(dict(row) for row in chosen["cells"])
        for relation in ("disagreement", "agreement"):
            eligible = [row for row in chosen["pairs"] if row["relation"] == relation]
            pair = sorted(
                eligible,
                key=lambda row: (
                    -float(row["robust_margin"]),
                    row["first_candidate_id"],
                    row["second_candidate_id"],
                ),
            )[0]
            selected_contrasts.append(
                {
                    "contrast_id": f"e7b:{direction_id}:{chosen['quartet_id']}:{relation}",
                    "direction_id": direction_id,
                    "quartet_id": chosen["quartet_id"],
                    "receiver_band": desired_band,
                    "predictor_relation": relation,
                    "control_family": "same_receiver_l2_matched_trio_pair",
                    **{key: value for key, value in pair.items() if key != "relation"},
                }
            )
    selected_cells.sort(key=lambda row: (row["direction_id"], row["candidate_id"]))
    selected_contrasts.sort(key=lambda row: row["contrast_id"])
    if len(selected_cells) != 18 or len(selected_contrasts) != 12:
        raise ValueError("E7b topology selection must contain 18 cells and 12 contrasts")
    return selected_cells, selected_contrasts


def _route_cell(
    *,
    cell_id: str,
    panel: str,
    context_id: str,
    direction_id: str,
    pair_id: str | None,
    pair_type: str,
    target_prompt_key: str,
    source_prompt_key: str,
    source_feature: str,
    source_edit_id: str,
    source_edit_policy: str,
    source_activation_delta: float,
    source_edit_l2: float,
    receiver_layer: int,
    nominated_receiver_feature: str | None,
    topology: Mapping[str, Any] | None,
    role: str,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "panel": panel,
        "context_id": context_id,
        "direction_id": direction_id,
        "pair_id": pair_id,
        "pair_type": pair_type,
        "target_prompt_key": target_prompt_key,
        "source_prompt_key": source_prompt_key,
        "source_feature": source_feature,
        "source_edit_id": source_edit_id,
        "source_edit_policy": source_edit_policy,
        "source_activation_delta": float(source_activation_delta),
        "source_edit_l2": float(source_edit_l2),
        "receiver_layer": int(receiver_layer),
        "receiver_boundary": "module_output",
        "nominated_receiver_feature": nominated_receiver_feature,
        "topology": dict(topology) if topology is not None else None,
        "role": role,
        "route_signature": (
            f"{direction_id}|{source_edit_id}|L{receiver_layer}|module_output"
        ),
    }


def build_manifest(
    *,
    feature_path_plan_path: Path,
    feature_path_result_path: Path,
    e5c_exact_plan_path: Path,
    e7b_draft_path: Path,
    handoff_path: Path,
    e7a_report_path: Path,
    topology_source_path: Path,
    graph_toolkit_root: Path,
) -> dict[str, Any]:
    feature_plan = _read(feature_path_plan_path)
    feature_result = _read(feature_path_result_path)
    e5c_plan = _read(e5c_exact_plan_path)
    validate_path_plan(feature_plan)
    validate_path_result(feature_result, feature_plan)
    validate_e5c_exact_plan(e5c_plan)

    graph_specs = {
        row["graph_id"]: {
            "path": Path(row["selection_graph"]["path"]),
            "sha256": row["selection_graph"]["sha256"],
        }
        for row in e5c_plan["graph_panels"]
    }
    mechanics_graph_id = feature_plan["graph_evidence"]["graph_name"]
    graph_specs[mechanics_graph_id] = {
        "path": Path(feature_plan["graph_evidence"]["graph_path"]),
        "sha256": feature_plan["graph_evidence"]["graph_sha256"],
    }
    workspaces = {
        graph_id: GraphWorkspace(spec["path"], spec["sha256"])
        for graph_id, spec in sorted(graph_specs.items())
    }

    enriched: list[dict[str, Any]] = []
    for row in e5c_plan["exact_candidates"]:
        workspace = workspaces[row["graph_id"]]
        topology = workspace.topology(
            source_feature=row["source_feature"],
            receiver_feature=row["receiver_feature"],
            source_activation_delta=float(row["applied_activation_delta"]),
            receiver_target_preactivation=float(row["receiver_target_preactivation"]),
        )
        direct_prediction = (
            float(topology["signed"]["direct_jacobian"])
            * float(row["applied_activation_delta"])
        )
        if not math.isclose(
            direct_prediction,
            float(row["graph_predicted_receiver_preactivation_delta"]),
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{row['candidate_id']}: direct graph prediction differs")
        enriched.append({**dict(row), "topology": topology})

    selected_topology, topology_contrasts = select_topology_trios(enriched)
    selected_by_id = {row["candidate_id"]: row for row in selected_topology}
    topology_cells = []
    for row in selected_topology:
        topology_cells.append(
            _route_cell(
                cell_id=f"topology:{row['direction_id']}:{row['candidate_id']}",
                panel="topology_discrimination",
                context_id=row["direction_id"],
                direction_id=row["direction_id"],
                pair_id=row["pair_id"],
                pair_type=row["pair_type"],
                target_prompt_key=row["target_prompt_key"],
                source_prompt_key=row["source_prompt_key"],
                source_feature=row["source_feature"],
                source_edit_id=row["source_edit_id"],
                source_edit_policy=row["source_edit_policy"],
                source_activation_delta=float(row["applied_activation_delta"]),
                source_edit_l2=float(row["source_edit_l2"]),
                receiver_layer=int(row["receiver_layer"]),
                nominated_receiver_feature=row["receiver_feature"],
                topology=row["topology"],
                role=row["pool_role"],
            )
        )

    mechanics_workspace = workspaces[mechanics_graph_id]
    mechanics_cells = []
    mechanics_relation_topology: dict[str, dict[str, Any]] = {}
    relation_specs = {row["relation_id"]: row for row in feature_plan["relations"]}
    for row in feature_result["relations"]:
        relation = relation_specs[row["relation_id"]]
        receiver_layer, _ = parse_feature(row["receiver_feature"])
        topology_key = row["relation_id"]
        if topology_key not in mechanics_relation_topology:
            if mechanics_workspace.has_feature(
                row["source_feature"]
            ) and mechanics_workspace.has_feature(row["receiver_feature"]):
                mechanics_relation_topology[topology_key] = mechanics_workspace.topology(
                    source_feature=row["source_feature"],
                    receiver_feature=row["receiver_feature"],
                    source_activation_delta=1.0,
                    receiver_target_preactivation=float(
                        row["edge_estimates"]["target_receiver_preactivation"]
                    ),
                )
            else:
                mechanics_relation_topology[topology_key] = {
                    "status": "not_available_in_reference_graph",
                    "missing_features": sorted(
                        feature
                        for feature in (row["source_feature"], row["receiver_feature"])
                        if not mechanics_workspace.has_feature(feature)
                    ),
                    "claim_boundary": (
                        "matched controls absent from the selected graph remain valid "
                        "causal controls but receive no invented topology features"
                    ),
                }
        reference_topology = json.loads(json.dumps(mechanics_relation_topology[topology_key]))
        reference_topology["reference_graph_id"] = mechanics_graph_id
        reference_topology["transported_context_warning"] = (
            None
            if row["target_prompt_key"] == mechanics_graph_id
            else "topology is an H03-D0 design reference, not a current-context prediction"
        )
        source_edit_id = (
            f"mechanics:{row['context_id']}:{row['relation_id']}:{row['source_feature']}"
        )
        context = next(item for item in feature_plan["contexts"] if item["context_id"] == row["context_id"])
        mechanics_cells.append(
            _route_cell(
                cell_id=f"mechanics:{row['context_id']}:{row['relation_id']}",
                panel="mechanics_core",
                context_id=row["context_id"],
                direction_id=row["context_id"],
                pair_id=None,
                pair_type=row["context_class"],
                target_prompt_key=row["target_prompt_key"],
                source_prompt_key=context["delta_source_prompt_key"],
                source_feature=row["source_feature"],
                source_edit_id=source_edit_id,
                source_edit_policy=(
                    "natural_counterfactual"
                    if relation["graph_status"] == "graph_proposed"
                    else "exact_l2_matched_to_graph_relation"
                ),
                source_activation_delta=float(row["source_activation_delta"]),
                source_edit_l2=float(row["source_edit_l2"]),
                receiver_layer=receiver_layer,
                nominated_receiver_feature=row["receiver_feature"],
                topology=reference_topology,
                role=relation["graph_status"],
            )
        )

    mechanics_cells.sort(key=lambda row: row["cell_id"])
    topology_cells.sort(key=lambda row: row["cell_id"])
    all_primary = mechanics_cells + topology_cells
    route_signatures = [row["route_signature"] for row in all_primary]
    if len(route_signatures) != len(set(route_signatures)):
        raise ValueError("E7b primary full-module route cells contain aliases")

    input_calibration_ids = [
        row["cell_id"]
        for row in mechanics_cells
        if row["role"] == "graph_proposed"
    ] + [
        row["cell_id"]
        for row in topology_cells
        if row["role"] == "anchor_high"
    ]

    offpath_controls = []
    mechanics_a = {
        row["context_id"]: row
        for row in mechanics_cells
        if row["cell_id"].endswith(":a_to_c")
    }
    mechanics_low_layer = mechanics_workspace.low_coverage_layer(
        source_feature="L29:F18221",
        primary_receiver_layer=31,
        excluded_layers=(31, 32),
    )
    for context_id, source_cell in sorted(mechanics_a.items()):
        offpath_controls.append(
            {
                "control_id": f"low_coverage:mechanics:{context_id}",
                "panel": "mechanics_core",
                "source_cell_id": source_cell["cell_id"],
                "direction_id": source_cell["direction_id"],
                "source_edit_id": source_cell["source_edit_id"],
                "source_feature": source_cell["source_feature"],
                "receiver_layer": mechanics_low_layer["layer"],
                "receiver_boundary": "module_output",
                "selection": mechanics_low_layer,
            }
        )
    anchors = [row for row in selected_topology if row["pool_role"] == "anchor_high"]
    for anchor in sorted(anchors, key=lambda row: row["direction_id"]):
        workspace = workspaces[anchor["graph_id"]]
        low_layer = workspace.low_coverage_layer(
            source_feature=anchor["source_feature"],
            primary_receiver_layer=int(anchor["receiver_layer"]),
            excluded_layers=(int(anchor["receiver_layer"]),),
        )
        source_cell = next(
            row for row in topology_cells if row["cell_id"].endswith(anchor["candidate_id"])
        )
        offpath_controls.append(
            {
                "control_id": f"low_coverage:topology:{anchor['direction_id']}",
                "panel": "topology_discrimination",
                "source_cell_id": source_cell["cell_id"],
                "direction_id": anchor["direction_id"],
                "source_edit_id": anchor["source_edit_id"],
                "source_feature": anchor["source_feature"],
                "receiver_layer": low_layer["layer"],
                "receiver_boundary": "module_output",
                "selection": low_layer,
            }
        )
    offpath_controls.sort(key=lambda row: row["control_id"])

    joint_receiver_sets = []
    for context_id in ("h03_d0_from_f0", "h09_d0_from_f0"):
        source_cell = mechanics_a[context_id]
        joint_receiver_sets.append(
            {
                "cell_id": f"joint_receiver_set:{context_id}:L31+L32",
                "context_id": context_id,
                "source_cell_id": source_cell["cell_id"],
                "direction_id": source_cell["direction_id"],
                "source_edit_id": source_cell["source_edit_id"],
                "source_feature": source_cell["source_feature"],
                "receiver_layers": [31, 32],
                "receiver_boundary": "module_output",
                "interpretation": "causally ordered receiver-set dependence, not redundancy",
            }
        )

    prompt_records: dict[str, dict[str, Any]] = {}
    for row in list(feature_plan["prompt_records"]) + list(e5c_plan["prompt_records"]):
        key = row["prompt_key"]
        if key in prompt_records and prompt_records[key] != row:
            raise ValueError(f"Prompt record differs across inputs: {key}")
        prompt_records[key] = dict(row)

    source_edits = {row["source_edit_id"] for row in all_primary}
    expected_forwards = {
        "unique_prompt_baseline_or_counterfactual_caches": len(prompt_records),
        "unique_sender_patched_worlds": len(source_edits),
        "primary_output_restoration_and_rescue": 2 * len(all_primary),
        "input_calibration_restoration_and_rescue": 2 * len(input_calibration_ids),
        "low_coverage_control_restoration_and_rescue": 2 * len(offpath_controls),
        "joint_receiver_set_restoration_and_rescue": 2 * len(joint_receiver_sets),
    }
    expected_forwards["total"] = sum(expected_forwards.values())
    expected_forwards["audit_ceiling"] = math.ceil(expected_forwards["total"] * 1.10)
    if expected_forwards["total"] != 223:
        raise ValueError(f"E7b exact forward accounting differs: {expected_forwards}")

    graph_toolkit_inputs = {
        name: graph_toolkit_root / "src" / "agtk" / name
        for name in ("io.py", "core.py", "mass.py")
    }
    inputs = {
        "feature_path_plan": feature_path_plan_path,
        "feature_path_result": feature_path_result_path,
        "e5c_exact_plan": e5c_exact_plan_path,
        "e7b_draft": e7b_draft_path,
        "e7_handoff": handoff_path,
        "e7a_report": e7a_report_path,
        "e7_topology": topology_source_path,
        "manifest_builder": Path(__file__),
        **{f"graph_toolkit_{name[:-3]}": path for name, path in graph_toolkit_inputs.items()},
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_semantics_version": SEMANTICS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "mode": "adaptive_exploration",
        "research_stage": "understanding_testing_hypotheses",
        "objective": (
            "freeze the smallest exact model-native receiver-route development "
            "panel before any E7 original-model outcome"
        ),
        "authorization": {
            "explicit_user_confirmation_received": True,
            "confirmation_scope": "E7c after exact E7b v1 pre-outcome freeze",
            "e7_outcomes_seen_by_builder": False,
        },
        "claim_boundary": {
            "development_only": True,
            "generalization_unit": "independent intent, not route cell",
            "receiver_feature_is_a_topology_marker_not_the_intervened_tensor": True,
            "complete_module_output_is_primary": True,
            "module_input_is_separate_calibration": True,
            "low_coverage_control_is_not_proven_physically_off_path": True,
            "no_generation": True,
        },
        "topology_semantics": {
            "unsigned": "0.90-pruned row-normalized absolute backward-walk sender coverage",
            "signed": (
                "dynamic programming over J(target<-source)=A(target,source)/activation(source), "
                "with intermediate ReLU gates and no multiplication of raw attribution edges"
            ),
            "signed_coherent_min_cancellation_ratio": SIGNED_COHERENT_MIN,
            "signed_conflicted_max_cancellation_ratio": SIGNED_CONFLICTED_MAX,
            "graph_edge_threshold": EDGE_THRESHOLD,
            "separate_predictor_families": True,
        },
        "topology_selection": {
            "direction_band_targets": dict(DIRECTION_BAND_TARGETS),
            "roles": list(TOPOLOGY_ROLES),
            "same_source_aliases_excluded": True,
            "alias_key": "direction + source_edit_id + receiver_layer + boundary",
            "selection_used_e7_outcomes": False,
            "selection_used_e5c_exact_receiver_outcomes": False,
            "selected_cells": len(topology_cells),
            "selected_contrasts": len(topology_contrasts),
        },
        "model": feature_plan["model"],
        "transcoder": feature_plan["transcoder"],
        "runtime_contract": feature_plan["runtime_contract"],
        "prompt_records": [prompt_records[key] for key in sorted(prompt_records)],
        "mechanics_cells": mechanics_cells,
        "topology_cells": topology_cells,
        "topology_contrasts": topology_contrasts,
        "input_calibration_cell_ids": sorted(input_calibration_ids),
        "low_coverage_controls": offpath_controls,
        "joint_receiver_sets": joint_receiver_sets,
        "expected_counts": {
            "mechanics_primary_cells": len(mechanics_cells),
            "topology_primary_cells": len(topology_cells),
            "primary_cells": len(all_primary),
            "input_calibration_cells": len(input_calibration_ids),
            "low_coverage_controls": len(offpath_controls),
            "joint_receiver_sets": len(joint_receiver_sets),
            "unique_source_edits": len(source_edits),
            "unique_prompts": len(prompt_records),
        },
        "expected_forwards": expected_forwards,
        "graph_inputs": {
            graph_id: _input(spec["path"])
            for graph_id, spec in sorted(graph_specs.items())
        },
        "inputs": {name: _input(path) for name, path in sorted(inputs.items())},
        "privacy": PRIVACY,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path-plan", type=Path, required=True)
    parser.add_argument("--feature-path-result", type=Path, required=True)
    parser.add_argument("--e5c-exact-plan", type=Path, required=True)
    parser.add_argument("--e7b-draft", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--e7a-report", type=Path, required=True)
    parser.add_argument("--topology-source", type=Path, required=True)
    parser.add_argument("--graph-toolkit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        feature_path_plan_path=args.feature_path_plan,
        feature_path_result_path=args.feature_path_result,
        e5c_exact_plan_path=args.e5c_exact_plan,
        e7b_draft_path=args.e7b_draft,
        handoff_path=args.handoff,
        e7a_report_path=args.e7a_report,
        topology_source_path=args.topology_source,
        graph_toolkit_root=args.graph_toolkit_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "expected_counts": payload["expected_counts"],
                "expected_forwards": payload["expected_forwards"],
                "payload_sha256": payload["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
