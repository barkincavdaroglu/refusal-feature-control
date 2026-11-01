#!/usr/bin/env python3
"""Analyze authenticated S16 Phase-B shards without another model forward."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

import s16_phase_b_contract as contract
from run_s16_phase_b import load_feature_row


Q = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise contract.ContractError("Phase-B summary input differs")
    return {
        "count": int(array.size),
        "active_fraction": float(np.mean(array > 0.0)),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sample_standard_deviation": float(array.std(ddof=1)) if array.size > 1 else None,
        "quantiles_linear": contract.quantiles(array, Q),
    }


def _load_shards(
    running: Mapping[str, Any], output_dir: pathlib.Path, row_count: int
) -> dict[str, Any]:
    from safetensors import safe_open

    finals = np.empty(row_count, dtype=np.float32)
    final_pre = np.empty(row_count, dtype=np.float32)
    last_user = np.empty(row_count, dtype=np.float32)
    prompt_max = np.empty(row_count, dtype=np.float32)
    prompt_mean = np.empty(row_count, dtype=np.float64)
    prompt_active = np.empty(row_count, dtype=np.float64)
    all_values: dict[int, np.ndarray] = {}
    references: dict[int, dict[str, np.ndarray]] = {}
    seen: set[int] = set()
    for record in running["completed_batches"]:
        path = output_dir / record["relative_path"]
        if path.is_symlink() or not path.is_file() or contract.file_sha256(path) != record["sha256"]:
            raise contract.ContractError("Phase-B analysis shard authentication failed")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            expected = {
                "row_indices",
                "c_preactivation_all",
                "c_activation_all",
                "c_preactivation_final",
                "c_activation_final",
                "c_preactivation_last_user",
                "c_activation_last_user",
                "reference_row_indices",
                "reference_resid_post",
                "reference_resid_pre",
                "reference_mlp_hook_in",
            }
            if set(handle.keys()) != expected:
                raise contract.ContractError("Phase-B analysis shard keys differ")
            indices = handle.get_tensor("row_indices").numpy().astype(np.int64)
            activation = handle.get_tensor("c_activation_all").float().numpy()
            shard_final = handle.get_tensor("c_activation_final").float().numpy()
            shard_final_pre = handle.get_tensor("c_preactivation_final").float().numpy()
            shard_last = handle.get_tensor("c_activation_last_user").float().numpy()
            ref_indices = handle.get_tensor("reference_row_indices").numpy().astype(np.int64)
            ref_values = {
                "resid_post": handle.get_tensor("reference_resid_post").float().numpy(),
                "resid_pre": handle.get_tensor("reference_resid_pre").float().numpy(),
                "mlp_hook_in": handle.get_tensor("reference_mlp_hook_in").float().numpy(),
            }
        if activation.shape[0] != indices.size or any(int(i) in seen for i in indices):
            raise contract.ContractError("Phase-B analysis row closure differs")
        for offset, index_value in enumerate(indices):
            index = int(index_value)
            seen.add(index)
            row = activation[offset]
            finals[index] = shard_final[offset]
            final_pre[index] = shard_final_pre[offset]
            last_user[index] = shard_last[offset]
            prompt_max[index] = float(row.max())
            prompt_mean[index] = float(row.mean(dtype=np.float64))
            prompt_active[index] = float(np.mean(row > 0.0))
            all_values[index] = row
        for ref_offset, index_value in enumerate(ref_indices):
            index = int(index_value)
            references[index] = {
                key: values[ref_offset] for key, values in ref_values.items()
            }
    if seen != set(range(row_count)):
        raise contract.ContractError("Phase-B analysis is missing corpus rows")
    return {
        "final": finals,
        "final_pre": final_pre,
        "last_user": last_user,
        "prompt_max": prompt_max,
        "prompt_mean": prompt_mean,
        "prompt_active": prompt_active,
        "all_values": all_values,
        "references": references,
    }


def _by_category(values: np.ndarray, manifest: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in manifest["rows"]:
        groups[row["category"]].append(float(values[row["row_index"]]))
    return {name: _summary(groups[name]) for name in sorted(groups)}


def _micro_by_category(
    all_values: Mapping[int, np.ndarray], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in manifest["rows"]:
        groups[row["category"]].append(all_values[row["row_index"]])
    return {
        name: _summary(np.concatenate(groups[name])) for name in sorted(groups)
    }


def _position_summaries(
    all_values: Mapping[int, np.ndarray], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    reverse_bins = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**9))
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in manifest["rows"]:
        values = all_values[row["row_index"]]
        category = row["category"]
        length = values.size
        for position, value in enumerate(values):
            reverse = length - 1 - position
            label = next(f"{low}-{high if high < 10**9 else 'plus'}" for low, high in reverse_bins if low <= reverse <= high)
            grouped[(category, label)].append(float(value))
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for (category, label), values in sorted(grouped.items()):
        output[category][label] = _summary(values)
    return dict(output)


def _role_and_user_position_summaries(
    all_values: Mapping[int, np.ndarray], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return row-balanced role/decile summaries and per-row position means."""

    row_count = len(manifest["rows"])
    full_mean = np.empty(row_count, dtype=np.float64)
    user_mean = np.empty(row_count, dtype=np.float64)
    template_mean = np.empty(row_count, dtype=np.float64)
    role_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    decile_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in manifest["rows"]:
        values = all_values[row["row_index"]]
        row_index = row["row_index"]
        category = row["category"]
        start, end = row["user_token_start"], row["user_token_end_exclusive"]
        width = end - start
        template = np.concatenate((values[:start], values[end:]))
        if width <= 0 or template.size <= 0:
            raise contract.ContractError("Phase-B row position partition differs")
        full_mean[row_index] = float(np.mean(values, dtype=np.float64))
        user_mean[row_index] = float(np.mean(values[start:end], dtype=np.float64))
        template_mean[row_index] = float(np.mean(template, dtype=np.float64))
        role_groups[(category, "user_content")].append(user_mean[row_index])
        role_groups[(category, "chat_template")].append(template_mean[row_index])
        for decile in range(10):
            positions = [
                position
                for position in range(start, end)
                if min(9, ((position - start) * 10) // width) == decile
            ]
            if positions:
                decile_groups[(category, f"decile_{decile}")].append(
                    float(np.mean(values[positions], dtype=np.float64))
                )
    roles: dict[str, dict[str, Any]] = defaultdict(dict)
    deciles: dict[str, dict[str, Any]] = defaultdict(dict)
    for (category, role), values in sorted(role_groups.items()):
        roles[category][role] = {
            "estimand": "summary_across_equal_weight_per_row_role_means",
            **_summary(values),
        }
    for (category, decile), values in sorted(decile_groups.items()):
        deciles[category][decile] = {
            "estimand": "summary_across_equal_weight_per_row_decile_means",
            **_summary(values),
        }
    return (
        {
            "template_vs_user_row_balanced": dict(roles),
            "normalized_user_deciles_row_balanced": dict(deciles),
        },
        {
            "full_input_mean": full_mean,
            "user_content_mean": user_mean,
            "template_context_mean": template_mean,
        },
    )


def _endpoint_values(
    data: Mapping[str, Any], row_positions: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        "formatted_final": np.asarray(data["final"], dtype=np.float64),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in row_positions.items()
        },
    }


def _helpsteer_wrapper_effects(
    endpoints: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    families: dict[str, dict[str, int]] = defaultdict(dict)
    for row in manifest["rows"]:
        if row.get("source_id") == "nvidia/HelpSteer2":
            family = row.get("family_id")
            variant = row.get("variant_id")
            if not isinstance(family, str) or variant not in {
                "base_v1",
                *contract.HELPSTEER_WRAPPER_VARIANTS,
            }:
                raise contract.ContractError("Phase-B HelpSteer family metadata differs")
            if variant in families[family]:
                raise contract.ContractError("Phase-B HelpSteer family contains a duplicate")
            families[family][variant] = row["row_index"]
    required = {"base_v1", *contract.HELPSTEER_WRAPPER_VARIANTS}
    if len(families) != contract.HELPSTEER_ANCHOR_COUNT or any(
        set(members) != required for members in families.values()
    ):
        raise contract.ContractError("Phase-B HelpSteer matched-family closure differs")
    ordered_families = [members for _family, members in sorted(families.items())]
    output: dict[str, Any] = {}
    for variant in contract.HELPSTEER_WRAPPER_VARIANTS:
        endpoint_output: dict[str, Any] = {}
        for endpoint, values in endpoints.items():
            deltas = [
                float(values[members[variant]] - values[members["base_v1"]])
                for members in ordered_families
            ]
            endpoint_output[endpoint] = {
                "orientation": "wrapper_minus_base_within_anchor",
                "anchor_count": len(deltas),
                "summary": _summary(deltas),
            }
        output[variant] = endpoint_output
    final_matrix = np.asarray(
        [
            [
                endpoints["formatted_final"][members[variant]]
                - endpoints["formatted_final"][members["base_v1"]]
                for variant in contract.HELPSTEER_WRAPPER_VARIANTS
            ]
            for members in ordered_families
        ],
        dtype=np.float64,
    )
    return {
        "effects_by_wrapper": output,
        "formatted_final_familywise_bootstrap": (
            contract.helpsteer_familywise_mean_intervals(final_matrix)
        ),
    }


def _prespecified_subgroup_summaries(
    endpoints: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    definitions = {
        "big_bench_validity_cells": (
            lambda row: row.get("source_id") == "tasksource/bigbench",
            "balance_cell",
            {"valid": 450, "invalid": 450},
        ),
        "project_risk_splits": (
            lambda row: row.get("category") == "project_panel",
            "risk_class",
            {"harmful": 202, "benign": 98},
        ),
    }
    output: dict[str, Any] = {}
    for name, (include, field, required_counts) in definitions.items():
        groups: dict[str, list[int]] = defaultdict(list)
        for row in manifest["rows"]:
            if include(row):
                value = row.get(field)
                if value not in required_counts:
                    raise contract.ContractError(f"Phase-B {name} metadata differs")
                groups[str(value)].append(row["row_index"])
        if {cell: len(indices) for cell, indices in groups.items()} != required_counts:
            raise contract.ContractError(f"Phase-B {name} closure differs")
        output[name] = {
            cell: {
                endpoint: _summary(values[indices])
                for endpoint, values in endpoints.items()
            }
            for cell, indices in sorted(groups.items())
        }
    return output


def _reference_analysis(
    *,
    data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    decoder: np.ndarray,
    probe: np.ndarray,
    private_output: pathlib.Path,
) -> dict[str, Any]:
    from safetensors.numpy import save_file

    pairs: dict[str, dict[str, int]] = defaultdict(dict)
    for row in manifest["rows"]:
        if row.get("reference_pair_id") is not None:
            pairs[row["reference_pair_id"]][row["reference_member"]] = row["row_index"]
    if len(pairs) != contract.EXPECTED_REFERENCE_PAIR_COUNT or any(set(members) != {"harmful", "benign"} for members in pairs.values()):
        raise contract.ContractError("Phase-B analysis reference pairs differ")
    pair_ids = sorted(pairs)
    directions: dict[str, np.ndarray] = {}
    output: dict[str, Any] = {}
    for name in ("resid_post", "resid_pre", "mlp_hook_in"):
        harmful = np.stack([data["references"][pairs[pair_id]["harmful"]][name] for pair_id in pair_ids])
        benign = np.stack([data["references"][pairs[pair_id]["benign"]][name] for pair_id in pair_ids])
        direction, deltas, raw_norm = contract.pair_balanced_direction(harmful, benign)
        directions[name] = direction
        comparators = {"c_decoder": decoder, "post_final_rmsnorm_probe_coordinate_proxy": probe}
        metrics = {}
        for comparator_name, comparator in comparators.items():
            lopo = contract.lopo_cosines(deltas, comparator)
            metrics[comparator_name] = {
                "cosine": contract.cosine(direction, comparator),
                "pair_bootstrap": contract.bootstrap_cosine_interval(deltas, comparator),
                "leave_one_pair_out_cosines": lopo,
                "leave_one_pair_out_range": [
                    min(value for value in lopo if value is not None),
                    max(value for value in lopo if value is not None),
                ],
            }
        output[name] = {
            "role": "primary" if name == "resid_post" else "secondary_coordinate_sensitivity",
            "hook": {
                "resid_post": contract.PRIMARY_REFERENCE_HOOK,
                "resid_pre": "blocks.31.hook_resid_pre",
                "mlp_hook_in": contract.FEATURE_HOOK,
            }[name],
            "pair_count": len(pair_ids),
            "orientation": "harmful_minus_benign",
            "raw_mean_difference_l2": raw_norm,
            "metrics": metrics,
            "probe_coordinate_caveat": "probe was trained post-final-RMSNorm; cosine here is a coordinate proxy",
        }
    contract.ensure_private_directory(private_output.parent)
    save_file(
        {f"{name}_unit": value.astype(np.float32) for name, value in directions.items()},
        str(private_output),
    )
    os.chmod(private_output, contract.PRIVATE_FILE_MODE)
    contract.require_private_file(private_output)
    return output


def analyze(
    *,
    manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    numerical_anchor: Mapping[str, Any],
    output_dir: pathlib.Path,
    derived_output_dir: pathlib.Path | None = None,
    feature_bundle: pathlib.Path,
    probe_path: pathlib.Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    contract.validate_corpus_manifest(manifest)
    if run_manifest.get("status") != "complete":
        raise contract.ContractError("Phase-B forwards are not terminal")
    if numerical_anchor.get("status") != "complete_mps_bf16_vs_cpu_fp32_anchor":
        raise contract.ContractError("Phase-B numerical anchor is not terminal")
    data = _load_shards(run_manifest, output_dir, contract.EXPECTED_ROW_COUNT)
    _encoder, decoder_tensor, _bias = load_feature_row(feature_bundle)
    decoder = decoder_tensor.float().numpy()
    with np.load(probe_path, allow_pickle=False) as archive:
        if "direction_unit" not in archive:
            raise contract.ContractError("Phase-B probe direction differs")
        probe = np.asarray(archive["direction_unit"], dtype=np.float32)
    if decoder.shape != (2560,) or probe.shape != (2560,):
        raise contract.ContractError("Phase-B comparator dimensions differ")

    macro = {
        "maximum": _by_category(data["prompt_max"], manifest),
        "mean": _by_category(data["prompt_mean"], manifest),
        "active_token_fraction": _by_category(data["prompt_active"], manifest),
    }
    pair_rows: dict[str, dict[str, int]] = defaultdict(dict)
    for row in manifest["rows"]:
        if row.get("reference_pair_id") is not None:
            pair_rows[row["reference_pair_id"]][row["reference_member"]] = row["row_index"]
    deltas = [
        float(data["final"][members["harmful"]] - data["final"][members["benign"]])
        for _pair, members in sorted(pair_rows.items())
    ]
    if len(pair_rows) != contract.EXPECTED_REFERENCE_PAIR_COUNT or any(
        set(members) != {"harmful", "benign"} for members in pair_rows.values()
    ):
        raise contract.ContractError("Phase-B activation-delta pair closure differs")
    pair_bootstrap = contract.bootstrap_mean_interval(deltas)
    position_summaries, row_positions = _role_and_user_position_summaries(
        data["all_values"], manifest
    )
    endpoints = _endpoint_values(data, row_positions)
    helpsteer = _helpsteer_wrapper_effects(endpoints, manifest)
    subgroup_summaries = _prespecified_subgroup_summaries(endpoints, manifest)
    prioritization = contract.phase_b_prioritization(
        helpsteer_familywise=helpsteer["formatted_final_familywise_bootstrap"],
        harmful_minus_benign_bootstrap=pair_bootstrap,
    )
    derived_root = output_dir if derived_output_dir is None else derived_output_dir
    reference_path = derived_root / "private" / "s16_phase_b_reference_directions.safetensors"
    references = _reference_analysis(
        data=data,
        manifest=manifest,
        decoder=decoder,
        probe=probe,
        private_output=reference_path,
    )
    per_prompt_maxima = sorted(
        (
            {
                "row_index": row["row_index"],
                "row_id": row["row_id"],
                "category": row["category"],
                "token_index": int(np.argmax(data["all_values"][row["row_index"]])),
                "activation": float(data["prompt_max"][row["row_index"]]),
            }
            for row in manifest["rows"]
        ),
        key=lambda item: (-item["activation"], item["row_id"], item["token_index"]),
    )
    private_context_path = derived_root / "private" / "s16_phase_b_max_context_index.json"
    contract.atomic_json(
        private_context_path,
        {
            "schema_version": "s16-phase-b-private-max-context-index/1",
            "status": "private_text_free_index_for_local_join_only",
            "records": per_prompt_maxima,
            "record_count": len(per_prompt_maxima),
            "raw_prompt_text_included": False,
        },
        private_parent=True,
    )
    result = {
        "schema_version": contract.RESULT_SCHEMA,
        "semantics_version": contract.SEMANTICS_VERSION,
        "status": "complete_forward_only_activation_census",
        "created_at_utc": _utc() if created_at_utc is None else created_at_utc,
        "run_manifest_payload_sha256": run_manifest["payload_sha256"],
        "corpus_payload_sha256": manifest["corpus_payload_sha256"],
        "corpus_manifest_payload_sha256": manifest["payload_sha256"],
        "row_count": contract.EXPECTED_ROW_COUNT,
        "forward_accounting": run_manifest["forward_accounting"],
        "numerical_audit": run_manifest["audit"],
        "numerical_anchor": numerical_anchor,
        "activation_census": {
            "formatted_final_e8_compatible": _by_category(data["final"], manifest),
            "last_user_content_token": _by_category(data["last_user"], manifest),
            "all_nonpadding_token_micro": _micro_by_category(data["all_values"], manifest),
            "per_prompt_macro": macro,
            "reverse_position_bins": _position_summaries(data["all_values"], manifest),
            **position_summaries,
            "helpsteer_within_anchor_wrapper_effects": helpsteer,
            "prespecified_subgroups": subgroup_summaries,
        },
        "dose_realism": {
            "historical_value": contract.HISTORICAL_DOSE,
            "historical_role": "fixed_E8_matched_activation_delta_comparator_not_presumed_high",
            "matched_final_activation_delta": {
                "summary": _summary(deltas),
                "pair_bootstrap_mean_95": pair_bootstrap,
                "historical_percentile_interval": contract.historical_percentile_interval(deltas),
            },
            "absolute_formatted_final_all_rows": {
                "summary": _summary(data["final"]),
                "historical_percentile_interval": contract.historical_percentile_interval(data["final"]),
            },
            "absolute_formatted_final_by_category": {
                category: {
                    "summary": summary,
                    "historical_percentile_interval": contract.historical_percentile_interval(
                        [
                            data["final"][row["row_index"]]
                            for row in manifest["rows"]
                            if row["category"] == category
                        ]
                    ),
                }
                for category, summary in _by_category(data["final"], manifest).items()
            },
        },
        "reference_directions": references,
        "phase_b_descriptive_prioritization": prioritization,
        "private_artifacts": {
            "reference_directions": {
                "relative_path": reference_path.relative_to(derived_root).as_posix(),
                "sha256": contract.file_sha256(reference_path),
                "byte_count": reference_path.stat().st_size,
            },
            "max_context_index": {
                "relative_path": private_context_path.relative_to(derived_root).as_posix(),
                "sha256": contract.file_sha256(private_context_path),
                "byte_count": private_context_path.stat().st_size,
            },
        },
        "claim_boundary": {
            "interventions": 0,
            "generations": 0,
            "backward_passes": 0,
            "causal_or_semantic_function_established": False,
            "phase_c_authorized": False,
        },
        "privacy": {
            "raw_prompt_text": False,
            "raw_token_sequences": False,
            "private_residuals": False,
            "max_context_text": False,
        },
    }
    result["scientific_payload_sha256"] = contract.scientific_payload_sha256(result)
    result["payload_sha256"] = contract.canonical_sha256(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--run-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--numerical-anchor", type=pathlib.Path, required=True)
    parser.add_argument("--feature-bundle", type=pathlib.Path, required=True)
    parser.add_argument("--probe-direction", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--staging-dir", type=pathlib.Path)
    return parser.parse_args(argv)


def _validate_analysis_transition(
    *,
    run_manifest: Mapping[str, Any],
    output_dir: pathlib.Path,
    run_manifest_path: pathlib.Path,
    result_path: pathlib.Path,
) -> bool:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise contract.ContractError("Phase-B analysis output root is unsafe")
    if run_manifest.get("status") != "complete":
        raise contract.ContractError("Phase-B analysis transition differs")
    allowed = {
        run_manifest_path.resolve(),
        (output_dir / "run.log").resolve(),
    }
    for record in run_manifest["completed_batches"]:
        allowed.add((output_dir / record["relative_path"]).resolve())
    allowed.add(result_path.resolve())
    allowed.add(
        (output_dir / "private/s16_phase_b_reference_directions.safetensors").resolve()
    )
    allowed.add(
        (output_dir / "private/s16_phase_b_max_context_index.json").resolve()
    )
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise contract.ContractError("Phase-B analysis input tree contains an unsafe symlink")
        if path.is_dir():
            continue
        if path.resolve() not in allowed:
            raise contract.ContractError("Phase-B analysis input tree contains an unexpected file")
    if not result_path.exists():
        return False
    result = contract.load_authenticated_json(result_path, schema=contract.RESULT_SCHEMA)
    if result.get("run_manifest_payload_sha256") != run_manifest.get("payload_sha256"):
        raise contract.ContractError("Phase-B terminal analysis binding differs")
    for record in result.get("private_artifacts", {}).values():
        path = output_dir / record["relative_path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or contract.file_sha256(path) != record.get("sha256")
            or path.stat().st_size != record.get("byte_count")
        ):
            raise contract.ContractError("Phase-B terminal private analysis artifact differs")
        contract.require_private_file(path)
    return True


def _require_exact_role_path(
    *,
    project_root: pathlib.Path,
    plan: Mapping[str, Any],
    role: str,
    supplied: pathlib.Path,
) -> None:
    expected = project_root / pathlib.Path(*pathlib.PurePosixPath(plan["inputs"][role]["path"]).parts)
    if supplied.is_symlink() or supplied.resolve(strict=True) != expected.resolve(strict=True):
        raise contract.ContractError(f"Phase-B analyzer input path differs: {role}")


def _analysis_stage(output_dir: pathlib.Path, requested: pathlib.Path | None) -> pathlib.Path:
    expected = output_dir.parent / f".{output_dir.name}.analysis-stage"
    stage = expected if requested is None else requested
    if stage.is_symlink() or stage.absolute() != expected.absolute():
        raise contract.ContractError("Phase-B analysis staging path differs")
    contract.ensure_private_directory(stage)
    allowed = {
        "s16_phase_b_result.json",
        "private/s16_phase_b_reference_directions.safetensors",
        "private/s16_phase_b_max_context_index.json",
    }
    for path in stage.rglob("*"):
        if path.is_symlink():
            raise contract.ContractError("Phase-B analysis staging tree contains a symlink")
        if path.is_file() and path.relative_to(stage).as_posix() not in allowed:
            raise contract.ContractError("Phase-B analysis staging tree contains an unexpected file")
    return stage


def _atomic_copy_private(source: pathlib.Path, target: pathlib.Path) -> None:
    contract.require_private_file(source)
    contract.ensure_private_directory(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.close(descriptor)
        shutil.copyfile(source, temporary)
        os.chmod(temporary, contract.PRIVATE_FILE_MODE)
        if target.exists():
            if (
                target.is_symlink()
                or not target.is_file()
                or contract.file_sha256(target) != contract.file_sha256(source)
            ):
                raise contract.ContractError("Phase-B committed analysis artifact differs")
            temporary.unlink()
        else:
            os.replace(temporary, target)
        contract.require_private_file(target)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = contract.load_authenticated_json(args.corpus_manifest, schema=contract.CORPUS_SCHEMA)
    contract.validate_corpus_manifest(manifest)
    run_manifest = contract.load_authenticated_json(args.run_manifest, schema="s16-phase-b-local-run-manifest/1")
    numerical_anchor = contract.load_authenticated_json(args.numerical_anchor, schema="s16-phase-b-numerical-anchor/1")
    for role, path in (
        ("corpus_manifest", args.corpus_manifest),
        ("feature_bundle", args.feature_bundle),
        ("probe_direction", args.probe_direction),
    ):
        if contract.file_sha256(path) != run_manifest.get("input_hashes", {}).get(role):
            raise contract.ContractError(f"Phase-B analyzer input hash differs: {role}")
    if numerical_anchor.get("mps_run_manifest_payload_sha256") != run_manifest.get(
        "payload_sha256"
    ):
        raise contract.ContractError("Phase-B numerical anchor/run binding differs")
    if _validate_analysis_transition(
        run_manifest=run_manifest,
        output_dir=args.output_dir,
        run_manifest_path=args.run_manifest,
        result_path=args.output,
    ):
        return 0
    if (
        args.run_manifest.resolve(strict=True)
        != (args.output_dir / "s16_phase_b_run_manifest.json").resolve(strict=True)
        or args.output.absolute()
        != (args.output_dir / "s16_phase_b_result.json").absolute()
    ):
        raise contract.ContractError("Phase-B analyzer output path differs")
    stage = _analysis_stage(args.output_dir, args.staging_dir)
    stage_result = stage / "s16_phase_b_result.json"
    committed_created_at = None
    if args.output.exists():
        committed = contract.load_authenticated_json(args.output, schema=contract.RESULT_SCHEMA)
        committed_created_at = committed.get("created_at_utc")
    result = analyze(
        manifest=manifest,
        run_manifest=run_manifest,
        numerical_anchor=numerical_anchor,
        output_dir=args.output_dir,
        derived_output_dir=stage,
        feature_bundle=args.feature_bundle,
        probe_path=args.probe_direction,
        created_at_utc=committed_created_at,
    )
    contract.atomic_json(stage_result, result, private_parent=True)
    for relative in (
        "private/s16_phase_b_reference_directions.safetensors",
        "private/s16_phase_b_max_context_index.json",
        "s16_phase_b_result.json",
    ):
        _atomic_copy_private(stage / relative, args.output_dir / relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
