#!/usr/bin/env python3
"""Hash-close E8b's authorized runtime bundle around immutable E8a science."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from s9_contract import sha256_file, stamp_payload
from s9_e8_contract import validate_e8a_plan
from s9_e8b_contract import (
    E8A_FROZEN_MANIFEST_FILE_SHA256,
    E8A_FROZEN_PLAN_FILE_SHA256,
    E8B_AUTHORIZATION_SCOPE,
    E8B_FORWARD_CONTRACT,
    E8B_INPUT_NAMES,
    E8B_NUMERICAL_CONTRACT,
    E8B_OUTPUT_CONTRACT,
    E8B_PLAN_SEMANTICS_VERSION,
    E8B_PLAN_STATUS,
    E8B_PRIVACY,
    E8B_RUNTIME_CONTRACT,
    E8B_SCHEMA_VERSION,
    E8B_SEMANTIC_RESOLUTION,
    e8a_scientific_snapshot_sha256,
    validate_e8b_execution_plan,
)


STEP = Path(__file__).resolve().parents[1]
ALL_WORK = STEP.parents[1]
S10 = ALL_WORK / "steps/10-target-and-dataset-repair"
OUTPUTS = STEP / "outputs"
SOURCE = STEP / "src"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def _input(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_execution_inputs(
    manifest_path: Path,
    e8a_plan_path: Path,
) -> dict[str, Path]:
    """Return the complete intended E8b upload/analysis byte closure.

    Several paths intentionally name future E8b runner/analyzer artifacts.  A
    plan cannot be materialized until every one exists and is reviewable.
    """

    paths = {
        "e8a_manifest": Path(manifest_path),
        "e8a_plan": Path(e8a_plan_path),
        "semantic_erratum": STEP / "E8B_PRE_OUTCOME_SEMANTIC_ERRATUM.md",
        "private_prompts": S10
        / "private/s10_phase_b_validation_expansion_prompts.csv",
        "probe_direction": ALL_WORK
        / "steps/06-lexical-controls-and-causal-tests/outputs/"
        "qwen3_4b_refusal_probe_direction.npz",
        "transcoder_row_bundle": STEP / "private/s9_e8b_transcoder_rows.safetensors",
        "transcoder_row_manifest": OUTPUTS
        / "s9_e8b_transcoder_rows_manifest.json",
        "transcoder_bundle_builder": SOURCE / "build_s9_e8b_transcoder_bundle.py",
        "range_helper": SOURCE / "extract_s9_e8b_transcoder_rows.py",
        "runner": SOURCE / "run_s9_e8b.py",
        "colab_runner": SOURCE / "run_s9_e8b_colab.py",
        "runtime_probe": SOURCE / "probe_s9_e8b_runtime_colab.py",
        "analyzer": SOURCE / "analyze_s9_e8b.py",
        "runtime_installer": SOURCE / "install_s9_colab_runtime.py",
        "base_runner": SOURCE / "run_s9_calibration.py",
        "base_contract": SOURCE / "s9_contract.py",
        "e8_contract": SOURCE / "s9_e8_contract.py",
        "e8b_contract": SOURCE / "s9_e8b_contract.py",
        "execution_plan_builder": Path(__file__),
    }
    if set(paths) != set(E8B_INPUT_NAMES):
        raise ValueError("E8b default input closure differs")
    return paths


def _verify_declared_e8a_inputs(payload: Mapping[str, Any], label: str) -> None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Immutable E8a {label} input closure is missing")
    for name, record in inputs.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Immutable E8a {label} input is malformed: {name}")
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Immutable E8a {label} input path is missing: {name}")
        path = Path(path_value)
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Immutable E8a {label} input hash differs: {name}")


def build_execution_plan(
    *,
    manifest_path: Path,
    e8a_plan_path: Path,
    input_paths: Mapping[str, Path] | None = None,
    authorization_reference: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    e8a_plan_path = Path(e8a_plan_path).resolve()
    if (
        not isinstance(authorization_reference, str)
        or not authorization_reference.strip()
        or authorization_reference != authorization_reference.strip()
    ):
        raise ValueError(
            "E8b exact user confirmation reference is missing or malformed"
        )
    if not manifest_path.is_file() or sha256_file(manifest_path) != (
        E8A_FROZEN_MANIFEST_FILE_SHA256
    ):
        raise ValueError("E8a manifest immutable frozen file hash differs")
    if not e8a_plan_path.is_file() or sha256_file(e8a_plan_path) != (
        E8A_FROZEN_PLAN_FILE_SHA256
    ):
        raise ValueError("E8a plan immutable frozen file hash differs")

    manifest = _read(manifest_path)
    e8a_plan = _read(e8a_plan_path)
    validate_e8a_plan(e8a_plan, manifest)
    if e8a_plan.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise ValueError("Immutable E8a plan references another manifest file")
    _verify_declared_e8a_inputs(manifest, "manifest")
    _verify_declared_e8a_inputs(e8a_plan, "plan")
    scientific_snapshot = e8a_scientific_snapshot_sha256(manifest, e8a_plan)

    paths = dict(
        input_paths or default_execution_inputs(manifest_path, e8a_plan_path)
    )
    if set(paths) != set(E8B_INPUT_NAMES):
        raise ValueError("E8b execution input closure differs")
    paths = {name: Path(path).resolve() for name, path in paths.items()}
    if paths["e8a_manifest"] != manifest_path:
        raise ValueError("E8b manifest input differs from manifest_path")
    if paths["e8a_plan"] != e8a_plan_path:
        raise ValueError("E8b E8a-plan input differs from e8a_plan_path")
    if sha256_file(paths["private_prompts"]) != manifest["private_prompt_source"][
        "sha256"
    ]:
        raise ValueError("E8b private validation-expansion prompt hash differs")

    payload: dict[str, Any] = {
        "schema_version": E8B_SCHEMA_VERSION,
        "analysis_semantics_version": E8B_PLAN_SEMANTICS_VERSION,
        "status": E8B_PLAN_STATUS,
        "mode": "prospective_route_replication",
        "research_stage": "understanding_testing_hypotheses",
        "created_at_utc": created_at_utc or _utc_now(),
        "objective": (
            "execute only the immutable E8a C-to-D replication with a fully "
            "hash-closed runtime, cumulative resume accounting, and no generation"
        ),
        "frozen_e8a_snapshot": {
            "scope": "entire_validated_e8a_manifest_and_plan_payloads",
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "plan_file_sha256": sha256_file(e8a_plan_path),
            "plan_payload_sha256": e8a_plan["payload_sha256"],
            "scientific_snapshot_sha256": scientific_snapshot,
        },
        "model": copy.deepcopy(e8a_plan["model"]),
        "transcoder": copy.deepcopy(e8a_plan["transcoder"]),
        "authorization": {
            "explicit_user_confirmation_received": True,
            "confirmation_reference": authorization_reference,
            "scope": E8B_AUTHORIZATION_SCOPE,
            "bundle_implementation_allowed": True,
            "gpu_allocation_allowed": True,
            "original_model_load_allowed": True,
            "original_model_outcomes_allowed": True,
            "validation_expansion_prompt_text_to_model_allowed": True,
            "generation_allowed": False,
            "e7d_allowed": False,
            "backward_pass_allowed": False,
            "confirmation_partition_allowed": False,
            "new_scientific_cells_allowed": False,
            "excluded_arms_allowed": False,
        },
        "outcome_disclosure": copy.deepcopy(e8a_plan["outcome_disclosure"]),
        "confirmation_boundary": copy.deepcopy(e8a_plan["confirmation_boundary"]),
        "excluded_arms": copy.deepcopy(e8a_plan["excluded_arms"]),
        "semantic_resolution": copy.deepcopy(E8B_SEMANTIC_RESOLUTION),
        "runtime_contract": copy.deepcopy(E8B_RUNTIME_CONTRACT),
        "numerical_contract": copy.deepcopy(E8B_NUMERICAL_CONTRACT),
        "forward_contract": copy.deepcopy(E8B_FORWARD_CONTRACT),
        "output_contract": copy.deepcopy(E8B_OUTPUT_CONTRACT),
        "inputs": {name: _input(paths[name]) for name in E8B_INPUT_NAMES},
        "privacy": copy.deepcopy(E8B_PRIVACY),
    }
    stamp_payload(payload)
    validate_e8b_execution_plan(payload, manifest, e8a_plan)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUTS / "s9_e8a_replication_manifest.json",
    )
    parser.add_argument(
        "--e8a-plan",
        type=Path,
        default=OUTPUTS / "s9_e8a_plan.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS / "s9_e8b_execution_plan.json",
    )
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--created-at-utc")
    args = parser.parse_args()
    payload = build_execution_plan(
        manifest_path=args.manifest,
        e8a_plan_path=args.e8a_plan,
        authorization_reference=args.authorization_reference,
        created_at_utc=args.created_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "payload_sha256": payload["payload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
