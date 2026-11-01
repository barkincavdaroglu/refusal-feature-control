#!/usr/bin/env python3
"""Authenticate E8b's runtime and uploaded bytes without a model forward."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXECUTION_PLAN = Path("/content/s9_e8b_execution_plan.json")
E8A_MANIFEST = Path("/content/s9_e8a_replication_manifest.json")
E8A_PLAN = Path("/content/s9_e8a_plan.json")
ROW_BUNDLE = Path("/content/s9_e8b_transcoder_rows.safetensors")
ROW_MANIFEST = Path("/content/s9_e8b_transcoder_rows_manifest.json")
OUTPUT = Path("/content/s9_e8b_runtime_probe.json")
REMOTE_SELF = Path("/content/probe_s9_e8b_runtime_colab.py")
PATHS = {
    "e8a_manifest": E8A_MANIFEST,
    "e8a_plan": E8A_PLAN,
    "semantic_erratum": Path("/content/E8B_PRE_OUTCOME_SEMANTIC_ERRATUM.md"),
    "private_prompts": Path(
        "/content/s10_phase_b_validation_expansion_prompts.csv"
    ),
    "probe_direction": Path("/content/qwen3_4b_refusal_probe_direction.npz"),
    "transcoder_row_bundle": ROW_BUNDLE,
    "transcoder_row_manifest": ROW_MANIFEST,
    "transcoder_bundle_builder": Path(
        "/content/build_s9_e8b_transcoder_bundle.py"
    ),
    "range_helper": Path("/content/extract_s9_e8b_transcoder_rows.py"),
    "runner": Path("/content/run_s9_e8b.py"),
    "colab_runner": Path("/content/run_s9_e8b_colab.py"),
    "runtime_probe": Path("/content/probe_s9_e8b_runtime_colab.py"),
    "analyzer": Path("/content/analyze_s9_e8b.py"),
    "runtime_installer": Path("/content/install_s9_colab_runtime.py"),
    "base_runner": Path("/content/run_s9_calibration.py"),
    "base_contract": Path("/content/s9_contract.py"),
    "e8_contract": Path("/content/s9_e8_contract.py"),
    "e8b_contract": Path("/content/s9_e8b_contract.py"),
    "execution_plan_builder": Path("/content/build_s9_e8b_execution_plan.py"),
}
_PROBE_CHILD_ENV = "S9_E8B_RUNTIME_PROBE_CHILD"
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _atomic_write(payload: dict[str, object], canonical_sha256) -> None:
    payload["payload_sha256"] = canonical_sha256(payload)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(OUTPUT)


def _nvidia_driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not versions or len(set(versions)) != 1:
        raise ValueError("E8b NVIDIA driver fingerprint is unavailable or heterogeneous")
    return versions[0]


def _runtime_fingerprint(torch, runtime_contract, versions, canonical_sha256):
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_gib": (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        ),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "nvidia_driver_version": _nvidia_driver_version(),
        "packages": dict(versions),
        "dtype": runtime_contract["dtype"],
        "random_seed": runtime_contract["random_seed"],
        "determinism": dict(runtime_contract["determinism"]),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    fields = runtime_contract["runtime_fingerprint_fields"]
    if set(fields) != set(runtime):
        raise ValueError("E8b runtime-probe fingerprint field closure differs")
    runtime["runtime_fingerprint_sha256"] = canonical_sha256(
        {"runtime_fingerprint": {key: runtime[key] for key in fields}}
    )
    return runtime


def _child_main() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("E8b CUBLAS workspace was not set before interpreter start")

    import torch

    from build_s9_e8b_transcoder_bundle import validate_transcoder_bundle_manifest
    from s9_contract import canonical_sha256, sha256_file
    from s9_e8b_contract import E8B_INPUT_NAMES, validate_e8b_execution_plan

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    execution_plan = json.loads(EXECUTION_PLAN.read_text(encoding="utf-8"))
    e8a_manifest = json.loads(E8A_MANIFEST.read_text(encoding="utf-8"))
    e8a_plan = json.loads(E8A_PLAN.read_text(encoding="utf-8"))
    validate_e8b_execution_plan(execution_plan, e8a_manifest, e8a_plan)
    if set(PATHS) != set(E8B_INPUT_NAMES):
        raise AssertionError("E8b runtime-probe input closure differs")

    hashes: dict[str, str] = {}
    for name, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Uploaded E8b input is missing: {name}")
        digest = sha256_file(path)
        if digest != execution_plan["inputs"][name]["sha256"]:
            raise ValueError(f"Uploaded E8b input hash differs: {name}")
        hashes[name] = digest

    row_manifest = json.loads(ROW_MANIFEST.read_text(encoding="utf-8"))
    validate_transcoder_bundle_manifest(
        row_manifest,
        e8a_manifest=e8a_manifest,
        e8a_plan=e8a_plan,
        e8a_manifest_path=E8A_MANIFEST,
        e8a_plan_path=E8A_PLAN,
        bundle_path=ROW_BUNDLE,
    )

    runtime_contract = execution_plan["runtime_contract"]
    if runtime_contract["determinism"]["cublas_workspace_config"] != (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    ):
        raise ValueError("E8b runtime-probe CUBLAS workspace differs")
    versions = {
        package: importlib.metadata.version(package)
        for package in runtime_contract["package_versions"]
    }
    if versions != runtime_contract["package_versions"]:
        raise ValueError("E8b runtime package contract differs")
    runtime = _runtime_fingerprint(
        torch, runtime_contract, versions, canonical_sha256
    )
    device = runtime["device"]
    if not any(
        name in device for name in runtime_contract["preferred_accelerators"]
    ):
        raise ValueError(f"Unsupported E8b accelerator: {device}")
    memory = runtime["cuda_total_memory_gib"]
    if memory < runtime_contract["minimum_cuda_memory_gib"]:
        raise ValueError(f"E8b GPU memory below contract: {memory:.2f} GiB")

    payload: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "runtime_and_uploaded_bytes_valid_before_e8b_model_load",
        "model_forward_executed_by_this_probe": False,
        "execution_plan_payload_sha256": execution_plan["payload_sha256"],
        "frozen_e8a_snapshot_sha256": execution_plan["frozen_e8a_snapshot"][
            "scientific_snapshot_sha256"
        ],
        "runtime": runtime,
        "uploaded_input_hashes": hashes,
        "confirmation_boundary": execution_plan["confirmation_boundary"],
        "privacy": execution_plan["privacy"],
    }
    _atomic_write(payload, canonical_sha256)
    print(json.dumps(payload, indent=2, allow_nan=False))


def main() -> None:
    if os.environ.get(_PROBE_CHILD_ENV) != "1":
        environment = dict(os.environ)
        environment["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        environment[_PROBE_CHILD_ENV] = "1"
        subprocess.run(
            [sys.executable, str(REMOTE_SELF)],
            check=True,
            env=environment,
        )
        return
    _child_main()


if __name__ == "__main__":
    main()
