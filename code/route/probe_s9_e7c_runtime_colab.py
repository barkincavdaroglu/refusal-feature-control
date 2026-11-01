#!/usr/bin/env python3
"""Authenticate E7c's GPU runtime and every uploaded semantic byte."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from s9_contract import canonical_sha256, sha256_file
from s9_e7_contract import E7_INPUT_NAMES, validate_e7b_plan


PLAN = Path("/content/s9_e7b_plan.json")
OUTPUT = Path("/content/s9_e7c_runtime_probe.json")
PATHS = {
    "prompts": Path("/content/qwen3_4b_lexical_control_prompts.csv"),
    "annotations": Path("/content/qwen3_4b_lexical_control_manual_annotations.csv"),
    "probe_direction": Path("/content/qwen3_4b_refusal_probe_direction.npz"),
    "candidate_manifest": Path("/content/s9_e7b_candidate_manifest.json"),
    "manifest_builder": Path("/content/build_s9_e7b_candidate_manifest.py"),
    "topology_source": Path("/content/e7_topology.py"),
    "e7a_semantics": Path("/content/receiver_route_semantics.py"),
    "e7_contract": Path("/content/s9_e7_contract.py"),
    "plan_builder": Path("/content/build_s9_e7b_plan.py"),
    "runner": Path("/content/run_s9_e7c.py"),
    "colab_runner": Path("/content/run_s9_e7c_colab.py"),
    "runtime_probe": Path("/content/probe_s9_e7c_runtime_colab.py"),
    "analyzer": Path("/content/analyze_s9_e7c.py"),
    "runtime_installer": Path("/content/install_s9_colab_runtime.py"),
    "base_runner": Path("/content/run_s9_calibration.py"),
    "base_contract": Path("/content/s9_contract.py"),
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    validate_e7b_plan(plan)
    if set(PATHS) != set(E7_INPUT_NAMES):
        raise AssertionError("E7c runtime-probe input closure differs")
    hashes = {}
    for name, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Uploaded E7c input is missing: {name}")
        digest = sha256_file(path)
        if digest != plan["inputs"][name]["sha256"]:
            raise ValueError(f"Uploaded E7c input hash differs: {name}")
        hashes[name] = digest
    versions = {
        package: importlib.metadata.version(package)
        for package in plan["runtime_contract"]["package_versions"]
    }
    if versions != plan["runtime_contract"]["package_versions"]:
        raise ValueError("E7c runtime package contract differs")
    device = torch.cuda.get_device_name(0)
    if not any(name in device for name in plan["runtime_contract"]["preferred_accelerators"]):
        raise ValueError(f"Unsupported E7c accelerator: {device}")
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory < plan["runtime_contract"]["minimum_cuda_memory_gib"]:
        raise ValueError(f"E7c GPU memory below contract: {memory:.2f} GiB")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "runtime_and_uploaded_bytes_valid_before_e7c_model_load",
        "model_forward_executed_by_this_probe": False,
        "plan_payload_sha256": plan["payload_sha256"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device,
            "cuda_total_memory_gib": memory,
            "packages": versions,
        },
        "uploaded_input_hashes": hashes,
        "privacy": plan["privacy"],
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
