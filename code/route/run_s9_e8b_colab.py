#!/usr/bin/env python3
"""Run or cumulatively resume the frozen E8b route replication in Colab."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = [
        sys.executable,
        "/content/run_s9_e8b.py",
        "--execution-plan",
        "/content/s9_e8b_execution_plan.json",
        "--e8a-manifest",
        "/content/s9_e8a_replication_manifest.json",
        "--e8a-plan",
        "/content/s9_e8a_plan.json",
        "--semantic-erratum",
        "/content/E8B_PRE_OUTCOME_SEMANTIC_ERRATUM.md",
        "--prompts",
        "/content/s10_phase_b_validation_expansion_prompts.csv",
        "--probe-direction",
        "/content/qwen3_4b_refusal_probe_direction.npz",
        "--transcoder-row-bundle",
        "/content/s9_e8b_transcoder_rows.safetensors",
        "--transcoder-row-manifest",
        "/content/s9_e8b_transcoder_rows_manifest.json",
        "--runtime-probe",
        "/content/s9_e8b_runtime_probe.json",
        "--output",
        "/content/s9_e8b_result.json",
        "--raw-tensors",
        "/content/private/s9_e8b_raw_tensors.safetensors",
        "--resume",
    ]
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
