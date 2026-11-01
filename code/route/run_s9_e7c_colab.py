#!/usr/bin/env python3
"""Run or resume the frozen E7c route panel in Colab."""

from __future__ import annotations

import runpy
import sys


sys.argv = [
    "run_s9_e7c.py",
    "--plan", "/content/s9_e7b_plan.json",
    "--manifest", "/content/s9_e7b_candidate_manifest.json",
    "--prompts", "/content/qwen3_4b_lexical_control_prompts.csv",
    "--annotations", "/content/qwen3_4b_lexical_control_manual_annotations.csv",
    "--probe-direction", "/content/qwen3_4b_refusal_probe_direction.npz",
    "--output", "/content/s9_e7c_result.json",
    "--raw-tensors", "/content/private/s9_e7c_raw_tensors.safetensors",
    "--resume",
]
_ = runpy.run_path("/content/run_s9_e7c.py", run_name="__main__")
