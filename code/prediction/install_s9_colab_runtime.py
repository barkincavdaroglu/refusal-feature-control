#!/usr/bin/env python3
"""Install the authenticated S9 Colab package stack without replacing torch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "transformers==4.57.3",
        "transformer-lens==3.2.1",
        "safetensors==0.8.0",
    ],
    check=True,
)
circuit_dir = Path("/content/circuit-tracer")
if not circuit_dir.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/decoderesearch/circuit-tracer.git",
            str(circuit_dir),
        ],
        check=True,
    )
subprocess.run(["git", "-C", str(circuit_dir), "fetch", "--all", "--quiet"], check=True)
subprocess.run(
    [
        "git",
        "-C",
        str(circuit_dir),
        "checkout",
        "8f1e2438df612464e229e44c4a00ff637bf9379b",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "-e", str(circuit_dir)],
    check=True,
)
print("S9 runtime installation complete", flush=True)
