#!/usr/bin/env python3
"""Build E9c's minimal, deterministic C/comparator decoder bundle locally."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from s9_contract import canonical_sha256, sha256_file, stamp_payload
from s9_e9c_contract import (
    E8B_SOURCE_BUNDLE_FILE_SHA256,
    E8B_SOURCE_MANIFEST_FILE_SHA256,
    E8B_SOURCE_MANIFEST_PAYLOAD_SHA256,
    E9A_DESIGN_FILE_SHA256,
    E9A_DESIGN_PAYLOAD_SHA256,
    E9C_SCHEMA_VERSION,
    E9C_SOURCE_BUNDLE_KEYS,
    E9C_SOURCE_BUNDLE_SEMANTICS_VERSION,
    E9C_SOURCE_BUNDLE_STATUS,
    expected_source_features,
    tensor_identity_sha256,
    validate_e9c_source_bundle_manifest,
)


STEP = Path(__file__).resolve().parents[1]
OUTPUTS = STEP / "outputs"
PRIVATE = STEP / "private"

_SAFETENSORS_DTYPES = {
    torch.bfloat16: "BF16",
    torch.int64: "I64",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return payload


def save_deterministic_safetensors(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> None:
    if set(tensors) != set(E9C_SOURCE_BUNDLE_KEYS):
        raise ValueError("E9c deterministic source-bundle key closure differs")
    header: dict[str, Any] = {
        "__metadata__": {key: str(metadata[key]) for key in sorted(metadata)}
    }
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        dtype = _SAFETENSORS_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported E9c safetensors dtype: {tensor.dtype}")
        blob = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"".join(blobs))


def build_source_bundle(
    *,
    e9a_design_path: Path,
    source_bundle_path: Path,
    source_manifest_path: Path,
    output_bundle_path: Path,
    output_manifest_path: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    e9a_design_path = Path(e9a_design_path).resolve()
    source_bundle_path = Path(source_bundle_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    output_bundle_path = Path(output_bundle_path).resolve()
    output_manifest_path = Path(output_manifest_path).resolve()

    if sha256_file(e9a_design_path) != E9A_DESIGN_FILE_SHA256:
        raise ValueError("E9c source bundle E9a frozen file hash differs")
    design = _read(e9a_design_path)
    if design.get("payload_sha256") != E9A_DESIGN_PAYLOAD_SHA256:
        raise ValueError("E9c source bundle E9a frozen payload differs")
    if sha256_file(source_bundle_path) != E8B_SOURCE_BUNDLE_FILE_SHA256:
        raise ValueError("E9c authenticated E8b source bundle hash differs")
    if sha256_file(source_manifest_path) != E8B_SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("E9c authenticated E8b source manifest hash differs")
    source_manifest = _read(source_manifest_path)
    if source_manifest.get("payload_sha256") != E8B_SOURCE_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("E9c authenticated E8b source manifest payload differs")
    observed_features = {
        record.get("feature"): record.get("role")
        for record in source_manifest.get("features", [])
        if isinstance(record, Mapping)
    }
    if observed_features.get("L31:F49541") != "C" or observed_features.get(
        "L31:F139261"
    ) != "M":
        raise ValueError("E9c source manifest lacks the frozen C/comparator rows")

    source = load_file(str(source_bundle_path), device="cpu")
    if "layer_31.feature_indices" not in source or "layer_31.W_dec" not in source:
        raise ValueError("E9c source decoder tensors are missing")
    source_indices = source["layer_31.feature_indices"]
    source_decoders = source["layer_31.W_dec"]
    if source_indices.dtype != torch.int64 or source_indices.tolist() != [49541, 139261]:
        raise ValueError("E9c source feature indices differ")
    if source_decoders.dtype != torch.bfloat16 or list(source_decoders.shape) != [2, 2560]:
        raise ValueError("E9c source decoder rows differ")

    tensors = {
        "feature_indices": source_indices.detach().contiguous().cpu(),
        "decoder_rows": source_decoders.detach().contiguous().cpu(),
    }
    save_deterministic_safetensors(
        tensors,
        output_bundle_path,
        metadata={
            "analysis_semantics_version": E9C_SOURCE_BUNDLE_SEMANTICS_VERSION,
            "e9a_design_payload_sha256": E9A_DESIGN_PAYLOAD_SHA256,
            "source_bundle_file_sha256": E8B_SOURCE_BUNDLE_FILE_SHA256,
        },
    )
    payload: dict[str, Any] = {
        "schema_version": E9C_SCHEMA_VERSION,
        "analysis_semantics_version": E9C_SOURCE_BUNDLE_SEMANTICS_VERSION,
        "status": E9C_SOURCE_BUNDLE_STATUS,
        "created_at_utc": created_at_utc or _utc_now(),
        "objective": (
            "copy only the two frozen layer-31 decoder rows required by E9c "
            "without target-model or network contact"
        ),
        "producer": {
            "path": Path(__file__).name,
            "file_sha256": sha256_file(Path(__file__).resolve()),
        },
        "e9a_design": {
            "file_sha256": E9A_DESIGN_FILE_SHA256,
            "payload_sha256": E9A_DESIGN_PAYLOAD_SHA256,
        },
        "source": {
            "bundle_file_sha256": E8B_SOURCE_BUNDLE_FILE_SHA256,
            "manifest_file_sha256": E8B_SOURCE_MANIFEST_FILE_SHA256,
            "manifest_payload_sha256": E8B_SOURCE_MANIFEST_PAYLOAD_SHA256,
            "selected_layer": 31,
            "selected_tensor": "W_dec",
            "unused_e8b_rows_copied": False,
        },
        "features": expected_source_features(),
        "bundle": {
            "path": "s9_e9c_source_decoders.safetensors",
            "file_sha256": sha256_file(output_bundle_path),
            "keys": list(E9C_SOURCE_BUNDLE_KEYS),
            "tensor_identity_sha256": {
                name: tensor_identity_sha256(tensors[name])
                for name in E9C_SOURCE_BUNDLE_KEYS
            },
            "public": False,
        },
        "privacy": {
            "raw_decoder_values_in_public_manifest": False,
            "private_bundle_external_transfer_allowed": False,
            "target_model_contact": False,
        },
    }
    stamp_payload(payload)
    validate_e9c_source_bundle_manifest(
        payload,
        e9a_design=design,
        bundle_path=output_bundle_path,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e9a-design", type=Path, default=OUTPUTS / "s9_e9a_design.json")
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=PRIVATE / "s9_e8b_transcoder_rows.safetensors",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=OUTPUTS / "s9_e8b_transcoder_rows_manifest.json",
    )
    parser.add_argument(
        "--output-bundle",
        type=Path,
        default=PRIVATE / "s9_e9c_source_decoders.safetensors",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=OUTPUTS / "s9_e9c_source_decoders_manifest.json",
    )
    parser.add_argument("--created-at-utc")
    args = parser.parse_args()
    payload = build_source_bundle(
        e9a_design_path=args.e9a_design,
        source_bundle_path=args.source_bundle,
        source_manifest_path=args.source_manifest,
        output_bundle_path=args.output_bundle,
        output_manifest_path=args.output_manifest,
        created_at_utc=args.created_at_utc,
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output_bundle.resolve()),
                "manifest": str(args.output_manifest.resolve()),
                "payload_sha256": payload["payload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
